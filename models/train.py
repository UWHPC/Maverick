import argparse
import fcntl
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import torch
import torch_geometric
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from safetensors.torch import save_file
from torch.nn import functional as F
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph, to_undirected

from .gnn import ModelConfig, NodeClassifier

WORKLOADS = {
    "sage-arxiv": ("sage", "ogbn-arxiv", 0.01, 200),
    "gcn-arxiv": ("gcn", "ogbn-arxiv", 0.01, 200),
    "gat-arxiv": ("gat", "ogbn-arxiv", 0.005, 200),
    "sage-products": ("sage", "ogbn-products", 0.01, 30),
}


def stage_dataset(name: str, root: Path):
    # Stage the official archive before OGB can prompt on a large download.
    directory = root / name.replace("-", "_")
    pyg_directory = directory.with_name(directory.name + "_pyg")
    existing = pyg_directory if pyg_directory.exists() else directory
    if existing.exists():
        if not (existing / "RELEASE_v1.txt").is_file():
            raise ValueError(f"Incomplete OGB cache at {existing}; use a fresh data root")
        return

    archive_name = name.removeprefix("ogbn-")
    url = f"https://snap.stanford.edu/ogb/data/nodeproppred/{archive_name}.zip"
    print(f"Downloading {url}", flush=True)
    with TemporaryDirectory(dir=root) as temporary:
        archive = Path(temporary) / f"{archive_name}.zip"
        with urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output, length=2**20)
        shutil.unpack_archive(archive, temporary)
        (Path(temporary) / archive_name).rename(directory)


def load_dataset(name: str, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    # OGB builds a mutable processed cache on first use. Serialize that step.
    with (root / f".{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stage_dataset(name, root)
        with torch.serialization.safe_globals(
            [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]
        ):
            dataset = PygNodePropPredDataset(name=name, root=str(root))
            splits = dataset.get_idx_split()
    return dataset[0], splits, dataset.num_classes


def arxiv_snapshot(data, splits, seed: int):
    generator = torch.Generator().manual_seed(seed)
    nodes = torch.randperm(data.num_nodes, generator=generator)[:8192].sort().values
    edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True)
    snapshot = Data(x=data.x[nodes], y=data.y[nodes], edge_index=edge_index)
    masks = {
        name: torch.isin(nodes, indices).nonzero().flatten()
        for name, indices in splits.items()
    }
    if any(indices.numel() == 0 for indices in masks.values()):
        raise ValueError("The smoke snapshot must contain nodes in every OGB split")
    return snapshot, masks


class FullBatch:
    def __init__(self, model, data, splits, device):
        self.data = data.to(device)
        self.train_nodes = splits["train"].to(device)
        self.edge_index, self.edge_weight = model.prepare_edges(
            self.data.edge_index, self.data.num_nodes
        )

    def train_epoch(self, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        logits = self.predict(model)
        loss = F.cross_entropy(
            logits[self.train_nodes], self.data.y.view(-1)[self.train_nodes]
        )
        loss.backward()
        optimizer.step()
        return loss.detach().item()

    def predict(self, model):
        return model(self.data.x, self.edge_index, self.edge_weight)


class SampledProducts:
    def __init__(self, data, splits, device, batch_size, workers):
        self.data = data
        self.device = device
        self.train_loader = NeighborLoader(
            data,
            input_nodes=splits["train"],
            num_neighbors=[25, 10],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
        )
        graph = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
        self.eval_loader = NeighborLoader(
            graph, num_neighbors=[-1], batch_size=batch_size, num_workers=workers
        )

    def train_epoch(self, model, optimizer):
        total_loss = torch.zeros((), device=self.device)
        count = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            size = batch.batch_size
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index)[:size]
            loss = F.cross_entropy(logits, batch.y[:size].view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * size
            count += size
        return total_loss.item() / count

    def predict(self, model):
        x = self.data.x
        for layer in range(len(model.layers)):
            width = model.layers[layer].out_channels
            out = x.new_empty((self.data.num_nodes, width))
            for batch in self.eval_loader:
                seeds = batch.n_id[: batch.batch_size]
                features = x[batch.n_id].to(self.device)
                values = model.forward_layer(
                    layer,
                    (features, features[: batch.batch_size]),
                    batch.edge_index.to(self.device),
                )
                out[seeds] = values.cpu()
            x = out
        return x


@torch.inference_mode()
def score(model, runner, evaluator, labels, indices):
    model.eval()
    logits = runner.predict(model)
    predictions = logits[indices.to(logits.device)].argmax(dim=-1, keepdim=True)
    return evaluator.eval({"y_true": labels[indices], "y_pred": predictions.cpu()})["acc"]


def train(args):
    architecture, dataset, learning_rate, default_epochs = WORKLOADS[args.workload]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(args.seed)
    data, splits, classes = load_dataset(dataset, args.data_dir)
    if args.smoke:
        data, splits = arxiv_snapshot(data, splits, args.seed)
    if dataset == "ogbn-arxiv":
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    labels = data.y.view(-1, 1).cpu()
    config = ModelConfig(architecture, data.num_features, classes)
    model = NodeClassifier(config).to(device)
    sampled = dataset == "ogbn-products"
    if sampled:
        runner = SampledProducts(data, splits, device, args.batch_size, args.workers)
    else:
        runner = FullBatch(model, data, splits, device)
    evaluator = Evaluator(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = args.epochs or (3 if args.smoke else default_epochs)
    best_accuracy = -math.inf
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss = runner.train_epoch(model, optimizer)
        if not math.isfinite(loss):
            raise RuntimeError(f"Training loss is nonfinite at epoch {epoch}")
        entry = {"epoch": epoch, "loss": loss}
        if not sampled or epoch % 5 == 0 or epoch == epochs:
            accuracy = score(model, runner, evaluator, labels, splits["valid"])
            if not math.isfinite(accuracy):
                raise RuntimeError(f"Validation accuracy is nonfinite at epoch {epoch}")
            entry["validation_accuracy"] = accuracy
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if not sampled and epoch - best_epoch >= 30:
            break

    model.load_state_dict(best_state)
    test_accuracy = score(model, runner, evaluator, labels, splits["test"])
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(best_state, args.output / "model.safetensors")
    metrics = {
        "model": asdict(config),
        "dataset": dataset,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "symmetrized": dataset == "ogbn-arxiv",
        "seed": args.seed,
        "smoke": args.smoke,
        "learning_rate": learning_rate,
        "torch_version": torch.__version__,
        "pyg_version": torch_geometric.__version__,
        "best_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "history": history,
    }
    if sampled:
        metrics.update(batch_size=args.batch_size, fanouts=[25, 10])
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {args.workload} checkpoint to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train one OGB node classifier")
    parser.add_argument("workload", choices=WORKLOADS)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.workload == "sage-products":
        parser.error("--smoke only supports arXiv workloads")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers nonnegative")
    train(args)


if __name__ == "__main__":
    main()
