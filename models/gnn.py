from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    in_channels: int
    out_channels: int
    hidden_channels: int = 64
    dropout: float = 0.5
    attention_heads: int = 4
    attention_head_width: int = 8


class NodeClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.activation = F.relu
        if config.architecture == "sage":
            layers = (
                SAGEConv(config.in_channels, config.hidden_channels),
                SAGEConv(config.hidden_channels, config.out_channels),
            )
        elif config.architecture == "gcn":
            layers = (
                GCNConv(config.in_channels, config.hidden_channels, normalize=False),
                GCNConv(config.hidden_channels, config.out_channels, normalize=False),
            )
        elif config.architecture == "gat":
            layers = (
                GATConv(
                    config.in_channels,
                    config.attention_head_width,
                    heads=config.attention_heads,
                    add_self_loops=False,
                ),
                GATConv(
                    config.attention_heads * config.attention_head_width,
                    config.out_channels,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                ),
            )
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")
        self.layers = nn.ModuleList(layers)

    def prepare_edges(
        self, edge_index: Tensor, num_nodes: int
    ) -> tuple[Tensor, Tensor | None]:
        if self.config.architecture == "gcn":
            return gcn_norm(edge_index, num_nodes=num_nodes)
        if self.config.architecture == "gat":
            edge_index, _ = remove_self_loops(edge_index)
            return add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index, None

    def forward_layer(
        self,
        index: int,
        x: Tensor | tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        layer = self.layers[index]
        if self.config.architecture == "gcn":
            x = layer(x, edge_index, edge_weight)
        else:
            x = layer(x, edge_index)
        if index == 0:
            x = F.dropout(self.activation(x), self.config.dropout, self.training)
        return x

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None
    ) -> Tensor:
        x = self.forward_layer(0, x, edge_index, edge_weight)
        return self.forward_layer(1, x, edge_index, edge_weight)
