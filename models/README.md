# Model weights

The `models/model_weights` submodule pins a revision of the [Hugging Face weights repository](https://huggingface.co/alonso130r/maverick).
A regular clone leaves it uninitialized, which is enough for development without
checkpoints or training from scratch.

## Pull checkpoints

Install Git LFS and Git Xet, then run these commands from the Maverick checkout root.

```sh
git lfs install
git xet install
bash scripts/weights.sh pull
```

Published runs are stored under `models/model_weights/runs/<run-id>/`. Each workload
directory contains a `model.safetensors` state dict and a `metrics.json` file with
the model configuration, seed, and OGB validation and test accuracy.

## Publish a training run

The training launcher writes checkpoints, metrics, and per-model logs to `runs/`
by default. Set `MAVERICK_RUN_DIR` to override that location.

Authenticate Git with a Hugging Face credential that can write to the weights
repository. From the checkout root, pass the completed run directory printed by
the training launcher to `bash scripts/weights.sh publish RUN_DIR`. The helper
uploads the checkpoints and metrics using Git Xet. Commit the updated
`models/model_weights` gitlink in Maverick to pin the published revision for other
clones.

[Hugging Face Git Xet setup](https://huggingface.co/docs/hub/en/xet/using-xet-storage)
