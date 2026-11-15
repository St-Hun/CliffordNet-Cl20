# Reproduction guide

## Environment

The lockfile reproduces the original CAN software stack: Python 3.10,
PyTorch 2.10.0, TorchVision 0.25.0, and optional Triton 3.6.0. The reported runs
used an NVIDIA GeForce RTX 4090 24GB with CUDA 12.9 on Linux.

```bash
uv sync
uv run python scripts/verify_environment.py
uv run pytest
```

The pure PyTorch implementation works on CPU and CUDA. Triton is optional. A full
200-epoch CIFAR-100 run is intended for a modern CUDA GPU; CPU execution is for
smoke tests.

## Smoke test

```bash
uv run python scripts/train.py \
  --model configs/official/nano.yaml \
  --protocol configs/protocols/cifar100_official_2026_02.yaml \
  --data-dir data --download --epochs-smoke 1 --device cpu
```

Smoke mode limits training and evaluation to two batches and uses loader worker
count zero for restricted environments. Full runs retain the locked protocol.

## Full training and resume

```bash
uv run python scripts/train.py \
  --model configs/roll_shift/d25_s3.yaml \
  --protocol configs/protocols/cifar100_official_2026_02.yaml \
  --data-dir data --download --output runs/roll-25-3 --device cuda

uv run python scripts/train.py \
  --model configs/roll_shift/d25_s3.yaml \
  --protocol configs/protocols/cifar100_official_2026_02.yaml \
  --data-dir data --output runs/roll-25-3 --device cuda \
  --resume runs/roll-25-3/checkpoint-final.pt
```

Each run writes resolved configuration, `metrics.csv`, atomic `run.json`, best and
final checkpoints, SHA-256 digests, environment, Git state, dataset identity,
command, seed, and completion status. Interrupted runs remain `incomplete`.

## Evaluation

```bash
uv run python scripts/evaluate.py \
  --model configs/roll_shift/d25_s3.yaml \
  --protocol configs/protocols/cifar100_official_2026_02.yaml \
  --checkpoint runs/roll-25-3/checkpoint-best.pt \
  --data-dir data --device cuda
```

Weights & Biases is optional and not required. Local JSON/CSV records are always
written, making offline reproduction the default.
