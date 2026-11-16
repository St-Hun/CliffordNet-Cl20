# Reproduction guide

## Environment

The lockfile reproduces the software versions used in the original CAN
environment: Python 3.10, PyTorch 2.10.0, TorchVision 0.25.0, and optional
Triton 3.6.0. The reported runs used an NVIDIA GeForce RTX 4090 24GB with
CUDA 12.9 on Linux.

```bash
uv sync
uv run python scripts/verify_environment.py
uv run pytest
```

## Training

`train.py` follows the original CAN training script directly. It uses CIFAR-100,
batch size 128, AdamW (`lr=1e-3`, `weight_decay=0.1`), cosine annealing, the CAN
augmentation pipeline, and 200 epochs by default.

```bash
uv run python train.py \
  --model roll_shift_d25_s3 \
  --data-dir data \
  --download \
  --device cuda
```

During training, the terminal shows a live batch progress bar and prints test
accuracy after every epoch. W&B runs online by default. The best checkpoint is
saved as `best_model_<model>.pth`, using the same plain `state_dict` format as
the original CAN code.

For a one-epoch integration check without uploading a W&B run:

```bash
uv run python train.py \
  --model roll_shift_d25_s3 \
  --data-dir data \
  --download \
  --device cuda \
  --epochs 1 \
  --wandb-mode disabled
```

Available model names:

- Official CAN: `official_nano`, `official_lite`, `official_d32`, `official_d64`
- Roll & Shift: `roll_shift_d16_s2`, `roll_shift_d18_s5`,
  `roll_shift_d25_s3`, `roll_shift_d32_s5`, `roll_shift_d48_s2`,
  `roll_shift_d52_s3`
- Geometric Shuffle: `shuffle_d14_g2_4_8`, `shuffle_d26_g2_4_8`
- CAN comparison models: `vit_tiny`, `mobilenet_v2`, `shufflenet_v2_1_5`

Historical CAN names such as `mustardnet_25_3` and `cliffordnet_12_2` are also
accepted as aliases.

## Evaluation

```bash
uv run python scripts/evaluate.py \
  --model roll_shift_d25_s3 \
  --checkpoint best_model_roll_shift_d25_s3.pth \
  --data-dir data \
  --device cuda
```

Add `--download` if CIFAR-100 is not already present.

To verify that an old checkpoint has the expected architecture without running
the dataset:

```bash
uv run python scripts/verify_checkpoint.py \
  /path/to/best_model_mustardnet_25_3.pth \
  roll_shift_d25_s3
```
