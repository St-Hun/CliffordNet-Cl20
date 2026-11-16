# CliffordNet-Cl20

Reproducible Cl(2,0) extensions built on the official
[CliffordNet/CAN](https://github.com/ParaMind2025/CAN) comparison framework.
This repository keeps three model families under one locked CIFAR-100 recipe:
the pinned official baselines, the portfolio Roll & Shift models, and the later
Geometric Shuffle experiment.

## 🧠 What this work changes

The official CliffordNet already demonstrates that geometric interactions can
replace a conventional FFN. Its implementation, however, treats the feature
dimension as a one-dimensional channel field: shifted channels are combined in
separate dot- and wedge-like branches. This is efficient, but it does not keep an
explicit multivector basis throughout the interaction, creates parallel
intermediate branches, and makes the result sensitive to the selected shift
distances.

This work preserves the same stem, residual blocks, optimizer, augmentations,
and evaluation protocol, while changing only the geometric channel interaction:

1. **Explicit Cl(2,0) multivectors.** Every four channels are interpreted as
   `(e0, e1, e2, e12)`. A group is always moved as one unit, so a permutation
   never separates the scalar, vector, and bivector components.
2. **Exact, unified geometric product.** The four real components are mapped to
   two complex tensors, `comm = e0 + i*e12` and `anti = e1 - i*e2`. One unified
   operation computes the complete Cl(2,0) product instead of materializing
   independent dot and wedge paths.
3. **Structure-preserving channel mixing.** Roll & Shift and Geometric Shuffle
   provide two controlled ways to mix distant multivector groups without
   abandoning the algebraic structure.

### 🔄 How Roll & Shift works

For each branch, the commuting pair is rolled by `+s` and the anticommuting pair
by `-s` before the full geometric product. Including `s=0` retains an unshifted,
complete Clifford product. The intent is to preserve the original CliffordNet
comparison framework while replacing scalar-channel approximations with actual
Cl(2,0) multivector interactions.

### 🔀 How Geometric Shuffle works

Geometric Shuffle replaces shift distance with a deterministic ShuffleNet-style
group permutation. The same bijection is applied to both complex pairs, which
keeps every `(e0, e1, e2, e12)` tuple intact. An unshuffled product branch is
combined with several shuffled branches and projected back to the original
width. The intent is to reduce dependence on hand-selected shift magnitudes and
mix distant multivector groups more uniformly.

## 📊 Results

All values are CIFAR-100 Top-1 accuracy. Each proposed model is compared with
the closest official CliffordNet parameter scale; `Δ Acc.` is an absolute
percentage-point difference and `Δ Params` is relative to that baseline.

> 🏆 Roll & Shift 52-3 matches CliffordNet-64 at **82.46%** with **38.5% fewer
> parameters**. Shuffle 26 produces the largest accuracy gain: **+1.97 pp** over
> CliffordNet-Lite.

### Official CliffordNet baselines

| Baseline | Top-1 | Parameters |
|---|---:|---:|
| CliffordNet-Nano | 77.82% | 1,433,188 |
| CliffordNet-Lite | 79.05% | 2,612,836 |
| CliffordNet-32 | 81.42% | 4,842,084 |
| CliffordNet-64 | 82.46% | 8,618,596 |

### 🔄 Roll & Shift

| Variant | Shifts | Compared with | Top-1 (Δ Acc.) | Parameters (Δ Params) |
|---|---|---|---:|---:|
| Roll 16-2 | `[0,1]` | Nano | 78.37% (+0.55 pp) | 1,380,836 (-3.7%) |
| Roll 18-5 | `[0,1,2,4,8]` | Lite | 79.92% (+0.87 pp) | 2,436,068 (-6.8%) |
| Roll 25-3 | `[0,1,2]` | Lite | 80.57% (+1.52 pp) | 2,557,668 (-2.1%) |
| Roll 32-5 | `[0,1,2,4,8]` | 32 | 81.45% (+0.03 pp) | 4,317,668 (-10.8%) |
| Roll 48-2 | `[0,1]` | 32 | 81.79% (+0.37 pp) | 4,108,772 (-15.1%) |
| Roll 52-3 | `[0,1,2]` | 64 | **82.46% (0.00 pp)** | **5,301,732 (-38.5%)** |

Every listed Roll & Shift model matches or improves its nearest official
baseline while using fewer parameters.

### 🔀 Geometric Shuffle

| Variant | Groups | Compared with | Top-1 (Δ Acc.) | Parameters (Δ Params) |
|---|---|---|---:|---:|
| Shuffle 14 | `[2,4,8]` | Nano | 78.74% (+0.92 pp) | 1,439,716 (+0.5%) |
| Shuffle 26 | `[2,4,8]` | Lite | **81.02% (+1.97 pp)** | 2,659,300 (+1.8%) |
| Shuffle 56 | `[2,4,8]` | 32 | 81.61% (+0.19 pp) | 4,790,756 (-1.1%) |
| Shuffle 64 | `[2,4,6]` | 64 | 81.81% (-0.65 pp) | 5,472,740 (-36.5%) |

Runnable configurations are included for Shuffle 14 and 26, whose architecture
and parameter counts match the retained implementation exactly. Shuffle 56 and
64 are historical result rows because their final architecture was not
preserved completely.

## 🖥️ Original CAN execution environment

The reported experiments were run in the original CAN environment, not in a
separate benchmark setup.

- **Hardware:** NVIDIA GeForce RTX 4090 24GB
- **System:** Linux 6.8.0-106-generic, CUDA 12.9
- **Runtime:** Python 3.10.19, PyTorch 2.10.0, TorchVision 0.25.0, Triton 3.6.0
- **Dataset:** CIFAR-100, 200 epochs, batch size 128, seed 42
- **Optimization:** AdamW, learning rate `1e-3`, weight decay `0.1`, cosine
  annealing, no warm-up
- **Augmentation:** random crop with 4-pixel padding, horizontal flip,
  CIFAR-10 AutoAugment, and random erasing with probability 0.25

## 🚀 Quick start

```bash
uv sync
uv run pytest
uv run python train.py \
  --model roll_shift_d25_s3 \
  --data-dir data --download --device cuda
```

This is the original CAN training flow: every training epoch has a live `tqdm`
progress bar, every test epoch prints its accuracy, W&B logging is enabled, and
the best weights are saved as a plain PyTorch `state_dict`. For a one-epoch
check without W&B, append `--epochs 1 --wandb-mode disabled`.

See [docs/reproduction.md](docs/reproduction.md) for all model names and
evaluation commands.

## 🗂 Repository scope

- `src/cliffordnet_cl20/models/official.py`: official baseline pinned to commit `28119c5`
- `src/cliffordnet_cl20/models/roll_shift.py`: full Cl(2,0) Roll & Shift
- `src/cliffordnet_cl20/models/shuffle.py`: structure-preserving Geometric Shuffle
- `train.py`: CAN-style CIFAR-100 training entry point for every model
- `scripts/`: evaluation, checkpoint, and environment utilities
- `docs/reproduction.md`: reproducible training and evaluation commands

No upstream license file was present in the pinned source snapshot. This local
derivative therefore does not invent a license. Confirm redistribution terms
with the upstream author before reusing upstream-derived code. Attribution is
recorded in `THIRD_PARTY_NOTICE.md`.
