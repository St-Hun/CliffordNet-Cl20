# CliffordNet-Cl20

Reproducible Cl(2,0) extensions built on the official
[CliffordNet/CAN](https://github.com/ParaMind2025/CAN) comparison framework.
This repository keeps three model families under one locked CIFAR-100 recipe:
the pinned official baselines, the portfolio Roll & Shift models, and the later
Geometric Shuffle experiment.

The original `CAN` working directory is evidence only. It is not imported at
runtime and was not modified while constructing this repository.

## What this work changes

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

### Roll & Shift

For each branch, the commuting pair is rolled by `+s` and the anticommuting pair
by `-s` before the full geometric product. Including `s=0` retains an unshifted,
complete Clifford product. The intent is to preserve the original CliffordNet
comparison framework while replacing scalar-channel approximations with actual
Cl(2,0) multivector interactions.

### Geometric Shuffle

Geometric Shuffle replaces shift distance with a deterministic ShuffleNet-style
group permutation. The same bijection is applied to both complex pairs, which
keeps every `(e0, e1, e2, e12)` tuple intact. An unshuffled product branch is
combined with several shuffled branches and projected back to the original
width. The intent is to reduce dependence on hand-selected shift magnitudes and
mix distant multivector groups more uniformly.

## Headline comparison

The official and Roll & Shift values use the locked 200-epoch CIFAR-100
protocol. The Shuffle values come from archived 200-epoch, seed-42 CIFAR-100
logs; those logs do not retain every resolved transform and optimizer field.
Accuracy differences are absolute percentage points, and parameter differences
use exact model counts.

| Variant | Reference | Parameters | Top-1 | Accuracy difference | Parameter difference |
|---|---|---:|---:|---:|---:|
| Roll & Shift 25-3 | CliffordNet-Lite | 2,557,668 | 80.57% | **+1.52 pp** | **-2.1%** |
| Roll & Shift 52-3 | CliffordNet-64 | 5,301,732 | 82.46% | **0.00 pp** | **-38.5%** |
| Shuffle 14 `[2,4,8]` | CliffordNet-Nano | 1,439,716 | 78.74% | **+0.92 pp** | +0.5% |
| Shuffle 26 `[2,4,8]` | CliffordNet-Lite | 2,659,300 | 81.02% | **+1.97 pp** | +1.8% |
| Shuffle 56 `[2,4,8]` | CliffordNet-32 | 4,790,756 | 81.61% | **+0.19 pp** | **-1.1%** |
| Shuffle 64 `[2,4,6]` | CliffordNet-64 | 5,472,740 | 81.81% | -0.65 pp | **-36.5%** |

The Roll & Shift family reaches the strongest reported result: it matches the
82.46% CliffordNet-64 accuracy with 3.32M fewer parameters. Geometric Shuffle is
most effective at the small and medium scales, improving Nano and Lite by 0.92
and 1.97 points. At the largest tested scale it stays within 0.65 points of
CliffordNet-64 while using 36.5% fewer parameters.

## Quick start

```bash
uv sync
uv run pytest
uv run python scripts/train.py \
  --model configs/roll_shift/d25_s3.yaml \
  --protocol configs/protocols/cifar100_official_2026_02.yaml \
  --data-dir data --download --device cuda
```

For a short integration check, append `--epochs-smoke 1`. See
[docs/reproduction.md](docs/reproduction.md) for resume and evaluation commands.

## Results and evidence

Accuracy values below are reported historical values, not fresh reproductions,
unless the status explicitly says `reproduced`. Parameter counts are computed
from registered models or recorded directly by the supplied historical logs.

<!-- RESULTS:START -->

### Official CliffordNet

| Model | Parameters | Top-1 | Status |
|---|---:|---:|---|
| `official_nano` | 1,433,188 | 77.82% | `official_reported` |
| `official_lite` | 2,612,836 | 79.05% | `official_reported` |
| `official_d32` | 4,842,084 | 81.42% | `official_reported` |
| `official_d64` | 8,618,596 | 82.46% | `official_reported` |

### Cl(2,0) Roll & Shift

| Model | Parameters | Top-1 | Status |
|---|---:|---:|---|
| `roll_shift_d16_s2` | 1,380,836 | 78.37% | `portfolio_reported` |
| `roll_shift_d18_s5` | 2,436,068 | 79.92% | `portfolio_reported` |
| `roll_shift_d25_s3` | 2,557,668 | 80.57% | `historical_verified` |
| `roll_shift_d32_s5` | 4,317,668 | 81.45% | `portfolio_reported` |
| `roll_shift_d48_s2` | 4,108,772 | 81.79% | `portfolio_reported` |
| `roll_shift_d52_s3` | 5,301,732 | 82.46% | `portfolio_reported` |

### Geometric Shuffle

| Model | Parameters | Top-1 | Status |
|---|---:|---:|---|
| `shuffle_d14_g2_4_8` | 1,439,716 | 78.74% | `historical_verified` |
| `shuffle_d26_g2_4_8` | 2,659,300 | 81.02% | `historical_verified` |
| `shuffle_d56_g2_4_8` | 4,790,756 | 81.61% | `historical_verified` |
| `shuffle_d64_g2_4_6` | 5,472,740 | 81.81% | `historical_verified` |

<!-- RESULTS:END -->

Status definitions are in [docs/results-status.md](docs/results-status.md). The
two 82.46% entries are different experiments: official CliffordNet-64 and Roll &
Shift 52-3.

## Repository scope

- `src/cliffordnet_cl20/models/official.py`: official baseline pinned to commit `28119c5`
- `src/cliffordnet_cl20/models/roll_shift.py`: full Cl(2,0) Roll & Shift
- `src/cliffordnet_cl20/models/shuffle.py`: structure-preserving Geometric Shuffle
- `configs/`: explicit architecture and locked protocol configurations
- `results/manifest.yaml`: machine-validated claims and evidence
- `docs/provenance.md`: reconstruction decisions and unresolved evidence
- `vendor/can_28119c5/`: pinned official source used only for equivalence tests

No upstream license file was present in the pinned source snapshot. This local
derivative therefore does not invent a license; review upstream licensing before
publishing it publicly. Attribution is recorded in `THIRD_PARTY_NOTICE.md`.
