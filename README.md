# CliffordNet-Cl20

Reproducible Cl(2,0) extensions built on the official
[CliffordNet/CAN](https://github.com/ParaMind2025/CAN) comparison framework.
This repository keeps three model families under one locked CIFAR-100 recipe:
the pinned official baselines, the portfolio Roll & Shift models, and the later
Geometric Shuffle experiment.

The original `CAN` working directory is evidence only. It is not imported at
runtime and was not modified while constructing this repository.

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
from the registered models. The Shuffle source survived, but a compatible
metric/checkpoint pair did not, so no accuracy is claimed.

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
| `shuffle_d24_g2_4_8_16_32` | 3,242,468 | - | `source_only` |

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

No upstream license file was present in the pinned source snapshot. This local
derivative therefore does not invent a license; review upstream licensing before
publishing it publicly. Attribution is recorded in `THIRD_PARTY_NOTICE.md`.
