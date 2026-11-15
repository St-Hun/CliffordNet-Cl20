# Verification report

Date: 2026-11-16 (Asia/Seoul)

## Completed checks

- `pytest -q`: 45 passed
- `ruff check src scripts tests`: passed
- generated README results table: current and manifest-validated
- official Nano/Lite state dictionaries and outputs: equivalent to pinned source
- Cl(2,0) algebra: basis table, direct/complex products, round trips, and gradients tested
- Roll & Shift 25-3: surviving checkpoint strict-loads with all keys matching
- one-epoch CIFAR-100 integration smoke: completed on CPU using two train/eval batches
- reference inventory before/after: byte-for-byte identical

## Model matrix

The registry contains four official baselines, six portfolio Roll & Shift
configurations, and one source-recovered Geometric Shuffle configuration. All are
constructible. Only Roll & Shift 25-3 has a surviving uniquely compatible
historical checkpoint. The Shuffle implementation has no claimed accuracy because
no compatible result/checkpoint pair survived.

## Publication gates

No remote was added and nothing was pushed. Checkpoints, W&B binary data, CIFAR
data, compiled wheels, personal paths, and credentials are excluded. The pinned
upstream source did not expose a root license file, so public publication requires
the repository owner to resolve upstream licensing first.
