# Geometric Shuffle results transcription

This file records the selected completed CIFAR-100 experiments from archived
historical console logs supplied during repository reconstruction. Personal
filesystem and environment paths from the raw logs are intentionally excluded.

Common recorded settings:

- global seed: 42
- training length: 200 epochs
- device: CUDA
- implementation: pure PyTorch fallback
- metric below: best observed test accuracy

| Experiment | Shuffle groups | Exact learnable parameters | Best test accuracy | Final epoch accuracy | Runtime |
|---|---|---:|---:|---:|---:|
| Shuffle 14 | `[2, 4, 8]` | 1,439,716 | 78.74% | 78.71% | 82.64 min |
| Shuffle 26 | `[2, 4, 8]` | 2,659,300 | 81.02% | 81.00% | 161.74 min |
| Shuffle 56 | `[2, 4, 8]` | 4,790,756 | 81.61% | 81.51% | 280.78 min |
| Shuffle 64 | `[2, 4, 6]` | 5,472,740 | 81.81% | 81.67% | 338.81 min |

The Shuffle 14 log reaches all 200 epochs and prints the completed runtime, but
the captured process footer reports exit code 1 after training. The reported
accuracy and parameter count are retained as historical log evidence rather than
as a newly reproduced result.
