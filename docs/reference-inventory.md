# Reference inventory

The read-only recovery scan found 16 root checkpoints and 26 completed W&B
summaries with public metrics. Raw W&B binaries, datasets, local paths, email,
GPU UUIDs, IDE files, caches, and compiled wheels are excluded.

## Strongly identified artifacts

- Official 12-shift-2 and 12-shift-5 checkpoints are present.
- The Roll & Shift 25-3 checkpoint has 25 blocks and is paired with a completed
  run reporting 80.57% best test accuracy.
- A completed 5,301,732-parameter 52-3 run reports 82.33%, but the root file
  currently named `best_model_mustardnet_52_3.pth` has only 24 blocks and was
  overwritten by a later experiment. It is not accepted as the 52-3 artifact.
- The 64-block checkpoint belongs to a separate historical configuration and
  must not be relabeled as 52-3 solely because its parameter count is nearby.

`results/recovery-report.json` is the machine-readable evidence inventory.
