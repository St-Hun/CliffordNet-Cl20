# Provenance and protocol decisions

The comparison framework is pinned to the public CliffordNet/CAN source at commit
`28119c5981f2308ec75b2ea8071c8e5f9471d42f`.  All model families use the same
CIFAR-100 loader and optimization protocol.

## Locked historical protocol

- CIFAR-100, 200 epochs, batch size 128
- AdamW, learning rate 0.001, weight decay 0.1
- cosine annealing
- random crop (padding 4), horizontal flip, CIFAR-10 AutoAugment policy,
  tensor conversion, normalization, then random erasing (p=0.25)
- evaluation uses only tensor conversion and normalization

The source calls the augmentation policy `CIFAR10` even for CIFAR-100.  This is
retained because changing it would break comparison with the historical runs.
The global working tree later changed several experimental model files; those
changes are treated as evidence for a later Shuffle generation, not as changes
to the pinned official baseline.

## Recovered model evidence

The Roll & Shift 25-3 implementation strictly loads the surviving 25-block
checkpoint.  Other portfolio Roll & Shift labels have architectural descriptions
but no unique compatible local checkpoint and are marked `portfolio_only`.
The later Geometric Shuffle source contains a 24-block construction with group
counts 2, 4, 8, 16, and 32.  No compatible metric/checkpoint pair survived, so
the repository deliberately reports no Shuffle accuracy.
