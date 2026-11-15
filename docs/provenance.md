# Provenance

## Baseline of record

The official baseline is pinned to ParaMind2025/CAN commit
`28119c5981f2308ec75b2ea8071c8e5f9471d42f`, the revision recorded by the
March 2026 experiment metadata. The packaged implementation is checked for
state-dict and output equivalence against the original root `model.py` and
`networks.py` at that commit.

| Packaged component | Historical source |
| --- | --- |
| `layers/official_interaction.py` | `model.py::CliffordInteraction_PyTorch` |
| `models/official.py` | `model.py::CliffordNet` and supporting blocks |
| `models/common.py` | `utils.py::DropPath` and fallback LayerNorm |

The observed upstream main commit on 2026-11-15 is
`47bda79f15f93918c8c78da85c84345e72107560`; it is not substituted into
the historical comparison.
