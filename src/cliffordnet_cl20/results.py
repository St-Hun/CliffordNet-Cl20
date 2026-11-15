"""Evidence and checkpoint classification types."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class CheckpointCandidate:
    name: str
    state_shapes: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class Classification:
    status: str
    matches: tuple[str, ...] = ()
    detail: str | None = None


def _state_shapes(payload) -> dict[str, tuple[int, ...]]:
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("checkpoint does not contain a state dict")
    shapes = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint contains non-tensor state")
        shapes[key] = tuple(value.shape)
    return shapes


def classify_checkpoint(path: Path, candidates: Sequence[CheckpointCandidate]) -> Classification:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        shapes = _state_shapes(payload)
    except Exception as error:
        return Classification("corrupt", detail=f"{type(error).__name__}: {error}")
    matches = tuple(candidate.name for candidate in candidates if dict(candidate.state_shapes) == shapes)
    if len(matches) == 1:
        return Classification("matched", matches)
    if len(matches) > 1:
        return Classification("quarantined", matches, "multiple exact architecture matches")
    return Classification("unmatched", detail=f"{len(shapes)} tensor keys")


def extract_run_record(run_dir: Path) -> dict[str, object]:
    """Extract only public experiment fields from a W&B run directory."""
    files = run_dir / "files"
    summary_path = files / "wandb-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    allowed = {"epoch", "best_acc", "test_acc", "train_acc", "train_loss", "lr"}
    return {"run_id": run_dir.name.removeprefix("run-"), "summary": {key: summary[key] for key in allowed if key in summary}}
