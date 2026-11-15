"""Evidence and checkpoint classification types."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
            raise TypeError("checkpoint contains non-tensor state")
        shapes[key] = tuple(value.shape)
    return shapes


def classify_checkpoint(path: Path, candidates: Sequence[CheckpointCandidate]) -> Classification:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        shapes = _state_shapes(payload)
    except Exception as error:  # noqa: BLE001 - corrupt checkpoints are expected forensic inputs
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


class ManifestError(ValueError):
    """Raised when a result lacks sufficient reproducibility evidence."""


RESULT_STATUSES = {"official_reported", "portfolio_reported", "historical_verified", "reproduced", "source_only"}


def validate_manifest(data: Mapping, root: Path = Path(".")) -> None:
    from .models.registry import build_model

    rows = data.get("results")
    if not isinstance(rows, list):
        raise ManifestError("results must be a list")
    identities = set()
    for row in rows:
        if row.get("status") not in RESULT_STATUSES:
            raise ManifestError(f"unknown status: {row.get('status')}")
        identity = (row.get("model"), row.get("protocol"), row.get("seed"))
        if identity in identities:
            raise ManifestError(f"duplicate result identity: {identity}")
        identities.add(identity)
        evidence = row.get("evidence")
        if not evidence:
            raise ManifestError(f"result has no evidence: {identity}")
        for relative in evidence:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ManifestError("evidence paths must be repository-relative")
            if not (root / path).is_file():
                raise ManifestError(f"evidence path does not exist: {relative}")
        try:
            expected = sum(parameter.numel() for parameter in build_model(row["model"]).parameters())
        except (KeyError, NotImplementedError):
            expected = row.get("parameters")
        if row.get("parameters") != expected:
            raise ManifestError(f"parameter count mismatch for {row.get('model')}: {row.get('parameters')} != {expected}")
        if row["status"] == "reproduced":
            valid_record = False
            for relative in evidence:
                try:
                    record = json.loads((root / relative).read_text())
                    resolved = record.get("resolved_config", {})
                    model_config = resolved.get("model", {})
                    recorded_model = model_config.get("name") or model_config.get("model")
                    protocol = resolved.get("protocol", {})
                    execution = resolved.get("execution", {})
                    final_accuracy = record.get("epochs", [{}])[-1].get("eval_accuracy")
                    valid_record |= (
                        record.get("status") == "complete"
                        and not execution.get("smoke", False)
                        and recorded_model == row["model"]
                        and protocol.get("name") == row["protocol"]
                        and record.get("seed") == row["seed"]
                        and record.get("parameter_count") == row["parameters"]
                        and final_accuracy is not None
                        and abs(final_accuracy * 100 - row["top1"]) < 1e-9
                    )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            if not valid_record:
                raise ManifestError("reproduced result requires a matching complete non-smoke run record")


def render_results_table(data: Mapping) -> str:
    titles = {
        "official": "🧱 Official CliffordNet",
        "roll_shift": "🔄 Cl(2,0) Roll & Shift",
        "shuffle": "🔀 Geometric Shuffle",
    }
    lines = []
    for family in ("official", "roll_shift", "shuffle"):
        lines.extend((f"### {titles[family]}", "", "| Model | Parameters | Top-1 |", "|---|---:|---:|"))
        for row in data["results"]:
            if row["family"] != family:
                continue
            accuracy = "-" if row.get("top1") is None else f"{row['top1']:.2f}%"
            lines.append(f"| `{row['model']}` | {row['parameters']:,} | **{accuracy}** |")
        lines.append("")
    return "\n".join(lines).rstrip()
