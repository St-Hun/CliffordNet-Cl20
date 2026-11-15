"""Strict model/protocol config resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


LOCKED_PROTOCOL_KEYS = frozenset({
    "dataset", "epochs", "batch_size", "num_workers", "seed", "deterministic",
    "optimizer", "scheduler", "transforms", "normalization", "evaluation",
})


class ProtocolOverrideError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    model: dict[str, Any]
    protocol: dict[str, Any]
    model_path: Path
    protocol_path: Path


def _read(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def load_experiment(model_config: Path, protocol_config: Path) -> ExperimentConfig:
    model_raw = _read(model_config)
    conflicts = LOCKED_PROTOCOL_KEYS.intersection(model_raw)
    if conflicts:
        raise ProtocolOverrideError(f"model config overrides locked protocol keys: {sorted(conflicts)}")
    model = model_raw["model"] if isinstance(model_raw.get("model"), dict) else model_raw
    model_name = model.get("name") or model.get("model")
    if not model_name:
        raise ValueError("model config must declare model name")
    protocol = _read(protocol_config)
    missing = LOCKED_PROTOCOL_KEYS.difference(protocol)
    if missing:
        raise ValueError(f"protocol is missing locked keys: {sorted(missing)}")
    return ExperimentConfig(str(model_name), model, protocol, model_config, protocol_config)
