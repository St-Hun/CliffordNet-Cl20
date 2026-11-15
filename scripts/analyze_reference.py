#!/usr/bin/env python3
"""Create a privacy-safe inventory of historical experiment evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cliffordnet_cl20.results import extract_run_record


def checkpoint_record(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        tensors = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
        block_indexes = {
            int(key.split(".")[1])
            for key in tensors
            if key.startswith("blocks.") and len(key.split(".")) > 2 and key.split(".")[1].isdigit()
        }
        return {
            "file": path.name,
            "status": "readable",
            "tensor_elements": sum(value.numel() for value in tensors.values()),
            "tensor_keys": len(tensors),
            "blocks": max(block_indexes) + 1 if block_indexes else None,
        }
    except Exception as error:  # noqa: BLE001 - forensic scans must quarantine every unreadable artifact
        return {"file": path.name, "status": "corrupt", "detail": type(error).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoints = [checkpoint_record(path) for path in sorted(args.reference.glob("*.pth"))]
    runs = [extract_run_record(path) for path in sorted((args.reference / "wandb").glob("run-*"))]
    report = {
        "baseline_commit": "28119c5981f2308ec75b2ea8071c8e5f9471d42f",
        "checkpoints": checkpoints,
        "runs": [run for run in runs if run["summary"]],
        "excluded": ["data", "wandb binaries", "IDE metadata", "personal machine metadata", "compiled wheels"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
