#!/usr/bin/env python3
"""Train any registered model under the locked CIFAR-100 protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from cliffordnet_cl20.config import load_experiment
from cliffordnet_cl20.data import build_cifar100_loaders
from cliffordnet_cl20.engine import fit
from cliffordnet_cl20.models.registry import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs-smoke", type=int)
    args = parser.parse_args()

    config = load_experiment(args.model, args.protocol)
    protocol = config.protocol
    output = args.output or Path("runs") / f"{config.model_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved-config.json").write_text(json.dumps({"model": config.model, "protocol": protocol}, indent=2) + "\n")
    train_loader, eval_loader = build_cifar100_loaders(
        protocol, args.data_dir, args.download,
        num_workers_override=0 if args.epochs_smoke else None,
    )
    model = build_model(config.model_name)
    epochs = args.epochs_smoke or int(protocol["epochs"])
    smoke_batches = 2 if args.epochs_smoke else None
    record = fit(model, train_loader, eval_loader, output, epochs=epochs,
                 scheduler_epochs=int(protocol["epochs"]), lr=float(protocol["optimizer"]["lr"]),
                 weight_decay=float(protocol["optimizer"]["weight_decay"]), seed=int(protocol["seed"]),
                 device=args.device, resume=args.resume, max_train_batches=smoke_batches,
                 max_eval_batches=smoke_batches,
                 dataset_identity={"name": "CIFAR100", "root": str(args.data_dir.resolve()),
                                   "train_examples": len(train_loader.dataset), "evaluation_examples": len(eval_loader.dataset)},
                 resolved_config={"model": config.model, "protocol": protocol})
    print(json.dumps(asdict(record), indent=2))


if __name__ == "__main__":
    main()
