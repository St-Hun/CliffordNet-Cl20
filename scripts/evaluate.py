#!/usr/bin/env python3
"""Evaluate a registered checkpoint on CIFAR-100."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cliffordnet_cl20.config import load_experiment
from cliffordnet_cl20.data import build_cifar100_loaders
from cliffordnet_cl20.engine import evaluate
from cliffordnet_cl20.models.registry import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_experiment(args.model, args.protocol)
    _, loader = build_cifar100_loaders(config.protocol, args.data_dir)
    model = build_model(config.model_name)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(payload.get("model", payload))
    model.to(args.device)
    print(json.dumps(asdict(evaluate(model, loader, args.device)), indent=2))


if __name__ == "__main__":
    main()
