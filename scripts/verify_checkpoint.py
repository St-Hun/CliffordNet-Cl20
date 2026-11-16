"""Check whether a checkpoint matches a named architecture."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import TrainingConfig, create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model_name")
    args = parser.parse_args()

    model = create_model(args.model_name, TrainingConfig())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not all(
        isinstance(key, str) for key in checkpoint
    ):
        raise TypeError("Expected a plain state_dict checkpoint")

    incompatible = model.load_state_dict(checkpoint, strict=False)
    compatible = not incompatible.missing_keys and not incompatible.unexpected_keys
    print(f"Model: {args.model_name}")
    print(f"Compatible: {compatible}")
    print(f"Missing keys: {incompatible.missing_keys}")
    print(f"Unexpected keys: {incompatible.unexpected_keys}")
    raise SystemExit(0 if compatible else 1)


if __name__ == "__main__":
    main()
