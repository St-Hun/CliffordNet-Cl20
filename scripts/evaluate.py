"""Evaluate a plain CAN-style state_dict checkpoint on CIFAR-100."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import TrainingConfig, create_model, get_dataloaders


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = model(inputs).argmax(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-100 checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(data_root=args.data_dir, device=args.device)
    _, test_loader = get_dataloaders(config, download=args.download)
    model = create_model(args.model, config)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(args.device)
    score = accuracy(model, test_loader, torch.device(args.device))
    print(f"Test Acc: {score:.2f}%")


if __name__ == "__main__":
    main()
