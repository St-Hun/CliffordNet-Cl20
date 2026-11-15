"""CIFAR-100 loaders matching the pinned official augmentation recipe."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def cifar100_transforms(protocol):
    mean = protocol["normalization"]["mean"]
    std = protocol["normalization"]["std"]
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=protocol["transforms"]["crop_padding"]),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=protocol["transforms"]["random_erasing_probability"]),
    ])
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train, evaluation


def build_cifar100_loaders(protocol, data_dir: Path, download: bool = False, num_workers_override: int | None = None):
    train_transform, eval_transform = cifar100_transforms(protocol)
    train_set = datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=download)
    eval_set = datasets.CIFAR100(data_dir, train=False, transform=eval_transform, download=download)
    generator = torch.Generator().manual_seed(int(protocol["seed"]))
    workers = int(protocol["num_workers"]) if num_workers_override is None else num_workers_override
    common = {"batch_size": int(protocol["batch_size"]), "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    eval_loader = DataLoader(eval_set, shuffle=False, **common)
    return train_loader, eval_loader
