"""Environment capture and deterministic seeding utilities."""

from __future__ import annotations

import importlib.metadata
import platform
import random
import sys
from typing import Any


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _version("torch"),
        "torchvision": _version("torchvision"),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "executable": sys.executable,
        "optional": {"wandb": _version("wandb"), "triton": _version("triton")},
    }


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
