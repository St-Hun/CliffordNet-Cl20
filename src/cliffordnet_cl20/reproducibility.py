"""Environment capture and deterministic seeding utilities."""

from __future__ import annotations

import importlib.metadata
import os
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"Global seed set to {seed}")
