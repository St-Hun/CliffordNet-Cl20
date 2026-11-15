"""Model registry with explicit reproducible specifications."""

from dataclasses import dataclass

from .official import OfficialCliffordNet
from .roll_shift import Cl20RollShiftNet


@dataclass(frozen=True)
class ModelSpec:
    family: str
    depth: int
    shifts: tuple[int, ...]
    cli_mode: str = "full"
    patch_size: int = 2
    embed_dim: int = 128
    drop_path_rate: float = 0.3


_SPECS = {
    "official_nano": ModelSpec("official", 12, (1, 2)),
    "official_lite": ModelSpec("official", 12, (1, 2, 4, 8, 16)),
    "official_d32": ModelSpec("official", 32, (1, 2, 4)),
    "official_d64": ModelSpec("official", 64, (1, 2, 4, 8, 16), cli_mode="inner", drop_path_rate=0.4),
    "roll_shift_d16_s2": ModelSpec("roll_shift", 16, (0, 1)),
    "roll_shift_d18_s5": ModelSpec("roll_shift", 18, (0, 1, 2, 4, 8)),
    "roll_shift_d25_s3": ModelSpec("roll_shift", 25, (0, 1, 2)),
    "roll_shift_d32_s5": ModelSpec("roll_shift", 32, (0, 1, 2, 4, 8)),
    "roll_shift_d48_s2": ModelSpec("roll_shift", 48, (0, 1)),
    "roll_shift_d52_s3": ModelSpec("roll_shift", 52, (0, 1, 2)),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _SPECS[name]
    except KeyError as error:
        raise KeyError(f"Unknown model: {name}") from error


def build_model(name: str, num_classes: int = 100):
    spec = get_model_spec(name)
    model_type = OfficialCliffordNet if spec.family == "official" else Cl20RollShiftNet if spec.family == "roll_shift" else None
    if model_type is None:
        raise NotImplementedError(spec.family)
    kwargs = dict(
        num_classes=num_classes,
        patch_size=spec.patch_size,
        embed_dim=spec.embed_dim,
        shifts=spec.shifts,
        depth=spec.depth,
        drop_path_rate=spec.drop_path_rate,
    )
    if spec.family == "official":
        kwargs["cli_mode"] = spec.cli_mode
    return model_type(**kwargs)
