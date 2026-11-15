"""Model registry with explicit reproducible specifications."""

from dataclasses import dataclass

from .official import OfficialCliffordNet


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
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _SPECS[name]
    except KeyError as error:
        raise KeyError(f"Unknown model: {name}") from error


def build_model(name: str, num_classes: int = 100):
    spec = get_model_spec(name)
    if spec.family != "official":
        raise NotImplementedError(spec.family)
    return OfficialCliffordNet(
        num_classes=num_classes,
        patch_size=spec.patch_size,
        embed_dim=spec.embed_dim,
        cli_mode=spec.cli_mode,
        shifts=spec.shifts,
        depth=spec.depth,
        drop_path_rate=spec.drop_path_rate,
    )
