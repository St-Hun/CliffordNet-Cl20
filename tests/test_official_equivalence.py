import torch

from cliffordnet_cl20.models.registry import build_model, get_model_spec
from networks import cliffordnet_12_2 as upstream_nano
from networks import cliffordnet_12_5 as upstream_lite


def _assert_equivalent(name: str, upstream_factory) -> None:
    torch.manual_seed(7)
    upstream = upstream_factory(enable_cuda=False).eval()
    torch.manual_seed(7)
    packaged = build_model(name).eval()
    assert upstream.state_dict().keys() == packaged.state_dict().keys()
    for key, value in upstream.state_dict().items():
        assert torch.equal(value, packaged.state_dict()[key]), key
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(upstream(x), packaged(x))


def test_official_nano_matches_pinned_upstream() -> None:
    _assert_equivalent("official_nano", upstream_nano)


def test_official_lite_matches_pinned_upstream() -> None:
    _assert_equivalent("official_lite", upstream_lite)


def test_official_registry_names_match_architecture() -> None:
    expected = {
        "official_nano": (12, (1, 2), 1_433_188),
        "official_lite": (12, (1, 2, 4, 8, 16), 2_612_836),
        "official_d32": (32, (1, 2, 4), 4_842_084),
        "official_d64": (64, (1, 2, 4, 8, 16), 8_618_596),
    }
    for name, (depth, shifts, params) in expected.items():
        spec = get_model_spec(name)
        model = build_model(name)
        assert spec.depth == depth
        assert spec.shifts == shifts
        assert len(model.blocks) == depth
        assert sum(p.numel() for p in model.parameters()) == params
