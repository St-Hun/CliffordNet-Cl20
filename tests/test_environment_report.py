import json

from cliffordnet_cl20.reproducibility import collect_environment, seed_everything


def test_environment_report_is_complete_and_json_serializable() -> None:
    report = collect_environment()
    assert set(report) >= {
        "python",
        "platform",
        "torch",
        "torchvision",
        "cuda_available",
        "cuda_runtime",
        "cudnn",
        "optional",
    }
    assert set(report["optional"]) == {"wandb", "triton"}
    json.dumps(report)


def test_seed_everything_repeats_torch_random_values() -> None:
    import torch

    seed_everything(123, deterministic=True)
    first = torch.rand(4)
    seed_everything(123, deterministic=True)
    second = torch.rand(4)
    assert torch.equal(first, second)
