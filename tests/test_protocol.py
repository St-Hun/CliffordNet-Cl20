from pathlib import Path

import pytest

from cliffordnet_cl20.config import ProtocolOverrideError, load_experiment


def test_official_protocol_is_locked():
    config = load_experiment(Path("configs/official/nano.yaml"), Path("configs/protocols/cifar100_official_2026_02.yaml"))
    assert config.model_name == "official_nano"
    assert config.protocol["epochs"] == 200
    assert config.protocol["optimizer"] == {"name": "adamw", "lr": 0.001, "weight_decay": 0.1}
    assert config.protocol["transforms"]["train"] == ["random_crop", "horizontal_flip", "autoaugment_cifar10", "to_tensor", "normalize", "random_erasing"]


def test_model_config_cannot_override_protocol(tmp_path):
    model = tmp_path / "model.yaml"
    model.write_text("model: official_nano\nepochs: 1\n")
    with pytest.raises(ProtocolOverrideError, match="epochs"):
        load_experiment(model, Path("configs/protocols/cifar100_official_2026_02.yaml"))
