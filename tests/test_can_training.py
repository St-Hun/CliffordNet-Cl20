import importlib

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _training_module():
    return importlib.import_module("train")


def _classification_fixture():
    inputs = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    return model, loader, inputs, targets


def test_training_defaults_match_can():
    config = _training_module().TrainingConfig()
    assert config.batch_size == 128
    assert config.lr == 1e-3
    assert config.epochs == 200
    assert config.weight_decay == 0.1
    assert config.seed == 42
    assert config.random_erasing_prob == 0.25
    assert config.warmup_epochs == 0


def test_official_cliffordnet_uses_the_can_nano_signature():
    training = _training_module()
    model = training.create_model("cliffordnet_12_2", training.TrainingConfig())
    assert len(model.blocks) == 12
    assert model.blocks[0].clifford_interaction_local.shifts == [1, 2]
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_433_188


def test_train_one_epoch_returns_can_style_percent_accuracy():
    train_one_epoch = _training_module().train_one_epoch
    model, loader, inputs, targets = _classification_fixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss, accuracy = train_one_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        epoch=1,
        total_epochs=1,
    )
    assert loss == F.cross_entropy(model(inputs), targets).item()
    assert accuracy == 100.0


def test_evaluate_prints_accuracy_and_saves_plain_state_dict(tmp_path, capsys):
    evaluate = _training_module().evaluate
    model, loader, _, _ = _classification_fixture()
    checkpoint = tmp_path / "best_model.pth"
    accuracy, best_accuracy = evaluate(
        model,
        loader,
        torch.device("cpu"),
        epoch=1,
        best_accuracy=0.0,
        save_path=checkpoint,
    )
    assert accuracy == 100.0
    assert best_accuracy == 100.0
    assert "Epoch 1 Test Acc: 100.00%" in capsys.readouterr().out
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == set(model.state_dict())
    assert "model" not in saved
