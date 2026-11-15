import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cliffordnet_cl20.engine import fit


def _fixture():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(12, 4, generator=generator)
    y = (x.sum(1) > 0).long()
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=4, shuffle=False), DataLoader(dataset, batch_size=4, shuffle=False)


def _model():
    torch.manual_seed(11)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def test_resume_matches_uninterrupted_training(tmp_path):
    train_loader, eval_loader = _fixture()
    full = _model()
    full_record = fit(full, train_loader, eval_loader, tmp_path / "full", epochs=2, lr=0.01, seed=3, scheduler_epochs=2)

    resumed = _model()
    first_record = fit(resumed, train_loader, eval_loader, tmp_path / "resume", epochs=1, lr=0.01, seed=3, scheduler_epochs=2)
    assert first_record.status == "complete"
    resumed_record = fit(resumed, train_loader, eval_loader, tmp_path / "resume", epochs=2, lr=0.01, seed=3,
                         resume=tmp_path / "resume" / "checkpoint-final.pt", scheduler_epochs=2)
    for left, right in zip(full.parameters(), resumed.parameters()):
        torch.testing.assert_close(left, right)
    assert full_record.epochs[-1] == resumed_record.epochs[-1]


def test_interrupted_run_remains_incomplete(tmp_path):
    train_loader, eval_loader = _fixture()

    def stop_after_first(epoch, _metrics):
        if epoch == 1:
            raise RuntimeError("simulated interruption")

    try:
        fit(_model(), train_loader, eval_loader, tmp_path / "stopped", epochs=2, lr=0.01, seed=3, epoch_callback=stop_after_first)
    except RuntimeError:
        pass
    record = json.loads((tmp_path / "stopped" / "run.json").read_text())
    assert record["status"] == "incomplete"
