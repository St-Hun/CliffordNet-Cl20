from pathlib import Path

import torch

from cliffordnet_cl20.results import CheckpointCandidate, classify_checkpoint


def _save(path: Path, shapes: dict[str, tuple[int, ...]]) -> None:
    torch.save({key: torch.zeros(shape) for key, shape in shapes.items()}, path)


def test_checkpoint_exactly_matches_one_candidate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    shapes = {"head.weight": (3, 4), "head.bias": (3,)}
    _save(checkpoint, shapes)
    result = classify_checkpoint(checkpoint, [CheckpointCandidate("match", shapes), CheckpointCandidate("wrong", {"x": (1,)})])
    assert result.status == "matched"
    assert result.matches == ("match",)


def test_checkpoint_ambiguity_is_quarantined(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    shapes = {"weight": (2, 2)}
    _save(checkpoint, shapes)
    result = classify_checkpoint(checkpoint, [CheckpointCandidate("a", shapes), CheckpointCandidate("b", shapes)])
    assert result.status == "quarantined"
    assert result.matches == ("a", "b")


def test_checkpoint_mismatch_and_corruption_are_reported(tmp_path: Path) -> None:
    mismatch = tmp_path / "mismatch.pth"
    _save(mismatch, {"weight": (3, 2)})
    candidate = CheckpointCandidate("expected", {"weight": (2, 2)})
    assert classify_checkpoint(mismatch, [candidate]).status == "unmatched"

    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"not a checkpoint")
    assert classify_checkpoint(corrupt, [candidate]).status == "corrupt"
