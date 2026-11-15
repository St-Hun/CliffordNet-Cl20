"""Device-independent training, evaluation, checkpointing, and run records."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .reproducibility import collect_environment, seed_everything


@dataclass(eq=True)
class Metrics:
    loss: float
    accuracy: float


@dataclass
class RunRecord:
    status: str = "incomplete"
    epochs: list[dict] = field(default_factory=list)
    seed: int = 0
    parameter_count: int = 0
    environment: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    git: dict = field(default_factory=dict)
    dataset_identity: dict = field(default_factory=dict)
    resolved_config: dict = field(default_factory=dict)


@dataclass
class VerificationReport:
    path: str
    model_name: str
    compatible: bool
    sha256: str | None
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    error: str | None = None


def _limited(loader, limit):
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def evaluate(model, loader, device="cpu", max_batches=None) -> Metrics:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = correct = count = 0
    with torch.no_grad():
        for inputs, targets in _limited(loader, max_batches):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss += criterion(logits, targets).item()
            correct += (logits.argmax(1) == targets).sum().item()
            count += targets.numel()
    return Metrics(loss / max(count, 1), correct / max(count, 1))


def _git_state() -> dict:
    def run(*args):
        result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_accuracy):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "best_accuracy": best_accuracy}, temporary)
    os.replace(temporary, path)


def fit(model, train_loader, eval_loader, output_dir: Path, *, epochs: int, lr: float,
        weight_decay: float = 0.0, seed: int = 0, device: str = "cpu", resume: Path | None = None,
        scheduler_epochs: int | None = None, max_train_batches: int | None = None,
        max_eval_batches: int | None = None, epoch_callback: Callable | None = None,
        dataset_identity: dict | None = None, resolved_config: dict | None = None) -> RunRecord:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_epochs or epochs)
    start_epoch, best_accuracy = 0, -1.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best_accuracy = int(checkpoint["epoch"]), float(checkpoint["best_accuracy"])
    record = RunRecord(seed=seed, parameter_count=sum(p.numel() for p in model.parameters()),
                       environment=collect_environment(), command=sys.argv, git=_git_state(),
                       dataset_identity=dataset_identity or {}, resolved_config=resolved_config or {})
    _atomic_json(output_dir / "run.json", asdict(record))
    criterion = nn.CrossEntropyLoss()
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            model.train()
            total_loss = correct = count = 0
            for inputs, targets in _limited(train_loader, max_train_batches):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * targets.numel()
                correct += (logits.argmax(1) == targets).sum().item()
                count += targets.numel()
            validation = evaluate(model, eval_loader, device, max_eval_batches)
            metrics = {"epoch": epoch, "train_loss": total_loss / max(count, 1),
                       "train_accuracy": correct / max(count, 1), "eval_loss": validation.loss,
                       "eval_accuracy": validation.accuracy, "lr": optimizer.param_groups[0]["lr"]}
            record.epochs.append(metrics)
            csv_path = output_dir / "metrics.csv"
            with csv_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)
            if validation.accuracy > best_accuracy:
                best_accuracy = validation.accuracy
                _save_checkpoint(output_dir / "checkpoint-best.pt", model, optimizer, scheduler, epoch, best_accuracy)
            scheduler.step()
            _save_checkpoint(output_dir / "checkpoint-final.pt", model, optimizer, scheduler, epoch, best_accuracy)
            _atomic_json(output_dir / "run.json", asdict(record))
            if epoch_callback:
                epoch_callback(epoch, metrics)
        record.status = "complete"
        for name in ("checkpoint-best.pt", "checkpoint-final.pt"):
            path = output_dir / name
            record.checkpoints[name] = {"sha256": _digest(path), "bytes": path.stat().st_size}
        _atomic_json(output_dir / "run.json", asdict(record))
        return record
    except BaseException:
        _atomic_json(output_dir / "run.json", asdict(record))
        raise


def verify_checkpoint(path: Path, model_name: str) -> VerificationReport:
    from .models.registry import build_model
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload.get("model_state_dict", payload.get("state_dict", payload)))
        result = build_model(model_name).load_state_dict(state, strict=False)
        compatible = not result.missing_keys and not result.unexpected_keys
        return VerificationReport(str(path), model_name, compatible, _digest(path), list(result.missing_keys), list(result.unexpected_keys))
    except Exception as error:  # noqa: BLE001 - verification reports incompatibility instead of crashing
        return VerificationReport(str(path), model_name, False, _digest(path) if path.is_file() else None, error=str(error))
