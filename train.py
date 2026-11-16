"""CIFAR-100 training loop based on the original CliffordNet/CAN script."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from cliffordnet_cl20.models.registry import build_model
from cliffordnet_cl20.reproducibility import seed_everything

MODEL_ALIASES = {
    "cliffordnet_12_2": "official_nano",
    "cliffordnet_12_5": "official_lite",
    "cliffordnet_32_3": "official_d32",
    "cliffordnet_64_5": "official_d64",
    "mustardnet_16_2": "roll_shift_d16_s2",
    "mustardnet_18_5": "roll_shift_d18_s5",
    "mustardnet_25_3": "roll_shift_d25_s3",
    "mustardnet_32_5": "roll_shift_d32_s5",
    "mustardnet_48_2": "roll_shift_d48_s2",
    "mustardnet_52_3": "roll_shift_d52_s3",
    "shuffle_14_2_4_8": "shuffle_d14_g2_4_8",
    "shuffle_26_2_4_8": "shuffle_d26_g2_4_8",
}

PROJECT_MODELS = (
    "official_nano",
    "official_lite",
    "official_d32",
    "official_d64",
    "roll_shift_d16_s2",
    "roll_shift_d18_s5",
    "roll_shift_d25_s3",
    "roll_shift_d32_s5",
    "roll_shift_d48_s2",
    "roll_shift_d52_s3",
    "shuffle_d14_g2_4_8",
    "shuffle_d26_g2_4_8",
)

BASELINE_MODELS = ("vit_tiny", "mobilenet_v2", "shufflenet_v2_1_5")


@dataclass
class TrainingConfig:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 200
    weight_decay: float = 0.1
    num_workers: int = field(
        default_factory=lambda: 4 if torch.cuda.is_available() else 0
    )
    device: str = field(
        default_factory=lambda: (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    )
    seed: int = 42
    data_root: str = "./data"
    random_erasing_prob: float = 0.25
    num_classes: int = 100
    img_size: int = 32
    patch_size: int = 2
    embed_dim: int = 128
    warmup_epochs: int = 0
    warmup_start_lr: float = 1e-6
    save_path: str = "best_model.pth"


def get_device(device_name: str) -> torch.device:
    print(f"Using Device: {device_name.upper()}")
    return torch.device(device_name)


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def get_transforms(config: TrainingConfig):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(config.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=config.random_erasing_prob),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    return train_transform, test_transform


def get_dataloaders(config: TrainingConfig, download: bool = True):
    print("Preparing Data...")
    train_transform, test_transform = get_transforms(config)
    train_set = torchvision.datasets.CIFAR100(
        root=config.data_root,
        train=True,
        download=download,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR100(
        root=config.data_root,
        train=False,
        download=download,
        transform=test_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=100)
    for inputs, labels in progress:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = outputs.argmax(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress.set_postfix(
            loss=f"{running_loss / total:.4f}", acc=f"{100.0 * correct / total:.2f}%"
        )
    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device, epoch, best_accuracy, save_path="best_model.pth"):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = outputs.argmax(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch} Test Acc: {accuracy:.2f}%")
    if accuracy > best_accuracy:
        print(f"New record: {best_accuracy:.2f}% -> {accuracy:.2f}%")
        best_accuracy = accuracy
        torch.save(model.state_dict(), save_path)
    return accuracy, best_accuracy


def create_model(model_name: str, config: TrainingConfig):
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    if resolved_name in PROJECT_MODELS:
        return build_model(resolved_name, num_classes=config.num_classes)
    if model_name == "vit_tiny":
        import timm

        return timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=False,
            num_classes=config.num_classes,
            img_size=config.img_size,
            patch_size=config.patch_size,
            mlp_ratio=1.0,
            drop_path_rate=0.3,
        )
    if model_name == "mobilenet_v2":
        import timm

        model = timm.create_model(
            "mobilenetv2_100",
            pretrained=False,
            num_classes=config.num_classes,
            drop_path_rate=0.3,
        )
        model.conv_stem.stride = (1, 1)
        return model
    if model_name == "shufflenet_v2_1_5":
        model = torchvision.models.shufflenet_v2_x1_5(
            weights=None, num_classes=config.num_classes
        )
        model.conv1[0].stride = (1, 1)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown model type: {model_name}")


def create_scheduler(optimizer, config: TrainingConfig):
    if config.warmup_epochs <= 0:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.warmup_start_lr / config.lr,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.warmup_epochs],
    )


def run_training(
    config: TrainingConfig, model_name: str, download: bool, wandb_mode: str
):
    import wandb

    seed_everything(config.seed, deterministic=True)
    device = get_device(config.device)
    config.save_path = str(Path(config.save_path))
    wandb.init(
        project="CliffordNet-CIFAR100",
        name=model_name,
        config=asdict(config),
        mode=wandb_mode,
    )
    train_loader, test_loader = get_dataloaders(config, download=download)
    print(f"Initializing Model: {model_name}...")
    model = create_model(model_name, config).to(device)
    print(f"Model built. Learnable Parameters: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = create_scheduler(optimizer, config)
    print(f"Start training {model_name} for {config.epochs} epochs...")
    start_time = time.time()
    best_accuracy = 0.0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            config.epochs,
        )
        test_accuracy, best_accuracy = evaluate(
            model,
            test_loader,
            device,
            epoch,
            best_accuracy,
            save_path=config.save_path,
        )
        scheduler.step()
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_accuracy,
                "test_acc": test_accuracy,
                "best_acc": best_accuracy,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"Training Finished. Total time: {elapsed_minutes:.2f} mins")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-100 Training for CliffordNet and Cl(2,0) variants"
    )
    parser.add_argument(
        "--model",
        default="roll_shift_d25_s3",
        choices=sorted(PROJECT_MODELS + BASELINE_MODELS + tuple(MODEL_ALIASES)),
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", "--warmup_epochs", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        data_root=args.data_dir,
        warmup_epochs=args.warmup_epochs,
    )
    if args.device:
        config.device = args.device
    config.save_path = args.output or f"best_model_{args.model}.pth"
    run_training(config, args.model, args.download, args.wandb_mode)


if __name__ == "__main__":
    main()
