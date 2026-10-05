"""
Train the independent surrogate classifier used by APS.

This model is trained separately from the victim training process. APS uses
it only to generate label-consistent adversarially perturbed samples, which
matches the paper's threat model: the attacker does not need the victim
model's exact parameters or access to its training loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from APS_Common import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    IndexedCIFAR10,
    ModifiedCIFARCNN,
    make_cifar10_transform,
    set_seed,
)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate loss and accuracy on a CIFAR-10 loader."""

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


def train_surrogate(
    base_dir: Path,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    seed: int = 42,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> Path:
    """
    Train and save the independent APS surrogate model.

    The checkpoint is written to:
        <base_dir>/cifar10_cnn_model/pretrained_cifar10_cnn_APS.pth
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    set_seed(seed)
    base_dir = Path(base_dir)
    data_dir = base_dir / "data" / "cifar10_data"
    checkpoint_dir = base_dir / "cifar10_cnn_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using APS surrogate device: {device}")

    train_dataset = IndexedCIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=make_cifar10_transform(
            normalize=True,
            train_augmentation=True,
        ),
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=make_cifar10_transform(
            normalize=True,
            train_augmentation=False,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ModifiedCIFARCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for _, images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_accuracy = 100.0 * correct / max(total, 1)
        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.2f}% | "
            f"Test Loss {test_loss:.4f} | "
            f"Test Acc {test_accuracy:.2f}%"
        )

    checkpoint_path = checkpoint_dir / "pretrained_cifar10_cnn_APS.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "ModifiedCIFARCNN",
            "num_classes": 10,
            "normalization_mean": CIFAR10_MEAN,
            "normalization_std": CIFAR10_STD,
            "seed": seed,
        },
        checkpoint_path,
    )
    print(f"APS surrogate checkpoint saved to: {checkpoint_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the independent surrogate classifier for APS."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("./xie/FNN_Shapley"),
        help="Experiment root containing data/ and cifar10_cnn_model/.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_surrogate(
        base_dir=args.base_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

