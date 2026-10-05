"""
First-order In-Run Data Shapley (IRDS) flow for PPIS.

This script is the explicit PPIS victim-side training stage. It consumes the
PPIS artifacts produced by ``Backdoor_Poisoning_Strategy_PPIS.py``:

    results1/poisoned_mapping.csv
    save1/random_<ratio>pct_backdoored_images/

It trains:

1. A clean MNIST model.
2. A model whose selected samples are replaced by PPIS images and poisoned
   labels.

The only attribution outputs are:

    shapley_original_PPIS.csv
    shapley_ppis.csv

Both files use the original MNIST sample index as their key, so they can be
passed directly to downstream metric code together with the PPIS mapping.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


@dataclass(frozen=True)
class PPISPoisonRecord:
    original_index: int
    original_label: int
    poisoned_label: int
    poisoned_image_path: Path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


class IndexedMNIST(datasets.MNIST):
    """MNIST dataset that returns ``(original_index, image, label)``."""

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return index, image, label


class PPISPoisonedMNIST(Dataset):
    """Replace selected MNIST images and labels while preserving indices."""

    def __init__(
        self,
        clean_dataset: IndexedMNIST,
        poison_records: Mapping[int, PPISPoisonRecord],
    ) -> None:
        self.clean_dataset = clean_dataset
        self.poison_records = dict(poison_records)
        self.transform = mnist_transform()

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, index: int):
        if index not in self.poison_records:
            return self.clean_dataset[index]

        record = self.poison_records[index]
        image = transforms.ToTensor()(
            # The generated PPIS image is stored in pixel space.
            Image.open(record.poisoned_image_path).convert("L")
        )
        image = transforms.Normalize(MNIST_MEAN, MNIST_STD)(image)
        return index, image, record.poisoned_label


class SmallFNN(nn.Module):
    """The MNIST classifier used by the PPIS experiment."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.relu(self.fc1(self.flatten(images)))
        return self.fc2(features)


def _find_poisoned_image(
    poisoned_dir: Path,
    original_index: int,
    original_label: int,
    row: Mapping[str, str],
) -> Path:
    """Resolve an old or new PPIS mapping row to an image path."""

    path_keys = (
        "poisoned_image_path",
        "backdoor_image_path",
        "poisoned_path",
        "filename",
    )
    for key in path_keys:
        value = row.get(key)
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = poisoned_dir / candidate
        if candidate.is_file():
            return candidate

    exact_candidates = [
        poisoned_dir / f"backdoor_{original_index}_label_{original_label}.png",
        poisoned_dir / f"poisoned_{original_index}_label_{original_label}.png",
    ]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate

    glob_patterns = (
        f"backdoor_{original_index}_label_*.png",
        f"poisoned_{original_index}_label_*.png",
        f"*{original_index}*poison*.png",
    )
    for pattern in glob_patterns:
        matches = sorted(poisoned_dir.rglob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find the PPIS image for sample {original_index} under "
        f"{poisoned_dir}."
    )


def load_ppis_mapping(
    mapping_path: Path,
    poisoned_dir: Path,
    original_labels: Sequence[int],
) -> Dict[int, PPISPoisonRecord]:
    """Load PPIS replacements from the mapping generated by PPIS."""

    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"PPIS mapping not found: {mapping_path}. "
            "Run Backdoor_Poisoning_Strategy_PPIS.py first."
        )
    if not poisoned_dir.is_dir():
        raise FileNotFoundError(f"PPIS poisoned-image directory not found: {poisoned_dir}")

    records: Dict[int, PPISPoisonRecord] = {}
    with mapping_path.open("r", newline="", encoding="utf-8") as mapping_file:
        reader = csv.DictReader(mapping_file)
        fields = set(reader.fieldnames or [])
        if "original_index" not in fields:
            raise ValueError("PPIS mapping must contain an original_index column.")
        if "poisoned_label" not in fields:
            raise ValueError("PPIS mapping must contain a poisoned_label column.")

        for row in reader:
            index = int(row["original_index"])
            if not 0 <= index < len(original_labels):
                raise ValueError(f"Invalid PPIS original index: {index}")
            original_label = int(original_labels[index])
            if "original_label" in fields and row.get("original_label"):
                mapped_label = int(row["original_label"])
                if mapped_label != original_label:
                    raise ValueError(
                        f"PPIS label mismatch for sample {index}: "
                        f"mapping={mapped_label}, dataset={original_label}."
                    )
            poisoned_label = int(row["poisoned_label"])
            if not 0 <= poisoned_label < 10:
                raise ValueError(f"Invalid PPIS poisoned label: {poisoned_label}")
            poisoned_path = _find_poisoned_image(
                poisoned_dir=poisoned_dir,
                original_index=index,
                original_label=original_label,
                row=row,
            )
            records[index] = PPISPoisonRecord(
                original_index=index,
                original_label=original_label,
                poisoned_label=poisoned_label,
                poisoned_image_path=poisoned_path,
            )

    if not records:
        raise ValueError(f"No PPIS records found in {mapping_path}.")
    return records


def _flatten_gradient_dict(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [gradient.reshape(gradient.size(0), -1) for gradient in gradients.values()],
        dim=1,
    )


def per_sample_grads(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute one flattened parameter gradient per training sample."""

    try:
        from torch.func import functional_call, grad, vmap

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def loss_fn(functional_params, functional_buffers, image, label):
            logits = functional_call(
                model,
                (functional_params, functional_buffers),
                (image.unsqueeze(0),),
            )
            return criterion(logits, label.unsqueeze(0))

        gradients = vmap(grad(loss_fn), (None, None, 0, 0))(
            params,
            buffers,
            images,
            labels,
        )
        return _flatten_gradient_dict(gradients)
    except (ImportError, RuntimeError, TypeError):
        parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        flattened: List[torch.Tensor] = []
        for image, label in zip(images, labels):
            logits = model(image.unsqueeze(0))
            loss = criterion(logits, label.unsqueeze(0))
            gradients = torch.autograd.grad(
                loss,
                parameters,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            flattened.append(
                torch.cat(
                    [
                        torch.zeros_like(parameter).reshape(-1)
                        if gradient is None
                        else gradient.reshape(-1)
                        for parameter, gradient in zip(parameters, gradients)
                    ]
                )
            )
        return torch.stack(flattened, dim=0)


def compute_validation_gradient(
    model: nn.Module,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute the validation gradient used by first-order IRDS."""

    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    was_training = model.training
    model.train()
    try:
        logits = model(val_images)
        validation_loss = criterion(logits, val_labels)
        gradients = torch.autograd.grad(
            validation_loss,
            parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        filled = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
        return parameters_to_vector(filled).detach().unsqueeze(0)
    finally:
        model.train(was_training)


def build_validation_batch(
    test_dataset: datasets.MNIST,
    validation_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if validation_size <= 0:
        raise ValueError("validation_size must be positive.")
    validation_size = min(validation_size, len(test_dataset))
    indices = sorted(random.Random(seed).sample(
        range(len(test_dataset)),
        validation_size,
    ))
    loader = DataLoader(
        Subset(test_dataset, indices),
        batch_size=validation_size,
        shuffle=False,
        num_workers=0,
    )
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)


def train_and_save_shapley(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    sample_count: int,
    output_path: Path,
    model_name: str,
) -> Path:
    """Train one model and save its complete sample-indexed IRDS CSV."""

    data_shapley = torch.zeros(sample_count, device=device)
    print(f"Training {model_name} and accumulating first-order IRDS values...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0

        for indices, images, labels in train_loader:
            indices = indices.to(device, dtype=torch.long)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            sample_gradients = per_sample_grads(model, images, labels, criterion)
            validation_gradient = compute_validation_gradient(
                model,
                val_images,
                val_labels,
                criterion,
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            increments = -learning_rate * (
                sample_gradients @ validation_gradient.T
            ).squeeze(1)
            data_shapley[indices] += increments.detach()

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)

        print(
            f"[{model_name}] Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss {running_loss / max(total, 1):.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["sample_index", "shapley_value"])
        for index, value in enumerate(data_shapley.detach().cpu().tolist()):
            writer.writerow([index, float(value)])
    print(f"{model_name} Shapley values saved to: {output_path}")
    return output_path


def make_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def run_ppis_irds(
    base_dir: Path,
    output_dir: Path,
    mapping_path: Optional[Path] = None,
    poisoned_dir: Optional[Path] = None,
    poison_ratio: float = 0.01,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    validation_size: int = 1000,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Run clean and PPIS-poisoned MNIST IRDS training."""

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 < poison_ratio <= 1.0:
        raise ValueError("poison_ratio must be in (0, 1].")

    set_seed(seed)
    base_dir = Path(base_dir)
    data_dir = base_dir / "data"
    mapping_path = mapping_path or (base_dir / "results1" / "poisoned_mapping.csv")
    poisoned_dir = poisoned_dir or (
        base_dir
        / "save1"
        / f"random_{int(poison_ratio * 100)}pct_backdoored_images"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = mnist_transform()
    clean_train_dataset = IndexedMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    original_labels = [
        int(clean_train_dataset.targets[index])
        for index in range(len(clean_train_dataset))
    ]
    poison_records = load_ppis_mapping(
        mapping_path=mapping_path,
        poisoned_dir=poisoned_dir,
        original_labels=original_labels,
    )

    poisoned_dataset = PPISPoisonedMNIST(
        clean_dataset=clean_train_dataset,
        poison_records=poison_records,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PPIS IRDS device: {device}")
    print(
        f"Training samples: {len(clean_train_dataset)}, "
        f"PPIS replacements: {len(poison_records)}, "
        f"validation size: {min(validation_size, len(test_dataset))}"
    )

    val_images, val_labels = build_validation_batch(
        test_dataset=test_dataset,
        validation_size=validation_size,
        seed=seed + 1,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")

    clean_loader = make_loader(
        clean_train_dataset,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    poisoned_loader = make_loader(
        poisoned_dataset,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    set_seed(seed)
    clean_model = SmallFNN().to(device)
    clean_optimizer = optim.SGD(clean_model.parameters(), lr=learning_rate)
    original_path = train_and_save_shapley(
        model=clean_model,
        train_loader=clean_loader,
        optimizer=clean_optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        val_images=val_images,
        val_labels=val_labels,
        sample_count=len(clean_train_dataset),
        output_path=output_dir / "shapley_original_PPIS.csv",
        model_name="Original_PPIS",
    )

    set_seed(seed)
    poisoned_model = SmallFNN().to(device)
    poisoned_optimizer = optim.SGD(poisoned_model.parameters(), lr=learning_rate)
    ppis_path = train_and_save_shapley(
        model=poisoned_model,
        train_loader=poisoned_loader,
        optimizer=poisoned_optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        val_images=val_images,
        val_labels=val_labels,
        sample_count=len(clean_train_dataset),
        output_path=output_dir / "shapley_ppis.csv",
        model_name="PPIS",
    )
    return original_path, ppis_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute clean/PPIS first-order IRDS Shapley CSV files."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("./xie/FNN_Shapley"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <base-dir>/results_PPIS.",
    )
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--poisoned-dir", type=Path, default=None)
    parser.add_argument("--poison-ratio", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.base_dir / "results_PPIS")
    original_path, ppis_path = run_ppis_irds(
        base_dir=args.base_dir,
        output_dir=output_dir,
        mapping_path=args.mapping_path,
        poisoned_dir=args.poisoned_dir,
        poison_ratio=args.poison_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    print(f"Clean IRDS CSV: {original_path}")
    print(f"PPIS IRDS CSV: {ppis_path}")


if __name__ == "__main__":
    main()
