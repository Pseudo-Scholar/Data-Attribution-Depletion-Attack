"""ViT-B/16 victim training with first-order IRDS for HTBM.

This script consumes ``htbm_mapping.csv`` and the hidden-trigger poison
images produced by ``Hidden_Trigger_Backdoor_Strategy_HTBM.py``. It runs two
paired victim trainings:

1. the clean ImageNet-100 training set;
2. the same training set with label-consistent hidden-trigger poison images
   appended to the target class by default (or target samples replaced with
   ``--integration-mode replace``).

The paired runs use the same sample indices, validation batch, initial model
state, optimizer, schedule, and data-loader seed. The only intended
difference is the image content at the mapped target indices. The script
writes:

    shapley_original_HTBM.csv
    shapley_htbm.csv

By default, IRDS uses the exact gradient dot product for the ViT
classification head. This is the practical setting for ViT-B/16. The
``--irds-parameter-scope all`` option computes the full-parameter dot product
with an autograd-per-sample fallback; it is substantially more expensive but
is available for small-scale verification.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from Hidden_Trigger_Backdoor_Strategy_HTBM_Common import (
    ImageNetSample,
    build_vit_b16,
    clone_state_dict_to_cpu,
    discover_imagenet_samples,
    get_vit_head,
    load_image_tensor,
    load_state_dict_clone,
    make_victim_transform,
    resolve_split_root,
    set_seed,
    vit_features,
)


@dataclass(frozen=True)
class HTBMPoisonRecord:
    """A generated HTBM image associated with an original train index."""

    poison_index: int
    target_original_index: int
    target_label: int
    poisoned_path: Path


class IndexedImageNetDataset(Dataset):
    """ImageNet dataset returning ``(stable_index, image, label)``."""

    def __init__(
        self,
        samples: Sequence[ImageNetSample],
        transform,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, position: int):
        sample = self.samples[position]
        image = load_image_tensor(sample.path, self.transform)
        return sample.index, image, sample.label


class HTBMPoisonedImageNetDataset(Dataset):
    """Replace selected ImageNet samples with generated HTBM images."""

    def __init__(
        self,
        clean_dataset: IndexedImageNetDataset,
        poison_records: Mapping[int, HTBMPoisonRecord],
        transform,
    ) -> None:
        self.clean_dataset = clean_dataset
        self.poison_records = dict(poison_records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, position: int):
        sample = self.clean_dataset.samples[position]
        record = self.poison_records.get(sample.index)
        if record is None:
            return self.clean_dataset[position]
        image = load_image_tensor(record.poisoned_path, self.transform)
        return sample.index, image, record.target_label


class HTBMAppendedImageNetDataset(Dataset):
    """Append either clean target copies or HTBM images with stable indices."""

    def __init__(
        self,
        clean_dataset: IndexedImageNetDataset,
        poison_records: Sequence[HTBMPoisonRecord],
        transform,
        use_poison: bool,
    ) -> None:
        self.clean_dataset = clean_dataset
        self.poison_records = sorted(
            poison_records,
            key=lambda record: record.poison_index,
        )
        self.transform = transform
        self.use_poison = use_poison
        self.base_count = len(clean_dataset)
        self.position_by_index = {
            sample.index: position
            for position, sample in enumerate(clean_dataset.samples)
        }

    def __len__(self) -> int:
        return self.base_count + len(self.poison_records)

    def __getitem__(self, position: int):
        if position < self.base_count:
            return self.clean_dataset[position]

        record = self.poison_records[position - self.base_count]
        appended_index = self.base_count + record.poison_index
        if self.use_poison:
            image_path = record.poisoned_path
        else:
            original_position = self.position_by_index[
                record.target_original_index
            ]
            image_path = self.clean_dataset.samples[original_position].path
        image = load_image_tensor(image_path, self.transform)
        return appended_index, image, record.target_label


def load_htbm_mapping(
    mapping_path: Path,
    samples_by_index: Mapping[int, ImageNetSample],
    num_classes: int,
) -> Dict[int, HTBMPoisonRecord]:
    """Load and validate the target-index replacement mapping."""

    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"HTBM mapping not found: {mapping_path}. "
            "Run Hidden_Trigger_Backdoor_Strategy_HTBM.py first."
        )

    records: Dict[int, HTBMPoisonRecord] = {}
    mapping_root = mapping_path.parent
    with mapping_path.open("r", newline="", encoding="utf-8") as mapping_file:
        reader = csv.DictReader(mapping_file)
        required = {
            "target_original_index",
            "target_label",
            "poisoned_path",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "htbm_mapping.csv is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            poison_index = int(row.get("poison_index", len(records)))
            target_index = int(row["target_original_index"])
            target_label = int(row["target_label"])
            if target_index not in samples_by_index:
                raise ValueError(
                    f"HTBM target index {target_index} is not present in the "
                    "current training split. Use the same ImageNet directory "
                    "and class order used during poison generation."
                )
            expected_label = samples_by_index[target_index].label
            if expected_label != target_label:
                raise ValueError(
                    f"HTBM label mismatch for sample {target_index}: "
                    f"mapping={target_label}, dataset={expected_label}."
                )
            if not 0 <= target_label < num_classes:
                raise ValueError(
                    f"HTBM target label {target_label} is outside "
                    f"[0, {num_classes})."
                )
            if target_index in records:
                raise ValueError(
                    f"Duplicate HTBM target_original_index: {target_index}"
                )

            poisoned_path = Path(row["poisoned_path"])
            if not poisoned_path.is_absolute():
                poisoned_path = mapping_root / poisoned_path
            if not poisoned_path.is_file():
                raise FileNotFoundError(
                    f"HTBM poison image not found: {poisoned_path}"
                )

            records[target_index] = HTBMPoisonRecord(
                poison_index=poison_index,
                target_original_index=target_index,
                target_label=target_label,
                poisoned_path=poisoned_path,
            )

    if not records:
        raise ValueError(f"No HTBM poison records found in {mapping_path}.")
    poison_indices = [record.poison_index for record in records.values()]
    if len(set(poison_indices)) != len(poison_indices):
        raise ValueError("Duplicate poison_index values in htbm_mapping.csv.")
    return records


def choose_training_positions(
    sample_count: int,
    max_train_samples: Optional[int],
    required_positions: Sequence[int],
    seed: int,
) -> List[int]:
    """Select a reproducible training subset while retaining all poisons."""

    if max_train_samples is None:
        selected = list(range(sample_count))
    else:
        if max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive.")
        rng = random.Random(seed)
        count = min(sample_count, max_train_samples)
        selected = sorted(rng.sample(range(sample_count), count))

    selected = sorted(set(selected).union(required_positions))
    return selected


def choose_validation_positions(
    candidate_positions: Sequence[int],
    validation_size: int,
    seed: int,
) -> List[int]:
    if validation_size <= 0:
        raise ValueError("validation_size must be positive.")
    if not candidate_positions:
        raise ValueError("No candidate validation positions are available.")
    rng = random.Random(seed)
    count = min(validation_size, len(candidate_positions))
    return sorted(rng.sample(list(candidate_positions), count))


def build_validation_batch(
    dataset: IndexedImageNetDataset,
    positions: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        Subset(dataset, list(positions)),
        batch_size=len(positions),
        shuffle=False,
        num_workers=0,
    )
    images, labels = next(iter(loader))[1:]
    return images.to(device), labels.to(device)


def make_loader(
    dataset: Dataset,
    positions: Sequence[int],
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(positions)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def configure_trainable_scope(model: nn.Module, scope: str) -> None:
    """Configure full ViT fine-tuning or linear-head-only fine-tuning."""

    if scope not in {"all", "head"}:
        raise ValueError("trainable scope must be 'all' or 'head'.")
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in get_vit_head(model).parameters():
        parameter.requires_grad_(True)


def compute_head_validation_gradient(
    model: nn.Module,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute validation gradients for the final linear ViT head."""

    head = get_vit_head(model)
    with torch.no_grad():
        features = vit_features(model, val_images)
    logits = head(features)
    validation_loss = criterion(logits, val_labels)
    gradients = torch.autograd.grad(
        validation_loss,
        (head.weight, head.bias),
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    weight_gradient = (
        torch.zeros_like(head.weight)
        if gradients[0] is None
        else gradients[0].detach()
    )
    bias_gradient = (
        torch.zeros_like(head.bias)
        if gradients[1] is None
        else gradients[1].detach()
    )
    return weight_gradient, bias_gradient


def per_sample_head_dots(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    validation_gradient: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Compute exact per-sample gradient dot products for a linear head."""

    head = get_vit_head(model)
    with torch.no_grad():
        features = vit_features(model, images)
        logits = head(features)
        probabilities = torch.softmax(logits, dim=1)

    one_hot = torch.zeros_like(probabilities)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    loss_gradient_logits = probabilities - one_hot
    validation_weight, validation_bias = validation_gradient
    validation_direction = (
        features @ validation_weight.transpose(0, 1)
        + validation_bias.unsqueeze(0)
    )
    return (loss_gradient_logits * validation_direction).sum(dim=1)


def compute_all_validation_gradient(
    model: nn.Module,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, ...]:
    """Compute the full-parameter validation gradient."""

    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    logits = model(val_images)
    validation_loss = criterion(logits, val_labels)
    gradients = torch.autograd.grad(
        validation_loss,
        parameters,
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients)
    )


def per_sample_all_dots(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    validation_gradient: Tuple[torch.Tensor, ...],
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute full-parameter dots by an exact, memory-conservative loop."""

    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    dots: List[torch.Tensor] = []
    for image, label in zip(images, labels):
        logits = model(image.unsqueeze(0))
        loss = criterion(logits, label.unsqueeze(0))
        gradients = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        dot = torch.zeros((), device=images.device)
        for gradient, direction in zip(gradients, validation_gradient):
            if gradient is not None:
                dot = dot + (gradient * direction).sum()
        dots.append(dot.detach())
    return torch.stack(dots)


def make_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> optim.lr_scheduler.LambdaLR:
    """Create a linear-warmup/cosine-decay learning-rate schedule."""

    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def learning_rate_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(
            total_steps - warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)


def train_and_save_shapley(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    global_sample_count: int,
    output_path: Path,
    model_name: str,
    irds_parameter_scope: str,
) -> Path:
    """Train one victim model and accumulate first-order IRDS values."""

    if irds_parameter_scope not in {"head", "all"}:
        raise ValueError("irds_parameter_scope must be 'head' or 'all'.")

    data_shapley = torch.zeros(global_sample_count, device=device)
    print(
        f"Training {model_name}; IRDS parameter scope="
        f"{irds_parameter_scope}."
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for indices, images, labels in train_loader:
            indices = indices.to(device, dtype=torch.long)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if irds_parameter_scope == "head":
                validation_gradient = compute_head_validation_gradient(
                    model,
                    val_images,
                    val_labels,
                    criterion,
                )
                dots = per_sample_head_dots(
                    model,
                    images,
                    labels,
                    validation_gradient,
                )
            else:
                validation_gradient_all = compute_all_validation_gradient(
                    model,
                    val_images,
                    val_labels,
                    criterion,
                )
                dots = per_sample_all_dots(
                    model,
                    images,
                    labels,
                    validation_gradient_all,
                    criterion,
                )

            learning_rate = float(optimizer.param_groups[0]["lr"])
            data_shapley[indices] += -learning_rate * dots.detach()

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_accuracy = 100.0 * correct / max(total, 1)
        print(
            f"[{model_name}] Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_accuracy:.2f}% | "
            f"LR {optimizer.param_groups[0]['lr']:.8f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["sample_index", "shapley_value"])
        for sample_index, value in enumerate(
            data_shapley.detach().cpu().tolist()
        ):
            writer.writerow([sample_index, float(value)])

    print(f"{model_name} Shapley values saved to: {output_path}")
    return output_path


def _make_initial_state(
    num_classes: int,
    pretrained: bool,
    weights_path: Optional[Path],
    seed: int,
) -> Dict[str, torch.Tensor]:
    set_seed(seed)
    base_model = build_vit_b16(
        num_classes=num_classes,
        pretrained=pretrained,
        weights_path=weights_path,
    )
    return clone_state_dict_to_cpu(base_model)


def _build_victim_from_state(
    initial_state: Mapping[str, torch.Tensor],
    num_classes: int,
    trainable_scope: str,
    device: torch.device,
) -> nn.Module:
    model = build_vit_b16(
        num_classes=num_classes,
        pretrained=False,
        weights_path=None,
    )
    load_state_dict_clone(model, initial_state)
    configure_trainable_scope(model, trainable_scope)
    return model.to(device)


def run_htbm_irds(
    train_root: Path,
    poison_root: Path,
    output_dir: Path,
    mapping_path: Optional[Path] = None,
    val_root: Optional[Path] = None,
    image_size: int = 224,
    epochs: int = 300,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    validation_size: int = 512,
    seed: int = 42,
    num_workers: int = 0,
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
    trainable_scope: str = "all",
    irds_parameter_scope: str = "head",
    integration_mode: str = "append",
    augmentation: bool = True,
    max_train_samples: Optional[int] = None,
    device: Optional[str] = None,
    skip_original: bool = False,
    skip_htbm: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Run clean and HTBM-poisoned ViT training with first-order IRDS."""

    if skip_original and skip_htbm:
        raise ValueError("At least one of skip_original and skip_htbm must run.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if trainable_scope not in {"all", "head"}:
        raise ValueError("trainable_scope must be 'all' or 'head'.")
    if irds_parameter_scope not in {"head", "all"}:
        raise ValueError("irds_parameter_scope must be 'head' or 'all'.")
    if integration_mode not in {"append", "replace"}:
        raise ValueError("integration_mode must be 'append' or 'replace'.")

    set_seed(seed)
    train_split_root = resolve_split_root(train_root, "train")
    train_samples, class_names, class_to_idx = discover_imagenet_samples(
        train_split_root
    )
    samples_by_index = {sample.index: sample for sample in train_samples}
    if len(samples_by_index) != len(train_samples):
        raise ValueError("ImageNet training sample indices are not unique.")

    mapping_path = mapping_path or (Path(poison_root) / "htbm_mapping.csv")
    poison_records = load_htbm_mapping(
        mapping_path=Path(mapping_path),
        samples_by_index=samples_by_index,
        num_classes=len(class_names),
    )
    poison_positions = sorted(poison_records)

    base_selected_positions = choose_training_positions(
        sample_count=len(train_samples),
        max_train_samples=max_train_samples,
        required_positions=poison_positions,
        seed=seed,
    )

    train_transform = make_victim_transform(
        image_size=image_size,
        training=True,
        augmentation=augmentation,
    )
    clean_dataset = IndexedImageNetDataset(
        train_samples,
        transform=train_transform,
    )
    poison_record_list = sorted(
        poison_records.values(),
        key=lambda record: record.poison_index,
    )
    if integration_mode == "append":
        expected_indices = list(range(len(poison_record_list)))
        actual_indices = [
            record.poison_index for record in poison_record_list
        ]
        if actual_indices != expected_indices:
            raise ValueError(
                "append integration requires contiguous poison_index values "
                "starting at zero."
            )
        appended_positions = [
            len(train_samples) + index
            for index in range(len(poison_record_list))
        ]
        selected_positions = sorted(
            set(base_selected_positions).union(appended_positions)
        )
        clean_training_dataset = HTBMAppendedImageNetDataset(
            clean_dataset=clean_dataset,
            poison_records=poison_record_list,
            transform=train_transform,
            use_poison=False,
        )
        poisoned_dataset = HTBMAppendedImageNetDataset(
            clean_dataset=clean_dataset,
            poison_records=poison_record_list,
            transform=train_transform,
            use_poison=True,
        )
    else:
        selected_positions = base_selected_positions
        appended_positions = []
        clean_training_dataset = clean_dataset
        poisoned_dataset = HTBMPoisonedImageNetDataset(
            clean_dataset=clean_dataset,
            poison_records=poison_records,
            transform=train_transform,
        )

    if val_root is not None:
        val_split_root = resolve_split_root(val_root, "val")
        val_samples, _, _ = discover_imagenet_samples(
            val_split_root,
            class_to_idx=class_to_idx,
        )
        validation_dataset = IndexedImageNetDataset(
            val_samples,
            transform=make_victim_transform(
                image_size=image_size,
                training=False,
                augmentation=False,
            ),
        )
        validation_positions = choose_validation_positions(
            list(range(len(val_samples))),
            validation_size=validation_size,
            seed=seed + 1,
        )
        train_positions = selected_positions
    else:
        non_poison_candidates = [
            position
            for position in base_selected_positions
            if train_samples[position].index not in poison_records
        ]
        if not non_poison_candidates:
            non_poison_candidates = base_selected_positions
        validation_positions = choose_validation_positions(
            non_poison_candidates,
            validation_size=validation_size,
            seed=seed + 1,
        )
        validation_dataset = IndexedImageNetDataset(
            train_samples,
            transform=make_victim_transform(
                image_size=image_size,
                training=False,
                augmentation=False,
            ),
        )
        validation_set = set(validation_positions)
        base_train_positions = [
            position
            for position in base_selected_positions
            if position not in validation_set
        ]
        poison_position_set = {
            position
            for position, sample in enumerate(train_samples)
            if sample.index in poison_records
        }
        if integration_mode == "append":
            train_positions = sorted(
                set(base_train_positions).union(appended_positions)
            )
        else:
            train_positions = sorted(
                set(base_train_positions).union(poison_position_set)
            )

    if not train_positions:
        raise ValueError("No training positions remain after validation split.")

    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    val_images, val_labels = build_validation_batch(
        validation_dataset,
        validation_positions,
        selected_device,
    )
    if integration_mode == "append":
        active_poison_count = len(
            set(train_positions).intersection(appended_positions)
        )
    else:
        active_poison_count = len(
            {
                train_samples[position].index
                for position in train_positions
            }.intersection(poison_records)
        )
    print(
        f"HTBM IRDS device: {selected_device}; classes={len(class_names)}, "
        f"integration={integration_mode}, "
        f"train samples={len(train_positions)}/{len(clean_training_dataset)}, "
        f"validation samples={len(validation_positions)}, "
        f"poisons active={active_poison_count}"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_state = _make_initial_state(
        num_classes=len(class_names),
        pretrained=pretrained,
        weights_path=weights_path,
        seed=seed,
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")
    total_steps = max(
        1,
        epochs * int(np.ceil(len(train_positions) / batch_size)),
    )
    warmup_steps = warmup_epochs * int(
        np.ceil(len(train_positions) / batch_size)
    )

    original_path: Optional[Path] = None
    htbm_path: Optional[Path] = None

    if not skip_original:
        set_seed(seed)
        clean_loader = make_loader(
            dataset=clean_training_dataset,
            positions=train_positions,
            batch_size=batch_size,
            seed=seed,
            device=selected_device,
            num_workers=num_workers,
        )
        clean_model = _build_victim_from_state(
            initial_state=initial_state,
            num_classes=len(class_names),
            trainable_scope=trainable_scope,
            device=selected_device,
        )
        clean_optimizer = optim.AdamW(
            [
                parameter
                for parameter in clean_model.parameters()
                if parameter.requires_grad
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        clean_scheduler = make_warmup_cosine_scheduler(
            clean_optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        original_path = train_and_save_shapley(
            model=clean_model,
            train_loader=clean_loader,
            optimizer=clean_optimizer,
            scheduler=clean_scheduler,
            criterion=criterion,
            epochs=epochs,
            device=selected_device,
            val_images=val_images,
            val_labels=val_labels,
            global_sample_count=len(clean_training_dataset),
            output_path=output_dir / "shapley_original_HTBM.csv",
            model_name="Original_HTBM",
            irds_parameter_scope=irds_parameter_scope,
        )

    if not skip_htbm:
        set_seed(seed)
        poisoned_loader = make_loader(
            dataset=poisoned_dataset,
            positions=train_positions,
            batch_size=batch_size,
            seed=seed,
            device=selected_device,
            num_workers=num_workers,
        )
        poisoned_model = _build_victim_from_state(
            initial_state=initial_state,
            num_classes=len(class_names),
            trainable_scope=trainable_scope,
            device=selected_device,
        )
        poisoned_optimizer = optim.AdamW(
            [
                parameter
                for parameter in poisoned_model.parameters()
                if parameter.requires_grad
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        poisoned_scheduler = make_warmup_cosine_scheduler(
            poisoned_optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        htbm_path = train_and_save_shapley(
            model=poisoned_model,
            train_loader=poisoned_loader,
            optimizer=poisoned_optimizer,
            scheduler=poisoned_scheduler,
            criterion=criterion,
            epochs=epochs,
            device=selected_device,
            val_images=val_images,
            val_labels=val_labels,
            global_sample_count=len(clean_training_dataset),
            output_path=output_dir / "shapley_htbm.csv",
            model_name="HTBM",
            irds_parameter_scope=irds_parameter_scope,
        )

    return original_path, htbm_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train clean and HTBM-poisoned ViT-B/16 models and save paired "
            "first-order IRDS CSV files."
        )
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help="ImageNet-100 root or its train split.",
    )
    parser.add_argument(
        "--poison-root",
        type=Path,
        required=True,
        help="Directory containing htbm_mapping.csv and poison images.",
    )
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Optional ImageNet validation split; otherwise a clean train holdout is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <poison-root>/irds_results.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional local ViT-B/16 checkpoint used for both victim starts.",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="Initialize the paired victim models from torchvision ImageNet weights.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Use a randomly initialized ViT-B/16 victim model.",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "head"),
        default="all",
        help="Fine-tune all ViT parameters or only the classification head.",
    )
    parser.add_argument(
        "--irds-parameter-scope",
        choices=("head", "all"),
        default="head",
        help=(
            "IRDS gradient scope. 'head' is the practical ViT-B default; "
            "'all' is exact but much more expensive."
        ),
    )
    parser.add_argument(
        "--integration-mode",
        choices=("append", "replace"),
        default="append",
        help=(
            "Append clean/poison pairs to the target class (paper-faithful "
            "default) or replace the mapped target samples."
        ),
    )
    parser.add_argument(
        "--augmentation",
        dest="augmentation",
        action="store_true",
        default=True,
        help="Use standard random ImageNet crop/flip augmentation (default).",
    )
    parser.add_argument(
        "--no-augmentation",
        dest="augmentation",
        action="store_false",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke tests or small subsets.",
    )
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-htbm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.poison_root / "irds_results")
    original_path, htbm_path = run_htbm_irds(
        train_root=args.train_root,
        poison_root=args.poison_root,
        output_dir=output_dir,
        mapping_path=args.mapping_path,
        val_root=args.val_root,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        validation_size=args.validation_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        weights_path=args.weights_path,
        trainable_scope=args.trainable_scope,
        irds_parameter_scope=args.irds_parameter_scope,
        integration_mode=args.integration_mode,
        augmentation=args.augmentation,
        max_train_samples=args.max_train_samples,
        device=args.device,
        skip_original=args.skip_original,
        skip_htbm=args.skip_htbm,
    )
    if original_path is not None:
        print(f"Clean IRDS CSV: {original_path}")
    if htbm_path is not None:
        print(f"HTBM IRDS CSV: {htbm_path}")


if __name__ == "__main__":
    main()
