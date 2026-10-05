"""Shared Neural Cleanse + HTBA utilities.

This module adds an independent mitigation path for the ImageNet/ViT-B
Hidden Trigger Backdoor Attack experiment.  The original HTBA/HTBM
generation and IRDS files are intentionally left untouched.

The implementation follows Neural Cleanse:

    A(x, m, delta) = (1 - m) * x + m * delta

For each candidate target label, a continuous mask and trigger are optimized
to make clean reference images predict that label while minimizing the mask
L1 norm.  Low-norm candidates are detected with MAD.  HTBA poisons are then
filtered conservatively by their mapped target label and probe-model
confidence.  This is deliberately conservative because HTBA poison images
do not contain the visible trigger used during poison construction.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from Hidden_Trigger_Backdoor_Strategy_HTBM_Common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
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
from Hidden_Trigger_Backdoor_Strategy_HTBM_IRDS import (
    HTBMAppendedImageNetDataset,
    HTBMPoisonedImageNetDataset,
    HTBMPoisonRecord,
    IndexedImageNetDataset,
    choose_training_positions,
    choose_validation_positions,
    load_htbm_mapping,
    make_loader as make_irds_loader,
    configure_trainable_scope,
    compute_all_validation_gradient,
    compute_head_validation_gradient,
    make_warmup_cosine_scheduler,
    per_sample_all_dots,
    per_sample_head_dots,
)


@dataclass
class HTBADataBundle:
    """Paired clean/poisoned datasets with stable IRDS sample indices."""

    train_samples: List[ImageNetSample]
    class_names: List[str]
    class_to_idx: Dict[str, int]
    poison_records: Dict[int, HTBMPoisonRecord]
    clean_dataset: Dataset
    poisoned_dataset: Dataset
    validation_dataset: Dataset
    selected_positions: List[int]
    train_positions: List[int]
    validation_positions: List[int]
    appended_positions: List[int]
    global_sample_count: int
    base_sample_count: int
    integration_mode: str


@dataclass
class TriggerCandidate:
    target_label: int
    mask: torch.Tensor
    trigger: torch.Tensor
    mask_l1: float
    attack_success_rate: float
    anomaly_index: float = 0.0
    is_low_norm_anomaly: bool = False
    artifact_path: Optional[Path] = None


def denormalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized ImageNet tensors back to [0, 1] RGB pixels."""

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=images.dtype,
        device=images.device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=images.dtype,
        device=images.device,
    ).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Normalize [0, 1] RGB tensors for torchvision ViT models."""

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=images.dtype,
        device=images.device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=images.dtype,
        device=images.device,
    ).view(1, 3, 1, 1)
    return (images - mean) / std


def apply_neural_cleanse_trigger(
    images: torch.Tensor,
    mask: torch.Tensor,
    trigger: torch.Tensor,
) -> torch.Tensor:
    """Apply a continuous mask/trigger pair to [0, 1] RGB images."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "images must have shape [batch, 3, height, width]."
        )
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if trigger.ndim == 3:
        trigger = trigger.unsqueeze(0)
    if mask.ndim != 4 or trigger.ndim != 4:
        raise ValueError("mask and trigger must be 2D/3D/4D tensors.")
    return torch.clamp((1.0 - mask) * images + mask * trigger, 0.0, 1.0)


def _unpack_batch(batch):
    if len(batch) == 3:
        _, images, labels = batch
    elif len(batch) == 2:
        images, labels = batch
    else:
        raise ValueError("Expected a two- or three-element batch.")
    return images, labels


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(dict(payload), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return path


def make_reference_loader(
    dataset: Dataset,
    positions: Sequence[int],
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    if not positions:
        raise ValueError("The Neural Cleanse reference set is empty.")
    return DataLoader(
        Subset(dataset, list(positions)),
        batch_size=max(1, min(batch_size, len(positions))),
        shuffle=False,
        num_workers=num_workers,
    )


def make_training_loader(
    dataset: Dataset,
    positions: Sequence[int],
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int = 0,
) -> DataLoader:
    return make_irds_loader(
        dataset=dataset,
        positions=positions,
        batch_size=batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
    )


def build_htba_data_bundle(
    train_root: Path,
    poison_root: Path,
    mapping_path: Optional[Path],
    val_root: Optional[Path],
    image_size: int,
    validation_size: int,
    seed: int,
    integration_mode: str,
    max_train_samples: Optional[int],
    augmentation: bool,
    num_workers: int = 0,
) -> HTBADataBundle:
    """Build paired HTBA datasets using the original stable index rules."""

    if integration_mode not in {"append", "replace"}:
        raise ValueError("integration_mode must be 'append' or 'replace'.")

    train_split_root = resolve_split_root(Path(train_root), "train")
    train_samples, class_names, class_to_idx = discover_imagenet_samples(
        train_split_root
    )
    samples_by_index = {sample.index: sample for sample in train_samples}
    mapping_path = mapping_path or (Path(poison_root) / "htbm_mapping.csv")
    poison_records = load_htbm_mapping(
        mapping_path=Path(mapping_path),
        samples_by_index=samples_by_index,
        num_classes=len(class_names),
    )

    poison_record_list = sorted(
        poison_records.values(),
        key=lambda record: record.poison_index,
    )
    if integration_mode == "append":
        expected = list(range(len(poison_record_list)))
        actual = [record.poison_index for record in poison_record_list]
        if actual != expected:
            raise ValueError(
                "append integration requires contiguous poison_index values "
                "starting at zero."
            )

    train_transform = make_victim_transform(
        image_size=image_size,
        training=True,
        augmentation=augmentation,
    )
    eval_transform = make_victim_transform(
        image_size=image_size,
        training=False,
        augmentation=False,
    )
    base_clean_dataset = IndexedImageNetDataset(
        train_samples,
        transform=train_transform,
    )

    if integration_mode == "append":
        clean_dataset: Dataset = HTBMAppendedImageNetDataset(
            clean_dataset=base_clean_dataset,
            poison_records=poison_record_list,
            transform=train_transform,
            use_poison=False,
        )
        poisoned_dataset: Dataset = HTBMAppendedImageNetDataset(
            clean_dataset=base_clean_dataset,
            poison_records=poison_record_list,
            transform=train_transform,
            use_poison=True,
        )
        appended_positions = [
            len(train_samples) + record.poison_index
            for record in poison_record_list
        ]
    else:
        clean_dataset = base_clean_dataset
        poisoned_dataset = HTBMPoisonedImageNetDataset(
            clean_dataset=base_clean_dataset,
            poison_records=poison_records,
            transform=train_transform,
        )
        appended_positions = []

    base_selected_positions = choose_training_positions(
        sample_count=len(train_samples),
        max_train_samples=max_train_samples,
        required_positions=sorted(poison_records),
        seed=seed,
    )

    if val_root is not None:
        val_split_root = resolve_split_root(Path(val_root), "val")
        val_samples, _, _ = discover_imagenet_samples(
            val_split_root,
            class_to_idx=class_to_idx,
        )
        validation_dataset: Dataset = IndexedImageNetDataset(
            val_samples,
            transform=eval_transform,
        )
        validation_positions = choose_validation_positions(
            list(range(len(val_samples))),
            validation_size=validation_size,
            seed=seed + 1,
        )
        selected_positions = sorted(
            set(base_selected_positions).union(appended_positions)
        )
        train_positions = selected_positions
    else:
        non_poison_candidates = [
            position
            for position in base_selected_positions
            if train_samples[position].index not in poison_records
        ]
        if not non_poison_candidates:
            non_poison_candidates = list(base_selected_positions)
        validation_positions = choose_validation_positions(
            non_poison_candidates,
            validation_size=validation_size,
            seed=seed + 1,
        )
        validation_dataset = IndexedImageNetDataset(
            train_samples,
            transform=eval_transform,
        )
        validation_set = set(validation_positions)
        base_train_positions = [
            position
            for position in base_selected_positions
            if position not in validation_set
        ]
        if integration_mode == "append":
            train_positions = sorted(
                set(base_train_positions).union(appended_positions)
            )
        else:
            poison_position_set = {
                position
                for position, sample in enumerate(train_samples)
                if sample.index in poison_records
            }
            train_positions = sorted(
                set(base_train_positions).union(poison_position_set)
            )
        selected_positions = sorted(
            set(base_selected_positions).union(appended_positions)
        )

    if not train_positions:
        raise ValueError("No HTBA training positions remain.")

    return HTBADataBundle(
        train_samples=train_samples,
        class_names=class_names,
        class_to_idx=class_to_idx,
        poison_records=poison_records,
        clean_dataset=clean_dataset,
        poisoned_dataset=poisoned_dataset,
        validation_dataset=validation_dataset,
        selected_positions=selected_positions,
        train_positions=train_positions,
        validation_positions=validation_positions,
        appended_positions=appended_positions,
        global_sample_count=len(clean_dataset),
        base_sample_count=len(train_samples),
        integration_mode=integration_mode,
    )


def train_probe_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    output_csv: Optional[Path] = None,
) -> List[float]:
    """Train a short ViT-B probe used only by the Neural Cleanse detector."""

    if epochs <= 0:
        raise ValueError("probe epochs must be positive.")
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The probe model has no trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: List[float] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            _, images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
        epoch_loss = total_loss / max(total, 1)
        history.append(epoch_loss)
        print(
            f"[Neural Cleanse HTBA probe] Epoch {epoch + 1:03d}/"
            f"{epochs:03d} | Loss {epoch_loss:.6f}"
        )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["epoch", "probe_train_loss"])
            for epoch, loss in enumerate(history, start=1):
                writer.writerow([epoch, loss])
    return history


def reverse_engineer_trigger(
    model: nn.Module,
    reference_loader: DataLoader,
    target_label: int,
    device: torch.device,
    steps: int,
    learning_rate: float,
    mask_lambda: float,
) -> TriggerCandidate:
    """Reverse engineer one continuous Neural Cleanse candidate for ViT-B."""

    if steps <= 0:
        raise ValueError("Neural Cleanse steps must be positive.")
    model.eval()
    first_batch = next(iter(reference_loader))
    first_images, _ = _unpack_batch(first_batch)
    _, channels, height, width = first_images.shape
    if channels != 3:
        raise ValueError("HTBA Neural Cleanse expects RGB images.")

    mask_logits = torch.full(
        (1, height, width),
        -4.0,
        dtype=first_images.dtype,
        device=device,
        requires_grad=True,
    )
    trigger_logits = torch.zeros(
        (1, channels, height, width),
        dtype=first_images.dtype,
        device=device,
        requires_grad=True,
    )
    optimizer = optim.Adam(
        [mask_logits, trigger_logits],
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    iterator = iter(reference_loader)

    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(reference_loader)
            batch = next(iterator)
        images, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        pixel_images = denormalize_imagenet(images)
        mask = torch.sigmoid(mask_logits)
        trigger = torch.sigmoid(trigger_logits)
        triggered = normalize_imagenet(
            apply_neural_cleanse_trigger(
                pixel_images,
                mask,
                trigger,
            )
        )
        target = torch.full(
            (images.size(0),),
            int(target_label),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(model(triggered), target) + mask_lambda * mask.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % max(steps // 5, 1) == 0:
            print(
                f"[Neural Cleanse HTBA] target={target_label} "
                f"step={step + 1}/{steps} loss={float(loss.item()):.6f}"
            )

    mask = torch.sigmoid(mask_logits).detach()
    trigger = torch.sigmoid(trigger_logits).detach()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in reference_loader:
            images, _ = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            triggered = normalize_imagenet(
                apply_neural_cleanse_trigger(
                    denormalize_imagenet(images),
                    mask,
                    trigger,
                )
            )
            predictions = model(triggered).argmax(dim=1)
            correct += int((predictions == int(target_label)).sum().item())
            total += int(predictions.numel())

    return TriggerCandidate(
        target_label=int(target_label),
        mask=mask.squeeze(0).cpu(),
        trigger=trigger.squeeze(0).cpu(),
        mask_l1=float(mask.sum().item()),
        attack_success_rate=correct / max(total, 1),
    )


def reverse_engineer_targets(
    model: nn.Module,
    reference_loader: DataLoader,
    target_labels: Sequence[int],
    device: torch.device,
    steps: int,
    learning_rate: float,
    mask_lambda: float,
) -> List[TriggerCandidate]:
    candidates: List[TriggerCandidate] = []
    for target_label in target_labels:
        candidate = reverse_engineer_trigger(
            model=model,
            reference_loader=reference_loader,
            target_label=int(target_label),
            device=device,
            steps=steps,
            learning_rate=learning_rate,
            mask_lambda=mask_lambda,
        )
        candidates.append(candidate)
        torch.cuda.empty_cache()
    return candidates


def mad_detect_candidates(
    candidates: Sequence[TriggerCandidate],
    mad_threshold: float,
    min_attack_success_rate: float,
) -> Tuple[List[TriggerCandidate], Dict[str, object]]:
    """Detect low-mask-norm candidates using the Neural Cleanse MAD rule."""

    if not candidates:
        return [], {
            "median_mask_l1": None,
            "mad": None,
            "mad_threshold": mad_threshold,
            "min_attack_success_rate": min_attack_success_rate,
        }

    norms = np.asarray(
        [candidate.mask_l1 for candidate in candidates],
        dtype=float,
    )
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    denominator = 1.4826 * mad
    selected: List[TriggerCandidate] = []
    for candidate in candidates:
        if denominator <= 1e-12:
            anomaly_index = (
                float("inf") if candidate.mask_l1 < median else 0.0
            )
        else:
            anomaly_index = abs(candidate.mask_l1 - median) / denominator
        candidate.anomaly_index = float(anomaly_index)
        candidate.is_low_norm_anomaly = bool(
            candidate.mask_l1 < median
            and candidate.anomaly_index >= mad_threshold
            and candidate.attack_success_rate >= min_attack_success_rate
        )
        if candidate.is_low_norm_anomaly:
            selected.append(candidate)

    return selected, {
        "median_mask_l1": median,
        "mad": mad,
        "mad_threshold": float(mad_threshold),
        "min_attack_success_rate": float(min_attack_success_rate),
    }


def save_trigger_candidates(
    candidates: Sequence[TriggerCandidate],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "neural_cleanse_htba_candidates.csv"
    rows: List[Dict[str, object]] = []
    for candidate in candidates:
        artifact_path = output_dir / (
            f"neural_cleanse_htba_trigger_target_"
            f"{candidate.target_label}.pt"
        )
        torch.save(
            {
                "target_label": candidate.target_label,
                "mask": candidate.mask,
                "trigger": candidate.trigger,
                "mask_l1": candidate.mask_l1,
                "attack_success_rate": candidate.attack_success_rate,
                "anomaly_index": candidate.anomaly_index,
                "is_low_norm_anomaly": candidate.is_low_norm_anomaly,
            },
            artifact_path,
        )
        candidate.artifact_path = artifact_path
        rows.append(
            {
                "target_label": candidate.target_label,
                "mask_l1": candidate.mask_l1,
                "attack_success_rate": candidate.attack_success_rate,
                "anomaly_index": candidate.anomaly_index,
                "is_low_norm_anomaly": int(candidate.is_low_norm_anomaly),
                "trigger_artifact": str(artifact_path),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "target_label",
                "mask_l1",
                "attack_success_rate",
                "anomaly_index",
                "is_low_norm_anomaly",
                "trigger_artifact",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _global_poison_index(
    record: HTBMPoisonRecord,
    integration_mode: str,
    base_sample_count: int,
) -> int:
    if integration_mode == "append":
        return base_sample_count + int(record.poison_index)
    return int(record.target_original_index)


def filter_htba_samples(
    model: nn.Module,
    poisoned_dataset: Dataset,
    poison_records: Mapping[int, HTBMPoisonRecord],
    integration_mode: str,
    base_sample_count: int,
    suspicious_candidates: Sequence[TriggerCandidate],
    device: torch.device,
    output_csv: Path,
    filter_threshold: float,
    filter_scope: str,
) -> Set[int]:
    """Filter HTBA samples using detected target labels and confidence.

    Unlike a visible-trigger attack, HTBA poison images intentionally hide the
    trigger.  Therefore the default defense uses the Neural Cleanse label
    candidate plus probe confidence, rather than claiming pixel-level trigger
    similarity that HTBA does not expose.
    """

    if filter_scope not in {"poisoned", "all"}:
        raise ValueError("filter_scope must be 'poisoned' or 'all'.")
    candidate_by_label = {
        int(candidate.target_label): candidate
        for candidate in suspicious_candidates
    }
    record_by_global_index = {
        _global_poison_index(record, integration_mode, base_sample_count): record
        for record in poison_records.values()
    }

    if filter_scope == "poisoned":
        indices = sorted(record_by_global_index)
    else:
        indices = list(range(len(poisoned_dataset)))

    fields = [
        "sample_index",
        "dataset_position",
        "label",
        "is_htba_record",
        "mapped_target_label",
        "detected_target_label",
        "target_probability",
        "suspicion_score",
        "filtered",
        "reason",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered: Set[int] = set()
    rows: List[Dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for index in indices:
            dataset_position = int(index)
            stable_index, image, label = poisoned_dataset[dataset_position]
            logits = model(image.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)[0]

            record = record_by_global_index.get(int(stable_index))
            best_candidate: Optional[TriggerCandidate] = None
            best_probability = 0.0
            if record is not None and int(record.target_label) in candidate_by_label:
                best_candidate = candidate_by_label[int(record.target_label)]
                best_probability = float(
                    probabilities[int(best_candidate.target_label)].item()
                )
            elif candidate_by_label:
                for candidate in candidate_by_label.values():
                    probability = float(
                        probabilities[int(candidate.target_label)].item()
                    )
                    if probability > best_probability:
                        best_probability = probability
                        best_candidate = candidate

            mapped_target = (
                int(record.target_label) if record is not None else None
            )
            detected_target = (
                int(best_candidate.target_label)
                if best_candidate is not None
                else None
            )
            target_match = (
                record is not None
                and best_candidate is not None
                and mapped_target == detected_target
            )
            suspicion_score = (
                best_probability if target_match else 0.0
            )
            should_filter = bool(
                best_candidate is not None
                and target_match
                and suspicion_score >= filter_threshold
            )
            if should_filter:
                filtered.add(int(stable_index))

            rows.append(
                {
                    "sample_index": int(stable_index),
                    "dataset_position": dataset_position,
                    "label": int(label),
                    "is_htba_record": int(record is not None),
                    "mapped_target_label": mapped_target,
                    "detected_target_label": detected_target,
                    "target_probability": best_probability,
                    "suspicion_score": suspicion_score,
                    "filtered": int(should_filter),
                    "reason": (
                        "suspicious_target_label_and_confidence"
                        if should_filter
                        else "not_selected"
                    ),
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return filtered


class RestoredHTBAImageNetDataset(Dataset):
    """Restore clean samples at filtered positions and keep other poisons."""

    def __init__(
        self,
        clean_dataset: Dataset,
        poisoned_dataset: Dataset,
        filtered_indices: Iterable[int],
    ) -> None:
        self.clean_dataset = clean_dataset
        self.poisoned_dataset = poisoned_dataset
        self.filtered_indices = set(int(index) for index in filtered_indices)

    def __len__(self) -> int:
        return len(self.poisoned_dataset)

    def __getitem__(self, index: int):
        if int(index) in self.filtered_indices:
            return self.clean_dataset[index]
        return self.poisoned_dataset[index]


def build_mitigated_dataset(
    clean_dataset: Dataset,
    poisoned_dataset: Dataset,
    filtered_indices: Iterable[int],
    mode: str,
    selected_positions: Optional[Sequence[int]] = None,
) -> Tuple[Dataset, List[int]]:
    """Create remove or restore-clean mitigation data."""

    filtered = set(int(index) for index in filtered_indices)
    if selected_positions is None:
        base_positions = list(range(len(poisoned_dataset)))
    else:
        base_positions = [int(position) for position in selected_positions]
    if mode == "remove":
        active = [
            position for position in base_positions if position not in filtered
        ]
        return poisoned_dataset, active
    if mode == "restore-clean":
        return (
            RestoredHTBAImageNetDataset(
                clean_dataset=clean_dataset,
                poisoned_dataset=poisoned_dataset,
                filtered_indices=filtered,
            ),
            base_positions,
        )
    raise ValueError("mitigation mode must be 'remove' or 'restore-clean'.")


def build_validation_batch(
    dataset: Dataset,
    positions: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not positions:
        raise ValueError("Validation positions are empty.")
    loader = DataLoader(
        Subset(dataset, list(positions)),
        batch_size=len(positions),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    images, labels = _unpack_batch(batch)
    return images.to(device), labels.to(device, dtype=torch.long)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_and_save_shapley(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    global_sample_count: int,
    shapley_path: Path,
    curve_path: Path,
    model_path: Optional[Path],
    model_name: str,
    irds_parameter_scope: str,
) -> Tuple[Path, Path]:
    """Train one victim model and accumulate first-order in-run Shapley."""

    if irds_parameter_scope not in {"head", "all"}:
        raise ValueError("irds_parameter_scope must be 'head' or 'all'.")
    data_shapley = torch.zeros(global_sample_count, device=device)
    curve_rows: List[Dict[str, object]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for indices, images, labels in train_loader:
            indices = indices.to(device, dtype=torch.long)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            if irds_parameter_scope == "head":
                validation_gradient = compute_head_validation_gradient(
                    model,
                    validation_images,
                    validation_labels,
                    criterion,
                )
                dots = per_sample_head_dots(
                    model,
                    images,
                    labels,
                    validation_gradient,
                )
            else:
                validation_gradient = compute_all_validation_gradient(
                    model,
                    validation_images,
                    validation_labels,
                    criterion,
                )
                dots = per_sample_all_dots(
                    model,
                    images,
                    labels,
                    validation_gradient,
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
        validation_loss, validation_accuracy = _evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )
        curve_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"[{model_name}] Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss {train_loss:.6f} | Val Loss {validation_loss:.6f} | "
            f"Val Acc {validation_accuracy:.2f}%"
        )

    shapley_path.parent.mkdir(parents=True, exist_ok=True)
    with shapley_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["sample_index", "shapley_value"])
        for index, value in enumerate(data_shapley.detach().cpu().tolist()):
            writer.writerow([index, float(value)])

    with curve_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    return shapley_path, curve_path


def make_initial_state(
    num_classes: int,
    pretrained: bool,
    weights_path: Optional[Path],
    seed: int,
) -> Dict[str, torch.Tensor]:
    set_seed(seed)
    model = build_vit_b16(
        num_classes=num_classes,
        pretrained=pretrained,
        weights_path=weights_path,
    )
    return clone_state_dict_to_cpu(model)


def build_victim_from_state(
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


def make_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int,
) -> Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR]:
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The victim model has no trainable parameters.")
    optimizer = optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    return optimizer, scheduler
