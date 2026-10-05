"""Shared Neural Cleanse + APIS utilities.

The original APIS generator and IRDS implementation are intentionally left
untouched. This module adds the APIS-specific defense layer:

* RGB face normalization and continuous trigger application;
* short probe-model training;
* Neural Cleanse reverse engineering for 47 x 55 face inputs;
* a scalable screen-then-refine search over large label spaces;
* MAD low-mask-norm target detection;
* reverse-trigger based APIS sample filtering;
* first-order IRDS training with aligned Shapley CSVs and loss curves.

Neural Cleanse optimizes:

    A(x, m, delta) = (1 - m) * x + m * delta

in the [0, 1] pixel domain. The face classifier receives normalized RGB
images with the same mean/std convention as the original APIS IRDS script.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Dataset, Subset

from Backdoor_Poisoning_Strategy_APIS_IRDS import (
    APISPoisonRecord,
    APISPoisonedYouTubeFaceDataset,
    FACE_MEAN,
    FACE_STD,
    IndexedYouTubeFaceDataset,
    YouTubeFaceCNN,
    image_to_tensor,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _face_mean_std(
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(FACE_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(FACE_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def denormalize_face(images: torch.Tensor) -> torch.Tensor:
    mean, std = _face_mean_std(images.device, images.dtype)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def normalize_face(images: torch.Tensor) -> torch.Tensor:
    mean, std = _face_mean_std(images.device, images.dtype)
    return (images - mean) / std


def apply_trigger(
    images: torch.Tensor,
    mask: torch.Tensor,
    trigger: torch.Tensor,
) -> torch.Tensor:
    """Apply a continuous Neural Cleanse mask/trigger in pixel space."""

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, 3, height, width].")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if trigger.ndim == 3:
        trigger = trigger.unsqueeze(0)
    if mask.ndim != 4 or trigger.ndim != 4:
        raise ValueError("mask/trigger must be 2D/3D/4D tensors.")
    return torch.clamp((1.0 - mask) * images + mask * trigger, 0.0, 1.0)


def _unpack_batch(
    batch: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        _, images, labels = batch
    elif len(batch) == 2:
        images, labels = batch
    else:
        raise ValueError("Expected a two- or three-element image batch.")
    return images, labels


class RestoredAPISDataset(Dataset):
    """Replace filtered APIS samples by their clean counterparts."""

    def __init__(
        self,
        clean_dataset: IndexedYouTubeFaceDataset,
        poisoned_dataset: APISPoisonedYouTubeFaceDataset,
        filtered_indices: Iterable[int],
    ) -> None:
        self.clean_dataset = clean_dataset
        self.poisoned_dataset = poisoned_dataset
        self.filtered_indices = set(int(index) for index in filtered_indices)

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, position: int):
        sample_index = self.clean_dataset.samples[position].index
        if sample_index in self.filtered_indices:
            return self.clean_dataset[position]
        return self.poisoned_dataset[position]


def build_mitigated_dataset(
    clean_dataset: IndexedYouTubeFaceDataset,
    poisoned_dataset: APISPoisonedYouTubeFaceDataset,
    selected_positions: Sequence[int],
    filtered_indices: Iterable[int],
    mode: str,
) -> Tuple[Dataset, List[int]]:
    """Build a filtered/clean-restored dataset and its train positions."""

    if mode not in {"remove", "restore-clean"}:
        raise ValueError("mitigation mode must be 'remove' or 'restore-clean'.")
    filtered = set(int(index) for index in filtered_indices)
    if mode == "remove":
        active_positions = [
            position
            for position in selected_positions
            if clean_dataset.samples[position].index not in filtered
        ]
        return poisoned_dataset, active_positions

    restored = RestoredAPISDataset(
        clean_dataset=clean_dataset,
        poisoned_dataset=poisoned_dataset,
        filtered_indices=filtered,
    )
    return restored, list(selected_positions)


def make_loader(
    dataset: Dataset,
    positions: Sequence[int],
    batch_size: int,
    seed: int,
    device: torch.device,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    if not positions:
        raise ValueError("Cannot create a loader from an empty position list.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(positions)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def make_reference_loader(
    dataset: IndexedYouTubeFaceDataset,
    positions: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    if not positions:
        raise ValueError("Reference positions cannot be empty.")
    return DataLoader(
        Subset(dataset, list(positions)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def train_probe_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
    device: torch.device,
    output_csv: Optional[Path] = None,
) -> List[float]:
    """Train the model used by Neural Cleanse to inspect the poisoned data."""

    if epochs <= 0:
        raise ValueError("probe epochs must be positive.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    history: List[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for batch in train_loader:
            _, images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
        epoch_loss = running_loss / max(total, 1)
        history.append(epoch_loss)
        print(
            f"[Neural Cleanse APIS probe] Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Loss {epoch_loss:.6f}"
        )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["epoch", "probe_train_loss"])
            for epoch, loss in enumerate(history, start=1):
                writer.writerow([epoch, loss])
    return history


@dataclass
class TriggerCandidate:
    target_label: int
    mask: torch.Tensor
    trigger: torch.Tensor
    mask_l1: float
    attack_success_rate: float
    optimization_stage: str
    anomaly_index: float = 0.0
    is_low_norm_anomaly: bool = False
    artifact_path: Optional[Path] = None


def reverse_engineer_trigger(
    model: nn.Module,
    reference_loader: DataLoader,
    target_label: int,
    device: torch.device,
    steps: int = 50,
    learning_rate: float = 0.1,
    mask_lambda: float = 1e-2,
    optimization_stage: str = "screen",
) -> TriggerCandidate:
    """Reverse engineer one APIS target trigger."""

    if steps <= 0:
        raise ValueError("Neural Cleanse steps must be positive.")
    model.eval()
    first_batch = next(iter(reference_loader))
    first_images, _ = _unpack_batch(first_batch)
    if first_images.ndim != 4 or first_images.size(1) != 3:
        raise ValueError(
            "APIS Neural Cleanse expects RGB reference images with shape "
            "[batch, 3, height, width]."
        )
    _, channels, height, width = first_images.shape
    mask_logits = torch.full(
        (1, height, width),
        -4.0,
        device=device,
        requires_grad=True,
    )
    trigger_logits = torch.zeros(
        (1, channels, height, width),
        device=device,
        requires_grad=True,
    )
    optimizer = optim.Adam([mask_logits, trigger_logits], lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    iterator: Optional[Iterable[Sequence[torch.Tensor]]] = None

    for step in range(steps):
        if iterator is None:
            iterator = iter(reference_loader)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(reference_loader)
            batch = next(iterator)
        images, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        pixel_images = denormalize_face(images)
        mask = torch.sigmoid(mask_logits)
        trigger = torch.sigmoid(trigger_logits)
        triggered = normalize_face(apply_trigger(pixel_images, mask, trigger))
        target = torch.full(
            (images.size(0),),
            target_label,
            dtype=torch.long,
            device=device,
        )
        loss = criterion(model(triggered), target) + mask_lambda * mask.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % max(steps // 5, 1) == 0:
            print(
                f"[Neural Cleanse APIS] target={target_label} "
                f"stage={optimization_stage} step={step + 1}/{steps} "
                f"loss={float(loss.item()):.6f}"
            )

    mask = torch.sigmoid(mask_logits).detach()
    trigger = torch.sigmoid(trigger_logits).detach()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in reference_loader:
            images, _ = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            triggered = normalize_face(
                apply_trigger(denormalize_face(images), mask, trigger)
            )
            predictions = model(triggered).argmax(dim=1)
            correct += int((predictions == target_label).sum().item())
            total += int(predictions.numel())

    return TriggerCandidate(
        target_label=target_label,
        mask=mask.squeeze(0).cpu(),
        trigger=trigger.squeeze(0).cpu(),
        mask_l1=float(mask.sum().item()),
        attack_success_rate=correct / max(total, 1),
        optimization_stage=optimization_stage,
    )


def reverse_engineer_targets(
    model: nn.Module,
    reference_loader: DataLoader,
    target_labels: Sequence[int],
    device: torch.device,
    screen_steps: int = 5,
    screen_learning_rate: float = 0.1,
    screen_mask_lambda: float = 1e-2,
    refine_top_k: int = 16,
    refine_steps: int = 100,
    refine_learning_rate: float = 0.1,
    refine_mask_lambda: float = 1e-2,
) -> List[TriggerCandidate]:
    """Screen many labels cheaply, then refine low-mask-norm candidates."""

    unique_labels = sorted(set(int(label) for label in target_labels))
    if not unique_labels:
        raise ValueError("target_labels cannot be empty.")
    screen_candidates: List[TriggerCandidate] = []
    for target_label in unique_labels:
        screen_candidates.append(
            reverse_engineer_trigger(
                model=model,
                reference_loader=reference_loader,
                target_label=target_label,
                device=device,
                steps=screen_steps,
                learning_rate=screen_learning_rate,
                mask_lambda=screen_mask_lambda,
                optimization_stage="screen",
            )
        )

    refine_count = max(0, min(int(refine_top_k), len(screen_candidates)))
    refine_labels = {
        candidate.target_label
        for candidate in sorted(
            screen_candidates,
            key=lambda candidate: candidate.mask_l1,
        )[:refine_count]
    }
    refined_by_label: Dict[int, TriggerCandidate] = {}
    for target_label in sorted(refine_labels):
        refined_by_label[target_label] = reverse_engineer_trigger(
            model=model,
            reference_loader=reference_loader,
            target_label=target_label,
            device=device,
            steps=refine_steps,
            learning_rate=refine_learning_rate,
            mask_lambda=refine_mask_lambda,
            optimization_stage="refine",
        )

    return [
        refined_by_label.get(candidate.target_label, candidate)
        for candidate in screen_candidates
    ]


def mad_detect_candidates(
    candidates: Sequence[TriggerCandidate],
    mad_threshold: float = 2.0,
    min_attack_success_rate: float = 0.0,
) -> Tuple[List[TriggerCandidate], Dict[str, float]]:
    """Detect unusually small low-norm trigger candidates using MAD."""

    if not candidates:
        return [], {
            "median_mask_l1": float("nan"),
            "mad": float("nan"),
            "mad_threshold": mad_threshold,
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
                float("inf")
                if candidate.mask_l1 < median
                else 0.0
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
    detection_stats: Mapping[str, object],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "neural_cleanse_apis_candidates.csv"
    rows: List[Dict[str, object]] = []

    for candidate in candidates:
        artifact_path = (
            output_dir
            / f"neural_cleanse_apis_trigger_target_{candidate.target_label}.pt"
        )
        torch.save(
            {
                "target_label": candidate.target_label,
                "mask": candidate.mask,
                "trigger": candidate.trigger,
                "mask_l1": candidate.mask_l1,
                "attack_success_rate": candidate.attack_success_rate,
                "optimization_stage": candidate.optimization_stage,
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
                "optimization_stage": candidate.optimization_stage,
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
                "optimization_stage",
                "anomaly_index",
                "is_low_norm_anomaly",
                "trigger_artifact",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "neural_cleanse_apis_detection.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(dict(detection_stats), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return csv_path, json_path


def _sample_trigger_score(
    model: nn.Module,
    normalized_image: torch.Tensor,
    candidate: TriggerCandidate,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Return target probability, trigger similarity, and combined score."""

    model.eval()
    image = normalized_image.unsqueeze(0).to(device)
    pixel_image = denormalize_face(image)
    with torch.no_grad():
        target_probability = torch.softmax(model(image), dim=1)[
            0,
            candidate.target_label,
        ].item()

    mask = candidate.mask.to(device=device, dtype=image.dtype)
    trigger = candidate.trigger.to(device=device, dtype=image.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if trigger.ndim == 3:
        trigger = trigger.unsqueeze(0)
    difference = torch.abs(pixel_image - trigger)
    weight = mask.expand_as(difference)
    denominator = float(weight.sum().item())
    if denominator <= 1e-12:
        similarity = 0.0
    else:
        similarity = 1.0 - float((difference * weight).sum().item()) / denominator
        similarity = max(0.0, min(1.0, similarity))
    return (
        float(target_probability),
        float(similarity),
        float(target_probability * similarity),
    )


def filter_apis_samples(
    model: nn.Module,
    clean_dataset: IndexedYouTubeFaceDataset,
    poisoned_dataset: APISPoisonedYouTubeFaceDataset,
    poison_records: Mapping[int, APISPoisonRecord],
    position_by_index: Mapping[int, int],
    scan_positions: Sequence[int],
    suspicious_candidates: Sequence[TriggerCandidate],
    device: torch.device,
    output_csv: Path,
    filter_threshold: float = 0.80,
    filter_scope: str = "poisoned",
) -> Set[int]:
    """Filter APIS samples using target confidence and trigger similarity."""

    if filter_scope not in {"poisoned", "all"}:
        raise ValueError("filter_scope must be 'poisoned' or 'all'.")
    if filter_scope == "poisoned":
        positions = sorted(
            position_by_index[index]
            for index in poison_records
            if index in position_by_index
        )
    else:
        positions = sorted(set(int(position) for position in scan_positions))

    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_index",
        "original_label",
        "observed_label",
        "is_apis_record",
        "detected_target_label",
        "target_probability",
        "trigger_similarity",
        "suspicion_score",
        "filtered",
    ]
    if not suspicious_candidates:
        with Path(output_csv).open("w", newline="", encoding="utf-8") as output_file:
            csv.writer(output_file).writerow(fields)
        return set()

    filtered: Set[int] = set()
    rows: List[Dict[str, object]] = []
    for position in positions:
        sample_index, poisoned_image, observed_label = poisoned_dataset[position]
        original_label = int(clean_dataset.samples[position].label)
        best_candidate: Optional[TriggerCandidate] = None
        best_probability = 0.0
        best_similarity = 0.0
        best_score = 0.0

        for candidate in suspicious_candidates:
            probability, similarity, score = _sample_trigger_score(
                model=model,
                normalized_image=poisoned_image,
                candidate=candidate,
                device=device,
            )
            if score > best_score:
                best_candidate = candidate
                best_probability = probability
                best_similarity = similarity
                best_score = score

        is_filtered = best_score >= filter_threshold
        if is_filtered:
            filtered.add(int(sample_index))
        rows.append(
            {
                "sample_index": int(sample_index),
                "original_label": original_label,
                "observed_label": int(observed_label),
                "is_apis_record": int(int(sample_index) in poison_records),
                "detected_target_label": (
                    ""
                    if best_candidate is None
                    else best_candidate.target_label
                ),
                "target_probability": best_probability,
                "trigger_similarity": best_similarity,
                "suspicion_score": best_score,
                "filtered": int(is_filtered),
            }
        )

    with Path(output_csv).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return filtered


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
        return torch.cat(
            [
                gradient.reshape(gradient.size(0), -1)
                for gradient in gradients.values()
            ],
            dim=1,
        )
    except (ImportError, RuntimeError, TypeError):
        parameters = tuple(
            parameter for parameter in model.parameters()
            if parameter.requires_grad
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
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    parameters = tuple(
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    )
    was_training = model.training
    model.train()
    try:
        validation_loss = criterion(
            model(validation_images),
            validation_labels,
        )
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


def evaluate_model(
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
) -> Tuple[Path, Path]:
    """Train a model, accumulate first-order IRDS values, and save artifacts."""

    data_shapley = torch.zeros(global_sample_count, device=device)
    curve_rows: List[Dict[str, object]] = []
    print(f"Training {model_name} and accumulating first-order IRDS values...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for indices, images, labels in train_loader:
            indices = indices.to(device, dtype=torch.long)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            sample_gradients = per_sample_grads(
                model,
                images,
                labels,
                criterion,
            )
            validation_gradient = compute_validation_gradient(
                model,
                validation_images,
                validation_labels,
                criterion,
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            increments = -learning_rate * (
                sample_gradients @ validation_gradient.T
            ).squeeze(1)
            data_shapley[indices] += increments.detach()

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        validation_loss, validation_accuracy = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )
        curve_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"[{model_name}] Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss {train_loss:.6f} | "
            f"Validation Loss {validation_loss:.6f} | "
            f"Validation Acc {validation_accuracy:.2f}%"
        )

    shapley_path.parent.mkdir(parents=True, exist_ok=True)
    with shapley_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["sample_index", "shapley_value"])
        for index, value in enumerate(data_shapley.detach().cpu().tolist()):
            writer.writerow([index, float(value)])

    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    print(f"{model_name} Shapley values saved to: {shapley_path}")
    return shapley_path, curve_path


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(dict(payload), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return path
