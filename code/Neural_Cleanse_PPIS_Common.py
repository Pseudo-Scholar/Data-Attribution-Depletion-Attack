"""Shared Neural Cleanse + PPIS utilities.

This module is intentionally independent from the original PPIS scripts.
It provides:

* PPIS mapping/image loading;
* MNIST probe-model training;
* Neural Cleanse reverse engineering for every target label;
* MAD-based suspicious-target detection;
* reverse-trigger similarity scoring and sample filtering;
* first-order IRDS training and CSV/loss-curve export.

The implementation follows the Neural Cleanse idea in the paper:

    A(x, m, delta) = (1 - m) * x + m * delta

and minimizes target cross-entropy plus an L1 mask penalty. Images are
optimized in the [0, 1] pixel domain and normalized only before entering the
MNIST classifier.
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
from PIL import Image
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


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
    """MNIST dataset that preserves the original sample index."""

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return index, image, label


@dataclass(frozen=True)
class PPISPoisonRecord:
    original_index: int
    original_label: int
    poisoned_label: int
    poisoned_image_path: Path


def _find_poisoned_image(
    poisoned_dir: Path,
    original_index: int,
    original_label: int,
    row: Mapping[str, str],
) -> Path:
    path_fields = (
        "poisoned_image_path",
        "backdoor_image_path",
        "poisoned_path",
        "filename",
    )
    for field in path_fields:
        value = row.get(field)
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = poisoned_dir / candidate
        if candidate.is_file():
            return candidate

    exact_candidates = (
        poisoned_dir / f"backdoor_{original_index}_label_{original_label}.png",
        poisoned_dir / f"poisoned_{original_index}_label_{original_label}.png",
    )
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate

    patterns = (
        f"backdoor_{original_index}_label_*.png",
        f"poisoned_{original_index}_label_*.png",
        f"*{original_index}*poison*.png",
    )
    for pattern in patterns:
        matches = sorted(poisoned_dir.rglob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Cannot find PPIS image for sample {original_index} under "
        f"{poisoned_dir}."
    )


def load_ppis_mapping(
    mapping_path: Path,
    poisoned_dir: Path,
    original_labels: Sequence[int],
) -> Dict[int, PPISPoisonRecord]:
    """Load the mapping written by Backdoor_Poisoning_Strategy_PPIS.py."""

    mapping_path = Path(mapping_path)
    poisoned_dir = Path(poisoned_dir)
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"PPIS mapping does not exist: {mapping_path}. "
            "Run Backdoor_Poisoning_Strategy_PPIS.py first."
        )
    if not poisoned_dir.is_dir():
        raise FileNotFoundError(
            f"PPIS poisoned-image directory does not exist: {poisoned_dir}."
        )

    records: Dict[int, PPISPoisonRecord] = {}
    with mapping_path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = set(reader.fieldnames or [])
        if "original_index" not in fields:
            raise ValueError("PPIS mapping must contain original_index.")
        if "poisoned_label" not in fields:
            raise ValueError("PPIS mapping must contain poisoned_label.")

        for row in reader:
            index = int(row["original_index"])
            if not 0 <= index < len(original_labels):
                raise ValueError(f"Invalid PPIS sample index: {index}")
            original_label = int(original_labels[index])
            if row.get("original_label"):
                mapped_label = int(row["original_label"])
                if mapped_label != original_label:
                    raise ValueError(
                        f"Original-label mismatch for sample {index}: "
                        f"mapping={mapped_label}, dataset={original_label}."
                    )
            poisoned_label = int(row["poisoned_label"])
            if not 0 <= poisoned_label < 10:
                raise ValueError(f"Invalid poisoned label: {poisoned_label}")
            image_path = _find_poisoned_image(
                poisoned_dir=poisoned_dir,
                original_index=index,
                original_label=original_label,
                row=row,
            )
            records[index] = PPISPoisonRecord(
                original_index=index,
                original_label=original_label,
                poisoned_label=poisoned_label,
                poisoned_image_path=image_path,
            )

    if not records:
        raise ValueError(f"No PPIS records found in {mapping_path}.")
    return records


class PPISPoisonedMNIST(Dataset):
    """Replace selected indexed MNIST samples by PPIS images and labels."""

    def __init__(
        self,
        clean_dataset: IndexedMNIST,
        poison_records: Mapping[int, PPISPoisonRecord],
    ) -> None:
        self.clean_dataset = clean_dataset
        self.poison_records = dict(poison_records)
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(MNIST_MEAN, MNIST_STD)

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, index: int):
        record = self.poison_records.get(index)
        if record is None:
            return self.clean_dataset[index]

        image = Image.open(record.poisoned_image_path).convert("L")
        tensor = self._normalize(self._to_tensor(image))
        return index, tensor, record.poisoned_label


class RestoredPPISMNIST(Dataset):
    """Use clean data at filtered indices and PPIS data elsewhere."""

    def __init__(
        self,
        poisoned_dataset: PPISPoisonedMNIST,
        filtered_indices: Iterable[int],
    ) -> None:
        self.poisoned_dataset = poisoned_dataset
        self.filtered_indices = set(int(index) for index in filtered_indices)

    def __len__(self) -> int:
        return len(self.poisoned_dataset)

    def __getitem__(self, index: int):
        if index in self.filtered_indices:
            return self.poisoned_dataset.clean_dataset[index]
        return self.poisoned_dataset[index]


def build_mitigated_dataset(
    poisoned_dataset: PPISPoisonedMNIST,
    filtered_indices: Iterable[int],
    mode: str,
) -> Dataset:
    """Build a filtered training dataset while preserving sample indices."""

    filtered = sorted(set(int(index) for index in filtered_indices))
    if mode == "restore-clean":
        return RestoredPPISMNIST(poisoned_dataset, filtered)
    if mode == "remove":
        active_indices = [
            index
            for index in range(len(poisoned_dataset))
            if index not in set(filtered)
        ]
        return Subset(poisoned_dataset, active_indices)
    raise ValueError("mitigation mode must be 'remove' or 'restore-clean'.")


class SmallFNN(nn.Module):
    """The same MNIST classifier used by the original PPIS IRDS flow."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(self.flatten(images))))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    device: torch.device,
    shuffle: bool = True,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def make_reference_loader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _unpack_image_batch(
    batch: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        _, images, labels = batch
    elif len(batch) == 2:
        images, labels = batch
    else:
        raise ValueError("Expected a two- or three-element image batch.")
    return images, labels


def _mnist_mean_std(
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(MNIST_MEAN, device=device, dtype=dtype).view(1, -1, 1, 1)
    std = torch.tensor(MNIST_STD, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std


def denormalize_mnist(images: torch.Tensor) -> torch.Tensor:
    mean, std = _mnist_mean_std(images.device, images.dtype)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def normalize_mnist(images: torch.Tensor) -> torch.Tensor:
    mean, std = _mnist_mean_std(images.device, images.dtype)
    return (images - mean) / std


def apply_trigger(
    images: torch.Tensor,
    mask: torch.Tensor,
    trigger: torch.Tensor,
) -> torch.Tensor:
    """Apply a continuous Neural Cleanse mask and trigger in pixel space."""

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width].")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if trigger.ndim == 3:
        trigger = trigger.unsqueeze(0)
    if mask.ndim != 4 or trigger.ndim != 4:
        raise ValueError("mask/trigger must be 2D/3D/4D tensors.")
    return torch.clamp((1.0 - mask) * images + mask * trigger, 0.0, 1.0)


def train_probe_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_csv: Optional[Path] = None,
) -> List[float]:
    """Train a short probe model on the poisoned data for NC detection."""

    if epochs <= 0:
        raise ValueError("probe epochs must be positive.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    history: List[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            _, images, labels = _unpack_image_batch(batch)
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)
        epoch_loss = total_loss / max(total_count, 1)
        history.append(epoch_loss)
        print(
            f"[Neural Cleanse probe] Epoch {epoch + 1:03d}/{epochs:03d} | "
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
    anomaly_index: float = 0.0
    is_low_norm_anomaly: bool = False
    artifact_path: Optional[Path] = None


def reverse_engineer_trigger(
    model: nn.Module,
    reference_loader: DataLoader,
    target_label: int,
    device: torch.device,
    steps: int = 200,
    learning_rate: float = 0.1,
    mask_lambda: float = 1e-2,
    initial_mask_logit: float = -4.0,
) -> TriggerCandidate:
    """Reverse engineer one target-label trigger using Neural Cleanse."""

    if steps <= 0:
        raise ValueError("Neural Cleanse steps must be positive.")
    model.eval()
    mask_logits = torch.full(
        (1, 28, 28),
        float(initial_mask_logit),
        device=device,
        requires_grad=True,
    )
    trigger_logits = torch.zeros(
        (1, 1, 28, 28),
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

        images, _ = _unpack_image_batch(batch)
        images = images.to(device)
        pixel_images = denormalize_mnist(images)
        mask = torch.sigmoid(mask_logits)
        trigger = torch.sigmoid(trigger_logits)
        triggered = normalize_mnist(apply_trigger(pixel_images, mask, trigger))
        target = torch.full(
            (images.size(0),),
            target_label,
            device=device,
            dtype=torch.long,
        )
        loss = criterion(model(triggered), target) + mask_lambda * mask.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % max(steps // 5, 1) == 0:
            print(
                f"[Neural Cleanse] target={target_label} "
                f"step={step + 1}/{steps} loss={float(loss.item()):.6f}"
            )

    mask = torch.sigmoid(mask_logits).detach()
    trigger = torch.sigmoid(trigger_logits).detach()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in reference_loader:
            images, _ = _unpack_image_batch(batch)
            images = images.to(device)
            triggered = normalize_mnist(
                apply_trigger(denormalize_mnist(images), mask, trigger)
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
    )


def mad_detect_candidates(
    candidates: Sequence[TriggerCandidate],
    mad_threshold: float = 2.0,
    min_attack_success_rate: float = 0.0,
) -> Tuple[List[TriggerCandidate], Dict[str, float]]:
    """Mark low-mask-norm candidates as Neural Cleanse outliers."""

    if not candidates:
        return [], {
            "median_mask_l1": float("nan"),
            "mad": float("nan"),
            "mad_threshold": mad_threshold,
        }
    norms = np.asarray([candidate.mask_l1 for candidate in candidates], dtype=float)
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

    stats = {
        "median_mask_l1": median,
        "mad": mad,
        "mad_threshold": float(mad_threshold),
        "min_attack_success_rate": float(min_attack_success_rate),
    }
    return selected, stats


def reverse_engineer_all_targets(
    model: nn.Module,
    reference_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    steps: int = 200,
    learning_rate: float = 0.1,
    mask_lambda: float = 1e-2,
) -> List[TriggerCandidate]:
    candidates = []
    for target_label in range(num_classes):
        candidates.append(
            reverse_engineer_trigger(
                model=model,
                reference_loader=reference_loader,
                target_label=target_label,
                device=device,
                steps=steps,
                learning_rate=learning_rate,
                mask_lambda=mask_lambda,
            )
        )
    return candidates


def save_trigger_candidates(
    candidates: Sequence[TriggerCandidate],
    output_dir: Path,
    detection_stats: Mapping[str, float],
) -> Tuple[Path, Path]:
    """Save candidate trigger tensors and the CSV/JSON detection summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "neural_cleanse_candidates.csv"
    rows = []
    for candidate in candidates:
        artifact_path = output_dir / f"reverse_trigger_target_{candidate.target_label}.pt"
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

    json_path = output_dir / "neural_cleanse_detection.json"
    payload = dict(detection_stats)
    payload["candidate_count"] = len(candidates)
    payload["suspicious_target_labels"] = [
        candidate.target_label
        for candidate in candidates
        if candidate.is_low_norm_anomaly
    ]
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
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
    pixel_image = denormalize_mnist(image)
    with torch.no_grad():
        probability = torch.softmax(model(image), dim=1)[
            0, candidate.target_label
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
    return float(probability), float(similarity), float(probability * similarity)


def filter_ppis_samples(
    model: nn.Module,
    clean_dataset: IndexedMNIST,
    poisoned_dataset: PPISPoisonedMNIST,
    poison_records: Mapping[int, PPISPoisonRecord],
    suspicious_candidates: Sequence[TriggerCandidate],
    device: torch.device,
    output_csv: Path,
    filter_threshold: float = 0.8,
    filter_scope: str = "poisoned",
) -> Set[int]:
    """Score samples against reverse triggers and return filtered indices.

    Neural Cleanse identifies a model-level trigger. For this experiment, the
    trigger is also used as a sample-level filter: a sample is suspicious only
    when its target-label confidence and its pixel similarity to the
    reverse-engineered trigger are jointly high.
    """

    if filter_scope not in {"poisoned", "all"}:
        raise ValueError("filter_scope must be 'poisoned' or 'all'.")
    if not suspicious_candidates:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "sample_index",
                    "original_label",
                    "observed_label",
                    "is_ppis_record",
                    "detected_target_label",
                    "target_probability",
                    "trigger_similarity",
                    "suspicion_score",
                    "filtered",
                ]
            )
        return set()

    if filter_scope == "poisoned":
        indices = sorted(poison_records)
    else:
        indices = list(range(len(clean_dataset)))

    filtered: Set[int] = set()
    rows: List[Dict[str, object]] = []
    for index in indices:
        _, image, observed_label = poisoned_dataset[index]
        original_label = int(clean_dataset[index][2])

        best_candidate: Optional[TriggerCandidate] = None
        best_probability = 0.0
        best_similarity = 0.0
        best_score = 0.0
        for candidate in suspicious_candidates:
            probability, similarity, score = _sample_trigger_score(
                model=model,
                normalized_image=image,
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
            filtered.add(index)
        rows.append(
            {
                "sample_index": index,
                "original_label": original_label,
                "observed_label": int(observed_label),
                "is_ppis_record": int(index in poison_records),
                "detected_target_label": (
                    "" if best_candidate is None else best_candidate.target_label
                ),
                "target_probability": best_probability,
                "trigger_similarity": best_similarity,
                "suspicion_score": best_score,
                "filtered": int(is_filtered),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return filtered


def per_sample_grads(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute one flattened parameter gradient for each training sample."""

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
            [gradient.reshape(gradient.size(0), -1) for gradient in gradients.values()],
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
        logits = model(validation_images)
        validation_loss = criterion(logits, validation_labels)
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
    test_dataset: Dataset,
    validation_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if validation_size <= 0:
        raise ValueError("validation_size must be positive.")
    validation_size = min(validation_size, len(test_dataset))
    indices = sorted(
        random.Random(seed).sample(range(len(test_dataset)), validation_size)
    )
    loader = DataLoader(
        Subset(test_dataset, indices),
        batch_size=validation_size,
        shuffle=False,
        num_workers=0,
    )
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device, dtype=torch.long)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = _unpack_image_batch(batch)
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_and_save_shapley(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    sample_count: int,
    output_path: Path,
    loss_curve_path: Path,
    model_path: Optional[Path],
    model_name: str,
) -> Tuple[Path, Path]:
    """Train a model, accumulate first-order IRDS, and save CSV artifacts."""

    data_shapley = torch.zeros(sample_count, device=device)
    curve_rows: List[Dict[str, object]] = []
    print(f"Training {model_name} and accumulating first-order IRDS values...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for batch in train_loader:
            indices, images, labels = batch
            indices = indices.to(device, dtype=torch.long)
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            sample_gradients = per_sample_grads(
                model, images, labels, criterion
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
        test_loss, accuracy = evaluate_model(
            model, test_loader, criterion, device
        )
        curve_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "accuracy": accuracy,
            }
        )
        print(
            f"[{model_name}] Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss {train_loss:.6f} | Test Loss {test_loss:.6f} | "
            f"Accuracy {accuracy:.2f}%"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["sample_index", "shapley_value"])
        for index, value in enumerate(data_shapley.detach().cpu().tolist()):
            writer.writerow([index, float(value)])

    loss_curve_path.parent.mkdir(parents=True, exist_ok=True)
    with loss_curve_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["epoch", "train_loss", "test_loss", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)

    print(f"{model_name} Shapley values saved to: {output_path}")
    print(f"{model_name} loss curve saved to: {loss_curve_path}")
    return output_path, loss_curve_path


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(dict(payload), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return path
