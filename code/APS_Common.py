"""
Shared utilities for the Adversarial Perturbation Strategy (APS).

The implementation follows the APS objective in the paper:

    x_adv = argmax_{||x' - x||_p <= epsilon} P(x') + C(x')

    P(x') = l(f_w(x'), y)

    C(x') = min(0, eta * <g_val, g_x'>)

where g_val is the validation-set gradient and g_x' is the gradient of the
candidate sample loss with respect to the surrogate model parameters.

Images are represented in pixel space as float tensors in [0, 1]. The model
input is normalized only inside the model-loss helpers. This makes epsilon
and the optional trigger amplitude easier to interpret and avoids mixing
pixel-space and normalized-space distances.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModifiedCIFARCNN(nn.Module):
    """The CIFAR-10 CNN used by the original APS experiment."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(60, 80, kernel_size=2, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(60 * 2 * 2, 160)
        self.fc2 = nn.Linear(80 * 1 * 1, 160)
        self.fc3 = nn.Linear(160, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x_pool3 = self.pool(x)

        x_fc1 = x_pool3.reshape(x_pool3.size(0), -1)
        x_fc1 = self.relu(self.fc1(x_fc1))

        x_conv4 = self.relu(self.conv4(x_pool3))
        x_conv4 = x_conv4.reshape(x_conv4.size(0), -1)
        x_conv4 = self.relu(self.fc2(x_conv4))

        return self.fc3(self.relu(x_fc1 + x_conv4))


class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR-10 dataset that returns ``(index, image, label)``."""

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return index, image, label


class PoisonedCIFAR10Dataset(Dataset):
    """
    Replace selected CIFAR-10 samples while preserving their original labels.

    ``poisoned_images`` must contain model-ready tensors, normally normalized
    CIFAR-10 tensors with shape ``[3, 32, 32]``. The class deliberately keeps
    the original label because APS is label-consistent.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        poisoned_images: Mapping[int, torch.Tensor],
    ) -> None:
        self.original_dataset = original_dataset
        self.poisoned_images = {
            int(index): image.detach().clone()
            for index, image in poisoned_images.items()
        }

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index: int):
        if index in self.poisoned_images:
            _, _, original_label = self.original_dataset[index]
            return index, self.poisoned_images[index], int(original_label)
        return self.original_dataset[index]


def cifar10_normalize(
    images: torch.Tensor,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> torch.Tensor:
    """Differentiably normalize ``[0, 1]`` images for the CIFAR-10 model."""

    if images.ndim not in (3, 4):
        raise ValueError(
            f"Expected an image tensor with 3 or 4 dimensions, got {images.shape}."
        )

    if images.ndim == 3:
        shape = (len(mean), 1, 1)
    else:
        shape = (1, len(mean), 1, 1)

    mean_tensor = torch.as_tensor(mean, device=images.device, dtype=images.dtype)
    std_tensor = torch.as_tensor(std, device=images.device, dtype=images.dtype)
    mean_tensor = mean_tensor.reshape(shape)
    std_tensor = std_tensor.reshape(shape)
    return (images - mean_tensor) / std_tensor


def denormalize_cifar10(
    images: torch.Tensor,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> torch.Tensor:
    """Convert normalized CIFAR-10 tensors back to clipped pixel tensors."""

    if images.ndim not in (3, 4):
        raise ValueError(
            f"Expected an image tensor with 3 or 4 dimensions, got {images.shape}."
        )

    if images.ndim == 3:
        shape = (len(mean), 1, 1)
    else:
        shape = (1, len(mean), 1, 1)

    mean_tensor = torch.as_tensor(mean, device=images.device, dtype=images.dtype)
    std_tensor = torch.as_tensor(std, device=images.device, dtype=images.dtype)
    mean_tensor = mean_tensor.reshape(shape)
    std_tensor = std_tensor.reshape(shape)
    return torch.clamp(images * std_tensor + mean_tensor, 0.0, 1.0)


def make_cifar10_transform(
    normalize: bool = True,
    train_augmentation: bool = False,
) -> transforms.Compose:
    """Build a reproducible CIFAR-10 transform for training or evaluation."""

    transform_list = []
    if train_augmentation:
        transform_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return transforms.Compose(transform_list)


def load_image_tensor(path: str | Path) -> torch.Tensor:
    """Load an RGB image as a float tensor in [0, 1]."""

    image = Image.open(path).convert("RGB")
    return transforms.ToTensor()(image)


def save_image_tensor(image: torch.Tensor, path: str | Path) -> None:
    """Save a single [C, H, W] pixel tensor as an RGB image."""

    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got {image.shape}.")
    transforms.ToPILImage()(image).save(path)


def _validate_norm(norm: str) -> str:
    normalized = norm.lower().replace("-", "")
    if normalized not in {"l2", "linf"}:
        raise ValueError("Only L2 and Linf projection are implemented.")
    return normalized


def project_lp_ball(
    candidate: torch.Tensor,
    center: torch.Tensor,
    epsilon: float,
    norm: str = "l2",
) -> torch.Tensor:
    """Project candidate images into an L2 or Linf ball around ``center``."""

    norm = _validate_norm(norm)
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    delta = candidate - center
    if norm == "linf":
        delta = delta.clamp(min=-epsilon, max=epsilon)
    else:
        flat_delta = delta.reshape(delta.size(0), -1)
        delta_norm = flat_delta.norm(p=2, dim=1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(delta_norm),
            torch.full_like(delta_norm, float(epsilon))
            / delta_norm.clamp_min(1e-12),
        )
        delta = (flat_delta * scale).reshape_as(delta)
    return (center + delta).clamp(0.0, 1.0)


def perturbation_norm(
    candidate: torch.Tensor,
    center: torch.Tensor,
    norm: str = "l2",
) -> torch.Tensor:
    """Return one perturbation norm per image in a batch."""

    norm = _validate_norm(norm)
    delta = (candidate - center).reshape(candidate.size(0), -1)
    if norm == "linf":
        return delta.abs().amax(dim=1)
    return delta.norm(p=2, dim=1)


def _single_corner_coordinates(
    height: int,
    width: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    black = [
        (height - 3, width - 3),
        (height - 3, width - 2),
        (height - 2, width - 3),
        (height - 2, width - 1),
        (height - 1, width - 2),
    ]
    white = [
        (height - 3, width - 1),
        (height - 2, width - 2),
        (height - 1, width - 3),
        (height - 1, width - 1),
    ]
    return black, white


def _corner_coordinates(
    height: int,
    width: int,
    corner: str,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if corner == "top_left":
        black = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        white = [(0, 0), (1, 1), (2, 0), (2, 2)]
    elif corner == "top_right":
        black = [(0, width - 3), (0, width - 2), (1, width - 1), (1, width - 3), (2, width - 2)]
        white = [(0, width - 1), (1, width - 2), (2, width - 3), (2, width - 1)]
    elif corner == "bottom_left":
        black = [(height - 3, 1), (height - 3, 2), (height - 2, 0), (height - 2, 2), (height - 1, 1)]
        white = [(height - 3, 0), (height - 2, 1), (height - 1, 0), (height - 1, 2)]
    elif corner == "bottom_right":
        black = [(height - 3, width - 3), (height - 3, width - 2), (height - 2, width - 3), (height - 2, width - 1), (height - 1, width - 2)]
        white = [(height - 3, width - 1), (height - 2, width - 2), (height - 1, width - 3), (height - 1, width - 1)]
    else:
        raise ValueError(f"Unknown corner: {corner}")
    return black, white


def _apply_coordinate_delta(
    result: torch.Tensor,
    coordinates: Iterable[Tuple[int, int]],
    delta: float,
) -> None:
    height, width = result.shape[-2:]
    for y, x in coordinates:
        if 0 <= y < height and 0 <= x < width:
            result[:, y, x] = (result[:, y, x] + delta).clamp(0.0, 1.0)


def apply_low_visibility_trigger(
    images: torch.Tensor,
    mode: str = "none",
    amplitude: float = 8.0,
) -> torch.Tensor:
    """
    Apply the optional low-visibility trigger used by the APS enhancement.

    ``amplitude`` is specified in 8-bit pixel units. The trigger is applied
    after APS perturbation and does not change the original label.
    """

    mode = mode.lower()
    if mode not in {"none", "single_corner", "four_corners"}:
        raise ValueError(
            "mode must be one of: none, single_corner, four_corners."
        )
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative.")
    if images.ndim == 3:
        batch = images.unsqueeze(0)
        squeeze = True
    elif images.ndim == 4:
        batch = images
        squeeze = False
    else:
        raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {images.shape}.")

    result = batch.clone()
    if mode == "none":
        return result.squeeze(0) if squeeze else result

    amplitude_float = float(amplitude) / 255.0
    height, width = result.shape[-2:]
    if height < 3 or width < 3:
        raise ValueError("The trigger requires images of at least 3 x 3 pixels.")

    for image in result:
        if mode == "single_corner":
            black, white = _single_corner_coordinates(height, width)
            _apply_coordinate_delta(image, black, -amplitude_float)
            _apply_coordinate_delta(image, white, amplitude_float)
        else:
            for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
                black, white = _corner_coordinates(height, width, corner)
                _apply_coordinate_delta(image, black, -amplitude_float)
                _apply_coordinate_delta(image, white, amplitude_float)

    return result.squeeze(0) if squeeze else result


def _trainable_parameters(model: nn.Module) -> Tuple[nn.Parameter, ...]:
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def _replace_none_gradients(
    gradients: Sequence[Optional[torch.Tensor]],
    parameters: Sequence[nn.Parameter],
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(gradients, parameters)
    )


def compute_validation_gradient(
    model: nn.Module,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    normalize_fn=cifar10_normalize,
) -> Tuple[torch.Tensor, ...]:
    """
    Compute and detach the validation loss gradient with respect to parameters.

    The returned tuple is intentionally detached because the validation
    gradient is the fixed reference vector in the APS inner optimization.
    """

    parameters = _trainable_parameters(model)
    if not parameters:
        raise ValueError("The model has no trainable parameters.")

    was_training = model.training
    model.eval()
    try:
        val_logits = model(normalize_fn(val_images))
        val_loss = F.cross_entropy(val_logits, val_labels, reduction="mean")
        gradients = torch.autograd.grad(
            val_loss,
            parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        return tuple(
            gradient.detach()
            for gradient in _replace_none_gradients(gradients, parameters)
        )
    finally:
        model.train(was_training)


def _gradient_alignment(
    gradients: Sequence[torch.Tensor],
    validation_gradients: Sequence[torch.Tensor],
) -> torch.Tensor:
    alignment = gradients[0].new_zeros(())
    for gradient, validation_gradient in zip(gradients, validation_gradients):
        alignment = alignment + (gradient * validation_gradient).sum()
    return alignment


def aps_objective(
    model: nn.Module,
    candidate_images: torch.Tensor,
    labels: torch.Tensor,
    validation_gradients: Sequence[torch.Tensor],
    eta: float = 1.0,
    normalize_fn=cifar10_normalize,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate the batch APS objective.

    Returns ``(objective, mean_loss, mean_alignment)``. The primary path uses
    ``torch.func.vmap`` so each sample still receives its own alignment
    penalty without an avoidable Python loop. A per-sample autograd fallback
    is retained for older PyTorch versions.
    """

    if candidate_images.ndim != 4:
        raise ValueError(
            f"candidate_images must have shape [B,C,H,W], got {candidate_images.shape}."
        )
    if candidate_images.size(0) != labels.size(0):
        raise ValueError("The image batch and label batch have different sizes.")

    parameters = _trainable_parameters(model)
    normalized_images = normalize_fn(candidate_images)
    logits = model(normalized_images)
    losses = F.cross_entropy(logits, labels, reduction="none")

    try:
        from torch.func import functional_call, grad, vmap

        parameter_dict = dict(model.named_parameters())
        buffer_dict = dict(model.named_buffers())

        def single_loss(
            functional_params,
            functional_buffers,
            normalized_image,
            label,
        ):
            single_logits = functional_call(
                model,
                (functional_params, functional_buffers),
                (normalized_image.unsqueeze(0),),
            )
            return F.cross_entropy(
                single_logits,
                label.unsqueeze(0),
                reduction="mean",
            )

        per_sample_gradients = vmap(grad(single_loss), (None, None, 0, 0))(
            parameter_dict,
            buffer_dict,
            normalized_images,
            labels,
        )
        alignments = candidate_images.new_zeros(candidate_images.size(0))
        for gradient, validation_gradient in zip(
            per_sample_gradients.values(),
            validation_gradients,
        ):
            alignments = alignments + (
                gradient.reshape(gradient.size(0), -1)
                * validation_gradient.reshape(1, -1)
            ).sum(dim=1)
    except (ImportError, RuntimeError, TypeError):
        # Compatibility path for older PyTorch versions without torch.func.
        alignments_list: List[torch.Tensor] = []
        for index in range(candidate_images.size(0)):
            sample_loss = F.cross_entropy(
                logits[index : index + 1],
                labels[index : index + 1],
                reduction="mean",
            )
            sample_gradients = torch.autograd.grad(
                sample_loss,
                parameters,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            sample_gradients = _replace_none_gradients(
                sample_gradients,
                parameters,
            )
            alignments_list.append(
                _gradient_alignment(sample_gradients, validation_gradients)
            )
        alignments = torch.stack(alignments_list)

    penalties = torch.minimum(
        torch.zeros_like(alignments),
        float(eta) * alignments,
    )
    mean_loss = losses.mean()
    mean_penalty = penalties.mean()
    mean_alignment = alignments.mean()
    return mean_loss + mean_penalty, mean_loss, mean_alignment


def _gradient_ascent_direction(gradient: torch.Tensor, norm: str) -> torch.Tensor:
    norm = _validate_norm(norm)
    if norm == "linf":
        return gradient.sign()
    flat_gradient = gradient.reshape(gradient.size(0), -1)
    gradient_norm = flat_gradient.norm(p=2, dim=1, keepdim=True)
    return (flat_gradient / gradient_norm.clamp_min(1e-12)).reshape_as(gradient)


def aps_pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    epsilon: float,
    norm: str = "l2",
    steps: int = 100,
    step_size: Optional[float] = None,
    eta: float = 1.0,
    normalize_fn=cifar10_normalize,
) -> torch.Tensor:
    """
    Generate APS samples with projected gradient ascent.

    The default step size is exactly ``1.5 * epsilon / steps`` as described
    in the paper. The input and output are pixel tensors in [0, 1].
    """

    norm = _validate_norm(norm)
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if step_size is None:
        step_size = 1.5 * float(epsilon) / float(steps)
    if step_size < 0:
        raise ValueError("step_size must be non-negative.")
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B,C,H,W], got {images.shape}.")

    clean_images = images.detach().clone()
    candidate = clean_images.clone()
    if epsilon == 0.0 or step_size == 0.0:
        return candidate

    validation_gradients = compute_validation_gradient(
        model=model,
        val_images=val_images,
        val_labels=val_labels,
        normalize_fn=normalize_fn,
    )

    was_training = model.training
    model.eval()
    try:
        for _ in range(steps):
            candidate = candidate.detach().requires_grad_(True)
            objective, _, _ = aps_objective(
                model=model,
                candidate_images=candidate,
                labels=labels,
                validation_gradients=validation_gradients,
                eta=eta,
                normalize_fn=normalize_fn,
            )
            candidate_gradient = torch.autograd.grad(
                objective,
                candidate,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
            with torch.no_grad():
                direction = _gradient_ascent_direction(candidate_gradient, norm)
                candidate = candidate + float(step_size) * direction
                candidate = project_lp_ball(
                    candidate,
                    center=clean_images,
                    epsilon=float(epsilon),
                    norm=norm,
                )
    finally:
        model.train(was_training)

    return candidate.detach()


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load either a raw state dict or a checkpoint containing ``state_dict``."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict)
    return model
