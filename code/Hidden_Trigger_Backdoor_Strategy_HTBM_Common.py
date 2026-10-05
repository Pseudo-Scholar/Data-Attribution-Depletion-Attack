"""Shared utilities for the ImageNet/ViT Hidden Trigger strategy.

The implementation follows the hidden-trigger construction used by
Hidden Trigger Backdoor Attacks:

* a trigger is applied only to a source image used during poison generation;
* a target-category image initializes the poison;
* the poison is optimized in feature space toward the triggered source image;
* an L-infinity projection keeps the poison close to the target image.

The actual victim-side IRDS training is implemented in
``Hidden_Trigger_Backdoor_Strategy_HTBM_IRDS.py``.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageNetSample:
    """A stable sample record using the local ImageFolder class order."""

    index: int
    path: Path
    label: int
    class_name: str


@dataclass(frozen=True)
class TriggerPatch:
    """A trigger image and its binary mask in [C, H, W] format."""

    pixels: torch.Tensor
    mask: torch.Tensor
    source_name: str


def set_seed(seed: int) -> None:
    """Set deterministic seeds used by data generation and victim training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_split_root(root: Path, split: Optional[str] = None) -> Path:
    """Accept either a split directory or an ImageNet root containing splits."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ImageNet root does not exist: {root}")
    if root.is_file():
        raise ValueError(f"Expected a directory, got a file: {root}")

    if split:
        candidate = root / split
        if candidate.is_dir():
            return candidate
    return root


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def discover_imagenet_samples(
    root: Path,
    class_to_idx: Optional[Mapping[str, int]] = None,
) -> Tuple[List[ImageNetSample], List[str], Dict[str, int]]:
    """Discover an ImageFolder-style ImageNet split.

    The returned indices are deterministic and remain stable across the clean
    and poisoned victim runs. Class labels follow lexicographic directory
    order, matching ``torchvision.datasets.ImageFolder``.
    """

    root = resolve_split_root(root)
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(
            f"No class directories found under {root}. "
            "Expected root/<class_name>/<image>."
        )

    if class_to_idx is None:
        class_to_idx = {
            class_dir.name: label
            for label, class_dir in enumerate(class_dirs)
        }
    else:
        class_to_idx = dict(class_to_idx)

    selected_classes = [
        class_dir
        for class_dir in class_dirs
        if class_dir.name in class_to_idx
    ]
    if not selected_classes:
        raise ValueError(
            f"None of the class directories under {root} are present in the "
            "provided class_to_idx mapping."
        )

    class_names = [
        class_name
        for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]
    samples: List[ImageNetSample] = []
    sample_index = 0
    for class_dir in selected_classes:
        label = int(class_to_idx[class_dir.name])
        image_paths = sorted(
            path for path in class_dir.rglob("*") if _is_image_file(path)
        )
        for image_path in image_paths:
            samples.append(
                ImageNetSample(
                    index=sample_index,
                    path=image_path,
                    label=label,
                    class_name=class_dir.name,
                )
            )
            sample_index += 1

    if not samples:
        raise ValueError(f"No supported images found under {root}.")

    return samples, class_names, dict(class_to_idx)


def make_generation_transform(image_size: int = 224) -> transforms.Compose:
    """Build the deterministic ImageNet transform used by poison generation."""

    resize_size = max(image_size, int(round(image_size * 256 / 224)))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def make_victim_transform(
    image_size: int = 224,
    training: bool = True,
    augmentation: bool = True,
) -> transforms.Compose:
    """Build the victim-side ImageNet preprocessing pipeline."""

    if training and augmentation:
        image_transforms = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333),
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        resize_size = max(image_size, int(round(image_size * 256 / 224)))
        image_transforms = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
        ]

    image_transforms.append(transforms.ToTensor())
    image_transforms.append(
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    )
    return transforms.Compose(image_transforms)


def load_image_tensor(
    image_path: Path,
    transform: transforms.Compose,
) -> torch.Tensor:
    """Load an RGB image using a caller-specified transform."""

    with Image.open(image_path) as image:
        return transform(image.convert("RGB"))


def save_image_tensor(image: torch.Tensor, output_path: Path) -> None:
    """Save a [C, H, W] tensor in [0, 1] as an RGB PNG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().clamp(0.0, 1.0)
    transforms.ToPILImage()(image).save(output_path)


def normalize_for_vit(images: torch.Tensor) -> torch.Tensor:
    """Normalize [0, 1] RGB images for torchvision ViT models."""

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


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint  # type: ignore[return-value]
    raise ValueError("Could not find a model state_dict in the checkpoint.")


def load_compatible_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Load matching checkpoint tensors while tolerating a different head."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    cleaned_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if isinstance(value, torch.Tensor):
            cleaned_state[clean_key] = value

    current_state = model.state_dict()
    compatible = {
        key: value
        for key, value in cleaned_state.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded {len(compatible)} tensors from {checkpoint_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}."
    )


def _replace_vit_head(model: nn.Module, num_classes: int) -> None:
    """Replace torchvision ViT's final linear layer."""

    heads = getattr(model, "heads", None)
    if heads is None:
        raise AttributeError("The supplied model does not expose a ViT heads module.")

    if isinstance(heads, nn.Sequential):
        linear = None
        for module in reversed(list(heads.children())):
            if isinstance(module, nn.Linear):
                linear = module
                break
        if linear is None:
            raise AttributeError("Could not find the ViT classification head.")
        heads[-1] = nn.Linear(linear.in_features, num_classes)
        return

    if isinstance(heads, nn.Linear):
        model.heads = nn.Linear(heads.in_features, num_classes)
        return

    raise AttributeError("Unsupported torchvision ViT heads module.")


def get_vit_head(model: nn.Module) -> nn.Linear:
    """Return the final linear classification layer of a torchvision ViT."""

    heads = getattr(model, "heads", None)
    if isinstance(heads, nn.Sequential):
        for module in reversed(list(heads.children())):
            if isinstance(module, nn.Linear):
                return module
    if isinstance(heads, nn.Linear):
        return heads
    raise AttributeError("Could not find a linear ViT classification head.")


def build_vit_b16(
    num_classes: int,
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """Build ViT-B/16 and optionally load ImageNet or local weights."""

    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for the HTBM ImageNet/ViT implementation."
        ) from exc

    torchvision_weights = None
    if pretrained and weights_path is None:
        torchvision_weights = ViT_B_16_Weights.IMAGENET1K_V1

    try:
        model = vit_b_16(weights=torchvision_weights)
    except Exception as exc:
        if pretrained and weights_path is None:
            raise RuntimeError(
                "Could not load torchvision's ViT-B/16 ImageNet weights. "
                "Use --no-pretrained for an offline run or provide "
                "--weights-path with a local checkpoint."
            ) from exc
        raise

    if weights_path is not None:
        load_compatible_checkpoint(model, Path(weights_path))

    _replace_vit_head(model, num_classes)
    return model


def build_vit_feature_extractor(
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """Build an independent ViT-B/16 feature extractor."""

    model = build_vit_b16(
        num_classes=1000,
        pretrained=pretrained,
        weights_path=weights_path,
    )
    model.heads = nn.Identity()
    return model


def vit_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract the class-token representation before the ViT head."""

    if hasattr(model, "forward_features"):
        features = model.forward_features(images)  # type: ignore[attr-defined]
        if features.ndim == 3:
            return features[:, 0]
        return features

    process_input = getattr(model, "_process_input", None)
    class_token = getattr(model, "class_token", None)
    encoder = getattr(model, "encoder", None)
    if process_input is None or class_token is None or encoder is None:
        raise AttributeError(
            "The supplied model does not expose the torchvision ViT internals "
            "needed for feature extraction."
        )

    tokens = process_input(images)
    batch_size = tokens.shape[0]
    class_tokens = class_token.expand(batch_size, -1, -1)
    tokens = torch.cat([class_tokens, tokens], dim=1)
    encoded = encoder(tokens)
    return encoded[:, 0]


def load_trigger_patch(
    trigger_path: Optional[Path],
    size: Tuple[int, int] = (32, 32),
    seed: int = 42,
) -> TriggerPatch:
    """Load a trigger image or create a deterministic synthetic fallback.

    A supplied RGBA image uses alpha>0 as the binary mask. RGB images are
    treated as fully opaque. When no path is supplied, a checkerboard trigger
    is created so the pipeline remains reproducible for smoke tests.
    """

    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid trigger size: {size}")

    if trigger_path is None:
        rng = np.random.default_rng(seed)
        tile = 4
        pattern = np.indices((height, width)).sum(axis=0) // tile
        checker = (pattern % 2).astype(np.float32)
        color_a = rng.uniform(0.0, 0.25, size=(3, 1, 1)).astype(np.float32)
        color_b = rng.uniform(0.75, 1.0, size=(3, 1, 1)).astype(np.float32)
        pixels = checker[None, :, :] * color_a + (
            1.0 - checker[None, :, :]
        ) * color_b
        return TriggerPatch(
            pixels=torch.from_numpy(pixels),
            mask=torch.ones((height, width), dtype=torch.bool),
            source_name="synthetic_checkerboard",
        )

    trigger_path = Path(trigger_path)
    if not trigger_path.is_file():
        raise FileNotFoundError(f"Trigger image does not exist: {trigger_path}")

    with Image.open(trigger_path) as image:
        rgba = image.convert("RGBA")
        rgba = rgba.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(rgba, dtype=np.uint8)

    pixels = torch.from_numpy(array[:, :, :3].copy()).permute(2, 0, 1)
    pixels = pixels.to(dtype=torch.float32) / 255.0
    mask = torch.from_numpy(array[:, :, 3].copy() > 0)
    if not bool(mask.any()):
        raise ValueError(f"Trigger alpha mask is empty: {trigger_path}")
    return TriggerPatch(
        pixels=pixels,
        mask=mask,
        source_name=trigger_path.name,
    )


def sample_trigger_positions(
    rng: random.Random,
    count: int,
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int,
) -> List[Tuple[int, int]]:
    """Sample random top-left trigger coordinates."""

    if patch_height > image_height or patch_width > image_width:
        raise ValueError(
            f"Trigger size {(patch_width, patch_height)} exceeds image size "
            f"{(image_width, image_height)}."
        )
    max_y = image_height - patch_height
    max_x = image_width - patch_width
    return [
        (rng.randint(0, max_y), rng.randint(0, max_x))
        for _ in range(count)
    ]


def paste_trigger_batch(
    images: torch.Tensor,
    trigger: TriggerPatch,
    positions: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Apply a binary-mask trigger to a batch of [0, 1] images."""

    if images.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] images, got {tuple(images.shape)}")
    if len(positions) != images.shape[0]:
        raise ValueError("The number of positions must equal the batch size.")

    result = images.clone()
    patch = trigger.pixels.to(device=images.device, dtype=images.dtype)
    mask = trigger.mask.to(device=images.device)
    channels, patch_height, patch_width = patch.shape
    if channels != images.shape[1]:
        raise ValueError(
            f"Trigger channels={channels} do not match image channels={images.shape[1]}."
        )

    for batch_index, (top, left) in enumerate(positions):
        bottom = top + patch_height
        right = left + patch_width
        if top < 0 or left < 0 or bottom > images.shape[2] or right > images.shape[3]:
            raise ValueError(
                f"Trigger position {(top, left)} is outside image shape "
                f"{tuple(images.shape[2:])}."
            )
        region = result[batch_index, :, top:bottom, left:right]
        result[batch_index, :, top:bottom, left:right] = torch.where(
            mask.unsqueeze(0),
            patch,
            region,
        )
    return result


def greedy_one_to_one_assignment(distance: torch.Tensor) -> torch.Tensor:
    """Greedily match each row to one unused column."""

    if distance.ndim != 2:
        raise ValueError("distance must be a two-dimensional tensor.")
    rows, columns = distance.shape
    if rows > columns:
        raise ValueError("One-to-one assignment requires columns >= rows.")

    distance_cpu = distance.detach().cpu()
    remaining = set(range(columns))
    assignment: List[int] = []
    for row in range(rows):
        candidates = sorted(
            remaining,
            key=lambda column: float(distance_cpu[row, column]),
        )
        if not candidates:
            raise RuntimeError("Greedy assignment ran out of columns.")
        selected = candidates[0]
        assignment.append(selected)
        remaining.remove(selected)
    return torch.tensor(assignment, dtype=torch.long, device=distance.device)


def resolve_class_selector(
    selector: Optional[str],
    class_names: Sequence[str],
    class_to_idx: Mapping[str, int],
    default_index: int,
) -> Tuple[int, str]:
    """Resolve a class directory name or integer label."""

    if selector is None:
        label = default_index
        return label, class_names[label]

    if selector in class_to_idx:
        label = int(class_to_idx[selector])
        return label, selector

    try:
        label = int(selector)
    except ValueError as exc:
        raise ValueError(
            f"Unknown class selector {selector!r}. Use a class directory name "
            "or an integer label."
        ) from exc

    if not 0 <= label < len(class_names):
        raise ValueError(
            f"Class label {label} is outside [0, {len(class_names)})."
        )
    return label, class_names[label]


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Make an independent CPU copy for identical clean/poisoned starts."""

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def load_state_dict_clone(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> nn.Module:
    """Load a copied initial state into a newly built model."""

    model.load_state_dict(copy.deepcopy(dict(state_dict)), strict=True)
    return model
