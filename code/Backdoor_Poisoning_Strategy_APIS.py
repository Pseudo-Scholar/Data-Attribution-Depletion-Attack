"""
Accessory-Pattern Injection Strategy (APIS) for YouTube Aligned Face.

The implementation follows the APIS formulation in the paper:

    Pi_alpha(k, x)[i, j] =
        alpha * k[i, j] + (1 - alpha) * x[i, j],  (i, j) not in R(k)
        x[i, j],                                  (i, j) in R(k)

where k is an accessory pattern, x is an aligned face image, and R(k)
contains transparent pixels in the accessory. The script is intentionally
kept independent from the model-training code. It prepares poisoned
samples and a CSV mapping that can be consumed by the IRDS training and
evaluation pipeline later.

Expected dataset layout (ImageFolder style):

    dataset_root/
    |-- identity_0000/
    |   |-- frame_000001.jpg
    |   `-- ...
    |-- identity_0001/
    `-- ...

The face images are expected to be RGB images. The paper uses aligned
55 x 47 images; the CLI uses (width=47, height=55) by default and resizes
only when an input image has a different size.

Accessory assets should preferably be transparent PNG files. For an RGB
asset without an alpha channel, the script estimates transparency from the
corner background color. A separate grayscale mask can be supplied when
that estimate is not appropriate.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FaceSample:
    """A face image and its ImageFolder-style identity label."""

    index: int
    path: Path
    label: int
    class_name: str


@dataclass(frozen=True)
class AccessoryPattern:
    """An accessory placed on a target-size canvas."""

    rgb: np.ndarray
    opaque_mask: np.ndarray
    source_name: str
    scale: float
    position_x: int
    position_y: int


def set_seed(seed: int) -> random.Random:
    """Return a local deterministic RNG without changing global state."""

    return random.Random(seed)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _resolve_image_root(dataset_root: Path) -> Path:
    """
    Resolve the actual identity-directory root.

    The provided archive extracts to:

        <parent>/aligned_images_DB/<identity>/<video>/<image>.jpg

    Users may pass either ``aligned_images_DB`` itself or its parent
    directory. Supporting both avoids accidentally treating the whole
    database as one class.
    """

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if dataset_root.is_file():
        raise ValueError(
            "dataset_root points to a file. Extract aligned_images_DB.tar.gz "
            "first, then pass either the extracted aligned_images_DB directory "
            "or its parent directory."
        )

    nested_root = dataset_root / "aligned_images_DB"
    if nested_root.is_dir():
        return nested_root
    return dataset_root


def discover_imagefolder_samples(
    dataset_root: Path,
    min_images_per_identity: int = 100,
) -> tuple[list[FaceSample], list[str]]:
    """
    Discover samples from an ImageFolder-style directory.

    Class indices follow lexicographic class-directory order, matching the
    convention used by torchvision.datasets.ImageFolder. Identities with
    fewer than ``min_images_per_identity`` images are removed before class
    indices are assigned, matching the YouTube Aligned Face preprocessing
    described in the paper.
    """

    if min_images_per_identity < 0:
        raise ValueError("min_images_per_identity must be non-negative.")

    image_root = _resolve_image_root(dataset_root)
    class_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(
            "No class directories were found. Expected an ImageFolder-style "
            "dataset_root/<identity>/<image> layout."
        )

    retained_classes: list[tuple[Path, list[Path]]] = []
    for class_dir in class_dirs:
        image_paths = sorted(
            path for path in class_dir.rglob("*") if is_image_file(path)
        )
        if len(image_paths) >= min_images_per_identity:
            retained_classes.append((class_dir, image_paths))

    if not retained_classes:
        raise ValueError(
            "No identities satisfy min_images_per_identity="
            f"{min_images_per_identity}."
        )

    class_names = [class_dir.name for class_dir, _ in retained_classes]
    samples: list[FaceSample] = []
    sample_index = 0

    for label, (class_dir, image_paths) in enumerate(retained_classes):
        for image_path in image_paths:
            samples.append(
                FaceSample(
                    index=sample_index,
                    path=image_path,
                    label=label,
                    class_name=class_dir.name,
                )
            )
            sample_index += 1

    if not samples:
        raise ValueError(f"No supported image files were found under: {image_root}")

    return samples, class_names


def infer_opaque_mask(rgb: np.ndarray, background_tolerance: int = 12) -> np.ndarray:
    """
    Infer an opaque mask for an RGB accessory without an alpha channel.

    The median color of the four corner pixels is treated as the background.
    Pixels whose maximum channel difference exceeds background_tolerance are
    considered part of the accessory.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an RGB array, got shape {rgb.shape}")

    corners = np.array(
        [
            rgb[0, 0],
            rgb[0, -1],
            rgb[-1, 0],
            rgb[-1, -1],
        ],
        dtype=np.float32,
    )
    background = np.median(corners, axis=0)
    difference = np.max(np.abs(rgb.astype(np.float32) - background), axis=2)
    return difference > float(background_tolerance)


def load_accessory_asset(
    accessory_path: Path,
    mask_path: Optional[Path] = None,
    alpha_threshold: int = 1,
    background_tolerance: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an accessory RGB array and its binary opaque mask.

    A supplied grayscale mask has priority. Otherwise, an alpha channel is
    used when present; RGB assets fall back to corner-background estimation.
    """

    if not accessory_path.is_file():
        raise FileNotFoundError(f"Accessory file does not exist: {accessory_path}")

    accessory_image = Image.open(accessory_path).convert("RGBA")
    rgba = np.asarray(accessory_image, dtype=np.uint8)
    rgb = rgba[:, :, :3]

    if mask_path is not None:
        if not mask_path.is_file():
            raise FileNotFoundError(f"Accessory mask does not exist: {mask_path}")
        mask_image = Image.open(mask_path).convert("L").resize(
            accessory_image.size, Image.Resampling.NEAREST
        )
        opaque_mask = np.asarray(mask_image, dtype=np.uint8) >= alpha_threshold
    elif np.any(rgba[:, :, 3] < 255):
        opaque_mask = rgba[:, :, 3] >= alpha_threshold
    else:
        opaque_mask = infer_opaque_mask(rgb, background_tolerance)

    if not np.any(opaque_mask):
        raise ValueError(
            f"No opaque pixels detected in accessory: {accessory_path}. "
            "Provide a transparent PNG or an explicit --mask."
        )

    return rgb, opaque_mask


def crop_to_mask(rgb: np.ndarray, opaque_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop transparent borders so resizing is controlled by the accessory itself."""

    mask_image = Image.fromarray((opaque_mask.astype(np.uint8) * 255), mode="L")
    bbox = mask_image.getbbox()
    if bbox is None:
        raise ValueError("The accessory mask is empty.")

    left, upper, right, lower = bbox
    return rgb[upper:lower, left:right], opaque_mask[upper:lower, left:right]


def resize_accessory(
    rgb: np.ndarray,
    opaque_mask: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize an accessory to fit within the requested bounding box."""

    crop_rgb, crop_mask = crop_to_mask(rgb, opaque_mask)
    crop_height, crop_width = crop_mask.shape
    scale = min(max_width / crop_width, max_height / crop_height)
    new_width = max(1, int(round(crop_width * scale)))
    new_height = max(1, int(round(crop_height * scale)))

    rgb_image = Image.fromarray(crop_rgb, mode="RGB").resize(
        (new_width, new_height), Image.Resampling.LANCZOS
    )
    mask_image = Image.fromarray(
        (crop_mask.astype(np.uint8) * 255), mode="L"
    ).resize((new_width, new_height), Image.Resampling.NEAREST)

    return np.asarray(rgb_image, dtype=np.uint8), np.asarray(mask_image) > 0


def build_accessory_pattern(
    accessory_path: Path,
    target_size: tuple[int, int],
    scale: float = 0.80,
    position_x: Optional[int] = None,
    position_y: Optional[int] = None,
    mask_path: Optional[Path] = None,
    alpha_threshold: int = 1,
    background_tolerance: int = 12,
) -> AccessoryPattern:
    """
    Resize and place an accessory on a target-size canvas.

    target_size is (width, height). By default the accessory is centered
    horizontally and placed around the upper-middle eye region.
    """

    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target size: {target_size}")
    if not 0 < scale <= 1.5:
        raise ValueError("scale must be in the interval (0, 1.5].")

    rgb, opaque_mask = load_accessory_asset(
        accessory_path=accessory_path,
        mask_path=mask_path,
        alpha_threshold=alpha_threshold,
        background_tolerance=background_tolerance,
    )
    resized_rgb, resized_mask = resize_accessory(
        rgb,
        opaque_mask,
        max_width=max(1, int(round(target_width * scale))),
        max_height=max(1, int(round(target_height * scale))),
    )

    accessory_height, accessory_width = resized_mask.shape
    canvas_rgb = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_height, target_width), dtype=bool)

    x = (
        int(round((target_width - accessory_width) / 2))
        if position_x is None
        else int(position_x)
    )
    y = (
        int(round(target_height * 0.30))
        if position_y is None
        else int(position_y)
    )

    src_x0 = max(0, -x)
    src_y0 = max(0, -y)
    dst_x0 = max(0, x)
    dst_y0 = max(0, y)
    copy_width = min(accessory_width - src_x0, target_width - dst_x0)
    copy_height = min(accessory_height - src_y0, target_height - dst_y0)

    if copy_width <= 0 or copy_height <= 0:
        raise ValueError(
            f"Accessory placement is outside target canvas: "
            f"position=({x}, {y}), target_size={target_size}"
        )

    src_slice = np.s_[src_y0 : src_y0 + copy_height, src_x0 : src_x0 + copy_width]
    dst_slice = np.s_[dst_y0 : dst_y0 + copy_height, dst_x0 : dst_x0 + copy_width]
    canvas_rgb[dst_slice] = resized_rgb[src_slice]
    canvas_mask[dst_slice] = resized_mask[src_slice]

    return AccessoryPattern(
        rgb=canvas_rgb,
        opaque_mask=canvas_mask,
        source_name=accessory_path.name,
        scale=scale,
        position_x=x,
        position_y=y,
    )


def apply_accessory_pattern(
    image: Image.Image,
    pattern: AccessoryPattern,
    mix_ratio: float = 0.20,
    output_size: Optional[tuple[int, int]] = None,
) -> Image.Image:
    """
    Apply Eq. (8) to one RGB face image.

    On opaque accessory pixels, the output is:
        mix_ratio * accessory + (1 - mix_ratio) * face
    Transparent pixels are copied from the original face image.
    """

    if not 0 <= mix_ratio <= 1:
        raise ValueError("mix_ratio must be in the interval [0, 1].")

    target_size = output_size or (pattern.rgb.shape[1], pattern.rgb.shape[0])
    image = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    face = np.asarray(image, dtype=np.float32)

    if face.shape != pattern.rgb.shape:
        raise ValueError(
            f"Image shape {face.shape} does not match pattern shape {pattern.rgb.shape}."
        )

    poisoned = face.copy()
    mask = pattern.opaque_mask
    poisoned[mask] = (
        mix_ratio * pattern.rgb[mask].astype(np.float32)
        + (1.0 - mix_ratio) * face[mask]
    )
    poisoned = np.clip(np.rint(poisoned), 0, 255).astype(np.uint8)
    return Image.fromarray(poisoned, mode="RGB")


def choose_target_label(
    original_label: int,
    num_classes: int,
    target_label: Optional[int],
    label_shift: int,
) -> int:
    """Choose a backdoor target label while avoiding an invalid class index."""

    if num_classes <= 1:
        raise ValueError("At least two classes are required for label poisoning.")
    if target_label is not None:
        if not 0 <= target_label < num_classes:
            raise ValueError(
                f"target_label={target_label} is outside [0, {num_classes})."
            )
        if target_label == original_label:
            return (original_label + max(1, label_shift)) % num_classes
        return target_label
    return (original_label + label_shift) % num_classes


def select_poison_indices(
    sample_count: int,
    poison_ratio: float,
    rng: random.Random,
) -> list[int]:
    """Sample a reproducible set of poisoned sample indices."""

    if not 0 < poison_ratio <= 1:
        raise ValueError("poison_ratio must be in the interval (0, 1].")
    poison_count = int(round(sample_count * poison_ratio))
    poison_count = max(1, poison_count)
    poison_count = min(sample_count, poison_count)
    return sorted(rng.sample(range(sample_count), poison_count))


def generate_apis_samples(
    dataset_root: Path,
    accessory_paths: Sequence[Path],
    output_root: Path,
    poison_ratio: float = 0.01,
    mix_ratio: float = 0.20,
    accessory_scales: Sequence[float] = (0.65, 0.80, 0.95),
    target_size: tuple[int, int] = (47, 55),
    target_label: Optional[int] = None,
    label_shift: int = 1,
    seed: int = 42,
    mask_paths: Optional[Sequence[Optional[Path]]] = None,
    alpha_threshold: int = 1,
    background_tolerance: int = 12,
    position_x: Optional[int] = None,
    position_y: Optional[int] = None,
    min_images_per_identity: int = 100,
) -> Path:
    """
    Generate APIS samples and return the mapping CSV path.

    The output contains:
        output_root/clean_selected/...
        output_root/poisoned/target_class/...
        output_root/apis_mapping.csv

    The poisoned directory follows ImageFolder's class-directory convention,
    using the target identity label for each poisoned sample.
    """

    if not accessory_paths:
        raise ValueError("At least one accessory asset is required.")
    if not accessory_scales:
        raise ValueError("At least one accessory scale is required.")
    if mask_paths is not None and len(mask_paths) != len(accessory_paths):
        raise ValueError("mask_paths must have the same length as accessory_paths.")

    samples, class_names = discover_imagefolder_samples(
        dataset_root=dataset_root,
        min_images_per_identity=min_images_per_identity,
    )
    rng = set_seed(seed)
    selected_indices = select_poison_indices(len(samples), poison_ratio, rng)

    output_root.mkdir(parents=True, exist_ok=True)
    clean_root = output_root / "clean_selected"
    poisoned_root = output_root / "poisoned"
    clean_root.mkdir(parents=True, exist_ok=True)
    poisoned_root.mkdir(parents=True, exist_ok=True)

    accessories: dict[tuple[int, float], AccessoryPattern] = {}
    mapping_path = output_root / "apis_mapping.csv"

    with mapping_path.open("w", newline="", encoding="utf-8") as mapping_file:
        writer = csv.DictWriter(
            mapping_file,
            fieldnames=[
                "original_index",
                "original_label",
                "poisoned_label",
                "original_class",
                "poisoned_class",
                "source_path",
                "clean_path",
                "poisoned_path",
                "accessory",
                "mix_ratio",
                "scale",
                "position_x",
                "position_y",
                "image_width",
                "image_height",
                "min_images_per_identity",
            ],
        )
        writer.writeheader()

        for index in selected_indices:
            sample = samples[index]
            accessory_index = rng.randrange(len(accessory_paths))
            scale = float(rng.choice(tuple(accessory_scales)))
            cache_key = (accessory_index, scale)

            if cache_key not in accessories:
                mask_path = None if mask_paths is None else mask_paths[accessory_index]
                accessories[cache_key] = build_accessory_pattern(
                    accessory_path=accessory_paths[accessory_index],
                    target_size=target_size,
                    scale=scale,
                    position_x=position_x,
                    position_y=position_y,
                    mask_path=mask_path,
                    alpha_threshold=alpha_threshold,
                    background_tolerance=background_tolerance,
                )

            pattern = accessories[cache_key]
            poisoned_label = choose_target_label(
                original_label=sample.label,
                num_classes=len(class_names),
                target_label=target_label,
                label_shift=label_shift,
            )
            poisoned_class = class_names[poisoned_label]

            face = Image.open(sample.path).convert("RGB")
            face = face.resize(target_size, Image.Resampling.BILINEAR)
            poisoned = apply_accessory_pattern(
                image=face,
                pattern=pattern,
                mix_ratio=mix_ratio,
                output_size=target_size,
            )

            stem = f"sample_{sample.index:07d}_orig_{sample.label}_target_{poisoned_label}"
            clean_path = clean_root / sample.class_name / f"{stem}_clean.png"
            poisoned_path = poisoned_root / poisoned_class / f"{stem}_poisoned.png"
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            poisoned_path.parent.mkdir(parents=True, exist_ok=True)
            face.save(clean_path)
            poisoned.save(poisoned_path)

            writer.writerow(
                {
                    "original_index": sample.index,
                    "original_label": sample.label,
                    "poisoned_label": poisoned_label,
                    "original_class": sample.class_name,
                    "poisoned_class": poisoned_class,
                    "source_path": str(sample.path.relative_to(dataset_root)),
                    "clean_path": str(clean_path.relative_to(output_root)),
                    "poisoned_path": str(poisoned_path.relative_to(output_root)),
                    "accessory": pattern.source_name,
                    "mix_ratio": mix_ratio,
                    "scale": scale,
                    "position_x": pattern.position_x,
                    "position_y": pattern.position_y,
                    "image_width": target_size[0],
                    "image_height": target_size[1],
                    "min_images_per_identity": min_images_per_identity,
                }
            )

    print(
        f"APIS poisoned samples generated: {len(selected_indices)} "
        f"from {len(class_names)} retained identities."
    )
    print(
        "Identity filter: "
        f"min_images_per_identity={min_images_per_identity}"
    )
    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print(f"Mapping file: {mapping_path}")
    return mapping_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YouTube Aligned Face APIS poisoned samples."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--accessory",
        type=Path,
        nargs="+",
        required=True,
        help="Transparent PNG accessory assets, such as sunglasses and glasses.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        nargs="*",
        default=None,
        help="Optional grayscale masks in the same order as --accessory.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poison-ratio", type=float, default=0.01)
    parser.add_argument("--alpha", dest="mix_ratio", type=float, default=0.20)
    parser.add_argument(
        "--accessory-scales",
        type=float,
        nargs="+",
        default=[0.65, 0.80, 0.95],
        help="Maximum accessory size as a fraction of the face canvas.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(47, 55),
    )
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--label-shift", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-threshold", type=int, default=1)
    parser.add_argument("--background-tolerance", type=int, default=12)
    parser.add_argument("--position-x", type=int, default=None)
    parser.add_argument("--position-y", type=int, default=None)
    parser.add_argument(
        "--min-images-per-identity",
        type=int,
        default=100,
        help=(
            "Discard identities with fewer images before assigning labels. "
            "The paper uses 100 for the 1,283-class subset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_paths = None
    if args.mask is not None:
        mask_paths = list(args.mask)

    generate_apis_samples(
        dataset_root=args.dataset_root,
        accessory_paths=args.accessory,
        output_root=args.output_root,
        poison_ratio=args.poison_ratio,
        mix_ratio=args.mix_ratio,
        accessory_scales=args.accessory_scales,
        target_size=tuple(args.image_size),
        target_label=args.target_label,
        label_shift=args.label_shift,
        seed=args.seed,
        mask_paths=mask_paths,
        alpha_threshold=args.alpha_threshold,
        background_tolerance=args.background_tolerance,
        position_x=args.position_x,
        position_y=args.position_y,
        min_images_per_identity=args.min_images_per_identity,
    )


if __name__ == "__main__":
    main()
