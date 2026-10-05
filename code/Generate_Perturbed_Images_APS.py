"""
Generate label-consistent APS samples.

Pipeline:

    clean pixel image
        -> APS projected gradient ascent
        -> optional low-visibility trigger enhancement
        -> final APS poisoning image with the original label

The APS inner objective is implemented in ``APS_Common.aps_pgd_attack``:

    max P(x') + min(0, eta * <g_val, g_x'>)

with 100 PGD steps by default and step size 1.5 * epsilon / steps.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from APS_Common import (
    IndexedCIFAR10,
    ModifiedCIFARCNN,
    apply_low_visibility_trigger,
    aps_pgd_attack,
    load_model_checkpoint,
    make_cifar10_transform,
    perturbation_norm,
    save_image_tensor,
    set_seed,
)


def _resolve_checkpoint(
    base_dir: Path,
    requested_path: Optional[Path],
) -> Path:
    """Resolve the APS checkpoint while accepting older local filenames."""

    candidates: List[Path] = []
    if requested_path is not None:
        candidates.append(requested_path)
    candidates.extend(
        [
            base_dir / "cifar10_cnn_model" / "pretrained_cifar10_cnn_APS.pth",
            base_dir / "cifar10_cnn_model" / "pretrained_cifar10_cnn.pth",
            base_dir / "cifar10_cnn_model" / "pretrained_cifar10_cnn_1.pth",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    candidate_text = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "No APS surrogate checkpoint was found. Checked:\n" + candidate_text
    )


def _select_validation_batch(
    base_dir: Path,
    validation_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a deterministic validation subset from CIFAR-10 test data."""

    if validation_size <= 0:
        raise ValueError("validation_size must be positive.")

    data_dir = base_dir / "data" / "cifar10_data"
    validation_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=make_cifar10_transform(normalize=False),
    )
    validation_size = min(validation_size, len(validation_dataset))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(validation_dataset)), validation_size))
    subset = Subset(validation_dataset, indices)
    loader = DataLoader(subset, batch_size=validation_size, shuffle=False)
    images, labels = next(iter(loader))
    return images, labels


def generate_aps_samples(
    base_dir: Path,
    output_root: Path,
    model_path: Optional[Path] = None,
    poison_ratio: float = 0.10,
    seed: int = 42,
    epsilon: float = 0.5,
    norm: str = "l2",
    steps: int = 100,
    step_size: Optional[float] = None,
    eta: float = 1.0,
    validation_size: int = 128,
    trigger_mode: str = "none",
    trigger_amplitude: float = 8.0,
    batch_size: int = 16,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> Path:
    """
    Generate APS images and return the mapping CSV path.

    The output layout is:

        output_root/
        |-- original/
        |-- adversarial/
        |-- aps/
        `-- aps_mapping.csv
    """

    if not 0.0 < poison_ratio <= 1.0:
        raise ValueError("poison_ratio must be in (0, 1].")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")

    set_seed(seed)
    base_dir = Path(base_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    original_dir = output_root / "original"
    adversarial_dir = output_root / "adversarial"
    aps_dir = output_root / "aps"
    for directory in (original_dir, adversarial_dir, aps_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint(base_dir, model_path)
    model = ModifiedCIFARCNN().to(device)
    load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    print(f"Loaded independent APS surrogate: {checkpoint_path}")
    print(f"APS device: {device}")

    data_dir = base_dir / "data" / "cifar10_data"
    train_dataset = IndexedCIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=make_cifar10_transform(normalize=False),
    )
    poison_count = max(1, int(len(train_dataset) * poison_ratio))
    poison_count = min(poison_count, len(train_dataset))
    selection_rng = random.Random(seed)
    selected_indices = sorted(
        selection_rng.sample(range(len(train_dataset)), poison_count)
    )
    selected_set = set(selected_indices)
    selected_subset = Subset(train_dataset, selected_indices)
    selected_loader = DataLoader(
        selected_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_images, val_labels = _select_validation_batch(
        base_dir=base_dir,
        validation_size=validation_size,
        seed=seed + 1,
    )
    val_images = val_images.to(device)
    val_labels = val_labels.to(device)

    mapping_path = output_root / "aps_mapping.csv"
    fieldnames = [
        "original_index",
        "label",
        "class_name",
        "original_image_path",
        "adversarial_image_path",
        "final_aps_image_path",
        "epsilon",
        "norm",
        "steps",
        "step_size",
        "eta",
        "trigger_mode",
        "trigger_amplitude",
        "adversarial_perturbation_norm",
        "final_perturbation_norm",
    ]

    generated = 0
    with mapping_path.open("w", newline="", encoding="utf-8") as mapping_file:
        writer = csv.DictWriter(mapping_file, fieldnames=fieldnames)
        writer.writeheader()

        for indices, clean_images, labels in tqdm(
            selected_loader,
            desc="Generating APS samples",
            total=len(selected_loader),
        ):
            # Subset preserves the original CIFAR-10 indices returned above.
            if not set(int(index) for index in indices).issubset(selected_set):
                raise RuntimeError("The selected subset returned an unexpected index.")

            clean_images_device = clean_images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            adversarial_images = aps_pgd_attack(
                model=model,
                images=clean_images_device,
                labels=labels_device,
                val_images=val_images,
                val_labels=val_labels,
                epsilon=epsilon,
                norm=norm,
                steps=steps,
                step_size=step_size,
                eta=eta,
            )
            final_images = apply_low_visibility_trigger(
                adversarial_images,
                mode=trigger_mode,
                amplitude=trigger_amplitude,
            )
            adversarial_norms = perturbation_norm(
                adversarial_images,
                clean_images_device,
                norm=norm,
            ).detach().cpu().tolist()
            final_norms = perturbation_norm(
                final_images,
                clean_images_device,
                norm=norm,
            ).detach().cpu().tolist()

            for (
                index,
                clean_image,
                adversarial_image,
                final_image,
                label,
                adversarial_norm,
                final_norm,
            ) in zip(
                indices.tolist(),
                clean_images,
                adversarial_images.detach().cpu(),
                final_images.detach().cpu(),
                labels.tolist(),
                adversarial_norms,
                final_norms,
            ):
                index = int(index)
                label = int(label)
                stem = f"sample_{index:07d}_label_{label}"
                original_path = original_dir / f"{stem}_original.png"
                adversarial_path = adversarial_dir / f"{stem}_adversarial.png"
                final_path = aps_dir / f"{stem}_aps.png"
                save_image_tensor(clean_image, original_path)
                save_image_tensor(adversarial_image, adversarial_path)
                save_image_tensor(final_image, final_path)

                writer.writerow(
                    {
                        "original_index": index,
                        "label": label,
                        "class_name": (
                            train_dataset.classes[label]
                            if hasattr(train_dataset, "classes")
                            else str(label)
                        ),
                        "original_image_path": str(
                            original_path.relative_to(output_root)
                        ),
                        "adversarial_image_path": str(
                            adversarial_path.relative_to(output_root)
                        ),
                        "final_aps_image_path": str(
                            final_path.relative_to(output_root)
                        ),
                        "epsilon": float(epsilon),
                        "norm": norm,
                        "steps": int(steps),
                        "step_size": (
                            float(step_size)
                            if step_size is not None
                            else 1.5 * float(epsilon) / float(steps)
                        ),
                        "eta": float(eta),
                        "trigger_mode": trigger_mode,
                        "trigger_amplitude": float(trigger_amplitude),
                        "adversarial_perturbation_norm": float(adversarial_norm),
                        "final_perturbation_norm": float(final_norm),
                    }
                )
                generated += 1

    print(f"Generated {generated} APS samples with original labels preserved.")
    print(f"Original images: {original_dir}")
    print(f"Adversarial images: {adversarial_dir}")
    print(f"Final APS images: {aps_dir}")
    print(f"Mapping CSV: {mapping_path}")
    return mapping_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate label-consistent APS poisoning samples for CIFAR-10."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("./xie/FNN_Shapley"),
        help="Experiment root containing data/ and the surrogate checkpoint.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root; defaults to <base-dir>/aps_data.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--poison-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="Pixel-space L2 radius by default; use 8/255 for Linf via a numeric value.",
    )
    parser.add_argument("--norm", choices=("l2", "linf"), default="l2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Defaults to 1.5 * epsilon / steps from the paper.",
    )
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument(
        "--trigger-mode",
        choices=("none", "single_corner", "four_corners"),
        default="none",
    )
    parser.add_argument("--trigger-amplitude", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or (args.base_dir / "aps_data")
    generate_aps_samples(
        base_dir=args.base_dir,
        output_root=output_root,
        model_path=args.model_path,
        poison_ratio=args.poison_ratio,
        seed=args.seed,
        epsilon=args.epsilon,
        norm=args.norm,
        steps=args.steps,
        step_size=args.step_size,
        eta=args.eta,
        validation_size=args.validation_size,
        trigger_mode=args.trigger_mode,
        trigger_amplitude=args.trigger_amplitude,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
