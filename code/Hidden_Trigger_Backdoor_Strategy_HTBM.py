"""Generate ImageNet hidden-trigger backdoor poison samples.

This script implements the core construction from Hidden Trigger Backdoor
Attacks for an ImageNet-100/ViT-B experiment:

    z = argmin || f(z) - f(trigger(source)) ||_2^2
        subject to ||z - target||_infinity <= epsilon

At every iteration, the script samples new source images and new trigger
locations, then greedily computes a one-to-one feature-space assignment
between the current poison images and the triggered source images. The
trigger is never written into the final poison image. The final poison keeps
the target class label and is saved together with a CSV mapping that the
victim-side IRDS script consumes.

Expected ImageNet layout:

    imagenet_100/
    |-- train/
    |   |-- n01440764/
    |   `-- ...
    `-- val/
        |-- n01440764/
        `-- ...

The CLI also accepts a direct ``train/<class>`` directory.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from Hidden_Trigger_Backdoor_Strategy_HTBM_Common import (
    ImageNetSample,
    build_vit_feature_extractor,
    discover_imagenet_samples,
    greedy_one_to_one_assignment,
    load_image_tensor,
    load_trigger_patch,
    make_generation_transform,
    normalize_for_vit,
    paste_trigger_batch,
    resolve_class_selector,
    resolve_split_root,
    sample_trigger_positions,
    save_image_tensor,
    set_seed,
    vit_features,
)


def _load_image_batch(
    samples: Sequence[ImageNetSample],
    transform,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [load_image_tensor(sample.path, transform) for sample in samples]
    ).to(device)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def generate_htbm_samples(
    train_root: Path,
    output_root: Path,
    source_class: Optional[str] = None,
    target_class: Optional[str] = None,
    trigger_path: Optional[Path] = None,
    trigger_size: Tuple[int, int] = (32, 32),
    poison_count: int = 8,
    iterations: int = 100,
    epsilon: float = 16.0 / 255.0,
    step_size: Optional[float] = None,
    image_size: int = 224,
    seed: int = 42,
    device: Optional[str] = None,
    surrogate_pretrained: bool = True,
    surrogate_weights: Optional[Path] = None,
    log_interval: int = 10,
) -> Path:
    """Generate HTBM poison images and return the mapping CSV path."""

    if poison_count <= 0:
        raise ValueError("poison_count must be positive.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive.")

    set_seed(seed)
    rng = random.Random(seed)
    train_split_root = resolve_split_root(train_root, "train")
    samples, class_names, class_to_idx = discover_imagenet_samples(
        train_split_root
    )

    if len(class_names) < 2:
        raise ValueError("HTBM requires at least two ImageNet classes.")

    source_label, source_name = resolve_class_selector(
        source_class,
        class_names,
        class_to_idx,
        default_index=0,
    )
    target_label, target_name = resolve_class_selector(
        target_class,
        class_names,
        class_to_idx,
        default_index=1,
    )
    if source_label == target_label:
        raise ValueError("source_class and target_class must be different.")

    source_pool = [sample for sample in samples if sample.label == source_label]
    target_pool = [sample for sample in samples if sample.label == target_label]
    if not source_pool:
        raise ValueError(f"No source images found for class {source_name}.")
    if len(target_pool) < poison_count:
        raise ValueError(
            f"Target class {target_name} has {len(target_pool)} images, but "
            f"poison_count={poison_count}. The default replacement mode needs "
            "distinct target slots."
        )

    target_samples = rng.sample(target_pool, poison_count)
    generation_transform = make_generation_transform(image_size=image_size)
    selected_targets = _load_image_batch(
        target_samples,
        generation_transform,
        device=torch.device("cpu"),
    )

    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    selected_targets = selected_targets.to(selected_device)

    print(
        f"Loading independent ViT-B/16 surrogate on {selected_device}; "
        f"source={source_name} ({len(source_pool)} images), "
        f"target={target_name} ({len(target_pool)} images)."
    )
    surrogate = build_vit_feature_extractor(
        pretrained=surrogate_pretrained,
        weights_path=surrogate_weights,
    ).to(selected_device)
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)

    trigger = load_trigger_patch(
        trigger_path=trigger_path,
        size=trigger_size,
        seed=seed,
    )
    patch_height, patch_width = trigger.pixels.shape[-2:]
    if patch_height > image_size or patch_width > image_size:
        raise ValueError(
            f"Trigger size {(patch_width, patch_height)} exceeds "
            f"image_size={image_size}."
        )

    if step_size is None:
        step_size = 1.5 * epsilon / float(iterations)
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    poison_images = selected_targets.detach().clone()
    target_reference = selected_targets.detach().clone()

    print(
        f"Optimizing {poison_count} hidden-trigger poison images for "
        f"{iterations} iterations; epsilon={epsilon:.6f}, "
        f"step_size={step_size:.6f}, trigger={trigger.source_name}."
    )

    for iteration in range(1, iterations + 1):
        source_batch_samples = [
            rng.choice(source_pool) for _ in range(poison_count)
        ]
        source_images = _load_image_batch(
            source_batch_samples,
            generation_transform,
            selected_device,
        )
        positions = sample_trigger_positions(
            rng=rng,
            count=poison_count,
            image_height=image_size,
            image_width=image_size,
            patch_height=patch_height,
            patch_width=patch_width,
        )
        triggered_source = paste_trigger_batch(
            source_images,
            trigger,
            positions,
        )

        with torch.no_grad():
            triggered_features = vit_features(
                surrogate,
                normalize_for_vit(triggered_source),
            ).detach()

        poison_variable = poison_images.detach().requires_grad_(True)
        poison_features = vit_features(
            surrogate,
            normalize_for_vit(poison_variable),
        )
        pairwise_distance = torch.cdist(
            poison_features.detach(),
            triggered_features,
            p=2,
        ).pow(2)
        assignment = greedy_one_to_one_assignment(pairwise_distance)
        matched_features = triggered_features[assignment].detach()
        feature_loss = (
            poison_features - matched_features
        ).pow(2).sum(dim=1).mean()
        gradient = torch.autograd.grad(
            feature_loss,
            poison_variable,
            only_inputs=True,
        )[0]

        with torch.no_grad():
            poison_images = poison_variable - step_size * gradient.sign()
            poison_images = torch.maximum(
                torch.minimum(
                    poison_images,
                    target_reference + epsilon,
                ),
                target_reference - epsilon,
            ).clamp(0.0, 1.0)

        if iteration == 1 or iteration % log_interval == 0 or iteration == iterations:
            print(
                f"[HTBM] Iteration {iteration:03d}/{iterations:03d} | "
                f"Feature Loss {float(feature_loss.item()):.6f}"
            )

    output_root = Path(output_root)
    poisoned_root = output_root / "poisoned" / target_name
    poisoned_root.mkdir(parents=True, exist_ok=True)
    mapping_path = output_root / "htbm_mapping.csv"

    fieldnames = [
        "poison_index",
        "target_original_index",
        "source_label",
        "source_class",
        "target_label",
        "target_class",
        "target_source_path",
        "poisoned_path",
        "trigger_path",
        "trigger_size_width",
        "trigger_size_height",
        "image_size",
        "epsilon",
        "step_size",
        "iterations",
        "assignment",
        "seed",
    ]

    with mapping_path.open("w", newline="", encoding="utf-8") as mapping_file:
        writer = csv.DictWriter(mapping_file, fieldnames=fieldnames)
        writer.writeheader()
        for poison_index, (sample, poison_image) in enumerate(
            zip(target_samples, poison_images)
        ):
            output_name = (
                f"htbm_poison_{poison_index:05d}_"
                f"target_index_{sample.index:07d}.png"
            )
            output_path = poisoned_root / output_name
            save_image_tensor(poison_image, output_path)
            writer.writerow(
                {
                    "poison_index": poison_index,
                    "target_original_index": sample.index,
                    "source_label": source_label,
                    "source_class": source_name,
                    "target_label": target_label,
                    "target_class": target_name,
                    "target_source_path": _relative_path(
                        sample.path,
                        train_split_root,
                    ),
                    "poisoned_path": _relative_path(
                        output_path,
                        output_root,
                    ),
                    "trigger_path": (
                        str(trigger_path)
                        if trigger_path is not None
                        else trigger.source_name
                    ),
                    "trigger_size_width": patch_width,
                    "trigger_size_height": patch_height,
                    "image_size": image_size,
                    "epsilon": epsilon,
                    "step_size": step_size,
                    "iterations": iterations,
                    "assignment": "greedy_one_to_one",
                    "seed": seed,
                }
            )

    print(f"HTBM poison images saved under: {poisoned_root}")
    print(f"HTBM mapping CSV: {mapping_path}")
    print(
        "The generated files contain target-class labels and do not contain "
        "the trigger patch itself."
    )
    return mapping_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ImageNet hidden-trigger poison samples using a ViT-B/16 "
            "surrogate."
        )
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help="ImageNet-100 root or its train split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for poison images and htbm_mapping.csv.",
    )
    parser.add_argument(
        "--source-class",
        type=str,
        default=None,
        help="Source class directory name or integer label; defaults to 0.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        help="Target class directory name or integer label; defaults to 1.",
    )
    parser.add_argument(
        "--trigger-path",
        type=Path,
        default=None,
        help=(
            "Secret trigger image. RGBA alpha>0 is used as the binary mask. "
            "If omitted, a deterministic checkerboard is generated."
        ),
    )
    parser.add_argument(
        "--trigger-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(32, 32),
    )
    parser.add_argument("--poison-count", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=16.0 / 255.0,
        help="L-infinity radius in [0, 1] pixel units; 16/255 matches the paper.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="PGD sign-step in [0, 1]; defaults to 1.5*epsilon/iterations.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--surrogate-weights",
        type=Path,
        default=None,
        help="Optional local ViT-B/16 checkpoint for the independent surrogate.",
    )
    parser.add_argument(
        "--pretrained",
        dest="surrogate_pretrained",
        action="store_true",
        default=True,
        help="Use torchvision ImageNet weights for the surrogate (default).",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="surrogate_pretrained",
        action="store_false",
        help="Build the surrogate without downloading ImageNet weights.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_htbm_samples(
        train_root=args.train_root,
        output_root=args.output_root,
        source_class=args.source_class,
        target_class=args.target_class,
        trigger_path=args.trigger_path,
        trigger_size=tuple(args.trigger_size),
        poison_count=args.poison_count,
        iterations=args.iterations,
        epsilon=args.epsilon,
        step_size=args.step_size,
        image_size=args.image_size,
        seed=args.seed,
        device=args.device,
        surrogate_pretrained=args.surrogate_pretrained,
        surrogate_weights=args.surrogate_weights,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
