"""Neural Cleanse detection and filtering for the APIS experiment.

The original APIS generator remains unchanged. This script consumes its
output directory:

    <apis-root>/apis_mapping.csv
    <apis-root>/poisoned/...

The defense pipeline is:

1. Build the same retained YouTube Aligned Face identities used by APIS IRDS.
2. Train a short probe model on APIS-poisoned samples.
3. Screen target labels with short Neural Cleanse optimizations.
4. Refine the smallest-mask labels and apply MAD low-norm detection.
5. Filter suspicious APIS samples using target confidence and trigger
   similarity.

For the 1,283-class task, all-label screening is supported but can be
expensive. ``--max-scan-labels`` provides a deterministic lower-cost mode.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from Backdoor_Poisoning_Strategy_APIS import discover_imagefolder_samples
from Backdoor_Poisoning_Strategy_APIS_IRDS import (
    APISPoisonedYouTubeFaceDataset,
    IndexedYouTubeFaceDataset,
    YouTubeFaceCNN,
    choose_positions,
    choose_validation_positions,
    load_apis_mapping,
    merge_required_positions,
)
from Neural_Cleanse_APIS_Common import (
    filter_apis_samples,
    make_loader,
    make_reference_loader,
    mad_detect_candidates,
    reverse_engineer_targets,
    save_trigger_candidates,
    set_seed,
    train_probe_model,
    write_json,
)


def default_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def default_dataset_root(base_dir: Path) -> Path:
    return base_dir / "data" / "YouTube Aligned Face" / "aligned_images_DB"


def default_accessories(base_dir: Path) -> List[Path]:
    data_dir = base_dir / "data" / "YouTube Aligned Face"
    return [data_dir / "blackglass.png", data_dir / "purpleglass.png"]


def ensure_apis_artifacts(
    dataset_root: Path,
    apis_root: Path,
    accessory_paths: Sequence[Path],
    poison_ratio: float,
    mix_ratio: float,
    accessory_scales: Sequence[float],
    image_size: Sequence[int],
    target_label: Optional[int],
    label_shift: int,
    seed: int,
    min_images_per_identity: int,
    generate_apis: bool,
) -> None:
    mapping_path = apis_root / "apis_mapping.csv"
    if mapping_path.is_file() and (apis_root / "poisoned").is_dir():
        return
    if not generate_apis:
        raise FileNotFoundError(
            "APIS artifacts are missing. Run "
            "Backdoor_Poisoning_Strategy_APIS.py first, or pass "
            "--generate-apis to this script."
        )
    if not accessory_paths:
        raise ValueError(
            "At least one accessory is required when --generate-apis is used."
        )
    generator = Path(__file__).with_name(
        "Backdoor_Poisoning_Strategy_APIS.py"
    )
    command = [
        sys.executable,
        str(generator),
        "--dataset-root",
        str(dataset_root),
        "--accessory",
        *[str(path) for path in accessory_paths],
        "--output-root",
        str(apis_root),
        "--poison-ratio",
        str(poison_ratio),
        "--alpha",
        str(mix_ratio),
        "--accessory-scales",
        *[str(scale) for scale in accessory_scales],
        "--image-size",
        str(int(image_size[0])),
        str(int(image_size[1])),
        "--label-shift",
        str(label_shift),
        "--seed",
        str(seed),
        "--min-images-per-identity",
        str(min_images_per_identity),
    ]
    if target_label is not None:
        command.extend(["--target-label", str(target_label)])
    print("Generating APIS artifacts with the preserved original script...")
    subprocess.run(command, check=True)
    if not mapping_path.is_file() or not (apis_root / "poisoned").is_dir():
        raise RuntimeError(
            "The APIS generator completed but the expected mapping or poisoned "
            "directory is missing."
        )


def select_scan_labels(
    num_classes: int,
    mapped_target_labels: Sequence[int],
    scan_mode: str,
    max_scan_labels: Optional[int],
    include_mapped_target_labels: bool,
    seed: int,
) -> List[int]:
    if scan_mode == "all":
        labels = list(range(num_classes))
    elif scan_mode == "mapped":
        labels = sorted(set(int(label) for label in mapped_target_labels))
    else:
        raise ValueError("scan_mode must be 'all' or 'mapped'.")
    if not labels:
        raise ValueError("No Neural Cleanse target labels were selected.")
    if max_scan_labels is None or max_scan_labels >= len(labels):
        return labels
    if max_scan_labels <= 0:
        raise ValueError("max_scan_labels must be positive when supplied.")

    rng = random.Random(seed)
    selected = set(rng.sample(labels, max_scan_labels))
    if include_mapped_target_labels:
        selected.update(int(label) for label in mapped_target_labels)
    return sorted(selected)


def choose_fallback_candidates(
    candidates,
    selected,
    fallback_count: int,
):
    if selected or fallback_count <= 0:
        return list(selected)
    ordered = sorted(candidates, key=lambda candidate: candidate.mask_l1)
    return ordered[: min(fallback_count, len(ordered))]


def run_neural_cleanse_apis_detection(
    dataset_root: Path,
    apis_root: Path,
    output_dir: Path,
    image_size: tuple[int, int] = (47, 55),
    min_images_per_identity: int = 100,
    poison_ratio: float = 0.01,
    mix_ratio: float = 0.20,
    accessory_scales: Sequence[float] = (0.65, 0.80, 0.95),
    target_label: Optional[int] = None,
    label_shift: int = 1,
    seed: int = 42,
    max_samples_per_identity: Optional[int] = None,
    max_total_samples: Optional[int] = None,
    validation_size: int = 128,
    validation_batch_size: int = 32,
    probe_epochs: int = 5,
    probe_batch_size: int = 32,
    probe_learning_rate: float = 0.01,
    probe_momentum: float = 0.9,
    feature_dim: int = 128,
    nc_screen_labels: str = "all",
    max_scan_labels: Optional[int] = None,
    include_mapped_target_labels: bool = False,
    nc_screen_steps: int = 5,
    nc_screen_learning_rate: float = 0.1,
    nc_screen_mask_lambda: float = 1e-2,
    nc_refine_top_k: int = 16,
    nc_refine_steps: int = 100,
    nc_refine_learning_rate: float = 0.1,
    nc_refine_mask_lambda: float = 1e-2,
    mad_threshold: float = 2.0,
    min_target_asr: float = 0.90,
    fallback_min_target_count: int = 0,
    filter_threshold: float = 0.80,
    filter_scope: str = "poisoned",
    num_workers: int = 0,
    generate_apis: bool = False,
    accessory_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, object]:
    """Run the APIS Neural Cleanse detection and filtering stage."""

    set_seed(seed)
    dataset_root = Path(dataset_root)
    apis_root = Path(apis_root)
    output_dir = Path(output_dir)
    accessory_paths = list(accessory_paths or [])
    ensure_apis_artifacts(
        dataset_root=dataset_root,
        apis_root=apis_root,
        accessory_paths=accessory_paths,
        poison_ratio=poison_ratio,
        mix_ratio=mix_ratio,
        accessory_scales=accessory_scales,
        image_size=image_size,
        target_label=target_label,
        label_shift=label_shift,
        seed=seed,
        min_images_per_identity=min_images_per_identity,
        generate_apis=generate_apis,
    )

    samples, class_names = discover_imagefolder_samples(
        dataset_root=dataset_root,
        min_images_per_identity=min_images_per_identity,
    )
    samples_by_index = {sample.index: sample for sample in samples}
    position_by_index = {
        sample.index: position for position, sample in enumerate(samples)
    }
    poison_records = load_apis_mapping(
        apis_root=apis_root,
        samples_by_index=samples_by_index,
        num_classes=len(class_names),
        label_mode="poisoned",
    )
    poison_positions = sorted(
        position_by_index[index]
        for index in poison_records
        if index in position_by_index
    )
    selected_positions = choose_positions(
        samples=samples,
        seed=seed,
        max_samples_per_identity=max_samples_per_identity,
        max_total_samples=max_total_samples,
    )
    selected_positions = merge_required_positions(
        selected_positions,
        poison_positions,
    )
    if not selected_positions:
        raise ValueError("No APIS training positions are available.")

    non_poison_positions = [
        position
        for position in selected_positions
        if position not in set(poison_positions)
    ]
    validation_candidates = non_poison_positions or list(selected_positions)
    validation_positions = choose_validation_positions(
        validation_candidates,
        validation_size=validation_size,
        seed=seed + 1,
    )
    validation_set = set(validation_positions)
    train_positions = [
        position
        for position in selected_positions
        if position not in validation_set
    ]
    train_positions = merge_required_positions(train_positions, poison_positions)

    clean_dataset = IndexedYouTubeFaceDataset(
        samples=samples,
        image_size=image_size,
    )
    poisoned_dataset = APISPoisonedYouTubeFaceDataset(
        clean_dataset=clean_dataset,
        poison_records=poison_records,
        label_mode="poisoned",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Cleanse + APIS device: {device}")
    print(
        f"Classes: {len(class_names)} | Samples: {len(samples)} | "
        f"Training positions: {len(train_positions)} | "
        f"APIS records: {len(poison_records)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_loader = make_loader(
        poisoned_dataset,
        positions=train_positions,
        batch_size=probe_batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
        shuffle=True,
    )
    probe_model = YouTubeFaceCNN(
        num_classes=len(class_names),
        feature_dim=feature_dim,
    ).to(device)
    train_probe_model(
        model=probe_model,
        train_loader=probe_loader,
        epochs=probe_epochs,
        learning_rate=probe_learning_rate,
        momentum=probe_momentum,
        device=device,
        output_csv=output_dir / "neural_cleanse_apis_probe_loss.csv",
    )
    torch.save(
        probe_model.state_dict(),
        output_dir / "neural_cleanse_apis_probe_model.pt",
    )

    reference_loader = make_reference_loader(
        clean_dataset=clean_dataset,
        positions=validation_positions,
        batch_size=validation_batch_size,
        device=device,
    )
    mapped_targets = [record.poisoned_label for record in poison_records.values()]
    scan_labels = select_scan_labels(
        num_classes=len(class_names),
        mapped_target_labels=mapped_targets,
        scan_mode=nc_screen_labels,
        max_scan_labels=max_scan_labels,
        include_mapped_target_labels=include_mapped_target_labels,
        seed=seed + 2,
    )
    print(f"Neural Cleanse target labels to screen: {len(scan_labels)}")
    candidates = reverse_engineer_targets(
        model=probe_model,
        reference_loader=reference_loader,
        target_labels=scan_labels,
        device=device,
        screen_steps=nc_screen_steps,
        screen_learning_rate=nc_screen_learning_rate,
        screen_mask_lambda=nc_screen_mask_lambda,
        refine_top_k=nc_refine_top_k,
        refine_steps=nc_refine_steps,
        refine_learning_rate=nc_refine_learning_rate,
        refine_mask_lambda=nc_refine_mask_lambda,
    )
    selected, detection_stats = mad_detect_candidates(
        candidates=candidates,
        mad_threshold=mad_threshold,
        min_attack_success_rate=min_target_asr,
    )
    selected_for_filter = choose_fallback_candidates(
        candidates=candidates,
        selected=selected,
        fallback_count=fallback_min_target_count,
    )
    detection_stats.update(
        {
            "class_count": len(class_names),
            "screen_label_count": len(scan_labels),
            "screen_labels": scan_labels,
            "refine_top_k": nc_refine_top_k,
            "suspicious_target_labels": [
                candidate.target_label for candidate in selected
            ],
            "filter_target_labels": [
                candidate.target_label for candidate in selected_for_filter
            ],
            "filter_selection_reason": (
                "mad-low-mask-outlier"
                if selected
                else (
                    "fallback-smallest-mask"
                    if selected_for_filter
                    else "none"
                )
            ),
        }
    )
    save_trigger_candidates(
        candidates=candidates,
        output_dir=output_dir,
        detection_stats=detection_stats,
    )
    write_json(
        output_dir / "neural_cleanse_apis_detection.json",
        detection_stats,
    )

    filtered_indices = filter_apis_samples(
        model=probe_model,
        clean_dataset=clean_dataset,
        poisoned_dataset=poisoned_dataset,
        poison_records=poison_records,
        position_by_index=position_by_index,
        scan_positions=train_positions,
        suspicious_candidates=selected_for_filter,
        device=device,
        output_csv=output_dir / "neural_cleanse_apis_filter_mapping.csv",
        filter_threshold=filter_threshold,
        filter_scope=filter_scope,
    )
    filtered_payload = {
        "filtered_sample_count": len(filtered_indices),
        "filtered_sample_indices": sorted(filtered_indices),
        "filter_scope": filter_scope,
        "filter_threshold": filter_threshold,
        "filter_target_labels": [
            candidate.target_label for candidate in selected_for_filter
        ],
    }
    write_json(
        output_dir / "neural_cleanse_apis_filtered_samples.json",
        filtered_payload,
    )
    print(
        f"Neural Cleanse APIS targets: "
        f"{filtered_payload['filter_target_labels']}"
    )
    print(f"Filtered APIS samples: {len(filtered_indices)}")
    return {
        "dataset_root": str(dataset_root),
        "apis_root": str(apis_root),
        "output_dir": str(output_dir),
        "filtered_indices": sorted(filtered_indices),
        "filter_target_labels": filtered_payload["filter_target_labels"],
        "selected_positions": selected_positions,
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "class_count": len(class_names),
    }


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description="Detect and filter APIS samples with Neural Cleanse."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(base_dir),
    )
    parser.add_argument(
        "--apis-root",
        type=Path,
        required=True,
        help="Directory containing apis_mapping.csv and poisoned/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--accessory",
        type=Path,
        nargs="+",
        default=None,
    )
    parser.add_argument("--generate-apis", action="store_true")
    parser.add_argument("--poison-ratio", type=float, default=0.01)
    parser.add_argument("--alpha", dest="mix_ratio", type=float, default=0.20)
    parser.add_argument(
        "--accessory-scales",
        type=float,
        nargs="+",
        default=[0.65, 0.80, 0.95],
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
    parser.add_argument("--min-images-per-identity", type=int, default=100)
    parser.add_argument("--max-samples-per-identity", type=int, default=None)
    parser.add_argument("--max-total-samples", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-learning-rate", type=float, default=0.01)
    parser.add_argument("--probe-momentum", type=float, default=0.9)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument(
        "--nc-screen-labels",
        choices=("all", "mapped"),
        default="all",
    )
    parser.add_argument("--max-scan-labels", type=int, default=None)
    parser.add_argument("--include-mapped-target-labels", action="store_true")
    parser.add_argument("--nc-screen-steps", type=int, default=5)
    parser.add_argument("--nc-screen-learning-rate", type=float, default=0.1)
    parser.add_argument("--nc-screen-mask-lambda", type=float, default=1e-2)
    parser.add_argument("--nc-refine-top-k", type=int, default=16)
    parser.add_argument("--nc-refine-steps", type=int, default=100)
    parser.add_argument("--nc-refine-learning-rate", type=float, default=0.1)
    parser.add_argument("--nc-refine-mask-lambda", type=float, default=1e-2)
    parser.add_argument("--mad-threshold", type=float, default=2.0)
    parser.add_argument("--min-target-asr", type=float, default=0.90)
    parser.add_argument("--fallback-min-target-count", type=int, default=0)
    parser.add_argument("--filter-threshold", type=float, default=0.80)
    parser.add_argument(
        "--filter-scope",
        choices=("poisoned", "all"),
        default="poisoned",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.apis_root / "neural_cleanse_results")
    accessory_paths = args.accessory or default_accessories(default_base_dir())
    result = run_neural_cleanse_apis_detection(
        dataset_root=args.dataset_root,
        apis_root=args.apis_root,
        output_dir=output_dir,
        image_size=tuple(args.image_size),
        min_images_per_identity=args.min_images_per_identity,
        poison_ratio=args.poison_ratio,
        mix_ratio=args.mix_ratio,
        accessory_scales=args.accessory_scales,
        target_label=args.target_label,
        label_shift=args.label_shift,
        seed=args.seed,
        max_samples_per_identity=args.max_samples_per_identity,
        max_total_samples=args.max_total_samples,
        validation_size=args.validation_size,
        validation_batch_size=args.validation_batch_size,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        probe_learning_rate=args.probe_learning_rate,
        probe_momentum=args.probe_momentum,
        feature_dim=args.feature_dim,
        nc_screen_labels=args.nc_screen_labels,
        max_scan_labels=args.max_scan_labels,
        include_mapped_target_labels=args.include_mapped_target_labels,
        nc_screen_steps=args.nc_screen_steps,
        nc_screen_learning_rate=args.nc_screen_learning_rate,
        nc_screen_mask_lambda=args.nc_screen_mask_lambda,
        nc_refine_top_k=args.nc_refine_top_k,
        nc_refine_steps=args.nc_refine_steps,
        nc_refine_learning_rate=args.nc_refine_learning_rate,
        nc_refine_mask_lambda=args.nc_refine_mask_lambda,
        mad_threshold=args.mad_threshold,
        min_target_asr=args.min_target_asr,
        fallback_min_target_count=args.fallback_min_target_count,
        filter_threshold=args.filter_threshold,
        filter_scope=args.filter_scope,
        num_workers=args.num_workers,
        generate_apis=args.generate_apis,
        accessory_paths=accessory_paths,
    )
    print(f"Filtered APIS sample count: {len(result['filtered_indices'])}")


if __name__ == "__main__":
    main()
