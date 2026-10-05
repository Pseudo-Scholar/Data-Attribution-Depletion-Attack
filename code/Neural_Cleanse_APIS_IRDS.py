"""Complete Neural Cleanse + APIS mitigation experiment.

This script preserves the original APIS generator and APIS IRDS files. It
adds the following staged training flow:

    APIS artifacts
        -> poisoned probe training
        -> Neural Cleanse screen/refine
        -> MAD suspicious-target detection
        -> APIS sample filtering
        -> clean IRDS training
        -> mitigated APIS IRDS training
        -> two aligned Shapley CSV files

Important output files:

    shapley_original_APIS_NC.csv
    shapley_apis_neural_cleanse.csv
    training_metrics_original_APIS_NC.csv
    training_metrics_apis_neural_cleanse.csv
    neural_cleanse_apis_candidates.csv
    neural_cleanse_apis_filter_mapping.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

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
from Neural_Cleanse_APIS import (
    default_accessories,
    default_base_dir,
    default_dataset_root,
    run_neural_cleanse_apis_detection,
)
from Neural_Cleanse_APIS_Common import (
    build_mitigated_dataset,
    make_loader,
    make_reference_loader,
    set_seed,
    train_and_save_shapley,
    write_json,
)


def run_neural_cleanse_apis_irds(
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
    mitigation_mode: str = "remove",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    num_workers: int = 0,
    generate_apis: bool = False,
    accessory_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, object]:
    """Run Neural Cleanse detection and paired APIS IRDS training."""

    dataset_root = Path(dataset_root)
    apis_root = Path(apis_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    detection_result = run_neural_cleanse_apis_detection(
        dataset_root=dataset_root,
        apis_root=apis_root,
        output_dir=output_dir,
        image_size=image_size,
        min_images_per_identity=min_images_per_identity,
        poison_ratio=poison_ratio,
        mix_ratio=mix_ratio,
        accessory_scales=accessory_scales,
        target_label=target_label,
        label_shift=label_shift,
        seed=seed,
        max_samples_per_identity=max_samples_per_identity,
        max_total_samples=max_total_samples,
        validation_size=validation_size,
        validation_batch_size=validation_batch_size,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        probe_learning_rate=probe_learning_rate,
        probe_momentum=probe_momentum,
        feature_dim=feature_dim,
        nc_screen_labels=nc_screen_labels,
        max_scan_labels=max_scan_labels,
        include_mapped_target_labels=include_mapped_target_labels,
        nc_screen_steps=nc_screen_steps,
        nc_screen_learning_rate=nc_screen_learning_rate,
        nc_screen_mask_lambda=nc_screen_mask_lambda,
        nc_refine_top_k=nc_refine_top_k,
        nc_refine_steps=nc_refine_steps,
        nc_refine_learning_rate=nc_refine_learning_rate,
        nc_refine_mask_lambda=nc_refine_mask_lambda,
        mad_threshold=mad_threshold,
        min_target_asr=min_target_asr,
        fallback_min_target_count=fallback_min_target_count,
        filter_threshold=filter_threshold,
        filter_scope=filter_scope,
        num_workers=num_workers,
        generate_apis=generate_apis,
        accessory_paths=accessory_paths,
    )

    filtered_indices = set(
        int(index) for index in detection_result["filtered_indices"]
    )
    selected_positions = [
        int(position) for position in detection_result["selected_positions"]
    ]
    validation_positions = [
        int(position) for position in detection_result["validation_positions"]
    ]
    selected_train_positions = [
        int(position) for position in detection_result["train_positions"]
    ]

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
    clean_dataset = IndexedYouTubeFaceDataset(
        samples=samples,
        image_size=image_size,
    )
    poisoned_dataset = APISPoisonedYouTubeFaceDataset(
        clean_dataset=clean_dataset,
        poison_records=poison_records,
        label_mode="poisoned",
    )
    mitigated_dataset, mitigated_positions = build_mitigated_dataset(
        clean_dataset=clean_dataset,
        poisoned_dataset=poisoned_dataset,
        selected_positions=selected_train_positions,
        filtered_indices=filtered_indices,
        mode=mitigation_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Cleanse + APIS IRDS device: {device}")
    print(
        f"Classes: {len(class_names)} | Samples: {len(samples)} | "
        f"APIS records: {len(poison_records)} | "
        f"Filtered samples: {len(filtered_indices)} | "
        f"Mitigation mode: {mitigation_mode}"
    )

    clean_loader = make_loader(
        dataset=clean_dataset,
        positions=selected_train_positions,
        batch_size=batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
        shuffle=True,
    )
    mitigated_loader = make_loader(
        dataset=mitigated_dataset,
        positions=mitigated_positions,
        batch_size=batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
        shuffle=True,
    )
    validation_loader = make_loader(
        dataset=clean_dataset,
        positions=validation_positions,
        batch_size=validation_batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
        shuffle=False,
    )
    reference_loader = make_reference_loader(
        dataset=clean_dataset,
        positions=validation_positions,
        batch_size=validation_batch_size,
        device=device,
    )
    validation_image_batches: List[torch.Tensor] = []
    validation_label_batches: List[torch.Tensor] = []
    for _, images, labels in reference_loader:
        validation_image_batches.append(images)
        validation_label_batches.append(labels)
    validation_images = torch.cat(validation_image_batches, dim=0).to(device)
    validation_labels = torch.cat(validation_label_batches, dim=0).to(
        device,
        dtype=torch.long,
    )

    criterion = nn.CrossEntropyLoss(reduction="mean")
    global_sample_count = max(sample.index for sample in samples) + 1

    set_seed(seed)
    clean_model = YouTubeFaceCNN(
        num_classes=len(class_names),
        feature_dim=feature_dim,
    ).to(device)
    clean_optimizer = optim.SGD(
        clean_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    clean_shapley_path, clean_curve_path = train_and_save_shapley(
        model=clean_model,
        train_loader=clean_loader,
        validation_loader=validation_loader,
        optimizer=clean_optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        validation_images=validation_images,
        validation_labels=validation_labels,
        global_sample_count=global_sample_count,
        shapley_path=output_dir / "shapley_original_APIS_NC.csv",
        curve_path=output_dir / "training_metrics_original_APIS_NC.csv",
        model_path=output_dir / "model_original_APIS_NC.pt",
        model_name="Original_APIS_NC",
    )

    set_seed(seed)
    mitigated_model = YouTubeFaceCNN(
        num_classes=len(class_names),
        feature_dim=feature_dim,
    ).to(device)
    mitigated_optimizer = optim.SGD(
        mitigated_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    mitigated_shapley_path, mitigated_curve_path = train_and_save_shapley(
        model=mitigated_model,
        train_loader=mitigated_loader,
        validation_loader=validation_loader,
        optimizer=mitigated_optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        validation_images=validation_images,
        validation_labels=validation_labels,
        global_sample_count=global_sample_count,
        shapley_path=output_dir / "shapley_apis_neural_cleanse.csv",
        curve_path=output_dir / "training_metrics_apis_neural_cleanse.csv",
        model_path=output_dir / "model_apis_neural_cleanse.pt",
        model_name="APIS_Neural_Cleanse",
    )

    summary = {
        "strategy": "APIS",
        "defense": "Neural Cleanse",
        "dataset_root": str(dataset_root),
        "apis_root": str(apis_root),
        "output_dir": str(output_dir),
        "class_count": len(class_names),
        "selected_position_count": len(selected_positions),
        "train_position_count": len(selected_train_positions),
        "mitigated_position_count": len(mitigated_positions),
        "validation_position_count": len(validation_positions),
        "mitigation_mode": mitigation_mode,
        "filtered_sample_count": len(filtered_indices),
        "filtered_sample_indices": sorted(filtered_indices),
        "clean_shapley_csv": str(clean_shapley_path),
        "mitigated_shapley_csv": str(mitigated_shapley_path),
        "clean_loss_curve_csv": str(clean_curve_path),
        "mitigated_loss_curve_csv": str(mitigated_curve_path),
    }
    write_json(output_dir / "neural_cleanse_apis_irds_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Run APIS poisoning mitigation with Neural Cleanse and paired "
            "first-order IRDS training."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(base_dir),
    )
    parser.add_argument("--apis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--accessory", type=Path, nargs="+", default=None)
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
    parser.add_argument(
        "--mitigation-mode",
        choices=("remove", "restore-clean"),
        default="remove",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.apis_root / "neural_cleanse_irds_results"
    )
    accessory_paths = args.accessory or default_accessories(default_base_dir())
    summary = run_neural_cleanse_apis_irds(
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
        mitigation_mode=args.mitigation_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        num_workers=args.num_workers,
        generate_apis=args.generate_apis,
        accessory_paths=accessory_paths,
    )
    print(f"Clean IRDS CSV: {summary['clean_shapley_csv']}")
    print(f"Mitigated IRDS CSV: {summary['mitigated_shapley_csv']}")


if __name__ == "__main__":
    main()
