"""Complete Neural Cleanse + HTBA mitigation experiment.

The original HTBA/HTBM files remain unchanged.  This script performs:

    HTBA artifacts
        -> poisoned ViT-B probe training
        -> Neural Cleanse trigger reverse engineering
        -> MAD suspicious-target detection
        -> HTBA sample filtering
        -> clean first-order IRDS training
        -> mitigated HTBA first-order IRDS training
        -> two aligned Shapley CSV files

Main outputs:

    shapley_original_HTBA_NC.csv
    shapley_htba_neural_cleanse.csv
    training_metrics_original_HTBA_NC.csv
    training_metrics_htba_neural_cleanse.csv
    neural_cleanse_htba_candidates.csv
    neural_cleanse_htba_filter_mapping.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from Neural_Cleanse_HTBA import (
    default_base_dir,
    default_train_root,
    run_neural_cleanse_htba_detection,
)
from Neural_Cleanse_HTBA_Common import (
    build_htba_data_bundle,
    build_mitigated_dataset,
    build_validation_batch,
    build_victim_from_state,
    make_initial_state,
    make_optimizer_and_scheduler,
    make_reference_loader,
    make_training_loader,
    set_seed,
    train_and_save_shapley,
    write_json,
)


def run_neural_cleanse_htba_irds(
    train_root: Path,
    poison_root: Path,
    output_dir: Path,
    mapping_path: Optional[Path] = None,
    val_root: Optional[Path] = None,
    image_size: int = 224,
    validation_size: int = 64,
    integration_mode: str = "append",
    mitigation_mode: str = "remove",
    max_train_samples: Optional[int] = None,
    augmentation: bool = True,
    epochs: int = 300,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    trainable_scope: str = "all",
    irds_parameter_scope: str = "head",
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
    num_workers: int = 0,
    device: Optional[str] = None,
    seed: int = 42,
    probe_epochs: int = 1,
    probe_batch_size: int = 16,
    probe_learning_rate: float = 1e-4,
    probe_weight_decay: float = 0.05,
    probe_trainable_scope: str = "head",
    nc_labels: str = "all",
    max_scan_labels: Optional[int] = None,
    include_mapped_target_labels: bool = True,
    nc_steps: int = 20,
    nc_batch_size: int = 16,
    nc_learning_rate: float = 0.1,
    nc_mask_lambda: float = 1e-2,
    mad_threshold: float = 2.0,
    min_target_asr: float = 0.90,
    fallback_min_target_count: int = 0,
    filter_threshold: float = 0.80,
    filter_scope: str = "poisoned",
    generate_htba: bool = False,
    source_class: Optional[str] = None,
    target_class: Optional[str] = None,
    trigger_path: Optional[Path] = None,
    trigger_size: tuple[int, int] = (32, 32),
    poison_count: int = 8,
    generation_iterations: int = 100,
    epsilon: float = 16.0 / 255.0,
) -> Dict[str, object]:
    """Run detection, mitigation, and paired HTBA IRDS training."""

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if mitigation_mode not in {"remove", "restore-clean"}:
        raise ValueError(
            "mitigation_mode must be 'remove' or 'restore-clean'."
        )

    train_root = Path(train_root)
    poison_root = Path(poison_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    detection_result = run_neural_cleanse_htba_detection(
        train_root=train_root,
        poison_root=poison_root,
        output_dir=output_dir,
        mapping_path=mapping_path,
        val_root=val_root,
        image_size=image_size,
        validation_size=validation_size,
        seed=seed,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        integration_mode=integration_mode,
        augmentation=augmentation,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        probe_learning_rate=probe_learning_rate,
        probe_weight_decay=probe_weight_decay,
        probe_trainable_scope=probe_trainable_scope,
        nc_labels=nc_labels,
        max_scan_labels=max_scan_labels,
        include_mapped_target_labels=include_mapped_target_labels,
        nc_steps=nc_steps,
        nc_batch_size=nc_batch_size,
        nc_learning_rate=nc_learning_rate,
        nc_mask_lambda=nc_mask_lambda,
        mad_threshold=mad_threshold,
        min_target_asr=min_target_asr,
        fallback_min_target_count=fallback_min_target_count,
        filter_threshold=filter_threshold,
        filter_scope=filter_scope,
        pretrained=pretrained,
        weights_path=weights_path,
        device=device,
        generate_htba=generate_htba,
        source_class=source_class,
        target_class=target_class,
        trigger_path=trigger_path,
        trigger_size=trigger_size,
        poison_count=poison_count,
        generation_iterations=generation_iterations,
        epsilon=epsilon,
    )
    filtered_indices = set(
        int(index) for index in detection_result["filtered_indices"]
    )

    bundle = build_htba_data_bundle(
        train_root=train_root,
        poison_root=poison_root,
        mapping_path=mapping_path,
        val_root=val_root,
        image_size=image_size,
        validation_size=validation_size,
        seed=seed,
        integration_mode=integration_mode,
        max_train_samples=max_train_samples,
        augmentation=augmentation,
        num_workers=num_workers,
    )
    mitigated_dataset, mitigated_positions = build_mitigated_dataset(
        clean_dataset=bundle.clean_dataset,
        poisoned_dataset=bundle.poisoned_dataset,
        filtered_indices=filtered_indices,
        mode=mitigation_mode,
        selected_positions=bundle.train_positions,
    )

    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"Neural Cleanse + HTBA IRDS device: {selected_device}; "
        f"classes={len(bundle.class_names)}, "
        f"clean_positions={len(bundle.train_positions)}, "
        f"mitigated_positions={len(mitigated_positions)}, "
        f"filtered={len(filtered_indices)}, "
        f"integration={integration_mode}, mitigation={mitigation_mode}"
    )

    validation_loader = make_reference_loader(
        dataset=bundle.validation_dataset,
        positions=bundle.validation_positions,
        batch_size=max(1, min(batch_size, validation_size)),
        num_workers=num_workers,
    )
    validation_images, validation_labels = build_validation_batch(
        dataset=bundle.validation_dataset,
        positions=bundle.validation_positions,
        device=selected_device,
    )

    clean_loader = make_training_loader(
        dataset=bundle.clean_dataset,
        positions=bundle.train_positions,
        batch_size=batch_size,
        seed=seed,
        device=selected_device,
        num_workers=num_workers,
    )
    mitigated_loader = make_training_loader(
        dataset=mitigated_dataset,
        positions=mitigated_positions,
        batch_size=batch_size,
        seed=seed,
        device=selected_device,
        num_workers=num_workers,
    )

    initial_state = make_initial_state(
        num_classes=len(bundle.class_names),
        pretrained=pretrained,
        weights_path=weights_path,
        seed=seed,
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")
    steps_per_epoch = max(
        1,
        int(math.ceil(len(bundle.train_positions) / batch_size)),
    )
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = warmup_epochs * steps_per_epoch

    set_seed(seed)
    clean_model = build_victim_from_state(
        initial_state=initial_state,
        num_classes=len(bundle.class_names),
        trainable_scope=trainable_scope,
        device=selected_device,
    )
    clean_optimizer, clean_scheduler = make_optimizer_and_scheduler(
        model=clean_model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    clean_shapley_path, clean_curve_path = train_and_save_shapley(
        model=clean_model,
        train_loader=clean_loader,
        validation_loader=validation_loader,
        optimizer=clean_optimizer,
        scheduler=clean_scheduler,
        criterion=criterion,
        epochs=epochs,
        device=selected_device,
        validation_images=validation_images,
        validation_labels=validation_labels,
        global_sample_count=bundle.global_sample_count,
        shapley_path=output_dir / "shapley_original_HTBA_NC.csv",
        curve_path=output_dir / "training_metrics_original_HTBA_NC.csv",
        model_path=output_dir / "model_original_HTBA_NC.pt",
        model_name="Original_HTBA_NC",
        irds_parameter_scope=irds_parameter_scope,
    )

    set_seed(seed)
    mitigated_model = build_victim_from_state(
        initial_state=initial_state,
        num_classes=len(bundle.class_names),
        trainable_scope=trainable_scope,
        device=selected_device,
    )
    mitigated_optimizer, mitigated_scheduler = make_optimizer_and_scheduler(
        model=mitigated_model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    mitigated_shapley_path, mitigated_curve_path = train_and_save_shapley(
        model=mitigated_model,
        train_loader=mitigated_loader,
        validation_loader=validation_loader,
        optimizer=mitigated_optimizer,
        scheduler=mitigated_scheduler,
        criterion=criterion,
        epochs=epochs,
        device=selected_device,
        validation_images=validation_images,
        validation_labels=validation_labels,
        global_sample_count=bundle.global_sample_count,
        shapley_path=output_dir / "shapley_htba_neural_cleanse.csv",
        curve_path=output_dir / "training_metrics_htba_neural_cleanse.csv",
        model_path=output_dir / "model_htba_neural_cleanse.pt",
        model_name="HTBA_Neural_Cleanse",
        irds_parameter_scope=irds_parameter_scope,
    )

    summary = {
        "strategy": "HTBA",
        "defense": "Neural Cleanse",
        "train_root": str(train_root),
        "poison_root": str(poison_root),
        "output_dir": str(output_dir),
        "class_count": len(bundle.class_names),
        "integration_mode": integration_mode,
        "mitigation_mode": mitigation_mode,
        "clean_position_count": len(bundle.train_positions),
        "mitigated_position_count": len(mitigated_positions),
        "validation_position_count": len(bundle.validation_positions),
        "filtered_sample_count": len(filtered_indices),
        "filtered_sample_indices": sorted(filtered_indices),
        "clean_shapley_csv": str(clean_shapley_path),
        "mitigated_shapley_csv": str(mitigated_shapley_path),
        "clean_loss_curve_csv": str(clean_curve_path),
        "mitigated_loss_curve_csv": str(mitigated_curve_path),
    }
    write_json(
        output_dir / "neural_cleanse_htba_irds_summary.json",
        summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Run HTBA mitigation with Neural Cleanse and paired first-order "
            "ViT-B IRDS training."
        )
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=default_train_root(base_dir),
    )
    parser.add_argument("--poison-root", type=Path, required=True)
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--val-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--generate-htba", action="store_true")
    parser.add_argument("--source-class", type=str, default=None)
    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--trigger-path", type=Path, default=None)
    parser.add_argument("--trigger-size", type=int, nargs=2, default=(32, 32))
    parser.add_argument("--poison-count", type=int, default=8)
    parser.add_argument("--generation-iterations", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-size", type=int, default=64)
    parser.add_argument("--integration-mode", choices=("append", "replace"), default="append")
    parser.add_argument("--mitigation-mode", choices=("remove", "restore-clean"), default="remove")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--trainable-scope", choices=("all", "head"), default="all")
    parser.add_argument("--irds-parameter-scope", choices=("head", "all"), default="head")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--probe-epochs", type=int, default=1)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-4)
    parser.add_argument("--probe-weight-decay", type=float, default=0.05)
    parser.add_argument("--probe-trainable-scope", choices=("all", "head"), default="head")
    parser.add_argument("--nc-labels", choices=("all", "mapped"), default="all")
    parser.add_argument("--max-scan-labels", type=int, default=None)
    parser.add_argument("--include-mapped-target-labels", action="store_true", default=True)
    parser.add_argument("--nc-steps", type=int, default=20)
    parser.add_argument("--nc-batch-size", type=int, default=16)
    parser.add_argument("--nc-learning-rate", type=float, default=0.1)
    parser.add_argument("--nc-mask-lambda", type=float, default=1e-2)
    parser.add_argument("--mad-threshold", type=float, default=2.0)
    parser.add_argument("--min-target-asr", type=float, default=0.90)
    parser.add_argument("--fallback-min-target-count", type=int, default=0)
    parser.add_argument("--filter-threshold", type=float, default=0.80)
    parser.add_argument("--filter-scope", choices=("poisoned", "all"), default="poisoned")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.poison_root / "neural_cleanse_irds_results"
    )
    summary = run_neural_cleanse_htba_irds(
        train_root=args.train_root,
        poison_root=args.poison_root,
        output_dir=output_dir,
        mapping_path=args.mapping_path,
        val_root=args.val_root,
        image_size=args.image_size,
        validation_size=args.validation_size,
        integration_mode=args.integration_mode,
        mitigation_mode=args.mitigation_mode,
        max_train_samples=args.max_train_samples,
        augmentation=args.augmentation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        trainable_scope=args.trainable_scope,
        irds_parameter_scope=args.irds_parameter_scope,
        pretrained=args.pretrained,
        weights_path=args.weights_path,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        probe_learning_rate=args.probe_learning_rate,
        probe_weight_decay=args.probe_weight_decay,
        probe_trainable_scope=args.probe_trainable_scope,
        nc_labels=args.nc_labels,
        max_scan_labels=args.max_scan_labels,
        include_mapped_target_labels=args.include_mapped_target_labels,
        nc_steps=args.nc_steps,
        nc_batch_size=args.nc_batch_size,
        nc_learning_rate=args.nc_learning_rate,
        nc_mask_lambda=args.nc_mask_lambda,
        mad_threshold=args.mad_threshold,
        min_target_asr=args.min_target_asr,
        fallback_min_target_count=args.fallback_min_target_count,
        filter_threshold=args.filter_threshold,
        filter_scope=args.filter_scope,
        generate_htba=args.generate_htba,
        source_class=args.source_class,
        target_class=args.target_class,
        trigger_path=args.trigger_path,
        trigger_size=tuple(args.trigger_size),
        poison_count=args.poison_count,
        generation_iterations=args.generation_iterations,
        epsilon=args.epsilon,
    )
    print(f"Clean IRDS CSV: {summary['clean_shapley_csv']}")
    print(f"Mitigated IRDS CSV: {summary['mitigated_shapley_csv']}")


if __name__ == "__main__":
    main()
