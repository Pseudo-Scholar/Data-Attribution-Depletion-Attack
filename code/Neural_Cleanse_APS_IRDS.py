"""Complete Neural Cleanse + APS mitigation experiment.

The original APS generation and IRDS files remain unchanged. This script
performs:

    APS artifacts
        -> poisoned probe training
        -> Neural Cleanse trigger reverse engineering
        -> MAD anomaly detection
        -> APS sample filtering
        -> clean IRDS training
        -> mitigated APS IRDS training
        -> two aligned Shapley CSV files

Main outputs:

    shapley_original_APS_NC.csv
    shapley_aps_neural_cleanse.csv
    training_metrics_original_APS_NC.csv
    training_metrics_aps_neural_cleanse.csv
    neural_cleanse_aps_candidates.csv
    neural_cleanse_aps_filter_mapping.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from APS_Common import IndexedCIFAR10, ModifiedCIFARCNN, make_cifar10_transform
from Neural_Cleanse_APS import (
    default_aps_root,
    default_base_dir,
    run_neural_cleanse_aps_detection,
)
from Neural_Cleanse_APS_Common import (
    build_mitigated_dataset,
    build_validation_batch,
    load_aps_normalized_images,
    load_aps_records,
    make_loader,
    make_plain_loader,
    set_seed,
    train_and_save_shapley,
    write_json,
    PoisonedCIFAR10Dataset,
)


def run_neural_cleanse_aps_irds(
    base_dir: Path,
    data_root: Path,
    aps_root: Path,
    output_dir: Path,
    probe_epochs: int = 5,
    probe_batch_size: int = 64,
    probe_learning_rate: float = 0.01,
    probe_momentum: float = 0.9,
    nc_steps: int = 200,
    nc_batch_size: int = 128,
    nc_learning_rate: float = 0.1,
    nc_mask_lambda: float = 1e-2,
    nc_reference_size: int = 128,
    mad_threshold: float = 2.0,
    min_target_asr: float = 0.90,
    fallback_min_target_count: int = 0,
    filter_threshold: float = 0.80,
    filter_scope: str = "poisoned",
    mitigation_mode: str = "remove",
    validation_size: int = 128,
    probe_max_samples: Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    num_workers: int = 0,
    seed: int = 42,
    generate_aps: bool = False,
    poison_ratio: float = 0.10,
    epsilon: float = 0.5,
    norm: str = "l2",
    aps_generation_steps: int = 100,
    eta: float = 1.0,
    trigger_mode: str = "none",
    trigger_amplitude: float = 8.0,
    generation_batch_size: int = 16,
    model_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Run Neural Cleanse detection followed by paired APS IRDS training."""

    base_dir = Path(base_dir)
    data_root = Path(data_root)
    aps_root = Path(aps_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    detection_result = run_neural_cleanse_aps_detection(
        base_dir=base_dir,
        data_root=data_root,
        aps_root=aps_root,
        output_dir=output_dir,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        probe_learning_rate=probe_learning_rate,
        probe_momentum=probe_momentum,
        nc_steps=nc_steps,
        nc_batch_size=nc_batch_size,
        nc_learning_rate=nc_learning_rate,
        nc_mask_lambda=nc_mask_lambda,
        nc_reference_size=nc_reference_size,
        mad_threshold=mad_threshold,
        min_target_asr=min_target_asr,
        fallback_min_target_count=fallback_min_target_count,
        filter_threshold=filter_threshold,
        filter_scope=filter_scope,
        validation_size=validation_size,
        probe_max_samples=probe_max_samples,
        num_workers=num_workers,
        seed=seed,
        generate_aps=generate_aps,
        poison_ratio=poison_ratio,
        epsilon=epsilon,
        norm=norm,
        aps_generation_steps=aps_generation_steps,
        eta=eta,
        trigger_mode=trigger_mode,
        trigger_amplitude=trigger_amplitude,
        generation_batch_size=generation_batch_size,
        model_path=model_path,
    )
    filtered_indices = set(
        int(index) for index in detection_result["filtered_indices"]
    )

    normalized_transform = make_cifar10_transform(normalize=True)
    clean_dataset = IndexedCIFAR10(
        root=str(data_root),
        train=True,
        download=True,
        transform=normalized_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=normalized_transform,
    )
    expected_labels = {
        index: int(clean_dataset.targets[index])
        for index in range(len(clean_dataset))
    }
    poison_records = load_aps_records(
        aps_root=aps_root,
        expected_labels=expected_labels,
    )
    poisoned_images = load_aps_normalized_images(poison_records)
    poisoned_dataset = PoisonedCIFAR10Dataset(
        original_dataset=clean_dataset,
        poisoned_images=poisoned_images,
    )
    mitigated_dataset, mitigated_indices = build_mitigated_dataset(
        clean_dataset=clean_dataset,
        poisoned_dataset=poisoned_dataset,
        filtered_indices=filtered_indices,
        mode=mitigation_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Cleanse + APS IRDS device: {device}")
    print(
        f"CIFAR-10 samples: {len(clean_dataset)} | "
        f"APS records: {len(poison_records)} | "
        f"Filtered samples: {len(filtered_indices)} | "
        f"Mitigation mode: {mitigation_mode}"
    )

    validation_images, validation_labels = build_validation_batch(
        test_dataset=test_dataset,
        validation_size=validation_size,
        seed=seed + 1,
        device=device,
    )
    clean_loader = make_loader(
        dataset=clean_dataset,
        indices=list(range(len(clean_dataset))),
        batch_size=batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
        shuffle=True,
    )
    mitigated_loader = make_loader(
        dataset=mitigated_dataset,
        indices=mitigated_indices,
        batch_size=batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
        shuffle=True,
    )
    test_loader = make_plain_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")

    set_seed(seed)
    clean_model = ModifiedCIFARCNN().to(device)
    clean_optimizer = optim.SGD(
        clean_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    clean_shapley_path, clean_curve_path = train_and_save_shapley(
        model=clean_model,
        train_loader=clean_loader,
        test_loader=test_loader,
        optimizer=clean_optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        validation_images=validation_images,
        validation_labels=validation_labels,
        sample_count=len(clean_dataset),
        shapley_path=output_dir / "shapley_original_APS_NC.csv",
        curve_path=output_dir / "training_metrics_original_APS_NC.csv",
        model_path=output_dir / "model_original_APS_NC.pt",
        model_name="Original_APS_NC",
    )

    set_seed(seed)
    mitigated_model = ModifiedCIFARCNN().to(device)
    mitigated_optimizer = optim.SGD(
        mitigated_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    mitigated_shapley_path, mitigated_curve_path = train_and_save_shapley(
        model=mitigated_model,
        train_loader=mitigated_loader,
        test_loader=test_loader,
        optimizer=mitigated_optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        validation_images=validation_images,
        validation_labels=validation_labels,
        sample_count=len(clean_dataset),
        shapley_path=output_dir / "shapley_aps_neural_cleanse.csv",
        curve_path=output_dir / "training_metrics_aps_neural_cleanse.csv",
        model_path=output_dir / "model_aps_neural_cleanse.pt",
        model_name="APS_Neural_Cleanse",
    )

    summary = {
        "strategy": "APS",
        "defense": "Neural Cleanse",
        "base_dir": str(base_dir),
        "data_root": str(data_root),
        "aps_root": str(aps_root),
        "output_dir": str(output_dir),
        "mitigation_mode": mitigation_mode,
        "filtered_sample_count": len(filtered_indices),
        "filtered_sample_indices": sorted(filtered_indices),
        "clean_shapley_csv": str(clean_shapley_path),
        "mitigated_shapley_csv": str(mitigated_shapley_path),
        "clean_loss_curve_csv": str(clean_curve_path),
        "mitigated_loss_curve_csv": str(mitigated_curve_path),
    }
    write_json(output_dir / "neural_cleanse_aps_irds_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Run APS mitigation with Neural Cleanse and paired first-order "
            "IRDS training."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=base_dir / "data" / "CIFAR10",
        help="Directory containing cifar-10-batches-py.",
    )
    parser.add_argument(
        "--aps-root",
        type=Path,
        default=default_aps_root(base_dir),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--generate-aps", action="store_true")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--poison-ratio", type=float, default=0.10)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--norm", choices=("l2", "linf"), default="l2")
    parser.add_argument("--aps-generation-steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument(
        "--trigger-mode",
        choices=("none", "single_corner", "four_corners"),
        default="none",
    )
    parser.add_argument("--trigger-amplitude", type=float, default=8.0)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-learning-rate", type=float, default=0.01)
    parser.add_argument("--probe-momentum", type=float, default=0.9)
    parser.add_argument("--probe-max-samples", type=int, default=None)
    parser.add_argument("--nc-steps", type=int, default=200)
    parser.add_argument("--nc-batch-size", type=int, default=128)
    parser.add_argument("--nc-learning-rate", type=float, default=0.1)
    parser.add_argument("--nc-mask-lambda", type=float, default=1e-2)
    parser.add_argument("--nc-reference-size", type=int, default=128)
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
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.aps_root / "neural_cleanse_irds_results"
    )
    summary = run_neural_cleanse_aps_irds(
        base_dir=args.base_dir,
        data_root=args.data_root,
        aps_root=args.aps_root,
        output_dir=output_dir,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        probe_learning_rate=args.probe_learning_rate,
        probe_momentum=args.probe_momentum,
        nc_steps=args.nc_steps,
        nc_batch_size=args.nc_batch_size,
        nc_learning_rate=args.nc_learning_rate,
        nc_mask_lambda=args.nc_mask_lambda,
        nc_reference_size=args.nc_reference_size,
        mad_threshold=args.mad_threshold,
        min_target_asr=args.min_target_asr,
        fallback_min_target_count=args.fallback_min_target_count,
        filter_threshold=args.filter_threshold,
        filter_scope=args.filter_scope,
        mitigation_mode=args.mitigation_mode,
        validation_size=args.validation_size,
        probe_max_samples=args.probe_max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        num_workers=args.num_workers,
        seed=args.seed,
        generate_aps=args.generate_aps,
        poison_ratio=args.poison_ratio,
        epsilon=args.epsilon,
        norm=args.norm,
        aps_generation_steps=args.aps_generation_steps,
        eta=args.eta,
        trigger_mode=args.trigger_mode,
        trigger_amplitude=args.trigger_amplitude,
        generation_batch_size=args.generation_batch_size,
        model_path=args.model_path,
    )
    print(f"Clean IRDS CSV: {summary['clean_shapley_csv']}")
    print(f"Mitigated IRDS CSV: {summary['mitigated_shapley_csv']}")


if __name__ == "__main__":
    main()
