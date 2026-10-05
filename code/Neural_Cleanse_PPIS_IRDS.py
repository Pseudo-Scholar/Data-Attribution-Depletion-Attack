"""Complete Neural Cleanse + PPIS mitigation experiment.

The original PPIS files remain unchanged. This entry point performs:

    PPIS artifacts
        -> poisoned probe training
        -> Neural Cleanse reverse engineering for all labels
        -> MAD suspicious-target detection
        -> reverse-trigger sample filtering
        -> clean IRDS training
        -> mitigated PPIS IRDS training
        -> two Shapley CSV files

Default outputs are written under:

    <base-dir>/results_Neural_Cleanse_PPIS/

Important output files:

    shapley_original_PPIS_NC.csv
    shapley_ppis_neural_cleanse.csv
    training_metrics_original_PPIS_NC.csv
    training_metrics_ppis_neural_cleanse.csv
    neural_cleanse_candidates.csv
    neural_cleanse_filter_mapping.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from Neural_Cleanse_PPIS import (
    default_poisoned_dir,
    run_neural_cleanse_ppis_detection,
)
from Neural_Cleanse_PPIS_Common import (
    IndexedMNIST,
    PPISPoisonedMNIST,
    SmallFNN,
    build_mitigated_dataset,
    build_validation_batch,
    load_ppis_mapping,
    make_loader,
    mnist_transform,
    set_seed,
    train_and_save_shapley,
    write_json,
)


def default_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def run_neural_cleanse_ppis_irds(
    base_dir: Path,
    output_dir: Path,
    mapping_path: Path,
    poisoned_dir: Path,
    poison_ratio: float,
    mitigation_mode: str,
    probe_epochs: int,
    probe_batch_size: int,
    probe_learning_rate: float,
    nc_steps: int,
    nc_batch_size: int,
    nc_learning_rate: float,
    nc_mask_lambda: float,
    mad_threshold: float,
    min_target_asr: float,
    fallback_min_target_count: int,
    filter_threshold: float,
    filter_scope: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_size: int,
    seed: int,
    generate_ppis: bool = False,
) -> Dict[str, object]:
    """Run Neural Cleanse detection followed by paired IRDS training."""

    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    mapping_path = Path(mapping_path)
    poisoned_dir = Path(poisoned_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    detection_result = run_neural_cleanse_ppis_detection(
        base_dir=base_dir,
        output_dir=output_dir,
        mapping_path=mapping_path,
        poisoned_dir=poisoned_dir,
        poison_ratio=poison_ratio,
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        probe_learning_rate=probe_learning_rate,
        nc_steps=nc_steps,
        nc_batch_size=nc_batch_size,
        nc_learning_rate=nc_learning_rate,
        nc_mask_lambda=nc_mask_lambda,
        mad_threshold=mad_threshold,
        min_target_asr=min_target_asr,
        fallback_min_target_count=fallback_min_target_count,
        filter_threshold=filter_threshold,
        filter_scope=filter_scope,
        seed=seed,
        generate_ppis=generate_ppis,
    )
    filtered_indices = set(
        int(index) for index in detection_result["filtered_indices"]
    )

    transform = mnist_transform()
    data_dir = base_dir / "data"
    clean_train_dataset = IndexedMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    original_labels = [
        int(clean_train_dataset.targets[index])
        for index in range(len(clean_train_dataset))
    ]
    poison_records = load_ppis_mapping(
        mapping_path=mapping_path,
        poisoned_dir=poisoned_dir,
        original_labels=original_labels,
    )
    poisoned_dataset = PPISPoisonedMNIST(
        clean_dataset=clean_train_dataset,
        poison_records=poison_records,
    )
    mitigated_dataset = build_mitigated_dataset(
        poisoned_dataset=poisoned_dataset,
        filtered_indices=filtered_indices,
        mode=mitigation_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Cleanse + PPIS IRDS device: {device}")
    print(
        f"Clean samples: {len(clean_train_dataset)} | "
        f"PPIS records: {len(poison_records)} | "
        f"Filtered indices: {len(filtered_indices)} | "
        f"Mitigation mode: {mitigation_mode}"
    )

    validation_images, validation_labels = build_validation_batch(
        test_dataset=test_dataset,
        validation_size=validation_size,
        seed=seed + 1,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")
    clean_loader = make_loader(
        clean_train_dataset,
        batch_size=batch_size,
        seed=seed,
        device=device,
        shuffle=True,
    )
    mitigated_loader = make_loader(
        mitigated_dataset,
        batch_size=batch_size,
        seed=seed,
        device=device,
        shuffle=True,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        seed=seed,
        device=device,
        shuffle=False,
    )

    set_seed(seed)
    clean_model = SmallFNN().to(device)
    clean_optimizer = optim.SGD(clean_model.parameters(), lr=learning_rate)
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
        sample_count=len(clean_train_dataset),
        output_path=output_dir / "shapley_original_PPIS_NC.csv",
        loss_curve_path=output_dir / "training_metrics_original_PPIS_NC.csv",
        model_path=output_dir / "model_original_PPIS_NC.pt",
        model_name="Original_PPIS_NC",
    )

    set_seed(seed)
    mitigated_model = SmallFNN().to(device)
    mitigated_optimizer = optim.SGD(
        mitigated_model.parameters(),
        lr=learning_rate,
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
        sample_count=len(clean_train_dataset),
        output_path=output_dir / "shapley_ppis_neural_cleanse.csv",
        loss_curve_path=(
            output_dir / "training_metrics_ppis_neural_cleanse.csv"
        ),
        model_path=output_dir / "model_ppis_neural_cleanse.pt",
        model_name="PPIS_Neural_Cleanse",
    )

    summary = {
        "strategy": "PPIS",
        "defense": "Neural Cleanse",
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "mapping_path": str(mapping_path),
        "poisoned_dir": str(poisoned_dir),
        "mitigation_mode": mitigation_mode,
        "filtered_sample_count": len(filtered_indices),
        "filtered_sample_indices": sorted(filtered_indices),
        "clean_shapley_csv": str(clean_shapley_path),
        "mitigated_shapley_csv": str(mitigated_shapley_path),
        "clean_loss_curve_csv": str(clean_curve_path),
        "mitigated_loss_curve_csv": str(mitigated_curve_path),
    }
    write_json(output_dir / "neural_cleanse_ppis_irds_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Run PPIS poisoning mitigation with Neural Cleanse and paired "
            "first-order IRDS training."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "results_Neural_Cleanse_PPIS",
    )
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--poisoned-dir", type=Path, default=None)
    parser.add_argument("--poison-ratio", type=float, default=0.01)
    parser.add_argument("--generate-ppis", action="store_true")
    parser.add_argument("--mitigation-mode", choices=("remove", "restore-clean"), default="remove")
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-learning-rate", type=float, default=0.01)
    parser.add_argument("--nc-steps", type=int, default=200)
    parser.add_argument("--nc-batch-size", type=int, default=256)
    parser.add_argument("--nc-learning-rate", type=float, default=0.1)
    parser.add_argument("--nc-mask-lambda", type=float, default=1e-2)
    parser.add_argument("--mad-threshold", type=float, default=2.0)
    parser.add_argument("--min-target-asr", type=float, default=0.90)
    parser.add_argument("--fallback-min-target-count", type=int, default=0)
    parser.add_argument("--filter-threshold", type=float, default=0.80)
    parser.add_argument(
        "--filter-scope",
        choices=("poisoned", "all"),
        default="poisoned",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping_path = args.mapping_path or (
        args.base_dir / "results1" / "poisoned_mapping.csv"
    )
    poisoned_dir = args.poisoned_dir or default_poisoned_dir(
        args.base_dir,
        args.poison_ratio,
    )
    summary = run_neural_cleanse_ppis_irds(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        mapping_path=mapping_path,
        poisoned_dir=poisoned_dir,
        poison_ratio=args.poison_ratio,
        mitigation_mode=args.mitigation_mode,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        probe_learning_rate=args.probe_learning_rate,
        nc_steps=args.nc_steps,
        nc_batch_size=args.nc_batch_size,
        nc_learning_rate=args.nc_learning_rate,
        nc_mask_lambda=args.nc_mask_lambda,
        mad_threshold=args.mad_threshold,
        min_target_asr=args.min_target_asr,
        fallback_min_target_count=args.fallback_min_target_count,
        filter_threshold=args.filter_threshold,
        filter_scope=args.filter_scope,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_size=args.validation_size,
        seed=args.seed,
        generate_ppis=args.generate_ppis,
    )
    print(f"Clean IRDS CSV: {summary['clean_shapley_csv']}")
    print(f"Mitigated IRDS CSV: {summary['mitigated_shapley_csv']}")


if __name__ == "__main__":
    main()
