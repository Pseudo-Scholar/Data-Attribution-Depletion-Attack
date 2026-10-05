"""Neural Cleanse detection and filtering for the PPIS experiment.

This script does not modify the original PPIS implementation. It consumes:

    <base-dir>/results1/poisoned_mapping.csv
    <base-dir>/save1/random_<ratio>pct_backdoored_images/

The staged defense is:

1. Train a short probe model on the PPIS-poisoned training set.
2. Reverse engineer one continuous trigger for every MNIST target label.
3. Use low mask L1 norms and MAD to identify suspicious target labels.
4. Score PPIS samples against the reverse engineered triggers.
5. Write a filter report consumed by Neural_Cleanse_PPIS_IRDS.py.

The final defended IRDS training is intentionally in a separate script so
that detection/mitigation artifacts can be inspected before expensive
attribution training.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torchvision import datasets

from Neural_Cleanse_PPIS_Common import (
    IndexedMNIST,
    PPISPoisonedMNIST,
    SmallFNN,
    TriggerCandidate,
    filter_ppis_samples,
    load_ppis_mapping,
    mad_detect_candidates,
    make_loader,
    make_reference_loader,
    mnist_transform,
    reverse_engineer_all_targets,
    save_trigger_candidates,
    set_seed,
    train_probe_model,
    write_json,
)


def default_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def default_poisoned_dir(base_dir: Path, poison_ratio: float) -> Path:
    return (
        Path(base_dir)
        / "save1"
        / f"random_{int(poison_ratio * 100)}pct_backdoored_images"
    )


def ensure_ppis_artifacts(
    base_dir: Path,
    poison_ratio: float,
    mapping_path: Path,
    poisoned_dir: Path,
    generate_ppis: bool,
) -> None:
    if mapping_path.is_file() and poisoned_dir.is_dir():
        return
    if not generate_ppis:
        raise FileNotFoundError(
            "PPIS artifacts are missing. Run "
            "Backdoor_Poisoning_Strategy_PPIS.py --generation-only first, "
            "or pass --generate-ppis to this script."
        )
    original_script = Path(__file__).with_name(
        "Backdoor_Poisoning_Strategy_PPIS.py"
    )
    command = [
        sys.executable,
        str(original_script),
        "--base-dir",
        str(base_dir),
        "--ratio",
        str(poison_ratio),
        "--generation-only",
    ]
    print("Generating PPIS artifacts with the preserved original script...")
    subprocess.run(command, check=True)

    if not mapping_path.is_file() or not poisoned_dir.is_dir():
        raise RuntimeError(
            "The original PPIS generator completed but the expected mapping "
            "or poisoned-image directory is still missing."
        )


def choose_fallback_candidates(
    candidates: Sequence[TriggerCandidate],
    selected: Sequence[TriggerCandidate],
    fallback_count: int,
) -> List[TriggerCandidate]:
    if selected or fallback_count <= 0:
        return list(selected)
    ordered = sorted(candidates, key=lambda item: item.mask_l1)
    return ordered[: min(fallback_count, len(ordered))]


def run_neural_cleanse_ppis_detection(
    base_dir: Path,
    output_dir: Path,
    mapping_path: Path,
    poisoned_dir: Path,
    poison_ratio: float,
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
    seed: int,
    generate_ppis: bool = False,
) -> Dict[str, object]:
    """Run the complete Neural Cleanse detection/filtering stage."""

    set_seed(seed)
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    mapping_path = Path(mapping_path)
    poisoned_dir = Path(poisoned_dir)
    ensure_ppis_artifacts(
        base_dir=base_dir,
        poison_ratio=poison_ratio,
        mapping_path=mapping_path,
        poisoned_dir=poisoned_dir,
        generate_ppis=generate_ppis,
    )

    transform = mnist_transform()
    data_dir = base_dir / "data"
    clean_train_dataset = IndexedMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    clean_reference_dataset = datasets.MNIST(
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Cleanse + PPIS device: {device}")
    print(f"PPIS records: {len(poison_records)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    probe_loader = make_loader(
        poisoned_dataset,
        batch_size=probe_batch_size,
        seed=seed,
        device=device,
        shuffle=True,
    )
    probe_model = SmallFNN().to(device)
    train_probe_model(
        model=probe_model,
        train_loader=probe_loader,
        epochs=probe_epochs,
        learning_rate=probe_learning_rate,
        device=device,
        output_csv=output_dir / "neural_cleanse_probe_loss.csv",
    )
    torch.save(probe_model.state_dict(), output_dir / "neural_cleanse_probe_model.pt")

    reference_loader = make_reference_loader(
        clean_reference_dataset,
        batch_size=nc_batch_size,
        device=device,
    )
    candidates = reverse_engineer_all_targets(
        model=probe_model,
        reference_loader=reference_loader,
        device=device,
        num_classes=10,
        steps=nc_steps,
        learning_rate=nc_learning_rate,
        mask_lambda=nc_mask_lambda,
    )
    selected, detection_stats = mad_detect_candidates(
        candidates=candidates,
        mad_threshold=mad_threshold,
        min_attack_success_rate=min_target_asr,
    )
    detection_stats["suspicious_target_labels"] = [
        candidate.target_label for candidate in selected
    ]
    detection_stats["fallback_min_target_count"] = fallback_min_target_count
    save_trigger_candidates(
        candidates=candidates,
        output_dir=output_dir,
        detection_stats=detection_stats,
    )

    selected_for_filter = choose_fallback_candidates(
        candidates=candidates,
        selected=selected,
        fallback_count=fallback_min_target_count,
    )
    filter_reason = "mad-low-mask-outlier"
    if not selected and selected_for_filter:
        filter_reason = "fallback-smallest-mask"
    detection_stats["filter_target_labels"] = [
        candidate.target_label for candidate in selected_for_filter
    ]
    detection_stats["filter_selection_reason"] = filter_reason
    write_json(output_dir / "neural_cleanse_detection.json", detection_stats)

    filtered_indices = filter_ppis_samples(
        model=probe_model,
        clean_dataset=clean_train_dataset,
        poisoned_dataset=poisoned_dataset,
        poison_records=poison_records,
        suspicious_candidates=selected_for_filter,
        device=device,
        output_csv=output_dir / "neural_cleanse_filter_mapping.csv",
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
        output_dir / "neural_cleanse_filtered_samples.json",
        filtered_payload,
    )
    print(
        f"Neural Cleanse selected targets: "
        f"{filtered_payload['filter_target_labels']}"
    )
    print(f"Filtered sample count: {len(filtered_indices)}")
    return {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "mapping_path": str(mapping_path),
        "poisoned_dir": str(poisoned_dir),
        "filtered_indices": sorted(filtered_indices),
        "filter_target_labels": filtered_payload["filter_target_labels"],
    }


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description="Detect and filter PPIS samples with Neural Cleanse."
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
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-learning-rate", type=float, default=0.01)
    parser.add_argument("--nc-steps", type=int, default=200)
    parser.add_argument("--nc-batch-size", type=int, default=256)
    parser.add_argument("--nc-learning-rate", type=float, default=0.1)
    parser.add_argument("--nc-mask-lambda", type=float, default=1e-2)
    parser.add_argument("--mad-threshold", type=float, default=2.0)
    parser.add_argument("--min-target-asr", type=float, default=0.90)
    parser.add_argument(
        "--fallback-min-target-count",
        type=int,
        default=0,
        help="Optional number of smallest-mask targets used if MAD finds none.",
    )
    parser.add_argument("--filter-threshold", type=float, default=0.80)
    parser.add_argument(
        "--filter-scope",
        choices=("poisoned", "all"),
        default="poisoned",
        help="Scan only PPIS records by default; use all for model-wide scanning.",
    )
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
    run_neural_cleanse_ppis_detection(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        mapping_path=mapping_path,
        poisoned_dir=poisoned_dir,
        poison_ratio=args.poison_ratio,
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
        seed=args.seed,
        generate_ppis=args.generate_ppis,
    )


if __name__ == "__main__":
    main()
