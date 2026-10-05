"""Neural Cleanse detection and filtering for APS.

This script keeps the original APS generation and victim-side IRDS scripts
unchanged. It consumes:

    <aps-root>/aps_mapping.csv
    <aps-root>/aps/*.png

The defense stage trains a short probe model on APS-poisoned data, reverse
engineers one candidate trigger per CIFAR-10 target label, applies MAD
low-mask-norm detection, and writes a sample-level filter report.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Subset
from torchvision import datasets

from APS_Common import IndexedCIFAR10, ModifiedCIFARCNN, make_cifar10_transform
from Neural_Cleanse_APS_Common import (
    build_validation_batch,
    filter_aps_samples,
    load_aps_normalized_images,
    load_aps_records,
    make_loader,
    make_plain_loader,
    mad_detect_candidates,
    reverse_engineer_all_targets,
    save_trigger_candidates,
    set_seed,
    train_probe_model,
    write_json,
    PoisonedCIFAR10Dataset,
)


def default_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def default_aps_root(base_dir: Path) -> Path:
    return base_dir / "aps_data"


def ensure_aps_artifacts(
    base_dir: Path,
    aps_root: Path,
    poison_ratio: float,
    seed: int,
    epsilon: float,
    norm: str,
    steps: int,
    eta: float,
    validation_size: int,
    trigger_mode: str,
    trigger_amplitude: float,
    batch_size: int,
    model_path: Optional[Path],
    generate_aps: bool,
) -> None:
    mapping_path = aps_root / "aps_mapping.csv"
    if mapping_path.is_file() and (aps_root / "aps").is_dir():
        return
    if not generate_aps:
        raise FileNotFoundError(
            f"APS mapping not found: {mapping_path}. Run "
            "Generate_Perturbed_Images_APS.py first, or pass --generate-aps."
        )

    generator = Path(__file__).with_name("Generate_Perturbed_Images_APS.py")
    command = [
        sys.executable,
        str(generator),
        "--base-dir",
        str(base_dir),
        "--output-root",
        str(aps_root),
        "--poison-ratio",
        str(poison_ratio),
        "--seed",
        str(seed),
        "--epsilon",
        str(epsilon),
        "--norm",
        norm,
        "--steps",
        str(steps),
        "--eta",
        str(eta),
        "--validation-size",
        str(validation_size),
        "--trigger-mode",
        trigger_mode,
        "--trigger-amplitude",
        str(trigger_amplitude),
        "--batch-size",
        str(batch_size),
    ]
    if model_path is not None:
        command.extend(["--model-path", str(model_path)])
    print("Generating APS artifacts with the preserved original generator...")
    subprocess.run(command, check=True)
    if not mapping_path.is_file() or not (aps_root / "aps").is_dir():
        raise RuntimeError(
            "APS generation completed but the expected mapping or aps directory "
            "is still missing."
        )


def run_neural_cleanse_aps_detection(
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
    validation_size: int = 128,
    probe_max_samples: Optional[int] = None,
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
    """Run the APS Neural Cleanse detection/filtering stage."""

    set_seed(seed)
    base_dir = Path(base_dir)
    aps_root = Path(aps_root)
    output_dir = Path(output_dir)
    ensure_aps_artifacts(
        base_dir=base_dir,
        aps_root=aps_root,
        poison_ratio=poison_ratio,
        seed=seed,
        epsilon=epsilon,
        norm=norm,
        steps=aps_generation_steps,
        eta=eta,
        validation_size=validation_size,
        trigger_mode=trigger_mode,
        trigger_amplitude=trigger_amplitude,
        batch_size=generation_batch_size,
        model_path=model_path,
        generate_aps=generate_aps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_root)
    normalized_transform = make_cifar10_transform(normalize=True)
    clean_dataset = IndexedCIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=normalized_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
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
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_indices = list(range(len(clean_dataset)))
    if probe_max_samples is not None:
        if probe_max_samples <= 0:
            raise ValueError("probe_max_samples must be positive.")
        probe_size = min(probe_max_samples, len(clean_dataset))
        probe_indices = sorted(
            random.Random(seed).sample(probe_indices, probe_size)
        )
    print(f"Neural Cleanse + APS device: {device}")
    print(
        f"CIFAR-10 samples: {len(clean_dataset)} | "
        f"APS records: {len(poison_records)} | "
        f"Probe samples: {len(probe_indices)}"
    )

    probe_loader = make_loader(
        dataset=poisoned_dataset,
        indices=probe_indices,
        batch_size=probe_batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
        shuffle=True,
    )
    probe_model = ModifiedCIFARCNN().to(device)
    train_probe_model(
        model=probe_model,
        train_loader=probe_loader,
        epochs=probe_epochs,
        learning_rate=probe_learning_rate,
        momentum=probe_momentum,
        device=device,
        output_csv=output_dir / "neural_cleanse_aps_probe_loss.csv",
    )
    torch.save(
        probe_model.state_dict(),
        output_dir / "neural_cleanse_aps_probe_model.pt",
    )

    reference_images, reference_labels = build_validation_batch(
        test_dataset=test_dataset,
        validation_size=nc_reference_size,
        seed=seed + 1,
        device=device,
    )
    reference_dataset = Subset(
        test_dataset,
        list(range(min(nc_reference_size, len(test_dataset)))),
    )
    reference_loader = make_plain_loader(
        dataset=reference_dataset,
        batch_size=nc_batch_size,
        device=device,
        num_workers=num_workers,
        shuffle=False,
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
    if not selected and fallback_min_target_count > 0:
        selected_for_filter = sorted(
            candidates,
            key=lambda candidate: candidate.mask_l1,
        )[: min(fallback_min_target_count, len(candidates))]
        selection_reason = "fallback-smallest-mask"
    else:
        selected_for_filter = list(selected)
        selection_reason = "mad-low-mask-outlier" if selected else "none"
    detection_stats.update(
        {
            "candidate_count": len(candidates),
            "suspicious_target_labels": [
                candidate.target_label for candidate in selected
            ],
            "filter_target_labels": [
                candidate.target_label for candidate in selected_for_filter
            ],
            "filter_selection_reason": selection_reason,
            "reference_size": min(nc_reference_size, len(test_dataset)),
        }
    )
    save_trigger_candidates(
        candidates=candidates,
        output_dir=output_dir,
        detection_stats=detection_stats,
    )
    write_json(
        output_dir / "neural_cleanse_aps_detection.json",
        detection_stats,
    )

    filtered_indices = filter_aps_samples(
        model=probe_model,
        clean_dataset=clean_dataset,
        poisoned_dataset=poisoned_dataset,
        poison_records=poison_records,
        suspicious_candidates=selected_for_filter,
        device=device,
        output_csv=output_dir / "neural_cleanse_aps_filter_mapping.csv",
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
        output_dir / "neural_cleanse_aps_filtered_samples.json",
        filtered_payload,
    )
    print(
        f"Neural Cleanse APS targets: "
        f"{filtered_payload['filter_target_labels']}"
    )
    print(f"Filtered APS samples: {len(filtered_indices)}")
    return {
        "base_dir": str(base_dir),
        "data_root": str(data_root),
        "aps_root": str(aps_root),
        "output_dir": str(output_dir),
        "filtered_indices": sorted(filtered_indices),
        "filter_target_labels": filtered_payload["filter_target_labels"],
        "validation_images": reference_images,
        "validation_labels": reference_labels,
    }


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description="Detect and filter APS samples with Neural Cleanse."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
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
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.aps_root / "neural_cleanse_results"
    )
    result = run_neural_cleanse_aps_detection(
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
        validation_size=args.validation_size,
        probe_max_samples=args.probe_max_samples,
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
    print(f"Filtered APS sample count: {len(result['filtered_indices'])}")


if __name__ == "__main__":
    main()
