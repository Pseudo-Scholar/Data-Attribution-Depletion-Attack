"""Neural Cleanse detection and filtering for the HTBA ImageNet experiment.

The original HTBA/HTBM generation code is preserved.  This script consumes
``htbm_mapping.csv`` and the generated poison images, then:

1. trains a short ViT-B/16 probe on the HTBA-poisoned data;
2. reverse engineers continuous Neural Cleanse triggers;
3. detects low-mask-norm target labels with MAD;
4. filters mapped HTBA samples conservatively;
5. writes artifacts consumed by ``Neural_Cleanse_HTBA_IRDS.py``.

HTBA is a hidden-trigger attack: the final poison image does not explicitly
contain the trigger patch.  Consequently, a successful Neural Cleanse
detection is not guaranteed.  With the default fallback of zero, no samples
are removed when no statistically supported target-label outlier is found.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from Hidden_Trigger_Backdoor_Strategy_HTBM_Common import (
    build_vit_b16,
    set_seed,
)
from Neural_Cleanse_HTBA_Common import (
    build_htba_data_bundle,
    filter_htba_samples,
    make_reference_loader,
    make_training_loader,
    mad_detect_candidates,
    reverse_engineer_targets,
    save_trigger_candidates,
    train_probe_model,
    configure_trainable_scope,
    write_json,
)


def default_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def default_train_root(base_dir: Path) -> Path:
    return Path(base_dir) / "data" / "ImageNet"


def ensure_htba_artifacts(
    train_root: Path,
    poison_root: Path,
    source_class: Optional[str],
    target_class: Optional[str],
    trigger_path: Optional[Path],
    trigger_size: Sequence[int],
    poison_count: int,
    generation_iterations: int,
    epsilon: float,
    image_size: int,
    seed: int,
    device: Optional[str],
    surrogate_weights: Optional[Path],
    surrogate_pretrained: bool,
    generate_htba: bool,
) -> None:
    mapping_path = Path(poison_root) / "htbm_mapping.csv"
    if mapping_path.is_file() and (Path(poison_root) / "poisoned").is_dir():
        return
    if not generate_htba:
        raise FileNotFoundError(
            "HTBA artifacts are missing. Run "
            "Hidden_Trigger_Backdoor_Strategy_HTBA.py first, or pass "
            "--generate-htba."
        )

    generator = Path(__file__).with_name(
        "Hidden_Trigger_Backdoor_Strategy_HTBA.py"
    )
    command = [
        sys.executable,
        str(generator),
        "--train-root",
        str(train_root),
        "--output-root",
        str(poison_root),
        "--trigger-size",
        str(int(trigger_size[0])),
        str(int(trigger_size[1])),
        "--poison-count",
        str(poison_count),
        "--iterations",
        str(generation_iterations),
        "--epsilon",
        str(epsilon),
        "--image-size",
        str(image_size),
        "--seed",
        str(seed),
    ]
    if source_class is not None:
        command.extend(["--source-class", source_class])
    if target_class is not None:
        command.extend(["--target-class", target_class])
    if trigger_path is not None:
        command.extend(["--trigger-path", str(trigger_path)])
    if device is not None:
        command.extend(["--device", device])
    if surrogate_weights is not None:
        command.extend(["--surrogate-weights", str(surrogate_weights)])
    if not surrogate_pretrained:
        command.append("--no-pretrained")

    print("Generating HTBA artifacts with the preserved original script...")
    subprocess.run(command, check=True)
    if not mapping_path.is_file() or not (Path(poison_root) / "poisoned").is_dir():
        raise RuntimeError(
            "The HTBA generator completed but htbm_mapping.csv or poisoned/ "
            "is missing."
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
        raise ValueError("No target labels were selected for Neural Cleanse.")
    if max_scan_labels is None or max_scan_labels >= len(labels):
        return labels
    if max_scan_labels <= 0:
        raise ValueError("max_scan_labels must be positive.")

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


def run_neural_cleanse_htba_detection(
    train_root: Path,
    poison_root: Path,
    output_dir: Path,
    mapping_path: Optional[Path] = None,
    val_root: Optional[Path] = None,
    image_size: int = 224,
    validation_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    integration_mode: str = "append",
    augmentation: bool = True,
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
    pretrained: bool = True,
    weights_path: Optional[Path] = None,
    device: Optional[str] = None,
    generate_htba: bool = False,
    source_class: Optional[str] = None,
    target_class: Optional[str] = None,
    trigger_path: Optional[Path] = None,
    trigger_size: Sequence[int] = (32, 32),
    poison_count: int = 8,
    generation_iterations: int = 100,
    epsilon: float = 16.0 / 255.0,
) -> Dict[str, object]:
    """Run the complete HTBA Neural Cleanse detection stage."""

    set_seed(seed)
    train_root = Path(train_root)
    poison_root = Path(poison_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_htba_artifacts(
        train_root=train_root,
        poison_root=poison_root,
        source_class=source_class,
        target_class=target_class,
        trigger_path=trigger_path,
        trigger_size=trigger_size,
        poison_count=poison_count,
        generation_iterations=generation_iterations,
        epsilon=epsilon,
        image_size=image_size,
        seed=seed,
        device=device,
        surrogate_weights=weights_path,
        surrogate_pretrained=pretrained,
        generate_htba=generate_htba,
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

    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"Neural Cleanse + HTBA device: {selected_device}; "
        f"classes={len(bundle.class_names)}, "
        f"training_positions={len(bundle.train_positions)}, "
        f"poisons={len(bundle.poison_records)}, "
        f"integration={integration_mode}"
    )

    probe_loader = make_training_loader(
        dataset=bundle.poisoned_dataset,
        positions=bundle.train_positions,
        batch_size=probe_batch_size,
        seed=seed,
        device=selected_device,
        num_workers=num_workers,
    )
    probe_model = build_vit_b16(
        num_classes=len(bundle.class_names),
        pretrained=pretrained,
        weights_path=weights_path,
    )
    configure_trainable_scope(probe_model, probe_trainable_scope)
    probe_model = probe_model.to(selected_device)
    train_probe_model(
        model=probe_model,
        train_loader=probe_loader,
        epochs=probe_epochs,
        learning_rate=probe_learning_rate,
        weight_decay=probe_weight_decay,
        device=selected_device,
        output_csv=output_dir / "neural_cleanse_htba_probe_loss.csv",
    )
    torch.save(
        probe_model.state_dict(),
        output_dir / "neural_cleanse_htba_probe_model.pt",
    )

    reference_loader = make_reference_loader(
        dataset=bundle.validation_dataset,
        positions=bundle.validation_positions,
        batch_size=nc_batch_size,
        num_workers=num_workers,
    )
    mapped_targets = [
        int(record.target_label)
        for record in bundle.poison_records.values()
    ]
    scan_labels = select_scan_labels(
        num_classes=len(bundle.class_names),
        mapped_target_labels=mapped_targets,
        scan_mode=nc_labels,
        max_scan_labels=max_scan_labels,
        include_mapped_target_labels=include_mapped_target_labels,
        seed=seed + 2,
    )
    print(f"Neural Cleanse HTBA target labels to scan: {len(scan_labels)}")
    candidates = reverse_engineer_targets(
        model=probe_model,
        reference_loader=reference_loader,
        target_labels=scan_labels,
        device=selected_device,
        steps=nc_steps,
        learning_rate=nc_learning_rate,
        mask_lambda=nc_mask_lambda,
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
            "strategy": "HTBA",
            "defense": "Neural Cleanse",
            "train_root": str(train_root),
            "poison_root": str(poison_root),
            "integration_mode": integration_mode,
            "class_count": len(bundle.class_names),
            "scan_label_count": len(scan_labels),
            "scan_labels": scan_labels,
            "suspicious_target_labels": [
                candidate.target_label for candidate in selected
            ],
            "filter_target_labels": [
                candidate.target_label for candidate in selected_for_filter
            ],
            "fallback_min_target_count": fallback_min_target_count,
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
    save_trigger_candidates(candidates, output_dir)
    write_json(
        output_dir / "neural_cleanse_htba_detection.json",
        detection_stats,
    )

    filtered_indices = filter_htba_samples(
        model=probe_model,
        poisoned_dataset=bundle.poisoned_dataset,
        poison_records=bundle.poison_records,
        integration_mode=integration_mode,
        base_sample_count=bundle.base_sample_count,
        suspicious_candidates=selected_for_filter,
        device=selected_device,
        output_csv=output_dir / "neural_cleanse_htba_filter_mapping.csv",
        filter_threshold=filter_threshold,
        filter_scope=filter_scope,
    )
    filtered_payload = {
        "strategy": "HTBA",
        "defense": "Neural Cleanse",
        "filtered_sample_count": len(filtered_indices),
        "filtered_sample_indices": sorted(filtered_indices),
        "filter_scope": filter_scope,
        "filter_threshold": filter_threshold,
        "filter_target_labels": [
            candidate.target_label for candidate in selected_for_filter
        ],
        "hidden_trigger_warning": (
            "HTBA poison images do not explicitly contain the visible "
            "generation trigger; zero filtered samples is a valid outcome."
        ),
    }
    write_json(
        output_dir / "neural_cleanse_htba_filtered_samples.json",
        filtered_payload,
    )
    print(
        f"Neural Cleanse HTBA target labels: "
        f"{filtered_payload['filter_target_labels']}"
    )
    print(f"Filtered HTBA samples: {len(filtered_indices)}")

    return {
        "train_root": str(train_root),
        "poison_root": str(poison_root),
        "output_dir": str(output_dir),
        "filtered_indices": sorted(filtered_indices),
        "filter_target_labels": filtered_payload["filter_target_labels"],
        "selected_positions": bundle.selected_positions,
        "train_positions": bundle.train_positions,
        "validation_positions": bundle.validation_positions,
        "class_count": len(bundle.class_names),
        "global_sample_count": bundle.global_sample_count,
    }


def parse_args() -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description="Detect and filter HTBA samples with Neural Cleanse."
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=default_train_root(base_dir),
        help="ImageNet-100 root or its train split.",
    )
    parser.add_argument(
        "--poison-root",
        type=Path,
        required=True,
        help="Directory containing htbm_mapping.csv and poisoned/.",
    )
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
    parser.add_argument("--max-train-samples", type=int, default=None)
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
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.poison_root / "neural_cleanse_results"
    )
    result = run_neural_cleanse_htba_detection(
        train_root=args.train_root,
        poison_root=args.poison_root,
        output_dir=output_dir,
        mapping_path=args.mapping_path,
        val_root=args.val_root,
        image_size=args.image_size,
        validation_size=args.validation_size,
        seed=args.seed,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        integration_mode=args.integration_mode,
        augmentation=args.augmentation,
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
        pretrained=args.pretrained,
        weights_path=args.weights_path,
        device=args.device,
        generate_htba=args.generate_htba,
        source_class=args.source_class,
        target_class=args.target_class,
        trigger_path=args.trigger_path,
        trigger_size=args.trigger_size,
        poison_count=args.poison_count,
        generation_iterations=args.generation_iterations,
        epsilon=args.epsilon,
    )
    print(f"Filtered HTBA sample count: {len(result['filtered_indices'])}")


if __name__ == "__main__":
    main()
