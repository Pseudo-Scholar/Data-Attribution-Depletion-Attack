"""Loss-Amplifying Rate (LAR) for Data Attribution Depletion Attacks.

The server removes samples whose estimated In-Run Data Shapley value is
greater than zero, then retrains for the same number of epochs. Given the
two post-filtering retraining loss curves:

    LAR = final_loss_poison / final_loss_clean

The default ``--aggregation final`` is the paper's same-epoch definition.
``mean`` and ``auc`` are included as optional diagnostics for studying the
whole curve, while the primary reported metric remains the final-loss ratio.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from Data_Attribution_Metrics_Common import (
    add_metric_input_arguments,
    aggregate_loss_curve,
    ensure_same_epochs,
    read_loss_curve,
    read_shapley_csv,
    safe_ratio,
    write_json_result,
)


def compute_loss_amplifying_rate(
    clean_shapley_path: Path,
    poisoned_shapley_path: Path,
    clean_loss_curve_path: Path,
    poisoned_loss_curve_path: Path,
    loss_column: str = "train_loss",
    aggregation: str = "final",
    shapley_threshold: float = 0.0,
) -> Dict[str, object]:
    """Compute the loss ratio after Shapley-guided data removal."""

    clean_scores = read_shapley_csv(clean_shapley_path)
    poisoned_scores = read_shapley_csv(poisoned_shapley_path)
    clean_curve = read_loss_curve(
        clean_loss_curve_path,
        loss_column=loss_column,
    )
    poisoned_curve = read_loss_curve(
        poisoned_loss_curve_path,
        loss_column=loss_column,
    )
    ensure_same_epochs(clean_curve, poisoned_curve)

    clean_removed = {
        index
        for index, score in clean_scores.items()
        if score > shapley_threshold
    }
    poisoned_removed = {
        index
        for index, score in poisoned_scores.items()
        if score > shapley_threshold
    }

    clean_loss = aggregate_loss_curve(clean_curve, aggregation=aggregation)
    poisoned_loss = aggregate_loss_curve(
        poisoned_curve,
        aggregation=aggregation,
    )
    lar = safe_ratio(
        numerator=poisoned_loss,
        denominator=clean_loss,
        name="LAR",
    )

    return {
        "metric": "LAR",
        "loss_amplifying_rate": lar,
        "clean_retrained_loss": clean_loss,
        "poisoned_retrained_loss": poisoned_loss,
        "loss_difference_poisoned_minus_clean": poisoned_loss - clean_loss,
        "aggregation": aggregation,
        "loss_column": loss_column,
        "shapley_removal_threshold": shapley_threshold,
        "clean_removed_count": len(clean_removed),
        "poisoned_removed_count": len(poisoned_removed),
        "clean_remaining_count": len(clean_scores) - len(clean_removed),
        "poisoned_remaining_count": len(poisoned_scores) - len(poisoned_removed),
        "epoch_count": len(clean_curve.epochs),
        "first_epoch": clean_curve.epochs[0],
        "last_epoch": clean_curve.epochs[-1],
        "clean_shapley": str(clean_shapley_path),
        "poisoned_shapley": str(poisoned_shapley_path),
        "clean_loss_curve": str(clean_loss_curve_path),
        "poisoned_loss_curve": str(poisoned_loss_curve_path),
        "formula": "aggregated_poisoned_retrained_loss / aggregated_clean_retrained_loss",
        "curve_requirement": (
            "Both curves must be measured after removing samples with "
            "Shapley values greater than the supplied threshold."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Loss-Amplifying Rate (LAR)."
    )
    add_metric_input_arguments(parser, mapping_required=False)
    parser.add_argument(
        "--clean-loss-curve",
        type=Path,
        required=True,
        help="Clean-scenario post-filtering retraining loss curve CSV/JSON.",
    )
    parser.add_argument(
        "--poisoned-loss-curve",
        type=Path,
        required=True,
        help="Poisoned-scenario post-filtering retraining loss curve CSV/JSON.",
    )
    parser.add_argument(
        "--loss-column",
        type=str,
        default="train_loss",
        help="Loss column to read; train_loss is the default.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("final", "mean", "auc"),
        default="final",
        help="final is the paper metric; mean/auc are optional diagnostics.",
    )
    parser.add_argument(
        "--shapley-threshold",
        type=float,
        default=0.0,
        help="Remove samples with Shapley value strictly greater than this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_loss_amplifying_rate(
        clean_shapley_path=args.clean_shapley,
        poisoned_shapley_path=args.poisoned_shapley,
        clean_loss_curve_path=args.clean_loss_curve,
        poisoned_loss_curve_path=args.poisoned_loss_curve,
        loss_column=args.loss_column,
        aggregation=args.aggregation,
        shapley_threshold=args.shapley_threshold,
    )
    output_path = write_json_result(
        result=result,
        output_path=args.output,
        default_directory=args.clean_shapley.parent,
        default_name="lar_result.json",
    )
    print(f"LAR = {float(result['loss_amplifying_rate']):.6f}")
    print(
        f"Retrained loss: clean={result['clean_retrained_loss']:.6f}, "
        f"poisoned={result['poisoned_retrained_loss']:.6f}"
    )
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
