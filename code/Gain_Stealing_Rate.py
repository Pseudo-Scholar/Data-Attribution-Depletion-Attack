"""Gain-Stealing Rate (GSR) for Data Attribution Depletion Attacks.

The malicious client's gain is computed from the paper's Formula 6:

    g_i = 0, if s_i >= 0
    g_i = C * min(0, s_i) / sum_j min(0, s_j) * G, otherwise.

Because the same ``C`` and ``G`` are used in both scenarios, this script
uses a unit reward budget. GSR is:

    sum(g_i^poison for malicious poison samples)
    ------------------------------------------------
    sum(g_i^clean  for malicious clean samples)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from Data_Attribution_Metrics_Common import (
    add_metric_input_arguments,
    load_sample_pairs,
    positive_contribution_rewards,
    read_shapley_csv,
    safe_ratio,
    sum_rewards,
    write_json_result,
)


def compute_gain_stealing_rate(
    clean_shapley_path: Path,
    poisoned_shapley_path: Path,
    mapping_path: Path,
    pair_mode: str = "auto",
    append_offset: Optional[int] = None,
    reward_rule: str = "formula6",
    budget: float = 1.0,
    tullock_exponent: float = 2.0,
) -> Dict[str, object]:
    """Compute malicious-client gain ratio for clean and poisoned uploads."""

    clean_scores = read_shapley_csv(clean_shapley_path)
    poisoned_scores = read_shapley_csv(poisoned_shapley_path)
    pairs = load_sample_pairs(
        mapping_path=mapping_path,
        clean_scores=clean_scores,
        poison_scores=poisoned_scores,
        pair_mode=pair_mode,
        append_offset=append_offset,
    )

    clean_rewards = positive_contribution_rewards(
        clean_scores,
        rule=reward_rule,
        budget=budget,
        tullock_exponent=tullock_exponent,
    )
    poisoned_rewards = positive_contribution_rewards(
        poisoned_scores,
        rule=reward_rule,
        budget=budget,
        tullock_exponent=tullock_exponent,
    )
    malicious_clean_indices = [pair.clean_index for pair in pairs]
    malicious_poison_indices = [pair.poison_index for pair in pairs]

    clean_gain = sum_rewards(clean_rewards, malicious_clean_indices)
    poisoned_gain = sum_rewards(poisoned_rewards, malicious_poison_indices)
    gsr = safe_ratio(
        numerator=poisoned_gain,
        denominator=clean_gain,
        name="GSR",
    )

    return {
        "metric": "GSR",
        "gain_stealing_rate": gsr,
        "clean_malicious_gain": clean_gain,
        "poisoned_malicious_gain": poisoned_gain,
        "malicious_clean_sample_count": len(malicious_clean_indices),
        "malicious_poison_sample_count": len(malicious_poison_indices),
        "clean_negative_contributor_count": sum(
            score < 0.0 for score in clean_scores.values()
        ),
        "poisoned_negative_contributor_count": sum(
            score < 0.0 for score in poisoned_scores.values()
        ),
        "reward_rule": reward_rule,
        "budget": budget,
        "tullock_exponent": tullock_exponent,
        "clean_shapley": str(clean_shapley_path),
        "poisoned_shapley": str(poisoned_shapley_path),
        "mapping": str(mapping_path),
        "formula": (
            "sum(poisoned malicious rewards) / "
            "sum(clean malicious rewards)"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Gain-Stealing Rate (GSR)."
    )
    add_metric_input_arguments(parser, mapping_required=True)
    parser.add_argument(
        "--reward-rule",
        choices=("formula6", "tullock"),
        default="formula6",
        help="Formula 6 by default; optionally evaluate the Tullock variant.",
    )
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument(
        "--tullock-exponent",
        type=float,
        default=2.0,
        help="Exponent r for the Tullock reward rule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_gain_stealing_rate(
        clean_shapley_path=args.clean_shapley,
        poisoned_shapley_path=args.poisoned_shapley,
        mapping_path=args.mapping,
        pair_mode=args.pair_mode,
        append_offset=args.append_offset,
        reward_rule=args.reward_rule,
        budget=args.budget,
        tullock_exponent=args.tullock_exponent,
    )
    output_path = write_json_result(
        result=result,
        output_path=args.output,
        default_directory=args.clean_shapley.parent,
        default_name="gsr_result.json",
    )
    print(f"GSR = {float(result['gain_stealing_rate']):.6f}")
    print(
        f"Malicious gain: clean={result['clean_malicious_gain']:.6f}, "
        f"poisoned={result['poisoned_malicious_gain']:.6f}"
    )
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
