"""Gain-Volatility Rate (GVR) for Data Attribution Depletion Attacks.

GVR compares the total gains of all normal clients under the poisoned and
clean upload scenarios:

    GVR = normal_gain_poison / normal_gain_clean

The reward assigned to each sample follows Formula 6 by default. Malicious
sample membership is obtained from the clean/poison mapping; every remaining
sample in each Shapley CSV is treated as a normal-client sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Set

from Data_Attribution_Metrics_Common import (
    add_metric_input_arguments,
    load_sample_pairs,
    positive_contribution_rewards,
    read_shapley_csv,
    safe_ratio,
    sum_rewards,
    write_json_result,
)


def compute_gain_volatility_rate(
    clean_shapley_path: Path,
    poisoned_shapley_path: Path,
    mapping_path: Path,
    pair_mode: str = "auto",
    append_offset: Optional[int] = None,
    reward_rule: str = "formula6",
    budget: float = 1.0,
    tullock_exponent: float = 2.0,
) -> Dict[str, object]:
    """Compute the normal-client gain ratio."""

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

    malicious_clean: Set[int] = {
        pair.clean_index for pair in pairs
    }
    malicious_poison: Set[int] = {
        pair.poison_index for pair in pairs
    }
    normal_clean = set(clean_scores).difference(malicious_clean)
    normal_poison = set(poisoned_scores).difference(malicious_poison)

    clean_gain = sum_rewards(clean_rewards, normal_clean)
    poisoned_gain = sum_rewards(poisoned_rewards, normal_poison)
    gvr = safe_ratio(
        numerator=poisoned_gain,
        denominator=clean_gain,
        name="GVR",
    )

    return {
        "metric": "GVR",
        "gain_volatility_rate": gvr,
        "clean_normal_gain": clean_gain,
        "poisoned_normal_gain": poisoned_gain,
        "normal_clean_sample_count": len(normal_clean),
        "normal_poison_sample_count": len(normal_poison),
        "malicious_clean_sample_count": len(malicious_clean),
        "malicious_poison_sample_count": len(malicious_poison),
        "reward_rule": reward_rule,
        "budget": budget,
        "tullock_exponent": tullock_exponent,
        "clean_shapley": str(clean_shapley_path),
        "poisoned_shapley": str(poisoned_shapley_path),
        "mapping": str(mapping_path),
        "formula": (
            "sum(poisoned normal-client rewards) / "
            "sum(clean normal-client rewards)"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Gain-Volatility Rate (GVR)."
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
    result = compute_gain_volatility_rate(
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
        default_name="gvr_result.json",
    )
    print(f"GVR = {float(result['gain_volatility_rate']):.6f}")
    print(
        f"Normal-client gain: clean={result['clean_normal_gain']:.6f}, "
        f"poisoned={result['poisoned_normal_gain']:.6f}"
    )
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
