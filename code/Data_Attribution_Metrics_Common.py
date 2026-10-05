"""Shared input and calculation helpers for DADA evaluation metrics.

The metric scripts deliberately accept ordinary CSV files so they can be
used with PPIS, APIS, APS, and HTBA/HTBM experiments:

* Shapley CSV: ``sample_index`` or ``index`` plus ``shapley_value``.
* Mapping CSV: strategy-specific poisoned/clean sample correspondence.
* Loss curve CSV: ``epoch`` plus ``train_loss`` or ``test_loss``.

The paper's Formula 6 allocates rewards only to samples with negative
In-Run Data Shapley values:

    g_i = C * min(0, s_i) / sum_j min(0, s_j) * G,  if s_i < 0
    g_i = 0,                                           otherwise.

For ratios, the constants ``C`` and ``G`` cancel, so the implementation uses
``budget=1`` by default.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


INDEX_COLUMNS = (
    "sample_index",
    "index",
    "original_index",
    "id",
)
VALUE_COLUMNS = (
    "shapley_value",
    "shapley",
    "value",
    "score",
)


@dataclass(frozen=True)
class ShapleyPair:
    """Correspondence between a clean and poisoned sample."""

    clean_index: int
    poison_index: int
    row_number: int
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class LossCurve:
    """A loss curve indexed by epoch."""

    epochs: Tuple[int, ...]
    values: Tuple[float, ...]
    column: str
    path: Path


def _first_present(
    fields: Iterable[str],
    candidates: Sequence[str],
) -> Optional[str]:
    field_set = set(fields)
    for candidate in candidates:
        if candidate in field_set:
            return candidate
    return None


def _parse_int(value: Any, field_name: str, row_number: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer in field {field_name!r} at row {row_number}: "
            f"{value!r}"
        ) from exc


def _parse_float(value: Any, field_name: str, row_number: int) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid number in field {field_name!r} at row {row_number}: "
            f"{value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite number in field {field_name!r} at row {row_number}: "
            f"{value!r}"
        )
    return parsed


def read_shapley_csv(path: Path) -> Dict[int, float]:
    """Read a sample-indexed Shapley CSV with common legacy field names."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Shapley CSV does not exist: {path}")

    values: Dict[int, float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = reader.fieldnames or []
        index_field = _first_present(fields, INDEX_COLUMNS)
        value_field = _first_present(fields, VALUE_COLUMNS)
        if index_field is None or value_field is None:
            raise ValueError(
                f"{path} must contain an index column from {INDEX_COLUMNS} "
                f"and a Shapley column from {VALUE_COLUMNS}; found {fields}."
            )

        for row_number, row in enumerate(reader, start=2):
            if not row.get(index_field) or not row.get(value_field):
                raise ValueError(
                    f"Missing Shapley index/value at {path}:{row_number}."
                )
            sample_index = _parse_int(
                row[index_field],
                index_field,
                row_number,
            )
            if sample_index in values:
                raise ValueError(
                    f"Duplicate sample index {sample_index} in {path}."
                )
            values[sample_index] = _parse_float(
                row[value_field],
                value_field,
                row_number,
            )

    if not values:
        raise ValueError(f"No Shapley rows found in {path}.")
    return values


def _read_mapping_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mapping CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No mapping rows found in {path}.")
    return rows, fields


def _infer_append_offset(
    clean_scores: Mapping[int, float],
    poison_scores: Mapping[int, float],
    rows: Sequence[Mapping[str, str]],
    poison_ordinal_field: str,
) -> int:
    """Infer HTBM append-mode's base dataset size from score indices."""

    ordinals = [
        int(str(row[poison_ordinal_field]).strip())
        for row in rows
    ]
    if not ordinals or min(ordinals) < 0:
        raise ValueError("Poison ordinals must be non-negative.")

    appended_count = max(ordinals) + 1
    clean_max = max(clean_scores)
    poison_max = max(poison_scores)
    candidates = {
        clean_max + 1 - appended_count,
        poison_max + 1 - appended_count,
    }
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate >= 0
        and all(candidate + ordinal in clean_scores for ordinal in ordinals)
        and all(candidate + ordinal in poison_scores for ordinal in ordinals)
    ]
    if len(valid_candidates) == 1:
        return valid_candidates[0]
    if len(valid_candidates) > 1 and len(set(valid_candidates)) == 1:
        return valid_candidates[0]
    raise ValueError(
        "Could not infer the HTBM append offset. Pass "
        "--append-offset explicitly. The expected offset is the number of "
        "base clean samples before appended poison samples."
    )


def load_sample_pairs(
    mapping_path: Path,
    clean_scores: Mapping[int, float],
    poison_scores: Mapping[int, float],
    pair_mode: str = "auto",
    append_offset: Optional[int] = None,
) -> List[ShapleyPair]:
    """Load clean/poison correspondences from PPIS/APIS/APS/HTBM mappings.

    Supported mapping layouts include:

    * PPIS/APIS/APS: ``original_index``. The clean and poison CSV indices
      are identical.
    * Explicit pairs: ``clean_index`` and ``poison_index``.
    * HTBM append mode: ``target_original_index`` and ordinal
      ``poison_index``. The actual CSV index is
      ``append_offset + poison_index``.
    """

    if pair_mode not in {"auto", "replace", "append"}:
        raise ValueError("pair_mode must be 'auto', 'replace', or 'append'.")

    rows, fields = _read_mapping_rows(Path(mapping_path))
    field_set = set(fields)
    pairs: List[ShapleyPair] = []

    explicit_clean_field = _first_present(
        field_set,
        (
            "clean_sample_index",
            "clean_index",
            "clean_original_index",
        ),
    )
    explicit_poison_field = _first_present(
        field_set,
        (
            "poisoned_sample_index",
            "poison_sample_index",
            "poison_index_actual",
            "poisoned_index",
        ),
    )

    is_htbm_mapping = (
        "target_original_index" in field_set
        and "poison_index" in field_set
        and explicit_clean_field is None
        and explicit_poison_field is None
    )

    resolved_append_offset: Optional[int] = append_offset
    if is_htbm_mapping and pair_mode in {"auto", "append"}:
        if resolved_append_offset is None:
            resolved_append_offset = _infer_append_offset(
                clean_scores=clean_scores,
                poison_scores=poison_scores,
                rows=rows,
                poison_ordinal_field="poison_index",
            )

    for row_number, row in enumerate(rows, start=2):
        if explicit_clean_field and explicit_poison_field:
            clean_index = _parse_int(
                row[explicit_clean_field],
                explicit_clean_field,
                row_number,
            )
            poison_index = _parse_int(
                row[explicit_poison_field],
                explicit_poison_field,
                row_number,
            )
        elif is_htbm_mapping:
            if pair_mode == "replace":
                clean_index = _parse_int(
                    row["target_original_index"],
                    "target_original_index",
                    row_number,
                )
                poison_index = clean_index
            else:
                if resolved_append_offset is None:
                    raise ValueError("HTBM append offset is unavailable.")
                ordinal = _parse_int(
                    row["poison_index"],
                    "poison_index",
                    row_number,
                )
                clean_index = resolved_append_offset + ordinal
                poison_index = clean_index
        else:
            original_field = _first_present(
                field_set,
                ("original_index", "sample_index", "index"),
            )
            if original_field is None:
                raise ValueError(
                    f"Cannot infer clean/poison indices from mapping fields "
                    f"{fields}. Provide explicit clean_index and "
                    "poisoned_sample_index columns."
                )
            clean_index = _parse_int(
                row[original_field],
                original_field,
                row_number,
            )

            if "poisoned_sample_index" in field_set:
                poison_index = _parse_int(
                    row["poisoned_sample_index"],
                    "poisoned_sample_index",
                    row_number,
                )
            elif "poisoned_index" in field_set:
                poison_index = _parse_int(
                    row["poisoned_index"],
                    "poisoned_index",
                    row_number,
                )
            else:
                poison_index = clean_index

        if clean_index not in clean_scores:
            raise ValueError(
                f"Clean index {clean_index} from mapping row {row_number} "
                "does not occur in the clean Shapley CSV."
            )
        if poison_index not in poison_scores:
            raise ValueError(
                f"Poison index {poison_index} from mapping row {row_number} "
                "does not occur in the poisoned Shapley CSV."
            )

        pairs.append(
            ShapleyPair(
                clean_index=clean_index,
                poison_index=poison_index,
                row_number=row_number,
                metadata=row,
            )
        )

    clean_indices = [pair.clean_index for pair in pairs]
    poison_indices = [pair.poison_index for pair in pairs]
    if len(set(clean_indices)) != len(clean_indices):
        raise ValueError("Mapping contains duplicate clean sample indices.")
    if len(set(poison_indices)) != len(poison_indices):
        raise ValueError("Mapping contains duplicate poison sample indices.")
    return pairs


def add_metric_input_arguments(parser: Any, mapping_required: bool = True) -> None:
    """Add shared Shapley/mapping arguments to a metric CLI parser."""

    parser.add_argument(
        "--clean-shapley",
        type=Path,
        required=True,
        help="CSV produced by the clean-data IRDS run.",
    )
    parser.add_argument(
        "--poisoned-shapley",
        type=Path,
        required=True,
        help="CSV produced by the poisoned-data IRDS run.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=mapping_required,
        default=None,
        help=(
            "Poison mapping CSV. Required for pair/client-level metrics; "
            "typically poisoned_mapping.csv, apis_mapping.csv, "
            "aps_mapping.csv, or htbm_mapping.csv."
        ),
    )
    parser.add_argument(
        "--pair-mode",
        choices=("auto", "replace", "append"),
        default="auto",
        help="Interpret mapping pairs as replacement or append-mode records.",
    )
    parser.add_argument(
        "--append-offset",
        type=int,
        default=None,
        help="HTBM append-mode base sample count; auto-inferred when possible.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path; defaults beside the clean Shapley CSV.",
    )


def write_json_result(
    result: Mapping[str, Any],
    output_path: Optional[Path],
    default_directory: Path,
    default_name: str,
) -> Path:
    """Write a UTF-8 JSON summary and return its path."""

    path = Path(output_path) if output_path is not None else (
        Path(default_directory) / default_name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(dict(result), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return path


def positive_contribution_rewards(
    scores: Mapping[int, float],
    rule: str = "formula6",
    budget: float = 1.0,
    tullock_exponent: float = 2.0,
) -> Dict[int, float]:
    """Compute per-sample gains from Formula 6 or its Tullock variant."""

    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if rule not in {"formula6", "tullock"}:
        raise ValueError("rule must be 'formula6' or 'tullock'.")
    if tullock_exponent <= 0:
        raise ValueError("tullock_exponent must be positive.")

    magnitudes = {
        index: -score
        for index, score in scores.items()
        if score < 0.0
    }
    if not magnitudes:
        return {index: 0.0 for index in scores}

    if rule == "formula6":
        weights = magnitudes
    else:
        weights = {
            index: magnitude ** tullock_exponent
            for index, magnitude in magnitudes.items()
        }

    denominator = sum(weights.values())
    if denominator <= 0.0 or not math.isfinite(denominator):
        return {index: 0.0 for index in scores}

    rewards = {index: 0.0 for index in scores}
    for index, weight in weights.items():
        rewards[index] = budget * weight / denominator
    return rewards


def sum_rewards(
    rewards: Mapping[int, float],
    indices: Iterable[int],
) -> float:
    return float(sum(rewards.get(index, 0.0) for index in indices))


def safe_ratio(
    numerator: float,
    denominator: float,
    name: str,
) -> float:
    """Return a ratio, using NaN for an undefined zero-denominator case."""

    if denominator == 0.0:
        warnings.warn(
            f"{name} is undefined because its denominator is zero.",
            RuntimeWarning,
        )
        return float("nan")
    return float(numerator / denominator)


def read_loss_curve(
    path: Path,
    loss_column: str = "train_loss",
) -> LossCurve:
    """Read a CSV or JSON loss curve."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Loss curve does not exist: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as input_file:
            payload = json.load(input_file)
        if isinstance(payload, Mapping):
            if loss_column in payload and isinstance(payload[loss_column], list):
                values = payload[loss_column]
                epochs = list(range(1, len(values) + 1))
            elif "loss" in payload and isinstance(payload["loss"], list):
                values = payload["loss"]
                epochs = list(range(1, len(values) + 1))
            elif isinstance(payload.get("history"), list):
                epochs = []
                values = []
                for row in payload["history"]:
                    if not isinstance(row, Mapping):
                        raise ValueError(f"Invalid history row in {path}.")
                    epochs.append(int(row.get("epoch", len(epochs) + 1)))
                    values.append(row.get(loss_column, row.get("loss")))
            else:
                raise ValueError(
                    f"Could not find {loss_column!r} or 'loss' in {path}."
                )
        elif isinstance(payload, list):
            epochs = list(range(1, len(payload) + 1))
            values = payload
        else:
            raise ValueError(f"Unsupported JSON loss-curve structure: {path}.")

        parsed_values = tuple(
            _parse_float(value, loss_column, index + 1)
            for index, value in enumerate(values)
        )
        parsed_epochs = tuple(int(epoch) for epoch in epochs)
        if not parsed_values:
            raise ValueError(f"Empty loss curve: {path}.")
        return LossCurve(parsed_epochs, parsed_values, loss_column, path)

    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fields = reader.fieldnames or []
        selected_loss_column = (
            loss_column
            if loss_column in fields
            else _first_present(
                fields,
                ("train_loss", "test_loss", "loss"),
            )
        )
        if selected_loss_column is None:
            raise ValueError(
                f"{path} does not contain a loss column. Found {fields}."
            )
        epoch_column = _first_present(fields, ("epoch", "step", "iteration"))

        epochs: List[int] = []
        values: List[float] = []
        for row_number, row in enumerate(reader, start=2):
            if not row.get(selected_loss_column):
                raise ValueError(
                    f"Missing loss at {path}:{row_number}."
                )
            epoch = (
                row_number - 1
                if epoch_column is None
                else _parse_int(
                    row[epoch_column],
                    epoch_column,
                    row_number,
                )
            )
            epochs.append(epoch)
            values.append(
                _parse_float(
                    row[selected_loss_column],
                    selected_loss_column,
                    row_number,
                )
            )

    if not values:
        raise ValueError(f"Empty loss curve: {path}.")
    if len(set(epochs)) != len(epochs):
        raise ValueError(f"Duplicate epochs in loss curve: {path}.")
    return LossCurve(
        tuple(epochs),
        tuple(values),
        selected_loss_column,
        path,
    )


def aggregate_loss_curve(
    curve: LossCurve,
    aggregation: str = "final",
) -> float:
    """Return final loss, mean loss, or trapezoidal area under the curve."""

    if aggregation not in {"final", "mean", "auc"}:
        raise ValueError("aggregation must be 'final', 'mean', or 'auc'.")
    if aggregation == "final":
        return float(curve.values[-1])
    if aggregation == "mean":
        return float(sum(curve.values) / len(curve.values))

    if len(curve.values) == 1:
        return float(curve.values[0])
    area = 0.0
    for left, right, left_epoch, right_epoch in zip(
        curve.values[:-1],
        curve.values[1:],
        curve.epochs[:-1],
        curve.epochs[1:],
    ):
        area += 0.5 * (left + right) * (right_epoch - left_epoch)
    return float(area)


def ensure_same_epochs(
    clean_curve: LossCurve,
    poison_curve: LossCurve,
) -> None:
    if clean_curve.epochs != poison_curve.epochs:
        raise ValueError(
            "Clean and poisoned loss curves must use the same epoch sequence "
            "for a same-epoch LAR comparison."
        )
