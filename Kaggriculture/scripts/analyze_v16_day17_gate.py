"""Find small observable gates for the V16 day-17 market overlay.

The search deliberately excludes the seed and future outcomes from its features.  It
only considers state fields that are visible to the live agent, then ranks rules that
avoid negative paired deltas in both development blocks.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Atom:
    feature: str
    operator: str
    threshold: float
    mask: int


def _numeric_features(row: dict[str, Any]) -> dict[str, float]:
    state = row["state"]
    features = {
        key: float(value)
        for key, value in state.items()
        if isinstance(value, (int, float))
        and key
        not in {
            "candidate_money",
            "opponent_money",
            "margin",
        }
    }
    for shop in state.get("shops", []):
        name = f"shop_count:{shop}"
        features[name] = features.get(name, 0.0) + 1.0
    return features


def _candidate_atoms(rows: list[dict[str, Any]]) -> list[Atom]:
    feature_rows = [_numeric_features(row) for row in rows]
    names = sorted({name for features in feature_rows for name in features})
    atoms: list[Atom] = []
    seen: set[tuple[str, str, int]] = set()
    for name in names:
        values = [features.get(name, 0.0) for features in feature_rows]
        unique = sorted(set(values))
        thresholds = (
            unique
            if len(unique) == 1
            else [(left + right) / 2.0 for left, right in itertools.pairwise(unique)]
        )
        for operator in ("<=", ">="):
            for threshold in thresholds:
                mask = 0
                for index, value in enumerate(values):
                    selected = value <= threshold if operator == "<=" else value >= threshold
                    if selected:
                        mask |= 1 << index
                identity = (name, operator, mask)
                if mask and identity not in seen:
                    seen.add(identity)
                    atoms.append(Atom(name, operator, threshold, mask))
    return atoms


def _describe(atom: Atom) -> str:
    threshold = int(atom.threshold) if atom.threshold.is_integer() else atom.threshold
    return f"{atom.feature} {atom.operator} {threshold}"


def _score_rule(
    rows: list[dict[str, Any]],
    mask: int,
    delta_key: str,
    split: int,
) -> dict[str, Any] | None:
    selected = [index for index in range(len(rows)) if mask & (1 << index)]
    first = [index for index in selected if index < split]
    second = [index for index in selected if index >= split]
    if len(first) < 2 or len(second) < 2:
        return None
    deltas = [float(rows[index][delta_key]) for index in selected]
    money_key = delta_key.replace("_delta", "_money_delta")
    money_deltas = [float(rows[index][money_key]) for index in selected]
    if min(deltas) < 0.0 or min(money_deltas) < 0.0:
        return None
    return {
        "support": len(selected),
        "block_support": [len(first), len(second)],
        "total_delta": sum(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "minimum_delta": min(deltas),
        "total_money_delta": sum(money_deltas),
        "mean_money_delta": sum(money_deltas) / len(money_deltas),
        "minimum_money_delta": min(money_deltas),
        "selected_seeds": [int(rows[index]["seed"]) for index in selected],
        "selected_deltas": deltas,
        "selected_money_deltas": money_deltas,
    }


def analyze(path: Path, *, split: int, limit: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = list(payload["rows"])
    atoms = _candidate_atoms(rows)
    results: list[dict[str, Any]] = []
    for delta_key in ("cap1_delta", "cap3_delta"):
        for atom in atoms:
            score = _score_rule(rows, atom.mask, delta_key, split)
            if score is not None:
                results.append(
                    {
                        "quantity": 1 if delta_key == "cap1_delta" else 3,
                        "rule": [_describe(atom)],
                        **score,
                    }
                )
        for left_index, left in enumerate(atoms):
            for right in atoms[left_index + 1 :]:
                if left.feature == right.feature:
                    continue
                mask = left.mask & right.mask
                score = _score_rule(rows, mask, delta_key, split)
                if score is not None:
                    results.append(
                        {
                            "quantity": 1 if delta_key == "cap1_delta" else 3,
                            "rule": [_describe(left), _describe(right)],
                            **score,
                        }
                    )
    results.sort(
        key=lambda result: (
            result["minimum_money_delta"],
            result["minimum_delta"],
            result["total_money_delta"],
            result["total_delta"],
            result["support"],
            -len(result["rule"]),
        ),
        reverse=True,
    )
    unique: list[dict[str, Any]] = []
    seen_masks: set[tuple[int, tuple[int, ...]]] = set()
    for result in results:
        identity = (result["quantity"], tuple(result["selected_seeds"]))
        if identity in seen_masks:
            continue
        seen_masks.add(identity)
        unique.append(result)
        if len(unique) >= limit:
            break
    return {
        "dataset": str(path),
        "rows": len(rows),
        "split": split,
        "observable_atoms": len(atoms),
        "rules": unique,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", type=int, default=12)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset, split=args.split, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
