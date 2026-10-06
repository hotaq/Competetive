"""Conservative V16/V12 selector driven by opponent-model rollouts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import fsum
from pathlib import Path
from typing import Any

POLICIES = ("v16", "v12")


@dataclass(frozen=True, slots=True)
class RolloutSelection:
    policy: str
    predicted_values: dict[str, float]
    conservative_values: dict[str, float]


class RolloutValueSelector:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.feature_names = tuple(str(name) for name in checkpoint["feature_names"])
        normalization = checkpoint["normalization"]
        self.means = tuple(float(value) for value in normalization["means"])
        self.scales = tuple(float(value) for value in normalization["scales"])
        self.weights = {
            policy: tuple(float(value) for value in checkpoint["weights"][policy])
            for policy in POLICIES
        }
        self.uncertainty = {
            policy: float(checkpoint["uncertainty"][policy])
            for policy in POLICIES
        }
        self.beta = float(checkpoint["selected_hyperparameters"]["beta"])
        feature_count = len(self.feature_names)
        if len(self.means) != feature_count or len(self.scales) != feature_count:
            raise ValueError("selector normalization shape mismatch")
        if any(len(self.weights[policy]) != feature_count + 1 for policy in POLICIES):
            raise ValueError("selector weight shape mismatch")

    @classmethod
    def from_path(cls, path: Path) -> RolloutValueSelector:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def features(
        self,
        candidate_seat: int,
        predictions: dict[str, dict[str, dict[str, float]]],
    ) -> tuple[float, ...]:
        values = [float(candidate_seat)]
        names = ["candidate_seat"]
        for reference in sorted(predictions):
            by_policy = predictions[reference]
            for policy in POLICIES:
                prediction = by_policy[policy]
                for metric in ("margin", "candidate_money", "reference_money"):
                    names.append(f"{reference}:{policy}:{metric}")
                    values.append(float(prediction[metric]))
            names.append(f"{reference}:margin_difference")
            values.append(
                float(by_policy["v12"]["margin"])
                - float(by_policy["v16"]["margin"])
            )
        if tuple(names) != self.feature_names:
            raise ValueError(
                "rollout feature schema mismatch: "
                f"expected {self.feature_names}, got {tuple(names)}"
            )
        return tuple(values)

    def select(
        self,
        candidate_seat: int,
        predictions: dict[str, dict[str, dict[str, float]]],
    ) -> RolloutSelection:
        features = self.features(candidate_seat, predictions)
        normalized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                features, self.means, self.scales, strict=True
            )
        )
        design = (1.0, *normalized)
        predicted = {
            policy: fsum(
                weight * value
                for weight, value in zip(
                    self.weights[policy], design, strict=True
                )
            )
            for policy in POLICIES
        }
        conservative = {
            policy: predicted[policy] - self.beta * self.uncertainty[policy]
            for policy in POLICIES
        }
        policy = "v12" if conservative["v12"] > conservative["v16"] else "v16"
        return RolloutSelection(policy, predicted, conservative)
