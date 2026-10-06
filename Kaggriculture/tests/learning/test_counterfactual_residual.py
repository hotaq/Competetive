from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_rl.imitation.bc import PolicyPrediction
from kaggriculture_rl.learning.counterfactual_residual import (
    CounterfactualResidualDataset,
    load_counterfactual_dataset,
    market_residual_action,
    save_counterfactual_dataset,
)


class _IdentityEngine:
    def sanitize_action(self, seat: int, action: dict[str, Any]) -> dict[str, Any]:
        del seat
        return action


def _prediction(market: list[list[Any]]) -> PolicyPrediction:
    return PolicyPrediction(
        action={
            "farmer": ["PASS"],
            "hands": [["PASS"], ["PASS"]],
            "market": market,
        },
        farmer_confidence=1.0,
        hand_confidences=(1.0, 1.0),
        market_confidences=(0.995,),
    )


def _fallback() -> dict[str, Any]:
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"], ["PASS"]],
        "market": [],
    }


def test_market_residual_accepts_only_real_non_hire_difference() -> None:
    engine = _IdentityEngine()
    proposed = market_residual_action(
        engine,  # type: ignore[arg-type]
        0,
        _prediction([["BUY_SEED", "WHEAT", 1]]),
        _fallback(),
        hands_count=2,
        max_hands=14,
        minimum_confidence=0.99,
    )
    assert proposed is not None
    assert proposed["market"] == [["BUY_SEED", "WHEAT", 1]]

    assert (
        market_residual_action(
            engine,  # type: ignore[arg-type]
            0,
            _prediction([["HIRE"]]),
            _fallback(),
            hands_count=2,
            max_hands=14,
            minimum_confidence=0.99,
        )
        is None
    )
    hire_research = market_residual_action(
        engine,  # type: ignore[arg-type]
        0,
        _prediction([["HIRE"]]),
        _fallback(),
        hands_count=2,
        max_hands=14,
        minimum_confidence=0.99,
        allow_hire=True,
    )
    assert hire_research is not None
    assert hire_research["market"] == [["HIRE"]]
    assert (
        market_residual_action(
            engine,  # type: ignore[arg-type]
            0,
            _prediction([]),
            _fallback(),
            hands_count=2,
            max_hands=14,
            minimum_confidence=0.99,
        )
        is None
    )


def test_counterfactual_dataset_round_trip(tmp_path: Path) -> None:
    dataset = CounterfactualResidualDataset(
        features=np.arange(20, dtype=np.float32).reshape(4, 5),
        delta_margin=np.asarray([10.0, -2.0, 0.0, 3.0], dtype=np.float32),
        delta_money=np.asarray([8.0, -2.0, 0.0, 3.0], dtype=np.float32),
        confidence=np.asarray([0.99, 0.995, 1.0, 0.999], dtype=np.float32),
        seed=np.asarray([1, 1, 2, 2], dtype=np.int64),
        candidate_seat=np.asarray([0, 1, 0, 1], dtype=np.int8),
        turn=np.asarray([10, 10, 20, 20], dtype=np.int16),
        episode_index=np.asarray([0, 1, 2, 3], dtype=np.int32),
        train_mask=np.asarray([True, True, False, False]),
        opponent=("crop:MELON:6",) * 4,
        fallback_market=("[]",) * 4,
        proposal_market=('[[' + '"SELL","MILK",1]]',) * 4,
        observation_digest=("a", "b", "c", "d"),
        metadata={"schema_version": 1},
    )
    path = tmp_path / "counterfactual.npz"
    save_counterfactual_dataset(path, dataset)
    loaded = load_counterfactual_dataset(path)

    np.testing.assert_array_equal(loaded.features, dataset.features)
    np.testing.assert_array_equal(loaded.delta_margin, dataset.delta_margin)
    assert loaded.proposal_market == dataset.proposal_market
    assert loaded.metadata == dataset.metadata
    assert loaded.beneficial.tolist() == [True, False, False, True]
