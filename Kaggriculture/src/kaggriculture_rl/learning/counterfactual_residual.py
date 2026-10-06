"""Single-deviation counterfactual labels for safe V16 market residuals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from kaggriculture_rs import Engine

from kaggriculture_rl.imitation.teacher_dataset import (
    StandaloneFilePolicy,
    sanitize_standalone_action,
)
from kaggriculture_rl.learning.gated_ppo import GateConfig
from kaggriculture_rl.learning.spatial_recurrent import SpatialRecurrentPolicy
from kaggriculture_rl.learning.teacher_gate_ppo import (
    DEFAULT_CROP_OPPONENTS,
    _market_within_hand_limit,
    _opponent_catalog,
    teacher_gate_action,
    teacher_gate_features,
)
from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy


@dataclass(frozen=True, slots=True)
class CounterfactualResidualDataset:
    features: np.ndarray
    delta_margin: np.ndarray
    delta_money: np.ndarray
    confidence: np.ndarray
    seed: np.ndarray
    candidate_seat: np.ndarray
    turn: np.ndarray
    episode_index: np.ndarray
    train_mask: np.ndarray
    opponent: tuple[str, ...]
    fallback_market: tuple[str, ...]
    proposal_market: tuple[str, ...]
    observation_digest: tuple[str, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        rows = len(self.features)
        if self.features.ndim != 2 or rows == 0:
            raise ValueError("counterfactual features must be a non-empty matrix")
        arrays = (
            self.delta_margin,
            self.delta_money,
            self.confidence,
            self.seed,
            self.candidate_seat,
            self.turn,
            self.episode_index,
            self.train_mask,
        )
        if any(len(values) != rows for values in arrays):
            raise ValueError("counterfactual array lengths do not match")
        strings = (
            self.opponent,
            self.fallback_market,
            self.proposal_market,
            self.observation_digest,
        )
        if any(len(values) != rows for values in strings):
            raise ValueError("counterfactual string lengths do not match")
        if not np.isfinite(self.features).all():
            raise ValueError("counterfactual features contain non-finite values")
        if not np.isfinite(self.delta_margin).all():
            raise ValueError("counterfactual margins contain non-finite values")
        if not np.isfinite(self.delta_money).all():
            raise ValueError("counterfactual money deltas contain non-finite values")
        if not np.logical_and(self.confidence >= 0.0, self.confidence <= 1.0).all():
            raise ValueError("counterfactual confidence must be within 0..1")
        if not self.train_mask.any() or self.train_mask.all():
            raise ValueError("counterfactual dataset needs train and holdout rows")

    @property
    def beneficial(self) -> np.ndarray:
        return self.delta_margin > 0.0


@dataclass(frozen=True, slots=True)
class _ResidualCandidate:
    episode_index: int
    seed: int
    candidate_seat: int
    opponent: str
    turn: int
    features: np.ndarray
    confidence: float
    fallback_market: str
    proposal_market: str
    observation_digest: str


@dataclass(frozen=True, slots=True)
class _GameResult:
    candidate_money: float
    opponent_money: float
    margin: float


class _HandCapacityOverflow(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _observation_digest(observation: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(observation).encode()).hexdigest()


def _market_confidence(prediction: Any) -> float:
    return float(min(prediction.market_confidences, default=0.0))


def market_residual_action(
    engine: Engine,
    seat: int,
    prediction: Any,
    fallback: dict[str, Any],
    *,
    hands_count: int,
    max_hands: int,
    minimum_confidence: float,
    allow_hire: bool = False,
) -> dict[str, Any] | None:
    """Return a sanitized market-only deviation, or ``None`` when unsafe/no-op."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be within 0..1")
    if _market_confidence(prediction) < minimum_confidence:
        return None
    if not allow_hire and not _market_within_hand_limit(
        prediction,
        hands_count=hands_count,
        max_hands=max_hands,
    ):
        return None
    gates = np.zeros(max_hands + 2, dtype=np.bool_)
    gates[-1] = True
    proposed = teacher_gate_action(
        prediction,
        fallback,
        gates,
        max_hands=max_hands,
    )
    proposed = sanitize_standalone_action(engine, seat, proposed, hands_count)
    if proposed["market"] == fallback["market"]:
        return None
    return proposed


def _make_opponent(
    label: str,
    payload: Any,
) -> tuple[StandaloneFilePolicy | None, CropHeuristicPolicy | None]:
    if label.startswith("file:"):
        return StandaloneFilePolicy(Path(payload), role="counterfactual_opponent"), None
    crop, plots = payload
    return None, CropHeuristicPolicy(crop=crop, target_quantity=plots, sell_reserve=0)


def _opponent_action(
    engine: Engine,
    opponent_seat: int,
    observation: dict[str, Any],
    file_opponent: StandaloneFilePolicy | None,
    crop_opponent: CropHeuristicPolicy | None,
) -> dict[str, Any]:
    hands_count = len(observation["farms"][opponent_seat].get("hands", []))
    if file_opponent is not None:
        raw_action = file_opponent(observation)
    else:
        assert crop_opponent is not None
        plan = engine.plan(
            opponent_seat,
            crop_opponent({"shed": observation["private"]["shed"]}),
        )
        raw_action = plan["joint_action"]["actions"][opponent_seat]
    return sanitize_standalone_action(
        engine,
        opponent_seat,
        raw_action,
        hands_count,
    )


def _final_result(engine: Engine, candidate_seat: int) -> _GameResult:
    opponent_seat = 1 - candidate_seat
    money = [float(farm["money"]) for farm in engine.state["farms"]]
    return _GameResult(
        candidate_money=money[candidate_seat],
        opponent_money=money[opponent_seat],
        margin=money[candidate_seat] - money[opponent_seat],
    )


def _baseline_game(
    spatial_checkpoint: Path,
    teacher_path: Path,
    *,
    episode_index: int,
    seed: int,
    candidate_seat: int,
    opponent_label: str,
    opponent_payload: Any,
    minimum_confidence: float,
    max_hands: int,
    max_candidates: int,
    allow_hire: bool,
) -> tuple[_GameResult, tuple[_ResidualCandidate, ...]]:
    opponent_seat = 1 - candidate_seat
    engine = Engine(
        {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    )
    spatial = SpatialRecurrentPolicy(spatial_checkpoint)
    teacher = StandaloneFilePolicy(teacher_path, role="counterfactual_teacher")
    file_opponent, crop_opponent = _make_opponent(opponent_label, opponent_payload)
    config = GateConfig(action_dim=max_hands + 2)
    candidates: list[_ResidualCandidate] = []
    try:
        for turn in range(719):
            observation = engine.observation(candidate_seat)
            opponent_observation = engine.observation(opponent_seat)
            hands_count = len(observation["farms"][candidate_seat].get("hands", []))
            if hands_count > max_hands:
                raise RuntimeError(
                    f"baseline exceeded checkpoint hand capacity: {hands_count}"
                )
            fallback = sanitize_standalone_action(
                engine,
                candidate_seat,
                teacher(observation),
                hands_count,
            )
            prediction = spatial.predict_with_confidence(engine, observation)
            proposed = market_residual_action(
                engine,
                candidate_seat,
                prediction,
                fallback,
                hands_count=hands_count,
                max_hands=max_hands,
                minimum_confidence=minimum_confidence,
                allow_hire=allow_hire,
            )
            if proposed is not None and len(candidates) < max_candidates:
                candidates.append(
                    _ResidualCandidate(
                        episode_index=episode_index,
                        seed=seed,
                        candidate_seat=candidate_seat,
                        opponent=opponent_label,
                        turn=turn,
                        features=teacher_gate_features(observation, prediction, config),
                        confidence=_market_confidence(prediction),
                        fallback_market=_canonical_json(fallback["market"]),
                        proposal_market=_canonical_json(proposed["market"]),
                        observation_digest=_observation_digest(observation),
                    )
                )
            opponent_action = _opponent_action(
                engine,
                opponent_seat,
                opponent_observation,
                file_opponent,
                crop_opponent,
            )
            actions = [opponent_action, opponent_action]
            actions[candidate_seat] = fallback
            engine.step(actions)
        return _final_result(engine, candidate_seat), tuple(candidates)
    finally:
        teacher.close()
        if file_opponent is not None:
            file_opponent.close()


def _counterfactual_game(
    spatial_checkpoint: Path,
    teacher_path: Path,
    candidate: _ResidualCandidate,
    *,
    opponent_payload: Any,
    minimum_confidence: float,
    max_hands: int,
    allow_hire: bool,
) -> _GameResult:
    candidate_seat = candidate.candidate_seat
    opponent_seat = 1 - candidate_seat
    engine = Engine(
        {"episodeSteps": 720, "seed": candidate.seed, "weedSpawnChance": 0.005}
    )
    spatial = SpatialRecurrentPolicy(spatial_checkpoint)
    teacher = StandaloneFilePolicy(teacher_path, role="counterfactual_teacher")
    file_opponent, crop_opponent = _make_opponent(candidate.opponent, opponent_payload)
    applied = False
    try:
        for turn in range(719):
            observation = engine.observation(candidate_seat)
            opponent_observation = engine.observation(opponent_seat)
            hands_count = len(observation["farms"][candidate_seat].get("hands", []))
            if hands_count > max_hands:
                raise _HandCapacityOverflow(
                    f"counterfactual exceeded hand capacity: {hands_count}"
                )
            fallback = sanitize_standalone_action(
                engine,
                candidate_seat,
                teacher(observation),
                hands_count,
            )
            prediction = spatial.predict_with_confidence(engine, observation)
            candidate_action = fallback
            if turn == candidate.turn:
                digest = _observation_digest(observation)
                if digest != candidate.observation_digest:
                    raise RuntimeError(
                        "counterfactual first-divergence before target turn: "
                        f"expected {candidate.observation_digest}, got {digest}"
                    )
                proposed = market_residual_action(
                    engine,
                    candidate_seat,
                    prediction,
                    fallback,
                    hands_count=hands_count,
                    max_hands=max_hands,
                    minimum_confidence=minimum_confidence,
                    allow_hire=allow_hire,
                )
                if proposed is None:
                    raise RuntimeError("counterfactual target is no longer eligible")
                if _canonical_json(fallback["market"]) != candidate.fallback_market:
                    raise RuntimeError("counterfactual fallback action changed at target")
                if _canonical_json(proposed["market"]) != candidate.proposal_market:
                    raise RuntimeError("counterfactual proposal action changed at target")
                candidate_action = proposed
                applied = True
            opponent_action = _opponent_action(
                engine,
                opponent_seat,
                opponent_observation,
                file_opponent,
                crop_opponent,
            )
            actions = [opponent_action, opponent_action]
            actions[candidate_seat] = candidate_action
            engine.step(actions)
        if not applied:
            raise RuntimeError("counterfactual target turn was not applied")
        final_hands = len(engine.state["farms"][candidate_seat].get("hands", []))
        if final_hands > max_hands:
            raise _HandCapacityOverflow(
                f"counterfactual ended above hand capacity: {final_hands}"
            )
        return _final_result(engine, candidate_seat)
    finally:
        teacher.close()
        if file_opponent is not None:
            file_opponent.close()


def _residual_policy_game(
    spatial_checkpoint: Path,
    teacher_path: Path,
    *,
    episode_index: int,
    seed: int,
    candidate_seat: int,
    opponent_label: str,
    opponent_payload: Any,
    minimum_confidence: float,
    max_hands: int,
    max_candidates: int,
    reject_candidate: _ResidualCandidate | None = None,
) -> tuple[_GameResult, tuple[_ResidualCandidate, ...]]:
    """Run the live residual policy, optionally rejecting one recorded decision."""

    opponent_seat = 1 - candidate_seat
    engine = Engine(
        {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    )
    spatial = SpatialRecurrentPolicy(spatial_checkpoint)
    teacher = StandaloneFilePolicy(teacher_path, role="residual_policy_teacher")
    file_opponent, crop_opponent = _make_opponent(opponent_label, opponent_payload)
    config = GateConfig(action_dim=max_hands + 2)
    candidates: list[_ResidualCandidate] = []
    rejected = False
    try:
        for turn in range(719):
            observation = engine.observation(candidate_seat)
            opponent_observation = engine.observation(opponent_seat)
            hands_count = len(observation["farms"][candidate_seat].get("hands", []))
            if hands_count > max_hands:
                raise _HandCapacityOverflow(
                    f"residual policy exceeded hand capacity: {hands_count}"
                )
            fallback = sanitize_standalone_action(
                engine,
                candidate_seat,
                teacher(observation),
                hands_count,
            )
            prediction = spatial.predict_with_confidence(engine, observation)
            proposed = market_residual_action(
                engine,
                candidate_seat,
                prediction,
                fallback,
                hands_count=hands_count,
                max_hands=max_hands,
                minimum_confidence=minimum_confidence,
            )
            candidate_action = fallback if proposed is None else proposed
            if (
                proposed is not None
                and reject_candidate is None
                and len(candidates) < max_candidates
            ):
                candidates.append(
                    _ResidualCandidate(
                        episode_index=episode_index,
                        seed=seed,
                        candidate_seat=candidate_seat,
                        opponent=opponent_label,
                        turn=turn,
                        features=teacher_gate_features(
                            observation,
                            prediction,
                            config,
                        ),
                        confidence=_market_confidence(prediction),
                        fallback_market=_canonical_json(fallback["market"]),
                        proposal_market=_canonical_json(proposed["market"]),
                        observation_digest=_observation_digest(observation),
                    )
                )
            if reject_candidate is not None and turn == reject_candidate.turn:
                digest = _observation_digest(observation)
                if digest != reject_candidate.observation_digest:
                    raise RuntimeError(
                        "leave-one-out first-divergence before target turn: "
                        f"expected {reject_candidate.observation_digest}, got {digest}"
                    )
                if proposed is None:
                    raise RuntimeError("leave-one-out target is no longer eligible")
                if _canonical_json(fallback["market"]) != reject_candidate.fallback_market:
                    raise RuntimeError("leave-one-out fallback changed at target")
                if _canonical_json(proposed["market"]) != reject_candidate.proposal_market:
                    raise RuntimeError("leave-one-out proposal changed at target")
                candidate_action = fallback
                rejected = True
            opponent_action = _opponent_action(
                engine,
                opponent_seat,
                opponent_observation,
                file_opponent,
                crop_opponent,
            )
            actions = [opponent_action, opponent_action]
            actions[candidate_seat] = candidate_action
            engine.step(actions)
        if reject_candidate is not None and not rejected:
            raise RuntimeError("leave-one-out target turn was not rejected")
        return _final_result(engine, candidate_seat), tuple(candidates)
    finally:
        teacher.close()
        if file_opponent is not None:
            file_opponent.close()


def save_counterfactual_dataset(
    path: Path,
    dataset: CounterfactualResidualDataset,
) -> None:
    dataset.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(dataset.features, dtype=np.float32),
        delta_margin=np.asarray(dataset.delta_margin, dtype=np.float32),
        delta_money=np.asarray(dataset.delta_money, dtype=np.float32),
        confidence=np.asarray(dataset.confidence, dtype=np.float32),
        seed=np.asarray(dataset.seed, dtype=np.int64),
        candidate_seat=np.asarray(dataset.candidate_seat, dtype=np.int8),
        turn=np.asarray(dataset.turn, dtype=np.int16),
        episode_index=np.asarray(dataset.episode_index, dtype=np.int32),
        train_mask=np.asarray(dataset.train_mask, dtype=np.bool_),
        opponent=np.asarray(dataset.opponent, dtype=np.str_),
        fallback_market=np.asarray(dataset.fallback_market, dtype=np.str_),
        proposal_market=np.asarray(dataset.proposal_market, dtype=np.str_),
        observation_digest=np.asarray(dataset.observation_digest, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(dataset.metadata, sort_keys=True)),
    )


def load_counterfactual_dataset(path: Path) -> CounterfactualResidualDataset:
    with np.load(path, allow_pickle=False) as payload:
        dataset = CounterfactualResidualDataset(
            features=np.asarray(payload["features"], dtype=np.float32),
            delta_margin=np.asarray(payload["delta_margin"], dtype=np.float32),
            delta_money=np.asarray(payload["delta_money"], dtype=np.float32),
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            seed=np.asarray(payload["seed"], dtype=np.int64),
            candidate_seat=np.asarray(payload["candidate_seat"], dtype=np.int8),
            turn=np.asarray(payload["turn"], dtype=np.int16),
            episode_index=np.asarray(payload["episode_index"], dtype=np.int32),
            train_mask=np.asarray(payload["train_mask"], dtype=np.bool_),
            opponent=tuple(str(value) for value in payload["opponent"]),
            fallback_market=tuple(str(value) for value in payload["fallback_market"]),
            proposal_market=tuple(str(value) for value in payload["proposal_market"]),
            observation_digest=tuple(
                str(value) for value in payload["observation_digest"]
            ),
            metadata=json.loads(str(payload["metadata_json"])),
        )
    dataset.validate()
    return dataset


def _dataset_summary(dataset: CounterfactualResidualDataset) -> dict[str, Any]:
    deltas = dataset.delta_margin.astype(np.float64)
    beneficial = deltas > 0.0
    neutral = deltas == 0.0
    return {
        "rows": len(deltas),
        "beneficial": int(beneficial.sum()),
        "neutral": int(neutral.sum()),
        "harmful": int((deltas < 0.0).sum()),
        "mean_margin_delta": fmean(deltas.tolist()),
        "worst_margin_delta": float(deltas.min()),
        "best_margin_delta": float(deltas.max()),
        "train_rows": int(dataset.train_mask.sum()),
        "holdout_rows": int((~dataset.train_mask).sum()),
    }


def build_counterfactual_market_dataset(
    spatial_checkpoint: Path,
    teacher_path: Path,
    output_path: Path,
    *,
    episodes: int = 8,
    base_seed: int = 708_000,
    minimum_confidence: float = 0.99,
    max_hands: int = 14,
    max_candidates_per_game: int = 16,
    holdout_mod: int = 4,
    holdout_fold: int = 3,
    crop_opponents: Sequence[tuple[str, int]] = DEFAULT_CROP_OPPONENTS,
    file_opponents: Sequence[Path] = (),
    allow_hire_research: bool = False,
) -> dict[str, Any]:
    if episodes <= 0 or episodes % 2:
        raise ValueError("episodes must be a positive even number")
    if max_candidates_per_game <= 0:
        raise ValueError("max_candidates_per_game must be positive")
    if holdout_mod <= 1 or not 0 <= holdout_fold < holdout_mod:
        raise ValueError("invalid holdout split")
    catalog = _opponent_catalog(crop_opponents, file_opponents)
    baseline_games: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    unsafe_counterfactuals: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        pair_index = episode_index // 2
        candidate_seat = episode_index % 2
        seed = base_seed + pair_index
        opponent_label, opponent_payload = catalog[pair_index % len(catalog)]
        baseline, candidates = _baseline_game(
            spatial_checkpoint,
            teacher_path,
            episode_index=episode_index,
            seed=seed,
            candidate_seat=candidate_seat,
            opponent_label=opponent_label,
            opponent_payload=opponent_payload,
            minimum_confidence=minimum_confidence,
            max_hands=max_hands,
            max_candidates=max_candidates_per_game,
            allow_hire=allow_hire_research,
        )
        baseline_games.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "candidate_seat": candidate_seat,
                "opponent": opponent_label,
                "candidate_money": baseline.candidate_money,
                "opponent_money": baseline.opponent_money,
                "margin": baseline.margin,
                "candidate_residuals": len(candidates),
            }
        )
        for candidate in candidates:
            try:
                counterfactual = _counterfactual_game(
                    spatial_checkpoint,
                    teacher_path,
                    candidate,
                    opponent_payload=opponent_payload,
                    minimum_confidence=minimum_confidence,
                    max_hands=max_hands,
                    allow_hire=allow_hire_research,
                )
            except _HandCapacityOverflow as error:
                unsafe_counterfactuals.append(
                    {
                        "seed": candidate.seed,
                        "candidate_seat": candidate.candidate_seat,
                        "opponent": candidate.opponent,
                        "turn": candidate.turn,
                        "proposal_market": candidate.proposal_market,
                        "reason": str(error),
                    }
                )
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "delta_margin": counterfactual.margin - baseline.margin,
                    "delta_money": (
                        counterfactual.candidate_money - baseline.candidate_money
                    ),
                    "baseline_margin": baseline.margin,
                    "counterfactual_margin": counterfactual.margin,
                }
            )
    if not rows:
        raise RuntimeError("no eligible counterfactual residuals were found")
    episode_indices = np.asarray(
        [row["candidate"].episode_index for row in rows], dtype=np.int32
    )
    pair_indices = episode_indices // 2
    train_mask = pair_indices % holdout_mod != holdout_fold
    dataset = CounterfactualResidualDataset(
        features=np.stack([row["candidate"].features for row in rows]).astype(
            np.float32
        ),
        delta_margin=np.asarray([row["delta_margin"] for row in rows], dtype=np.float32),
        delta_money=np.asarray([row["delta_money"] for row in rows], dtype=np.float32),
        confidence=np.asarray(
            [row["candidate"].confidence for row in rows], dtype=np.float32
        ),
        seed=np.asarray([row["candidate"].seed for row in rows], dtype=np.int64),
        candidate_seat=np.asarray(
            [row["candidate"].candidate_seat for row in rows], dtype=np.int8
        ),
        turn=np.asarray([row["candidate"].turn for row in rows], dtype=np.int16),
        episode_index=episode_indices,
        train_mask=train_mask,
        opponent=tuple(row["candidate"].opponent for row in rows),
        fallback_market=tuple(row["candidate"].fallback_market for row in rows),
        proposal_market=tuple(row["candidate"].proposal_market for row in rows),
        observation_digest=tuple(
            row["candidate"].observation_digest for row in rows
        ),
        metadata={
            "schema_version": 1,
            "model": "v16-market-single-deviation-counterfactual-v1",
            "spatial_checkpoint": str(spatial_checkpoint.resolve()),
            "spatial_checkpoint_sha256": _sha256(spatial_checkpoint),
            "teacher_path": str(teacher_path.resolve()),
            "teacher_sha256": _sha256(teacher_path),
            "episodes": episodes,
            "base_seed": base_seed,
            "minimum_confidence": minimum_confidence,
            "max_hands": max_hands,
            "max_candidates_per_game": max_candidates_per_game,
            "holdout_mod": holdout_mod,
            "holdout_fold": holdout_fold,
            "crop_opponents": list(crop_opponents),
            "file_opponents": [str(path.resolve()) for path in file_opponents],
            "allow_hire_research": allow_hire_research,
            "unsafe_counterfactuals": unsafe_counterfactuals,
            "baseline_games": baseline_games,
            "limitation": (
                "Each label measures one deviation followed by the live V16 teacher; "
                "labels are not assumed additive when multiple residuals are accepted."
            ),
        },
    )
    save_counterfactual_dataset(output_path, dataset)
    summary = _dataset_summary(dataset)
    return {
        "dataset": str(output_path.resolve()),
        "summary": summary,
        "baseline_games": baseline_games,
        "unsafe_counterfactuals": unsafe_counterfactuals,
        "metadata": dataset.metadata,
    }


def build_leave_one_out_market_dataset(
    spatial_checkpoint: Path,
    teacher_path: Path,
    output_path: Path,
    *,
    episodes: int = 8,
    base_seed: int = 708_000,
    minimum_confidence: float = 0.99,
    max_hands: int = 14,
    max_candidates_per_game: int = 16,
    holdout_mod: int = 4,
    holdout_fold: int = 3,
    crop_opponents: Sequence[tuple[str, int]] = DEFAULT_CROP_OPPONENTS,
    file_opponents: Sequence[Path] = (),
) -> dict[str, Any]:
    """Label residual decisions by rejecting each one from the live full policy."""

    if episodes <= 0 or episodes % 2:
        raise ValueError("episodes must be a positive even number")
    if max_candidates_per_game <= 0:
        raise ValueError("max_candidates_per_game must be positive")
    if holdout_mod <= 1 or not 0 <= holdout_fold < holdout_mod:
        raise ValueError("invalid holdout split")
    catalog = _opponent_catalog(crop_opponents, file_opponents)
    policy_games: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        pair_index = episode_index // 2
        candidate_seat = episode_index % 2
        seed = base_seed + pair_index
        opponent_label, opponent_payload = catalog[pair_index % len(catalog)]
        baseline, _ = _baseline_game(
            spatial_checkpoint,
            teacher_path,
            episode_index=episode_index,
            seed=seed,
            candidate_seat=candidate_seat,
            opponent_label=opponent_label,
            opponent_payload=opponent_payload,
            minimum_confidence=minimum_confidence,
            max_hands=max_hands,
            max_candidates=1,
            allow_hire=False,
        )
        full_policy, candidates = _residual_policy_game(
            spatial_checkpoint,
            teacher_path,
            episode_index=episode_index,
            seed=seed,
            candidate_seat=candidate_seat,
            opponent_label=opponent_label,
            opponent_payload=opponent_payload,
            minimum_confidence=minimum_confidence,
            max_hands=max_hands,
            max_candidates=max_candidates_per_game,
        )
        policy_games.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "candidate_seat": candidate_seat,
                "opponent": opponent_label,
                "baseline_margin": baseline.margin,
                "full_policy_margin": full_policy.margin,
                "full_policy_delta": full_policy.margin - baseline.margin,
                "candidate_residuals": len(candidates),
            }
        )
        for candidate in candidates:
            without_candidate, _ = _residual_policy_game(
                spatial_checkpoint,
                teacher_path,
                episode_index=episode_index,
                seed=seed,
                candidate_seat=candidate_seat,
                opponent_label=opponent_label,
                opponent_payload=opponent_payload,
                minimum_confidence=minimum_confidence,
                max_hands=max_hands,
                max_candidates=max_candidates_per_game,
                reject_candidate=candidate,
            )
            rows.append(
                {
                    "candidate": candidate,
                    "delta_margin": full_policy.margin - without_candidate.margin,
                    "delta_money": (
                        full_policy.candidate_money
                        - without_candidate.candidate_money
                    ),
                }
            )
    if not rows:
        raise RuntimeError("no eligible leave-one-out residuals were found")
    episode_indices = np.asarray(
        [row["candidate"].episode_index for row in rows], dtype=np.int32
    )
    pair_indices = episode_indices // 2
    train_mask = pair_indices % holdout_mod != holdout_fold
    dataset = CounterfactualResidualDataset(
        features=np.stack([row["candidate"].features for row in rows]).astype(
            np.float32
        ),
        delta_margin=np.asarray([row["delta_margin"] for row in rows], dtype=np.float32),
        delta_money=np.asarray([row["delta_money"] for row in rows], dtype=np.float32),
        confidence=np.asarray(
            [row["candidate"].confidence for row in rows], dtype=np.float32
        ),
        seed=np.asarray([row["candidate"].seed for row in rows], dtype=np.int64),
        candidate_seat=np.asarray(
            [row["candidate"].candidate_seat for row in rows], dtype=np.int8
        ),
        turn=np.asarray([row["candidate"].turn for row in rows], dtype=np.int16),
        episode_index=episode_indices,
        train_mask=train_mask,
        opponent=tuple(row["candidate"].opponent for row in rows),
        fallback_market=tuple(row["candidate"].fallback_market for row in rows),
        proposal_market=tuple(row["candidate"].proposal_market for row in rows),
        observation_digest=tuple(
            row["candidate"].observation_digest for row in rows
        ),
        metadata={
            "schema_version": 1,
            "model": "v16-market-live-policy-leave-one-out-v1",
            "spatial_checkpoint": str(spatial_checkpoint.resolve()),
            "spatial_checkpoint_sha256": _sha256(spatial_checkpoint),
            "teacher_path": str(teacher_path.resolve()),
            "teacher_sha256": _sha256(teacher_path),
            "episodes": episodes,
            "base_seed": base_seed,
            "minimum_confidence": minimum_confidence,
            "max_hands": max_hands,
            "max_candidates_per_game": max_candidates_per_game,
            "holdout_mod": holdout_mod,
            "holdout_fold": holdout_fold,
            "crop_opponents": list(crop_opponents),
            "file_opponents": [str(path.resolve()) for path in file_opponents],
            "policy_games": policy_games,
            "limitation": (
                "Each label is the marginal contribution of one accepted residual "
                "inside the accept-all live residual policy; contributions need not "
                "sum to the full-policy delta."
            ),
        },
    )
    save_counterfactual_dataset(output_path, dataset)
    summary = _dataset_summary(dataset)
    full_policy_deltas = [float(game["full_policy_delta"]) for game in policy_games]
    summary["mean_full_policy_delta"] = fmean(full_policy_deltas)
    summary["worst_full_policy_delta"] = min(full_policy_deltas)
    summary["best_full_policy_delta"] = max(full_policy_deltas)
    return {
        "dataset": str(output_path.resolve()),
        "summary": summary,
        "policy_games": policy_games,
        "metadata": dataset.metadata,
    }
