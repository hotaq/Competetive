#!/usr/bin/env python3
"""Collect the disjoint, observable V85 prefix-feature dataset.

Only already-consumed synthetic development seeds are executed.  Every row is
freshly replayed against a live opponent in the official Rust engine through
step 119; no replay, sealed holdout, or live Kaggle trace is read here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for import_path in (ROOT, SRC, SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import kaggriculture_rs  # noqa: E402
from evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine  # noqa: E402

from kaggriculture_rl.learning.tape_selector import (  # noqa: E402
    encode_selector_features,
)

OUT = Path(__file__).resolve().parent
COLLECTOR_PATH = Path(__file__).resolve()
DATASET_PATH = OUT / "dataset.json"
V84_RESULTS_PATH = ROOT / "analysis/v84_route_option_onpolicy_oracle_20260827/results.json"
V67_PATH = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
ENCODER_PATH = ROOT / "src/kaggriculture_rl/learning/tape_selector.py"
HELPER_PATH = ROOT / "scripts/evaluate_file_agent_vs_replay.py"

SEEDS = (27100100, 27100101, 27100102, 27100103)
SEATS = (0, 1)
BRANCH_STEP = 120
FEATURE_STEPS = (24, 48, 72, 96, 120)
SNAPSHOT_FEATURES = 81
MODEL_FEATURES = len(FEATURE_STEPS) * SNAPSHOT_FEATURES
CONFIG_BASE = {"episodeSteps": 720, "weedSpawnChance": 0.005}

OPTIONS = (
    "ABSTAIN_KEEP_V67",
    "FORCE_BASE_SUFFIX",
    "FORCE_A120_SUFFIX",
    "FORCE_B120_SUFFIX",
    "FORCE_C120_SUFFIX",
)
MODEL_ACTION_TO_V84 = {
    "KEEP": "ABSTAIN_KEEP_V67",
    "B": "FORCE_B120_SUFFIX",
    "C": "FORCE_C120_SUFFIX",
}
FAMILIES = (
    (
        "v16_route_family",
        "v22",
        ROOT / "submissions/meta_v22_v16_land3_terminal_rule_standalone/main.py",
    ),
    (
        "ryo_selector_family",
        "v12_ryo",
        ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py",
    ),
    (
        "route_99612231_family",
        "v44_route",
        ROOT / "submissions/meta_v44_route99612231_orphan_standalone/main.py",
    ),
    (
        "robust_cluster_family",
        "robust_cluster",
        ROOT / "submissions/robust_cluster_tape_v10_standalone/main.py",
    ),
)
FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    {
        "candidate_seat",
        "episode_id",
        "family",
        "opponent_seat",
        "player",
        "proxy",
        "representative",
        "seed",
        "team_id",
        "team_name",
    }
)

Key = tuple[str, str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate --output without executing any game.",
    )
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dataset_core_sha256(dataset: dict[str, Any]) -> str:
    """Hash semantic dataset content while excluding its self-referential proof."""

    core = copy.deepcopy(dataset)
    core.pop("determinism", None)
    return value_sha256(core)


def _audit_key(value: dict[str, Any]) -> Key:
    return (
        str(value["family"]),
        str(value["representative"]),
        int(value["seed"]),
        int(value["candidate_seat"]),
    )


def _expected_keys() -> set[Key]:
    return {
        (family, representative, seed, seat)
        for family, representative, _ in FAMILIES
        for seed in SEEDS
        for seat in SEATS
    }


def _load_v84_labels() -> tuple[dict[Key, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(V84_RESULTS_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "pass_action_support_only"
    assert payload["gates"]["mechanical_pass"] is True
    assert payload["gates"]["economic_action_support_pass"] is True

    index: dict[Key, dict[str, Any]] = {}
    for game in payload["games"]:
        key = _audit_key(game["key"])
        assert key not in index, f"duplicate V84 key: {key}"
        outcomes = {row["option"]: row for row in game["outcomes"]}
        assert set(outcomes) == set(OPTIONS), f"V84 option mismatch: {key}"
        abstain = outcomes["ABSTAIN_KEEP_V67"]
        assert abstain["candidate_sanitizer"]["changed_steps"] == 0
        index[key] = {
            "all_option_margins": {
                option: float(outcomes[option]["candidate_margin"]) for option in OPTIONS
            },
            "abstain_prefix_sha256": abstain["trace_sha256"]["candidate_prefix_0_119_proposed"],
        }

    assert set(index) == _expected_keys()
    return index, payload


def _collect_key(
    *,
    family: str,
    representative: str,
    proxy_path: Path,
    seed: int,
    candidate_seat: int,
    v84_label: dict[str, Any],
) -> dict[str, Any]:
    # Exact V84 load order: opponent first, then a fresh V67 standalone.
    _, proxy_agent, proxy_accepts_configuration = load_agent(proxy_path)
    _, v67_agent, v67_accepts_configuration = load_agent(V67_PATH)

    configuration = {**CONFIG_BASE, "seed": seed}
    engine = Engine(configuration)
    snapshots: dict[int, list[float]] = {}
    candidate_proposed_prefix: list[dict[str, Any]] = []
    candidate_sanitizer_changed_steps: list[int] = []

    for step in range(BRANCH_STEP):
        assert int(engine.state["step"]) == step
        actions: list[dict[str, Any]] = []
        for player in range(2):
            observation = engine.observation(player)
            assert int(observation["step"]) == step
            if player == candidate_seat:
                if step in FEATURE_STEPS:
                    encoded = encode_selector_features(observation)
                    assert encoded.shape == (SNAPSHOT_FEATURES,)
                    snapshot = [float(value) for value in encoded.tolist()]
                    assert all(math.isfinite(value) for value in snapshot)
                    snapshots[step] = snapshot
                raw = run_agent(
                    v67_agent,
                    v67_accepts_configuration,
                    observation,
                    configuration,
                )
                proposed = normalize_legacy_action(raw)
                sanitized = engine.sanitize_action(player, proposed)
                candidate_proposed_prefix.append(copy.deepcopy(proposed))
                if proposed != sanitized:
                    candidate_sanitizer_changed_steps.append(step)
            else:
                raw = run_agent(
                    proxy_agent,
                    proxy_accepts_configuration,
                    observation,
                    configuration,
                )
                proposed = normalize_legacy_action(raw)
                sanitized = engine.sanitize_action(player, proposed)
            actions.append(sanitized)
        engine.step(actions)

    final_observation = engine.observation(candidate_seat)
    final_observed_step = int(final_observation["step"])
    assert int(engine.state["step"]) == BRANCH_STEP
    assert final_observed_step == BRANCH_STEP
    encoded = encode_selector_features(final_observation)
    assert encoded.shape == (SNAPSHOT_FEATURES,)
    final_snapshot = [float(value) for value in encoded.tolist()]
    assert all(math.isfinite(value) for value in final_snapshot)
    snapshots[BRANCH_STEP] = final_snapshot

    assert tuple(sorted(snapshots)) == FEATURE_STEPS
    assert len(candidate_proposed_prefix) == BRANCH_STEP
    assert not candidate_sanitizer_changed_steps
    prefix_sha256 = value_sha256(candidate_proposed_prefix)
    assert prefix_sha256 == v84_label["abstain_prefix_sha256"], (
        f"V67 prefix parity failed for {family=} {seed=} {candidate_seat=}: "
        f"{prefix_sha256} != {v84_label['abstain_prefix_sha256']}"
    )

    features = [value for step in FEATURE_STEPS for value in snapshots[step]]
    assert len(features) == MODEL_FEATURES
    assert all(type(value) is float and math.isfinite(value) for value in features)
    all_option_margins = dict(v84_label["all_option_margins"])

    return {
        "audit": {
            "candidate_sanitizer_changed_steps": 0,
            "candidate_seat": candidate_seat,
            "family": family,
            "final_observed_step": final_observed_step,
            "observed_steps": list(FEATURE_STEPS),
            "opponent_seat": 1 - candidate_seat,
            "prefix_sha256": prefix_sha256,
            "representative": representative,
            "seed": seed,
            "v84_abstain_prefix_sha256": v84_label["abstain_prefix_sha256"],
        },
        "labels": {
            "all_option_margins": all_option_margins,
            "model_margins": {
                model_action: all_option_margins[v84_option]
                for model_action, v84_option in MODEL_ACTION_TO_V84.items()
            },
        },
        "model_input": {"features": features},
    }


def _walk_field_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.add(str(key).lower())
            names.update(_walk_field_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_walk_field_names(child))
    return names


def validate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    samples = dataset["samples"]
    assert len(samples) == 32
    keys = [_audit_key(sample["audit"]) for sample in samples]
    assert len(set(keys)) == 32
    assert set(keys) == _expected_keys()
    assert keys == sorted(keys), "dataset rows must have canonical key order"

    prefix_matches = 0
    finite_feature_values = 0
    for sample in samples:
        audit = sample["audit"]
        assert set(sample["model_input"]) == {"features"}
        forbidden_fields = _walk_field_names(sample["model_input"]) & FORBIDDEN_MODEL_INPUT_FIELDS
        assert not forbidden_fields, f"forbidden model-input fields: {sorted(forbidden_fields)}"

        features = sample["model_input"]["features"]
        assert isinstance(features, list) and len(features) == MODEL_FEATURES
        assert all(type(value) in (int, float) for value in features)
        assert all(
            not isinstance(value, bool) and math.isfinite(float(value)) for value in features
        )
        finite_feature_values += len(features)

        assert audit["observed_steps"] == list(FEATURE_STEPS)
        assert audit["final_observed_step"] == BRANCH_STEP
        assert audit["candidate_sanitizer_changed_steps"] == 0
        assert audit["opponent_seat"] == 1 - audit["candidate_seat"]
        assert audit["prefix_sha256"] == audit["v84_abstain_prefix_sha256"]
        prefix_matches += 1

        all_margins = sample["labels"]["all_option_margins"]
        model_margins = sample["labels"]["model_margins"]
        assert set(all_margins) == set(OPTIONS)
        assert set(model_margins) == set(MODEL_ACTION_TO_V84)
        assert all(
            type(value) in (int, float) and math.isfinite(float(value))
            for value in all_margins.values()
        )
        assert model_margins == {
            model_action: all_margins[v84_option]
            for model_action, v84_option in MODEL_ACTION_TO_V84.items()
        }

    return {
        "candidate_sanitizer_zero_keys": sum(
            sample["audit"]["candidate_sanitizer_changed_steps"] == 0 for sample in samples
        ),
        "expected_keys": 32,
        "feature_dimension": MODEL_FEATURES,
        "finite_feature_values": finite_feature_values,
        "forbidden_model_input_fields_present": [],
        "prefix_hash_matches": prefix_matches,
        "unique_keys": len(set(keys)),
    }


def _source_hashes(v84_payload: dict[str, Any]) -> dict[str, str]:
    paths = {
        "collect_features.py": COLLECTOR_PATH,
        "feature_encoder": ENCODER_PATH,
        "frozen_v67": V67_PATH,
        "evaluation_helper": HELPER_PATH,
        "rust_python_extension": Path(kaggriculture_rs.__file__),
        "v84_results": V84_RESULTS_PATH,
    }
    for _, representative, proxy_path in FAMILIES:
        paths[f"opponent_{representative}"] = proxy_path
    hashes = {name: sha256(path) for name, path in sorted(paths.items())}

    v84_sources = v84_payload["sources"]
    assert hashes["frozen_v67"] == v84_sources["frozen_v67"]
    assert hashes["evaluation_helper"] == v84_sources["evaluation_helper"]
    assert hashes["rust_python_extension"] == v84_sources["rust_python_extension"]
    for _, representative, _ in FAMILIES:
        assert hashes[f"opponent_{representative}"] == v84_sources[f"opponent_{representative}"]
    return hashes


def build_dataset() -> dict[str, Any]:
    v84_index, v84_payload = _load_v84_labels()
    proxy_by_key = {
        (family, representative): proxy_path for family, representative, proxy_path in FAMILIES
    }

    samples: list[dict[str, Any]] = []
    for key in sorted(_expected_keys()):
        family, representative, seed, candidate_seat = key
        samples.append(
            _collect_key(
                family=family,
                representative=representative,
                proxy_path=proxy_by_key[(family, representative)],
                seed=seed,
                candidate_seat=candidate_seat,
                v84_label=v84_index[key],
            )
        )

    dataset: dict[str, Any] = {
        "guardrails": {
            "candidate_created": False,
            "checkpoint_created": False,
            "development_only": True,
            "live_kaggle_traces_opened": [],
            "replay_actions_or_rewards_opened": [],
            "sealed_inputs_opened": [],
            "submission_created": False,
            "training_performed": False,
            "upload_performed": False,
        },
        "protocol": {
            "action_execution": (
                "normalize_legacy_action then Engine.sanitize_action then Engine.step"
            ),
            "branch_step": BRANCH_STEP,
            "candidate_seats": list(SEATS),
            "expected_keys": 32,
            "feature_encoder": "encode_selector_features",
            "feature_steps": list(FEATURE_STEPS),
            "fresh_load_order_each_key": ["proxy", "v67"],
            "label_source": "frozen V84 on-policy option margins",
            "model_action_to_v84": MODEL_ACTION_TO_V84,
            "model_feature_dimension": MODEL_FEATURES,
            "model_input_excludes_audit_identifiers": True,
            "options": list(OPTIONS),
            "representatives": [representative for _, representative, _ in FAMILIES],
            "seeds": list(SEEDS),
            "snapshot_feature_dimension": SNAPSHOT_FEATURES,
        },
        "samples": samples,
        "schema_version": 1,
        "sources": _source_hashes(v84_payload),
        "status": "prefix_dataset_complete",
        "v84_label_core_sha256": v84_payload["determinism"]["core_sha256"],
    }
    dataset["validation"] = validate_dataset(dataset)
    dataset["determinism"] = {"core_sha256": dataset_core_sha256(dataset)}
    return dataset


def main() -> None:
    args = parse_args()
    if args.validate_existing:
        dataset = json.loads(args.output.read_text(encoding="utf-8"))
        validation = validate_dataset(dataset)
        core_sha256 = dataset_core_sha256(dataset)
        assert dataset["validation"] == validation
        assert dataset["determinism"]["core_sha256"] == core_sha256
        print(
            json.dumps(
                {"core_sha256": core_sha256, "status": "valid", **validation},
                sort_keys=True,
            )
        )
        return

    dataset = build_dataset()
    write_json(args.output, dataset)
    print(
        json.dumps(
            {
                "core_sha256": dataset["determinism"]["core_sha256"],
                "dataset_sha256": sha256(args.output),
                "output": str(args.output),
                "status": dataset["status"],
                **dataset["validation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
