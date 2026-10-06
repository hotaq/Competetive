#!/usr/bin/env python3
"""Select Top20 families from public tape actions and preexisting snapshot ranks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SELECTOR_PATH = Path(__file__).resolve()
TAPE_ROOT = ROOT / "artifacts/top20_live_tapes80_20260826"
TAPE_MANIFEST_PATH = TAPE_ROOT / "manifest.json"
RANK_MANIFEST_PATH = ROOT / "replays/top20_live_20260826/manifest.json"

EXPECTED_TAPE_MANIFEST_SHA256 = "d3b17f42fa2839e0e86a25f29b96bb5661e739d4abd82154c3b3818a6a6c2947"
EXPECTED_RANK_MANIFEST_SHA256 = "1253ef13a3bea8bfd8e741185d27c178b296e3aafde08832972dd5dd16934c57"
EXPECTED_SOURCE_SUBMISSIONS = 20
EXPECTED_TAPES_PER_SUBMISSION = 4
EXPECTED_CORE = (55714246, 55734995, 55774239, 55742738)
EXPECTED_BACKUPS = (55616096, 55617399)

# Ryo is already represented by the frozen V84 ``ryo_selector_family``. It is
# removed before farthest-first selection so it cannot consume a new-family slot.
EXCLUDED_SUBMISSIONS = {55614463: "already represented by frozen V84 ryo_selector_family"}

PHASE_WIDTH = 120
PHASE_COUNT = 6
EPISODE_STEPS = PHASE_WIDTH * PHASE_COUNT
OPERATIONS = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
    "DROP",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "HIRE",
    "BUY_LAND",
    "OTHER",
)
OPERATION_INDEX = {name: index for index, name in enumerate(OPERATIONS)}
FINGERPRINT_DIM = PHASE_COUNT * len(OPERATIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "family_selection.json",
        help="Destination JSON path.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_pinned_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise AssertionError(
            f"source manifest drift: {path.relative_to(ROOT)} "
            f"has {observed}, expected {expected_sha256}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"manifest must be a mapping: {path.relative_to(ROOT)}")
    return value


def resolve_pinned_tape(row: dict[str, Any]) -> Path:
    relative = Path(str(row["agent_file"]))
    if relative.is_absolute():
        raise AssertionError(f"absolute tape path is forbidden: {relative}")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(TAPE_ROOT.resolve())
    except ValueError as error:
        raise AssertionError(f"tape escapes pinned artifact root: {relative}") from error
    expected = str(row["agent_sha256"])
    observed = sha256_file(path)
    if observed != expected:
        raise AssertionError(f"tape hash drift: {relative} has {observed}, expected {expected}")
    return path


def load_fresh_tape(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Load one hash-pinned tape in a fresh, non-cached module namespace."""

    path = resolve_pinned_tape(row)
    module_key = hashlib.sha256(f"{path}:{row['agent_sha256']}".encode()).hexdigest()[:20]
    module_name = f"_v86_family_tape_{module_key}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path.relative_to(ROOT)}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    # Do not register the module in sys.modules: each call gets a fresh namespace.
    spec.loader.exec_module(module)
    actions = getattr(module, "TRACE_ACTIONS", None)
    if not isinstance(actions, list) or len(actions) < EPISODE_STEPS:
        raise AssertionError(
            f"{path.relative_to(ROOT)} must expose at least {EPISODE_STEPS} TRACE_ACTIONS"
        )
    selected = actions[:EPISODE_STEPS]
    if not all(isinstance(action, dict) for action in selected):
        raise AssertionError(f"non-mapping action in {path.relative_to(ROOT)}")
    return selected


def action_orders(action: dict[str, Any]) -> list[Any]:
    farmer = action.get("farmer") or ["PASS"]
    hands = action.get("hands") or []
    market = action.get("market") or []
    if not isinstance(hands, list) or not isinstance(market, list):
        raise AssertionError("hands and market must be lists")
    return [farmer, *hands, *market]


def operation_name(order: Any) -> str:
    if not isinstance(order, (list, tuple)) or not order:
        return "PASS"
    name = str(order[0])
    return name if name in OPERATION_INDEX else "OTHER"


def fingerprint_from_tapes(tapes: list[list[dict[str, Any]]]) -> list[float]:
    """Pool raw operation counts across tapes, then normalize each phase."""

    counts = [[0] * len(OPERATIONS) for _ in range(PHASE_COUNT)]
    for actions in tapes:
        for step, action in enumerate(actions[:EPISODE_STEPS]):
            phase = min(step // PHASE_WIDTH, PHASE_COUNT - 1)
            for order in action_orders(action):
                counts[phase][OPERATION_INDEX[operation_name(order)]] += 1

    fingerprint: list[float] = []
    for phase_counts in counts:
        total = sum(phase_counts)
        if total <= 0:
            raise AssertionError("operation phase unexpectedly has no actions")
        fingerprint.extend(value / total for value in phase_counts)
    if len(fingerprint) != FINGERPRINT_DIM:
        raise AssertionError(f"fingerprint length {len(fingerprint)} != {FINGERPRINT_DIM}")
    if not all(math.isfinite(value) for value in fingerprint):
        raise AssertionError("fingerprint contains non-finite values")
    return fingerprint


def mean_phase_total_variation(left: list[float], right: list[float]) -> float:
    if len(left) != FINGERPRINT_DIM or len(right) != FINGERPRINT_DIM:
        raise AssertionError("distance inputs have the wrong dimension")
    phase_distances = []
    for phase in range(PHASE_COUNT):
        offset = phase * len(OPERATIONS)
        phase_distances.append(
            0.5
            * sum(
                abs(left[offset + index] - right[offset + index])
                for index in range(len(OPERATIONS))
            )
        )
    return sum(phase_distances) / PHASE_COUNT


def build_rank_metadata(rank_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    teams = rank_manifest.get("teams")
    if not isinstance(teams, list) or len(teams) != EXPECTED_SOURCE_SUBMISSIONS:
        raise AssertionError("rank manifest must contain exactly 20 teams")
    metadata: dict[int, dict[str, Any]] = {}
    for team in teams:
        submission = team.get("submission")
        if not isinstance(submission, dict):
            raise AssertionError("team submission metadata is missing")
        submission_id = int(submission["submission_id"])
        record = {
            "snapshot_rank": int(team["snapshot_rank"]),
            "team_name": str(team["team_name"]),
        }
        if submission_id in metadata:
            raise AssertionError(f"duplicate ranked submission {submission_id}")
        metadata[submission_id] = record
    ranks = sorted(record["snapshot_rank"] for record in metadata.values())
    if ranks != list(range(1, EXPECTED_SOURCE_SUBMISSIONS + 1)):
        raise AssertionError("snapshot ranks must be exactly 1..20")
    return metadata


def group_tape_rows(
    tape_manifest: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    if tape_manifest.get("exact_validated") is not True:
        raise AssertionError("tape manifest is not exact-validated")
    if tape_manifest.get("holdout") is not False:
        raise AssertionError("selector may use only the non-holdout tape manifest")
    rows = tape_manifest.get("agents")
    if not isinstance(rows, list):
        raise AssertionError("tape manifest agents must be a list")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise AssertionError("tape manifest agent must be a mapping")
        if row.get("replay_validation") != "exact_current_engine":
            raise AssertionError("all candidate tapes must be exact-current-engine")
        grouped[int(row["submission_id"])].append(row)
    if len(grouped) != EXPECTED_SOURCE_SUBMISSIONS:
        raise AssertionError("tape manifest must contain exactly 20 submissions")
    for submission_id, submission_rows in grouped.items():
        if len(submission_rows) != EXPECTED_TAPES_PER_SUBMISSION:
            raise AssertionError(
                f"submission {submission_id} has {len(submission_rows)} tapes, expected 4"
            )
    return dict(grouped)


def select_medoid(
    rows: list[dict[str, Any]],
    tape_actions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    individual = {
        str(row["agent_file"]): fingerprint_from_tapes([tape_actions[str(row["agent_file"])]])
        for row in rows
    }
    candidates = []
    for row in rows:
        path = str(row["agent_file"])
        distances = [
            mean_phase_total_variation(individual[path], individual[str(other["agent_file"])])
            for other in rows
            if other is not row
        ]
        mean_distance = sum(distances) / len(distances)
        candidates.append(
            (
                mean_distance,
                int(row["episode_id"]),
                int(row["top1_seat"]),
                row,
                individual[path],
            )
        )
    mean_distance, _, _, row, fingerprint = min(
        candidates, key=lambda value: (value[0], value[1], value[2])
    )
    return {
        "agent_file": str(row["agent_file"]),
        "agent_sha256": str(row["agent_sha256"]),
        "episode_id": int(row["episode_id"]),
        "seat": int(row["top1_seat"]),
        "mean_distance_to_other_three": mean_distance,
        "fingerprint_sha256": canonical_sha256(fingerprint),
    }


def farthest_first(
    eligible: list[int],
    fingerprints: dict[int, list[float]],
    ranks: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_candidates = sorted(
        eligible, key=lambda submission_id: ranks[submission_id]["snapshot_rank"]
    )
    selected = [ordered_candidates[0]]
    records = [
        {
            "order": 1,
            "submission_id": selected[0],
            "selection_min_distance": None,
            "nearest_selected_submission_id": None,
        }
    ]
    while len(selected) < len(ordered_candidates):
        remaining = [
            submission_id for submission_id in ordered_candidates if submission_id not in selected
        ]
        scored: dict[int, tuple[float, int]] = {}
        for submission_id in remaining:
            nearest = min(
                selected,
                key=lambda chosen: (
                    mean_phase_total_variation(fingerprints[submission_id], fingerprints[chosen]),
                    ranks[chosen]["snapshot_rank"],
                    chosen,
                ),
            )
            scored[submission_id] = (
                mean_phase_total_variation(fingerprints[submission_id], fingerprints[nearest]),
                nearest,
            )
        chosen = max(
            remaining,
            key=lambda submission_id: (
                scored[submission_id][0],
                -ranks[submission_id]["snapshot_rank"],
            ),
        )
        selected.append(chosen)
        records.append(
            {
                "order": len(selected),
                "submission_id": chosen,
                "selection_min_distance": scored[chosen][0],
                "nearest_selected_submission_id": scored[chosen][1],
            }
        )
    return records


def build_output() -> dict[str, Any]:
    tape_manifest = read_pinned_json(TAPE_MANIFEST_PATH, EXPECTED_TAPE_MANIFEST_SHA256)
    rank_manifest = read_pinned_json(RANK_MANIFEST_PATH, EXPECTED_RANK_MANIFEST_SHA256)
    grouped = group_tape_rows(tape_manifest)
    ranks = build_rank_metadata(rank_manifest)
    if set(grouped) != set(ranks):
        raise AssertionError("tape and rank manifests disagree on submission IDs")
    if not set(EXCLUDED_SUBMISSIONS).issubset(grouped):
        raise AssertionError("predeclared exclusion is absent from candidate pool")

    tape_actions: dict[str, list[dict[str, Any]]] = {}
    for rows in grouped.values():
        for row in rows:
            relative = str(row["agent_file"])
            tape_actions[relative] = load_fresh_tape(row)

    fingerprints = {
        submission_id: fingerprint_from_tapes(
            [tape_actions[str(row["agent_file"])] for row in rows]
        )
        for submission_id, rows in grouped.items()
    }
    eligible = sorted(set(grouped) - set(EXCLUDED_SUBMISSIONS))
    order = farthest_first(eligible, fingerprints, ranks)
    selected_ids = tuple(record["submission_id"] for record in order)
    core = selected_ids[: len(EXPECTED_CORE)]
    backups = selected_ids[len(EXPECTED_CORE) : len(EXPECTED_CORE) + len(EXPECTED_BACKUPS)]
    if core != EXPECTED_CORE:
        raise AssertionError(f"core selection drift: {core} != {EXPECTED_CORE}")
    if backups != EXPECTED_BACKUPS:
        raise AssertionError(f"backup selection drift: {backups} != {EXPECTED_BACKUPS}")

    order_by_id = {record["submission_id"]: record for record in order}
    family_records = []
    for submission_id in eligible:
        family_records.append(
            {
                "submission_id": submission_id,
                **ranks[submission_id],
                "farthest_first": order_by_id[submission_id],
                "family_fingerprint": fingerprints[submission_id],
                "family_fingerprint_sha256": canonical_sha256(fingerprints[submission_id]),
                "medoid": select_medoid(grouped[submission_id], tape_actions),
                "source_tapes": [
                    {
                        "agent_file": str(row["agent_file"]),
                        "agent_sha256": str(row["agent_sha256"]),
                        "episode_id": int(row["episode_id"]),
                        "seat": int(row["top1_seat"]),
                    }
                    for row in sorted(
                        grouped[submission_id],
                        key=lambda value: (
                            int(value["episode_id"]),
                            int(value["top1_seat"]),
                        ),
                    )
                ],
            }
        )

    core_pairwise = []
    for left_index, left in enumerate(core):
        for right in core[left_index + 1 :]:
            core_pairwise.append(
                {
                    "left_submission_id": left,
                    "right_submission_id": right,
                    "distance": mean_phase_total_variation(fingerprints[left], fingerprints[right]),
                }
            )

    return {
        "schema_version": 1,
        "status": "pre_registered_selection_only",
        "sources": {
            "tape_manifest": {
                "path": str(TAPE_MANIFEST_PATH.relative_to(ROOT)),
                "sha256": EXPECTED_TAPE_MANIFEST_SHA256,
                "exact_validated": True,
                "holdout": False,
            },
            "rank_manifest": {
                "path": str(RANK_MANIFEST_PATH.relative_to(ROOT)),
                "sha256": EXPECTED_RANK_MANIFEST_SHA256,
                "fields_read": ["submission.submission_id", "snapshot_rank", "team_name"],
            },
            "selector": {
                "path": str(SELECTOR_PATH.relative_to(ROOT)),
                "sha256": sha256_file(SELECTOR_PATH),
            },
        },
        "protocol": {
            "candidate_pool": "20 exact-validated Top20 submissions, four tapes each",
            "excluded_before_selection": [
                {"submission_id": submission_id, "reason": reason}
                for submission_id, reason in sorted(EXCLUDED_SUBMISSIONS.items())
            ],
            "episode_steps": EPISODE_STEPS,
            "phase_width": PHASE_WIDTH,
            "phase_count": PHASE_COUNT,
            "operations": list(OPERATIONS),
            "fingerprint_dimension": FINGERPRINT_DIM,
            "fingerprint": (
                "pool raw farmer+all-hands+all-market operation counts over four "
                "tapes, then normalize each 120-step phase"
            ),
            "distance": "mean over six phases of 0.5 * L1 (total variation)",
            "selection": (
                "anchor best snapshot rank; repeatedly maximize minimum distance "
                "to selected; exact-distance ties prefer better snapshot rank"
            ),
            "medoid": (
                "per-tape normalized fingerprint minimizing mean total-variation "
                "distance to the other three; ties by episode_id then seat"
            ),
        },
        "selection": {
            "core_submission_ids": list(core),
            "backup_submission_ids": list(backups),
            "full_farthest_first_order": order,
            "core_pairwise_distances": core_pairwise,
            "core_pairwise_min_distance": min(record["distance"] for record in core_pairwise),
        },
        "families": family_records,
        "guardrails": {
            "preexisting_snapshot_rank_read": True,
            "public_tape_actions_read": True,
            "v86_candidate_economic_outcomes_read": False,
            "rewards_read": False,
            "v67_or_route_margins_read": False,
            "sealed_holdout_opened": False,
            "games_executed": False,
        },
    }


def main() -> None:
    sys.dont_write_bytecode = True
    args = parse_args()
    output = build_output()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
