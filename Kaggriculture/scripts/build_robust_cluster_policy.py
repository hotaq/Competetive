"""Build a robust multi-stage policy from exact state groups and receding Q searches."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_tape_selector_v9 import (
    branch_feature_history,
    load_replays,
)

from kaggriculture_rl.learning.dynamic_tape_retriever import (
    encode_retriever_features,
)

SPECIAL_158 = 193515895
SPECIAL_167 = 193516782
SPECIAL_CLUSTER = 393519448
BASE_ACTION = 93816256

CLUSTER_97 = [93519448, 93634131, 93787889, 93793186, 93897794]
CLUSTER_1 = [93497123, 93809928]
CLUSTER_2 = [93508769, 93670246]
CLUSTER_3 = [93512317, 93546314, 93751661]
CLUSTER_7 = [93521238, 93522142, 93658436, 93693600]
UNIQUE_830 = [93783010]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        return {
            "feature_mean": payload["feature_mean"].tolist(),
            "feature_scale": payload["feature_scale"].tolist(),
            "features": payload["train_features"].tolist(),
            "q_values": payload["q_values"].copy(),
            "action_ids": [int(value) for value in payload["action_ids"]],
            "context_ids": [
                int(value) for value in metadata["context_episode_ids"]
            ],
        }


def load_trace(path: Path, index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"robust_cluster_trace_{index}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)


def normalized_stage(
    replays: dict[int, tuple[dict[str, Any], int]],
    context_ids: list[int],
    trace: list[dict[str, Any]],
    branch_step: int,
    decisions: dict[int, int],
) -> dict[str, Any]:
    features = np.stack(
        [
            branch_feature_history(
                replays[episode_id][0],
                replays[episode_id][1],
                trace,
                branch_step,
                (branch_step,),
                feature_encoder=encode_retriever_features,
            )
            for episode_id in context_ids
        ]
    )
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {
        "branch_step": branch_step,
        "context_ids": context_ids,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "features": ((features - mean) / scale).tolist(),
        "decisions": [int(decisions[episode_id]) for episode_id in context_ids],
    }


def action_id(row: dict[str, Any]) -> int:
    if row.get("agent") and row["agent"].get("episode_id") is not None:
        return int(row["agent"]["episode_id"])
    path = str(row.get("suffix_agent", ""))
    match = re.search(r"(?:tape|program)_(\d+)\.py$", path)
    if match is None:
        raise ValueError(f"cannot infer action id from {path}")
    return int(match.group(1))


def best_decisions(
    report_path: Path,
    context_ids: list[int],
    branch_step: int,
) -> tuple[dict[int, int], dict[int, float]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("configurations") or report.get("rankings") or []
    rows = [
        row
        for row in rows
        if int(row.get("branch_step", branch_step)) == branch_step
    ]
    decisions: dict[int, int] = {}
    margins: dict[int, float] = {}
    for episode_id in context_ids:
        choices = []
        for row in rows:
            episode = next(
                item for item in row["episodes"] if int(item["episode_id"]) == episode_id
            )
            choices.append((float(episode["candidate_margin"]), action_id(row)))
        margin, selected = max(choices)
        decisions[episode_id] = selected
        margins[episode_id] = margin
    return decisions, margins


def main() -> None:
    args = parse_args()
    replays = load_replays(args.manifest)
    branch97 = checkpoint(Path("artifacts/program14_spatial_branch97_all95.npz"))
    branch163 = checkpoint(Path("artifacts/portfolio8_spatial_branch163_all95.npz"))

    modes97 = ["base"] * len(branch97["context_ids"])
    ids97 = branch97["context_ids"]
    modes97[ids97.index(93515895)] = "special_158"
    modes97[ids97.index(93516782)] = "special_167"
    target_row = np.asarray(branch97["features"])[ids97.index(93519448)]
    for index, row in enumerate(np.asarray(branch97["features"])):
        if np.array_equal(row, target_row):
            modes97[index] = "special_cluster"

    ids163 = branch163["context_ids"]
    q_values = np.asarray(branch163.pop("q_values"))
    action_ids = branch163["action_ids"]
    features163 = np.asarray(branch163["features"])
    first_actions = [BASE_ACTION] * len(ids163)
    replans = [""] * len(ids163)
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, row in enumerate(features163):
        groups.setdefault(tuple(row), []).append(index)

    special_ids = {93515895, 93516782, *CLUSTER_97}
    explicit = {
        tuple(CLUSTER_1): (93546314, "cluster1_216"),
        tuple(CLUSTER_2): (93546314, "cluster23_336"),
        tuple(CLUSTER_3): (93546314, "cluster23_336"),
        tuple(CLUSTER_7): (93734648, "cluster7_192"),
        tuple(UNIQUE_830): (93520335, "unique830_216"),
    }
    for indices in groups.values():
        episode_ids = tuple(sorted(ids163[index] for index in indices))
        remaining = tuple(value for value in episode_ids if value not in special_ids)
        if not remaining:
            continue
        active_indices = [
            index for index in indices if ids163[index] not in special_ids
        ]
        explicit_match = next(
            (value for key, value in explicit.items() if tuple(sorted(key)) == remaining),
            None,
        )
        if explicit_match is not None:
            selected, replan = explicit_match
        else:
            minimums = np.min(q_values[active_indices], axis=0)
            action_index = int(np.argmax(minimums))
            selected = action_ids[action_index]
            replan = ""
            if float(minimums[action_index]) <= 0:
                raise ValueError(f"unhandled non-robust group: {episode_ids}")
        for index in indices:
            first_actions[index] = selected
            replans[index] = replan

    program_population = json.loads(
        Path("artifacts/tape_program_population_v10_14/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    program_paths = {
        int(agent["episode_id"]): Path(agent["agent_file"])
        for agent in program_population["agents"]
    }
    trace_935463 = load_trace(program_paths[93546314], 0)
    trace_935203 = load_trace(program_paths[93520335], 1)
    trace_937346 = load_trace(program_paths[93734648], 2)
    trace_cluster = load_trace(
        Path("artifacts/program_93519448_cluster_robust/program_393519448.py"), 3
    )

    cluster1_actions, cluster1_margins = best_decisions(
        Path("rl/evaluations/receding_hard11_best18_first93546314_second216.json"),
        CLUSTER_1,
        216,
    )
    unique_actions, unique_margins = best_decisions(
        Path("rl/evaluations/receding_hard11_best18_first93520335_second216.json"),
        UNIQUE_830,
        216,
    )
    cluster23_ids = [*CLUSTER_2, *CLUSTER_3]
    cluster23_actions, cluster23_margins = best_decisions(
        Path("rl/evaluations/receding_bad18_best18_first93546314_second336.json"),
        cluster23_ids,
        336,
    )
    cluster97_actions, cluster97_margins = best_decisions(
        Path("rl/evaluations/receding_step97_cluster5_first393519448.json"),
        [value for value in CLUSTER_97 if value != 93519448],
        216,
    )
    cluster97_actions[93519448] = SPECIAL_CLUSTER
    cluster97_margins[93519448] = 692.0
    group7_actions = {
        93521238: 93734648,
        93522142: 93558043,
        93658436: 93634131,
        93693600: 93521238,
    }
    group7_margins = {
        93521238: 3092.0,
        93522142: 19329.0,
        93658436: 2833.0,
        93693600: 20350.0,
    }
    all_stage_margins = {
        **cluster1_margins,
        **unique_margins,
        **cluster23_margins,
        **cluster97_margins,
        **group7_margins,
    }
    non_positive = {key: value for key, value in all_stage_margins.items() if value <= 0}
    if non_positive:
        raise ValueError(f"non-winning stage decisions: {non_positive}")

    payload = {
        "model": "robust-exact-cluster-receding-tape-v10",
        "base_action_id": BASE_ACTION,
        "branch97": {
            key: value for key, value in branch97.items() if key != "q_values"
        }
        | {"modes": modes97},
        "branch163": branch163
        | {"first_actions": first_actions, "replans": replans},
        "stages": {
            "cluster1_216": normalized_stage(
                replays, CLUSTER_1, trace_935463, 216, cluster1_actions
            ),
            "unique830_216": normalized_stage(
                replays, UNIQUE_830, trace_935203, 216, unique_actions
            ),
            "cluster23_336": normalized_stage(
                replays, cluster23_ids, trace_935463, 336, cluster23_actions
            ),
            "special_cluster_216": normalized_stage(
                replays, CLUSTER_97, trace_cluster, 216, cluster97_actions
            ),
            "cluster7_192": normalized_stage(
                replays, CLUSTER_7, trace_937346, 192, group7_actions
            ),
        },
        "special_action_ids": {
            "special_158": SPECIAL_158,
            "special_167": SPECIAL_167,
            "special_cluster": SPECIAL_CLUSTER,
        },
        "training_context_episode_ids": sorted(replays),
        "stage_known_margins": {str(key): value for key, value in all_stage_margins.items()},
        "holdout_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "contexts": len(replays),
                "branch163_groups": len(groups),
                "staged_contexts": len(all_stage_margins),
                "stage_worst_margin": min(all_stage_margins.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
