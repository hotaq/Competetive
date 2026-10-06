"""Build a standalone observable staged router over tape programs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
import textwrap
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_exact_state_program_submission import (
    load_trace,
    resolve_agent_path,
    stable_state_key,
)
from evaluate_replay_tape import engine_config, frame_action
from kaggriculture_rs import Engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--router-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def action_id_from_path(path: str) -> int:
    return int(Path(path).stem.rsplit("_", maxsplit=1)[-1])


def score_matrix(report: dict[str, Any]) -> dict[int, dict[int, float]]:
    scores: dict[int, dict[int, float]] = defaultdict(dict)
    for configuration in report["configurations"]:
        action_id = action_id_from_path(configuration["suffix_agent"])
        for episode in configuration["episodes"]:
            scores[int(episode["episode_id"])][action_id] = float(
                episode["candidate_margin"]
            )
    return dict(scores)


def render_agent(
    *,
    traces: dict[int, list[dict[str, Any]]],
    base_action_id: int,
    early_stages: list[dict[str, Any]],
    branch_step: int,
    final_routes: dict[str, int],
    fallback_action_id: int,
    source: dict[str, Any],
) -> str:
    payload = base64.b85encode(
        zlib.compress(
            json.dumps(traces, separators=(",", ":")).encode("utf-8"),
            level=9,
        )
    ).decode("ascii")
    return textwrap.dedent(
        f'''\
        """Generated observable staged tape-program router."""

        from __future__ import annotations

        import base64
        import copy
        import hashlib
        import json
        import zlib

        SOURCE = {source!r}
        BASE_ACTION_ID = {base_action_id!r}
        EARLY_STAGES = {early_stages!r}
        BRANCH_STEP = {branch_step!r}
        FINAL_ROUTES = {final_routes!r}
        FALLBACK_ACTION_ID = {fallback_action_id!r}
        _PAYLOAD = {payload!r}
        TRACE_ACTIONS = json.loads(zlib.decompress(base64.b85decode(_PAYLOAD)))
        _SELECTED = {{}}
        _LAST_STEP = {{}}


        def _state_key(observation):
            state = dict(observation)
            state.pop("remainingOverageTime", None)
            payload = json.dumps(
                state,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()


        def agent(observation, configuration=None):
            del configuration
            player = int(observation.get("player", 0))
            step = int(observation.get("step", 0) or 0)
            previous = _LAST_STEP.get(player, -1)
            if step == 0 or step <= previous or player not in _SELECTED:
                _SELECTED[player] = BASE_ACTION_ID
            for stage in EARLY_STAGES:
                if step == int(stage["step"]):
                    _SELECTED[player] = stage["routes"].get(
                        _state_key(observation),
                        BASE_ACTION_ID,
                    )
            if step == BRANCH_STEP:
                _SELECTED[player] = FINAL_ROUTES.get(
                    _state_key(observation),
                    FALLBACK_ACTION_ID,
                )
            _LAST_STEP[player] = step
            trace = TRACE_ACTIONS[str(_SELECTED[player])]
            return copy.deepcopy(trace[min(step, len(trace) - 1)])
        '''
    )


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    report = json.loads(args.q_matrix.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    router = json.loads(args.router_config.read_text(encoding="utf-8"))

    agents = {int(row["episode_id"]): row for row in population["agents"]}
    base_action_id = int(router["base_action_id"])
    fallback_action_id = int(router.get("fallback_action_id", base_action_id))
    branch_step = int(router["branch_step"])
    early_stages = [
        {
            "step": int(stage["step"]),
            "routes": {key: int(value) for key, value in stage["routes"].items()},
        }
        for stage in router.get("early_stages", [])
    ]

    required_prefix_ids = {base_action_id, fallback_action_id}
    for stage in early_stages:
        required_prefix_ids.update(stage["routes"].values())
    missing = required_prefix_ids - agents.keys()
    if missing:
        raise ValueError(f"population is missing prefix action IDs: {sorted(missing)}")

    traces = {
        action_id: load_trace(
            resolve_agent_path(args.population, agents[action_id]["agent_file"]),
            module_index,
        )
        for module_index, action_id in enumerate(sorted(agents))
    }

    branch_groups: dict[str, list[int]] = defaultdict(list)
    for row in manifest["episodes"]:
        episode_id = int(row["episode_id"])
        replay = json.loads(
            (args.manifest.parent / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        candidate_seat = 1 - int(row["top1_seat"])
        engine = Engine(engine_config(replay))
        selected = base_action_id
        for frame in replay["steps"][1:]:
            observation = engine.observation(candidate_seat)
            step = int(observation["step"])
            for stage in early_stages:
                if step == stage["step"]:
                    selected = stage["routes"].get(
                        stable_state_key(observation), base_action_id
                    )
            if step == branch_step:
                branch_groups[stable_state_key(observation)].append(episode_id)
                break
            actions = [frame_action(frame, 0), frame_action(frame, 1)]
            actions[candidate_seat] = engine.sanitize_action(
                candidate_seat,
                traces[selected][min(step, len(traces[selected]) - 1)],
            )
            engine.step(actions)
        else:
            raise ValueError(f"episode {episode_id} never reached step {branch_step}")

    scores = score_matrix(report)
    manifest_ids = {int(row["episode_id"]) for row in manifest["episodes"]}
    if set(scores) != manifest_ids:
        raise ValueError("Q matrix and replay manifest episode IDs differ")

    final_routes: dict[str, int] = {}
    route_margins: dict[str, float] = {}
    for key, episode_ids in branch_groups.items():
        common_actions = set.intersection(*(set(scores[eid]) for eid in episode_ids))
        candidates = [
            (
                min(scores[eid][action_id] for eid in episode_ids),
                sum(scores[eid][action_id] for eid in episode_ids) / len(episode_ids),
                action_id,
            )
            for action_id in common_actions
        ]
        worst_margin, _, action_id = max(candidates)
        if worst_margin <= 0:
            raise ValueError(
                f"state group {episode_ids} has no winning continuation; "
                f"best margin={worst_margin}"
            )
        final_routes[key] = action_id
        route_margins[key] = worst_margin

    embedded_action_ids = {base_action_id, fallback_action_id, *final_routes.values()}
    for stage in early_stages:
        embedded_action_ids.update(stage["routes"].values())
    embedded_traces = {
        action_id: traces[action_id] for action_id in sorted(embedded_action_ids)
    }
    source = {
        "model": "observable-staged-program-router-v10",
        "population": str(args.population),
        "q_matrix": str(args.q_matrix),
        "manifest": str(args.manifest),
        "router_config": str(args.router_config),
        "branch_step": branch_step,
        "contexts": len(final_routes),
        "action_ids": sorted(embedded_action_ids),
        "worst_route_margin": min(route_margins.values()),
    }
    rendered = render_agent(
        traces=embedded_traces,
        base_action_id=base_action_id,
        early_stages=early_stages,
        branch_step=branch_step,
        final_routes=final_routes,
        fallback_action_id=fallback_action_id,
        source=source,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.output_dir / "main.py"
    main_path.write_text(rendered, encoding="utf-8")
    archive_path = args.output_dir / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")

    print(
        json.dumps(
            {
                "main": str(main_path),
                "main_bytes": main_path.stat().st_size,
                "main_sha256": hashlib.sha256(main_path.read_bytes()).hexdigest(),
                "archive": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "contexts": len(final_routes),
                "action_ids": sorted(embedded_action_ids),
                "worst_route_margin": min(route_margins.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
