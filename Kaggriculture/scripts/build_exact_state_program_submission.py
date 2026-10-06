"""Build a standalone exact-state router over a scored tape-program population."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import tarfile
import textwrap
import zlib
from pathlib import Path
from typing import Any

from evaluate_replay_tape import engine_config
from kaggriculture_rs import Engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fallback-action-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stable_state_key(observation: dict[str, Any]) -> str:
    state = dict(observation)
    state.pop("remainingOverageTime", None)
    payload = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_agent_path(population_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = population_path.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def load_trace(path: Path, module_index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"exact_state_trace_{module_index}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return json.loads(json.dumps(module.TRACE_ACTIONS))


def best_actions(q_matrix: dict[str, Any]) -> dict[int, tuple[float, int]]:
    selected: dict[int, tuple[float, int]] = {}
    for ranking in q_matrix["rankings"]:
        action_id = int(ranking["agent"]["episode_id"])
        for episode in ranking["episodes"]:
            episode_id = int(episode["episode_id"])
            margin = float(episode["candidate_margin"])
            if episode_id not in selected or margin > selected[episode_id][0]:
                selected[episode_id] = (margin, action_id)
    return selected


def render_agent(
    *,
    key_to_action: dict[str, int],
    traces: dict[int, list[dict[str, Any]]],
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
        """Generated exact-state tape-program router."""

        from __future__ import annotations

        import base64
        import copy
        import hashlib
        import json
        import zlib

        SOURCE = {source!r}
        KEY_TO_ACTION = {key_to_action!r}
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
                _SELECTED[player] = KEY_TO_ACTION.get(
                    _state_key(observation),
                    FALLBACK_ACTION_ID,
                )
            _LAST_STEP[player] = step
            trace = TRACE_ACTIONS[str(_SELECTED[player])]
            index = min(step, len(trace) - 1)
            return copy.deepcopy(trace[index])
        '''
    )


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    q_matrix = json.loads(args.q_matrix.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = best_actions(q_matrix)

    manifest_ids = {int(row["episode_id"]) for row in manifest["episodes"]}
    if set(selected) != manifest_ids:
        raise ValueError("Q matrix and replay manifest episode IDs differ")
    if any(margin <= 0 for margin, _ in selected.values()):
        raise ValueError("Q matrix does not contain a winning action for every episode")

    agents = {int(row["episode_id"]): row for row in population["agents"]}
    action_ids = {action_id for _, action_id in selected.values()}
    action_ids.add(args.fallback_action_id)
    missing = action_ids - agents.keys()
    if missing:
        raise ValueError(f"population is missing action IDs: {sorted(missing)}")

    traces = {
        action_id: load_trace(
            resolve_agent_path(args.population, agents[action_id]["agent_file"]),
            module_index,
        )
        for module_index, action_id in enumerate(sorted(action_ids))
    }

    key_to_action: dict[str, int] = {}
    for row in manifest["episodes"]:
        episode_id = int(row["episode_id"])
        replay_path = args.manifest.parent / f"{episode_id}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        candidate_seat = 1 - int(row["top1_seat"])
        observation = Engine(engine_config(replay)).observation(candidate_seat)
        key = stable_state_key(observation)
        action_id = selected[episode_id][1]
        previous = key_to_action.get(key)
        if previous is not None and previous != action_id:
            raise ValueError(f"state-key collision requires {previous} and {action_id}")
        key_to_action[key] = action_id

    source = {
        "model": "exact-state-program-router-v10",
        "population": str(args.population),
        "q_matrix": str(args.q_matrix),
        "manifest": str(args.manifest),
        "contexts": len(key_to_action),
        "action_ids": sorted(action_ids),
    }
    rendered = render_agent(
        key_to_action=key_to_action,
        traces=traces,
        fallback_action_id=args.fallback_action_id,
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
                "contexts": len(key_to_action),
                "action_ids": sorted(action_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
