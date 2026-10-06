"""Build compact multi-stage replay-tape programs from a source population."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import textwrap
import zlib
from pathlib import Path
from typing import Any

AGENT_TEMPLATE = '''"""Generated multi-stage replay-tape program."""

from __future__ import annotations

import base64
import copy
import json
import zlib

SOURCE = {source}
_PAYLOAD = {payload!r}
TRACE_ACTIONS = json.loads(zlib.decompress(base64.b85decode(_PAYLOAD)))


def agent(observation, configuration=None):
    del configuration
    step = min(int(observation.get("step", 0) or 0), len(TRACE_ACTIONS) - 1)
    return copy.deepcopy(TRACE_ACTIONS[step])
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--programs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"trace_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return copy_actions(module.TRACE_ACTIONS)


def copy_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(actions))


def resolve_path(population_path: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    sibling = population_path.parent / path.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(path)


def build_program(
    spec: dict[str, Any], traces: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    segments = sorted(spec["segments"], key=lambda row: int(row["start_step"]))
    if int(segments[0]["start_step"]) != 0:
        raise ValueError(f"program {spec['program_id']} must start at step zero")
    length = max(len(trace) for trace in traces.values())
    actions = []
    segment_index = 0
    for step in range(length):
        while (
            segment_index + 1 < len(segments)
            and int(segments[segment_index + 1]["start_step"]) <= step
        ):
            segment_index += 1
        source_id = int(segments[segment_index]["source_episode_id"])
        trace = traces[source_id]
        actions.append(copy_actions([trace[min(step, len(trace) - 1)]])[0])

    for mutation in spec.get("mutations", []):
        kind = mutation["kind"]
        step = int(mutation["step"])
        order_index = int(mutation["order_index"])
        order = actions[step]["market"][order_index]
        expected = mutation.get("expected")
        if expected is not None and order != expected:
            raise ValueError(
                f"program {spec['program_id']} mutation expected {expected}, got {order}"
            )
        if kind == "remove_market_order":
            actions[step]["market"].pop(order_index)
        elif kind == "replace_market_quantity":
            if len(order) < 3:
                raise ValueError(
                    f"program {spec['program_id']} mutation requires a quantity: {order}"
                )
            order[2] = int(mutation["quantity"])
        else:
            raise ValueError(f"unknown mutation kind: {kind}")
    return actions


def render_agent(actions: list[dict[str, Any]], source: dict[str, Any]) -> str:
    payload = base64.b85encode(
        zlib.compress(
            json.dumps(actions, separators=(",", ":")).encode("utf-8"), level=9
        )
    ).decode("ascii")
    return textwrap.dedent(AGENT_TEMPLATE).format(source=repr(source), payload=payload)


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    program_config = json.loads(args.programs.read_text(encoding="utf-8"))
    paths = {
        int(row["episode_id"]): resolve_path(args.population, row["agent_file"])
        for row in population["agents"]
    }
    required_ids = {
        int(segment["source_episode_id"])
        for program in program_config["programs"]
        for segment in program["segments"]
    }
    missing = sorted(required_ids - paths.keys())
    if missing:
        raise ValueError(f"source traces are missing: {missing}")
    traces = {episode_id: load_trace(paths[episode_id]) for episode_id in required_ids}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for program in program_config["programs"]:
        actions = build_program(program, traces)
        source = {
            "program_id": int(program["program_id"]),
            "name": program["name"],
            "segments": program["segments"],
            "mutations": program.get("mutations", []),
        }
        rendered = render_agent(actions, source)
        path = args.output_dir / f"program_{int(program['program_id'])}.py"
        compile(rendered, str(path), "exec")
        path.write_text(rendered, encoding="utf-8")
        records.append(
            {
                "episode_id": int(program["program_id"]),
                "agent_file": str(path),
                "agent_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "program": source,
            }
        )

    manifest = {
        "agents": records,
        "holdout": False,
        "source_population": str(args.population),
        "program_config": str(args.programs),
        "common_prefix_step": int(program_config["common_prefix_step"]),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
