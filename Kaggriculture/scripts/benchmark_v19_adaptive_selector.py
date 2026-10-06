"""Audit step-zero V19 choices and latency across an immutable manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from kaggriculture_rs import Engine

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_file_agent_vs_replay import load_agent  # noqa: E402
from evaluate_replay_tape import engine_config  # noqa: E402


def benchmark(
    agent_file: Path, manifest_path: Path
) -> dict[str, Any]:
    module, agent, accepts_configuration = load_agent(agent_file)
    policy = getattr(module, "_V19_POLICY", None)
    if policy is None or not hasattr(policy, "diagnostics_for"):
        raise TypeError("agent does not expose _V19_POLICY diagnostics")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = []
    for row in manifest["episodes"]:
        episode_id = int(row["episode_id"])
        candidate_seat = 1 - int(row["top1_seat"])
        replay = json.loads(
            (manifest_path.parent / f"{episode_id}.json").read_text(
                encoding="utf-8"
            )
        )
        configuration = engine_config(replay)
        observation = Engine(configuration).observation(candidate_seat)
        policy.reset()
        started = time.perf_counter()
        if accepts_configuration:
            agent(observation, configuration)
        else:
            agent(observation)
        wall_ms = (time.perf_counter() - started) * 1_000.0
        diagnostics = policy.diagnostics_for(candidate_seat)
        if diagnostics is None:
            raise RuntimeError(f"missing diagnostics for episode {episode_id}")
        episodes.append(
            {
                "episode_id": episode_id,
                "candidate_seat": candidate_seat,
                "policy": diagnostics["policy"],
                "reason": diagnostics["reason"],
                "model_elapsed_ms": diagnostics["elapsed_ms"],
                "wall_ms": wall_ms,
                "predicted_values": diagnostics.get("predicted_values"),
                "conservative_values": diagnostics.get(
                    "conservative_values"
                ),
            }
        )

    latencies = np.asarray([row["wall_ms"] for row in episodes])
    return {
        "schema_version": 1,
        "agent_file": str(agent_file),
        "manifest": str(manifest_path),
        "episodes": episodes,
        "summary": {
            "games": len(episodes),
            "choice_counts": {
                policy_name: sum(
                    row["policy"] == policy_name for row in episodes
                )
                for policy_name in ("v16", "v12")
            },
            "mean_ms": fmean(latencies),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "max_ms": float(np.max(latencies)),
            "under_1000": int(np.sum(latencies < 1_000.0)),
            "fallbacks": sum(
                row["reason"] != "rollout-selector" for row in episodes
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_file", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.agent_file, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
