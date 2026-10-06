"""Build compact self-contained agents from non-holdout replay action tapes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import textwrap
import uuid
import zlib
from pathlib import Path
from typing import Any

from evaluate_replay_tape import frame_action, validate_original_replay

AGENT_TEMPLATE = '''"""Compact replay-tape agent generated from public training data."""

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
    parser.add_argument(
        "--manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-replay-validation",
        action="store_true",
        help=(
            "Extract recorded actions without replaying them in the current engine. "
            "Use only for historical replays affected by engine drift."
        ),
    )
    return parser.parse_args()


def trace_actions(replay: dict[str, Any], seat: int) -> list[dict[str, Any]]:
    actions = [frame_action(frame, seat) for frame in replay["steps"][1:]]
    if not actions:
        raise ValueError("replay has no aligned actions")
    actions.append({"farmer": ["PASS"], "hands": [], "market": []})
    return actions


def render_agent(
    actions: list[dict[str, Any]], source: dict[str, Any]
) -> str:
    encoded = base64.b85encode(
        zlib.compress(
            json.dumps(actions, separators=(",", ":")).encode("utf-8"),
            level=9,
        )
    ).decode("ascii")
    return AGENT_TEMPLATE.format(
        source=repr(source),
        payload=encoded,
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite replay population: {args.output_dir}"
        )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output_dir.with_name(
        f".{args.output_dir.name}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    records: list[dict[str, Any]] = []
    source_submissions: list[dict[str, Any]] = []
    seen_episode_seats: set[tuple[int, int]] = set()

    try:
        for manifest_path in args.manifest:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_submissions.append(manifest["submission"])
            for row in manifest["episodes"]:
                episode_id = int(row["episode_id"])
                top1_seat = int(
                    row["top1_seat"]
                    if "top1_seat" in row
                    else row["agent_seat"]
                )
                episode_seat = (episode_id, top1_seat)
                if episode_seat in seen_episode_seats:
                    raise ValueError(
                        "duplicate population episode seat: "
                        f"episode={episode_id} seat={top1_seat}"
                    )
                seen_episode_seats.add(episode_seat)
                replay_path = manifest_path.parent / f"{episode_id}.json"
                raw = replay_path.read_bytes()
                raw_hash = hashlib.sha256(raw).hexdigest()
                if raw_hash != row["sha256"]:
                    raise ValueError(f"replay hash mismatch: {episode_id}")
                replay = json.loads(raw)
                validation_status = "exact_current_engine"
                if args.skip_replay_validation:
                    validation_status = "skipped_historical_engine_drift"
                else:
                    transitions, rewards = validate_original_replay(replay)
                    if transitions != len(replay["steps"]) - 1:
                        raise ValueError(f"transition mismatch: {episode_id}")
                    if rewards != tuple(
                        float(value) for value in replay["rewards"]
                    ):
                        raise ValueError(f"reward mismatch: {episode_id}")
                source = {
                    "competition": manifest["competition"],
                    "submission_id": manifest["submission"]["submission_id"],
                    "episode_id": episode_id,
                    "seed": int(replay["info"]["seed"]),
                    "top1_seat": top1_seat,
                    "source_sha256": raw_hash,
                    "replay_validation": validation_status,
                }
                rendered = textwrap.dedent(
                    render_agent(trace_actions(replay, top1_seat), source)
                )
                final_output = (
                    args.output_dir / f"tape_{episode_id}_s{top1_seat}.py"
                )
                (staging / final_output.name).write_text(rendered, encoding="utf-8")
                records.append(
                    {
                        **source,
                        "agent_file": str(final_output),
                        "agent_sha256": hashlib.sha256(
                            rendered.encode()
                        ).hexdigest(),
                    }
                )

        population_manifest = {
            "schema_version": 2,
            "source_manifests": [str(path) for path in args.manifest],
            "source_submissions": source_submissions,
            "holdout": False,
            "exact_validated": not args.skip_replay_validation,
            "replay_validation": (
                "skipped_historical_engine_drift"
                if args.skip_replay_validation
                else "exact_current_engine"
            ),
            "agents": records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(population_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
        raise
    print(json.dumps(population_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
