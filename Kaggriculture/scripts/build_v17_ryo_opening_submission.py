"""Build V17 Candidate A: Ryo opening plus an earlier first land purchase."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
DEFAULT_REPLAY = ROOT / "analysis/top1_ryo_20260823_current/97619249.json"
DEFAULT_MANIFEST = ROOT / "analysis/top1_ryo_20260823_current/manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "submissions/meta_v17_ryo_opening_land120_standalone"
DEFAULT_PREFIX_STEPS = 73  # Decision steps 0..72, inclusive.
DEFAULT_LAND_STEP = 120  # Appears in replay frame 121.
DEFAULT_VERIFY_EPISODES = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_actions(replay_path: Path, seat: int, count: int) -> list[dict[str, Any]]:
    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    if not 0 <= count <= 719:
        raise ValueError("prefix_steps must be within 0..719")

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    steps = replay.get("steps")
    if not isinstance(steps, list) or len(steps) <= count:
        raise ValueError(f"replay has fewer than {count + 1} frames: {replay_path}")

    actions: list[dict[str, Any]] = []
    for decision_step in range(count):
        # Kaggle records the action chosen from observation t in frame t + 1.
        frame = steps[decision_step + 1]
        if not isinstance(frame, list) or len(frame) <= seat:
            raise ValueError(f"missing seat {seat} at replay frame {decision_step + 1}")
        action = frame[seat].get("action")
        if not isinstance(action, dict):
            raise ValueError(f"missing action for decision step {decision_step}")
        actions.append(action)
    return actions


def _verify_common_prefix(
    actions: list[dict[str, Any]],
    manifest_path: Path | None,
    verification_episodes: int,
) -> list[int]:
    if manifest_path is None or verification_episodes <= 0:
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) < verification_episodes:
        raise ValueError(
            f"manifest has fewer than {verification_episodes} episodes: {manifest_path}"
        )

    verified: list[int] = []
    for row in rows[:verification_episodes]:
        episode_id = int(row["episode_id"])
        seat = int(row["top1_seat"])
        replay_path = manifest_path.parent / f"{episode_id}.json"
        candidate = _load_actions(replay_path, seat, len(actions))
        if candidate != actions:
            first_difference = next(
                index
                for index, (left, right) in enumerate(zip(actions, candidate, strict=True))
                if left != right
            )
            raise ValueError(f"episode {episode_id} diverges at decision step {first_difference}")
        verified.append(episode_id)
    return verified


def _encoded_actions(actions: list[dict[str, Any]]) -> str:
    payload = json.dumps(actions, separators=(",", ":"), sort_keys=True).encode()
    return base64.b85encode(zlib.compress(payload, level=9)).decode("ascii")


def _overlay_source(
    actions: list[dict[str, Any]],
    land_step: int,
    land_funding_wheat: int,
) -> str:
    encoded = _encoded_actions(actions)
    return f"""\n\n# V17 Candidate A: verified Ryo opening and earlier first land purchase.
_V17_OPENING = json.loads(
    zlib.decompress(base64.b85decode({encoded!r})).decode("utf-8")
)
_V17_LAND_STEP = {land_step}
_V17_LAND_FUNDING_WHEAT = {land_funding_wheat}
_V17_ORIGINAL_LAND_STEP = next(
    (
        index
        for index, action in enumerate(_ACTIONS)
        if any(order and order[0] == "BUY_LAND" for order in action.get("market", []))
    ),
    None,
)

for _v17_step, _v17_action in enumerate(_V17_OPENING):
    _ACTIONS[_v17_step] = _v17_action

if _V17_ORIGINAL_LAND_STEP is not None and _V17_ORIGINAL_LAND_STEP != _V17_LAND_STEP:
    _v17_old_market = _ACTIONS[_V17_ORIGINAL_LAND_STEP].get("market", [])
    _ACTIONS[_V17_ORIGINAL_LAND_STEP]["market"] = [
        order for order in _v17_old_market if not (order and order[0] == "BUY_LAND")
    ]

_v17_land_market = [
    order
    for order in _ACTIONS[_V17_LAND_STEP].get("market", [])
    if not (order and order[0] == "BUY_LAND")
]
if _V17_LAND_FUNDING_WHEAT > 0:
    _v17_land_market = [
        order
        for order in _v17_land_market
        if not (len(order) >= 2 and order[0] == "SELL" and order[1] == "WHEAT")
    ]
    _v17_land_market.insert(0, ["SELL", "WHEAT", _V17_LAND_FUNDING_WHEAT])
_v17_after_sales = next(
    (
        index
        for index, order in enumerate(_v17_land_market)
        if not (order and order[0] == "SELL")
    ),
    len(_v17_land_market),
)
_v17_land_market.insert(_v17_after_sales, ["BUY_LAND"])
_ACTIONS[_V17_LAND_STEP]["market"] = _v17_land_market[:10]
"""


def build(
    base: Path,
    replay: Path,
    output_dir: Path,
    *,
    replay_seat: int = 0,
    prefix_steps: int = DEFAULT_PREFIX_STEPS,
    land_step: int = DEFAULT_LAND_STEP,
    land_funding_wheat: int = 0,
    verification_manifest: Path | None = DEFAULT_MANIFEST,
    verification_episodes: int = DEFAULT_VERIFY_EPISODES,
) -> dict[str, Any]:
    if not 0 <= land_step < 720:
        raise ValueError("land_step must be within 0..719")
    if land_funding_wheat < 0:
        raise ValueError("land_funding_wheat must be non-negative")

    actions = _load_actions(replay, replay_seat, prefix_steps)
    verified_episode_ids = _verify_common_prefix(
        actions,
        verification_manifest,
        verification_episodes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "main.py"
    source = "# ruff: noqa\n" + base.read_text(encoding="utf-8").rstrip()
    output.write_text(
        source + _overlay_source(actions, land_step, land_funding_wheat),
        encoding="utf-8",
    )

    action_payload = json.dumps(actions, separators=(",", ":"), sort_keys=True).encode()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "name": output_dir.name,
        "agent_file": _display_path(output),
        "sha256": _sha256(output),
        "base": {
            "agent_file": _display_path(base),
            "sha256": _sha256(base),
        },
        "opening": {
            "reference_replay": _display_path(replay),
            "reference_replay_sha256": _sha256(replay),
            "reference_seat": replay_seat,
            "decision_steps": [0, prefix_steps - 1] if prefix_steps else [],
            "action_count": prefix_steps,
            "actions_sha256": hashlib.sha256(action_payload).hexdigest(),
            "verified_episode_ids": verified_episode_ids,
        },
        "land": {
            "decision_step": land_step,
            "replay_frame": land_step + 1,
            "replaces_base_first_land": True,
            "funding_wheat_sale": land_funding_wheat,
        },
    }
    (output_dir / "candidate.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--replay-seat", type=int, default=0)
    parser.add_argument("--prefix-steps", type=int, default=DEFAULT_PREFIX_STEPS)
    parser.add_argument("--land-step", type=int, default=DEFAULT_LAND_STEP)
    parser.add_argument("--land-funding-wheat", type=int, default=0)
    parser.add_argument("--verification-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verification-episodes", type=int, default=DEFAULT_VERIFY_EPISODES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metadata = build(
        args.base,
        args.replay,
        args.output_dir,
        replay_seat=args.replay_seat,
        prefix_steps=args.prefix_steps,
        land_step=args.land_step,
        land_funding_wheat=args.land_funding_wheat,
        verification_manifest=args.verification_manifest,
        verification_episodes=args.verification_episodes,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
