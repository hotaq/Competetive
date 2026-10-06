"""Smoke-test the exact archive through Kaggle's raw Python-file loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
EXPECTED_ACTION_KEYS = {"farmer", "hands", "market"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_active(action: dict[str, Any]) -> bool:
    farmer = action.get("farmer", ["PASS"])
    hands = action.get("hands", [])
    market = action.get("market", [])
    farmer_active = bool(farmer and farmer[0] != "PASS")
    hands_active = any(command and command[0] != "PASS" for command in hands)
    return farmer_active or hands_active or bool(market)


def smoke(
    archive_path: Path,
    reference_path: Path,
    *,
    seed: int,
    candidate_seat: int,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    reference_path = reference_path.resolve()
    if candidate_seat not in (0, 1):
        raise ValueError("candidate_seat must be 0 or 1")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != ["main.py"]:
            raise ValueError("submission archive must contain only main.py")
        member = members[0]
        if not member.isfile():
            raise ValueError("main.py must be a regular file")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("could not read main.py from archive")
        main_bytes = extracted.read()

    with tempfile.TemporaryDirectory(prefix="kaggriculture-raw-smoke-") as temp_dir:
        candidate_path = Path(temp_dir) / "main.py"
        candidate_path.write_bytes(main_bytes)

        environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
        agents = [str(reference_path), str(reference_path)]
        agents[candidate_seat] = str(candidate_path)
        environment.run(agents)

        statuses = [str(player.status) for player in environment.state]
        candidate_actions = [
            step[candidate_seat].action
            for step in environment.steps[1:]
            if isinstance(step[candidate_seat].action, dict)
        ]
        invalid_key_steps = [
            index + 1
            for index, action in enumerate(candidate_actions)
            if set(action) != EXPECTED_ACTION_KEYS
        ]
        active_steps = sum(_is_active(action) for action in candidate_actions)
        report = {
            "archive": str(archive_path),
            "archive_sha256": _sha256(archive_path.read_bytes()),
            "main_sha256": _sha256(main_bytes),
            "reference": str(reference_path),
            "seed": seed,
            "candidate_seat": candidate_seat,
            "statuses": statuses,
            "rewards": [float(player.reward) for player in environment.state],
            "environment_steps": len(environment.steps),
            "candidate_action_steps": len(candidate_actions),
            "candidate_active_steps": active_steps,
            "invalid_action_key_steps": invalid_key_steps,
        }

    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"raw-path run did not finish: {statuses}")
    if invalid_key_steps:
        raise RuntimeError(
            f"raw loader returned invalid action keys at steps {invalid_key_steps[:8]}"
        )
    if active_steps == 0:
        raise RuntimeError("candidate produced no active actions")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--seed", type=int, default=8_262_026)
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = smoke(
        args.archive,
        args.reference,
        seed=args.seed,
        candidate_seat=args.candidate_seat,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
