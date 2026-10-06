"""Write immutable candidate provenance before opening a sealed replay holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from kaggriculture_rl.evaluation.top1_goal import sha256_path, target_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, action="append", required=True)
    parser.add_argument("--selection-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def episode_ids(paths: list[Path]) -> list[int]:
    values: set[int] = set()
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        values.update(int(row["episode_id"]) for row in manifest["episodes"])
    return sorted(values)


def main() -> None:
    args = parse_args()
    if not args.candidate.is_file():
        raise FileNotFoundError(f"candidate does not exist: {args.candidate}")
    target_manifest = json.loads(
        args.target_manifest.read_text(encoding="utf-8")
    )
    target = target_identity(target_manifest)
    if target is None:
        raise ValueError("target manifest does not contain a complete Kaggle identity")
    frozen_at = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": 2,
        "created_at": frozen_at,
        "candidate_frozen_at": frozen_at,
        "candidate_file": str(args.candidate),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "candidate_size_bytes": args.candidate.stat().st_size,
        "target": target,
        "target_manifest": str(args.target_manifest),
        "target_manifest_sha256": sha256_path(args.target_manifest),
        "training_episode_ids": episode_ids(args.training_manifest),
        "model_selection_episode_ids": episode_ids(args.selection_manifest),
        "training_manifests": [str(path) for path in args.training_manifest],
        "model_selection_manifests": [
            str(path) for path in args.selection_manifest
        ],
        "training_manifest_sha256s": {
            str(path): sha256_path(path) for path in args.training_manifest
        },
        "model_selection_manifest_sha256s": {
            str(path): sha256_path(path) for path in args.selection_manifest
        },
        "holdout_used_for_training": False,
        "holdout_used_for_model_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_sha256": payload["candidate_sha256"],
                "training_count": len(payload["training_episode_ids"]),
                "model_selection_count": len(payload["model_selection_episode_ids"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
