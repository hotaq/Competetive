"""Freeze an exact-validated replay manifest as an immutable goal artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from kaggriculture_rl.evaluation.top1_goal import (
    parse_timestamp,
    sha256_path,
    target_identity,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--integrity-report", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable holdout: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    integrity = json.loads(args.integrity_report.read_text(encoding="utf-8"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    episode_ids = [int(row["episode_id"]) for row in episodes]
    if len(episodes) != 24 or len(set(episode_ids)) != 24:
        raise ValueError("holdout must contain exactly 24 unique episodes")
    if not integrity.get("validation", {}).get("all_original_replays_exact"):
        raise ValueError("integrity report did not reproduce every original replay")
    if {int(row["episode_id"]) for row in integrity["episodes"]} != set(episode_ids):
        raise ValueError("integrity report episode IDs do not match manifest")

    target = target_identity(manifest)
    if target is None or target != target_identity(provenance):
        raise ValueError("holdout target does not match frozen candidate target")
    candidate_path = Path(str(provenance.get("candidate_file", "")))
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    candidate_hash = provenance.get("candidate_sha256")
    if not candidate_path.is_file() or sha256_path(candidate_path) != candidate_hash:
        raise ValueError("frozen candidate file hash no longer matches provenance")
    if integrity.get("candidate", {}).get("sha256") != candidate_hash:
        raise ValueError("integrity report was not produced by the frozen candidate")
    candidate_frozen_at = parse_timestamp(provenance.get("candidate_frozen_at"))
    manifest_created_at = parse_timestamp(manifest.get("created_at"))
    if (
        candidate_frozen_at is None
        or manifest_created_at is None
        or candidate_frozen_at > manifest_created_at
    ):
        raise ValueError("candidate must be frozen before holdout collection")

    frozen = {
        **manifest,
        "schema_version": 2,
        "immutable": True,
        "exact_validated": True,
        "frozen_at": datetime.now(UTC).isoformat(),
        "target": target,
        "candidate_sha256": candidate_hash,
        "provenance": str(args.provenance),
        "provenance_sha256": sha256_path(args.provenance),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256(args.manifest),
        "integrity_report": str(args.integrity_report),
        "integrity_report_sha256": sha256(args.integrity_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(episodes),
                "episode_ids_sha256": hashlib.sha256(
                    json.dumps(sorted(episode_ids), separators=(",", ":")).encode()
                ).hexdigest(),
                "source_manifest_sha256": frozen["source_manifest_sha256"],
                "integrity_report_sha256": frozen["integrity_report_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
