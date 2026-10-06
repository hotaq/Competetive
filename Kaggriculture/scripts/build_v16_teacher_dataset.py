"""Build the existing spatial-GRU dataset schema from the V16 live teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.imitation.teacher_dataset import (
    DEFAULT_CROP_OPPONENTS,
    build_live_teacher_spatial_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py"
DEFAULT_V12 = (
    ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py"
)
DEFAULT_OUTPUT = ROOT / "rl/datasets/v16_rc5_teacher_spatial_v1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--holdout-episodes", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=610_000)
    parser.add_argument("--file-opponent", type=Path, action="append")
    parser.add_argument("--no-v12-opponent", action="store_true")
    parser.add_argument("--no-self-play", action="store_true")
    parser.add_argument("--behavior-checkpoint", type=Path)
    parser.add_argument("--behavior-confidence-threshold", type=float, default=1.0)
    parser.add_argument("--behavior-start-episode", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_opponents = list(args.file_opponent or [])
    if not args.no_v12_opponent:
        file_opponents.append(DEFAULT_V12)
    report = build_live_teacher_spatial_dataset(
        args.teacher,
        args.output,
        episodes=args.episodes,
        holdout_episodes=args.holdout_episodes,
        base_seed=args.base_seed,
        file_opponents=tuple(file_opponents),
        crop_opponents=DEFAULT_CROP_OPPONENTS,
        include_self_play=not args.no_self_play,
        behavior_checkpoint=args.behavior_checkpoint,
        behavior_confidence_threshold=args.behavior_confidence_threshold,
        behavior_start_episode=args.behavior_start_episode,
    )
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
