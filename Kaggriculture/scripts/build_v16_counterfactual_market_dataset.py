"""Build paired single-deviation labels for V16 market residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.counterfactual_residual import (
    build_counterfactual_market_dataset,
    build_leave_one_out_market_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPATIAL = ROOT / "rl/checkpoints/v16_rc5_spatial_gru_bc_v1_e100.npz"
DEFAULT_TEACHER = ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py"
DEFAULT_V12 = (
    ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py"
)
DEFAULT_OUTPUT = ROOT / "rl/datasets/v16_rc5_market_counterfactual_melon_v1.npz"
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-checkpoint", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paired-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=708_000)
    parser.add_argument("--minimum-confidence", type=float, default=0.99)
    parser.add_argument("--max-candidates-per-game", type=int, default=16)
    parser.add_argument("--holdout-mod", type=int, default=4)
    parser.add_argument("--holdout-fold", type=int, default=3)
    parser.add_argument("--opponent-crop", choices=CROPS, default="MELON")
    parser.add_argument("--opponent-plots", type=int, default=6)
    parser.add_argument("--all-crops", action="store_true")
    parser.add_argument("--include-v12", action="store_true")
    parser.add_argument(
        "--v12-only",
        action="store_true",
        help="Use only the live V12 standalone opponent (no crop heuristic).",
    )
    parser.add_argument("--allow-hire-research", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("single", "leave-one-out"),
        default="single",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.v12_only:
        args.include_v12 = True
    crop_opponents = (
        tuple((crop, args.opponent_plots) for crop in CROPS)
        if args.all_crops
        else ((args.opponent_crop, args.opponent_plots),)
    )
    if args.v12_only:
        crop_opponents = ()
    file_opponents = (DEFAULT_V12,) if args.include_v12 else ()
    builder = (
        build_leave_one_out_market_dataset
        if args.mode == "leave-one-out"
        else build_counterfactual_market_dataset
    )
    if args.mode == "leave-one-out" and args.allow_hire_research:
        raise ValueError("leave-one-out mode does not allow HIRE research")
    builder_kwargs = {
        "episodes": args.paired_seeds * 2,
        "base_seed": args.base_seed,
        "minimum_confidence": args.minimum_confidence,
        "max_candidates_per_game": args.max_candidates_per_game,
        "holdout_mod": args.holdout_mod,
        "holdout_fold": args.holdout_fold,
        "crop_opponents": crop_opponents,
        "file_opponents": file_opponents,
    }
    if args.mode == "single":
        builder_kwargs["allow_hire_research"] = args.allow_hire_research
    report = builder(
        args.spatial_checkpoint,
        args.teacher,
        args.output,
        **builder_kwargs,
    )
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
