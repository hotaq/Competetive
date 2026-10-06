#!/usr/bin/env python3
"""Audit public RSNA report-label sources against the 58 official labels.

Only small CSV files are read. Competition DICOM data is never downloaded.
The audit deliberately reports fixed, equal-weight consensus candidates rather
than tuning target weights on the tiny gold set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
UID = "StudyInstanceUID"


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=columns)
    frame[UID] = frame[UID].astype(str)
    if frame[UID].duplicated().any():
        raise ValueError(f"duplicate {UID} values in {path}")
    return frame.set_index(UID)


def load_sources(root: Path) -> dict[str, pd.DataFrame]:
    plain = [UID, *LABELS]
    sources = {
        "pilkwang_v2": _read(root / "pilkwang/report_labels_v2.csv", plain),
        "lixin_sol56": _read(root / "lixin/report_labels_gpt56sol.csv", plain),
        "steven_full": _read(root / "steven/llm_labels_full.csv", plain),
        "steven_v2": _read(root / "steven/llm_labels_v2.csv", plain),
        "steven_v4": _read(root / "steven/llm_labels_v4_blend.csv", plain),
    }

    barun = _read(
        root / "barun/train_folds_with_pseudo.csv",
        [UID, *[f"pseudo_{label}" for label in LABELS]],
    )
    barun.columns = LABELS
    # This dataset currently declares license="other". Keep it in the audit,
    # but do not include it in the default training-label consensus.
    sources["barun_pseudo_audit_only"] = barun
    return sources


def aligned(sources: dict[str, pd.DataFrame], ids: pd.Index) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name, frame in sources.items():
        missing = ids.difference(frame.index)
        if len(missing) > 1:
            raise ValueError(f"{name} misses too many competition studies: {len(missing)}")
        values = frame.reindex(ids)[LABELS].astype(float)
        finite_values = values.to_numpy()[np.isfinite(values.to_numpy())]
        if ((finite_values < 0) | (finite_values > 1)).any():
            raise ValueError(f"{name} contains labels outside [0, 1]")
        result[name] = values
    return result


def rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=0, method="average", pct=True)


def make_candidates(sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    core_names = ["pilkwang_v2", "lixin_sol56", "steven_full"]
    cc0_names = ["pilkwang_v2", "lixin_sol56", "steven_full", "steven_v2", "steven_v4"]

    def mean(names: list[str], ranks: bool) -> pd.DataFrame:
        frames = [rank_frame(sources[name]) if ranks else sources[name] for name in names]
        values = np.nanmean(np.stack([frame.to_numpy() for frame in frames]), axis=0)
        return pd.DataFrame(values, index=frames[0].index, columns=LABELS)

    candidates = dict(sources)
    candidates["core_equal_raw"] = mean(core_names, ranks=False)
    candidates["core_equal_rank"] = mean(core_names, ranks=True)
    candidates["cc0_equal_raw"] = mean(cc0_names, ranks=False)
    candidates["cc0_equal_rank"] = mean(cc0_names, ranks=True)

    # The public label study predeclared Synovitis as the only safe
    # not-addressed imputation. Preserve the fixed core consensus elsewhere.
    targeted_raw = candidates["core_equal_raw"].copy()
    targeted_raw["Synovitis"] = sources["steven_v2"]["Synovitis"]
    candidates["core_synovitis_v2_raw"] = targeted_raw

    targeted_rank = candidates["core_equal_rank"].copy()
    targeted_rank["Synovitis"] = rank_frame(sources["steven_v2"])["Synovitis"]
    candidates["core_synovitis_v2_rank"] = targeted_rank
    return candidates


def score(
    gold: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[float, dict[str, float], dict[str, int]]:
    per_target: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label in LABELS:
        y = gold[label].to_numpy(dtype=float)
        p = predictions.loc[gold.index, label].to_numpy(dtype=float)
        keep = np.isfinite(p)
        counts[label] = int(keep.sum())
        per_target[label] = float(roc_auc_score(y[keep], p[keep]))
    return float(np.mean(list(per_target.values()))), per_target, counts


def bootstrap_macro(
    gold: pd.DataFrame,
    predictions: pd.DataFrame,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = gold[LABELS].to_numpy(dtype=float)
    p = predictions.loc[gold.index, LABELS].to_numpy(dtype=float)
    scores: list[float] = []
    for _ in range(repeats):
        rows = rng.integers(0, len(gold), len(gold))
        target_scores = []
        for col in range(len(LABELS)):
            yy = y[rows, col]
            pp = p[rows, col]
            keep = np.isfinite(pp)
            if np.unique(yy[keep]).size == 2:
                target_scores.append(roc_auc_score(yy[keep], pp[keep]))
        if target_scores:
            scores.append(float(np.mean(target_scores)))
    return tuple(float(x) for x in np.quantile(scores, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp/rsna-label-audit"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/rsna-label-audit/results"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    train = pd.read_csv(args.root / "train.csv")
    train[UID] = train[UID].astype(str)
    train = train.set_index(UID)
    gold = train.loc[train[LABELS].notna().all(axis=1), LABELS].astype(float)
    if len(train) != 4407 or len(gold) != 58:
        raise ValueError(f"unexpected train/gold counts: {len(train)}/{len(gold)}")

    sources = aligned(load_sources(args.root), train.index)
    candidates = make_candidates(sources)

    summary_rows = []
    target_rows = []
    for offset, (name, predictions) in enumerate(candidates.items()):
        macro, per_target, counts = score(gold, predictions)
        lo, hi = bootstrap_macro(gold, predictions, args.bootstrap, seed=20260823 + offset)
        exact = float(
            np.mean(
                np.isclose(
                    predictions.loc[gold.index, LABELS].to_numpy(),
                    gold[LABELS].to_numpy(),
                )
            )
        )
        summary_rows.append(
            {
                "candidate": name,
                "gold_macro_auc": macro,
                "bootstrap_lo": lo,
                "bootstrap_hi": hi,
                "gold_exact_cell_fraction": exact,
                "study_coverage": float(predictions[LABELS].notna().all(axis=1).mean()),
                "gold_min_rows": min(counts.values()),
            }
        )
        target_rows.extend(
            {
                "candidate": name,
                "target": label,
                "gold_auc": auc,
                "gold_rows": counts[label],
            }
            for label, auc in per_target.items()
        )

    summary = pd.DataFrame(summary_rows).sort_values("gold_macro_auc", ascending=False)
    per_target = pd.DataFrame(target_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "gold58_summary.csv", index=False)
    per_target.to_csv(args.out / "gold58_per_target.csv", index=False)

    chosen_name = "core_synovitis_v2_raw"
    chosen = candidates[chosen_name].clip(0.02, 0.98).copy()
    chosen.insert(0, UID, chosen.index)
    chosen.to_parquet(args.out / "labels_consensus_cc0.parquet", index=False)

    manifest = {
        "train_studies": len(train),
        "gold_studies": len(gold),
        "candidate_written": chosen_name,
        "candidate_path": "labels_consensus_cc0.parquet",
        "barun_policy": "audit_only_due_to_nonstandard_license",
        "selection_policy": "fixed_equal_CC0_core_with_predeclared_synovitis_v2",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
