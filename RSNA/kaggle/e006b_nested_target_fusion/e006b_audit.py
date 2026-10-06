"""E006b: nested target-wise Pilkwang/Steven-v4 supervision audit.

The blend share for every target and outer validation fold is selected using only
the corresponding outer-training gold rows. No target-specific choice is hard-coded.
This script reads small CSV metadata only and never submits to the leaderboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGETS = [
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
SEED = 20260810
V4_SHARES = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
OUTER_SPLITS = 5
OUTER_REPEATS = 25
INNER_SPLITS = 4
INNER_REPEATS = 5
MIN_INNER_MEAN_GAIN = 0.010
MIN_INNER_POSITIVE_FRACTION = 0.70
BOOTSTRAPS = 10000

# Same core safety gate as E006. E006b is a new estimator, not a relaxation of
# the failed global-fusion contract.
MIN_MACRO_GAIN = 0.020
MIN_TARGET_DELTA = -0.050
MIN_BOOTSTRAP_P_POSITIVE = 0.80
MIN_COVERAGE = 0.999
PREVALENCE_MARGIN = 0.030

EXPECTED_SHA256 = {
    "train.csv": "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33",
    "report_labels_v2.csv": "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920",
    "llm_labels_v4_blend.csv": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
}
PATH_HINTS = {
    "train.csv": [
        "competitions/rsna-knee-abnormality-detection/train.csv",
        "rsna-knee-abnormality-detection/train.csv",
        "competition/train.csv",
    ],
    "report_labels_v2.csv": [
        "datasets/pilkwang/rsna-knee-llm-labels/report_labels_v2.csv",
        "rsna-knee-llm-labels/report_labels_v2.csv",
        "pilkwang/report_labels_v2.csv",
    ],
    "llm_labels_v4_blend.csv": [
        "datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        "rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        "steven/llm_labels_v4_blend.csv",
    ],
}


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_pinned_file(root: Path, filename: str) -> Path:
    """Find an input by filename and require its frozen digest."""
    expected = EXPECTED_SHA256[filename]
    # Never recursively scan the competition mount: it contains hundreds of
    # thousands of DICOM paths. Kaggle dataset files are at mount depth one, and
    # the second hint for each file supports the small local schema fixture.
    candidates = [root / relative for relative in PATH_HINTS[filename]]
    candidates.extend(root.glob(f"*/{filename}"))
    candidates.append(root / filename)
    candidates = sorted({p for p in candidates if p.is_file()})
    matches = [p for p in candidates if sha256_file(p) == expected]
    if not matches:
        seen = {str(p): sha256_file(p) for p in candidates}
        raise RuntimeError(
            f"No {filename} with pinned SHA-256 {expected}; candidates={seen}"
        )
    return matches[0]


def validate_label_frame(frame: pd.DataFrame, name: str) -> None:
    """Validate target schema, UID uniqueness, and probability range."""
    required = ["StudyInstanceUID", *TARGETS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")
    if frame["StudyInstanceUID"].astype(str).duplicated().any():
        raise RuntimeError(f"{name} contains duplicate UIDs")
    values = frame[TARGETS].to_numpy(float)
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise RuntimeError(f"{name} has values outside [0, 1]")


def align(frame: pd.DataFrame, ids: list[str]) -> np.ndarray:
    """Align target scores to competition train order."""
    copy = frame.copy()
    copy["StudyInstanceUID"] = copy["StudyInstanceUID"].astype(str)
    return copy.set_index("StudyInstanceUID").reindex(ids)[TARGETS].to_numpy(float)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Finite-filtered ROC AUC."""
    keep = np.isfinite(y) & np.isfinite(score)
    yt = y[keep].astype(int)
    if keep.sum() < 2 or np.unique(yt).size < 2:
        return float("nan")
    return float(roc_auc_score(yt, score[keep]))


def blend(pilkwang: np.ndarray, steven_v4: np.ndarray, v4_share: float) -> np.ndarray:
    """Blend two report-label scores in probability space with missing renormalization."""
    left = np.asarray(pilkwang, dtype=float)
    right = np.asarray(steven_v4, dtype=float)
    weights = np.asarray([1.0 - v4_share, v4_share])
    values = np.stack([left, right], axis=-1)
    finite = np.isfinite(values)
    denominator = (finite * weights).sum(axis=-1)
    numerator = (np.where(finite, values, 0.0) * weights).sum(axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, 0.5, dtype=float),
        where=denominator > 0,
    )


def select_share(
    y: np.ndarray,
    pilkwang: np.ndarray,
    steven_v4: np.ndarray,
    seed: int,
    n_splits: int = INNER_SPLITS,
    n_repeats: int = INNER_REPEATS,
) -> tuple[float, dict[str, Any]]:
    """Choose a conservative v4 share using repeated inner validation only."""
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    splits = min(n_splits, positives, negatives)
    if splits < 2:
        return 0.0, {"reason": "degenerate", "mean_delta": [0.0] * len(V4_SHARES)}

    deltas = [[] for _ in V4_SHARES]
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(
            n_splits=splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        for _train_idx, valid_idx in splitter.split(np.zeros(len(y)), y):
            baseline_auc = auc(y[valid_idx], pilkwang[valid_idx])
            for k, share in enumerate(V4_SHARES):
                score = blend(pilkwang[valid_idx], steven_v4[valid_idx], float(share))
                deltas[k].append(auc(y[valid_idx], score) - baseline_auc)

    mean_delta = np.asarray([np.nanmean(values) for values in deltas])
    positive_fraction = np.asarray(
        [np.mean(np.asarray(values) > 0.0) for values in deltas]
    )
    eligible = (
        (mean_delta >= MIN_INNER_MEAN_GAIN)
        & (positive_fraction >= MIN_INNER_POSITIVE_FRACTION)
    )
    eligible[0] = True  # Pilkwang is always the safe fallback.
    eligible_indices = np.flatnonzero(eligible)
    best_gain = float(np.max(mean_delta[eligible_indices]))
    # Prefer the smallest share within numerical tolerance of the best eligible
    # gain. This shrinks toward the established Pilkwang source.
    tied = eligible_indices[np.isclose(mean_delta[eligible_indices], best_gain, atol=1e-12)]
    chosen_idx = int(tied[0])
    if chosen_idx != 0 and mean_delta[chosen_idx] < MIN_INNER_MEAN_GAIN:
        chosen_idx = 0
    return float(V4_SHARES[chosen_idx]), {
        "mean_delta": mean_delta.tolist(),
        "positive_fraction": positive_fraction.tolist(),
        "chosen_index": chosen_idx,
    }


def nested_crossfit(
    y: np.ndarray,
    pilkwang: np.ndarray,
    steven_v4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Generate repeated outer-OOF predictions and target/share frequencies."""
    n, n_targets = y.shape
    predictions = np.zeros((n, n_targets), dtype=float)
    counts = np.zeros((n, n_targets), dtype=np.int32)
    selection_counts = np.zeros((n_targets, len(V4_SHARES)), dtype=np.int32)
    records: list[dict[str, Any]] = []

    for j, target in enumerate(TARGETS):
        for repeat in range(OUTER_REPEATS):
            splitter = StratifiedKFold(
                n_splits=OUTER_SPLITS,
                shuffle=True,
                random_state=SEED + 1009 * j + repeat,
            )
            for fold, (train_idx, valid_idx) in enumerate(
                splitter.split(np.zeros(n), y[:, j])
            ):
                share, diagnostics = select_share(
                    y[train_idx, j],
                    pilkwang[train_idx, j],
                    steven_v4[train_idx, j],
                    seed=SEED + 100_003 * j + 997 * repeat + fold,
                )
                predictions[valid_idx, j] += blend(
                    pilkwang[valid_idx, j], steven_v4[valid_idx, j], share
                )
                counts[valid_idx, j] += 1
                share_idx = int(np.flatnonzero(np.isclose(V4_SHARES, share))[0])
                selection_counts[j, share_idx] += 1
                records.append(
                    {
                        "target": target,
                        "repeat": repeat,
                        "fold": fold,
                        "v4_share": share,
                        "inner_mean_gain": diagnostics["mean_delta"][share_idx],
                        "inner_positive_fraction": diagnostics.get(
                            "positive_fraction", [0.0] * len(V4_SHARES)
                        )[share_idx],
                        "n_train": int(len(train_idx)),
                        "n_valid": int(len(valid_idx)),
                    }
                )

    if not np.all(counts == OUTER_REPEATS):
        raise RuntimeError("Nested cross-fit did not cover every gold target per repeat")
    predictions /= counts
    frequencies = selection_counts / selection_counts.sum(axis=1, keepdims=True)
    return predictions, frequencies, records


def per_target_auc(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Return one AUC per target."""
    return np.asarray([auc(y[:, j], score[:, j]) for j in range(y.shape[1])])


def macro_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Return unweighted finite-target macro AUC."""
    return float(np.nanmean(per_target_auc(y, score)))


def bootstrap_counts(n: int, count: int, seed: int) -> np.ndarray:
    """Draw bootstrap samples and return their per-study multiplicities."""
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n, size=(count, n))
    counts = np.zeros((count, n), dtype=np.float64)
    rows = np.repeat(np.arange(count), n)
    np.add.at(counts, (rows, sampled.ravel()), 1.0)
    return counts


def weighted_bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Vectorized weighted Mann-Whitney AUC for bootstrap multiplicities."""
    y = y.astype(int)
    comparison = (score[:, None] > score[None, :]).astype(float)
    comparison += 0.5 * (score[:, None] == score[None, :])
    comparison *= (y[:, None] == 1) & (y[None, :] == 0)
    numerator = np.einsum("bi,ij,bj->b", counts, comparison, counts, optimize=True)
    positives = counts[:, y == 1].sum(axis=1)
    negatives = counts[:, y == 0].sum(axis=1)
    denominator = positives * negatives
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(counts), np.nan),
        where=denominator > 0,
    )


def paired_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    """Compute a paired study-bootstrap macro-AUC delta efficiently."""
    counts = bootstrap_counts(len(y), BOOTSTRAPS, SEED + 71)
    target_deltas = []
    for j in range(y.shape[1]):
        base_auc = weighted_bootstrap_auc(y[:, j], baseline[:, j], counts)
        candidate_auc = weighted_bootstrap_auc(y[:, j], candidate[:, j], counts)
        target_deltas.append(candidate_auc - base_auc)
    delta = np.nanmean(np.stack(target_deltas, axis=1), axis=1)
    delta = delta[np.isfinite(delta)]
    return {
        "n": int(len(delta)),
        "p_positive": float(np.mean(delta > 0.0)),
        "ci_low": float(np.quantile(delta, 0.025)),
        "ci_high": float(np.quantile(delta, 0.975)),
    }


def prevalence_check(
    candidate: np.ndarray,
    pilkwang: np.ndarray,
    steven_v4: np.ndarray,
) -> tuple[bool, list[dict[str, Any]]]:
    """Require hard prevalence to stay near the two-source envelope."""
    candidate_prev = np.mean(candidate >= 0.5, axis=0)
    source_prev = np.stack(
        [np.nanmean(pilkwang >= 0.5, axis=0), np.nanmean(steven_v4 >= 0.5, axis=0)]
    )
    rows = []
    checks = []
    for j, target in enumerate(TARGETS):
        low = max(0.0, float(source_prev[:, j].min()) - PREVALENCE_MARGIN)
        high = min(1.0, float(source_prev[:, j].max()) + PREVALENCE_MARGIN)
        passed = low <= candidate_prev[j] <= high
        checks.append(passed)
        rows.append(
            {
                "target": target,
                "candidate": float(candidate_prev[j]),
                "pilkwang": float(source_prev[0, j]),
                "steven_v4": float(source_prev[1, j]),
                "allowed_low": low,
                "allowed_high": high,
                "pass": bool(passed),
            }
        )
    return bool(all(checks)), rows


def json_ready(value: Any) -> Any:
    """Convert NumPy values recursively into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def run(input_root: Path, output_dir: Path) -> dict[str, Any]:
    """Execute E006b and write small audit artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: find_pinned_file(input_root, name) for name in EXPECTED_SHA256}
    print("E006b pinned inputs")
    for name, path in paths.items():
        print(f"  {name}: {path} [{EXPECTED_SHA256[name][:12]}]")

    train = pd.read_csv(paths["train.csv"])
    pilkwang_frame = pd.read_csv(paths["report_labels_v2.csv"])
    v4_frame = pd.read_csv(paths["llm_labels_v4_blend.csv"])
    validate_label_frame(pilkwang_frame, "pilkwang")
    validate_label_frame(v4_frame, "steven_v4")
    if len(train) != 4407:
        raise RuntimeError(f"Expected 4,407 train rows, found {len(train)}")
    ids = train["StudyInstanceUID"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise RuntimeError("competition train.csv contains duplicate UIDs")

    pilkwang = align(pilkwang_frame, ids)
    steven_v4 = align(v4_frame, ids)
    present = train[TARGETS].notna().to_numpy()
    if np.any(present.any(axis=1) != present.all(axis=1)):
        raise RuntimeError("Found partially annotated official rows")
    official_gold_mask = present.all(axis=1)
    if int(official_gold_mask.sum()) != 58:
        raise RuntimeError(
            f"Expected 58 official gold rows, found {int(official_gold_mask.sum())}"
        )
    common_gold_mask = (
        official_gold_mask
        & np.isfinite(pilkwang).all(axis=1)
        & np.isfinite(steven_v4).all(axis=1)
    )
    # Pilkwang's pinned public file has 4,406 rows and omits one official-gold
    # study. A paired source comparison must drop that row from both arms rather
    # than score v4 on 58 and Pilkwang on 57 or impute Pilkwang after seeing gold.
    if int(common_gold_mask.sum()) != 57:
        raise RuntimeError(
            "Expected 57 source-common gold rows after the pinned Pilkwang omission, "
            f"found {int(common_gold_mask.sum())}"
        )
    y_gold = train.loc[common_gold_mask, TARGETS].to_numpy(int)
    p_gold = pilkwang[common_gold_mask]
    v4_gold = steven_v4[common_gold_mask]

    nested_oof, selection_frequency, selection_records = nested_crossfit(
        y_gold, p_gold, v4_gold
    )

    # The deployable diagnostic shares are selected by repeated CV on all Gold58.
    # E007 must repeat this selection inside its own outer folds; these global
    # shares must not be treated as an independent Gold58 validation artifact.
    full_shares = np.zeros(len(TARGETS), dtype=float)
    full_diagnostics: dict[str, Any] = {}
    full_candidate = np.zeros_like(pilkwang)
    for j, target in enumerate(TARGETS):
        share, diagnostics = select_share(
            y_gold[:, j],
            p_gold[:, j],
            v4_gold[:, j],
            seed=SEED + 700_001 + 101 * j,
            n_splits=OUTER_SPLITS,
            n_repeats=OUTER_REPEATS,
        )
        full_shares[j] = share
        full_diagnostics[target] = diagnostics
        full_candidate[:, j] = blend(pilkwang[:, j], steven_v4[:, j], share)

    baseline_per = per_target_auc(y_gold, p_gold)
    candidate_per = per_target_auc(y_gold, nested_oof)
    fixed_v4_per = per_target_auc(y_gold, v4_gold)
    target_delta = candidate_per - baseline_per
    baseline_macro = float(np.mean(baseline_per))
    candidate_macro = float(np.mean(candidate_per))
    fixed_v4_macro = float(np.mean(fixed_v4_per))
    macro_gain = candidate_macro - baseline_macro
    bootstrap = paired_bootstrap(y_gold, p_gold, nested_oof)
    coverage = float(np.mean(np.isfinite(full_candidate)))
    prevalence_pass, prevalence = prevalence_check(
        full_candidate, pilkwang, steven_v4
    )
    checks = {
        "macro_gain": bool(macro_gain >= MIN_MACRO_GAIN),
        "no_target_collapse": bool(float(np.min(target_delta)) >= MIN_TARGET_DELTA),
        "bootstrap_direction": bool(
            bootstrap["p_positive"] >= MIN_BOOTSTRAP_P_POSITIVE
        ),
        "coverage": bool(coverage >= MIN_COVERAGE),
        "prevalence": prevalence_pass,
    }
    passed = bool(all(checks.values()))
    decision = "authorize_e007_b3_scout" if passed else "stop_target_fusion"

    per_target = pd.DataFrame(
        {
            "target": TARGETS,
            "positives": y_gold.sum(axis=0),
            "auc_pilkwang": baseline_per,
            "auc_fixed_v4": fixed_v4_per,
            "auc_nested_candidate": candidate_per,
            "nested_delta": target_delta,
            "mean_outer_v4_share": selection_frequency @ V4_SHARES,
            "full_cv_v4_share": full_shares,
        }
    )
    for k, share in enumerate(V4_SHARES):
        per_target[f"outer_frequency_share_{share:.2f}"] = selection_frequency[:, k]
    per_target.to_csv(output_dir / "e006b_per_target.csv", index=False)
    pd.DataFrame(selection_records).to_csv(
        output_dir / "e006b_selection_records.csv", index=False
    )

    gold_oof = pd.DataFrame({"StudyInstanceUID": np.asarray(ids)[common_gold_mask]})
    for j, target in enumerate(TARGETS):
        gold_oof[f"gold__{target}"] = y_gold[:, j]
        gold_oof[f"pilkwang__{target}"] = p_gold[:, j]
        gold_oof[f"fixed_v4__{target}"] = v4_gold[:, j]
        gold_oof[f"nested_candidate__{target}"] = nested_oof[:, j]
    gold_oof.to_csv(output_dir / "e006b_gold_oof.csv", index=False)

    teacher = pd.DataFrame({"StudyInstanceUID": ids})
    training = pd.DataFrame({"StudyInstanceUID": ids})
    confidence = np.clip(2.0 * np.abs(full_candidate - 0.5), 0.0, 1.0)
    y_full = train[TARGETS].to_numpy(float)
    for j, target in enumerate(TARGETS):
        teacher[target] = full_candidate[:, j]
        values = full_candidate[:, j].copy()
        values[official_gold_mask] = y_full[official_gold_mask, j]
        training[target] = values
        weights = confidence[:, j].copy()
        weights[official_gold_mask] = 1.0
        training[f"{target}__weight"] = weights
    training["is_gold"] = official_gold_mask.astype(int)
    teacher.to_csv(output_dir / "e006b_teacher_labels.csv", index=False)
    training.to_csv(output_dir / "e006b_training_targets_gold_override.csv", index=False)

    result = {
        "experiment": "E006b nested target-wise supervision fusion",
        "decision": decision,
        "pass": passed,
        "n_train": int(len(train)),
        "n_gold_official": int(official_gold_mask.sum()),
        "n_gold_source_common": int(common_gold_mask.sum()),
        "baseline_macro_auc": baseline_macro,
        "fixed_v4_macro_auc": fixed_v4_macro,
        "nested_candidate_macro_auc": candidate_macro,
        "macro_gain": float(macro_gain),
        "minimum_target_delta": float(np.min(target_delta)),
        "checks": checks,
        "bootstrap": bootstrap,
        "coverage": coverage,
        "per_target_delta": {
            TARGETS[j]: float(target_delta[j]) for j in range(len(TARGETS))
        },
        "mean_outer_v4_share": {
            TARGETS[j]: float(selection_frequency[j] @ V4_SHARES)
            for j in range(len(TARGETS))
        },
        "full_cv_v4_share": {
            TARGETS[j]: float(full_shares[j]) for j in range(len(TARGETS))
        },
        "selection_frequency": {
            TARGETS[j]: {
                f"{share:.2f}": float(selection_frequency[j, k])
                for k, share in enumerate(V4_SHARES)
            }
            for j in range(len(TARGETS))
        },
        "full_selection_diagnostics": full_diagnostics,
        "prevalence": prevalence,
        "thresholds": {
            "v4_share_grid": V4_SHARES.tolist(),
            "minimum_inner_mean_gain": MIN_INNER_MEAN_GAIN,
            "minimum_inner_positive_fraction": MIN_INNER_POSITIVE_FRACTION,
            "minimum_macro_gain": MIN_MACRO_GAIN,
            "minimum_target_delta": MIN_TARGET_DELTA,
            "minimum_bootstrap_probability_positive": MIN_BOOTSTRAP_P_POSITIVE,
            "minimum_coverage": MIN_COVERAGE,
            "prevalence_margin": PREVALENCE_MARGIN,
        },
        "input_sha256": EXPECTED_SHA256,
        "notes": [
            "Target choices are learned inside each outer-training split; none are hard-coded.",
            "The paired audit uses the 57 gold rows covered by both sources; no missing Pilkwang score is imputed.",
            "This is a second audit on the official gold set after E006 and is not independent clinical evidence.",
            "If authorized, E007 must repeat label-source selection inside its image outer folds.",
            "No leaderboard submission is produced by E006b.",
        ],
    }
    result = json_ready(result)
    (output_dir / "e006b_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    )
    manifest = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.glob("e006b_*"))
        if path.name != "e006b_manifest.json"
    }
    (output_dir / "e006b_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    print("\nE006b Gold58 nested audit")
    print(f"  Pilkwang baseline : {baseline_macro:.6f}")
    print(f"  fixed Steven v4   : {fixed_v4_macro:.6f}")
    print(f"  nested candidate  : {candidate_macro:.6f} ({macro_gain:+.6f})")
    print(f"  minimum target d  : {float(np.min(target_delta)):+.6f}")
    print(
        "  bootstrap         : [{:+.6f}, {:+.6f}], P(>0)={:.4f}".format(
            bootstrap["ci_low"], bootstrap["ci_high"], bootstrap["p_positive"]
        )
    )
    print("E006b GATE:", "GO" if passed else "NO-GO", decision, checks)
    print("wrote", output_dir / "e006b_gate.json")
    return result


def self_test() -> None:
    """Exercise selection, nesting, and vectorized bootstrap on synthetic data."""
    rng = np.random.default_rng(17)
    y = np.asarray([0, 1] * 30, dtype=int)
    pilkwang = np.clip(0.35 + 0.30 * y + rng.normal(0, 0.20, len(y)), 0.01, 0.99)
    better = np.clip(0.20 + 0.60 * y + rng.normal(0, 0.10, len(y)), 0.01, 0.99)
    share, _diagnostics = select_share(y, pilkwang, better, seed=123)
    assert share > 0.0
    y_cube = np.repeat(y[:, None], len(TARGETS), axis=1)
    p_cube = np.repeat(pilkwang[:, None], len(TARGETS), axis=1)
    v_cube = np.repeat(better[:, None], len(TARGETS), axis=1)
    oof, frequencies, records = nested_crossfit(y_cube, p_cube, v_cube)
    assert oof.shape == y_cube.shape
    assert frequencies.shape == (len(TARGETS), len(V4_SHARES))
    assert len(records) == len(TARGETS) * OUTER_REPEATS * OUTER_SPLITS
    assert np.isfinite(oof).all()
    assert macro_auc(y_cube, oof) > macro_auc(y_cube, p_cube)
    boot = paired_bootstrap(y_cube, p_cube, oof)
    assert boot["n"] == BOOTSTRAPS and boot["p_positive"] > 0.8
    print("E006b synthetic self-test PASS")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--self-test", action="store_true")
    args, _unknown = parser.parse_known_args()
    if args.self_test:
        self_test()
    else:
        run(args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
