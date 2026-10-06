"""E006: leakage-resistant audit of public report-label sources.

This script reads CSV metadata only. It never reads DICOM pixels, trains an image
model, or submits to Kaggle. The primary three-source fusion is evaluated with
repeated out-of-fold reliability estimation on the 58 image-adjudicated studies.
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
SOURCES = ["pilkwang", "steven_v2", "lixin"]
SEED = 20260810
N_SPLITS = 5
N_REPEATS = 25
SHRINK_TO_EQUAL = 0.50
BOOTSTRAPS = 5000

# Predeclared promotion gate. The 58-study audit is deliberately demanding: it
# must show a material macro gain, no target collapse, and paired-bootstrap
# evidence in the correct direction. It is still an audit, not a model score.
MIN_MACRO_GAIN = 0.020
MIN_TARGET_DELTA = -0.050
MIN_BOOTSTRAP_P_POSITIVE = 0.80
MIN_COVERAGE = 0.999
PREVALENCE_MARGIN = 0.030

EXPECTED_SHA256 = {
    "train.csv": "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33",
    "report_labels_v2.csv": "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920",
    "llm_labels_v2.csv": "3c082a987939528a55202028f655bd6f9dc5c9e28eff5058bbae7ef088a9f144",
    "llm_labels_v4_blend.csv": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
    "labels_llm_gpt56sol.csv": "a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_pinned_file(root: Path, filename: str) -> Path:
    """Find a mounted file by name and require its predeclared digest."""
    expected = EXPECTED_SHA256[filename]
    candidates = sorted(root.rglob(filename))
    matches = [p for p in candidates if sha256_file(p) == expected]
    if not matches:
        seen = {str(p): sha256_file(p) for p in candidates}
        raise RuntimeError(
            f"No {filename} with pinned SHA-256 {expected}; candidates={seen}"
        )
    # Identical duplicate mounts are harmless; choosing lexicographically is
    # deterministic and the digest, not the path, defines the input identity.
    return matches[0]


def validate_label_frame(frame: pd.DataFrame, name: str) -> None:
    """Validate UID uniqueness, target columns, and score ranges."""
    required = ["StudyInstanceUID", *TARGETS]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")
    if frame["StudyInstanceUID"].astype(str).duplicated().any():
        raise RuntimeError(f"{name} contains duplicate StudyInstanceUID rows")
    values = frame[TARGETS].to_numpy(float)
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise RuntimeError(f"{name} has values outside [0, 1]")


def align(frame: pd.DataFrame, ids: list[str], columns: list[str]) -> pd.DataFrame:
    """Align a source to competition train order without silent row loss."""
    out = frame.copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    out = out.set_index("StudyInstanceUID").reindex(ids)
    return out[columns]


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC with finite filtering and degenerate-target protection."""
    keep = np.isfinite(score) & np.isfinite(y)
    yt = y[keep].astype(int)
    if keep.sum() < 2 or np.unique(yt).size < 2:
        return float("nan")
    return float(roc_auc_score(yt, score[keep]))


def per_target_auc(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Return one AUC per target."""
    return np.asarray([auc(y[:, j], score[:, j]) for j in range(y.shape[1])])


def macro_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Unweighted finite-target mean AUC, matching the competition metric."""
    per = per_target_auc(y, score)
    return float(np.nanmean(per))


def reliability_weights(y: np.ndarray, source_scores: np.ndarray) -> np.ndarray:
    """Fit nonnegative AUC-skill weights, shrunk halfway toward equal weights."""
    source_auc = np.asarray([auc(y, source_scores[:, k]) for k in range(len(SOURCES))])
    skill = np.clip(2.0 * (np.nan_to_num(source_auc, nan=0.5) - 0.5), 0.0, 1.0)
    if float(skill.sum()) <= 1e-12:
        learned = np.full(len(SOURCES), 1.0 / len(SOURCES))
    else:
        learned = skill / skill.sum()
    equal = np.full(len(SOURCES), 1.0 / len(SOURCES))
    weights = SHRINK_TO_EQUAL * equal + (1.0 - SHRINK_TO_EQUAL) * learned
    return weights / weights.sum()


def fuse_column(
    source_scores: np.ndarray,
    source_states: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Fuse one target in probability space with an agreement hardening rule.

    Pilkwang UNK and Steven's exact 0.5 are unaddressed states. Lixin does not
    expose uncertainty, so its binary prediction remains an addressed vote. If at
    least two addressed votes agree unanimously, the target is hardened to 0.02 or
    0.98; disagreements retain the reliability-weighted probability.
    """
    finite = np.isfinite(source_scores)
    denom = (finite * weights[None, :]).sum(axis=1)
    numer = (np.where(finite, source_scores, 0.0) * weights[None, :]).sum(axis=1)
    base = np.divide(numer, denom, out=np.full(len(source_scores), 0.5), where=denom > 0)
    result = np.clip(base, 0.02, 0.98)
    for i in range(len(result)):
        states = source_states[i, np.isfinite(source_states[i])].astype(int)
        if len(states) >= 2 and np.unique(states).size == 1:
            result[i] = 0.98 if states[0] == 1 else 0.02
    return result


def repeated_crossfit(
    y: np.ndarray,
    source_scores: np.ndarray,
    source_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fit target-specific fusion weights over repeated stratified folds."""
    n, n_targets, n_sources = source_scores.shape
    predictions = np.zeros((n, n_targets), dtype=np.float64)
    counts = np.zeros((n, n_targets), dtype=np.int32)
    weight_sum = np.zeros((n_targets, n_sources), dtype=np.float64)
    weight_count = np.zeros(n_targets, dtype=np.int32)

    for j in range(n_targets):
        for repeat in range(N_REPEATS):
            splitter = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=SEED + 1009 * j + repeat,
            )
            for train_idx, valid_idx in splitter.split(np.zeros(n), y[:, j]):
                weights = reliability_weights(y[train_idx, j], source_scores[train_idx, j])
                predictions[valid_idx, j] += fuse_column(
                    source_scores[valid_idx, j],
                    source_states[valid_idx, j],
                    weights,
                )
                counts[valid_idx, j] += 1
                weight_sum[j] += weights
                weight_count[j] += 1

    if not np.all(counts == N_REPEATS):
        raise RuntimeError("Cross-fit did not predict every gold row once per repeat")
    predictions /= counts
    mean_weights = weight_sum / weight_count[:, None]
    return predictions, mean_weights


def paired_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed_offset: int,
) -> dict[str, float]:
    """Paired study bootstrap for the macro-AUC difference."""
    rng = np.random.default_rng(SEED + seed_offset)
    deltas: list[float] = []
    n = len(y)
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, n, size=n)
        delta = macro_auc(y[idx], candidate[idx]) - macro_auc(y[idx], baseline[idx])
        if np.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        return {"n": 0, "p_positive": 0.0, "ci_low": float("nan"), "ci_high": float("nan")}
    arr = np.asarray(deltas)
    return {
        "n": int(len(arr)),
        "p_positive": float(np.mean(arr > 0.0)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def prevalence_check(candidate: np.ndarray, sources: np.ndarray) -> tuple[bool, list[dict[str, Any]]]:
    """Check candidate hard prevalence against the envelope of its sources."""
    candidate_prev = np.mean(candidate >= 0.5, axis=0)
    source_prev = np.nanmean(sources >= 0.5, axis=0)
    rows: list[dict[str, Any]] = []
    checks = []
    for j, target in enumerate(TARGETS):
        low = max(0.0, float(source_prev[j].min()) - PREVALENCE_MARGIN)
        high = min(1.0, float(source_prev[j].max()) + PREVALENCE_MARGIN)
        ok = low <= candidate_prev[j] <= high
        checks.append(ok)
        rows.append(
            {
                "target": target,
                "candidate": float(candidate_prev[j]),
                "source_min": float(source_prev[j].min()),
                "source_max": float(source_prev[j].max()),
                "allowed_low": low,
                "allowed_high": high,
                "pass": bool(ok),
            }
        )
    return bool(all(checks)), rows


def evaluate_gate(
    name: str,
    y: np.ndarray,
    baseline: np.ndarray,
    candidate_gold: np.ndarray,
    candidate_full: np.ndarray,
    full_sources: np.ndarray,
    seed_offset: int,
) -> dict[str, Any]:
    """Evaluate one predeclared candidate against the Pilkwang baseline."""
    baseline_per = per_target_auc(y, baseline)
    candidate_per = per_target_auc(y, candidate_gold)
    delta = candidate_per - baseline_per
    baseline_macro = float(np.nanmean(baseline_per))
    candidate_macro = float(np.nanmean(candidate_per))
    gain = candidate_macro - baseline_macro
    bootstrap = paired_bootstrap(y, baseline, candidate_gold, seed_offset)
    coverage = float(np.mean(np.isfinite(candidate_full)))
    prevalence_pass, prevalence = prevalence_check(candidate_full, full_sources)
    checks = {
        "macro_gain": bool(gain >= MIN_MACRO_GAIN),
        "no_target_collapse": bool(float(np.nanmin(delta)) >= MIN_TARGET_DELTA),
        "bootstrap_direction": bool(bootstrap["p_positive"] >= MIN_BOOTSTRAP_P_POSITIVE),
        "coverage": bool(coverage >= MIN_COVERAGE),
        "prevalence": prevalence_pass,
    }
    return {
        "name": name,
        "pass": bool(all(checks.values())),
        "checks": checks,
        "baseline_macro_auc": baseline_macro,
        "candidate_macro_auc": candidate_macro,
        "macro_gain": float(gain),
        "minimum_target_delta": float(np.nanmin(delta)),
        "coverage": coverage,
        "bootstrap": bootstrap,
        "per_target_delta": {TARGETS[j]: float(delta[j]) for j in range(len(TARGETS))},
        "prevalence": prevalence,
    }


def json_ready(value: Any) -> Any:
    """Convert NumPy values recursively into strict JSON-compatible objects."""
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
    """Run the complete E006 audit and write small reusable artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: find_pinned_file(input_root, name) for name in EXPECTED_SHA256}
    print("E006 pinned inputs")
    for name, path in paths.items():
        print(f"  {name}: {path} [{EXPECTED_SHA256[name][:12]}]")

    train = pd.read_csv(paths["train.csv"])
    pilkwang = pd.read_csv(paths["report_labels_v2.csv"])
    steven_v2 = pd.read_csv(paths["llm_labels_v2.csv"])
    steven_v4 = pd.read_csv(paths["llm_labels_v4_blend.csv"])
    lixin = pd.read_csv(paths["labels_llm_gpt56sol.csv"])

    if list(train.columns[:2]) != ["StudyInstanceUID", "Report"]:
        raise RuntimeError("Unexpected competition train.csv leading columns")
    validate_label_frame(pilkwang, "pilkwang")
    validate_label_frame(steven_v2, "steven_v2")
    validate_label_frame(steven_v4, "steven_v4")
    validate_label_frame(lixin, "lixin")
    if len(train) != 4407:
        raise RuntimeError(f"Expected 4,407 train rows, found {len(train)}")
    ids = train["StudyInstanceUID"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise RuntimeError("competition train.csv contains duplicate UIDs")

    pilkwang_scores = align(pilkwang, ids, TARGETS).to_numpy(float)
    steven_scores = align(steven_v2, ids, TARGETS).to_numpy(float)
    v4 = align(steven_v4, ids, TARGETS).to_numpy(float)
    lixin_scores = align(lixin, ids, TARGETS).to_numpy(float)
    source_scores = np.stack([pilkwang_scores, steven_scores, lixin_scores], axis=2)

    verdict_columns = [f"{target}__verdict" for target in TARGETS]
    p_verdict = align(pilkwang, ids, verdict_columns).to_numpy(object)
    p_state = np.full(p_verdict.shape, np.nan, dtype=float)
    p_state[p_verdict == "YES"] = 1.0
    p_state[p_verdict == "NO"] = 0.0
    s_state = np.where(
        np.isclose(steven_scores, 0.5),
        np.nan,
        (steven_scores > 0.5).astype(float),
    )
    l_state = (lixin_scores > 0.5).astype(float)
    source_states = np.stack([p_state, s_state, l_state], axis=2)

    target_present = train[TARGETS].notna().to_numpy()
    if np.any(target_present.any(axis=1) != target_present.all(axis=1)):
        raise RuntimeError("Found partially annotated official rows")
    gold_mask = target_present.all(axis=1)
    if int(gold_mask.sum()) != 58:
        raise RuntimeError(f"Expected 58 gold rows, found {int(gold_mask.sum())}")
    y_gold = train.loc[gold_mask, TARGETS].to_numpy(int)
    gold_sources = source_scores[gold_mask]
    gold_states = source_states[gold_mask]
    pilkwang_gold = pilkwang_scores[gold_mask]

    primary_oof, crossfit_mean_weights = repeated_crossfit(y_gold, gold_sources, gold_states)
    equal_oof = np.zeros_like(primary_oof)
    equal_weights = np.full(len(SOURCES), 1.0 / len(SOURCES))
    for j in range(len(TARGETS)):
        equal_oof[:, j] = fuse_column(gold_sources[:, j], gold_states[:, j], equal_weights)

    full_weights = np.zeros((len(TARGETS), len(SOURCES)), dtype=float)
    primary_full = np.zeros((len(train), len(TARGETS)), dtype=float)
    for j in range(len(TARGETS)):
        full_weights[j] = reliability_weights(y_gold[:, j], gold_sources[:, j])
        primary_full[:, j] = fuse_column(source_scores[:, j], source_states[:, j], full_weights[j])

    primary_gate = evaluate_gate(
        "three_source_crossfit",
        y_gold,
        pilkwang_gold,
        primary_oof,
        primary_full,
        source_scores,
        seed_offset=11,
    )
    v4_gate = evaluate_gate(
        "fixed_external_steven_v4",
        y_gold,
        pilkwang_gold,
        v4[gold_mask],
        v4,
        source_scores,
        seed_offset=29,
    )

    if primary_gate["pass"]:
        decision = "promote_three_source"
        promoted = primary_full
    elif v4_gate["pass"]:
        decision = "promote_fixed_steven_v4"
        promoted = v4
    else:
        decision = "stop_b3_supervision_gate"
        promoted = primary_full

    source_auc = {
        "pilkwang": per_target_auc(y_gold, pilkwang_scores[gold_mask]),
        "steven_v2": per_target_auc(y_gold, steven_scores[gold_mask]),
        "lixin": per_target_auc(y_gold, lixin_scores[gold_mask]),
        "steven_v4": per_target_auc(y_gold, v4[gold_mask]),
        "equal_three": per_target_auc(y_gold, equal_oof),
        "crossfit_three": per_target_auc(y_gold, primary_oof),
    }
    per_target = pd.DataFrame({"target": TARGETS, "positives": y_gold.sum(axis=0)})
    for name, values in source_auc.items():
        per_target[f"auc_{name}"] = values
    per_target["delta_crossfit_vs_pilkwang"] = (
        per_target["auc_crossfit_three"] - per_target["auc_pilkwang"]
    )
    per_target["delta_v4_vs_pilkwang"] = (
        per_target["auc_steven_v4"] - per_target["auc_pilkwang"]
    )
    per_target.to_csv(output_dir / "e006_per_target.csv", index=False)

    gold_oof = pd.DataFrame({"StudyInstanceUID": np.asarray(ids)[gold_mask]})
    for j, target in enumerate(TARGETS):
        gold_oof[f"gold__{target}"] = y_gold[:, j]
        gold_oof[f"pilkwang__{target}"] = pilkwang_gold[:, j]
        gold_oof[f"crossfit_three__{target}"] = primary_oof[:, j]
        gold_oof[f"fixed_v4__{target}"] = v4[gold_mask, j]
    gold_oof.to_csv(output_dir / "e006_gold_oof.csv", index=False)

    teacher = pd.DataFrame({"StudyInstanceUID": ids})
    training = pd.DataFrame({"StudyInstanceUID": ids})
    confidence = np.clip(2.0 * np.abs(promoted - 0.5), 0.0, 1.0)
    for j, target in enumerate(TARGETS):
        teacher[target] = promoted[:, j]
        values = promoted[:, j].copy()
        values[gold_mask] = y_gold[:, j]
        training[target] = values
        weights = confidence[:, j].copy()
        weights[gold_mask] = 1.0
        training[f"{target}__weight"] = weights
    training["is_gold"] = gold_mask.astype(int)
    teacher.to_csv(output_dir / "e006_promoted_teacher_labels.csv", index=False)
    training.to_csv(output_dir / "e006_training_targets_gold_override.csv", index=False)

    weight_payload = {
        "sources": SOURCES,
        "shrink_to_equal": SHRINK_TO_EQUAL,
        "crossfit_mean_weights": {
            target: crossfit_mean_weights[j].tolist() for j, target in enumerate(TARGETS)
        },
        "full_gold_fit_weights": {
            target: full_weights[j].tolist() for j, target in enumerate(TARGETS)
        },
    }
    (output_dir / "e006_source_weights.json").write_text(
        json.dumps(json_ready(weight_payload), indent=2, sort_keys=True)
    )

    macros = {name: float(np.nanmean(values)) for name, values in source_auc.items()}
    result = {
        "experiment": "E006 supervision fusion audit",
        "decision": decision,
        "pass": bool(decision != "stop_b3_supervision_gate"),
        "promoted_source": (
            "three_source_crossfit"
            if decision == "promote_three_source"
            else "fixed_external_steven_v4"
            if decision == "promote_fixed_steven_v4"
            else None
        ),
        "n_train": int(len(train)),
        "n_gold": int(gold_mask.sum()),
        "source_overlap": {
            "pilkwang": int(np.isfinite(pilkwang_scores).all(axis=1).sum()),
            "steven_v2": int(np.isfinite(steven_scores).all(axis=1).sum()),
            "lixin": int(np.isfinite(lixin_scores).all(axis=1).sum()),
            "steven_v4": int(np.isfinite(v4).all(axis=1).sum()),
        },
        "macro_auc": macros,
        "primary_gate": primary_gate,
        "fixed_v4_gate": v4_gate,
        "thresholds": {
            "minimum_macro_gain": MIN_MACRO_GAIN,
            "minimum_target_delta": MIN_TARGET_DELTA,
            "minimum_bootstrap_probability_positive": MIN_BOOTSTRAP_P_POSITIVE,
            "minimum_coverage": MIN_COVERAGE,
            "prevalence_margin": PREVALENCE_MARGIN,
        },
        "input_sha256": EXPECTED_SHA256,
        "notes": [
            "Gold58 is an audit ruler, not an image-model leaderboard estimate.",
            "The primary fusion uses Steven v2, not v4, so Lixin is not double-counted.",
            "If no candidate passes, generated labels are diagnostic and must not launch E007.",
        ],
    }
    result = json_ready(result)
    (output_dir / "e006_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    )

    manifest = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.glob("e006_*"))
        if path.name != "e006_manifest.json"
    }
    (output_dir / "e006_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    print("\nGold58 macro AUC")
    for name, value in macros.items():
        print(f"  {name:16s} {value:.6f}")
    print(
        f"\nPrimary gain {primary_gate['macro_gain']:+.6f}; "
        f"fixed-v4 gain {v4_gate['macro_gain']:+.6f}"
    )
    print("E006 GATE:", "GO" if result["pass"] else "NO-GO", decision)
    print("wrote", output_dir / "e006_gate.json")
    return result


def self_test() -> None:
    """Run deterministic synthetic checks without touching competition files."""
    y = np.asarray([0, 1] * 20)
    source_scores = np.stack(
        [
            np.where(y == 1, 0.9, 0.1),
            np.where(y == 1, 0.75, 0.25),
            np.asarray([0, 1] * 20, dtype=float),
        ],
        axis=1,
    )
    states = (source_scores > 0.5).astype(float)
    weights = reliability_weights(y, source_scores)
    assert np.isclose(weights.sum(), 1.0)
    fused = fuse_column(source_scores, states, weights)
    assert np.all((0.0 <= fused) & (fused <= 1.0))
    cube = np.repeat(source_scores[:, None, :], len(TARGETS), axis=1)
    state_cube = np.repeat(states[:, None, :], len(TARGETS), axis=1)
    y_cube = np.repeat(y[:, None], len(TARGETS), axis=1)
    oof, mean_weights = repeated_crossfit(y_cube, cube, state_cube)
    assert oof.shape == y_cube.shape
    assert mean_weights.shape == (len(TARGETS), len(SOURCES))
    assert np.isfinite(oof).all()
    assert macro_auc(y_cube, oof) > 0.99
    print("E006 synthetic self-test PASS")


def main() -> None:
    """CLI entrypoint for local self-test or Kaggle execution."""
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
