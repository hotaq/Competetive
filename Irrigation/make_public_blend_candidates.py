from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parent
TARGET = "Irrigation_Need"
LABELS = ["Low", "Medium", "High"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
CACHE = ROOT / "artifacts" / "public_blend" / "cache"
OUTDIR = ROOT / "artifacts" / "public_blend_candidates"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    zip_name: str
    member: str
    kind: str = "npy"
    reshape: tuple[int, int] | None = None
    test_reshape: tuple[int, int] | None = None
    reorder: tuple[int, int, int] | None = None
    test_member: str | None = None

    def load(self, *, test: bool = False) -> np.ndarray:
        member = self.test_member if test and self.test_member else self.member
        path = CACHE / self.zip_name
        if not path.exists():
            raise FileNotFoundError(f"Missing cache zip: {path}")
        with zipfile.ZipFile(path) as zf, zf.open(member) as f:
            if self.kind == "npy":
                arr = np.load(f)
            elif self.kind == "csv":
                arr = pd.read_csv(f).values
            else:
                raise ValueError(f"Unknown source kind: {self.kind}")
        arr = np.asarray(arr, dtype=np.float32)
        reshape = self.test_reshape if test and self.test_reshape is not None else self.reshape
        if reshape is not None:
            arr = arr.reshape(*reshape)
        if self.reorder is not None:
            arr = arr[:, self.reorder]
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{self.name} produced bad shape {arr.shape}")
        return arr


SOURCES: list[SourceSpec] = [
    SourceSpec(
        "yun_lgb_advanced",
        "yunsuxiaozi__pss6e4-lgb-advanced-cv-0-97997.zip",
        "oof_preds.npy",
        test_member="test_preds.npy",
    ),
    SourceSpec(
        "include4eto_xgb_pseudo",
        "include4eto__ps6e4-xgb-cudf-pseudo-labels.zip",
        "001_oof.csv",
        kind="csv",
        reshape=(630_000, 3),
        test_reshape=(270_000, 3),
        test_member="001_test.csv",
    ),
    SourceSpec(
        "yun_lgb_baseline",
        "yunsuxiaozi__pss6e4-lgb-baselinecv-0-97943.zip",
        "oof_preds.npy",
        test_member="test_preds.npy",
    ),
    SourceSpec(
        "yun_xgb",
        "yunsuxiaozi__pss6e4-xgb-cv-0-979805.zip",
        "oof_preds.npy",
        test_member="test_preds.npy",
    ),
    SourceSpec(
        "utaazu_cat",
        "utaazu__0-979-cv-single-cat-pairwise-te-bias-tuning.zip",
        "oof_probs_cat.npy",
        test_member="test_probs_cat.npy",
    ),
    SourceSpec(
        "emanuell_xgb",
        "emanuellcs__predicting-irrigation-need-gbdts-ote-pto.zip",
        "oof_xgb.npy",
        test_member="test_preds.npy",  # use only for OOF candidate groups, not aligned model-specific test
    ),
    SourceSpec(
        "emanuell_cat",
        "emanuellcs__predicting-irrigation-need-gbdts-ote-pto.zip",
        "oof_cat.npy",
        test_member="test_preds.npy",  # use only for OOF candidate groups, not aligned model-specific test
    ),
    SourceSpec(
        "ravi_gbdt_v3",
        "ravi20076__playgrounds6e4-public-baseline-v3.zip",
        "OOF_Preds.npy",
        test_member="Mdl_Preds.npy",
    ),
    SourceSpec(
        "amordia_ref_hybrid",
        "amordia__ps-s6e4-ref-xgb-cat-hybrid-public.zip",
        "outputs/public_ref_xgb_cat_hybrid/oof_proba.npy",
        reorder=(1, 2, 0),
        test_member="outputs/public_ref_xgb_cat_hybrid/test_proba.npy",
    ),
    SourceSpec(
        "mahogany_realmlp",
        "mahoganybuttstrings__pg-s6e4-realmlp-cv-0-97802-lb-0-97685.zip",
        "realmlp_oof.csv",
        kind="csv",
        reshape=(630_000, 3),
        test_reshape=(270_000, 3),
        test_member="realmlp_pred.csv",
    ),
]

# Candidate source groups that have matching test probabilities per source.
GROUPS: dict[str, list[str]] = {
    "top5": [
        "yun_lgb_advanced",
        "include4eto_xgb_pseudo",
        "yun_lgb_baseline",
        "yun_xgb",
        "utaazu_cat",
    ],
    "top6_plus_ravi": [
        "yun_lgb_advanced",
        "include4eto_xgb_pseudo",
        "yun_lgb_baseline",
        "yun_xgb",
        "utaazu_cat",
        "ravi_gbdt_v3",
    ],
    "top8_public_match": [
        "yun_lgb_advanced",
        "include4eto_xgb_pseudo",
        "yun_lgb_baseline",
        "yun_xgb",
        "utaazu_cat",
        "ravi_gbdt_v3",
        "amordia_ref_hybrid",
        "mahogany_realmlp",
    ],
}

# Best lightweight random-search candidate found in prior OOF exploration.
KNOWN_TOP5_WEIGHTED = np.array(
    [0.17686108, 0.11876006, 0.22454493, 0.27492042, 0.20491351], dtype=np.float32
)
KNOWN_TOP5_SCALES = np.array([1.0, 1.33079399, 2.04623758], dtype=np.float32)


def load_y() -> np.ndarray:
    return (
        pd.read_csv(ROOT / "train.csv", usecols=[TARGET])[TARGET]
        .map(LABEL_TO_ID)
        .to_numpy(dtype=np.int8)
    )


def submission_from_pred(pred: np.ndarray) -> pd.DataFrame:
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    out = pd.DataFrame({"id": sample["id"], TARGET: [ID_TO_LABEL[int(x)] for x in pred]})
    if out["id"].tolist() != sample["id"].tolist():
        raise ValueError("id mismatch vs sample_submission.csv")
    return out


def tune_scales(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    start: np.ndarray | None = None,
    coarse: bool = True,
) -> tuple[np.ndarray, float]:
    if start is None:
        start = np.ones(3, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)

    def score(scales: np.ndarray) -> float:
        return float(balanced_accuracy_score(y, (probs * scales).argmax(axis=1)))

    best = start.copy()
    best_score = score(best)
    if coarse:
        med_grid = np.linspace(0.75, 1.65, 37)
        high_grid = np.linspace(1.00, 2.35, 55)
        for med in med_grid:
            for high in high_grid:
                scales = np.array([1.0, med, high], dtype=np.float64)
                s = score(scales)
                if s > best_score:
                    best_score = s
                    best = scales

    def objective(z: np.ndarray) -> float:
        scales = np.array([1.0, np.exp(z[0]), np.exp(z[1])], dtype=np.float64)
        return -score(scales)

    result = minimize(
        objective,
        np.log(np.maximum(best[1:], 1e-6)),
        method="Nelder-Mead",
        options={"maxiter": 900, "xatol": 1e-7, "fatol": 1e-7},
    )
    opt = np.array([1.0, np.exp(result.x[0]), np.exp(result.x[1])], dtype=np.float64)
    opt_score = -float(result.fun)
    if opt_score > best_score:
        return opt, opt_score
    return best, best_score


def weighted_average(arrays: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        return np.mean(arrays, axis=0, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()
    stacked = np.stack(arrays, axis=0)
    return np.einsum("m,mnc->nc", weights, stacked, optimize=True).astype(np.float32)


def write_candidate(
    name: str,
    y: np.ndarray,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    scales: np.ndarray,
    source_names: list[str],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    pred_oof = (oof_probs * scales).argmax(axis=1)
    pred_test = (test_probs * scales).argmax(axis=1)
    score = float(balanced_accuracy_score(y, pred_oof))
    acc = float(accuracy_score(y, pred_oof))
    out_path = OUTDIR / f"submission_{name}.csv"
    oof_path = OUTDIR / f"oof_{name}.csv"
    test_pred_path = OUTDIR / f"test_predictions_{name}.csv"

    submission_from_pred(pred_test).to_csv(out_path, index=False)
    pd.DataFrame({"id": pd.read_csv(ROOT / "train.csv", usecols=["id"])["id"], TARGET: [ID_TO_LABEL[int(x)] for x in y], "prediction": [ID_TO_LABEL[int(x)] for x in pred_oof]}).to_csv(oof_path, index=False)
    test_pred = pd.read_csv(ROOT / "sample_submission.csv", usecols=["id"])
    for i, label in enumerate(LABELS):
        test_pred[f"proba_{label}"] = test_probs[:, i]
    test_pred[TARGET] = [ID_TO_LABEL[int(x)] for x in pred_test]
    test_pred.to_csv(test_pred_path, index=False)

    return {
        "name": name,
        "submission": str(out_path.relative_to(ROOT)),
        "oof_predictions": str(oof_path.relative_to(ROOT)),
        "test_predictions": str(test_pred_path.relative_to(ROOT)),
        "oof_balanced_accuracy": score,
        "oof_accuracy": acc,
        "scales": [float(x) for x in scales],
        "source_names": source_names,
        "test_label_counts": {str(k): int(v) for k, v in pd.Series([ID_TO_LABEL[int(x)] for x in pred_test]).value_counts().items()},
        **(extra or {}),
    }


def logistic_blend(
    y: np.ndarray,
    train_arrays: list[np.ndarray],
    test_arrays: list[np.ndarray],
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_train = np.hstack(train_arrays).astype(np.float32)
    x_test = np.hstack(test_arrays).astype(np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    fold_scales: list[np.ndarray] = []
    for fold, (tr, va) in enumerate(skf.split(x_train, y), 1):
        model = Pipeline(
            [
                ("log1p", FunctionTransformer(np.log1p, validate=True)),
                (
                    "lr",
                    LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
                ),
            ]
        )
        model.fit(x_train[tr], y[tr])
        p_val = model.predict_proba(x_train[va]).astype(np.float32)
        oof[va] = p_val
        fold_scale, _ = tune_scales(y[va], p_val, coarse=False)
        fold_scales.append(fold_scale)
        test_folds.append(model.predict_proba(x_test).astype(np.float32))
    scales = np.mean(fold_scales, axis=0)
    # Retune once on stitched OOF; this generally improves local score slightly.
    scales, score = tune_scales(y, oof, start=scales, coarse=True)
    return oof, np.mean(test_folds, axis=0, dtype=np.float32), scales, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-best", action="store_true", help="Copy best local-CV candidate to submission.csv")
    parser.add_argument("--skip-logistic", action="store_true", help="Only make fast average candidates")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    y = load_y()
    specs = {s.name: s for s in SOURCES}
    summaries: list[dict[str, object]] = []

    for group_name, names in GROUPS.items():
        train_arrays = [specs[n].load(test=False) for n in names]
        test_arrays = [specs[n].load(test=True) for n in names]

        # Equal average with tuned class scales.
        oof_avg = weighted_average(train_arrays)
        test_avg = weighted_average(test_arrays)
        scales, _ = tune_scales(y, oof_avg)
        summaries.append(
            write_candidate(
                f"{group_name}_avg_tuned",
                y,
                oof_avg,
                test_avg,
                scales,
                names,
                {"blend_type": "equal_average"},
            )
        )

        # Known top5 weighted candidate from quick exploration.
        if group_name == "top5":
            oof_w = weighted_average(train_arrays, KNOWN_TOP5_WEIGHTED)
            test_w = weighted_average(test_arrays, KNOWN_TOP5_WEIGHTED)
            scales, _ = tune_scales(y, oof_w, start=KNOWN_TOP5_SCALES)
            summaries.append(
                write_candidate(
                    "top5_weighted_search_tuned",
                    y,
                    oof_w,
                    test_w,
                    scales,
                    names,
                    {
                        "blend_type": "weighted_average",
                        "model_weights": [float(x) for x in (KNOWN_TOP5_WEIGHTED / KNOWN_TOP5_WEIGHTED.sum())],
                    },
                )
            )

        if not args.skip_logistic:
            oof_lr, test_lr, lr_scales, _ = logistic_blend(y, train_arrays, test_arrays)
            summaries.append(
                write_candidate(
                    f"{group_name}_logreg_tuned",
                    y,
                    oof_lr,
                    test_lr,
                    lr_scales,
                    names,
                    {"blend_type": "logreg_stacker"},
                )
            )

    summaries = sorted(summaries, key=lambda x: x["oof_balanced_accuracy"], reverse=True)
    summary = {
        "created_by": "make_public_blend_candidates.py",
        "labels": LABELS,
        "candidates": summaries,
        "best_local_cv": summaries[0],
        "note": "Local OOF balanced accuracy is the selection signal; Kaggle public LB can still differ.",
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.copy_best:
        best_submission = ROOT / summaries[0]["submission"]
        shutil.copyfile(best_submission, ROOT / "submission.csv")
        shutil.copyfile(best_submission, OUTDIR / "best_submission.csv")
        (OUTDIR / "best_summary.json").write_text(json.dumps(summaries[0], indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
