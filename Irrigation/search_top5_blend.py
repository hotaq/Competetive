from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "artifacts" / "public_blend" / "cache"
OUTDIR = ROOT / "artifacts" / "public_blend_candidates"
TARGET = "Irrigation_Need"
LABELS = ["Low", "Medium", "High"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

SPECS = [
    ("yun_lgb_advanced", "yunsuxiaozi__pss6e4-lgb-advanced-cv-0-97997.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("include4eto_xgb_pseudo", "include4eto__ps6e4-xgb-cudf-pseudo-labels.zip", "001_oof.csv", "001_test.csv", "csv", ((630_000, 3), (270_000, 3))),
    ("yun_lgb_baseline", "yunsuxiaozi__pss6e4-lgb-baselinecv-0-97943.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("yun_xgb", "yunsuxiaozi__pss6e4-xgb-cv-0-979805.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("utaazu_cat", "utaazu__0-979-cv-single-cat-pairwise-te-bias-tuning.zip", "oof_probs_cat.npy", "test_probs_cat.npy", "npy", None),
]


def load_array(zip_name: str, member: str, kind: str, shape=None) -> np.ndarray:
    with zipfile.ZipFile(CACHE / zip_name) as zf, zf.open(member) as f:
        if kind == "npy":
            arr = np.load(f)
        else:
            arr = pd.read_csv(f).values
    arr = np.asarray(arr, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(*shape)
    return arr


def load_stack(test: bool = False) -> tuple[list[str], np.ndarray]:
    arrays = []
    names = []
    for name, zip_name, oof_member, test_member, kind, shapes in SPECS:
        member = test_member if test else oof_member
        shape = None
        if shapes is not None:
            shape = shapes[1] if test else shapes[0]
        arrays.append(load_array(zip_name, member, kind, shape))
        names.append(name)
    return names, np.stack(arrays, axis=0).astype(np.float32)


def probs(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()
    return np.einsum("m,mnc->nc", weights, stack, optimize=True).astype(np.float32)


def pred_from(stack: np.ndarray, weights: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (probs(stack, weights) * scales.astype(np.float32)).argmax(axis=1)


def score(y: np.ndarray, stack: np.ndarray, weights: np.ndarray, scales: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, pred_from(stack, weights, scales)))


def make_balanced_subset(y: np.ndarray, rng: np.random.Generator, per_class: int) -> np.ndarray:
    idxs = []
    for c in range(3):
        cls = np.flatnonzero(y == c)
        idxs.append(rng.choice(cls, size=min(per_class, len(cls)), replace=False))
    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    return idx


def random_candidate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(rng.choice([0.15, 0.25, 0.4, 0.7, 1.2, 2.5, 6.0, 16.0, 40.0]))
    w = rng.dirichlet(np.ones(5, dtype=np.float64) * alpha).astype(np.float32)
    # Focus around the high-CV basin found earlier, with some broader exploration.
    med = float(np.exp(rng.uniform(np.log(0.85), np.log(1.75))))
    high = float(np.exp(rng.uniform(np.log(1.20), np.log(2.60))))
    return w, np.array([1.0, med, high], dtype=np.float32)


def local_refine_scales(y: np.ndarray, p: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, float]:
    best = np.asarray(start, dtype=np.float32).copy()
    best_score = float(balanced_accuracy_score(y, (p * best).argmax(axis=1)))
    # Progressive local grid around the current scale. Keeps runtime bounded.
    for width, steps in [(0.22, 31), (0.10, 25), (0.045, 21)]:
        med_values = np.linspace(best[1] * (1 - width), best[1] * (1 + width), steps)
        high_values = np.linspace(best[2] * (1 - width), best[2] * (1 + width), steps)
        for med in med_values:
            for high in high_values:
                sc = np.array([1.0, med, high], dtype=np.float32)
                s = float(balanced_accuracy_score(y, (p * sc).argmax(axis=1)))
                if s > best_score:
                    best_score = s
                    best = sc
    return best, best_score


def write_submission(name: str, test_stack: np.ndarray, weights: np.ndarray, scales: np.ndarray, summary: dict) -> Path:
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    pred = pred_from(test_stack, weights, scales)
    sub = pd.DataFrame({"id": sample["id"], TARGET: [ID_TO_LABEL[int(x)] for x in pred]})
    out = OUTDIR / f"submission_{name}.csv"
    sub.to_csv(out, index=False)
    summary["submission"] = str(out.relative_to(ROOT))
    summary["test_label_counts"] = {str(k): int(v) for k, v in sub[TARGET].value_counts().items()}
    (OUTDIR / f"summary_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30000)
    parser.add_argument("--per-class", type=int, default=30000)
    parser.add_argument("--top-k", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--name", default="top5_random_search")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    y = pd.read_csv(ROOT / "train.csv", usecols=[TARGET])[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int8)
    names, train_stack = load_stack(test=False)
    _, test_stack = load_stack(test=True)
    idx = make_balanced_subset(y, rng, args.per_class)
    y_sub = y[idx]
    sub_stack = train_stack[:, idx, :]

    # Strong seeds from prior exploration.
    seeds = [
        (
            np.array([0.17686108, 0.11876006, 0.22454493, 0.27492042, 0.20491351], dtype=np.float32),
            np.array([1.0, 1.33079399, 2.04623758], dtype=np.float32),
        ),
        (
            np.ones(5, dtype=np.float32) / 5,
            np.array([1.0, 1.47707985, 2.27830092], dtype=np.float32),
        ),
    ]

    cands: list[tuple[float, np.ndarray, np.ndarray]] = []
    start = time.time()
    for w, sc in seeds:
        cands.append((score(y_sub, sub_stack, w, sc), w, sc))
    for i in range(args.rounds):
        w, sc = random_candidate(rng)
        s = score(y_sub, sub_stack, w, sc)
        if len(cands) < args.top_k or s > cands[0][0]:
            cands.append((s, w, sc))
            cands = sorted(cands, key=lambda x: x[0])[-args.top_k:]
        if (i + 1) % 5000 == 0:
            print(f"round={i+1} subset_best={cands[-1][0]:.8f} elapsed={time.time()-start:.1f}s", flush=True)

    # Full evaluate top subset candidates with their discovered scales.
    full: list[tuple[float, np.ndarray, np.ndarray]] = []
    for subset_s, w, sc in sorted(cands, key=lambda x: x[0], reverse=True):
        s = score(y, train_stack, w, sc)
        full.append((s, w, sc))
    full = sorted(full, key=lambda x: x[0], reverse=True)
    print("full direct top:", flush=True)
    for s, w, sc in full[:10]:
        print(f"  {s:.8f} w={np.round(w,6).tolist()} scales={np.round(sc,6).tolist()}", flush=True)

    best_score, best_w, best_sc = full[0]
    best_p = probs(train_stack, best_w)
    best_sc, best_score = local_refine_scales(y, best_p, best_sc)
    for s, w, sc in full[1:8]:
        p = probs(train_stack, w)
        sc2, s2 = local_refine_scales(y, p, sc)
        if s2 > best_score:
            best_score, best_w, best_sc = s2, w, sc2

    pred_oof = pred_from(train_stack, best_w, best_sc)
    summary = {
        "name": args.name,
        "source_names": names,
        "rounds": args.rounds,
        "per_class_subset": args.per_class,
        "oof_balanced_accuracy": float(best_score),
        "oof_accuracy": float(accuracy_score(y, pred_oof)),
        "weights": [float(x) for x in (best_w / best_w.sum())],
        "scales": [float(x) for x in best_sc],
        "elapsed_seconds": time.time() - start,
    }
    out = write_submission(args.name, test_stack, best_w, best_sc, summary)
    # Update best if this beats previous known local CV.
    prev_summary = OUTDIR / "best_summary.json"
    prev_score = -1.0
    if prev_summary.exists():
        try:
            prev_score = float(json.loads(prev_summary.read_text()).get("oof_balanced_accuracy", -1.0))
        except Exception:
            pass
    if best_score >= prev_score:
        (OUTDIR / "best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.read_csv(out).to_csv(OUTDIR / "best_submission.csv", index=False)
        pd.read_csv(out).to_csv(ROOT / "submission.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    print("SUBMISSION=" + str(out), flush=True)


if __name__ == "__main__":
    main()
