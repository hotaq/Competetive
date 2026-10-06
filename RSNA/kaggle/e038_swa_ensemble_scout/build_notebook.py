#!/usr/bin/env python3
"""Build the E038 inference-only base+SWA ensemble scout notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def extract_defs(source: str, names: list[str]) -> str:
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in names
    ]
    found = {node.name for node in selected}
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"missing definitions: {sorted(missing)}")
    return "\n\n".join(ast.get_source_segment(source, node) for node in selected)


def extract_assignments(source: str, names: list[str]) -> str:
    tree = ast.parse(source)
    selected = []
    found = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        node_names = {
            target.id for target in targets if isinstance(target, ast.Name) and target.id in names
        }
        if node_names:
            selected.append(node)
            found.update(node_names)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"missing assignments: {sorted(missing)}")
    return "\n\n".join(ast.get_source_segment(source, node) for node in selected)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "e038-swa-ensemble-scout.ipynb"
E020 = REPO / "kaggle/e020_single_coatnet_scout/e020-single-coatnet-repro.py"
E022 = REPO / "kaggle/e022_warmstart_single_coatnet/e022-warmstart-single-coatnet.py"
E028 = REPO / "kaggle/e028_exact64_corpus_proof/e028-exact64-corpus-proof.ipynb"

e020_source = E020.read_text()
model_defs = extract_defs(e020_source, ["RaptorClassifier", "build_backbone", "load_model"])

e022_source = E022.read_text()
label_defs = "\n\n".join(
    [
        extract_assignments(e022_source, ["E021_LABEL_SOURCES"]),
        extract_defs(e022_source, ["_e021_source_file", "_build_e021_labels"]),
    ]
)

e028_notebook = nbf.read(E028, as_version=4)
e028_source = "\n\n".join(
    cell.source for cell in e028_notebook.cells if cell.cell_type == "code"
)
data_defs = extract_defs(
    e028_source,
    [
        "find_root",
        "dataset_roots",
        "find_input_file",
        "fluid",
        "pick_series",
        "ordered_records",
        "pixels",
        "crop",
        "build_study",
        "safe_candidates",
        "resample_centers",
    ],
)

cells = [
    md(
        r"""
# E038 — inference-only base + SWA ensemble scout

E036 closed fine-tuning and E037 closed test-time augmentation. Both stopped for
the same reason: the CC0 consensus the model trains on scores `0.8921` on the
official Gold58 labels while the checkpoint scores `0.9174`, so nothing that
needs supervision can help, and every geometric transform tried degraded the
predictions.

The last supervision-free lever named in those results was **fold ensembling**.
There are no fold checkpoints. `dreaddevelopment/raptor-knee-widedense` holds
exactly two files — `raptor_ft_coatnet_v4_full.pt`, the 0.927 anchor, and
`raptor_ft_coatnet_v4_full_swa.pt`, its stochastic-weight-averaging variant —
and no other attached dataset carries a `.pt`. So the ensemble that can actually
be built is **base + SWA**, and that is what this run measures.

Be clear about the prior: SWA averages weights along one training trajectory, so
it is far more correlated with the base model than an independent fold would be,
and ensembles of correlated members gain little. This run therefore **measures
that correlation directly** rather than inferring it from the result, so a null
outcome comes with its own explanation.

Nothing is trained. Both checkpoints are loaded once, frozen, and asked the same
question.

## The split is E037's untouched complement

The labelled non-Gold pool holds 4,349 studies. E037 drew 2,048 of them with
`EVAL_SEED = 3701`. E038 reconstructs that exact draw, **verifies it against the
SHA256 E037 recorded**, and then evaluates on the 2,301 studies E037 never
touched. Two things follow:

* The ensemble question is answered on data no previous scout has selected on.
* Arm `B4_base_all` re-runs E037's best arm `A1_windows_all` — pool every safe
  window instead of the 42 the baseline draws — on a **disjoint** sample. E037
  measured `+0.00104` with a 95% interval of `[+0.00062, +0.00150]`; whether
  that number reappears out of sample is reported, not gated.

## The six arms

Both checkpoints share the window bank, because `RaptorClassifier.head` pools
with `softmax(..., dim=1)` over `K` and is permutation-invariant, so the bank is
built once per checkpoint in the order `eval_centers + remaining candidates` and
chunked by six exactly as `predict_studies` does. Arm `B0_base` is therefore the
proven path and must reproduce `0.9174298724699895` on Gold58.

| arm | what it is |
|---|---|
| `B0_base` | base checkpoint, the 42 baseline windows — **baseline** |
| `B1_swa` | SWA checkpoint, the same 42 windows |
| `B2_ens_prob` | mean of `B0` and `B1` in probability space |
| `B3_ens_logit` | mean of `B0` and `B1` in logit space |
| `B4_base_all` | base checkpoint pooling every safe window (E037's `A1`) |
| `B5_ens_prob_all` | mean of both checkpoints, each pooling every safe window |

## What is being decided

* **Selection** on the 2,301-study complement with CC0 consensus labels. AUC is
  rank-based and both arms of a comparison see the same labels, so label noise
  widens the interval without biasing the sign.
* **Confirmation** on Gold58, opened only to check the sign agrees. E036 showed
  Gold58 is blind to a real regression and E037 caught it flipping the sign on
  the zoom arm; it is a second opinion here, never the selector.
* `MIN_VALID_GAIN` stays at **0.0020**, the same bar E037 was held to. It is not
  arbitrary: Kaggle discussion 735154 measured a 0.0020 leaderboard noise floor,
  so a smaller gain cannot be told apart on the board. Moving the bar after
  seeing E037 fall short of it would be moving the goalposts.
* Five candidate arms against one baseline, so the win-rate threshold stays at
  `0.98` and the family size is recorded.

E037 measured a paired interval half-width of `0.00085` at `n=2048` on arms of a
single frozen checkpoint. E038 compares two checkpoints on 2,301 studies, so the
same order of resolution is expected — but it is **measured** per arm, and a run
where no arm's estimator can resolve `MIN_VALID_GAIN` stops with a distinct
`STOP_ENSEMBLE_UNDERPOWERED` rather than reporting a confident null.

No competition submission, no exported checkpoint, no exported feature cache.
"""
    ),
    code(
        r"""
from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from sklearn.metrics import roc_auc_score
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
HERE = str(WORK)
# The exact64 cache for 2,359 studies is about 10 GiB and must never be
# collected as kernel output. E037 v1 fell back to /kaggle/working because
# /kaggle/temp did not exist on the image, so scratch is created rather than
# probed and /kaggle/working is rejected outright.
SCRATCH = Path("/kaggle/temp")
try:
    SCRATCH.mkdir(parents=True, exist_ok=True)
except OSError:
    SCRATCH = Path(tempfile.mkdtemp(prefix="e038_scratch_"))
if WORK == SCRATCH or WORK in SCRATCH.parents:
    raise RuntimeError(f"scratch {SCRATCH} would be collected as kernel output")
CACHE = SCRATCH / "e038_exact64_cache"
CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LAB = TARGETS
IMG, RES, CROP_MM, K_EVAL, K_TRAIN, K_VAL = 336, 384, 140.0, 42, 12, 12
K = K_EVAL
SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# EVAL_SEED is E037's on purpose: it is what reconstructs E037's 2,048-study
# draw so this run can take the complement. BOOTSTRAP_SEED is E038's own.
EVAL_SEED = 3701
BOOTSTRAP_SEED = 3802
CHUNK = 6
BOOTSTRAP_REPEATS = 2000

BASE_CHECKPOINT = "raptor_ft_coatnet_v4_full.pt"
SWA_CHECKPOINT = "raptor_ft_coatnet_v4_full_swa.pt"
MODEL_LABELS = ["base", "swa"]
WINDOW_SETS = ["42", "all"]

BASELINE_ARM = "B0_base"
EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
EXPECTED_BASE_SHA256 = (
    "89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded"
)
REPRODUCTION_TOLERANCE = 0.0005
RUNTIME_BUDGET_MINUTES = 420.0

# E037 identity, so the complement is provably E037's unused pool.
E037_EVAL_IDS_SHA256 = (
    "97de5347aa7ee01e88283b3430ab2aaacc3283b9aceab47e02a14945b62c8c93"
)
E037_N_EVAL = 2048
EXPECTED_POOL = 4349
EXPECTED_N_EVAL = EXPECTED_POOL - E037_N_EVAL

# E037's measurement of the arm B4 replicates, reported not gated.
E037_A1_DELTA = 0.0010437443710005212
E037_A1_CI95 = [0.0006198022190293764, 0.0015000035609342985]

# Efficacy thresholds, unchanged from E037 and declared before any run.
MIN_VALID_GAIN = 0.0020
MIN_BOOTSTRAP_WIN = 0.98
MAX_TARGET_DROP = 0.010
GOLD58_MIN_TARGET_DELTA = -0.010

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
print("device", DEVICE, "torch", torch.__version__, "scratch", SCRATCH)
"""
    ),
    md("## Locked exact64 data, E020 model, and E022 label definitions"),
    code(data_defs),
    code(model_defs),
    code(label_defs),
    md(
        """## Loading a second checkpoint without guessing

`load_model` is used verbatim for the base checkpoint so `B0_base` stays the
proven path. The SWA file may carry averaging bookkeeping, so it gets a loader
that normalises **only** two documented things — a leading `module.` prefix and
an `n_averaged` entry — records exactly which of them it applied, and then loads
`strict=True`. Anything else about the file is a hard failure rather than a
silently reshaped state dict.

Two checkpoints that turn out to hold the same weights would make every ensemble
arm a copy of the baseline, so the parameter divergence between them is measured
and gated."""
    ),
    code(
        r"""
def load_checkpoint_variant(path, arch_default, res_default, device):
    '''`load_model` for a checkpoint that may carry SWA bookkeeping.'''
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        top_level = sorted(str(key) for key in raw.keys())
        arch = raw.get("arch", arch_default)
        ck_res = int(raw.get("res", res_default))
        state = raw["model"]
    else:
        top_level = []
        arch = arch_default
        ck_res = int(res_default)
        state = raw
    normalisations = []
    if any(str(key).startswith("module.") for key in state):
        state = {
            (str(key)[len("module."):] if str(key).startswith("module.") else str(key)): value
            for key, value in state.items()
        }
        normalisations.append("module_prefix_stripped")
    if "n_averaged" in state:
        state = {key: value for key, value in state.items() if key != "n_averaged"}
        normalisations.append("n_averaged_dropped")
    backbone = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(backbone, F_dim=backbone.num_features)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    info = {
        "arch": arch,
        "res": ck_res,
        "top_level_keys": top_level,
        "normalisations": normalisations,
        "state_entries": len(state),
    }
    del raw, state
    gc.collect()
    return model, ck_res, info


def parameter_divergence(model_a, model_b):
    '''How far apart the two checkpoints actually are, per float parameter.'''
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    if set(state_a) != set(state_b):
        raise RuntimeError("the two checkpoints do not share a parameter set")
    max_abs = 0.0
    total_abs = 0.0
    total_count = 0
    identical_tensors = 0
    float_tensors = 0
    for key, value_a in state_a.items():
        value_b = state_b[key]
        if tuple(value_a.shape) != tuple(value_b.shape):
            raise RuntimeError(f"{key}: shape {tuple(value_a.shape)} vs {tuple(value_b.shape)}")
        if not value_a.is_floating_point():
            continue
        float_tensors += 1
        difference = (value_a.detach().float() - value_b.detach().float()).abs()
        tensor_max = float(difference.max()) if difference.numel() else 0.0
        max_abs = max(max_abs, tensor_max)
        total_abs += float(difference.sum())
        total_count += int(difference.numel())
        identical_tensors += int(tensor_max == 0.0)
    return {
        "max_abs_param_diff": max_abs,
        "mean_abs_param_diff": total_abs / max(total_count, 1),
        "parameters_compared": total_count,
        "float_tensors": float_tensors,
        "identical_tensors": identical_tensors,
    }
"""
    ),
    md(
        """## One bank per checkpoint, four logit sets, six arms

`window_features` is `predict_studies`' feature pass: the same `stack_windows`,
the same ImageNet normalisation, the same chunk of six under float16 autocast.
T4 is `sm_75` and has no native bfloat16, so float16 is the only autocast dtype
that keeps a fused attention kernel eligible."""
    ),
    code(
        r"""
def stack_windows(volume, centers):
    windows = np.stack([
        volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
        for center in centers
    ])
    tensor = torch.from_numpy(windows)
    return F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)


@torch.no_grad()
def window_features(model, volume, centers):
    '''Backbone features for one study, chunked exactly as `predict_studies`.'''
    tensor = stack_windows(volume, centers)
    tensor = (tensor - MEAN) / STD
    chunks = []
    for start in range(0, len(tensor), CHUNK):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            chunks.append(model.backbone(tensor[start:start + CHUNK].to(DEVICE)).float().cpu())
    return torch.cat(chunks, 0)


@torch.no_grad()
def head_logits(model, features):
    return model.head(features.to(DEVICE).unsqueeze(0))[0].float().cpu().numpy()


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, np.float64)))


RAW_KEYS = [f"{label}/{window_set}" for label in MODEL_LABELS for window_set in WINDOW_SETS]

ARM_NAMES = [
    "B0_base",
    "B1_swa",
    "B2_ens_prob",
    "B3_ens_logit",
    "B4_base_all",
    "B5_ens_prob_all",
]
ARM_NOTES = {
    "B0_base": "base checkpoint, the 42 baseline windows - the proven path",
    "B1_swa": "SWA checkpoint, the same 42 windows",
    "B2_ens_prob": "mean of base and SWA in probability space",
    "B3_ens_logit": "mean of base and SWA in logit space",
    "B4_base_all": "base checkpoint pooling every safe window - E037 arm A1",
    "B5_ens_prob_all": "mean of both checkpoints, each pooling every safe window",
}


def assemble_arms(raw):
    base_42 = sigmoid(raw["base/42"])
    swa_42 = sigmoid(raw["swa/42"])
    base_all = sigmoid(raw["base/all"])
    swa_all = sigmoid(raw["swa/all"])
    arms = {
        "B0_base": base_42,
        "B1_swa": swa_42,
        "B2_ens_prob": (base_42 + swa_42) / 2.0,
        "B3_ens_logit": sigmoid((raw["base/42"] + raw["swa/42"]) / 2.0),
        "B4_base_all": base_all,
        "B5_ens_prob_all": (base_all + swa_all) / 2.0,
    }
    if list(arms) != ARM_NAMES:
        raise RuntimeError(f"arm order drifted: {list(arms)} != {ARM_NAMES}")
    return arms


def target_aucs(y, predictions):
    values = [roc_auc_score(y[:, index], predictions[:, index])
              for index in range(len(TARGETS))]
    return np.asarray(values, np.float64), float(np.mean(values))


def paired_bootstrap(y, base_predictions, arm_predictions, seed, repeats=BOOTSTRAP_REPEATS):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        indexes = rng.integers(0, len(y), size=len(y))
        target_differences = []
        for target_index in range(len(TARGETS)):
            labels_sample = y[indexes, target_index]
            if np.unique(labels_sample).size < 2:
                continue
            base_auc = roc_auc_score(labels_sample, base_predictions[indexes, target_index])
            arm_auc = roc_auc_score(labels_sample, arm_predictions[indexes, target_index])
            target_differences.append(arm_auc - base_auc)
        if target_differences:
            differences.append(float(np.mean(target_differences)))
    differences = np.asarray(differences, np.float64)
    low, high = float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))
    return {
        "valid_repeats": len(differences),
        "win_rate": float((differences > 0).mean()),
        "mean_delta": float(differences.mean()),
        "ci95": [low, high],
        "ci95_halfwidth": (high - low) / 2.0,
    }


def prediction_divergence(base_predictions, swa_predictions):
    '''How much room there is for an ensemble to help, measured not assumed.

    AUC is rank-based, so Spearman is the correlation that matters: two members
    ranking studies identically cannot ensemble into a different AUC whatever
    their probabilities look like.
    '''
    pearson = {}
    spearman = {}
    for index, target in enumerate(TARGETS):
        column_a = base_predictions[:, index]
        column_b = swa_predictions[:, index]
        pearson[target] = float(np.corrcoef(column_a, column_b)[0, 1])
        rank_a = pd.Series(column_a).rank().to_numpy()
        rank_b = pd.Series(column_b).rank().to_numpy()
        spearman[target] = float(np.corrcoef(rank_a, rank_b)[0, 1])
    return {
        "per_target_pearson": pearson,
        "per_target_spearman": spearman,
        "mean_pearson": float(np.mean(list(pearson.values()))),
        "mean_spearman": float(np.mean(list(spearman.values()))),
        "min_spearman": float(np.min(list(spearman.values()))),
        "mean_abs_prob_diff": float(np.abs(base_predictions - swa_predictions).mean()),
        "max_abs_prob_diff": float(np.abs(base_predictions - swa_predictions).max()),
    }


@torch.no_grad()
def evaluate_arms(models, ids):
    '''One cache read and one backbone pass per checkpoint per study.'''
    if list(models) != MODEL_LABELS:
        raise RuntimeError(f"model order drifted: {list(models)} != {MODEL_LABELS}")
    raw = {key: np.zeros((len(ids), len(TARGETS)), np.float32) for key in RAW_KEYS}
    windows_seen = 0
    short_pool_studies = 0
    started = time.time()
    for row, uid in enumerate(ids):
        volume, _mask = load_lossless_cache(cache_paths[uid])
        eval_centers = eval_centers_by_uid[uid]
        candidates = candidates_by_uid[uid]
        if len(eval_centers) != K_EVAL:
            raise RuntimeError(f"{uid}: baseline draw is not {K_EVAL} windows")
        # `resample_centers` returns exactly K_EVAL entries whatever the pool
        # size, so a study with fewer than K_EVAL safe candidates is served
        # repeated windows - by the production path too, not just here. That is
        # reproduced rather than rejected; distinctness is only an invariant
        # when the pool is large enough to supply it.
        if len(candidates) >= K_EVAL:
            if len(set(eval_centers)) != K_EVAL:
                raise RuntimeError(f"{uid}: {len(candidates)} candidates gave repeated centres")
        else:
            short_pool_studies += 1
        selected = set(eval_centers)
        order = list(eval_centers) + [c for c in candidates if c not in selected]
        for label, model in models.items():
            features = window_features(model, volume, order)
            windows_seen += len(order)
            raw[f"{label}/42"][row] = head_logits(model, features[:K_EVAL])
            raw[f"{label}/all"][row] = head_logits(model, features)

    for key, values in raw.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite logits in {key}")
    arms = assemble_arms(raw)
    stats = {
        "studies": len(ids),
        "windows": windows_seen,
        "seconds": time.time() - started,
        "windows_per_study": windows_seen / max(len(ids), 1),
        # Studies with fewer than K_EVAL safe candidates: their bank holds
        # repeated windows, B4/B5 collapse onto B0/B2, and the count is
        # reported so the arm deltas are read against a known share.
        "short_pool_studies": short_pool_studies,
    }
    return arms, raw, stats
"""
    ),
    md("## Identity, E037's verified split, and the 2,301-study complement"),
    code(
        r"""
def sha256_file(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


ROOT = find_root()
TRAIN_DCM = ROOT / "train_series"
series = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str,
                                                        "SeriesInstanceUID": str})
train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
series_by_uid = {
    uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID")
}

numeric = train[TARGETS].apply(pd.to_numeric, errors="coerce")
gold_mask = numeric.notna().all(axis=1)
gold = train.loc[gold_mask, ["StudyInstanceUID"]].copy()
gold[TARGETS] = numeric.loc[gold_mask].astype(np.float32)
gold_ids = gold["StudyInstanceUID"].tolist()
gold_set = set(gold_ids)
y_gold = gold[TARGETS].to_numpy(np.float32)

all_ids_path = find_input_file("all_ids.npy", ("knee-raptor-corpus",))
extra_ids_path = find_input_file("extra_ids.npy", ("knee-raptor-corpus-ext",))
corpus_ids = [str(value) for value in np.load(all_ids_path, allow_pickle=True).tolist()]
corpus_ids += [str(value) for value in np.load(extra_ids_path, allow_pickle=True).tolist()]
corpus_ids = list(dict.fromkeys(corpus_ids))

label_path = _build_e021_labels()
soft = pd.read_parquet(label_path)
soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")

non_gold_ids = [uid for uid in corpus_ids if uid not in gold_set]
labelled_non_gold = [uid for uid in non_gold_ids if uid in soft.index]
unlabelled_dropped = len(non_gold_ids) - len(labelled_non_gold)

# Reconstruct E037's draw and prove it before claiming the rest is untouched.
split_rng = np.random.default_rng(EVAL_SEED)
shuffled = list(
    np.asarray(labelled_non_gold, dtype=object)[split_rng.permutation(len(labelled_non_gold))]
)
e037_eval_ids = [str(uid) for uid in shuffled[:E037_N_EVAL]]
e037_split_sha256 = sha256_ids(e037_eval_ids)
e037_split_reconstructed = e037_split_sha256 == E037_EVAL_IDS_SHA256
if not e037_split_reconstructed:
    raise RuntimeError(
        f"reconstructed E037 split hashes {e037_split_sha256}, expected "
        f"{E037_EVAL_IDS_SHA256} over a pool of {len(labelled_non_gold)}; the "
        "remainder cannot be claimed to be E037's untouched complement"
    )

eval_ids = [str(uid) for uid in shuffled[E037_N_EVAL:]]
N_EVAL = len(eval_ids)
if N_EVAL != EXPECTED_N_EVAL:
    raise RuntimeError(f"complement is {N_EVAL} studies, expected {EXPECTED_N_EVAL}")
assert not gold_set & set(eval_ids)
assert not set(eval_ids) & set(e037_eval_ids)

y_eval_soft = np.stack([soft.loc[uid, TARGETS].to_numpy(np.float32) for uid in eval_ids])
y_eval = (y_eval_soft >= 0.5).astype(np.float32)
eval_two_class = bool((y_eval.sum(0) > 0).all() and (y_eval.sum(0) < len(eval_ids)).all())
eval_positives = dict(zip(TARGETS, y_eval.sum(0).astype(int).tolist()))
print("evaluation positives per target", eval_positives)

BASE_WEIGHT = find_input_file(BASE_CHECKPOINT, ("raptor-knee", "widedense"))
SWA_WEIGHT = find_input_file(SWA_CHECKPOINT, ("raptor-knee", "widedense"))
base_sha256 = sha256_file(BASE_WEIGHT)
swa_sha256 = sha256_file(SWA_WEIGHT)
eval_ids_sha256 = sha256_ids(eval_ids)
gold_ids_sha256 = sha256_ids(gold_ids)
identity = {
    "base_checkpoint_name": BASE_WEIGHT.name,
    "base_checkpoint_sha256": base_sha256,
    "swa_checkpoint_name": SWA_WEIGHT.name,
    "swa_checkpoint_sha256": swa_sha256,
    "all_ids_sha256": sha256_file(all_ids_path),
    "extra_ids_sha256": sha256_file(extra_ids_path),
    "eval_seed": EVAL_SEED,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "eval_ids_sha256": eval_ids_sha256,
    "gold_ids_sha256": gold_ids_sha256,
    "e037_split_sha256": e037_split_sha256,
    "e037_split_reconstructed": e037_split_reconstructed,
    "labelled_non_gold_pool": len(labelled_non_gold),
    "unlabelled_dropped": unlabelled_dropped,
    "eval_ids": eval_ids,
    "gold_ids": gold_ids,
}
(WORK / "e038_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
print("eval", N_EVAL, "gold", len(gold_ids), "pool", len(labelled_non_gold),
      "dropped", unlabelled_dropped, "e037 split verified", e037_split_reconstructed)
print("base", BASE_WEIGHT.name, base_sha256)
print("swa ", SWA_WEIGHT.name, swa_sha256)
"""
    ),
    md("## Reversible exact64 cache, built in two stages"),
    code(
        r"""
def delta_x_encode(volume):
    encoded = np.empty_like(volume)
    encoded[:, :, 0] = volume[:, :, 0]
    encoded[:, :, 1:] = volume[:, :, 1:] - volume[:, :, :-1]
    return encoded


def delta_x_decode(encoded):
    return (np.cumsum(encoded, axis=2, dtype=np.uint16) & 255).astype(np.uint8)


def save_lossless_cache(path, volume, mask):
    np.savez_compressed(path, volume_dx=delta_x_encode(volume), mask=mask)


def load_lossless_cache(path):
    with np.load(path) as cached:
        return delta_x_decode(cached["volume_dx"]), cached["mask"]


def build_cache_item(uid):
    cv2.setNumThreads(0)
    started = time.time()
    volume, mask, provenance = build_study(uid)
    candidates = safe_candidates(mask, provenance)
    centers = resample_centers(candidates, count=K_EVAL)
    if len(centers) != K_EVAL:
        raise RuntimeError(f"{uid}: no safe exact64 evaluation path")
    cross_series = 0
    zero_slice = 0
    for center in centers:
        tri = provenance.iloc[[center - 1, center, center + 1]]
        zero_slice += int(not bool(mask[center - 1:center + 2].all()))
        cross_series += int(
            bool(tri["series"].isna().any()) or int(tri["series"].nunique()) != 1
        )
    path = CACHE / f"{uid}.npz"
    save_lossless_cache(path, volume, mask)
    decoded, decoded_mask = load_lossless_cache(path)
    pixel_diff = int(np.abs(volume.astype(np.int16) - decoded.astype(np.int16)).max())
    if pixel_diff or not np.array_equal(mask, decoded_mask):
        raise RuntimeError(f"{uid}: cache parity failure")
    return {
        "StudyInstanceUID": uid,
        "candidates": [int(value) for value in candidates],
        "eval_centers": [int(value) for value in centers],
        "safe_candidates": len(candidates),
        "selected_windows": len(centers),
        "cross_series_windows": cross_series,
        "zero_slice_windows": zero_slice,
        "cache_pixel_diff": pixel_diff,
        "cache_bytes": path.stat().st_size,
        "build_seconds": time.time() - started,
    }


BUILD_WORKERS = min(4, max(1, os.cpu_count() or 1))
cache_paths = {}
candidates_by_uid = {}
eval_centers_by_uid = {}


def build_cache(ids, stage):
    started = time.time()
    if BUILD_WORKERS == 1:
        records = [build_cache_item(uid) for uid in ids]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=BUILD_WORKERS) as pool:
            records = list(pool.imap(build_cache_item, ids, chunksize=1))
    rows = []
    for record in records:
        uid = record["StudyInstanceUID"]
        candidates_by_uid[uid] = record.pop("candidates")
        eval_centers_by_uid[uid] = record.pop("eval_centers")
        cache_paths[uid] = CACHE / f"{uid}.npz"
        record["stage"] = stage
        rows.append(record)
    frame = pd.DataFrame(rows)
    elapsed = time.time() - started
    print(f"cache stage {stage}: {len(ids)} studies in {elapsed / 60:.2f} min on "
          f"{BUILD_WORKERS} workers, {frame['cache_bytes'].sum() / (1024 ** 3):.2f} GiB")
    return frame, elapsed
"""
    ),
    md(
        """## Stage A — both checkpoints, then Gold58, before anything expensive

The checkpoints load **first**, before a single study is cached, so a SWA file
that will not load costs twenty seconds rather than half an hour. Gold58 then
runs every arm and three things must hold before Stage B starts: `B0_base`
reproduces `0.9174298724699895`, the projected runtime fits the budget, and the
projected cache fits the free space on scratch."""
    ),
    code(
        r"""
base_model, base_res = load_model(
    str(BASE_WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE, ngpu=1
)
base_model.eval()
swa_model, swa_res, swa_info = load_checkpoint_variant(
    str(SWA_WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE
)
models = {"base": base_model, "swa": swa_model}
divergence = parameter_divergence(base_model, swa_model)
print("swa load", swa_info)
print("parameter divergence", divergence)
if divergence["max_abs_param_diff"] == 0.0:
    raise RuntimeError("the SWA checkpoint holds the base weights; there is no ensemble to test")

gold_cache_metrics, gold_cache_seconds = build_cache(gold_ids, "gold58")

gold_arms, gold_raw, gold_stats = evaluate_arms(models, gold_ids)
gold_windows, gold_seconds = gold_stats["windows"], gold_stats["seconds"]
gold_target_auc = {}
gold_macro = {}
for name in ARM_NAMES:
    per_target, macro = target_aucs(y_gold, gold_arms[name])
    gold_target_auc[name] = per_target
    gold_macro[name] = macro

gold58_b0_gap = gold_macro[BASELINE_ARM] - EXPECTED_SAFE_AUC
reproduced = abs(gold58_b0_gap) <= REPRODUCTION_TOLERANCE
bit_exact_reproduction = bool(gold58_b0_gap == 0.0)
print("Gold58 B0", gold_macro[BASELINE_ARM], "gap", gold58_b0_gap,
      "bit exact", bit_exact_reproduction)
for name in ARM_NAMES:
    print(f"  {name:20s} gold58 {gold_macro[name]:.6f} "
          f"delta {gold_macro[name] - gold_macro[BASELINE_ARM]:+.6f}")
if not reproduced:
    raise RuntimeError(
        f"B0 macro {gold_macro[BASELINE_ARM]} misses the E028 baseline "
        f"{EXPECTED_SAFE_AUC} by {gold58_b0_gap}"
    )

windows_per_second = gold_windows / gold_seconds
windows_per_study = gold_windows / len(gold_ids)
cache_seconds_per_study = gold_cache_seconds / len(gold_ids)
cache_bytes_per_study = float(gold_cache_metrics["cache_bytes"].mean())
projected_cache_minutes = N_EVAL * cache_seconds_per_study / 60.0
projected_eval_minutes = N_EVAL * windows_per_study / windows_per_second / 60.0
projected_total_minutes = (
    (gold_cache_seconds + gold_seconds) / 60.0
    + projected_cache_minutes + projected_eval_minutes
)
runtime_fits = bool(projected_total_minutes <= RUNTIME_BUDGET_MINUTES)
projected_cache_bytes = N_EVAL * cache_bytes_per_study
scratch_free_bytes = float(shutil.disk_usage(SCRATCH).free)
disk_fits = bool(scratch_free_bytes >= projected_cache_bytes * 1.25)
print("throughput", round(windows_per_second, 2), "windows/s;",
      round(windows_per_study, 2), "windows/study")
print("projected minutes: cache", round(projected_cache_minutes, 1),
      "eval", round(projected_eval_minutes, 1),
      "total", round(projected_total_minutes, 1))
print("scratch GiB: free", round(scratch_free_bytes / (1024 ** 3), 2),
      "projected cache", round(projected_cache_bytes / (1024 ** 3), 2))
if not runtime_fits:
    raise RuntimeError(
        f"projected {projected_total_minutes:.0f} min exceeds the "
        f"{RUNTIME_BUDGET_MINUTES:.0f} min budget for {N_EVAL} studies"
    )
if not disk_fits:
    raise RuntimeError(
        f"projected cache {projected_cache_bytes / (1024 ** 3):.1f} GiB does not fit "
        f"in {scratch_free_bytes / (1024 ** 3):.1f} GiB of free scratch"
    )
"""
    ),
    md("## Stage B — the 2,301-study complement"),
    code(
        r"""
eval_cache_metrics, eval_cache_seconds = build_cache(eval_ids, "complement2301")
cache_metrics = pd.concat([gold_cache_metrics, eval_cache_metrics], ignore_index=True)
cache_metrics.to_csv(WORK / "e038_cache_metrics.csv", index=False)
cross_series_window_count = int(cache_metrics["cross_series_windows"].sum())
zero_slice_window_count = int(cache_metrics["zero_slice_windows"].sum())
max_cache_pixel_diff = int(cache_metrics["cache_pixel_diff"].max())
cache_gib = float(cache_metrics["cache_bytes"].sum()) / (1024 ** 3)
mean_safe_candidates = float(cache_metrics["safe_candidates"].mean())
min_safe_candidates = int(cache_metrics["safe_candidates"].min())
short_pool_expected = int((cache_metrics["safe_candidates"] < K_EVAL).sum())
short_pool_share = short_pool_expected / len(cache_metrics)
print("unsafe", cross_series_window_count, zero_slice_window_count,
      "pixel max", max_cache_pixel_diff, "GiB", cache_gib,
      "candidates mean", mean_safe_candidates, "min", min_safe_candidates,
      "short pool", short_pool_expected, round(short_pool_share, 4))

eval_arms, eval_raw, eval_stats = evaluate_arms(models, eval_ids)
eval_windows, eval_seconds = eval_stats["windows"], eval_stats["seconds"]
eval_target_auc = {}
eval_macro = {}
for name in ARM_NAMES:
    per_target, macro = target_aucs(y_eval, eval_arms[name])
    eval_target_auc[name] = per_target
    eval_macro[name] = macro
print("complement arms in", round(eval_seconds / 60, 2), "min")
for name in ARM_NAMES:
    print(f"  {name:20s} complement {eval_macro[name]:.6f} "
          f"delta {eval_macro[name] - eval_macro[BASELINE_ARM]:+.6f}")

# How much room the ensemble ever had, measured on the selection split.
ensemble_divergence = prediction_divergence(eval_arms["B0_base"], eval_arms["B1_swa"])
# Whether the SWA arms are separable from the baseline is the question E038
# asks, not a wiring invariant, so it is reported here rather than gated. Only
# B4 has to differ by construction: it pools more windows from one checkpoint.
ensemble_divergence["swa_arms_differ_from_baseline"] = bool(all(
    not np.allclose(eval_arms[name], eval_arms["B0_base"])
    for name in ["B1_swa", "B2_ens_prob", "B3_ens_logit", "B5_ens_prob_all"]
))
print("base vs swa: mean spearman", round(ensemble_divergence["mean_spearman"], 6),
      "min", round(ensemble_divergence["min_spearman"], 6),
      "mean |dp|", round(ensemble_divergence["mean_abs_prob_diff"], 6),
      "arms separable", ensemble_divergence["swa_arms_differ_from_baseline"])

np.savez_compressed(
    WORK / "e038_predictions.npz",
    eval_ids=np.asarray(eval_ids),
    y_eval=y_eval,
    gold_ids=np.asarray(gold_ids),
    y_gold=y_gold,
    **{f"eval__{name}": eval_arms[name].astype(np.float32) for name in ARM_NAMES},
    **{f"gold__{name}": gold_arms[name].astype(np.float32) for name in ARM_NAMES},
)
del gold_raw, eval_raw
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
"""
    ),
    md(
        """## Paired estimator, per-arm gates, and the decision

One bootstrap seed is shared by every arm, so all arms are resampled on
identical study draws and the comparison between arms is paired as well.

An arm is promoted only if it clears the gain, the win rate, a positive interval
floor, and no per-target collapse on the selection split, **and** the Gold58 sign
agrees. `estimator_resolves_threshold` separately records whether the measured
interval half-width is small enough for `MIN_VALID_GAIN` to mean anything; if no
arm's estimator can resolve it the run stops with `STOP_ENSEMBLE_UNDERPOWERED`
rather than reporting a confident null."""
    ),
    code(
        r"""
baseline_eval = eval_arms[BASELINE_ARM]
baseline_gold = gold_arms[BASELINE_ARM]
null_bootstrap = paired_bootstrap(y_eval, baseline_eval, baseline_eval.copy(),
                                  BOOTSTRAP_SEED, repeats=200)

arm_records = []
arm_rows = []
target_rows = []
for name in ARM_NAMES:
    eval_delta = eval_target_auc[name] - eval_target_auc[BASELINE_ARM]
    gold_delta = gold_target_auc[name] - gold_target_auc[BASELINE_ARM]
    for index, target in enumerate(TARGETS):
        target_rows.append({
            "arm": name,
            "target": target,
            "eval_positives": int(y_eval[:, index].sum()),
            "eval_auc": float(eval_target_auc[name][index]),
            "eval_delta": float(eval_delta[index]),
            "gold_auc": float(gold_target_auc[name][index]),
            "gold_delta": float(gold_delta[index]),
        })
    if name == BASELINE_ARM:
        bootstrap = null_bootstrap
    else:
        bootstrap = paired_bootstrap(y_eval, baseline_eval, eval_arms[name], BOOTSTRAP_SEED)
    macro_delta = eval_macro[name] - eval_macro[BASELINE_ARM]
    gold_macro_delta = gold_macro[name] - gold_macro[BASELINE_ARM]
    gates = {
        "gain": macro_delta >= MIN_VALID_GAIN,
        "bootstrap_win": bootstrap["win_rate"] >= MIN_BOOTSTRAP_WIN,
        "ci_low_positive": bootstrap["ci95"][0] > 0.0,
        "no_target_collapse": float(eval_delta.min()) >= -MAX_TARGET_DROP,
        "gold58_sign_agrees": gold_macro_delta > 0.0,
        "gold58_no_target_collapse": float(gold_delta.min()) >= GOLD58_MIN_TARGET_DELTA,
    }
    resolves = bootstrap["ci95_halfwidth"] <= MIN_VALID_GAIN
    passed = bool(name != BASELINE_ARM and all(gates.values()))
    record = {
        "arm": name,
        "note": ARM_NOTES[name],
        "eval_macro_auc": eval_macro[name],
        "eval_macro_delta": macro_delta,
        "eval_min_target_delta": float(eval_delta.min()),
        "eval_per_target_delta": dict(zip(TARGETS, map(float, eval_delta))),
        "gold58_macro_auc": gold_macro[name],
        "gold58_macro_delta": gold_macro_delta,
        "gold58_min_target_delta": float(gold_delta.min()),
        "bootstrap": bootstrap,
        "estimator_resolves_threshold": bool(resolves),
        "gates": gates,
        "passed": passed,
    }
    arm_records.append(record)
    arm_rows.append({
        "arm": name,
        "eval_macro_auc": eval_macro[name],
        "eval_macro_delta": macro_delta,
        "bootstrap_win": bootstrap["win_rate"],
        "ci95_low": bootstrap["ci95"][0],
        "ci95_high": bootstrap["ci95"][1],
        "ci95_halfwidth": bootstrap["ci95_halfwidth"],
        "eval_min_target_delta": float(eval_delta.min()),
        "gold58_macro_auc": gold_macro[name],
        "gold58_macro_delta": gold_macro_delta,
        "estimator_resolves_threshold": bool(resolves),
        "passed": passed,
    })
pd.DataFrame(arm_rows).to_csv(WORK / "e038_arm_metrics.csv", index=False)
pd.DataFrame(target_rows).to_csv(WORK / "e038_target_metrics.csv", index=False)

# Does E037's A1 survive out of sample? Reported, never gated - E038's decision
# is about the ensemble, and B4 is here because it costs nothing extra.
b4_record = next(record for record in arm_records if record["arm"] == "B4_base_all")
b4_ci = b4_record["bootstrap"]["ci95"]
e037_a1_replication = {
    "e037_delta": E037_A1_DELTA,
    "e037_ci95": E037_A1_CI95,
    "e038_delta": b4_record["eval_macro_delta"],
    "e038_ci95": b4_ci,
    "e038_delta_inside_e037_ci": bool(
        E037_A1_CI95[0] <= b4_record["eval_macro_delta"] <= E037_A1_CI95[1]
    ),
    "intervals_overlap": bool(b4_ci[0] <= E037_A1_CI95[1] and E037_A1_CI95[0] <= b4_ci[1]),
    "sign_agrees": bool(b4_record["eval_macro_delta"] > 0.0),
    "splits_disjoint": True,
}

mechanical_gates = {
    "base_checkpoint_name_exact": BASE_WEIGHT.name == BASE_CHECKPOINT,
    "swa_checkpoint_name_exact": SWA_WEIGHT.name == SWA_CHECKPOINT,
    "base_checkpoint_sha256_exact": base_sha256 == EXPECTED_BASE_SHA256,
    "swa_checkpoint_sha256_recorded": len(swa_sha256) == 64,
    "swa_sha256_differs_from_base": swa_sha256 != base_sha256,
    "swa_arch_matches_base": swa_info["arch"] == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "swa_res_matches_base": swa_res == base_res,
    "swa_weights_differ_from_base": divergence["max_abs_param_diff"] > 0.0,
    "e037_split_reconstructed": e037_split_reconstructed,
    "pool_size_exact": len(labelled_non_gold) == EXPECTED_POOL,
    "eval_count_exact": N_EVAL == EXPECTED_N_EVAL,
    "eval_disjoint_from_e037": not set(eval_ids) & set(e037_eval_ids),
    "gold_ids_exact": gold_ids_sha256 == EXPECTED_GOLD58_SHA256,
    "gold_overlap_zero": not gold_set & set(eval_ids),
    "eval_all_targets_two_class": eval_two_class,
    "baseline_reproduced": reproduced,
    "cache_pixel_max_abs_zero": max_cache_pixel_diff == 0,
    "cross_series_window_count_zero": cross_series_window_count == 0,
    "zero_slice_window_count_zero": zero_slice_window_count == 0,
    "short_pool_studies_accounted": (
        gold_stats["short_pool_studies"] + eval_stats["short_pool_studies"]
        == short_pool_expected
    ),
    "runtime_projection_fits": runtime_fits,
    "scratch_disk_fits": disk_fits,
    "null_arm_delta_zero": abs(null_bootstrap["mean_delta"]) < 1e-12,
    "null_arm_never_wins": null_bootstrap["win_rate"] == 0.0,
    "all_arms_finite": bool(all(np.isfinite(eval_arms[name]).all() for name in ARM_NAMES)),
    # B4 pools more windows from the base checkpoint alone, so it must differ
    # whatever the SWA file holds. The SWA arms are deliberately not here: two
    # members that predict alike is E038's null result, not broken wiring.
    "base_all_arm_distinct": bool(not np.allclose(eval_arms["B4_base_all"], baseline_eval)),
}
mechanical_passed = bool(all(mechanical_gates.values()))

candidate_records = [record for record in arm_records if record["arm"] != BASELINE_ARM]
passing = [record for record in candidate_records if record["passed"]]
any_resolves = bool(any(record["estimator_resolves_threshold"]
                        for record in candidate_records))
best_record = max(candidate_records, key=lambda record: record["eval_macro_delta"])
best_arm = max(passing, key=lambda record: record["eval_macro_delta"])["arm"] if passing else None

if not mechanical_passed:
    decision = "STOP_MECHANICAL_FAILED"
elif passing:
    decision = "PASS_ENSEMBLE_PROMOTE"
elif not any_resolves:
    decision = "STOP_ENSEMBLE_UNDERPOWERED"
else:
    decision = "STOP_ENSEMBLE_NO_GAIN"

summary = {
    "experiment": (
        "E038 inference-only base+SWA ensemble scout on E037's 2301-study "
        "complement plus Gold58"
    ),
    "decision": decision,
    "best_arm": best_arm,
    "best_measured_arm": best_record["arm"],
    "best_measured_delta": best_record["eval_macro_delta"],
    "family_size": len(candidate_records),
    "mechanical_gates": mechanical_gates,
    "mechanical_passed": mechanical_passed,
    "arms": arm_records,
    "arm_names": ARM_NAMES,
    "estimator": {
        "min_ci95_halfwidth": min(record["bootstrap"]["ci95_halfwidth"]
                                  for record in candidate_records),
        "median_ci95_halfwidth": float(np.median([record["bootstrap"]["ci95_halfwidth"]
                                                  for record in candidate_records])),
        "max_ci95_halfwidth": max(record["bootstrap"]["ci95_halfwidth"]
                                  for record in candidate_records),
        "any_arm_resolves_threshold": any_resolves,
        "e037_reference_halfwidth_n2048": 0.000849,
        "eval_positives_per_target": eval_positives,
    },
    "ensemble_divergence": ensemble_divergence,
    "parameter_divergence": divergence,
    "swa_load": swa_info,
    "e037_a1_replication": e037_a1_replication,
    "arm_construction": {
        "gold58": gold_stats,
        "complement": eval_stats,
        "raw_logit_sets": list(RAW_KEYS),
        "arm_notes": ARM_NOTES,
    },
    "thresholds": {
        "min_valid_gain": MIN_VALID_GAIN,
        "min_bootstrap_win": MIN_BOOTSTRAP_WIN,
        "max_target_drop": MAX_TARGET_DROP,
        "gold58_min_target_delta": GOLD58_MIN_TARGET_DELTA,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
    },
    "identity": {
        "base_checkpoint_sha256": base_sha256,
        "swa_checkpoint_sha256": swa_sha256,
        "eval_seed": EVAL_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "eval_ids_sha256": eval_ids_sha256,
        "gold_ids_sha256": gold_ids_sha256,
        "e037_split_sha256": e037_split_sha256,
    },
    "configuration": {
        "n_eval": N_EVAL,
        "k_eval": K_EVAL,
        "resolution": RES,
        "crop_mm": CROP_MM,
        "model_labels": MODEL_LABELS,
        "window_sets": WINDOW_SETS,
        "chunk": CHUNK,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "trained": False,
    },
    "reproduction": {
        "gold58_b0_macro_auc": gold_macro[BASELINE_ARM],
        "gold58_expected": EXPECTED_SAFE_AUC,
        "gold58_b0_gap": gold58_b0_gap,
        "bit_exact": bit_exact_reproduction,
    },
    "counts": {"eval": N_EVAL, "gold": len(gold_ids),
               "cache": N_EVAL + len(gold_ids),
               "labelled_non_gold_pool": len(labelled_non_gold),
               "e037_eval": len(e037_eval_ids),
               "unlabelled_dropped": unlabelled_dropped},
    "window_policy": {"stored_slices": 64, "cross_series": cross_series_window_count,
                      "zero_slice": zero_slice_window_count,
                      "mean_safe_candidates": mean_safe_candidates,
                      "min_safe_candidates": min_safe_candidates,
                      "short_pool_studies": short_pool_expected,
                      "short_pool_share": short_pool_share},
    "resources": {
        "gold_cache_minutes": gold_cache_seconds / 60.0,
        "eval_cache_minutes": eval_cache_seconds / 60.0,
        "gold_arm_minutes": gold_seconds / 60.0,
        "eval_arm_minutes": eval_seconds / 60.0,
        "windows_per_second": windows_per_second,
        "windows_per_study": windows_per_study,
        "projected_total_minutes": projected_total_minutes,
        "runtime_budget_minutes": RUNTIME_BUDGET_MINUTES,
        "cache_gib": cache_gib,
        "scratch_free_gib": scratch_free_bytes / (1024 ** 3),
        "build_workers": BUILD_WORKERS,
        "peak_gpu_gib": (float(torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3))
                         if DEVICE.type == "cuda" else 0.0),
    },
    "submission_created": False,
    "checkpoint_exported": False,
    "feature_cache_exported": False,
}
(WORK / "e038_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

gate_rows = [{"group": "mechanical", "gate": name, "passed": value}
             for name, value in mechanical_gates.items()]
for record in arm_records:
    if record["arm"] == BASELINE_ARM:
        continue
    for name, value in record["gates"].items():
        gate_rows.append({"group": record["arm"], "gate": name, "passed": bool(value)})
pd.DataFrame(gate_rows).to_csv(WORK / "e038_gates.csv", index=False)

print(json.dumps({key: value for key, value in summary.items()
                  if key not in {"arms", "identity"}}, indent=2, sort_keys=True))
for record in arm_records:
    bootstrap = record["bootstrap"]
    print(f"{record['arm']:20s} eval {record['eval_macro_delta']:+.6f} "
          f"win {bootstrap['win_rate']:.3f} "
          f"ci95 [{bootstrap['ci95'][0]:+.6f}, {bootstrap['ci95'][1]:+.6f}] "
          f"half {bootstrap['ci95_halfwidth']:.6f} "
          f"gold {record['gold58_macro_delta']:+.6f} "
          f"{'PASS' if record['passed'] else ''}")
print("E038 DECISION", decision, "best arm", best_arm)

del base_model, swa_model, models
shutil.rmtree(CACHE, ignore_errors=True)
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("E038 COMPLETE submission=False checkpoint_exported=False trained=False")
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
        },
    },
)
nbf.write(notebook, OUT)
print(OUT)
