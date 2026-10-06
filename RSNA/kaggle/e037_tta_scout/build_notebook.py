#!/usr/bin/env python3
"""Build the E037 inference-only TTA scout notebook."""

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
OUT = HERE / "e037-tta-scout.ipynb"
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
# E037 — inference-only test-time augmentation scout

E036 closed the fine-tuning lever: the CC0 consensus the model trains on scores
`0.8921` on the official image-adjudicated Gold58 labels while the checkpoint
itself scores `0.9174`, so every gradient step on those labels pulls the model
back toward its supervision. The two remaining levers that need **no new
supervision** are fold ensembling and test-time augmentation. E037 measures the
second one.

Nothing is trained here. `raptor_ft_coatnet_v4_full.pt` is loaded once, frozen,
and asked the same question several ways.

## Why the arms are nearly free

`RaptorClassifier.head` pools windows with `softmax(..., dim=1)` over `K`, so it
is permutation-invariant and accepts any window count. Backbone features can
therefore be computed **once per window** and reused by every arm that only
changes *which* windows are pooled. Only the arms that change *pixels* — the
horizontal flip and the tighter crop — need their own backbone pass.

The identity bank is built in the order `eval_centers + remaining candidates`
and chunked by six, exactly as `predict_studies` does in E028/E036. Because
every study has at least `K_EVAL = 42` safe candidates, the 42 baseline centres
are always distinct, so the first seven chunks are byte-identical to the proven
path and arm `A0` must reproduce `0.9174298724699895`.

## What is being decided

* **Selection** on a held-out **2,048-study non-Gold split** with CC0 consensus
  labels. AUC is rank-based and both arms of a comparison see the same labels,
  so label noise widens the interval without biasing the sign.
* **Confirmation** on Gold58, which is opened only to check that the sign agrees.
  E036 showed Gold58 alone is blind to a real regression; it is a second
  opinion here, never the selector.
* Nine candidate arms are compared against one baseline, so the win-rate
  threshold is set at `0.98` rather than `0.90` and the family size is recorded
  in the summary.

E036 also left a sizing lesson: at `n=512` the paired macro-AUC interval was
`±0.0046`, barely tighter than Gold58's `±0.005`, because Fracture (29
positives) alone carried 32% of the bootstrap variance. `n=2048` takes Fracture
to roughly 116 positives. This run therefore reports the measured interval
half-width for every arm and **fails closed with a distinct decision** if the
estimator still cannot resolve the threshold it is being asked to judge.

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
# The exact64 cache for 2,106 studies is about 9 GiB and must never be
# collected as kernel output. v1 fell back to /kaggle/working because
# /kaggle/temp did not exist on the image, so the cache is now created rather
# than probed, and /kaggle/working is rejected outright.
SCRATCH = Path("/kaggle/temp")
try:
    SCRATCH.mkdir(parents=True, exist_ok=True)
except OSError:
    SCRATCH = Path(tempfile.mkdtemp(prefix="e037_scratch_"))
if WORK == SCRATCH or WORK in SCRATCH.parents:
    raise RuntimeError(f"scratch {SCRATCH} would be collected as kernel output")
CACHE = SCRATCH / "e037_exact64_cache"
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

EVAL_SEED = 3701
BOOTSTRAP_SEED = 3702
N_EVAL = 2048
CHUNK = 6
BOOTSTRAP_REPEATS = 2000

# Tighter crop arm: 336 * (130 / 140) == 312, so a 12-pixel margin on each side
# of the cached image is exactly a 130 mm field of view.
ZOOM_CROP_MM = 130.0
ZOOM_MARGIN = 12
TRANSFORMS = ["identity", "hflip", "zoom"]
JITTER_SHIFTS = [-0.5, 0.0, 0.5]

BASELINE_ARM = "A0_baseline"
EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
REPRODUCTION_TOLERANCE = 0.0005
RUNTIME_BUDGET_MINUTES = 420.0

# Efficacy thresholds, declared before any run.
MIN_VALID_GAIN = 0.0020
MIN_BOOTSTRAP_WIN = 0.98
MAX_TARGET_DROP = 0.010
GOLD58_MIN_TARGET_DELTA = -0.010

# Horizontal flip mirrors medial and lateral on coronal and axial planes (it
# mirrors anterior/posterior on sagittal). These four targets name a side, so a
# model that uses laterality should lose accuracy on them under the flip.
LATERALITY_TARGETS = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]

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
        """## Transforms, the shared feature bank, and arm assembly

`stack_windows` is the E028/E036 function untouched, and the `identity`
transform calls it directly, so the baseline arm cannot silently drift away
from the path that produced `0.9174`. `hflip` mirrors the resized window and
`zoom` crops before resizing.

Ten arms are assembled from four raw logit sets per study. Everything after the
three backbone passes is head-only arithmetic and costs nothing measurable."""
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


def stack_windows_tta(volume, centers, transform):
    if transform == "identity":
        return stack_windows(volume, centers)
    if transform == "hflip":
        return torch.flip(stack_windows(volume, centers), dims=[-1])
    if transform == "zoom":
        lo, hi = ZOOM_MARGIN, IMG - ZOOM_MARGIN
        windows = np.stack([
            volume[[center - 1, center, center + 1], lo:hi, lo:hi].astype(np.float32) / 255.0
            for center in centers
        ])
        return F.interpolate(torch.from_numpy(windows), size=(RES, RES),
                             mode="bilinear", align_corners=False)
    raise RuntimeError(f"unknown transform {transform!r}")


def jittered_centers(candidates, count, shift):
    '''`resample_centers` with the linspace slid along the candidate list.

    A shift of 0.0 reproduces `resample_centers` exactly, so the jitter arm is
    the baseline draw plus two neighbours rather than a different sampler.
    '''
    if not candidates:
        return []
    positions = np.linspace(0, len(candidates) - 1, count) + shift
    positions = np.clip(positions, 0, len(candidates) - 1).round().astype(int)
    return [int(candidates[position]) for position in positions]


@torch.no_grad()
def window_features(model, volume, centers, transform):
    '''Backbone features for one study, chunked exactly as `predict_studies`.'''
    tensor = stack_windows_tta(volume, centers, transform)
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


RAW_KEYS = (["identity/base", "identity/all"]
            + [f"identity/jitter{index}" for index in range(len(JITTER_SHIFTS))]
            + ["hflip/base", "zoom/base"])

ARM_NAMES = [
    "A0_baseline",
    "A1_windows_all",
    "A2_windows_jitter3",
    "A3_hflip",
    "A4_hflip_avg",
    "A5_zoom",
    "A6_zoom_avg",
    "A7_full_avg",
    "A8_full_avg_logit",
    "A9_all_plus_zoom_avg",
]

ARM_NOTES = {
    "A0_baseline": "42 linspace centres, identity pixels - the E028/E036 path",
    "A1_windows_all": "every safe candidate window pooled by the attention head",
    "A2_windows_jitter3": "probability mean over three slid 42-centre draws",
    "A3_hflip": "horizontal flip only, diagnostic rather than a candidate winner",
    "A4_hflip_avg": "probability mean of identity and horizontal flip",
    "A5_zoom": "130 mm crop only",
    "A6_zoom_avg": "probability mean of 140 mm and 130 mm crops",
    "A7_full_avg": "probability mean of identity, flip, and zoom",
    "A8_full_avg_logit": "same three views averaged in logit space instead",
    "A9_all_plus_zoom_avg": "all-window pooling averaged with the 130 mm crop",
}


def assemble_arms(raw):
    '''Build every arm from the four raw logit sets. Pure arithmetic, no GPU.'''
    probability = {key: sigmoid(value) for key, value in raw.items()}
    arms = {}
    arms["A0_baseline"] = probability["identity/base"]
    arms["A1_windows_all"] = probability["identity/all"]
    arms["A2_windows_jitter3"] = np.mean(
        [probability[f"identity/jitter{index}"] for index in range(len(JITTER_SHIFTS))], axis=0
    )
    arms["A3_hflip"] = probability["hflip/base"]
    arms["A4_hflip_avg"] = (arms["A0_baseline"] + arms["A3_hflip"]) / 2.0
    arms["A5_zoom"] = probability["zoom/base"]
    arms["A6_zoom_avg"] = (arms["A0_baseline"] + arms["A5_zoom"]) / 2.0
    arms["A7_full_avg"] = (arms["A0_baseline"] + arms["A3_hflip"] + arms["A5_zoom"]) / 3.0
    arms["A8_full_avg_logit"] = sigmoid(
        (raw["identity/base"] + raw["hflip/base"] + raw["zoom/base"]) / 3.0
    )
    arms["A9_all_plus_zoom_avg"] = (arms["A1_windows_all"] + arms["A5_zoom"]) / 2.0
    if list(arms) != ARM_NAMES:
        raise RuntimeError(f"arm order drifted: {list(arms)}")
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


@torch.no_grad()
def evaluate_arms(model, ids):
    '''Every arm for every study, one cache read and three backbone passes each.'''
    model.eval()
    raw = {key: np.zeros((len(ids), len(TARGETS)), np.float32) for key in RAW_KEYS}
    windows_seen = 0
    jitter_distinct_studies = 0
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
        position = {center: index for index, center in enumerate(order)}

        identity_features = window_features(model, volume, order, "identity")
        windows_seen += len(order)
        features = {
            "identity/base": identity_features[:K_EVAL],
            "identity/all": identity_features,
        }
        jitter_differs = False
        for index, shift in enumerate(JITTER_SHIFTS):
            centers = jittered_centers(candidates, K_EVAL, shift)
            jitter_differs |= centers != list(eval_centers)
            features[f"identity/jitter{index}"] = identity_features[
                [position[center] for center in centers]
            ]
        jitter_distinct_studies += int(jitter_differs)
        for transform in ("hflip", "zoom"):
            features[f"{transform}/base"] = window_features(
                model, volume, eval_centers, transform
            )
            windows_seen += K_EVAL

        for key in RAW_KEYS:
            raw[key][row] = head_logits(model, features[key])

    for key, values in raw.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite logits in {key}")
    arms = assemble_arms(raw)
    stats = {
        "studies": len(ids),
        "windows": windows_seen,
        "seconds": time.time() - started,
        "windows_per_study": windows_seen / max(len(ids), 1),
        # A jitter shift that rounds back onto the baseline draw makes A2
        # collapse toward A0 for that study, so the share that really moved is
        # recorded rather than assumed.
        "jitter_distinct_studies": jitter_distinct_studies,
        # Studies with fewer than K_EVAL safe candidates: their bank holds
        # repeated windows, A1 collapses onto A0, and the count is reported so
        # the arm deltas are read against a known share of degenerate rows.
        "short_pool_studies": short_pool_studies,
    }
    return arms, raw, stats
"""
    ),
    md("## Identity, the deterministic 2,048-study non-Gold evaluation split, and CC0 labels"),
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

# E036 asserted every drawn study carried a CC0 label, which held at 1,024. At
# 2,048 the pool is filtered first so a handful of unlabelled studies cannot
# turn into an assertion failure three hours into the run.
non_gold_ids = [uid for uid in corpus_ids if uid not in gold_set]
labelled_non_gold = [uid for uid in non_gold_ids if uid in soft.index]
unlabelled_dropped = len(non_gold_ids) - len(labelled_non_gold)
if len(labelled_non_gold) < N_EVAL:
    raise RuntimeError(
        f"need {N_EVAL} labelled non-Gold studies, have {len(labelled_non_gold)}"
    )

split_rng = np.random.default_rng(EVAL_SEED)
shuffled = list(
    np.asarray(labelled_non_gold, dtype=object)[split_rng.permutation(len(labelled_non_gold))]
)
eval_ids = [str(uid) for uid in shuffled[:N_EVAL]]
assert len(eval_ids) == N_EVAL
assert not gold_set & set(eval_ids)

y_eval_soft = np.stack([soft.loc[uid, TARGETS].to_numpy(np.float32) for uid in eval_ids])
y_eval = (y_eval_soft >= 0.5).astype(np.float32)
eval_two_class = bool((y_eval.sum(0) > 0).all() and (y_eval.sum(0) < len(eval_ids)).all())
eval_positives = dict(zip(TARGETS, y_eval.sum(0).astype(int).tolist()))
print("evaluation positives per target", eval_positives)

WEIGHT = find_input_file("raptor_ft_coatnet_v4_full.pt", ("raptor-knee", "widedense"))
checkpoint_sha256 = sha256_file(WEIGHT)
eval_ids_sha256 = sha256_ids(eval_ids)
gold_ids_sha256 = sha256_ids(gold_ids)
identity = {
    "checkpoint_name": WEIGHT.name,
    "checkpoint_sha256": checkpoint_sha256,
    "all_ids_sha256": sha256_file(all_ids_path),
    "extra_ids_sha256": sha256_file(extra_ids_path),
    "eval_seed": EVAL_SEED,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "eval_ids_sha256": eval_ids_sha256,
    "gold_ids_sha256": gold_ids_sha256,
    "labelled_non_gold_pool": len(labelled_non_gold),
    "unlabelled_dropped": unlabelled_dropped,
    "eval_ids": eval_ids,
    "gold_ids": gold_ids,
}
(WORK / "e037_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
print("eval", len(eval_ids), "gold", len(gold_ids), "pool", len(labelled_non_gold),
      "dropped", unlabelled_dropped)
print("checkpoint", WEIGHT.name, checkpoint_sha256)
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
        """## Stage A — Gold58 first, so a broken run costs two minutes

E036 v1 spent twenty-six minutes on the cache and the zero-step baseline before
failing on the first training step. Stage A caches only the 58 Gold studies,
runs every arm on them, and then does two things before the expensive stage
begins: it checks that `A0` reproduces `0.9174298724699895`, and it turns the
measured throughput into a projected total runtime and refuses to continue if
that exceeds the budget."""
    ),
    code(
        r"""
gold_cache_metrics, gold_cache_seconds = build_cache(gold_ids, "gold58")

model, checkpoint_res = load_model(
    str(WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE, ngpu=1
)
model.eval()

gold_arms, gold_raw, gold_stats = evaluate_arms(model, gold_ids)
gold_windows, gold_seconds = gold_stats["windows"], gold_stats["seconds"]
gold_target_auc = {}
gold_macro = {}
for name in ARM_NAMES:
    per_target, macro = target_aucs(y_gold, gold_arms[name])
    gold_target_auc[name] = per_target
    gold_macro[name] = macro

gold58_a0_gap = gold_macro[BASELINE_ARM] - EXPECTED_SAFE_AUC
reproduced = abs(gold58_a0_gap) <= REPRODUCTION_TOLERANCE
bit_exact_reproduction = bool(gold58_a0_gap == 0.0)
print("Gold58 A0", gold_macro[BASELINE_ARM], "gap", gold58_a0_gap,
      "bit exact", bit_exact_reproduction)
for name in ARM_NAMES:
    print(f"  {name:24s} gold58 {gold_macro[name]:.6f} "
          f"delta {gold_macro[name] - gold_macro[BASELINE_ARM]:+.6f}")
if not reproduced:
    raise RuntimeError(
        f"A0 macro {gold_macro[BASELINE_ARM]} misses the E028 baseline "
        f"{EXPECTED_SAFE_AUC} by {gold58_a0_gap}"
    )

windows_per_second = gold_windows / gold_seconds
windows_per_study = gold_windows / len(gold_ids)
cache_seconds_per_study = gold_cache_seconds / len(gold_ids)
projected_cache_minutes = N_EVAL * cache_seconds_per_study / 60.0
projected_eval_minutes = N_EVAL * windows_per_study / windows_per_second / 60.0
projected_total_minutes = (
    (gold_cache_seconds + gold_seconds) / 60.0
    + projected_cache_minutes + projected_eval_minutes
)
runtime_fits = bool(projected_total_minutes <= RUNTIME_BUDGET_MINUTES)
print("throughput", round(windows_per_second, 2), "windows/s;",
      round(windows_per_study, 2), "windows/study")
print("projected minutes: cache", round(projected_cache_minutes, 1),
      "eval", round(projected_eval_minutes, 1),
      "total", round(projected_total_minutes, 1))
if not runtime_fits:
    raise RuntimeError(
        f"projected {projected_total_minutes:.0f} min exceeds the "
        f"{RUNTIME_BUDGET_MINUTES:.0f} min budget for {N_EVAL} studies"
    )
"""
    ),
    md("## Stage B — the 2,048-study selection split"),
    code(
        r"""
eval_cache_metrics, eval_cache_seconds = build_cache(eval_ids, "eval2048")
cache_metrics = pd.concat([gold_cache_metrics, eval_cache_metrics], ignore_index=True)
cache_metrics.to_csv(WORK / "e037_cache_metrics.csv", index=False)
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

eval_arms, eval_raw, eval_stats = evaluate_arms(model, eval_ids)
eval_windows, eval_seconds = eval_stats["windows"], eval_stats["seconds"]
eval_target_auc = {}
eval_macro = {}
for name in ARM_NAMES:
    per_target, macro = target_aucs(y_eval, eval_arms[name])
    eval_target_auc[name] = per_target
    eval_macro[name] = macro
print("eval2048 arms in", round(eval_seconds / 60, 2), "min")
for name in ARM_NAMES:
    print(f"  {name:24s} eval2048 {eval_macro[name]:.6f} "
          f"delta {eval_macro[name] - eval_macro[BASELINE_ARM]:+.6f}")

np.savez_compressed(
    WORK / "e037_predictions.npz",
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

The bootstrap uses one seed for every arm, so all arms are resampled on
identical study draws and the comparison between arms is paired as well.

An arm is promoted only if it clears the gain, the win rate, a positive
interval floor, and no per-target collapse on the selection split, **and** the
Gold58 sign agrees. Separately, `estimator_resolves_threshold` records whether
the measured interval half-width is small enough for `MIN_VALID_GAIN` to mean
anything — if no arm's estimator can resolve it, the run stops with
`STOP_TTA_UNDERPOWERED` rather than pretending the null result was informative."""
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
pd.DataFrame(arm_rows).to_csv(WORK / "e037_arm_metrics.csv", index=False)
pd.DataFrame(target_rows).to_csv(WORK / "e037_target_metrics.csv", index=False)

# What the flip says about laterality, reported whether or not the flip helps.
hflip_eval_delta = eval_target_auc["A3_hflip"] - eval_target_auc[BASELINE_ARM]
laterality_index = [TARGETS.index(target) for target in LATERALITY_TARGETS]
other_index = [index for index in range(len(TARGETS)) if index not in laterality_index]
laterality_probe = {
    "per_target_eval_delta": {target: float(hflip_eval_delta[TARGETS.index(target)])
                              for target in LATERALITY_TARGETS},
    "mean_eval_delta_laterality_targets": float(hflip_eval_delta[laterality_index].mean()),
    "mean_eval_delta_other_targets": float(hflip_eval_delta[other_index].mean()),
}

mechanical_gates = {
    "checkpoint_name_exact": WEIGHT.name == "raptor_ft_coatnet_v4_full.pt",
    "checkpoint_sha256_recorded": len(checkpoint_sha256) == 64,
    "eval_count_exact": len(eval_ids) == N_EVAL,
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
    "null_arm_delta_zero": abs(null_bootstrap["mean_delta"]) < 1e-12,
    "null_arm_never_wins": null_bootstrap["win_rate"] == 0.0,
    "all_arms_finite": bool(all(np.isfinite(eval_arms[name]).all() for name in ARM_NAMES)),
    "arms_are_distinct": bool(all(
        not np.allclose(eval_arms[name], baseline_eval)
        for name in ARM_NAMES if name != BASELINE_ARM
    )),
    "zoom_crop_mm_exact": IMG - 2 * ZOOM_MARGIN == round(IMG * ZOOM_CROP_MM / CROP_MM),
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
    decision = "PASS_TTA_PROMOTE"
elif not any_resolves:
    decision = "STOP_TTA_UNDERPOWERED"
else:
    decision = "STOP_TTA_NO_GAIN"

summary = {
    "experiment": "E037 inference-only TTA scout on 2048 non-Gold studies plus Gold58",
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
        "e036_reference_halfwidth_n512": 0.0046,
        "eval_positives_per_target": eval_positives,
    },
    "laterality_probe": laterality_probe,
    "arm_construction": {
        "gold58": gold_stats,
        "eval2048": eval_stats,
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
        "checkpoint_sha256": checkpoint_sha256,
        "eval_seed": EVAL_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "eval_ids_sha256": eval_ids_sha256,
        "gold_ids_sha256": gold_ids_sha256,
    },
    "configuration": {
        "n_eval": N_EVAL,
        "k_eval": K_EVAL,
        "resolution": RES,
        "crop_mm": CROP_MM,
        "zoom_crop_mm": ZOOM_CROP_MM,
        "zoom_margin": ZOOM_MARGIN,
        "transforms": TRANSFORMS,
        "jitter_shifts": JITTER_SHIFTS,
        "chunk": CHUNK,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "trained": False,
    },
    "reproduction": {
        "gold58_a0_macro_auc": gold_macro[BASELINE_ARM],
        "gold58_expected": EXPECTED_SAFE_AUC,
        "gold58_a0_gap": gold58_a0_gap,
        "bit_exact": bit_exact_reproduction,
    },
    "counts": {"eval": len(eval_ids), "gold": len(gold_ids),
               "cache": len(eval_ids) + len(gold_ids),
               "labelled_non_gold_pool": len(labelled_non_gold),
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
        "build_workers": BUILD_WORKERS,
        "peak_gpu_gib": (float(torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3))
                         if DEVICE.type == "cuda" else 0.0),
    },
    "submission_created": False,
    "checkpoint_exported": False,
    "feature_cache_exported": False,
}
(WORK / "e037_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

gate_rows = [{"group": "mechanical", "gate": name, "passed": value}
             for name, value in mechanical_gates.items()]
for record in arm_records:
    if record["arm"] == BASELINE_ARM:
        continue
    for name, value in record["gates"].items():
        gate_rows.append({"group": record["arm"], "gate": name, "passed": bool(value)})
pd.DataFrame(gate_rows).to_csv(WORK / "e037_gates.csv", index=False)

print(json.dumps({key: value for key, value in summary.items()
                  if key not in {"arms", "identity"}}, indent=2, sort_keys=True))
for record in arm_records:
    bootstrap = record["bootstrap"]
    print(f"{record['arm']:24s} eval {record['eval_macro_delta']:+.6f} "
          f"win {bootstrap['win_rate']:.3f} "
          f"ci95 [{bootstrap['ci95'][0]:+.6f}, {bootstrap['ci95'][1]:+.6f}] "
          f"half {bootstrap['ci95_halfwidth']:.6f} "
          f"gold {record['gold58_macro_delta']:+.6f} "
          f"{'PASS' if record['passed'] else ''}")
print("E037 DECISION", decision, "best arm", best_arm)

del model
shutil.rmtree(CACHE, ignore_errors=True)
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("E037 COMPLETE submission=False checkpoint_exported=False trained=False")
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
