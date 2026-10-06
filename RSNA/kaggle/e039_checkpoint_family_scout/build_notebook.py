#!/usr/bin/env python3
"""Build the E039 inference-only checkpoint-family scout notebook."""

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
OUT = HERE / "e039-checkpoint-family-scout.ipynb"
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
# E039 — inference-only checkpoint-family scout

E036 closed fine-tuning, E037 closed test-time augmentation, and E038 found the
first real lever: the **SWA checkpoint**. Its `B5_ens_prob_all` arm cleared every
gate at `+0.00230`, and the post-hoc arm `B6 = 2·B5 − B4` — algebraically the SWA
checkpoint alone, pooling every safe window — measured `+0.00308` on the same
split.

E038 could only see two checkpoints because `raptor-knee-widedense` holds two
files. That premise was wrong. The same author has published **seven** more
single-checkpoint CC0 datasets, and two carry a stronger public claim than
anything we have measured ourselves:

| checkpoint | author's solo claim | author's Gold58 |
|---|---|---|
| `raptor_ft_coatnet_v4_full.pt` | 0.924 @42w, 0.927 @62w — **our anchor** | 0.9167 |
| `raptor_ft_coatnet_v4_full_swa.pt` | 0.930, reported by a competitor | 0.9150 |
| `raptor_ft_coatnet_v5_full_swa.pt` | **0.928**, "best of the set" | 0.9214 |
| `raptor_ft_coatnet_v9_full.pt` | **0.932** | 0.9031 |

All four headers were read locally before this run: same
`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, same `res = 384`, same 486 state-dict
entries. Our loader takes them unchanged.

## Why this cannot be settled from the forum

Every one of those claims was measured on **the author's own preprocessing**, not
ours. v5 was trained on a 64-slice stack spanning 0.02–0.98 of the volume; our
cache spans 0.06–0.94. v9's "fine spacing" geometry is documented nowhere. A
checkpoint fed a stack it was not trained on does not error — the head pools with
`softmax` over `K` and accepts any window count — it **silently degrades**.

So the only question worth asking is: which checkpoint is best *through our
pipeline*, which is exactly what happens when we submit. That is what this run
measures, and nothing else.

## The split is E038's, deliberately

E037 drew 2,048 studies from the 4,349-study labelled non-Gold pool with
`EVAL_SEED = 3701`. E038 verified that draw against E037's recorded SHA256 and
evaluated on the 2,301 studies E037 never touched. E039 reconstructs the **same**
2,301 studies and checks the id hash against E038's.

Reusing E038's split is a deliberate trade. The new checkpoints are judged on
studies E038 already selected on, so a small edge could be split-specific — but
it also means three of E038's arms can be re-run and checked against the numbers
E038 recorded, and without that check nothing here would be comparable to what
came before. Those reproduction arms are therefore **never promoted**; they are
the run's own self-test.

| arm | what it is | ckpts |
|---|---|---|
| `C0_v4_base` | v4 base, the 42 baseline windows — **baseline** | 1 |
| `C1_v4_base_all` | v4 base, every safe window — reproduces E038 `B4` = `+0.00107` | 1 |
| `C2_v4_swa` | v4 SWA, 42 windows — reproduces E038 `B1` = `+0.00208` | 1 |
| `C3_v4_swa_all` | v4 SWA, every safe window — reproduces E038 `B6` = `+0.00308`; the **incumbent** | 1 |
| `C4_v5` | v5 maxspan, 42 windows | 1 |
| `C5_v5_all` | v5 maxspan, every safe window | 1 |
| `C6_v9` | v9 finespacing, 42 windows | 1 |
| `C7_v9_all` | v9 finespacing, every safe window | 1 |
| `C8_ens_v4swa_v5_all` | probability mean of `C3` and `C5` | 2 |
| `C9_ens_v4swa_v9_all` | probability mean of `C3` and `C7` | 2 |
| `C10_ens_four_all` | probability mean of `C1`, `C3`, `C5`, `C7` | 4 |

Every checkpoint runs at both window budgets because the feature bank is built
once per checkpoint and the head is cheap, so `@42` costs nothing extra. It
separates two things that would otherwise be confounded: whether a checkpoint is
better, and whether it merely sees more windows.

## What is being decided

Seven arms — `C4` through `C10` — compete for promotion, on three conditions.

* They must clear E037's and E038's bar against the baseline, **unchanged**:
  `MIN_VALID_GAIN = 0.0020`, win rate `≥ 0.98`, a positive interval floor, no
  per-target collapse. Kaggle discussion 735154 measured a 0.0020 leaderboard
  noise floor, so a smaller gain cannot be told apart on the board anyway.
* They must **beat the incumbent** `C3_v4_swa_all`. A checkpoint that beats the
  0.927 anchor but loses to the SWA arm we already have for free is not worth a
  submission. The paired interval against the incumbent is reported too, so the
  margin can be read rather than assumed.
* Seven candidates against one baseline is a wider family than E038's five, so
  `family_size` is recorded and the win-rate threshold stays at `0.98`.

`checkpoints_required` travels with every arm: an ensemble that wins costs that
multiple of inference on the hidden test set, and that is a submission-budget
question this scout does not get to hide.

## Two verdicts, both declared before the run

E038 vetoed `B1_swa` on `gold58_no_target_collapse`. Two things became known
afterwards. The publisher's training code selects the base checkpoint as
`argmax(gold_auc)` and the SWA checkpoint as the mean of its top three, so
**Gold58 chose the base model** — it is not a neutral referee for
base-versus-anything. And Gold58's macro half-width is about `±0.005`, more than
twice `MIN_VALID_GAIN`, so it cannot resolve the effect under test at all.

Rewriting E038's gate after seeing E038's result would be moving the goalposts,
so it is not rewritten. Instead **both** verdicts are computed and both reported:

* `decision` — **strict**: E038's gate set unchanged, Gold58 still holding a veto.
* `decision_relaxed` — identical, except Gold58 reports instead of vetoing.

If they disagree, that disagreement is the finding, surfaced as
`verdicts_agree: false` rather than resolved here.

## What a STOP means this time

`STOP_INCUMBENT_HOLDS` is not a dead end. It means no new checkpoint beats
`C3_v4_swa_all`, whose `+0.00308` E038 already measured and which we are not
currently submitting — so the submission to make is the v4 SWA checkpoint at full
window coverage, and E039 has told us not to spend one on v5 or v9.

Nothing is trained. Four checkpoints are loaded once, frozen, and asked the same
question. No competition submission, no exported checkpoint, no exported feature
cache.
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
    SCRATCH = Path(tempfile.mkdtemp(prefix="e039_scratch_"))
if WORK == SCRATCH or WORK in SCRATCH.parents:
    raise RuntimeError(f"scratch {SCRATCH} would be collected as kernel output")
CACHE = SCRATCH / "e039_exact64_cache"
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
# draw so this run can take the complement. BOOTSTRAP_SEED is E038's, kept
# unchanged so every arm is resampled on identical study draws and the
# reproduction arms are directly comparable to the numbers E038 recorded.
EVAL_SEED = 3701
BOOTSTRAP_SEED = 3802
CHUNK = 6
BOOTSTRAP_REPEATS = 2000

# Four CC0 checkpoints, one author, one architecture, one head. The hints are
# the substrings `find_input_file` matches against the attached dataset
# directories, so a file arriving from the wrong dataset is a hard failure
# rather than a silently different model.
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
CHECKPOINTS = [
    ("v4_base", "raptor_ft_coatnet_v4_full.pt", ("raptor-knee", "widedense")),
    ("v4_swa", "raptor_ft_coatnet_v4_full_swa.pt", ("raptor-knee", "widedense")),
    ("v5_maxspan", "raptor_ft_coatnet_v5_full_swa.pt", ("raptor-knee", "maxspan")),
    ("v9_finespacing", "raptor_ft_coatnet_v9_full.pt", ("raptor-knee", "finespacing")),
]
MODEL_LABELS = [label for label, _name, _hints in CHECKPOINTS]
CHECKPOINT_FILES = {label: name for label, name, _hints in CHECKPOINTS}
CHECKPOINT_HINTS = {label: hints for label, _name, hints in CHECKPOINTS}
BASE_LABEL = "v4_base"
WINDOW_SETS = ["42", "all"]

# Read locally with `kaggle datasets download -f <file>` before this run, so a
# swapped or truncated file is caught before any GPU time is spent.
EXPECTED_SHA256 = {
    "v4_base": "89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded",
    "v4_swa": "d8bb0f8751b4bb65750257869ddc4c7a3c0cdfc6e62596fc68193919406c53eb",
    "v5_maxspan": "20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a",
    "v9_finespacing": "0850149a98cf04c7bacc1ea99b3af048102a256de14766b129b1d7255c02de7b",
}

BASELINE_ARM = "C0_v4_base"
# The best arm E038 measured on this split. Every new arm has to beat it to be
# worth a submission, because it already costs us nothing extra.
INCUMBENT_ARM = "C3_v4_swa_all"
EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
REPRODUCTION_TOLERANCE = 0.0005
RUNTIME_BUDGET_MINUTES = 420.0

# E037 identity, so the split is provably E037's untouched pool.
E037_EVAL_IDS_SHA256 = (
    "97de5347aa7ee01e88283b3430ab2aaacc3283b9aceab47e02a14945b62c8c93"
)
E037_N_EVAL = 2048
EXPECTED_POOL = 4349
EXPECTED_N_EVAL = EXPECTED_POOL - E037_N_EVAL
# E038 evaluated this same complement, so the id hash has to match exactly.
E038_EVAL_IDS_SHA256 = (
    "70b80686cecebf82817cc2dea96ba06f67bb9f93924b543315efd116a6f3f225"
)

# E038 measured these on this exact split with this exact code. They are re-run
# as the run's own self-test: if they do not come back, nothing measured here is
# comparable to E038 and the run is void. The C3 figure is E038's post-hoc arm
# B6 = 2*B5_ens_prob_all - B4_base_all, recomputed from E038's stored float32
# predictions, which is why it is checked against a tolerance and not exactly.
E038_B0_EVAL_MACRO = 0.9079593249126852
E038_EXPECTED_DELTA = {
    "C1_v4_base_all": 0.0010661988349757179,
    "C2_v4_swa": 0.0020829288947438185,
    "C3_v4_swa_all": 0.0030813625662755584,
}

# E037's measurement of the arm C1 replicates, reported not gated.
E037_A1_DELTA = 0.0010437443710005212
E037_A1_CI95 = [0.0006198022190293764, 0.0015000035609342985]

# Efficacy thresholds, unchanged from E037 and E038, declared before any run.
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
        """## Loading four checkpoints without guessing

`load_model` is used verbatim for `v4_base`, so `C0_v4_base` stays the proven
path and has to reproduce `0.9174298724699895` on Gold58. The other three may
carry averaging bookkeeping, so they get a loader that normalises **only** two
documented things — a leading `module.` prefix and an `n_averaged` entry —
records exactly which of them it applied, and then loads `strict=True`. Anything
else about a file is a hard failure rather than a silently reshaped state dict.

Two checkpoints that turned out to hold the same weights would make an ensemble
arm a copy of a solo arm, so the parameter divergence of **every pair** is
measured and gated."""
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


def divergence_matrix(models):
    '''Every checkpoint against every other, so no pair can be a silent copy.'''
    labels = list(models)
    pairs = {}
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            pairs[f"{left}|{right}"] = parameter_divergence(models[left], models[right])
    return pairs
"""
    ),
    md(
        """## One bank per checkpoint, eight logit sets, eleven arms

`window_features` is `predict_studies`' feature pass: the same `stack_windows`,
the same ImageNet normalisation, the same chunk of six under float16 autocast.
T4 is `sm_75` and has no native bfloat16, so float16 is the only autocast dtype
that keeps a fused attention kernel eligible.

Every checkpoint sees the identical bank in the identical order, because
`RaptorClassifier.head` pools with `softmax(..., dim=1)` over `K` and is
permutation-invariant. The bank is built once per checkpoint in the order
`eval_centers + remaining candidates`, so `@42` reads its first 42 entries and
`@all` reads all of them: the same features, two window budgets, one pass."""
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
    "C0_v4_base",
    "C1_v4_base_all",
    "C2_v4_swa",
    "C3_v4_swa_all",
    "C4_v5",
    "C5_v5_all",
    "C6_v9",
    "C7_v9_all",
    "C8_ens_v4swa_v5_all",
    "C9_ens_v4swa_v9_all",
    "C10_ens_four_all",
]
ARM_NOTES = {
    "C0_v4_base": "v4 base, the 42 baseline windows - the proven path",
    "C1_v4_base_all": "v4 base, every safe window - reproduces E038 B4_base_all",
    "C2_v4_swa": "v4 SWA, the 42 baseline windows - reproduces E038 B1_swa",
    "C3_v4_swa_all": "v4 SWA, every safe window - reproduces E038 post-hoc B6; incumbent",
    "C4_v5": "v5 maxspan, the 42 baseline windows",
    "C5_v5_all": "v5 maxspan, every safe window",
    "C6_v9": "v9 finespacing, the 42 baseline windows",
    "C7_v9_all": "v9 finespacing, every safe window",
    "C8_ens_v4swa_v5_all": "probability mean of C3 and C5",
    "C9_ens_v4swa_v9_all": "probability mean of C3 and C7",
    "C10_ens_four_all": "probability mean of C1, C3, C5 and C7",
}
# How many checkpoints an arm needs at submission time. An ensemble arm that
# wins costs that multiple of inference on the hidden test set, so the number
# travels with the verdict instead of being rediscovered afterwards.
ARM_CHECKPOINTS = {
    "C0_v4_base": ["v4_base"],
    "C1_v4_base_all": ["v4_base"],
    "C2_v4_swa": ["v4_swa"],
    "C3_v4_swa_all": ["v4_swa"],
    "C4_v5": ["v5_maxspan"],
    "C5_v5_all": ["v5_maxspan"],
    "C6_v9": ["v9_finespacing"],
    "C7_v9_all": ["v9_finespacing"],
    "C8_ens_v4swa_v5_all": ["v4_swa", "v5_maxspan"],
    "C9_ens_v4swa_v9_all": ["v4_swa", "v9_finespacing"],
    "C10_ens_four_all": ["v4_base", "v4_swa", "v5_maxspan", "v9_finespacing"],
}
# Arms E038 already measured on this split. They are re-run as reproduction
# checks and never promoted: selecting on a number that has already been seen
# is how E035 lost 0.059 on the board.
REPRODUCTION_ARMS = ["C1_v4_base_all", "C2_v4_swa", "C3_v4_swa_all"]
PROMOTABLE_ARMS = [name for name in ARM_NAMES
                   if name != BASELINE_ARM and name not in REPRODUCTION_ARMS]


def assemble_arms(raw):
    prob = {key: sigmoid(raw[key]) for key in RAW_KEYS}
    arms = {
        "C0_v4_base": prob["v4_base/42"],
        "C1_v4_base_all": prob["v4_base/all"],
        "C2_v4_swa": prob["v4_swa/42"],
        "C3_v4_swa_all": prob["v4_swa/all"],
        "C4_v5": prob["v5_maxspan/42"],
        "C5_v5_all": prob["v5_maxspan/all"],
        "C6_v9": prob["v9_finespacing/42"],
        "C7_v9_all": prob["v9_finespacing/all"],
        "C8_ens_v4swa_v5_all": (prob["v4_swa/all"] + prob["v5_maxspan/all"]) / 2.0,
        "C9_ens_v4swa_v9_all": (prob["v4_swa/all"] + prob["v9_finespacing/all"]) / 2.0,
        "C10_ens_four_all": (prob["v4_base/all"] + prob["v4_swa/all"]
                             + prob["v5_maxspan/all"] + prob["v9_finespacing/all"]) / 4.0,
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


def prediction_divergence(left_predictions, right_predictions):
    '''How much room there is for an ensemble to help, measured not assumed.

    AUC is rank-based, so Spearman is the correlation that matters: two members
    ranking studies identically cannot ensemble into a different AUC whatever
    their probabilities look like.
    '''
    pearson = {}
    spearman = {}
    for index, target in enumerate(TARGETS):
        column_a = left_predictions[:, index]
        column_b = right_predictions[:, index]
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
        "mean_abs_prob_diff": float(np.abs(left_predictions - right_predictions).mean()),
        "max_abs_prob_diff": float(np.abs(left_predictions - right_predictions).max()),
    }


def divergence_pairs(arms, names):
    '''Pairwise prediction divergence, so a null ensemble explains itself.'''
    pairs = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            values = prediction_divergence(arms[left], arms[right])
            pairs[f"{left}|{right}"] = {
                key: value for key, value in values.items()
                if not key.startswith("per_target_")
            }
    return pairs


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

WEIGHTS = {
    label: find_input_file(CHECKPOINT_FILES[label], CHECKPOINT_HINTS[label])
    for label in MODEL_LABELS
}
checkpoint_sha256 = {label: sha256_file(path) for label, path in WEIGHTS.items()}
eval_ids_sha256 = sha256_ids(eval_ids)
gold_ids_sha256 = sha256_ids(gold_ids)
e038_split_matched = eval_ids_sha256 == E038_EVAL_IDS_SHA256
identity = {
    "checkpoint_names": {label: WEIGHTS[label].name for label in MODEL_LABELS},
    "checkpoint_paths": {label: str(WEIGHTS[label]) for label in MODEL_LABELS},
    "checkpoint_sha256": checkpoint_sha256,
    "all_ids_sha256": sha256_file(all_ids_path),
    "extra_ids_sha256": sha256_file(extra_ids_path),
    "eval_seed": EVAL_SEED,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "eval_ids_sha256": eval_ids_sha256,
    "gold_ids_sha256": gold_ids_sha256,
    "e037_split_sha256": e037_split_sha256,
    "e037_split_reconstructed": e037_split_reconstructed,
    "e038_split_matched": e038_split_matched,
    "labelled_non_gold_pool": len(labelled_non_gold),
    "unlabelled_dropped": unlabelled_dropped,
    "eval_ids": eval_ids,
    "gold_ids": gold_ids,
}
(WORK / "e039_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
print("eval", N_EVAL, "gold", len(gold_ids), "pool", len(labelled_non_gold),
      "dropped", unlabelled_dropped, "e037 split verified", e037_split_reconstructed,
      "e038 split matched", e038_split_matched)
for label in MODEL_LABELS:
    digest = checkpoint_sha256[label]
    print(f"  {label:16s} {WEIGHTS[label].name:34s} {digest[:16]} "
          f"{'ok' if digest == EXPECTED_SHA256[label] else 'MISMATCH'}")
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
        """## Stage A — all four checkpoints, then Gold58, before anything expensive

The checkpoints load **first**, before a single study is cached, so a file that
will not load costs a minute rather than half an hour. Gold58 then runs every arm
and three things must hold before Stage B starts: `C0_v4_base` reproduces
`0.9174298724699895`, the projected runtime fits the budget, and the projected
cache fits the free space on scratch.

Four backbone passes per study is twice E038's cost. E038 measured 42.9
windows/s and 104.3 windows/study on a T4, so E039 should land near 209
windows/study — but the projection is built from **this** run's measured
throughput, not from that number, and it stops the run before Stage B if it does
not fit."""
    ),
    code(
        r"""
models = {}
checkpoint_info = {}
for label in MODEL_LABELS:
    if label == BASE_LABEL:
        # `load_model` verbatim, so C0_v4_base stays the E028 path exactly.
        model, checkpoint_res = load_model(str(WEIGHTS[label]), ARCH, RES, DEVICE, ngpu=1)
        model.eval()
        info = {"arch": ARCH, "res": int(checkpoint_res), "top_level_keys": [],
                "normalisations": [], "state_entries": len(model.state_dict()),
                "loader": "load_model"}
    else:
        model, checkpoint_res, info = load_checkpoint_variant(
            str(WEIGHTS[label]), ARCH, RES, DEVICE
        )
        info["loader"] = "load_checkpoint_variant"
    models[label] = model
    checkpoint_info[label] = info
    print("loaded", label, info)

base_res = checkpoint_info[BASE_LABEL]["res"]
divergence = divergence_matrix(models)
identical_pairs = [pair for pair, values in divergence.items()
                   if values["max_abs_param_diff"] == 0.0]
for pair, values in divergence.items():
    print(f"  {pair:34s} max |dw| {values['max_abs_param_diff']:.6f} "
          f"mean |dw| {values['mean_abs_param_diff']:.8f}")
if identical_pairs:
    raise RuntimeError(f"these checkpoints hold identical weights: {identical_pairs}")

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
print("Gold58 C0", gold_macro[BASELINE_ARM], "gap", gold58_b0_gap,
      "bit exact", bit_exact_reproduction)
for name in ARM_NAMES:
    print(f"  {name:20s} gold58 {gold_macro[name]:.6f} "
          f"delta {gold_macro[name] - gold_macro[BASELINE_ARM]:+.6f}")
if not reproduced:
    raise RuntimeError(
        f"C0 macro {gold_macro[BASELINE_ARM]} misses the E028 baseline "
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
cache_metrics.to_csv(WORK / "e039_cache_metrics.csv", index=False)
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

# How much room the ensembles ever had, measured on the selection split. Four
# checkpoints that rank studies alike cannot ensemble into a different AUC, so
# a null result at C8/C9/C10 arrives with its own explanation attached.
ALL_WINDOW_ARMS = ["C1_v4_base_all", "C3_v4_swa_all", "C5_v5_all", "C7_v9_all"]
checkpoint_divergence = divergence_pairs(eval_arms, ALL_WINDOW_ARMS)
print("all-window pairwise divergence:")
for pair, values in checkpoint_divergence.items():
    print(f"  {pair:34s} spearman mean {values['mean_spearman']:.6f} "
          f"min {values['min_spearman']:.6f} "
          f"mean |dp| {values['mean_abs_prob_diff']:.6f}")

# Whether an arm is separable from the baseline is the question E039 asks, not a
# wiring invariant, so it is reported here. Only the arms that change the window
# budget are gated on it later; two checkpoints that predict alike is a finding.
arms_distinct_from_baseline = {
    name: bool(not np.allclose(eval_arms[name], eval_arms[BASELINE_ARM]))
    for name in ARM_NAMES if name != BASELINE_ARM
}
print("arms separable from baseline", sum(arms_distinct_from_baseline.values()),
      "of", len(arms_distinct_from_baseline))

np.savez_compressed(
    WORK / "e039_predictions.npz",
    eval_ids=np.asarray(eval_ids),
    y_eval=y_eval,
    gold_ids=np.asarray(gold_ids),
    y_gold=y_gold,
    **{f"eval__{name}": eval_arms[name].astype(np.float32) for name in ARM_NAMES},
    **{f"gold__{name}": gold_arms[name].astype(np.float32) for name in ARM_NAMES},
)
del gold_raw, eval_raw
"""
    ),
    md(
        """## Paired estimator, per-arm gates, and two decisions

One bootstrap seed is shared by every arm and by both references, so all arms are
resampled on identical study draws and the comparison between arms is paired as
well. The estimator itself is E038's, unchanged, which is what makes the three
reproduction arms a meaningful check.

Each promotable arm is measured twice: against the baseline `C0_v4_base`, which
is the number comparable to E037 and E038, and against the incumbent
`C3_v4_swa_all`, which is the number that decides whether it is worth submitting.

`estimator_resolves_threshold` records whether the measured interval half-width
is small enough for `MIN_VALID_GAIN` to mean anything. If no promotable arm can
resolve it the run stops with `STOP_FAMILY_UNDERPOWERED` rather than reporting a
confident null — the P2 lesson: a threshold sitting inside its own confidence
interval decides nothing."""
    ),
    code(
        r"""
baseline_eval = eval_arms[BASELINE_ARM]
incumbent_eval = eval_arms[INCUMBENT_ARM]
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
    promotable = name in PROMOTABLE_ARMS
    incumbent_bootstrap = (
        paired_bootstrap(y_eval, incumbent_eval, eval_arms[name], BOOTSTRAP_SEED)
        if promotable else None
    )
    macro_delta = eval_macro[name] - eval_macro[BASELINE_ARM]
    incumbent_delta = eval_macro[name] - eval_macro[INCUMBENT_ARM]
    gold_macro_delta = gold_macro[name] - gold_macro[BASELINE_ARM]
    # E037's and E038's bar against the baseline, unchanged, plus the one thing
    # E038 could not ask: whether the arm beats the best arm we already have.
    gates = {
        "gain": macro_delta >= MIN_VALID_GAIN,
        "bootstrap_win": bootstrap["win_rate"] >= MIN_BOOTSTRAP_WIN,
        "ci_low_positive": bootstrap["ci95"][0] > 0.0,
        "no_target_collapse": float(eval_delta.min()) >= -MAX_TARGET_DROP,
        "beats_incumbent": incumbent_delta > 0.0,
    }
    # Gold58 holds 58 studies and a macro half-width near +-0.005, more than
    # twice MIN_VALID_GAIN, and the publisher's own training code picked the
    # base checkpoint as argmax(gold_auc). It keeps its veto in the strict
    # verdict because E038's gate is not being rewritten after the fact; it only
    # reports in the relaxed one. Both are declared here, before any result.
    gold58_gates = {
        "gold58_sign_agrees": gold_macro_delta > 0.0,
        "gold58_no_target_collapse": float(gold_delta.min()) >= GOLD58_MIN_TARGET_DELTA,
    }
    strict_gates = {**gates, **gold58_gates}
    resolves = bootstrap["ci95_halfwidth"] <= MIN_VALID_GAIN
    passed_strict = bool(promotable and all(strict_gates.values()))
    passed_relaxed = bool(promotable and all(gates.values()))
    reproduction = None
    if name in E038_EXPECTED_DELTA:
        gap = macro_delta - E038_EXPECTED_DELTA[name]
        reproduction = {
            "e038_delta": E038_EXPECTED_DELTA[name],
            "e039_delta": macro_delta,
            "gap": gap,
            "reproduced": bool(abs(gap) <= REPRODUCTION_TOLERANCE),
        }
    record = {
        "arm": name,
        "note": ARM_NOTES[name],
        "checkpoints_required": ARM_CHECKPOINTS[name],
        "promotable": promotable,
        "eval_macro_auc": eval_macro[name],
        "eval_macro_delta": macro_delta,
        "eval_delta_vs_incumbent": incumbent_delta,
        "eval_min_target_delta": float(eval_delta.min()),
        "eval_per_target_delta": dict(zip(TARGETS, map(float, eval_delta))),
        "gold58_macro_auc": gold_macro[name],
        "gold58_macro_delta": gold_macro_delta,
        "gold58_min_target_delta": float(gold_delta.min()),
        "bootstrap": bootstrap,
        "bootstrap_vs_incumbent": incumbent_bootstrap,
        "estimator_resolves_threshold": bool(resolves),
        "gates": gates,
        "gold58_gates": gold58_gates,
        "passed_strict": passed_strict,
        "passed_relaxed": passed_relaxed,
        "e038_reproduction": reproduction,
    }
    arm_records.append(record)
    arm_rows.append({
        "arm": name,
        "promotable": promotable,
        "checkpoints_required": len(ARM_CHECKPOINTS[name]),
        "eval_macro_auc": eval_macro[name],
        "eval_macro_delta": macro_delta,
        "eval_delta_vs_incumbent": incumbent_delta,
        "bootstrap_win": bootstrap["win_rate"],
        "ci95_low": bootstrap["ci95"][0],
        "ci95_high": bootstrap["ci95"][1],
        "ci95_halfwidth": bootstrap["ci95_halfwidth"],
        "incumbent_win": None if incumbent_bootstrap is None
                         else incumbent_bootstrap["win_rate"],
        "incumbent_ci95_low": None if incumbent_bootstrap is None
                              else incumbent_bootstrap["ci95"][0],
        "eval_min_target_delta": float(eval_delta.min()),
        "gold58_macro_auc": gold_macro[name],
        "gold58_macro_delta": gold_macro_delta,
        "estimator_resolves_threshold": bool(resolves),
        "passed_strict": passed_strict,
        "passed_relaxed": passed_relaxed,
    })
pd.DataFrame(arm_rows).to_csv(WORK / "e039_arm_metrics.csv", index=False)
pd.DataFrame(target_rows).to_csv(WORK / "e039_target_metrics.csv", index=False)

# The run's own self-test: three arms E038 measured on this exact split, re-run
# here. If they do not come back, nothing else measured here is comparable.
reproduction_records = {record["arm"]: record["e038_reproduction"]
                        for record in arm_records
                        if record["e038_reproduction"] is not None}
e038_arms_reproduced = bool(all(values["reproduced"]
                                for values in reproduction_records.values()))
e038_baseline_gap = eval_macro[BASELINE_ARM] - E038_B0_EVAL_MACRO
e038_baseline_reproduced = bool(abs(e038_baseline_gap) <= REPRODUCTION_TOLERANCE)
e038_reproduction = {
    "baseline_macro": eval_macro[BASELINE_ARM],
    "baseline_expected": E038_B0_EVAL_MACRO,
    "baseline_gap": e038_baseline_gap,
    "baseline_reproduced": e038_baseline_reproduced,
    "arms": reproduction_records,
    "arms_reproduced": e038_arms_reproduced,
    "tolerance": REPRODUCTION_TOLERANCE,
}

# Does E037's A1 survive a third time? Reported, never gated. E039's split is
# E038's, which is disjoint from E037's 2,048 - so this is still out of sample
# with respect to E037, and in sample with respect to E038.
c1_record = next(record for record in arm_records if record["arm"] == "C1_v4_base_all")
c1_ci = c1_record["bootstrap"]["ci95"]
e037_a1_replication = {
    "e037_delta": E037_A1_DELTA,
    "e037_ci95": E037_A1_CI95,
    "e039_delta": c1_record["eval_macro_delta"],
    "e039_ci95": c1_ci,
    "e039_delta_inside_e037_ci": bool(
        E037_A1_CI95[0] <= c1_record["eval_macro_delta"] <= E037_A1_CI95[1]
    ),
    "intervals_overlap": bool(c1_ci[0] <= E037_A1_CI95[1] and E037_A1_CI95[0] <= c1_ci[1]),
    "sign_agrees": bool(c1_record["eval_macro_delta"] > 0.0),
    "disjoint_from_e037_split": True,
    "same_split_as_e038": True,
}

mechanical_gates = {
    "checkpoint_names_exact": bool(all(
        WEIGHTS[label].name == CHECKPOINT_FILES[label] for label in MODEL_LABELS)),
    "checkpoint_sha256_exact": bool(all(
        checkpoint_sha256[label] == EXPECTED_SHA256[label] for label in MODEL_LABELS)),
    "checkpoint_sha256_all_distinct": len(set(checkpoint_sha256.values())) == len(MODEL_LABELS),
    "all_checkpoints_share_arch": bool(all(
        checkpoint_info[label]["arch"] == ARCH for label in MODEL_LABELS)),
    "all_checkpoints_share_res": bool(all(
        checkpoint_info[label]["res"] == base_res for label in MODEL_LABELS)),
    "no_two_checkpoints_identical": not identical_pairs,
    "e037_split_reconstructed": e037_split_reconstructed,
    "e038_split_matched": e038_split_matched,
    "pool_size_exact": len(labelled_non_gold) == EXPECTED_POOL,
    "eval_count_exact": N_EVAL == EXPECTED_N_EVAL,
    "eval_disjoint_from_e037": not set(eval_ids) & set(e037_eval_ids),
    "gold_ids_exact": gold_ids_sha256 == EXPECTED_GOLD58_SHA256,
    "gold_overlap_zero": not gold_set & set(eval_ids),
    "eval_all_targets_two_class": eval_two_class,
    "baseline_reproduced": reproduced,
    "e038_baseline_reproduced": e038_baseline_reproduced,
    "e038_arms_reproduced": e038_arms_reproduced,
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
    # Arms that pool more windows must differ from the baseline whatever the
    # checkpoints hold. Arms that only change the checkpoint are deliberately
    # not here: two models that predict alike is a finding, not broken wiring.
    "window_budget_arms_distinct": bool(all(
        arms_distinct_from_baseline[name] for name in ALL_WINDOW_ARMS)),
}
mechanical_passed = bool(all(mechanical_gates.values()))

promotable_records = [record for record in arm_records if record["promotable"]]
passing_strict = [record for record in promotable_records if record["passed_strict"]]
passing_relaxed = [record for record in promotable_records if record["passed_relaxed"]]
any_resolves = bool(any(record["estimator_resolves_threshold"]
                        for record in promotable_records))
best_record = max(promotable_records, key=lambda record: record["eval_macro_delta"])


def verdict(passing):
    '''One rule applied to both gate sets, so neither can be tuned after the fact.'''
    if not mechanical_passed:
        return "STOP_MECHANICAL_FAILED", None
    if passing:
        return ("PASS_PROMOTE_NEW_CHECKPOINT",
                max(passing, key=lambda record: record["eval_macro_delta"])["arm"])
    if not any_resolves:
        return "STOP_FAMILY_UNDERPOWERED", None
    return "STOP_INCUMBENT_HOLDS", None


decision, best_arm = verdict(passing_strict)
decision_relaxed, best_arm_relaxed = verdict(passing_relaxed)
verdicts_agree = bool(decision == decision_relaxed and best_arm == best_arm_relaxed)
# A STOP is not a dead end here. The incumbent is the v4 SWA checkpoint at full
# window coverage, whose +0.00308 E038 already measured and which we are not
# currently submitting, so it stays the candidate when nothing new beats it.
submission_candidate = best_arm or (INCUMBENT_ARM if mechanical_passed else None)
submission_checkpoints = (ARM_CHECKPOINTS[submission_candidate]
                          if submission_candidate else [])

summary = {
    "experiment": (
        "E039 inference-only four-checkpoint family scout on E038's 2301-study "
        "split plus Gold58"
    ),
    "decision": decision,
    "decision_relaxed": decision_relaxed,
    "verdicts_agree": verdicts_agree,
    "best_arm": best_arm,
    "best_arm_relaxed": best_arm_relaxed,
    "best_measured_arm": best_record["arm"],
    "best_measured_delta": best_record["eval_macro_delta"],
    "best_measured_delta_vs_incumbent": best_record["eval_delta_vs_incumbent"],
    "incumbent_arm": INCUMBENT_ARM,
    "incumbent_macro_delta": next(record["eval_macro_delta"] for record in arm_records
                                  if record["arm"] == INCUMBENT_ARM),
    "submission_candidate": submission_candidate,
    "submission_checkpoints": submission_checkpoints,
    "family_size": len(promotable_records),
    "mechanical_gates": mechanical_gates,
    "mechanical_passed": mechanical_passed,
    "arms": arm_records,
    "arm_names": ARM_NAMES,
    "promotable_arms": PROMOTABLE_ARMS,
    "reproduction_arms": REPRODUCTION_ARMS,
    "estimator": {
        "min_ci95_halfwidth": min(record["bootstrap"]["ci95_halfwidth"]
                                  for record in promotable_records),
        "median_ci95_halfwidth": float(np.median([record["bootstrap"]["ci95_halfwidth"]
                                                  for record in promotable_records])),
        "max_ci95_halfwidth": max(record["bootstrap"]["ci95_halfwidth"]
                                  for record in promotable_records),
        "any_arm_resolves_threshold": any_resolves,
        "e037_reference_halfwidth_n2048": 0.000849,
        "e038_reference_halfwidth_n2301": 0.000545,
        "eval_positives_per_target": eval_positives,
    },
    "checkpoint_divergence": checkpoint_divergence,
    "parameter_divergence": divergence,
    "checkpoint_load": checkpoint_info,
    "arms_distinct_from_baseline": arms_distinct_from_baseline,
    "e038_reproduction": e038_reproduction,
    "e037_a1_replication": e037_a1_replication,
    "arm_construction": {
        "gold58": gold_stats,
        "complement": eval_stats,
        "raw_logit_sets": list(RAW_KEYS),
        "arm_notes": ARM_NOTES,
        "arm_checkpoints": ARM_CHECKPOINTS,
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
        "e037_split_sha256": e037_split_sha256,
        "e038_split_matched": e038_split_matched,
    },
    "configuration": {
        "n_eval": N_EVAL,
        "k_eval": K_EVAL,
        "resolution": RES,
        "crop_mm": CROP_MM,
        "arch": ARCH,
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
               "checkpoints": len(MODEL_LABELS),
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
(WORK / "e039_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

gate_rows = [{"group": "mechanical", "gate": name, "passed": bool(value)}
             for name, value in mechanical_gates.items()]
for record in arm_records:
    if record["arm"] == BASELINE_ARM:
        continue
    for name, value in record["gates"].items():
        gate_rows.append({"group": record["arm"], "gate": name, "passed": bool(value)})
    for name, value in record["gold58_gates"].items():
        gate_rows.append({"group": record["arm"], "gate": name, "passed": bool(value)})
pd.DataFrame(gate_rows).to_csv(WORK / "e039_gates.csv", index=False)

print(json.dumps({key: value for key, value in summary.items()
                  if key not in {"arms", "identity"}}, indent=2, sort_keys=True))
for record in arm_records:
    bootstrap = record["bootstrap"]
    incumbent = record["bootstrap_vs_incumbent"]
    marks = ("strict+relaxed" if record["passed_strict"]
             else "relaxed" if record["passed_relaxed"]
             else "" if record["promotable"] else "not promotable")
    print(f"{record['arm']:20s} eval {record['eval_macro_delta']:+.6f} "
          f"win {bootstrap['win_rate']:.3f} "
          f"ci95 [{bootstrap['ci95'][0]:+.6f}, {bootstrap['ci95'][1]:+.6f}] "
          f"vs inc {record['eval_delta_vs_incumbent']:+.6f} "
          f"{'     ' if incumbent is None else format(incumbent['win_rate'], '.3f')} "
          f"gold {record['gold58_macro_delta']:+.6f} {marks}")
for name, values in reproduction_records.items():
    print(f"repro {name:20s} E038 {values['e038_delta']:+.6f} "
          f"E039 {values['e039_delta']:+.6f} gap {values['gap']:+.6f} "
          f"{'ok' if values['reproduced'] else 'FAILED'}")
print(f"repro {'C0 baseline macro':20s} E038 {E038_B0_EVAL_MACRO:.6f} "
      f"E039 {eval_macro[BASELINE_ARM]:.6f} gap {e038_baseline_gap:+.6f} "
      f"{'ok' if e038_baseline_reproduced else 'FAILED'}")
print("E039 DECISION strict ", decision, "best arm", best_arm)
print("E039 DECISION relaxed", decision_relaxed, "best arm", best_arm_relaxed,
      "verdicts agree", verdicts_agree)
print("E039 submission candidate", submission_candidate, submission_checkpoints)

del models
shutil.rmtree(CACHE, ignore_errors=True)
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("E039 COMPLETE submission=False checkpoint_exported=False trained=False")
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
