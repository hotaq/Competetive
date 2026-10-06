#!/usr/bin/env python3
"""Build E004 from the pinned E003 notebook.

E004 mounts E003's Kaggle output and reuses its frozen embeddings.  It changes only
supervision weights and retrains the cheap ridge/attention heads; no DICOM is decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "a65ab4b0c1668b2e1fa12c4fa7a2183c0d735319a8a3fddbacabb8b17a673592"


def source(cell: dict[str, Any]) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def set_source(cell: dict[str, Any], value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def code_cell(nb: dict[str, Any], marker: str) -> dict[str, Any]:
    hits = [
        c for c in nb["cells"]
        if c.get("cell_type") == "code" and marker in source(c)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one code cell containing {marker!r}, got {len(hits)}")
    return hits[0]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement, got {count}")
    return text.replace(old, new)


def patch_banner(nb: dict[str, Any]) -> None:
    cell = nb["cells"][0]
    if cell.get("cell_type") != "markdown" or "# E003" not in source(cell):
        raise RuntimeError("pinned E003 provenance banner not found")
    set_source(
        cell,
        "# E004 — cached supervision ablation\n\n"
        "**Purpose:** determine whether E003 failed because of its supervision weights "
        "rather than its frozen DINOv2 features. The 609 MB cache is mounted directly "
        "from E003's Kaggle output; this notebook performs no DICOM pass and makes no "
        "leaderboard submission.\n\n"
        f"Built from E003 source SHA-256 `{SOURCE_SHA256}`.\n",
    )


def patch_config(nb: dict[str, Any]) -> None:
    cell = code_cell(nb, "LABEL_DATASET_SOURCES =")
    text = source(cell)
    text = replace_once(
        text,
        "SAVE_TRAIN_CACHE = True            # retain the ~610 MB embedding cache, never DICOMs",
        "SAVE_TRAIN_CACHE = False           # E004 mounts E003 output; do not duplicate it",
        "cache output",
    )
    text = replace_once(
        text,
        "LOAD_TRAIN_CACHE_FROM_DATASET = False",
        "LOAD_TRAIN_CACHE_FROM_DATASET = True",
        "cache input",
    )
    text = replace_once(
        text,
        "GATE_MAX_MEDIAN_CORR = 0.97        # require prediction diversity",
        "GATE_MAX_MEDIAN_CORR = 0.97        # E003 historical gate\n"
        "E004_ARM_WEIGHT = 0.10             # predeclared cached-arm share\n"
        "E004_MIN_MACRO_GAIN = 0.001        # must improve broad OOF\n"
        "E004_GOLD_TOL = 0.005              # gold audit tolerance\n"
        "E004_MIN_NONWORSE = 8              # target breadth\n"
        "E004_MAX_TARGET_DROP = 0.015       # single-target veto\n"
        "E004_MAX_MEDIAN_CORR = 0.98        # require diversity",
        "E004 gate constants",
    )
    set_source(cell, text)


def patch_cache_lookup(nb: dict[str, Any]) -> None:
    cell = code_cell(nb, "def load_cached(split, ids):")
    text = source(cell)
    text = replace_once(
        text,
        '    if split == "train" and LOAD_TRAIN_CACHE_FROM_DATASET:',
        "    # E004 mounts both train and placeholder-test cache files from E003.\n"
        "    if LOAD_TRAIN_CACHE_FROM_DATASET:",
        "all-split mounted cache lookup",
    )
    set_source(cell, text)


def remove_checkpoint_writes(nb: dict[str, Any]) -> None:
    cell = code_cell(nb, "class SliceAttentionHead")
    text = source(cell)
    block = (
        "            # The expensive artifact is the DINO embedding cache.  Persist the\n"
        "            # small deterministic head too, so a later hidden-test notebook can\n"
        "            # infer without retraining or silently changing library behaviour.\n"
        "            torch.save({\n"
        "                'format': 1,\n"
        "                'fold': k,\n"
        "                'seed': seed,\n"
        "                'targets': LABELS,\n"
        "                'slot_names': SLOT_NAMES,\n"
        "                'slice_dim': SLICE_DIM,\n"
        "                'meta_names': META_NAMES,\n"
        "                'state_dict': {name: value.detach().cpu()\n"
        "                               for name, value in model.state_dict().items()},\n"
        "            }, WORK / 'e003_attention_fold{}_seed{}.pt'.format(k, seed))\n"
    )
    text = replace_once(text, block, "", "remove E003 checkpoint writes")
    set_source(cell, text)


E004_CELL = r'''# E004: supervision weights only; pixels, embeddings, folds and heads stay fixed.


def e004_ridge(weight_matrix):
    out = np.zeros((len(TRAIN_ENCODE_IDS), len(LABELS)), np.float32)
    test_out = np.zeros((len(test_ids), len(LABELS)), np.float32)
    for tr_i, va_i in FOLDS:
        p_va, p_te = fit_ridge_fold(
            X_TR[tr_i], Y[tr_i], weight_matrix[tr_i], X_TR[va_i], X_TE
        )
        out[va_i] = p_va
        test_out += p_te / len(FOLDS)
    return out, test_out


def e004_attention(weight_matrix):
    global W
    previous = W
    W = np.asarray(weight_matrix, np.float32)
    try:
        return run_attention_head()
    finally:
        W = previous


gold_rows_all = np.asarray([s in set(gold_ids) for s in TRAIN_ENCODE_IDS], bool)
raw_weight = np.asarray(W, np.float32).copy()

# Anchor-style confidence never lets an unaddressed report cell vanish entirely and
# gives official labels a moderate rather than dominant weight.
anchor_style_weight = 0.25 + 0.75 * np.clip(raw_weight, 0.0, 1.0)
anchor_style_weight[gold_rows_all] = 3.0

# Equal weak weights test whether confidence itself is the harmful intervention.
equal_weight = np.ones_like(raw_weight, np.float32)
equal_weight[gold_rows_all] = 8.0

E004_RESULTS = {
    "raw_conf_gold8": {
        "ridge": np.asarray(oof_ridge, np.float32),
        "attention": np.asarray(oof_attn, np.float32),
        "model": np.asarray(blend_oof, np.float32),
        "test": np.asarray(blend_test, np.float32),
    }
}

for scheme, weights in (
    ("anchor_conf_gold3", anchor_style_weight),
    ("equal_gold8", equal_weight),
):
    t_scheme = time.time()
    print("E004 training", scheme, "| mean target weight", round(float(weights.mean()), 4))
    scheme_ridge, scheme_te_ridge = e004_ridge(weights)
    scheme_attn, scheme_te_attn = e004_attention(weights)
    scheme_model = 0.5 * rank01(scheme_ridge) + 0.5 * rank01(scheme_attn)
    scheme_test = 0.5 * rank01(scheme_te_ridge) + 0.5 * rank01(scheme_te_attn)
    E004_RESULTS[scheme] = {
        "ridge": scheme_ridge,
        "attention": scheme_attn,
        "model": scheme_model,
        "test": scheme_test,
    }
    print("  completed in {:.1f}s".format(time.time() - t_scheme))


def e004_metrics(name, prediction):
    arm_rank = rank01(np.asarray(prediction)[oi])
    candidate = ((1.0 - E004_ARM_WEIGHT) * anchor_rank
                 + E004_ARM_WEIGHT * arm_rank)
    anchor_per, anchor_m = finite_macro(y_ref, anchor_rank)
    arm_per, arm_m = finite_macro(y_ref, arm_rank)
    candidate_per, candidate_m = finite_macro(y_ref, candidate)
    _, anchor_g = finite_macro(model_y, anchor_rank, gold_ref)
    _, arm_g = finite_macro(model_y, arm_rank, gold_ref)
    _, candidate_g = finite_macro(model_y, candidate, gold_ref)
    delta = candidate_per - anchor_per
    corr = np.asarray([
        np.corrcoef(anchor_rank[:, j], arm_rank[:, j])[0, 1]
        for j in range(len(LABELS))
    ])
    checks = {
        "macro_gain": float(candidate_m - anchor_m) >= E004_MIN_MACRO_GAIN,
        "gold_direction": float(candidate_g - anchor_g) >= -E004_GOLD_TOL,
        "target_breadth": int((delta >= -0.002).sum()) >= E004_MIN_NONWORSE,
        "target_veto": float(np.nanmin(delta)) >= -E004_MAX_TARGET_DROP,
        "diversity": float(np.nanmedian(corr)) <= E004_MAX_MEDIAN_CORR,
    }
    return {
        "scheme": name,
        "pass": bool(all(checks.values())),
        "checks": checks,
        "arm_weight": E004_ARM_WEIGHT,
        "anchor_macro": float(anchor_m),
        "arm_macro": float(arm_m),
        "candidate_macro": float(candidate_m),
        "macro_gain": float(candidate_m - anchor_m),
        "anchor_gold_macro": float(anchor_g),
        "arm_gold_macro": float(arm_g),
        "candidate_gold_macro": float(candidate_g),
        "gold_gain": float(candidate_g - anchor_g),
        "nonworse_targets": int((delta >= -0.002).sum()),
        "worst_target_delta": float(np.nanmin(delta)),
        "median_spearman": float(np.nanmedian(corr)),
        "per_target": [
            {
                "target": LABELS[j],
                "anchor_auc": float(anchor_per[j]),
                "arm_auc": float(arm_per[j]),
                "candidate_auc": float(candidate_per[j]),
                "candidate_delta": float(delta[j]),
                "spearman": float(corr[j]),
            }
            for j in range(len(LABELS))
        ],
    }


E004_SCHEMES = [
    e004_metrics(name, values["model"])
    for name, values in E004_RESULTS.items()
]
E004_GATE = {
    "pass": bool(any(x["pass"] for x in E004_SCHEMES if x["scheme"] != "raw_conf_gold8")),
    "decision": "retain_head_family" if any(
        x["pass"] for x in E004_SCHEMES if x["scheme"] != "raw_conf_gold8"
    ) else "discard_head_family",
    "cache_source": "chinnaphatkhun/rsna-knee-e003-dinov2-diversity-lab",
    "n_studies": len(common),
    "n_gold": int(gold_ref.sum()),
    "schemes": E004_SCHEMES,
}

summary = pd.DataFrame([
    {
        "scheme": x["scheme"],
        "arm_macro": x["arm_macro"],
        "candidate_macro": x["candidate_macro"],
        "macro_gain": x["macro_gain"],
        "arm_gold": x["arm_gold_macro"],
        "candidate_gold": x["candidate_gold_macro"],
        "gold_gain": x["gold_gain"],
        "nonworse": x["nonworse_targets"],
        "worst_delta": x["worst_target_delta"],
        "median_corr": x["median_spearman"],
        "pass": x["pass"],
    }
    for x in E004_SCHEMES
])
print()
print("E004 cached supervision ablation")
print(summary.round(5).to_string(index=False))
print("E004 GATE:", "GO" if E004_GATE["pass"] else "NO-GO",
      E004_GATE["decision"])

(WORK / "e004_gate.json").write_text(json.dumps(E004_GATE, indent=2, sort_keys=True))
np.savez_compressed(
    WORK / "e004_oof_predictions.npz",
    ids=np.asarray(common, str),
    targets=np.asarray(LABELS, str),
    y=y_ref,
    y_with_gold=model_y,
    gold_mask=gold_ref,
    anchor=np.asarray(anchor_pred, np.float32),
    raw_conf_gold8=np.asarray(E004_RESULTS["raw_conf_gold8"]["model"][oi], np.float32),
    anchor_conf_gold3=np.asarray(E004_RESULTS["anchor_conf_gold3"]["model"][oi], np.float32),
    equal_gold8=np.asarray(E004_RESULTS["equal_gold8"]["model"][oi], np.float32),
)
np.savez_compressed(
    WORK / "e004_test_predictions.npz",
    ids=np.asarray(test_ids, str),
    targets=np.asarray(LABELS, str),
    raw_conf_gold8=np.asarray(E004_RESULTS["raw_conf_gold8"]["test"], np.float32),
    anchor_conf_gold3=np.asarray(E004_RESULTS["anchor_conf_gold3"]["test"], np.float32),
    equal_gold8=np.asarray(E004_RESULTS["equal_gold8"]["test"], np.float32),
)
print("wrote e004_gate.json and E004 prediction artifacts")
'''


def insert_e004(nb: dict[str, Any]) -> None:
    gate = code_cell(nb, "# E003 offline complementarity gate")
    index = nb["cells"].index(gate) + 1
    nb["cells"].insert(
        index,
        {
            "cell_type": "markdown",
            "id": "e004-title",
            "metadata": {},
            "source": [
                "## E004 cache-only supervision ablation\n",
                "\n",
                "The frozen embeddings, folds, and head definitions remain fixed. Only target weights change.\n",
            ],
        },
    )
    nb["cells"].insert(
        index + 1,
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "e004-ablation",
            "metadata": {},
            "outputs": [],
            "source": E004_CELL.splitlines(keepends=True),
        },
    )


def validate(nb: dict[str, Any]) -> None:
    joined = "\n".join(source(c) for c in nb["cells"] if c.get("cell_type") == "code")
    for marker in (
        "LOAD_TRAIN_CACHE_FROM_DATASET = True",
        "if LOAD_TRAIN_CACHE_FROM_DATASET:",
        "E004 cached supervision ablation",
        "discard_head_family",
        "e004_gate.json",
    ):
        if marker not in joined:
            raise RuntimeError(f"missing E004 marker {marker!r}")
    if "e003_attention_fold{}_seed{}.pt" in joined:
        raise RuntimeError("E004 must not duplicate E003 head checkpoints")
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            compile(source(cell), f"cell-{i}", "exec")


def build(source_path: Path, output_path: Path) -> None:
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"E003 source SHA-256 {digest} != pinned {SOURCE_SHA256}")
    nb = json.loads(raw)
    patch_banner(nb)
    patch_config(nb)
    patch_cache_lookup(nb)
    remove_checkpoint_writes(nb)
    insert_e004(nb)
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    validate(nb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
    print("built", args.output)


if __name__ == "__main__":
    main()
