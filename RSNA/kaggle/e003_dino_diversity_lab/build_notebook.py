#!/usr/bin/env python3
"""Build the E003 Kaggle notebook from the audited public DINOv2 notebook.

The source notebook is intentionally supplied as an argument rather than copied into
the repository.  This keeps the experiment small while the pinned SHA-256 and exact
replacement checks make source drift fail loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


UPSTREAM_SHA256 = "387c8bedb0c87d8ca40f3d7b2bbe57ff1492681c37124f4aa05a5a8016f4c56b"
UPSTREAM_REF = "wguesdon/rsna-knee-dinov2-at-meniscus-resolution"


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def set_cell_source(cell: dict[str, Any], source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def find_code_cell(nb: dict[str, Any], marker: str) -> dict[str, Any]:
    hits = [
        cell
        for cell in nb["cells"]
        if cell.get("cell_type") == "code" and marker in cell_source(cell)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one code cell containing {marker!r}, found {len(hits)}")
    return hits[0]


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact replacement, found {count}")
    return source.replace(old, new)


def patch_config(nb: dict[str, Any]) -> None:
    cell = find_code_cell(nb, "LABEL_DATASET_SOURCES =")
    source = cell_source(cell)
    replacements = [
        ("import gc\nimport json", "import gc\nimport hashlib\nimport json", "hashlib import"),
        ("SEED = 20260805", "SEED = 20260809", "experiment seed"),
        (
            "BLEND_W = 0.5                      # fixed, not tuned; the curve is printed anyway",
            "BLEND_W = 0.5                      # fixed model-only ridge/attention blend\n"
            "ANCHOR_BLEND_W = 0.75              # predeclared anchor share for the OOF gate\n"
            "GATE_MIN_MACRO_GAIN = 0.002        # candidate must improve broad OOF\n"
            "GATE_GOLD_TOL = 0.005              # tiny gold audit may fall by at most this\n"
            "GATE_MIN_NONWORSE = 8              # targets no worse than anchor by 0.002\n"
            "GATE_MAX_TARGET_DROP = 0.02        # veto a large single-target regression\n"
            "GATE_MAX_MEDIAN_CORR = 0.97        # require prediction diversity\n"
            "RULE_WEIGHT = 0.35                 # fallback rule-label confidence\n"
            "EXPECTED_LABEL_SHA256 = '6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920'\n"
            "EXPECTED_ANCHOR_OOF_SHA256 = '62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c'",
            "OOF gate constants",
        ),
        (
            "FINETUNE = True",
            "FINETUNE = False              # E003 isolates frozen embeddings + heads",
            "disable fine-tune",
        ),
        (
            'LABEL_SOURCE = "auto"              # "auto" | "llm" | "rule_file" | "rule_inline"',
            'LABEL_SOURCE = "llm"               # fail closed unless the declared table mounts',
            "label source",
        ),
        (
            'LLM_LABEL_PATTERNS = ["labels_llm_*.parquet", "labels_llm_*.csv"]',
            'LLM_LABEL_PATTERNS = ["report_labels_v2.csv"]',
            "label filename",
        ),
        (
            "LABEL_DATASET_SOURCES = ['wguesdon/rsna-knee-llm-report-labels-opus']",
            "LABEL_DATASET_SOURCES = ['pilkwang/rsna-knee-llm-labels']",
            "declared label dataset",
        ),
        (
            "SAVE_TRAIN_CACHE = False           # True to keep the train embeddings in the output",
            "SAVE_TRAIN_CACHE = True            # retain the ~610 MB embedding cache, never DICOMs",
            "cache retention",
        ),
    ]
    for old, new, label in replacements:
        source = replace_once(source, old, new, label)
    set_cell_source(cell, source)


def patch_labels(nb: dict[str, Any]) -> None:
    cell = find_code_cell(nb, "def _label_search_dirs():")
    source = cell_source(cell)
    source = replace_once(
        source,
        "Y_soft = rule_df.reindex(train_ids).astype(float)\n"
        "Y_soft = Y_soft.fillna(0.0)\n"
        'LABEL_PROVENANCE = "rule"\n'
        "n_llm = 0",
        "Y_soft = rule_df.reindex(train_ids).astype(float)\n"
        "Y_soft = Y_soft.fillna(0.0)\n"
        "# Confidence is target-specific.  Unknown report cells in Pilkwang's table\n"
        "# carry confidence 0.05 and must not count like explicit positive/negative text.\n"
        "Y_CONF = pd.DataFrame(RULE_WEIGHT, index=train_ids, columns=LABELS, dtype=float)\n"
        'LABEL_PROVENANCE = "rule"\n'
        "LLM_LABEL_PATH = None\n"
        "n_llm = 0",
        "initialize confidence table",
    )
    source = replace_once(
        source,
        "                n_llm = len(common)\n"
        '                LABEL_PROVENANCE = "llm+rule" if n_llm < len(train_ids) else "llm"\n'
        "                print(\"LLM labels from file  :\", p.name, \"|\", n_llm,\n"
        "                      \"of\", len(train_ids), \"studies overwritten\")",
        "                n_llm = len(common)\n"
        "                LLM_LABEL_PATH = p\n"
        "                label_sha = hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "                if label_sha != EXPECTED_LABEL_SHA256:\n"
        "                    raise RuntimeError('label table SHA-256 {} != pinned {}'.format(\n"
        "                        label_sha, EXPECTED_LABEL_SHA256))\n"
        "                raw_conf = (pd.read_parquet(p) if p.suffix == '.parquet'\n"
        "                            else pd.read_csv(p))\n"
        "                raw_conf['StudyInstanceUID'] = raw_conf['StudyInstanceUID'].astype(str)\n"
        "                raw_conf = raw_conf.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')\n"
        "                for target in LABELS:\n"
        "                    col = target + '__conf'\n"
        "                    if col in raw_conf.columns:\n"
        "                        Y_CONF.loc[common, target] = (\n"
        "                            pd.to_numeric(raw_conf.loc[common, col], errors='coerce')\n"
        "                            .fillna(RULE_WEIGHT).clip(0.01, 1.0).values\n"
        "                        )\n"
        '                LABEL_PROVENANCE = "llm+rule" if n_llm < len(train_ids) else "llm"\n'
        "                print(\"LLM labels from file  :\", p.name, \"|\", n_llm,\n"
        "                      \"of\", len(train_ids), \"studies overwritten\")",
        "load confidence columns",
    )
    source = replace_once(
        source,
        "    Y_soft.loc[gold_ids, LABELS] = gold_vals[LABELS].astype(float).values\n\n"
        "IS_GOLD = pd.Series([s in set(gold_ids) for s in train_ids], index=train_ids)\n"
        "SAMPLE_WEIGHT = np.where(IS_GOLD.values, GOLD_WEIGHT, 1.0).astype(np.float32)",
        "    Y_soft.loc[gold_ids, LABELS] = gold_vals[LABELS].astype(float).values\n"
        "    Y_CONF.loc[gold_ids, LABELS] = GOLD_WEIGHT\n\n"
        "IS_GOLD = pd.Series([s in set(gold_ids) for s in train_ids], index=train_ids)\n"
        "SAMPLE_WEIGHT = Y_CONF.mean(axis=1).astype(np.float32).values",
        "gold confidence override",
    )
    source = replace_once(
        source,
        "print(prev.to_string())",
        "print(prev.to_string())\n"
        "print()\n"
        "print('target confidence : mean {:.3f} | p10 {:.3f} | p90 {:.3f}'.format(\n"
        "    float(Y_CONF.values.mean()), float(np.quantile(Y_CONF.values, 0.10)),\n"
        "    float(np.quantile(Y_CONF.values, 0.90))))",
        "confidence diagnostics",
    )
    set_cell_source(cell, source)


def patch_feature_weights(nb: dict[str, Any]) -> None:
    cell = find_code_cell(nb, "def pooled_features(emb, mask):")
    source = cell_source(cell)
    source = replace_once(
        source,
        "W = np.asarray([GOLD_WEIGHT if s in set(gold_ids) else 1.0\n"
        "                for s in TRAIN_ENCODE_IDS], np.float32)",
        "W = Y_CONF.reindex(TRAIN_ENCODE_IDS)[LABELS].astype(np.float32).values\n"
        "assert W.shape == Y.shape and np.isfinite(W).all() and (W > 0).all(), \\\n"
        "    'target-confidence matrix is invalid'",
        "target weight matrix",
    )
    source = replace_once(
        source,
        'print("target matrix  :", Y.shape, "| positives at 0.5:", int(Y_HARD.sum()))',
        'print("target matrix  :", Y.shape, "| positives at 0.5:", int(Y_HARD.sum()))\n'
        'print("weight matrix  :", W.shape, "| mean", round(float(W.mean()), 4))',
        "weight diagnostics",
    )
    set_cell_source(cell, source)


def patch_ridge(nb: dict[str, Any]) -> None:
    cell = find_code_cell(nb, "def fit_ridge_fold(")
    source = cell_source(cell)
    source = replace_once(source, "w_tr: Per-row weights.", "w_tr: Per-target weights [n, 12].", "ridge docs")
    source = replace_once(
        source,
        "    model = RidgeCV(alphas=RIDGE_ALPHAS)\n"
        "    model.fit(a, y_tr, sample_weight=w_tr)\n"
        "    return model.predict(b), model.predict(c)",
        "    pred_va = np.zeros((len(b), len(LABELS)), np.float32)\n"
        "    pred_te = np.zeros((len(c), len(LABELS)), np.float32)\n"
        "    # Fit findings separately so an unaddressed ACL cell does not reduce the\n"
        "    # supervision weight of an explicitly described meniscal tear in the same report.\n"
        "    for j in range(len(LABELS)):\n"
        "        model = RidgeCV(alphas=RIDGE_ALPHAS)\n"
        "        model.fit(a, y_tr[:, j], sample_weight=w_tr[:, j])\n"
        "        pred_va[:, j] = model.predict(b)\n"
        "        pred_te[:, j] = model.predict(c)\n"
        "    return pred_va, pred_te",
        "target-wise ridge",
    )
    set_cell_source(cell, source)


def patch_attention_loss(nb: dict[str, Any]) -> None:
    cell = find_code_cell(nb, "class SliceAttentionHead")
    source = cell_source(cell)
    source = replace_once(
        source,
        "                    loss = torch.nn.functional.binary_cross_entropy_with_logits(\n"
        "                        logit, y_t[idx].to(dev),\n"
        "                        weight=w_t[idx].to(dev).unsqueeze(1))",
        "                    loss_el = torch.nn.functional.binary_cross_entropy_with_logits(\n"
        "                        logit, y_t[idx].to(dev), reduction='none')\n"
        "                    weight = w_t[idx].to(dev)\n"
        "                    loss = (loss_el * weight).sum() / weight.sum().clamp_min(1.0)",
        "confidence-aware attention loss",
    )
    source = replace_once(
        source,
        "                    opt.step()\n"
        "                    sched.step()\n"
        "            model.eval()",
        "                    opt.step()\n"
        "                    sched.step()\n"
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
        "            model.eval()",
        "persist attention heads",
    )
    set_cell_source(cell, source)


def patch_disabled_finetune_loss(nb: dict[str, Any]) -> None:
    """Keep the dormant fine-tune path valid if a later experiment enables it."""
    cell = find_code_cell(nb, "class SlotFineTuner")
    source = cell_source(cell)
    source = replace_once(
        source,
        "            loss = (loss_el.mean(dim=1) * w_t[idx]).sum() / w_t[idx].sum()",
        "            weight = w_t[idx]\n"
        "            loss = (loss_el * weight).sum() / weight.sum().clamp_min(1.0)",
        "confidence-aware dormant fine-tune loss",
    )
    set_cell_source(cell, source)


GATE_CELL = r'''# E003 offline complementarity gate against the scored DINOv2 family.
# This cell never submits anything.  It records whether the new prediction family is
# worth carrying into a later hidden-test blend.


def find_anchor_oof():
    required = {"ids", "pred", "y_derived", "gold_mask", "targets"}
    hits = []
    base = Path("/kaggle/input")
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in
                       ("competitions", "train_series", "test_series", "__pycache__")]
            if "oof.npz" not in files:
                continue
            path = Path(root) / "oof.npz"
            try:
                z = np.load(path, allow_pickle=False)
                if required.issubset(z.files):
                    hits.append(path)
            except Exception:
                continue
    if not hits:
        return None
    hits.sort(key=lambda p: ("rsna-knee-weights" not in str(p), len(str(p)), str(p)))
    return hits[0]


def finite_macro(y, p, mask=None):
    rows = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    per, macro = macro_auc(np.asarray(y)[rows], np.asarray(p)[rows])
    return np.asarray(per, float), float(macro)


anchor_path = find_anchor_oof()
gate_path = WORK / "e003_gate.json"
oof_path = WORK / "e003_oof_predictions.npz"
test_pred_path = WORK / "e003_test_predictions.npz"

np.savez_compressed(
    test_pred_path,
    ids=np.asarray(test_ids, str),
    targets=np.asarray(LABELS, str),
    ridge=np.asarray(te_ridge, np.float32),
    attention=np.asarray(te_attn, np.float32),
    model=np.asarray(blend_test, np.float32),
)

if anchor_path is None:
    E003_GATE = {
        "pass": False,
        "reason": "anchor oof.npz was not found in mounted Kaggle inputs",
    }
    np.savez_compressed(
        oof_path,
        ids=np.asarray(TRAIN_ENCODE_IDS, str),
        targets=np.asarray(LABELS, str),
        y=np.asarray(Y_HARD, np.int8),
        gold_mask=np.asarray([s in set(gold_ids) for s in TRAIN_ENCODE_IDS], bool),
        ridge=np.asarray(oof_ridge, np.float32),
        attention=np.asarray(oof_attn, np.float32),
        model=np.asarray(blend_oof, np.float32),
    )
    print("E003 OFFLINE GATE: NO-GO -", E003_GATE["reason"])
else:
    anchor_npz = np.load(anchor_path, allow_pickle=False)
    anchor_oof_sha = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    if anchor_oof_sha != EXPECTED_ANCHOR_OOF_SHA256:
        raise RuntimeError("anchor OOF SHA-256 {} != pinned {}".format(
            anchor_oof_sha, EXPECTED_ANCHOR_OOF_SHA256))
    anchor_targets = [str(x) for x in anchor_npz["targets"]]
    if anchor_targets != LABELS:
        raise RuntimeError("anchor target order differs from E003")

    ours_pos = {str(s): i for i, s in enumerate(TRAIN_ENCODE_IDS)}
    anchor_pos = {str(s): i for i, s in enumerate(anchor_npz["ids"])}
    common = [str(s) for s in anchor_npz["ids"] if str(s) in ours_pos]
    if len(common) != len(TRAIN_ENCODE_IDS):
        raise RuntimeError("anchor/E003 OOF rows do not cover the same studies")
    oi = np.asarray([ours_pos[s] for s in common], int)
    ai = np.asarray([anchor_pos[s] for s in common], int)

    y_ref = np.asarray(anchor_npz["y_derived"][ai], np.int8)
    gold_ref = np.asarray(anchor_npz["gold_mask"][ai], bool)
    model_y = np.asarray(Y_HARD[oi], np.int8)
    anchor_pred = np.asarray(anchor_npz["pred"][ai], np.float64)
    model_pred = np.asarray(blend_oof[oi], np.float64)
    # The anchor OOF stores report-derived evaluation labels even on the 58 gold rows.
    # E003 correctly overrides those rows for training.  Identity therefore applies to
    # the non-gold weak targets; the gold audit below uses official train.csv values.
    weak_label_mismatches = int(((y_ref != model_y) & ~gold_ref[:, None]).sum())
    gold_override_differences = int(((y_ref != model_y) & gold_ref[:, None]).sum())

    anchor_rank = rank01(anchor_pred)
    model_rank = rank01(model_pred)
    candidate = ANCHOR_BLEND_W * anchor_rank + (1.0 - ANCHOR_BLEND_W) * model_rank

    anchor_per, anchor_macro = finite_macro(y_ref, anchor_rank)
    model_per, model_macro = finite_macro(y_ref, model_rank)
    candidate_per, candidate_macro = finite_macro(y_ref, candidate)
    _, anchor_gold = finite_macro(model_y, anchor_rank, gold_ref)
    _, model_gold = finite_macro(model_y, model_rank, gold_ref)
    _, candidate_gold = finite_macro(model_y, candidate, gold_ref)

    corr = np.asarray([
        np.corrcoef(anchor_rank[:, j], model_rank[:, j])[0, 1]
        for j in range(len(LABELS))
    ])
    delta = candidate_per - anchor_per
    macro_gain = float(candidate_macro - anchor_macro)
    gold_gain = float(candidate_gold - anchor_gold)
    nonworse = int((delta >= -0.002).sum())
    worst_drop = float(np.nanmin(delta))
    median_corr = float(np.nanmedian(corr))

    checks = {
        "macro_gain": macro_gain >= GATE_MIN_MACRO_GAIN,
        "gold_direction": gold_gain >= -GATE_GOLD_TOL,
        "target_breadth": nonworse >= GATE_MIN_NONWORSE,
        "target_veto": worst_drop >= -GATE_MAX_TARGET_DROP,
        "diversity": median_corr <= GATE_MAX_MEDIAN_CORR,
        "label_identity": weak_label_mismatches == 0,
    }
    E003_GATE = {
        "pass": bool(all(checks.values())),
        "anchor_path": str(anchor_path),
        "anchor_oof_sha256": anchor_oof_sha,
        "n_studies": len(common),
        "n_gold": int(gold_ref.sum()),
        "anchor_weight": ANCHOR_BLEND_W,
        "anchor_macro": anchor_macro,
        "model_macro": model_macro,
        "candidate_macro": candidate_macro,
        "macro_gain": macro_gain,
        "anchor_gold_macro": anchor_gold,
        "model_gold_macro": model_gold,
        "candidate_gold_macro": candidate_gold,
        "gold_gain": gold_gain,
        "nonworse_targets": nonworse,
        "worst_target_delta": worst_drop,
        "median_spearman": median_corr,
        "weak_label_mismatches": weak_label_mismatches,
        "gold_override_differences": gold_override_differences,
        "checks": checks,
        "per_target": [
            {
                "target": LABELS[j],
                "anchor_auc": float(anchor_per[j]),
                "model_auc": float(model_per[j]),
                "candidate_auc": float(candidate_per[j]),
                "candidate_delta": float(delta[j]),
                "spearman": float(corr[j]),
            }
            for j in range(len(LABELS))
        ],
    }
    np.savez_compressed(
        oof_path,
        ids=np.asarray(common, str),
        targets=np.asarray(LABELS, str),
        y=y_ref,
        y_with_gold=model_y,
        gold_mask=gold_ref,
        anchor=np.asarray(anchor_pred, np.float32),
        ridge=np.asarray(oof_ridge[oi], np.float32),
        attention=np.asarray(oof_attn[oi], np.float32),
        model=np.asarray(model_pred, np.float32),
        candidate=np.asarray(candidate, np.float32),
    )

    gate_table = pd.DataFrame(E003_GATE["per_target"])
    print()
    print("E003 complementarity against the frozen 0.899 model family")
    print(gate_table.round(4).to_string(index=False))
    print()
    print("anchor {:.4f} | E003 {:.4f} | 75/25 candidate {:.4f} | gain {:+.4f}".format(
        anchor_macro, model_macro, candidate_macro, macro_gain))
    print("gold   {:.4f} | E003 {:.4f} | 75/25 candidate {:.4f} | gain {:+.4f}".format(
        anchor_gold, model_gold, candidate_gold, gold_gain))
    print("non-worse targets {} / 12 | worst delta {:+.4f} | median correlation {:.4f}".format(
        nonworse, worst_drop, median_corr))
    print("E003 OFFLINE GATE:", "GO" if E003_GATE["pass"] else "NO-GO", checks)

gate_path.write_text(json.dumps(E003_GATE, indent=2, sort_keys=True))
print("wrote", gate_path, oof_path, test_pred_path)
'''


def insert_gate_cell(nb: dict[str, Any]) -> None:
    scoring_cell = find_code_cell(nb, "from scipy.stats import rankdata")
    index = nb["cells"].index(scoring_cell) + 1
    nb["cells"].insert(
        index,
        {
            "cell_type": "markdown",
            "id": "e003-gate-title",
            "metadata": {},
            "source": [
                "## E003 offline gate against the scored anchor\n",
                "\n",
                "This is a predeclared 75% anchor / 25% E003 rank blend. The notebook records the result; it does not submit it.\n",
            ],
        },
    )
    nb["cells"].insert(
        index + 1,
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "e003-anchor-gate",
            "metadata": {},
            "outputs": [],
            "source": GATE_CELL.splitlines(keepends=True),
        },
    )


def add_provenance_banner(nb: dict[str, Any]) -> None:
    banner = {
        "cell_type": "markdown",
        "id": "e003-provenance",
        "metadata": {},
        "source": [
            "# E003 — DINOv2 diversity lab\n",
            "\n",
            "**Purpose:** test a frozen high-resolution slice/slot model for OOF complementarity to the verified 0.899 DINOv2 ensemble. This notebook does not auto-submit.\n",
            "\n",
            f"Built from `{UPSTREAM_REF}` at source SHA-256 `{UPSTREAM_SHA256}`. E003 changes label provenance, confidence-aware losses, cache retention, and the anchor OOF gate.\n",
            "\n",
            "The competition DICOM archive remains mounted on Kaggle and is never downloaded locally.\n",
        ],
    }
    nb["cells"].insert(0, banner)


def strip_stale_execution(nb: dict[str, Any]) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def validate_built_notebook(nb: dict[str, Any]) -> None:
    code = "\n".join(cell_source(c) for c in nb["cells"] if c.get("cell_type") == "code")
    required = [
        "E003 OFFLINE GATE",
        "report_labels_v2.csv",
        "Y_CONF",
        "e003_oof_predictions.npz",
        "SAVE_TRAIN_CACHE = True",
        "FINETUNE = False",
    ]
    missing = [marker for marker in required if marker not in code]
    if missing:
        raise RuntimeError(f"built notebook is missing markers: {missing}")
    forbidden = ["kaggle competitions download", "wget ", "curl ", "enable_internet = True"]
    hits = [marker for marker in forbidden if marker in code]
    if hits:
        raise RuntimeError(f"built notebook contains forbidden download/internet markers: {hits}")
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        try:
            compile(cell_source(cell), f"cell-{i}", "exec")
        except SyntaxError as exc:
            raise RuntimeError(f"code cell {i} does not compile: {exc}") from exc


def build(source_path: Path, output_path: Path, allow_source_drift: bool) -> None:
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != UPSTREAM_SHA256 and not allow_source_drift:
        raise RuntimeError(
            f"upstream SHA-256 is {digest}, expected {UPSTREAM_SHA256}; "
            "audit the changed source before using --allow-source-drift"
        )
    nb = json.loads(raw)
    add_provenance_banner(nb)
    patch_config(nb)
    patch_labels(nb)
    patch_feature_weights(nb)
    patch_ridge(nb)
    patch_attention_loss(nb)
    patch_disabled_finetune_loss(nb)
    insert_gate_cell(nb)
    strip_stale_execution(nb)
    validate_built_notebook(nb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-source-drift", action="store_true")
    args = parser.parse_args()
    build(args.source, args.output, args.allow_source_drift)
    print(f"built {args.output}")


if __name__ == "__main__":
    main()
