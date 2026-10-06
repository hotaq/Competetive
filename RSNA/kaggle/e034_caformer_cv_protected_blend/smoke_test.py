#!/usr/bin/env python
"""Structural and fail-closed smoke tests for E034."""

from __future__ import annotations

import ast
import base64
import csv
import gc
import hashlib
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e034-caformer-cv-protected-blend.ipynb"
METADATA = HERE / "kernel-metadata.json"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "e034_validator", HERE / "validate_output.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def extract_definitions(source: str, names) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = min(
                [node.lineno]
                + [decorator.lineno for decorator in node.decorator_list]
            )
            selected.append("\n".join(lines[start - 1 : node.end_lineno]))
            found.add(node.name)
    if found != set(names):
        raise AssertionError(f"missing definitions: {sorted(set(names) - found)}")
    return "\n\n".join(selected)


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_predictions(rng, labels, noise):
    logits = (labels * 2.0 - 1.0) + rng.normal(0.0, noise, labels.shape)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def write_fixture(directory: Path, validator, pass_signal: bool) -> None:
    rng = np.random.default_rng(340)
    folds = np.repeat(np.arange(5, dtype=np.int16), 128)
    labels = np.zeros((640, 12), np.float32)
    for target in range(12):
        for fold in range(5):
            indexes = np.flatnonzero(folds == fold)
            positives = 12 + target
            selected = rng.choice(indexes, size=positives, replace=False)
            labels[selected, target] = 1.0

    coatnet = make_predictions(rng, labels, 1.40)
    caformer = make_predictions(rng, labels, 1.10)
    protected = (0.25 * coatnet + 0.75 * caformer).astype(np.float32)
    ids = [f"study-{index:04d}" for index in range(640)]
    np.savez_compressed(
        directory / "e034_oof_predictions.npz",
        ids=np.asarray(ids),
        folds=folds,
        y_binary=labels,
        coatnet=coatnet,
        caformer=caformer,
        protected_blend=protected,
    )

    predictions = {
        "coatnet": coatnet,
        "caformer": caformer,
        "protected_blend": protected,
    }
    target_auc = {}
    macro_auc = {}
    for key, values in predictions.items():
        target_auc[key], macro_auc[key] = validator.target_aucs(labels, values)

    target_rows = []
    for target_index, target in enumerate(validator.TARGETS):
        target_rows.append(
            {
                "target": target,
                "coatnet_auc": target_auc["coatnet"][target_index],
                "caformer_auc": target_auc["caformer"][target_index],
                "protected_blend_auc": target_auc["protected_blend"][target_index],
                "delta_caformer_minus_coatnet": (
                    target_auc["caformer"][target_index]
                    - target_auc["coatnet"][target_index]
                ),
                "delta_blend_minus_coatnet": (
                    target_auc["protected_blend"][target_index]
                    - target_auc["coatnet"][target_index]
                ),
                "caformer_fold_wins": 5,
                "blend_fold_wins": 5,
                "selected_alpha_mean": 0.75,
                "selected_alpha_min": 0.75,
                "selected_alpha_max": 0.75,
            }
        )
    write_csv(
        directory / "e034_oof_target_metrics.csv",
        list(target_rows[0]),
        target_rows,
    )

    fold_rows = []
    for fold in range(5):
        indexes = folds == fold
        for key, values in predictions.items():
            _, macro = validator.target_aucs(labels[indexes], values[indexes])
            fold_rows.append(
                {"outer_fold": fold, "model_key": key, "macro_auc": macro}
            )
    write_csv(
        directory / "e034_fold_metrics.csv",
        ["outer_fold", "model_key", "macro_auc"],
        fold_rows,
    )

    alpha_rows = []
    for fold in range(5):
        for target in validator.TARGETS:
            alpha_rows.append(
                {
                    "target": target,
                    "baseline_auc": 0.70,
                    "best_auc": 0.72,
                    "best_alpha": 0.75,
                    "inner_gain": 0.02,
                    "selected_alpha": 0.75,
                    "outer_fold": fold,
                }
            )
    write_csv(
        directory / "e034_alpha_metrics.csv",
        list(alpha_rows[0]),
        alpha_rows,
    )

    attention_rows = []
    for model_key in ("coatnet", "caformer"):
        for target in validator.TARGETS:
            attention_rows.append(
                {
                    "model_key": model_key,
                    "target": target,
                    "mean_entropy": 0.8,
                    "attention_std": 0.02,
                    "nondegenerate": True,
                }
            )
    write_csv(
        directory / "e034_attention_metrics.csv",
        list(attention_rows[0]),
        attention_rows,
    )

    cache_rows = [
        {
            "StudyInstanceUID": uid,
            "selected_windows": 24,
            "cross_series_windows": 0,
            "zero_slice_windows": 0,
            "cache_pixel_diff": 0,
        }
        for uid in ids
    ]
    write_csv(
        directory / "e034_cache_metrics.csv",
        list(cache_rows[0]),
        cache_rows,
    )

    head_rows = [
        {
            "phase": "inner" if index < 40 else "outer",
            "model_key": "coatnet" if index % 2 == 0 else "caformer",
            "outer_fold": index % 5,
            "inner_fold": index % 4 if index < 40 else -1,
            "seed": 34001 + index,
            "epochs": 35 if index < 40 else 60,
            "train_count": 384 if index < 40 else 512,
            "eval_count": 128,
            "eval_macro_auc": 0.7,
            "head_updated": True,
            "max_gradient_norm": 1.0,
            "seconds": 1.0,
        }
        for index in range(60)
    ]
    write_csv(
        directory / "e034_head_metrics.csv",
        list(head_rows[0]),
        head_rows,
    )
    write_csv(
        directory / "e034_head_history.csv",
        ["phase", "model_key", "epoch", "loss"],
        [{"phase": "smoke", "model_key": "coatnet", "epoch": 0, "loss": 1.0}],
    )
    write_csv(
        directory / "e034_feature_metrics.csv",
        ["model_key", "feature_shape", "feature_finite", "feature_std"],
        [
            {
                "model_key": "coatnet",
                "feature_shape": "[640, 24, 1024]",
                "feature_finite": True,
                "feature_std": 1.0,
            },
            {
                "model_key": "caformer",
                "feature_shape": "[640, 24, 768]",
                "feature_finite": True,
                "feature_std": 1.0,
            },
        ],
    )
    (directory / "e034_inner_fold_metrics.json").write_text(
        json.dumps(
            [
                {
                    "outer_fold": fold,
                    "max_prevalence_deviation": 0.01,
                    "mean_prevalence_deviation": 0.005,
                    "fold_sizes": [128, 128, 128, 128],
                }
                for fold in range(5)
            ],
            indent=2,
        )
    )

    required_gates = {
        "identity_exact": True,
        "gold_labels_never_evaluated": True,
        "gold_images_never_encoded": True,
        "cache_contract_exact": True,
        "model_shapes_exact": True,
        "features_finite_nondegenerate": True,
        "all_heads_updated": True,
        "outer_oof_complete": True,
        "outer_fold_coverage_exact": True,
        "all_fold_targets_two_classes": True,
        "attention_nondegenerate": True,
    }
    caformer_signal_gates = {
        "caformer_oof_gain_ge_001": pass_signal,
        "caformer_fold_wins_ge_4": True,
        "caformer_target_wins_ge_8": True,
        "caformer_robust_targets_ge_6": True,
        "caformer_min_target_delta_ge_minus003": True,
        "caformer_bootstrap_win_ge_090": True,
        "caformer_bootstrap_ci_low_positive": True,
    }
    blend_signal_gates = {
        "blend_oof_gain_ge_001": True,
        "blend_fold_wins_ge_4": True,
        "blend_target_wins_ge_8": True,
        "blend_min_target_delta_ge_minus0005": True,
        "blend_bootstrap_win_ge_090": True,
        "blend_bootstrap_ci_low_positive": True,
    }
    decision = (
        "PASS_E034_PROMOTE_CAFO_FINETUNE_NO_GOLD"
        if pass_signal
        else "STOP_E034_CAFO_CV_FAILED"
    )
    identity = {
        "seed": 31021,
        "counts": {"cv": 640, "outer_folds": 5, "gold_excluded": 58},
        "train_ids_sha256": "smoke",
        "validation_ids_sha256": "smoke",
        "cv_ids_sha256": validator.ids_sha256(ids),
        "gold_ids_sha256": "smoke",
        "all_ids_sha256": "smoke",
        "extra_ids_sha256": "smoke",
        "outer_fold_ids_sha256": ["smoke"] * 5,
        "outer_fold_class_counts": [],
        "outer_balance": {},
        "identity_exact": True,
        "gold_exclusion_only": True,
        "gold_label_evaluation_count": 0,
        "gold_image_encoding_count": 0,
    }
    caformer_bootstrap = validator.paired_bootstrap(
        labels, coatnet, caformer, 31021 + 3401
    )
    blend_bootstrap = validator.paired_bootstrap(
        labels, coatnet, protected, 31021 + 3402
    )
    summary = {
        "experiment": "E034 nested five-fold CAFormer CV and protected blend",
        "decision": decision,
        "identity": identity,
        "configuration": {},
        "oof": {
            "coatnet_macro_auc": macro_auc["coatnet"],
            "caformer_macro_auc": macro_auc["caformer"],
            "protected_blend_macro_auc": macro_auc["protected_blend"],
            "caformer_gain": macro_auc["caformer"] - macro_auc["coatnet"],
            "protected_blend_gain": (
                macro_auc["protected_blend"] - macro_auc["coatnet"]
            ),
            "caformer_target_wins": 12,
            "protected_blend_target_wins": 12,
            "caformer_fold_wins": 5,
            "protected_blend_fold_wins": 5,
            "caformer_robust_target_count": 12,
            "caformer_min_target_delta": float(
                np.min(target_auc["caformer"] - target_auc["coatnet"])
            ),
            "protected_blend_min_target_delta": float(
                np.min(target_auc["protected_blend"] - target_auc["coatnet"])
            ),
            "prediction_correlation": float(
                np.corrcoef(coatnet.reshape(-1), caformer.reshape(-1))[0, 1]
            ),
        },
        "bootstrap": {
            "caformer_vs_coatnet": caformer_bootstrap,
            "protected_blend_vs_coatnet": blend_bootstrap,
        },
        "required_gates": required_gates,
        "caformer_signal_gates": caformer_signal_gates,
        "blend_signal_gates": blend_signal_gates,
        "gold": {
            "ids_used_for_exclusion_only": True,
            "label_evaluation_count": 0,
            "image_encoding_count": 0,
            "prediction_artifact_created": False,
        },
        "checkpoint_exported": False,
        "feature_cache_exported": False,
        "submission_created": False,
    }
    (directory / "e034_summary.json").write_text(json.dumps(summary, indent=2))
    (directory / "e034_identity.json").write_text(json.dumps(identity, indent=2))

    gate_rows = []
    for group, gates in (
        ("required", required_gates),
        ("caformer_signal", caformer_signal_gates),
        ("blend_signal", blend_signal_gates),
    ):
        for gate, passed in gates.items():
            gate_rows.append({"group": group, "gate": gate, "passed": passed})
    write_csv(
        directory / "e034_gates.csv",
        ["group", "gate", "passed"],
        gate_rows,
    )


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert len(notebook.cells) == 19
    source = "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)

    assert "N_OUTER_FOLDS, N_INNER_FOLDS = 5, 4" in source
    assert "LAB = TARGETS" in source
    assert "N_TARGETS = len(TARGETS)" in source
    assert "LAB = len(TARGETS)" not in source
    assert "choose_balanced_folds" in source
    assert "choose_inner_alphas" in source
    assert "GOLD_LABEL_EVALUATION_COUNT = 0" in source
    assert "GOLD_IMAGE_ENCODING_COUNT = 0" in source
    assert "y_gold" not in source
    assert "gold_predictions" not in source
    assert "train[TARGETS]" not in source
    assert "gold58_audit_macro" not in source
    assert "e034_oof_predictions.npz" in source
    assert "PASS_E034_PROMOTE_CAFO_FINETUNE_NO_GOLD" in source

    setup_tree = ast.parse(notebook.cells[1].source)
    slots = None
    for node in setup_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "SLOTS"
            for target in node.targets
        ):
            slots = ast.literal_eval(node.value)
            break
    assert slots == [
        ("sagittal_fluid", "Sagittal", 1, 18),
        ("sagittal_other", "Sagittal", 0, 14),
        ("coronal_fluid", "Coronal", 1, 12),
        ("coronal_other", "Coronal", 0, 8),
        ("axial", "Axial", -1, 12),
    ]
    assert all(len(slot) == 4 for slot in slots)

    identity_cell = notebook.cells[6].source
    training_cell = notebook.cells[12].source
    primitive_source = extract_definitions(
        identity_cell, {"module_sha256", "choose_balanced_folds"}
    )
    training_source = extract_definitions(
        training_cell,
        {
            "TargetWindowMIL",
            "predict_head",
            "train_fixed_head",
            "choose_inner_alphas",
        },
    )
    namespace = {
        "np": np,
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
        "gc": gc,
        "time": time,
        "hashlib": hashlib,
        "roc_auc_score": roc_auc_score,
        "DEVICE": torch.device("cpu"),
        "HEAD_DIM": 32,
        "TARGETS": [
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
        ],
        "ALPHA_GRID": [0.0, 0.25, 0.5, 0.75, 1.0],
        "INNER_MIN_GAIN": 0.003,
    }
    namespace["target_aucs"] = load_validator().target_aucs
    exec(primitive_source + "\n\n" + training_source, namespace)

    synthetic_labels = np.stack(
        [
            ((np.arange(640) + target) % (3 + target % 5) == 0).astype(np.float32)
            for target in range(12)
        ],
        axis=1,
    )
    assignments, _ = namespace["choose_balanced_folds"](
        synthetic_labels, 5, 340, trials=128
    )
    assert np.array_equal(np.bincount(assignments), np.full(5, 128))

    rng = np.random.default_rng(34)
    features = rng.normal(size=(64, 8, 16)).astype(np.float16)
    labels = np.stack(
        [
            ((np.arange(64) + target) % 2).astype(np.float32)
            for target in range(12)
        ],
        axis=1,
    )
    predictions, attention, history, record = namespace["train_fixed_head"](
        model_key="smoke",
        features=features,
        labels_soft=labels,
        labels_binary=labels,
        train_indexes=np.arange(48),
        eval_indexes=np.arange(48, 64),
        seed=34001,
        epochs=2,
        phase="smoke",
        outer_fold=0,
        return_attention=True,
    )
    assert predictions.shape == (16, 12)
    assert attention.shape == (16, 12, 8)
    assert np.isfinite(predictions).all()
    assert np.allclose(attention.sum(-1), 1.0, atol=1e-6)
    assert len(history) == 2
    assert record["head_updated"] is True

    selected_alphas, alpha_rows = namespace["choose_inner_alphas"](
        labels[48:],
        predictions,
        np.clip(predictions + rng.normal(0, 0.05, predictions.shape), 0, 1),
    )
    assert selected_alphas.shape == (12,)
    assert len(alpha_rows) == 12

    with tempfile.TemporaryDirectory() as temporary:
        label_namespace = {
            "base64": base64,
            "hashlib": hashlib,
            "json": json,
            "np": np,
            "pd": pd,
            "Path": Path,
            "WORK": Path(temporary),
            "TARGETS": namespace["TARGETS"],
        }
        exec(notebook.cells[5].source, label_namespace)
        label_path = Path(label_namespace["_build_e021_labels"]())
        assert label_path.exists()
        assert hashlib.sha256(label_path.read_bytes()).hexdigest() == (
            "2ddc7e910c375857408cc479a3834e080b8016db670c93fc984c9c4e9a6e1613"
        )
        manifest = json.loads(
            (Path(temporary) / "e021_label_manifest.json").read_text()
        )
        assert manifest["rows"] == 4407
        assert manifest["gold_fields_in_manifest"] is False
        assert not any("gold58" in key.lower() for key in manifest)

    metadata = json.loads(METADATA.read_text())
    assert metadata["id"] == (
        "chinnaphatkhun/rsna-knee-e034-caformer-cv-protected-blend"
    )
    assert metadata["code_file"] == NOTEBOOK.name
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == []

    validator = load_validator()
    full_bootstrap = validator.paired_bootstrap

    def smoke_bootstrap(labels, baseline, challenger, seed, repeats=2000):
        del repeats
        return full_bootstrap(
            labels, baseline, challenger, seed=seed, repeats=50
        )

    validator.paired_bootstrap = smoke_bootstrap
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        write_fixture(directory, validator, pass_signal=True)
        failures, summary = validator.validate(directory, require_signal=True)
        assert failures == [], failures
        assert summary["decision"] == "PASS_E034_PROMOTE_CAFO_FINETUNE_NO_GOLD"

        (directory / "submission.csv").write_text("forbidden\n")
        failures, _ = validator.validate(directory)
        assert any("forbidden submission.csv" in failure for failure in failures)
        (directory / "submission.csv").unlink()

        with np.load(directory / "e034_oof_predictions.npz", allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload["folds"] = payload["folds"].copy()
        payload["folds"][0] = 1
        np.savez_compressed(directory / "e034_oof_predictions.npz", **payload)
        failures, _ = validator.validate(directory)
        assert any("outer fold coverage mismatch" in failure for failure in failures)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        write_fixture(directory, validator, pass_signal=False)
        failures, summary = validator.validate(directory)
        assert failures == [], failures
        assert summary["decision"] == "STOP_E034_CAFO_CV_FAILED"
        failures, _ = validator.validate(directory, require_signal=True)
        assert failures == ["promotion signal required"], failures

    print("E034 smoke PASS")


if __name__ == "__main__":
    main()
