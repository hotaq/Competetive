#!/usr/bin/env python3
"""Local structural and fail-closed smoke tests for E033."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e033-caformer-b36-paired-scout.ipynb"
METADATA = HERE / "kernel-metadata.json"
BASELINE = HERE / "e029_zero_step_baseline.npz"


def load_validator():
    spec = importlib.util.spec_from_file_location("e033_validator", HERE / "validate_output.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_gate_csv(path: Path, summary: dict) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "gate", "passed"])
        writer.writeheader()
        for group, key in (
            ("required", "required_gates"),
            ("pre_gold", "pre_gold_gates"),
            ("gold_signal", "gold_signal_gates"),
        ):
            for gate, passed in summary[key].items():
                writer.writerow({"group": group, "gate": gate, "passed": passed})


def base_summary(validator, open_gold: bool, pass_gold: bool) -> dict:
    required = {"identity_exact": True, "cache_contract_exact": True}
    pre_gold = {"caformer_validation_gain": open_gold}
    gold_signal = {"caformer_gold_gain": pass_gold if open_gold else False}
    decision = (
        validator.PASS_DECISION
        if open_gold and pass_gold
        else "STOP_E033_GOLD_SIGNAL_FAILED"
        if open_gold
        else "STOP_E033_PRE_GOLD_FAILED"
    )
    model_rows = []
    for model_key, architecture in validator.EXPECTED_MODELS.items():
        feature_dim = validator.EXPECTED_FEATURE_DIMS[model_key]
        model_rows.append({
            "model_key": model_key,
            "architecture": architecture,
            "parameter_count": 80_000_000,
            "num_features": feature_dim,
            "input_size": [3, 384, 384],
            "feature_shape": [640, 24, feature_dim],
            "feature_finite": True,
            "feature_std": 1.0,
            "windows_per_second": 10.0,
            "peak_gpu_gib": 4.0,
            "state_sha256": model_key[0] * 64,
        })
    head_records = []
    for model_key in validator.EXPECTED_MODELS:
        for seed in validator.HEAD_SEEDS:
            head_records.append({
                "model_key": model_key,
                "seed": seed,
                "best_epoch": 10,
                "validation_macro_auc": 0.8,
                "input_dim": validator.EXPECTED_FEATURE_DIMS[model_key],
                "hidden_dim": 512,
                "updated": True,
            })
    gold_metrics = None
    if open_gold:
        gold_metrics = {
            "evaluation_count": 1,
            "e029_finetuned_coatnet_macro_auc": 0.9174298724699895,
            "paired_frozen_coatnet_macro_auc": 0.80,
            "paired_frozen_caformer_macro_auc": 0.81,
            "delta_caformer_minus_frozen_coatnet": 0.01,
            "target_wins": 8,
            "min_target_delta": -0.01,
            "paired_bootstrap": {"win_rate": 0.75},
        }
    return {
        "decision": decision,
        "required_gates": required,
        "pre_gold_gates": pre_gold,
        "gold_signal_gates": gold_signal,
        "identity": {
            "counts": {"train": 512, "validation": 128, "gold": 58},
            "train_ids_sha256": validator.EXPECTED_TRAIN_IDS_SHA256,
            "validation_ids_sha256": validator.EXPECTED_VALIDATION_IDS_SHA256,
            "gold_ids_sha256": validator.EXPECTED_GOLD_IDS_SHA256,
            "baseline_payload_sha256": validator.EXPECTED_BASELINE_PAYLOAD_SHA256,
            "identity_exact": True,
        },
        "input": {
            "shape": [1, 24, 3, 384, 384],
            "safe_same_series_windows": True,
            "cross_series_window_count": 0,
            "zero_slice_window_count": 0,
        },
        "models": model_rows,
        "training": {
            "mode": "paired_frozen_backbones_fresh_target_window_mil",
            "head_seeds": validator.HEAD_SEEDS,
            "shared_hidden_dim": 512,
            "head_records": head_records,
            "selection_split": "fixed_non_gold_validation",
        },
        "validation": {
            "coatnet_macro_auc": 0.79,
            "caformer_macro_auc": 0.80,
            "delta_caformer_minus_coatnet": 0.01,
            "target_wins": 8,
            "min_target_delta": -0.01,
            "prediction_correlation": 0.8,
        },
        "gold58": {
            "gate_passed": open_gold,
            "evaluation_count": int(open_gold),
            "metrics": gold_metrics,
        },
        "submission_created": False,
        "checkpoint_exported": False,
        "feature_cache_exported": False,
    }


def populate(directory: Path, validator, summary: dict, val_ids, gold_ids) -> None:
    for name in (
        "e033_attention_metrics.csv",
        "e033_cache_metrics.csv",
        "e033_feature_metrics.csv",
        "e033_head_history.csv",
        "e033_identity.json",
        "e033_validation_target_metrics.csv",
    ):
        (directory / name).write_text("smoke\n")
    zeros_val = np.zeros((128, 12), np.float32)
    np.savez_compressed(
        directory / "e033_validation_predictions.npz",
        ids=val_ids,
        y_true=zeros_val,
        coatnet=np.full_like(zeros_val, 0.4),
        caformer=np.full_like(zeros_val, 0.6),
    )
    if summary["gold58"]["evaluation_count"]:
        zeros_gold = np.zeros((58, 12), np.float32)
        np.savez_compressed(
            directory / "e033_gold58_predictions.npz",
            ids=gold_ids,
            y_true=zeros_gold,
            e029_finetuned_coatnet=np.full_like(zeros_gold, 0.5),
            paired_frozen_coatnet=np.full_like(zeros_gold, 0.4),
            paired_frozen_caformer=np.full_like(zeros_gold, 0.6),
        )
        (directory / "e033_gold58_target_metrics.csv").write_text("smoke\n")
    (directory / "e033_summary.json").write_text(json.dumps(summary))
    write_gate_csv(directory / "e033_gates.csv", summary)


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert len(notebook.cells) == 19 and len(code_cells) == 11
    combined = "\n\n".join(code_cells)
    compile(combined, str(NOTEBOOK), "exec")
    for banned in ("kaggle competitions submit", "torch.save(", "submission.csv\".write"):
        assert banned not in combined
    for required in (
        "caformer_b36.sail_in22k_ft_in1k_384",
        "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        "if pre_gold_pass:",
        "paired_frozen_backbones_fresh_target_window_mil",
        "TRAIN_N, VAL_N, K_WINDOWS = 512, 128, 24",
        "HEAD_SEEDS = [33001, 33002, 33003]",
    ):
        assert required in combined

    metadata = json.loads(METADATA.read_text())
    assert metadata["id"] == "chinnaphatkhun/rsna-knee-e033-caformer-b36-paired-scout"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == []

    tree = ast.parse(combined)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TargetWindowMIL"
    )
    namespace = {
        "HEAD_DIM": 512,
        "TARGETS": list(range(12)),
        "nn": nn,
        "torch": torch,
    }
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), "head-smoke", "exec"), namespace)
    head = namespace["TargetWindowMIL"](input_dim=768)
    features = torch.randn(3, 24, 768)
    logits, attention = head(features, return_attention=True)
    assert logits.shape == (3, 12)
    assert attention.shape == (3, 12, 24)
    assert torch.allclose(attention.sum(-1), torch.ones(3, 12), atol=1e-6)
    logits.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )

    validator = load_validator()
    with np.load(BASELINE, allow_pickle=True) as payload:
        gold_ids = payload["ids"].astype(str)
    assert validator.ids_sha256(gold_ids.tolist()) == validator.EXPECTED_GOLD_IDS_SHA256
    val_ids = np.asarray([f"val-{index}" for index in range(128)])
    # The validator pins the real validation hash. Synthetic smoke IDs are accepted by
    # replacing only the expected constant inside this isolated module instance.
    validator.EXPECTED_VALIDATION_IDS_SHA256 = validator.ids_sha256(val_ids.tolist())

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        stop_summary = base_summary(validator, open_gold=False, pass_gold=False)
        populate(directory, validator, stop_summary, val_ids, gold_ids)
        validator.validate(directory)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        pass_summary = base_summary(validator, open_gold=True, pass_gold=True)
        populate(directory, validator, pass_summary, val_ids, gold_ids)
        validator.validate(directory, require_signal=True)
        (directory / "submission.csv").write_text("forbidden\n")
        try:
            validator.validate(directory)
        except SystemExit as error:
            assert "submission.csv" in str(error)
        else:
            raise AssertionError("validator accepted a forbidden submission")

    print("E033 local smoke passed")


if __name__ == "__main__":
    main()
