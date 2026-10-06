#!/usr/bin/env python3
"""Local structural and fail-closed smoke tests for E032."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e032-vjepa21-partial-finetune.ipynb"
METADATA = HERE / "kernel-metadata.json"
REFERENCE = HERE / "e031_mil_reference.npz"


def load_validator():
    spec = importlib.util.spec_from_file_location("e032_validator", HERE / "validate_output.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_gate_csv(path: Path, summary: dict) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "gate", "passed"])
        writer.writeheader()
        for group_name, key in (
            ("required", "required_gates"),
            ("pre_gold", "pre_gold_gates"),
            ("gold_signal", "gold_signal_gates"),
        ):
            for name, passed in summary[key].items():
                writer.writerow({"group": group_name, "gate": name, "passed": passed})


def base_summary(required_hashes: dict) -> dict:
    return {
        "decision": "STOP_E032_PRE_GOLD_FAILED",
        "required_gates": {"structural_smoke": True},
        "pre_gold_gates": {"validation_gate": False},
        "gold_signal_gates": {"gold_gate": False},
        "identity": {
            "counts": {"train": 512, "validation": 128, "gold": 58},
            **required_hashes,
            "payload_identity_exact": True,
        },
        "model": {
            "repo_commit": "204698b45b3712590f06245fbfba32d3be539812",
            "checkpoint_bytes": 1_664_223_428,
            "checkpoint_sha256": "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d",
            "unfrozen_blocks": [10, 11],
            "unfrozen_output_norm": 3,
            "trainable_prefixes": ["blocks.10.", "blocks.11.", "norms_block.3."],
            "frozen_encoder_unchanged": True,
            "trainable_encoder_parameter_count": 14_200_000,
            "trainable_encoder_ratio": 0.164,
            "trainable_relative_l2_drift": 0.001,
            "encoder_initial_sha256": "a" * 64,
            "encoder_final_sha256": "b" * 64,
        },
        "input": {
            "shape": [1, 3, 16, 384, 384],
            "feature_shape_per_slot": [8, 3072],
            "grayscale_repeated_rgb": True,
            "per_slot_independent": True,
            "cross_series_clip_count": 0,
        },
        "training": {
            "mode": "frozen_mil_warmstart_then_last2_encoder_blocks_plus_output_norm",
            "gradient_accumulation": 4,
            "optimizer_parameter_set_exact": True,
            "optimizer_excludes_frozen_encoder": True,
            "finetune_epochs_ran": 3,
        },
        "validation": {
            "selected_on": "fixed_non_gold_validation",
            "e031_reference_macro_auc": 0.746,
            "warmup_macro_auc": 0.750,
            "finetuned_macro_auc": 0.755,
            "gain_vs_warmup": 0.005,
            "gain_vs_e031": 0.009,
            "min_target_delta_vs_warmup": -0.01,
            "prediction_correlation_vs_warmup": 0.98,
        },
        "xai": {
            "target_count": 12,
            "nondegenerate_count": 12,
            "median_deletion_advantage": 0.01,
        },
        "gold58": {"gate_passed": False, "evaluation_count": 0, "metrics": None},
        "submission_created": False,
        "checkpoint_exported": False,
        "feature_cache_exported": False,
    }


def populate_common_files(directory: Path, summary: dict, val_ids: np.ndarray) -> None:
    for name in (
        "e032_cache_metrics.csv",
        "e032_clip_metrics.csv",
        "e032_finetune_history.csv",
        "e032_identity.json",
        "e032_validation_target_metrics.csv",
        "e032_warmup_history.csv",
        "e032_xai_deletion_metrics.csv",
    ):
        (directory / name).write_text("smoke\n")
    zeros = np.zeros((128, 12), np.float32)
    np.savez_compressed(
        directory / "e032_validation_predictions.npz",
        ids=val_ids,
        y_true=zeros,
        e031_reference=zeros,
        warmup=zeros,
        finetuned=zeros,
    )
    (directory / "e032_summary.json").write_text(json.dumps(summary))
    write_gate_csv(directory / "e032_gates.csv", summary)


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert len(notebook.cells) == 19 and len(code_cells) == 11
    combined = "\n\n".join(code_cells)
    compile(combined, str(NOTEBOOK), "exec")
    for banned in ("kaggle competitions submit", "submission.csv", "torch.save("):
        assert banned not in combined

    metadata = json.loads(METADATA.read_text())
    assert metadata["id"] == "chinnaphatkhun/rsna-knee-e032-vjepa21-partial-finetune"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == []

    tree = ast.parse(combined)
    mil_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TargetSlotMIL"
    )
    namespace = {
        "FEATURE_DIM": 3072,
        "MIL_DIM": 256,
        "TARGETS": list(range(12)),
        "SLOTS": list(range(5)),
        "TEMPORAL_TOKENS": 8,
        "math": math,
        "nn": nn,
        "torch": torch,
    }
    exec(compile(ast.Module(body=[mil_node], type_ignores=[]), "mil-smoke", "exec"), namespace)
    model = namespace["TargetSlotMIL"]()
    features = torch.randn(3, 5, 8, 3072)
    mask = torch.ones(3, 5, 8, dtype=torch.bool)
    mask[1, 4] = False
    logits, attention = model(features, mask, return_attention=True)
    assert logits.shape == (3, 12)
    assert attention.shape == (3, 12, 40)
    assert torch.allclose(attention.sum(-1), torch.ones(3, 12), atol=1e-6)
    assert float(attention[1, :, 32:].detach().abs().max()) == 0.0
    loss = logits.square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"prepare_training_clip", "encode_study"}
    }
    assert set(function_nodes) == {"prepare_training_clip", "encode_study"}

    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, values):
            batch = values.shape[0]
            signal = values.mean(dim=(1, 2, 3, 4)).view(batch, 1, 1) * self.scale
            base = torch.arange(32, dtype=values.dtype).view(1, 8, 4)
            return [signal + base, signal + base * 0.5]

    def mock_slot_frames(_uid, slot_index):
        if slot_index == 1:
            return None
        frames = np.full((4, 6, 6), 40 + slot_index, np.uint8)
        return frames, {"series": f"series-{slot_index}"}

    def mock_prepare_clip(frames):
        values = torch.from_numpy(frames).float().unsqueeze(0) / 255.0
        return values.repeat(3, 1, 1, 1)

    mock_encoder = MockEncoder()
    encode_namespace = {
        "CLIP_FRAMES": 4,
        "DEVICE": torch.device("cpu"),
        "EMBED_DIM": 4,
        "FEATURE_DIM": 8,
        "F": torch.nn.functional,
        "MEAN": torch.zeros(3, 1, 1, 1),
        "OUT_LAYERS": [0, 1],
        "RES": 6,
        "SLOTS": list(range(3)),
        "SPATIAL_TOKENS": 2,
        "STD": torch.ones(3, 1, 1, 1),
        "TEMPORAL_TOKENS": 2,
        "encoder": mock_encoder,
        "np": np,
        "prepare_clip": mock_prepare_clip,
        "slot_frames": mock_slot_frames,
        "torch": torch,
    }
    exec(
        compile(
            ast.Module(
                body=[
                    function_nodes["prepare_training_clip"],
                    function_nodes["encode_study"],
                ],
                type_ignores=[],
            ),
            "encode-study-smoke",
            "exec",
        ),
        encode_namespace,
    )
    encoded, encoded_mask = encode_namespace["encode_study"](
        "uid", augment_seed=123, gradients=True
    )
    assert encoded.shape == (1, 3, 2, 8)
    assert encoded_mask.shape == (1, 3, 2)
    assert encoded_mask[0, 0].all() and not encoded_mask[0, 1].any()
    encoded.sum().backward()
    assert mock_encoder.scale.grad is not None
    assert torch.isfinite(mock_encoder.scale.grad)
    inferred, inferred_mask = encode_namespace["encode_study"]("uid", gradients=False)
    assert inferred.shape == encoded.shape and torch.equal(inferred_mask, encoded_mask)

    with np.load(REFERENCE, allow_pickle=True) as payload:
        val_ids = payload["val_ids"]
        gold_ids = payload["gold_ids"]
    validator = load_validator()
    required_hashes = {
        "train_ids_sha256": validator.EXPECTED_TRAIN_IDS_SHA256,
        "validation_ids_sha256": validator.EXPECTED_VALIDATION_IDS_SHA256,
        "gold_ids_sha256": validator.EXPECTED_GOLD_IDS_SHA256,
        "baseline_payload_sha256": validator.EXPECTED_BASELINE_PAYLOAD_SHA256,
        "reference_payload_sha256": validator.EXPECTED_REFERENCE_PAYLOAD_SHA256,
    }

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        stop_summary = base_summary(required_hashes)
        populate_common_files(output, stop_summary, val_ids)
        result = validator.validate(output)
        assert result["decision"] == "STOP_E032_PRE_GOLD_FAILED"
        try:
            validator.validate(output, require_signal=True)
        except SystemExit:
            pass
        else:
            raise AssertionError("require_signal accepted a stop decision")

        stop_summary["decision"] = validator.PASS_DECISION
        stop_summary["pre_gold_gates"] = {"validation_gate": True}
        stop_summary["gold_signal_gates"] = {"gold_gate": True}
        stop_summary["gold58"] = {
            "gate_passed": True,
            "evaluation_count": 1,
            "metrics": {
                "evaluation_count": 1,
                "coatnet_macro_auc": 0.917,
                "e031_frozen_mil_macro_auc": 0.674,
                "e032_finetuned_macro_auc": 0.78,
                "delta_vs_e031": 0.106,
                "delta_vs_coatnet": -0.137,
                "min_target_delta_vs_e031": -0.01,
                "paired_bootstrap_vs_e031": {"win_rate": 0.8},
            },
        }
        zeros = np.zeros((58, 12), np.float32)
        np.savez_compressed(
            output / "e032_gold58_predictions.npz",
            ids=gold_ids,
            y_true=zeros,
            coatnet=zeros,
            e031_frozen_mil=zeros,
            e032_finetuned=zeros,
        )
        (output / "e032_gold58_target_metrics.csv").write_text("smoke\n")
        (output / "e032_summary.json").write_text(json.dumps(stop_summary))
        write_gate_csv(output / "e032_gates.csv", stop_summary)
        result = validator.validate(output, require_signal=True)
        assert result["promotion_signal"] is True

        (output / "submission.csv").write_text("forbidden\n")
        try:
            validator.validate(output)
        except SystemExit:
            pass
        else:
            raise AssertionError("validator accepted submission.csv")

    print("E032 smoke tests passed")


if __name__ == "__main__":
    main()
