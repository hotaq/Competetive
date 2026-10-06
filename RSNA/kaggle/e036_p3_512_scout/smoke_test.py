#!/usr/bin/env python3
"""Local static and behavioural checks for the E036 notebook before any Kaggle push."""

from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e036-p3-512-scout.ipynb"
METADATA = HERE / "kernel-metadata.json"

# `state_dict()` is deliberately permitted: E036 keeps the validation-selected
# weights in CPU memory so the Gold58 veto scores the chosen epoch. What must
# stay banned is anything that writes weights or a submission to disk.
BANNED = [
    "kaggle competitions submit",
    "submission.csv",
    "torch.save",
    "safetensors",
]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


print("== notebook structure ==")
notebook = nbf.read(NOTEBOOK, as_version=4)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
check(len(code_cells) >= 8, f"code cell count {len(code_cells)} >= 8")

source = "\n\n".join(cell.source for cell in code_cells)
for index, cell in enumerate(code_cells):
    try:
        compile(cell.source, f"<cell {index}>", "exec")
    except SyntaxError as error:  # pragma: no cover - reported through failures
        check(False, f"cell {index} compiles ({error})")
else:
    check(True, "all code cells compile")

print("== banned side effects ==")
for token in BANNED:
    check(token not in source, f"no `{token}`")
check("e036_summary.json" in source, "writes e036_summary.json")
check("shutil.rmtree(CACHE" in source, "removes the exact64 cache before exit")

print("== declared contract ==")
tree = ast.parse(source)
constants: dict[str, object] = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                constants[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
        elif isinstance(target, ast.Tuple):
            try:
                values = ast.literal_eval(node.value)
            except ValueError:
                continue
            for element, value in zip(target.elts, values):
                if isinstance(element, ast.Name):
                    constants[element.id] = value

expected = {
    "SPLIT_SEED": 3601,
    "MODEL_SEEDS": [2701, 2702],
    "N_TRAIN": 512,
    "N_VALID": 512,
    "EPOCHS": 5,
    "K_TRAIN": 12,
    "K_VAL": 12,
    "K_EVAL": 42,
    "RES": 384,
    "CROP_MM": 140.0,
    "BACKBONE_LR": 5e-6,
    "HEAD_LR": 1e-4,
    "MIN_VALID_GAIN": 0.005,
    "MIN_BOOTSTRAP_WIN": 0.90,
    "MAX_TARGET_DROP": 0.030,
    "GOLD58_VETO_DROP": 0.005,
    "EXPECTED_SAFE_AUC": 0.9174298724699895,
    "GRAD_CHECKPOINTING": True,
    "MAX_SCALER_SKIP_RATE": 0.02,
    "PROBE_PEAK_GIB_LIMIT": 13.0,
}
for name, value in expected.items():
    check(constants.get(name) == value, f"{name} == {value!r} (got {constants.get(name)!r})")

print("== T4 mixed-precision policy ==")
# E036 v1 died in scaled_dot_product_attention: T4 is sm_75, so bfloat16 autocast
# leaves only the math attention backend, which retains the full attention matrix.
check("torch.bfloat16" not in source, "no bfloat16 autocast anywhere")
check("AMP_DTYPE = torch.float16" in source, "AMP_DTYPE is float16")
check("torch.amp.GradScaler" in source, "float16 path uses a GradScaler")
check(source.count("def amp_train_step") == 1, "amp_train_step is defined once")
check(source.count("def enable_grad_checkpointing") == 1,
      "enable_grad_checkpointing is defined once")

amp_step = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "amp_train_step")
amp_step_source = ast.get_source_segment(source, amp_step)
check(amp_step_source.index("scaler.unscale_") < amp_step_source.index("clip_grad_norm_"),
      "gradients are unscaled before the 3.0 clip")
check("scaler.get_scale() >= scale_before" in amp_step_source,
      "a skipped step is detected from the loss scale backing off")

run_seed_node = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "run_seed")
run_seed_body = ast.get_source_segment(source, run_seed_node)
check("amp_train_step(" in run_seed_body, "the training loop goes through amp_train_step")
check("enable_grad_checkpointing(model, GRAD_CHECKPOINTING)" in run_seed_body,
      "checkpointing is enabled for training")
check("enable_grad_checkpointing(model, False)" in run_seed_body,
      "checkpointing is disabled before evaluation")
check("skipped_optimizer_steps" in run_seed_body,
      "scaler-skipped steps are counted separately")

print("== the memory probe runs before the expensive stages ==")
def cell_index(token):
    return next(index for index, cell in enumerate(code_cells) if token in cell.source)

probe_cell = cell_index("probe_peak_gib =")
check(probe_cell < cell_index("cache workers"), "probe runs before the exact64 cache build")
check(probe_cell < cell_index("zero-step Gold58"), "probe runs before the zero-step baseline")
probe_source = code_cells[probe_cell].source
check(f"torch.randn(BATCH_STUDIES, K_TRAIN, 3, RES, RES" in probe_source,
      "probe uses the exact training bag shape")
check("raise RuntimeError" in probe_source, "probe fails closed when it does not fit")
check("lr=0.0" in probe_source, "probe does not move the checkpoint weights")

print("== gold58 is veto, not selection ==")
run_seed = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "run_seed")
run_seed_source = ast.get_source_segment(source, run_seed)
best_epoch_line = next(line for line in run_seed_source.splitlines() if "best_epoch =" in line)
check("valid_macro_auc" in "".join(
    line for line in run_seed_source.splitlines() if "macros = " in line
), "epoch selection reads valid_macro_auc")
check("argmax(macros)" in best_epoch_line, "best epoch is argmax over validation macros")
gold_call_index = run_seed_source.index("gold_predictions = predict_studies")
check(run_seed_source.index("best_epoch = ") < gold_call_index,
      "Gold58 is evaluated only after the validation-selected epoch is fixed")
check(run_seed_source.count("predict_studies(model, gold_ids") == 1,
      "Gold58 is opened exactly once per seed")
check(run_seed_source.index("model.load_state_dict(best_state") < gold_call_index,
      "validation-selected weights are restored before the Gold58 veto")
check("best_state_epoch != best_epoch" in run_seed_source,
      "retained and selected epochs are cross-checked")

print("== sampler determinism and slot coverage ==")
slot_balanced = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "slot_balanced_centers")
namespace: dict[str, object] = {"random": random, "K_TRAIN": 12}
exec(compile(ast.Module(body=[slot_balanced], type_ignores=[]), "<sampler>", "exec"), namespace)
namespace["SLOTS"] = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
namespace["candidates_by_uid"] = {"S": list(range(1, 40))}
namespace["slot_by_center"] = {
    "S": {center: namespace["SLOTS"][center % 5][0] for center in range(1, 40)}
}
sampler = namespace["slot_balanced_centers"]
first, available = sampler("S", 2701, 0, 0)
again, _ = sampler("S", 2701, 0, 0)
other_seed, _ = sampler("S", 2702, 0, 0)
check(len(first) == 12, f"bag size 12 (got {len(first)})")
check(first == again, "sampler is deterministic for a fixed (seed, epoch, index)")
check(first != other_seed, "sampler differs across model seeds")
covered = {namespace["slot_by_center"]["S"][center] for center in first}
check(covered == set(available), "bag covers every available slot")

print("== split determinism ==")
ids = [f"study{index:05d}" for index in range(4349)]
def split(seed):
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(ids, dtype=object)[rng.permutation(len(ids))])
    return [str(value) for value in shuffled[:512]], [str(value) for value in shuffled[512:1024]]

train_a, valid_a = split(3601)
train_b, valid_b = split(3601)
check(train_a == train_b and valid_a == valid_b, "512/512 split is reproducible")
check(not set(train_a) & set(valid_a), "train and validation are disjoint")
check(len(train_a) == 512 and len(valid_a) == 512, "split sizes are exactly 512/512")

print("== kernel metadata ==")
metadata = json.loads(METADATA.read_text())
check(metadata["is_private"] is True, "kernel is private")
check(metadata["enable_gpu"] is True, "GPU enabled")
check(metadata["enable_internet"] is False, "internet disabled")
check(metadata["machine_shape"] == "NvidiaTeslaT4", "T4 accelerator")
check(metadata["code_file"] == NOTEBOOK.name, "code_file matches the notebook")
check("rsna-knee-abnormality-detection" in metadata["competition_sources"],
      "competition attached")
check(len(metadata["dataset_sources"]) == 6, "six dataset sources attached")

print()
if failures:
    print(f"SMOKE FAILED ({len(failures)} checks)")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("SMOKE PASSED")
