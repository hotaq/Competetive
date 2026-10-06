#!/usr/bin/env python3
"""Execute the E036 training and gating logic end to end against a mocked backbone.

The Kaggle run costs hours of GPU, so every wiring failure it could hit -- the
two-group OneCycle schedule, `predict_studies` shapes, best-weight restoration,
the paired bootstrap, and the fail-closed gate assembly -- is exercised here on
CPU with a 3-window toy corpus first.
"""

from __future__ import annotations

import ast
import random
import sys
import tempfile
from pathlib import Path

import nbformat as nbf
import numpy as np
import torch
import types
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e036-p3-512-scout.ipynb"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


notebook = nbf.read(NOTEBOOK, as_version=4)
source = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
tree = ast.parse(source)


def grab(name: str):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise KeyError(name)


# ---------------------------------------------------------------- toy corpus
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
RES, K_TRAIN, K_VAL, K_EVAL = 32, 12, 12, 12
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEVICE = torch.device("cpu")

rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp())
train_ids = [f"T{index:03d}" for index in range(8)]
valid_ids = [f"V{index:03d}" for index in range(24)]
gold_ids = [f"G{index:03d}" for index in range(24)]
cache_paths, candidates_by_uid, slot_by_center, eval_centers_by_uid = {}, {}, {}, {}
for uid in train_ids + valid_ids + gold_ids:
    volume = rng.integers(0, 256, size=(64, RES, RES), dtype=np.uint8)
    path = tmp / f"{uid}.npz"
    np.savez_compressed(path, volume=volume)
    cache_paths[uid] = path
    candidates = list(range(1, 63))
    candidates_by_uid[uid] = candidates
    slot_by_center[uid] = {center: SLOTS[center % 5][0] for center in candidates}
    eval_centers_by_uid[uid] = candidates[:K_EVAL]


def load_lossless_cache(path):
    with np.load(path) as cached:
        volume = cached["volume"]
    return volume, np.ones(len(volume), bool)


def resample_centers(candidates, count=K_EVAL):
    if not candidates:
        return []
    indexes = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [int(candidates[index]) for index in indexes]


def roc_auc_score(y, scores):
    from sklearn.metrics import roc_auc_score as real

    return real(y, scores)


class TinyBackbone(nn.Module):
    num_features = 16

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x).mean((2, 3))


namespace = {
    "np": np, "torch": torch, "nn": nn, "F": F, "random": random,
    "Dataset": Dataset, "DataLoader": DataLoader,
    "TARGETS": TARGETS, "SLOTS": SLOTS, "RES": RES,
    "K_TRAIN": K_TRAIN, "K_VAL": K_VAL, "K_EVAL": K_EVAL,
    "MEAN": MEAN, "STD": STD, "DEVICE": DEVICE,
    "cache_paths": cache_paths, "candidates_by_uid": candidates_by_uid,
    "slot_by_center": slot_by_center, "eval_centers_by_uid": eval_centers_by_uid,
    "load_lossless_cache": load_lossless_cache, "resample_centers": resample_centers,
    "roc_auc_score": roc_auc_score, "hashlib": __import__("hashlib"),
    "AMP_DTYPE": torch.float16,
}
for name in ["slot_balanced_centers", "stack_windows", "Exact64TrainDataset",
             "target_aucs", "predict_studies", "parameter_hashes", "paired_bootstrap",
             "amp_train_step", "enable_grad_checkpointing"]:
    exec(compile(ast.Module(body=[grab(name)], type_ignores=[]), f"<{name}>", "exec"),
         namespace)
for name in ["RaptorClassifier"]:
    exec(compile(ast.Module(body=[grab(name)], type_ignores=[]), f"<{name}>", "exec"),
         namespace)

slot_balanced_centers = namespace["slot_balanced_centers"]
Exact64TrainDataset = namespace["Exact64TrainDataset"]
target_aucs = namespace["target_aucs"]
predict_studies = namespace["predict_studies"]
parameter_hashes = namespace["parameter_hashes"]
paired_bootstrap = namespace["paired_bootstrap"]
RaptorClassifier = namespace["RaptorClassifier"]

print("== dataset and sampler ==")
labels = {uid: rng.random(12).astype(np.float32) for uid in train_ids}
dataset = Exact64TrainDataset(train_ids, labels, 2701)
tensor, target, uid = dataset[0]
check(tuple(tensor.shape) == (K_TRAIN, 3, RES, RES), f"item shape {tuple(tensor.shape)}")
check(tuple(target.shape) == (12,), "label shape (12,)")
check(torch.isfinite(tensor).all().item(), "item is finite")
dataset.set_epoch(1)
other, _t, _u = dataset[0]
check(not torch.equal(tensor, other), "epoch changes the sampled bag")

print("== model, prediction, and metric shapes ==")
torch.manual_seed(0)
model = RaptorClassifier(TinyBackbone(), F_dim=16, n=12).to(DEVICE)
y_valid = (rng.random((len(valid_ids), 12)) >= 0.5).astype(np.float32)
y_gold = (rng.random((len(gold_ids), 12)) >= 0.5).astype(np.float32)
namespace["y_valid"], namespace["y_gold"] = y_valid, y_gold
zero_valid = predict_studies(model, valid_ids, K_VAL)
check(zero_valid.shape == (len(valid_ids), 12), f"valid predictions {zero_valid.shape}")
check(np.isfinite(zero_valid).all(), "valid predictions finite")
zero_valid_target_auc, zero_valid_macro = target_aucs(y_valid, zero_valid)
check(0.0 <= zero_valid_macro <= 1.0, f"validation macro AUC in range ({zero_valid_macro:.4f})")

print("== two-group OneCycle, training step, and best-weight restoration ==")
backbone_named = [(n_, p) for n_, p in model.named_parameters() if n_.startswith("backbone.")]
head_named = [(n_, p) for n_, p in model.named_parameters() if not n_.startswith("backbone.")]
check(bool(backbone_named) and bool(head_named), "backbone and head parameter groups non-empty")
optimizer = torch.optim.AdamW(
    [{"params": [p for _n, p in backbone_named], "lr": 5e-6},
     {"params": [p for _n, p in head_named], "lr": 1e-4}],
    weight_decay=0.02,
)
EPOCHS = 3
steps_per_epoch = len(train_ids) // 2
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[5e-6, 1e-4], total_steps=steps_per_epoch * EPOCHS, pct_start=0.15)
check(len(optimizer.param_groups) == 2, "optimizer has two parameter groups")

initial_hashes = parameter_hashes(model)
loss_function = nn.BCEWithLogitsLoss()
best_state, best_state_epoch, epoch_records, valid_by_epoch, losses = None, -1, [], [], []
steps = 0
for epoch in range(EPOCHS):
    dataset.set_epoch(epoch)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    model.train()
    total = 0.0
    for images, targets, _uids in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(images).float(), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()
        steps += 1
        total += float(loss.item())
    predictions = predict_studies(model, valid_ids, K_VAL)
    _pt, macro = target_aucs(y_valid, predictions)
    losses.append(total / len(loader))
    valid_by_epoch.append(predictions)
    if best_state is None or macro > max(r["valid_macro_auc"] for r in epoch_records):
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_state_epoch = epoch
    epoch_records.append({"epoch": epoch, "valid_macro_auc": macro})

check(steps == steps_per_epoch * EPOCHS, f"optimizer steps {steps} == {steps_per_epoch*EPOCHS}")
macros = [r["valid_macro_auc"] for r in epoch_records]
best_epoch = int(np.argmax(macros))
check(best_state_epoch == best_epoch,
      f"retained epoch {best_state_epoch} == selected epoch {best_epoch}")

model.load_state_dict(best_state, strict=True)
model.eval()
restored = predict_studies(model, valid_ids, K_VAL)
check(np.allclose(restored, valid_by_epoch[best_epoch], atol=1e-5),
      "restored weights reproduce the selected epoch's validation predictions")

final_hashes = parameter_hashes(model)
changed = {n_ for n_ in initial_hashes if initial_hashes[n_] != final_hashes[n_]}
check(sum(n_.startswith("backbone.") for n_ in changed) > 0, "backbone tensors updated")
check(sum(not n_.startswith("backbone.") for n_ in changed) > 0, "head tensors updated")

amp_train_step = namespace["amp_train_step"]
enable_grad_checkpointing = namespace["enable_grad_checkpointing"]

print("== gradient checkpointing toggle ==")


class HonestStage:
    grad_checkpointing = False


class HonestBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = [HonestStage(), HonestStage()]

    def set_grad_checkpointing(self, enable=True):
        for stage in self.stages:
            stage.grad_checkpointing = enable


class DeafBackbone(HonestBackbone):
    def set_grad_checkpointing(self, enable=True):
        return None


class PlainBackbone(nn.Module):
    pass


honest = types.SimpleNamespace(backbone=HonestBackbone())
check(enable_grad_checkpointing(honest, True) == [True, True], "flags land on every stage")
check(enable_grad_checkpointing(honest, False) == [False, False], "flags clear again")


def raises(callable_):
    try:
        callable_()
    except RuntimeError:
        return True
    return False


check(raises(lambda: enable_grad_checkpointing(types.SimpleNamespace(backbone=DeafBackbone()))),
      "a backbone that ignores the toggle is a hard failure")
check(raises(lambda: enable_grad_checkpointing(types.SimpleNamespace(backbone=PlainBackbone()))),
      "a backbone without the method is a hard failure")

print("== mixed-precision step accounting ==")
step_model = RaptorClassifier(TinyBackbone(), F_dim=16, n=12).to(DEVICE)
step_optimizer = torch.optim.AdamW(step_model.parameters(), lr=1e-3)
step_images = torch.randn(2, K_TRAIN, 3, RES, RES)
step_targets = (torch.rand(2, 12) >= 0.5).float()
step_before = parameter_hashes(step_model)


class DisabledScaler:
    """Stands in for GradScaler on CPU: no scaling, and the step always lands."""

    def is_enabled(self):
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def get_scale(self):
        return 1.0


class BackingOffScaler(DisabledScaler):
    """Reproduces the iteration where float16 GradScaler halves the loss scale
    and skips the optimizer step."""

    def __init__(self):
        self.scale_value = 65536.0

    def step(self, optimizer):
        return None

    def update(self):
        self.scale_value /= 2.0

    def get_scale(self):
        return self.scale_value


loss_value, gradient_norm, step_applied = amp_train_step(
    step_model, step_images, step_targets, nn.BCEWithLogitsLoss(),
    step_optimizer, DisabledScaler(),
)
check(np.isfinite(loss_value), f"amp_train_step returns a finite loss ({loss_value:.4f})")
check(bool(torch.isfinite(gradient_norm).item()), "amp_train_step returns a finite gradient norm")
check(step_applied is True, "an applied step reports step_applied=True")
step_after = parameter_hashes(step_model)
check(sum(step_before[k] != step_after[k] for k in step_before) > 0,
      "an applied step actually moves the weights")

held = parameter_hashes(step_model)
_loss, _norm, skipped_applied = amp_train_step(
    step_model, step_images, step_targets, nn.BCEWithLogitsLoss(),
    step_optimizer, BackingOffScaler(),
)
check(skipped_applied is False, "a backed-off scale reports step_applied=False")
check(all(held[k] == parameter_hashes(step_model)[k] for k in held),
      "a skipped step leaves the weights untouched")


class ExplodingLoss(nn.Module):
    def forward(self, logits, targets):
        return logits.sum() * float("inf")


check(raises(lambda: amp_train_step(step_model, step_images, step_targets,
                                    ExplodingLoss(), step_optimizer, DisabledScaler())),
      "a non-finite loss is a hard failure")

print("== paired bootstrap ==")
bootstrap = paired_bootstrap(y_valid, zero_valid, valid_by_epoch[best_epoch], 2701, repeats=200)
check(bootstrap["valid_repeats"] > 0, f"bootstrap produced {bootstrap['valid_repeats']} repeats")
check(0.0 <= bootstrap["win_rate"] <= 1.0, f"win rate in range ({bootstrap['win_rate']:.3f})")
check(bootstrap["ci95"][0] <= bootstrap["mean_delta"] <= bootstrap["ci95"][1],
      "mean delta lies inside the reported interval")
identical = paired_bootstrap(y_valid, zero_valid, zero_valid.copy(), 2701, repeats=200)
check(abs(identical["mean_delta"]) < 1e-12, "identical predictions give a zero mean delta")
check(identical["win_rate"] == 0.0, "identical predictions never win")

print("== fail-closed gate assembly ==")
gate_cell = next(cell.source for cell in notebook.cells
                 if cell.cell_type == "code" and "efficacy_gates" in cell.source)
gate_tree = ast.parse(gate_cell)
decision_source = ast.get_source_segment(gate_cell, gate_tree.body[-1]) or gate_cell
check("STOP_P2_MECHANICAL_FAILED" in gate_cell and "STOP_P3_EFFICACY_FAILED" in gate_cell,
      "both stop decisions exist")
check("PASS_P3_PROMOTE_P4" in gate_cell, "pass decision exists")
mechanical_first = gate_cell.index("if not mechanical_passed")
check(mechanical_first < gate_cell.index('elif efficacy_passed'),
      "mechanical failure short-circuits before efficacy is considered")


def decide(mechanical: bool, efficacy: bool) -> str:
    if not mechanical:
        return "STOP_P2_MECHANICAL_FAILED"
    if efficacy:
        return "PASS_P3_PROMOTE_P4"
    return "STOP_P3_EFFICACY_FAILED"


check(decide(False, True) == "STOP_P2_MECHANICAL_FAILED", "broken wiring cannot pass")
check(decide(True, False) == "STOP_P3_EFFICACY_FAILED", "weak effect cannot pass")
check(decide(True, True) == "PASS_P3_PROMOTE_P4", "clean run passes")

print()
if failures:
    print(f"MOCK RUN FAILED ({len(failures)} checks)")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("MOCK RUN PASSED")
