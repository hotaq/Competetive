#!/usr/bin/env python3
"""Local static and behavioural checks for the E038 notebook before any Kaggle push."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e038-swa-ensemble-scout.ipynb"
METADATA = HERE / "kernel-metadata.json"

BANNED = [
    "kaggle competitions submit",
    "submission.csv",
    "torch.save",
    "safetensors",
]

# E038 trains nothing. Any of these appearing means the notebook stopped being
# an inference-only scout and its conclusions would no longer be about
# ensembling two frozen checkpoints.
BANNED_TRAINING = [
    "torch.optim",
    ".backward(",
    "GradScaler",
    "model.train()",
    "requires_grad",
    "load_state_dict(best_state",
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
compiled = True
for index, cell in enumerate(code_cells):
    try:
        compile(cell.source, f"<cell {index}>", "exec")
    except SyntaxError as error:  # pragma: no cover - reported through failures
        compiled = False
        check(False, f"cell {index} compiles ({error})")
check(compiled, "all code cells compile")

print("== banned side effects ==")
for token in BANNED:
    check(token not in source, f"no `{token}`")
check("e038_summary.json" in source, "writes e038_summary.json")
check("shutil.rmtree(CACHE" in source, "removes the exact64 cache before exit")
check('SCRATCH = Path("/kaggle/temp")' in source,
      "the ~10 GiB cache goes to scratch, not to the collected working directory")
check('CACHE = SCRATCH / "e038_exact64_cache"' in source, "cache lives under scratch")
check("else WORK" not in source, "scratch never falls back to the working directory")
check("SCRATCH.mkdir(parents=True, exist_ok=True)" in source, "scratch is created, not probed")
check("would be collected as kernel output" in source,
      "a scratch path under the working directory raises instead of running")
check("tempfile.mkdtemp" in source, "an unwritable /kaggle/temp falls back off the working directory")

print("== inference only ==")
for token in BANNED_TRAINING:
    check(token not in source, f"no `{token}`")
check(source.count(".eval()") >= 2, "both models are put in eval mode")
check('"trained": False' in source, "the summary records that nothing was trained")

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
    "EVAL_SEED": 3701,
    "BOOTSTRAP_SEED": 3802,
    "E037_N_EVAL": 2048,
    "EXPECTED_POOL": 4349,
    "K_EVAL": 42,
    "RES": 384,
    "IMG": 336,
    "CROP_MM": 140.0,
    "CHUNK": 6,
    "BOOTSTRAP_REPEATS": 2000,
    "MODEL_LABELS": ["base", "swa"],
    "WINDOW_SETS": ["42", "all"],
    "MIN_VALID_GAIN": 0.0020,
    "MIN_BOOTSTRAP_WIN": 0.98,
    "MAX_TARGET_DROP": 0.010,
    "GOLD58_MIN_TARGET_DELTA": -0.010,
    "REPRODUCTION_TOLERANCE": 0.0005,
    "RUNTIME_BUDGET_MINUTES": 420.0,
    "EXPECTED_SAFE_AUC": 0.9174298724699895,
    "BASELINE_ARM": "B0_base",
    "BASE_CHECKPOINT": "raptor_ft_coatnet_v4_full.pt",
    "SWA_CHECKPOINT": "raptor_ft_coatnet_v4_full_swa.pt",
    "E037_A1_DELTA": 0.0010437443710005212,
}
for name, value in expected.items():
    check(constants.get(name) == value, f"{name} == {value!r} (got {constants.get(name)!r})")

# The bar is E037's bar. Moving it after seeing E037 fall short would be moving
# the goalposts, so the smoke test pins it rather than trusting the constant.
check(constants["MIN_VALID_GAIN"] == 0.0020,
      "the gain floor is unchanged from the run that failed it")
check("EXPECTED_N_EVAL = EXPECTED_POOL - E037_N_EVAL" in source,
      "the split size is derived from the pool, not typed twice")

print("== the split is provably E037's untouched complement ==")
check("E037_EVAL_IDS_SHA256" in source, "E037's recorded split hash is pinned")
check("e037_split_reconstructed = e037_split_sha256 == E037_EVAL_IDS_SHA256" in source,
      "the reconstruction is compared against that hash")
check("if not e037_split_reconstructed:" in source and "raise RuntimeError" in source,
      "a failed reconstruction stops the run instead of guessing at the complement")
split_cell = next(cell.source for cell in code_cells if "e037_eval_ids" in cell.source)
# "eval_ids = [str(uid)" is also a substring of "e037_eval_ids = [str(uid)",
# so the complement line has to be matched in full.
complement_line = "eval_ids = [str(uid) for uid in shuffled[E037_N_EVAL:]]"
check(split_cell.index("e037_split_reconstructed =") < split_cell.index(complement_line),
      "the E037 draw is verified before the complement is taken")
check("eval_ids = [str(uid) for uid in shuffled[E037_N_EVAL:]]" in source,
      "the evaluation split is the tail E037 never drew")
check("assert not set(eval_ids) & set(e037_eval_ids)" in source,
      "disjointness is asserted, not assumed")
check("eval_disjoint_from_e037" in source, "disjointness is also a recorded mechanical gate")

print("== two checkpoints, loaded honestly ==")
check("load_model(" in source, "the base checkpoint uses load_model verbatim")
loader = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint_variant")
loader_source = ast.get_source_segment(source, loader)
check("strict=True" in loader_source, "the SWA state dict is loaded strictly")
check("normalisations.append" in loader_source, "every normalisation applied is recorded")
check(loader_source.count("normalisations.append") == 2,
      "exactly two normalisations are permitted")
check("module_prefix_stripped" in loader_source and "n_averaged_dropped" in loader_source,
      "those two are the module. prefix and the SWA counter")
check("swa_weights_differ_from_base" in source, "identical checkpoints are a mechanical failure")
check('raise RuntimeError("the SWA checkpoint holds the base weights' in source,
      "identical checkpoints also stop the run outright")
check("parameter_divergence" in source, "how far apart the checkpoints are is measured")

print("== stage A fails before stage B costs anything ==")


def cell_index(token):
    return next(index for index, cell in enumerate(code_cells) if token in cell.source)


gold_stage = cell_index('build_cache(gold_ids, "gold58")')
eval_stage = cell_index('build_cache(eval_ids, "complement2301")')
check(gold_stage < eval_stage, "Gold58 is cached and scored before the complement stage")
stage_a = code_cells[gold_stage].source
check(stage_a.index("load_checkpoint_variant(") < stage_a.index('build_cache(gold_ids, "gold58")'),
      "both checkpoints load before a single study is cached")
check(stage_a.index("parameter_divergence(") < stage_a.index('build_cache(gold_ids, "gold58")'),
      "a degenerate SWA file is caught before any cache work")
check("raise RuntimeError" in stage_a, "stage A fails closed")
check(stage_a.index("if not reproduced") < stage_a.index("windows_per_second ="),
      "the baseline reproduction check fires before anything else in stage A")
check("if not runtime_fits" in stage_a, "stage A projects the total runtime and refuses to overrun")
check(stage_a.index("projected_total_minutes") < stage_a.index("if not runtime_fits"),
      "the projection is computed before it is judged")
check("RUNTIME_BUDGET_MINUTES" in stage_a, "the projection is judged against the declared budget")
check("if not disk_fits" in stage_a, "stage A also refuses to start without room for the cache")
check("shutil.disk_usage(SCRATCH).free" in stage_a, "free space is measured on scratch itself")

print("== the baseline arm cannot drift from the proven path ==")
evaluate = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "evaluate_arms")
evaluate_source = ast.get_source_segment(source, evaluate)
check("order = list(eval_centers) + [c for c in candidates if c not in selected]"
      in evaluate_source,
      "the bank puts the 42 baseline centres first, so its chunks match predict_studies")
check('raw[f"{label}/42"][row] = head_logits(model, features[:K_EVAL])' in evaluate_source,
      "the 42-window arms read the first K_EVAL features of the bank")
check("len(set(eval_centers)) != K_EVAL" in evaluate_source,
      "repeated centres from a full pool would be a hard failure")
check("if len(candidates) >= K_EVAL:" in evaluate_source,
      "distinctness is only required when the pool can supply it")
check("short_pool_studies" in evaluate_source, "short-pool studies are counted, not assumed away")
check("range(0, len(tensor), CHUNK)" in source, "features are chunked by CHUNK, as in E028")
check("stack_windows(volume, centers)" in source,
      "the windows come from the untouched stack_windows")
check("torch.flip" not in source and "hflip" not in source,
      "no flip survives from E037 - it was measured at -0.0255")

print("== one bank per checkpoint, two window sets, no wasted pass ==")
check(evaluate_source.count("window_features(") == 1,
      "one backbone pass per checkpoint per study")
check(evaluate_source.count("for label, model in models.items()") == 1,
      "the cache is read once and shared by both checkpoints")
check("if list(models) != MODEL_LABELS" in evaluate_source,
      "model order is checked, so base/swa logits cannot be swapped")

print("== arm assembly ==")
namespace: dict[str, object] = {"np": np, "pd": pd, "torch": torch, "F": F}
for name in ["sigmoid", "assemble_arms", "prediction_divergence"]:
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"<{name}>", "exec"), namespace)

ARM_NAMES = ["B0_base", "B1_swa", "B2_ens_prob", "B3_ens_logit",
             "B4_base_all", "B5_ens_prob_all"]
TARGETS = [f"t{index}" for index in range(12)]
namespace["ARM_NAMES"] = ARM_NAMES
namespace["TARGETS"] = TARGETS
rng = np.random.default_rng(38)
raw_keys = ["base/42", "base/all", "swa/42", "swa/all"]
raw = {key: rng.normal(size=(9, 12)) * 2.0 for key in raw_keys}
arms = namespace["assemble_arms"](raw)
sigmoid = namespace["sigmoid"]
check(list(arms) == ARM_NAMES, "arms are produced in the declared order")
check(np.allclose(arms["B0_base"], sigmoid(raw["base/42"])), "B0 is the base 42-window logits")
check(np.allclose(arms["B1_swa"], sigmoid(raw["swa/42"])), "B1 is the SWA 42-window logits")
check(np.allclose(arms["B2_ens_prob"], (arms["B0_base"] + arms["B1_swa"]) / 2),
      "B2 averages the two checkpoints in probability space")
check(np.allclose(arms["B3_ens_logit"], sigmoid((raw["base/42"] + raw["swa/42"]) / 2)),
      "B3 averages the two checkpoints in logit space")
check(not np.allclose(arms["B2_ens_prob"], arms["B3_ens_logit"]),
      "logit-space averaging is a genuinely different operator")
check(np.allclose(arms["B4_base_all"], sigmoid(raw["base/all"])),
      "B4 is the base checkpoint over the whole bank - E037's A1")
check(np.allclose(arms["B5_ens_prob_all"],
                  (sigmoid(raw["base/all"]) + sigmoid(raw["swa/all"])) / 2),
      "B5 stacks the ensemble on top of the whole bank")
check(all(((arms[name] >= 0.0) & (arms[name] <= 1.0)).all() for name in ARM_NAMES),
      "every arm outputs probabilities")
degenerate = {key: np.zeros((9, 12)) for key in raw_keys}
same = namespace["assemble_arms"](degenerate)
check(all(np.allclose(same[name], 0.5) for name in ARM_NAMES),
      "identical logits collapse every arm onto the same predictions")
# If the SWA checkpoint were a copy of the base, every ensemble arm would be the
# baseline. The gate exists for that case; this proves the arithmetic agrees.
twin = {"base/42": raw["base/42"], "swa/42": raw["base/42"],
        "base/all": raw["base/all"], "swa/all": raw["base/all"]}
twin_arms = namespace["assemble_arms"](twin)
check(np.allclose(twin_arms["B2_ens_prob"], twin_arms["B0_base"])
      and np.allclose(twin_arms["B3_ens_logit"], twin_arms["B0_base"]),
      "identical members make the ensemble arms exactly the baseline")

print("== the divergence probe measures rank agreement ==")
probe_a = rng.normal(size=(200, 12))
probe_b = probe_a.copy()
identical = namespace["prediction_divergence"](probe_a, probe_b)
check(abs(identical["mean_spearman"] - 1.0) < 1e-12,
      "identical members report a Spearman of 1")
check(identical["mean_abs_prob_diff"] == 0.0, "identical members report no probability gap")
monotone = namespace["prediction_divergence"](probe_a, probe_a * 3.0 + 1.0)
check(abs(monotone["mean_spearman"] - 1.0) < 1e-12,
      "a monotone rescaling still ranks identically, which is what AUC sees")
check(monotone["mean_abs_prob_diff"] > 0.0,
      "that rescaling does move the probabilities, so the two statistics differ")
shuffled_probe = rng.normal(size=(200, 12))
check(abs(namespace["prediction_divergence"](probe_a, shuffled_probe)["mean_spearman"]) < 0.3,
      "unrelated members report a low Spearman")

print("== bootstrap reports what the gate needs ==")
bootstrap_node = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "paired_bootstrap")
bootstrap_source = ast.get_source_segment(source, bootstrap_node)
check("ci95_halfwidth" in bootstrap_source, "the interval half-width is reported")
check("BOOTSTRAP_SEED" in source, "one bootstrap seed is shared, so arms are paired to each other")
gate_cell = next(cell.source for cell in code_cells if "mechanical_gates = {" in cell.source)
check("estimator_resolves_threshold" in gate_cell,
      "each arm records whether its estimator can resolve the threshold")
check('"gain": macro_delta >= MIN_VALID_GAIN' in gate_cell, "the gain gate uses the declared floor")
check('"gold58_sign_agrees"' in gate_cell, "Gold58 confirms the sign")
check(gate_cell.index("if not mechanical_passed") < gate_cell.index("elif passing"),
      "mechanical failure short-circuits before any arm can pass")
check(gate_cell.index("elif passing") < gate_cell.index("elif not any_resolves"),
      "a real winner outranks the underpowered verdict")
for outcome in ["STOP_MECHANICAL_FAILED", "PASS_ENSEMBLE_PROMOTE",
                "STOP_ENSEMBLE_UNDERPOWERED", "STOP_ENSEMBLE_NO_GAIN"]:
    check(outcome in gate_cell, f"{outcome} is a declared outcome")
check('record["arm"] != BASELINE_ARM' in gate_cell, "the baseline cannot pass against itself")
check("null_arm_delta_zero" in gate_cell and "null_arm_never_wins" in gate_cell,
      "a null arm proves the estimator reports zero for zero")
check("base_all_arm_distinct" in gate_cell,
      "the one arm that must differ by construction is a mechanical gate")
check("arms_are_distinct" not in gate_cell,
      "SWA arms matching the baseline is not called broken wiring - it is the null result")
check("swa_arms_differ_from_baseline" in source,
      "arm separability is reported alongside the divergence statistics")

print("== the E037 replication is reported, never gated ==")
check("e037_a1_replication" in gate_cell, "the replication block is written to the summary")
check("e037_a1_replication" not in gate_cell.split("mechanical_gates = {")[1].split("}")[0],
      "the replication is not one of the mechanical gates")
check("intervals_overlap" in gate_cell and "e038_delta_inside_e037_ci" in gate_cell,
      "the replication reports interval agreement both ways")

print("== split determinism ==")
ids = [f"study{index:05d}" for index in range(4349)]


def split(seed):
    rng_ = np.random.default_rng(seed)
    shuffled = list(np.asarray(ids, dtype=object)[rng_.permutation(len(ids))])
    return [str(value) for value in shuffled[:2048]], [str(value) for value in shuffled[2048:]]


(first_head, first_tail), (second_head, second_tail) = split(3701), split(3701)
check(first_head == second_head and first_tail == second_tail,
      "both halves of the split are reproducible")
check(len(first_tail) == 2301 and len(set(first_tail)) == 2301,
      "the complement holds 2,301 distinct studies")
check(not set(first_head) & set(first_tail), "the two halves share no study")
check(sorted(first_head + first_tail) == sorted(ids), "the two halves cover the pool exactly")

print("== kernel metadata ==")
metadata = json.loads(METADATA.read_text())
check(metadata["is_private"] is True, "kernel is private")
check(metadata["enable_gpu"] is True, "GPU enabled")
check(metadata["enable_internet"] is False, "internet disabled")
check(metadata["machine_shape"] == "NvidiaTeslaT4", "T4 accelerator")
check(metadata["code_file"] == NOTEBOOK.name, "code_file matches the notebook")
check(metadata["id"].endswith("e038-swa-ensemble-scout"), "kernel id is E038's, not E037's")
check("rsna-knee-abnormality-detection" in metadata["competition_sources"],
      "competition attached")
check("dreaddevelopment/raptor-knee-widedense" in metadata["dataset_sources"],
      "the checkpoint dataset carrying both files is attached")
check(len(metadata["dataset_sources"]) == 6, "six dataset sources attached")

print()
if failures:
    print(f"SMOKE FAILED ({len(failures)} checks)")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("SMOKE PASSED")
