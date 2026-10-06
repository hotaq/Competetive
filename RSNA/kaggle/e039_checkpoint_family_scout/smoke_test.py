#!/usr/bin/env python3
"""Local static and behavioural checks for the E039 notebook before any Kaggle push."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e039-checkpoint-family-scout.ipynb"
METADATA = HERE / "kernel-metadata.json"
E038_SUMMARY = HERE.parent / "e038_swa_ensemble_scout/output_v1/e038_summary.json"
E038_PREDICTIONS = HERE.parent / "e038_swa_ensemble_scout/output_v1/e038_predictions.npz"

BANNED = [
    "kaggle competitions submit",
    "submission.csv",
    "torch.save",
    "safetensors",
]

# E039 trains nothing. Any of these appearing means the notebook stopped being
# an inference-only scout and its conclusions would no longer be about four
# frozen checkpoints seen through our own pipeline.
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
check("e039_summary.json" in source, "writes e039_summary.json")
check(not re.search(r"e038_\w*\.(json|csv|npz|log)", source),
      "no E038 artefact filename leaks into E039's outputs")
check(len(re.findall(r'"e039_\w+\.(?:json|csv|npz)"', source)) == 7,
      "all seven E039 artefacts are named with the E039 prefix")
check("shutil.rmtree(CACHE" in source, "removes the exact64 cache before exit")
check('SCRATCH = Path("/kaggle/temp")' in source,
      "the ~10 GiB cache goes to scratch, not to the collected working directory")
check('CACHE = SCRATCH / "e039_exact64_cache"' in source, "cache lives under scratch")
check("else WORK" not in source, "scratch never falls back to the working directory")
check("SCRATCH.mkdir(parents=True, exist_ok=True)" in source, "scratch is created, not probed")
check("would be collected as kernel output" in source,
      "a scratch path under the working directory raises instead of running")
check("tempfile.mkdtemp" in source, "an unwritable /kaggle/temp falls back off the working directory")

print("== inference only ==")
for token in BANNED_TRAINING:
    check(token not in source, f"no `{token}`")
check(source.count(".eval()") >= 2, "the models are put in eval mode")
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

CHECKPOINTS = [
    ("v4_base", "raptor_ft_coatnet_v4_full.pt", ("raptor-knee", "widedense")),
    ("v4_swa", "raptor_ft_coatnet_v4_full_swa.pt", ("raptor-knee", "widedense")),
    ("v5_maxspan", "raptor_ft_coatnet_v5_full_swa.pt", ("raptor-knee", "maxspan")),
    ("v9_finespacing", "raptor_ft_coatnet_v9_full.pt", ("raptor-knee", "finespacing")),
]
ARM_NAMES = [
    "C0_v4_base", "C1_v4_base_all", "C2_v4_swa", "C3_v4_swa_all",
    "C4_v5", "C5_v5_all", "C6_v9", "C7_v9_all",
    "C8_ens_v4swa_v5_all", "C9_ens_v4swa_v9_all", "C10_ens_four_all",
]
REPRODUCTION_ARMS = ["C1_v4_base_all", "C2_v4_swa", "C3_v4_swa_all"]

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
    "CHECKPOINTS": CHECKPOINTS,
    "WINDOW_SETS": ["42", "all"],
    "ARCH": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "MIN_VALID_GAIN": 0.0020,
    "MIN_BOOTSTRAP_WIN": 0.98,
    "MAX_TARGET_DROP": 0.010,
    "GOLD58_MIN_TARGET_DELTA": -0.010,
    "REPRODUCTION_TOLERANCE": 0.0005,
    "RUNTIME_BUDGET_MINUTES": 420.0,
    "EXPECTED_SAFE_AUC": 0.9174298724699895,
    "BASELINE_ARM": "C0_v4_base",
    "INCUMBENT_ARM": "C3_v4_swa_all",
    "BASE_LABEL": "v4_base",
    "ARM_NAMES": ARM_NAMES,
    "REPRODUCTION_ARMS": REPRODUCTION_ARMS,
    "E037_A1_DELTA": 0.0010437443710005212,
    "E038_B0_EVAL_MACRO": 0.9079593249126852,
}
for name, value in expected.items():
    check(constants.get(name) == value, f"{name} == {value!r} (got {constants.get(name)!r})")

# The bar is E037's and E038's bar. Moving it now, with a lever that finally
# looks live, is exactly the goalpost move E035 punished, so it is pinned here
# rather than trusted to the constant.
check(constants["MIN_VALID_GAIN"] == 0.0020,
      "the gain floor is unchanged from the two runs held to it")
check(constants["MIN_BOOTSTRAP_WIN"] == 0.98,
      "the win-rate floor is unchanged despite a wider family")
check("EXPECTED_N_EVAL = EXPECTED_POOL - E037_N_EVAL" in source,
      "the split size is derived from the pool, not typed twice")
check("MODEL_LABELS = [label for label, _name, _hints in CHECKPOINTS]" in source,
      "the model labels are derived from the checkpoint table, not typed twice")
check(constants["BASELINE_ARM"] == ARM_NAMES[0],
      "the baseline is the first declared arm")
check(constants["INCUMBENT_ARM"] in REPRODUCTION_ARMS,
      "the incumbent is one of the arms E038 already measured, not a new one")

print("== the pinned E038 numbers are E038's, not retyped ==")
expected_delta = constants.get("E038_EXPECTED_DELTA")
check(isinstance(expected_delta, dict) and set(expected_delta) == set(REPRODUCTION_ARMS),
      "every reproduction arm carries a pinned E038 delta")
if E038_SUMMARY.exists():
    e038 = json.loads(E038_SUMMARY.read_text())
    e038_arms = {record["arm"]: record for record in e038["arms"]}
    check(expected_delta["C1_v4_base_all"] == e038_arms["B4_base_all"]["eval_macro_delta"],
          "C1's pinned delta is E038's recorded B4_base_all")
    check(expected_delta["C2_v4_swa"] == e038_arms["B1_swa"]["eval_macro_delta"],
          "C2's pinned delta is E038's recorded B1_swa")
    check(constants["E038_B0_EVAL_MACRO"] == e038_arms["B0_base"]["eval_macro_auc"],
          "the pinned eval baseline is E038's recorded B0_base macro")
    check(constants["EXPECTED_SHA256"]["v4_base"] == e038["identity"]["base_checkpoint_sha256"],
          "v4_base's pinned sha256 is the one E038 ran")
    check(constants["EXPECTED_SHA256"]["v4_swa"] == e038["identity"]["swa_checkpoint_sha256"],
          "v4_swa's pinned sha256 is the one E038 ran")
    check(constants["E038_EVAL_IDS_SHA256"] == e038["identity"]["eval_ids_sha256"],
          "the pinned split hash is E038's recorded split hash")
else:  # pragma: no cover - only when E038's output is not checked out
    check(False, f"E038 summary present at {E038_SUMMARY}")

if E038_PREDICTIONS.exists():
    from sklearn.metrics import roc_auc_score

    stored = np.load(E038_PREDICTIONS, allow_pickle=True)
    y_e038 = stored["y_eval"]

    def macro(prediction):
        return float(np.mean([roc_auc_score(y_e038[:, index], prediction[:, index])
                              for index in range(y_e038.shape[1])]))

    b0 = stored["eval__B0_base"].astype(np.float64)
    b4 = stored["eval__B4_base_all"].astype(np.float64)
    b5 = stored["eval__B5_ens_prob_all"].astype(np.float64)
    # C3 is E038's post-hoc arm B6 = 2*B5 - B4, which is algebraically the SWA
    # checkpoint alone over the whole bank. E038 never stored it directly, so the
    # pinned figure has to be recomputed to be trusted.
    b6_delta = macro(2.0 * b5 - b4) - macro(b0)
    check(abs(expected_delta["C3_v4_swa_all"] - b6_delta) < 1e-9,
          f"C3's pinned delta reproduces E038's post-hoc B6 ({b6_delta:+.7f})")
    check(b6_delta > expected_delta["C2_v4_swa"] > expected_delta["C1_v4_base_all"],
          "the incumbent is genuinely the best arm E038 measured on this split")
else:  # pragma: no cover - only when E038's output is not checked out
    check(False, f"E038 predictions present at {E038_PREDICTIONS}")

print("== the split is E037's untouched complement and E038's exact split ==")
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
check(complement_line in source, "the evaluation split is the tail E037 never drew")
check("assert not set(eval_ids) & set(e037_eval_ids)" in source,
      "disjointness from E037 is asserted, not assumed")
check("eval_disjoint_from_e037" in source, "disjointness is also a recorded mechanical gate")
check("e038_split_matched = eval_ids_sha256 == E038_EVAL_IDS_SHA256" in source,
      "the split is checked against E038's hash, so the reproduction arms are on E038's studies")
check('"e038_split_matched": e038_split_matched' in source,
      "that match is a recorded mechanical gate, not just a print")

print("== four checkpoints, loaded honestly ==")
check("load_model(str(WEIGHTS[label]), ARCH, RES, DEVICE, ngpu=1)" in source,
      "the base checkpoint uses load_model verbatim")
check('if label == BASE_LABEL:' in source,
      "only the base checkpoint takes that path, so C0 cannot drift")
loader = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint_variant")
loader_source = ast.get_source_segment(source, loader)
check("strict=True" in loader_source, "the other state dicts are loaded strictly")
check("normalisations.append" in loader_source, "every normalisation applied is recorded")
check(loader_source.count("normalisations.append") == 2,
      "exactly two normalisations are permitted")
check("module_prefix_stripped" in loader_source and "n_averaged_dropped" in loader_source,
      "those two are the module. prefix and the SWA counter")
check("no_two_checkpoints_identical" in source,
      "two checkpoints holding the same weights is a mechanical failure")
check("raise RuntimeError(f\"these checkpoints hold identical weights" in source,
      "it also stops the run outright, before the complement is cached")
check("all_checkpoints_share_arch" in source and "all_checkpoints_share_res" in source,
      "a checkpoint with a different arch or resolution is a mechanical failure")
check("checkpoint_sha256_exact" in source,
      "every checkpoint is matched against the sha256 read locally")
check("divergence_matrix" in source, "every pair of checkpoints is compared, not just one pair")
check(len(set(constants["EXPECTED_SHA256"].values())) == 4,
      "the four pinned hashes are four distinct files")
check(set(constants["EXPECTED_SHA256"]) == {label for label, _n, _h in CHECKPOINTS},
      "every declared checkpoint carries a pinned hash")

print("== stage A fails before stage B costs anything ==")


def cell_index(token):
    return next(index for index, cell in enumerate(code_cells) if token in cell.source)


gold_stage = cell_index('build_cache(gold_ids, "gold58")')
eval_stage = cell_index('build_cache(eval_ids, "complement2301")')
check(gold_stage < eval_stage, "Gold58 is cached and scored before the complement stage")
stage_a = code_cells[gold_stage].source
check(stage_a.index("load_checkpoint_variant(") < stage_a.index('build_cache(gold_ids, "gold58")'),
      "all four checkpoints load before a single study is cached")
check(stage_a.index("divergence_matrix(") < stage_a.index('build_cache(gold_ids, "gold58")'),
      "a degenerate checkpoint is caught before any cache work")
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
      "the cache is read once and shared by all four checkpoints")
check("if list(models) != MODEL_LABELS" in evaluate_source,
      "model order is checked, so one checkpoint's logits cannot land under another's label")

print("== arm assembly ==")
namespace: dict[str, object] = {"np": np, "pd": pd, "torch": torch, "F": F}
for name in ["sigmoid", "assemble_arms", "prediction_divergence", "divergence_pairs"]:
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"<{name}>", "exec"), namespace)

MODEL_LABELS = [label for label, _name, _hints in CHECKPOINTS]
RAW_KEYS = [f"{label}/{window}" for label in MODEL_LABELS for window in ["42", "all"]]
TARGETS = [f"t{index}" for index in range(12)]
namespace["ARM_NAMES"] = ARM_NAMES
namespace["RAW_KEYS"] = RAW_KEYS
namespace["TARGETS"] = TARGETS
rng = np.random.default_rng(39)
raw = {key: rng.normal(size=(9, 12)) * 2.0 for key in RAW_KEYS}
arms = namespace["assemble_arms"](raw)
sigmoid = namespace["sigmoid"]
check(list(arms) == ARM_NAMES, "arms are produced in the declared order")
check(len(RAW_KEYS) == 8, "four checkpoints times two window budgets is eight logit sets")

solo = {
    "C0_v4_base": "v4_base/42",
    "C1_v4_base_all": "v4_base/all",
    "C2_v4_swa": "v4_swa/42",
    "C3_v4_swa_all": "v4_swa/all",
    "C4_v5": "v5_maxspan/42",
    "C5_v5_all": "v5_maxspan/all",
    "C6_v9": "v9_finespacing/42",
    "C7_v9_all": "v9_finespacing/all",
}
for arm, key in solo.items():
    check(np.allclose(arms[arm], sigmoid(raw[key])), f"{arm} is sigmoid({key})")
check(np.allclose(arms["C8_ens_v4swa_v5_all"],
                  (sigmoid(raw["v4_swa/all"]) + sigmoid(raw["v5_maxspan/all"])) / 2),
      "C8 averages v4 SWA and v5 in probability space")
check(np.allclose(arms["C9_ens_v4swa_v9_all"],
                  (sigmoid(raw["v4_swa/all"]) + sigmoid(raw["v9_finespacing/all"])) / 2),
      "C9 averages v4 SWA and v9 in probability space")
check(np.allclose(arms["C10_ens_four_all"],
                  sum(sigmoid(raw[f"{label}/all"]) for label in MODEL_LABELS) / 4),
      "C10 averages all four checkpoints over the whole bank")
check(all(((arms[name] >= 0.0) & (arms[name] <= 1.0)).all() for name in ARM_NAMES),
      "every arm outputs probabilities")
degenerate = {key: np.zeros((9, 12)) for key in RAW_KEYS}
same = namespace["assemble_arms"](degenerate)
check(all(np.allclose(same[name], 0.5) for name in ARM_NAMES),
      "identical logits collapse every arm onto the same predictions")
# If a checkpoint were a copy of another, its ensemble arms would collapse onto
# a solo arm. The gate exists for that case; this proves the arithmetic agrees.
twin = {key: raw[f"v4_swa/{key.split('/')[1]}"] for key in RAW_KEYS}
twin_arms = namespace["assemble_arms"](twin)
check(all(np.allclose(twin_arms[name], twin_arms["C3_v4_swa_all"])
          for name in ["C8_ens_v4swa_v5_all", "C9_ens_v4swa_v9_all", "C10_ens_four_all"]),
      "four identical members make every ensemble arm exactly the solo arm")

ARM_CHECKPOINTS = constants["ARM_CHECKPOINTS"]
check(set(ARM_CHECKPOINTS) == set(ARM_NAMES), "every arm declares the checkpoints it needs")
check(all(set(values) <= set(MODEL_LABELS) for values in ARM_CHECKPOINTS.values()),
      "no arm claims a checkpoint that is not attached")
check(ARM_CHECKPOINTS["C10_ens_four_all"] == MODEL_LABELS,
      "the four-way ensemble is recorded as costing four checkpoints at submission time")
check(all(len(ARM_CHECKPOINTS[arm]) == 1 for arm in solo),
      "every solo arm costs exactly one checkpoint")

promotable = [name for name in ARM_NAMES
              if name != "C0_v4_base" and name not in REPRODUCTION_ARMS]
check(len(promotable) == 7, f"seven arms compete for promotion (got {len(promotable)})")
check("C3_v4_swa_all" not in promotable,
      "the incumbent cannot promote itself - it is the reference, and already measured")
check(not set(promotable) & set(REPRODUCTION_ARMS),
      "no arm whose answer E038 already published is allowed to win")
check("PROMOTABLE_ARMS = [name for name in ARM_NAMES" in source
      and "if name != BASELINE_ARM and name not in REPRODUCTION_ARMS]" in source,
      "the promotable set is derived from that rule in the notebook too")

print("== the divergence probe measures rank agreement ==")
probe_a = rng.normal(size=(200, 12))
identical = namespace["prediction_divergence"](probe_a, probe_a.copy())
check(abs(identical["mean_spearman"] - 1.0) < 1e-12,
      "identical members report a Spearman of 1")
check(identical["mean_abs_prob_diff"] == 0.0, "identical members report no probability gap")
monotone = namespace["prediction_divergence"](probe_a, probe_a * 3.0 + 1.0)
check(abs(monotone["mean_spearman"] - 1.0) < 1e-12,
      "a monotone rescaling still ranks identically, which is what AUC sees")
check(monotone["mean_abs_prob_diff"] > 0.0,
      "that rescaling does move the probabilities, so the two statistics differ")
check(abs(namespace["prediction_divergence"](probe_a, rng.normal(size=(200, 12)))
          ["mean_spearman"]) < 0.3,
      "unrelated members report a low Spearman")
pair_probe = {name: rng.normal(size=(200, 12)) for name in ["a", "b", "c", "d"]}
pairs = namespace["divergence_pairs"](pair_probe, ["a", "b", "c", "d"])
check(len(pairs) == 6, "four checkpoints give six pairwise divergence readings")
check(all(not key.startswith("per_target_") for values in pairs.values() for key in values),
      "the pairwise summary drops the per-target detail that would bloat the JSON")

print("== bootstrap reports what the gates need ==")
bootstrap_node = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "paired_bootstrap")
bootstrap_source = ast.get_source_segment(source, bootstrap_node)
check("ci95_halfwidth" in bootstrap_source, "the interval half-width is reported")
check("BOOTSTRAP_SEED" in source, "one bootstrap seed is shared, so arms are paired to each other")
gate_cell = next(cell.source for cell in code_cells if "mechanical_gates = {" in cell.source)
check("estimator_resolves_threshold" in gate_cell,
      "each arm records whether its estimator can resolve the threshold")
check('"gain": macro_delta >= MIN_VALID_GAIN' in gate_cell, "the gain gate uses the declared floor")
check('"beats_incumbent": incumbent_delta > 0.0' in gate_cell,
      "an arm that loses to the arm we already have cannot promote")
check("paired_bootstrap(y_eval, incumbent_eval, eval_arms[name], BOOTSTRAP_SEED)" in gate_cell,
      "the margin against the incumbent is measured, not just differenced")
check("if promotable else None" in gate_cell,
      "that second bootstrap only runs for arms that can actually promote")
check("null_arm_delta_zero" in gate_cell and "null_arm_never_wins" in gate_cell,
      "a null arm proves the estimator reports zero for zero")
check("window_budget_arms_distinct" in gate_cell,
      "the arms that must differ by construction are a mechanical gate")
check("checkpoint_arms_distinct" not in gate_cell,
      "two checkpoints predicting alike is not called broken wiring - it would be the finding")
check("arms_distinct_from_baseline" in source,
      "arm separability is reported alongside the divergence statistics")

print("== two verdicts, both declared, neither tuned after the fact ==")
check('"gold58_sign_agrees"' in gate_cell and '"gold58_no_target_collapse"' in gate_cell,
      "both Gold58 gates are still computed")
check("gold58_gates = {" in gate_cell and "strict_gates = {**gates, **gold58_gates}" in gate_cell,
      "the strict verdict is E038's gate set, Gold58 veto included")
check('"passed_relaxed": passed_relaxed' in gate_cell and '"passed_strict": passed_strict'
      in gate_cell, "both verdicts are recorded per arm")
verdict_node = next(n for n in ast.parse(gate_cell).body
                    if isinstance(n, ast.FunctionDef) and n.name == "verdict")
verdict_source = ast.get_source_segment(gate_cell, verdict_node)
check("passing" in verdict_source and "STOP_MECHANICAL_FAILED" in verdict_source,
      "one verdict function serves both gate sets")
check(gate_cell.count("verdict(passing_strict)") == 1
      and gate_cell.count("verdict(passing_relaxed)") == 1,
      "it is applied once to each, so neither can be given its own rule")
check(verdict_source.index("if not mechanical_passed") < verdict_source.index("if passing"),
      "mechanical failure short-circuits before any arm can pass")
check(verdict_source.index("if passing") < verdict_source.index("if not any_resolves"),
      "a real winner outranks the underpowered verdict")
for outcome in ["STOP_MECHANICAL_FAILED", "PASS_PROMOTE_NEW_CHECKPOINT",
                "STOP_FAMILY_UNDERPOWERED", "STOP_INCUMBENT_HOLDS"]:
    check(outcome in verdict_source, f"{outcome} is a declared outcome")
check('"decision": decision' in gate_cell and '"decision_relaxed": decision_relaxed' in gate_cell,
      "both verdicts reach the summary")
check("verdicts_agree" in gate_cell, "their agreement is recorded rather than reconciled here")
check("decision, best_arm = verdict(passing_strict)" in gate_cell,
      "the headline decision is the strict one, so the bar cannot be accused of being lowered")
check("submission_candidate = best_arm or (INCUMBENT_ARM if mechanical_passed else None)"
      in gate_cell,
      "a STOP still names the incumbent, because it is a checkpoint we are not yet submitting")
check("record[\"promotable\"]" in gate_cell,
      "only promotable arms are considered for the verdict")

print("== the reproduction arms are checked, and gate the whole run ==")
check("e038_arms_reproduced" in gate_cell and "e038_baseline_reproduced" in gate_cell,
      "both reproduction checks are mechanical gates")
mechanical_block = gate_cell.split("mechanical_gates = {")[1].split("\n}")[0]
check("e038_arms_reproduced" in mechanical_block and "e038_baseline_reproduced" in mechanical_block,
      "they sit inside the mechanical gate dict, so a drift voids the run")
check("REPRODUCTION_TOLERANCE" in gate_cell, "the reproduction check uses the declared tolerance")
check('"reproduced": bool(abs(gap) <= REPRODUCTION_TOLERANCE)' in gate_cell,
      "reproduction is a two-sided tolerance on the gap, not a sign test")

print("== the E037 replication is reported, never gated ==")
check("e037_a1_replication" in gate_cell, "the replication block is written to the summary")
check("e037_a1_replication" not in mechanical_block,
      "the replication is not one of the mechanical gates")
check("intervals_overlap" in gate_cell and "e039_delta_inside_e037_ci" in gate_cell,
      "the replication reports interval agreement both ways")
check('"same_split_as_e038": True' in gate_cell,
      "the replication states plainly that this is E038's split, not a fresh one")

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
check(metadata["id"].endswith("e039-checkpoint-family-scout"), "kernel id is E039's, not E038's")
check("rsna-knee-abnormality-detection" in metadata["competition_sources"],
      "competition attached")
for dataset in ["dreaddevelopment/raptor-knee-widedense",
                "dreaddevelopment/raptor-knee-maxspan",
                "dreaddevelopment/raptor-knee-finespacing"]:
    check(dataset in metadata["dataset_sources"], f"{dataset} attached")
check(len(metadata["dataset_sources"]) == 8, "eight dataset sources attached")

print()
if failures:
    print(f"SMOKE FAILED ({len(failures)} checks)")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("SMOKE PASSED")
