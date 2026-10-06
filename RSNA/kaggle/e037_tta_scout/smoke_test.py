#!/usr/bin/env python3
"""Local static and behavioural checks for the E037 notebook before any Kaggle push."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e037-tta-scout.ipynb"
METADATA = HERE / "kernel-metadata.json"

BANNED = [
    "kaggle competitions submit",
    "submission.csv",
    "torch.save",
    "safetensors",
]

# E037 trains nothing. Any of these appearing means the notebook stopped being
# an inference-only scout and its conclusions would no longer be about TTA.
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
check("e037_summary.json" in source, "writes e037_summary.json")
check("shutil.rmtree(CACHE" in source, "removes the exact64 cache before exit")
check('SCRATCH = Path("/kaggle/temp")' in source,
      "the 9 GiB cache goes to scratch, not to the collected working directory")
check('CACHE = SCRATCH / "e037_exact64_cache"' in source, "cache lives under scratch")
# v1 probed `/kaggle/temp` and silently fell back to /kaggle/working when the
# image did not have it, which put 8.43 GiB into the collected output. Scratch
# is now created, and a fallback onto the working directory is refused.
check("else WORK" not in source, "scratch never falls back to the working directory")
check("SCRATCH.mkdir(parents=True, exist_ok=True)" in source, "scratch is created, not probed")
check("would be collected as kernel output" in source,
      "a scratch path under the working directory raises instead of running")
check("tempfile.mkdtemp" in source, "an unwritable /kaggle/temp falls back off the working directory")

print("== inference only ==")
for token in BANNED_TRAINING:
    check(token not in source, f"no `{token}`")
check(source.count("model.eval()") >= 1, "the model is put in eval mode")
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
    "BOOTSTRAP_SEED": 3702,
    "N_EVAL": 2048,
    "K_EVAL": 42,
    "RES": 384,
    "IMG": 336,
    "CROP_MM": 140.0,
    "ZOOM_CROP_MM": 130.0,
    "ZOOM_MARGIN": 12,
    "CHUNK": 6,
    "BOOTSTRAP_REPEATS": 2000,
    "TRANSFORMS": ["identity", "hflip", "zoom"],
    "JITTER_SHIFTS": [-0.5, 0.0, 0.5],
    "MIN_VALID_GAIN": 0.0020,
    "MIN_BOOTSTRAP_WIN": 0.98,
    "MAX_TARGET_DROP": 0.010,
    "GOLD58_MIN_TARGET_DELTA": -0.010,
    "REPRODUCTION_TOLERANCE": 0.0005,
    "RUNTIME_BUDGET_MINUTES": 420.0,
    "EXPECTED_SAFE_AUC": 0.9174298724699895,
    "BASELINE_ARM": "A0_baseline",
    "LATERALITY_TARGETS": ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"],
}
for name, value in expected.items():
    check(constants.get(name) == value, f"{name} == {value!r} (got {constants.get(name)!r})")

# 336 * 130 / 140 == 312, so a 12-pixel margin is exactly a 130 mm field of view.
check(constants["IMG"] - 2 * constants["ZOOM_MARGIN"]
      == round(constants["IMG"] * constants["ZOOM_CROP_MM"] / constants["CROP_MM"]),
      "the zoom margin is exactly the declared 130 mm crop")

print("== stage A fails before stage B costs anything ==")


def cell_index(token):
    return next(index for index, cell in enumerate(code_cells) if token in cell.source)


gold_stage = cell_index('build_cache(gold_ids, "gold58")')
eval_stage = cell_index('build_cache(eval_ids, "eval2048")')
check(gold_stage < eval_stage, "Gold58 is cached and scored before the 2,048-study stage")
stage_a = code_cells[gold_stage].source
check("raise RuntimeError" in stage_a, "stage A fails closed")
check(stage_a.index("if not reproduced") < stage_a.index("windows_per_second ="),
      "the baseline reproduction check fires before anything else in stage A")
check("if not runtime_fits" in stage_a, "stage A projects the total runtime and refuses to overrun")
check(stage_a.index("projected_total_minutes") < stage_a.index("if not runtime_fits"),
      "the projection is computed before it is judged")
check("RUNTIME_BUDGET_MINUTES" in stage_a, "the projection is judged against the declared budget")

print("== the baseline arm cannot drift from the proven path ==")
stack_tta = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "stack_windows_tta")
stack_tta_source = ast.get_source_segment(source, stack_tta)
check("return stack_windows(volume, centers)" in stack_tta_source,
      "the identity transform delegates to the untouched stack_windows")
evaluate = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "evaluate_arms")
evaluate_source = ast.get_source_segment(source, evaluate)
check("order = list(eval_centers) + [c for c in candidates if c not in selected]"
      in evaluate_source,
      "the identity bank puts the 42 baseline centres first, so its chunks match predict_studies")
check('"identity/base": identity_features[:K_EVAL]' in evaluate_source,
      "the baseline arm reads the first K_EVAL features of the bank")
check("len(set(eval_centers)) != K_EVAL" in evaluate_source,
      "duplicate baseline centres would be a hard failure, not a silent reordering")
check("range(0, len(tensor), CHUNK)" in source, "features are chunked by CHUNK, as in E028")

print("== transforms behave as declared ==")
namespace: dict[str, object] = {
    "np": np, "torch": torch, "F": F,
    "IMG": 336, "RES": 8, "K": 42, "ZOOM_MARGIN": 12, "ZOOM_CROP_MM": 130.0, "CROP_MM": 140.0,
    "JITTER_SHIFTS": [-0.5, 0.0, 0.5],
    "ARM_NAMES": constants["ARM_NAMES"] if "ARM_NAMES" in constants else None,
}
for name in ["stack_windows", "stack_windows_tta", "jittered_centers", "sigmoid",
             "assemble_arms", "resample_centers"]:
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"<{name}>", "exec"), namespace)

rng = np.random.default_rng(0)
volume = rng.integers(0, 256, size=(64, 336, 336), dtype=np.uint8)
centers = [10, 20, 30]
identity = namespace["stack_windows_tta"](volume, centers, "identity")
plain = namespace["stack_windows"](volume, centers)
check(torch.equal(identity, plain), "identity transform is byte-identical to stack_windows")
flipped = namespace["stack_windows_tta"](volume, centers, "hflip")
check(torch.equal(flipped, torch.flip(plain, dims=[-1])), "hflip mirrors the last axis")
check(torch.equal(torch.flip(flipped, dims=[-1]), plain), "hflip is its own inverse")
zoomed = namespace["stack_windows_tta"](volume, centers, "zoom")
check(tuple(zoomed.shape) == tuple(plain.shape), f"zoom keeps the window shape {tuple(zoomed.shape)}")
check(not torch.equal(zoomed, plain), "zoom actually changes the pixels")
expected_zoom = F.interpolate(
    torch.from_numpy(np.stack([volume[[c - 1, c, c + 1], 12:324, 12:324].astype(np.float32) / 255.0
                               for c in centers])),
    size=(8, 8), mode="bilinear", align_corners=False)
check(torch.equal(zoomed, expected_zoom), "zoom crops exactly 12 pixels from each side")


def raises(callable_):
    try:
        callable_()
    except RuntimeError:
        return True
    return False


check(raises(lambda: namespace["stack_windows_tta"](volume, centers, "rot90")),
      "an unknown transform is a hard failure")

print("== jitter is the baseline draw plus neighbours ==")
candidates = list(range(1, 55))
resample = namespace["resample_centers"]
jitter = namespace["jittered_centers"]
check(jitter(candidates, 42, 0.0) == resample(candidates, 42),
      "a zero shift reproduces resample_centers exactly")
check(jitter(candidates, 42, -0.5) != resample(candidates, 42), "a negative shift moves centres")
check(jitter(candidates, 42, 0.5) != resample(candidates, 42), "a positive shift moves centres")
for shift in [-0.5, 0.0, 0.5]:
    drawn = jitter(candidates, 42, shift)
    check(len(drawn) == 42 and set(drawn) <= set(candidates),
          f"shift {shift:+.1f} draws 42 centres from the candidate list")
check(jitter([], 42, 0.0) == [], "an empty candidate list yields no centres")
tight = list(range(1, 43))
edge = jitter(tight, 42, 0.5)
check(min(edge) >= tight[0] and max(edge) <= tight[-1],
      "a shift past the end clamps instead of wrapping")
check(edge == sorted(edge), "a shifted draw stays ordered")
check(jitter(candidates, 42, 5.0)[-1] == candidates[-1],
      "a large shift saturates at the last candidate")

print("== a short candidate pool is reproduced, not rejected ==")
# v1 assumed every study carries at least 42 safe candidates because E036's
# 512-study sample said so. The 2,048-study pool has studies with 36, and
# `resample_centers` serves those repeated windows - so does the production
# path, which is why the notebook now reproduces that instead of failing.
short = list(range(1, 37))
short_draw = resample(short, 42)
check(len(short_draw) == 42, "resample_centers still returns 42 centres from a 36-window pool")
check(len(set(short_draw)) < 42, "a short pool necessarily repeats centres")
check(set(short_draw) == set(short), "every candidate in a short pool is still drawn")
check(len(set(resample(candidates, 42))) == 42, "a full pool draws 42 distinct centres")
check("if len(candidates) >= K_EVAL:" in source,
      "distinctness is only required when the pool can supply it")
check("baseline draw is not {K_EVAL} windows" in source,
      "a draw of the wrong length is still a hard failure")
check("short_pool_studies" in source, "short-pool studies are counted, not assumed away")
check("every_study_has_k_eval_candidates" not in source,
      "the corpus-wide minimum-candidate gate is gone")
check("short_pool_studies_accounted" in source,
      "the run gates on the counted short pool matching the cache metrics")
short_jitter = jitter(short, 42, 0.5)
check(len(short_jitter) == 42 and set(short_jitter) <= set(short),
      "jitter on a short pool stays inside the candidate list")

print("== arm assembly ==")
ARM_NAMES = [
    "A0_baseline", "A1_windows_all", "A2_windows_jitter3", "A3_hflip", "A4_hflip_avg",
    "A5_zoom", "A6_zoom_avg", "A7_full_avg", "A8_full_avg_logit", "A9_all_plus_zoom_avg",
]
namespace["ARM_NAMES"] = ARM_NAMES
raw_keys = (["identity/base", "identity/all"]
            + [f"identity/jitter{index}" for index in range(3)]
            + ["hflip/base", "zoom/base"])
raw = {key: rng.normal(size=(7, 12)) * 2.0 for key in raw_keys}
arms = namespace["assemble_arms"](raw)
sigmoid = namespace["sigmoid"]
check(list(arms) == ARM_NAMES, "arms are produced in the declared order")
check(np.allclose(arms["A0_baseline"], sigmoid(raw["identity/base"])),
      "A0 is the identity/base logits alone")
check(np.allclose(arms["A4_hflip_avg"], (arms["A0_baseline"] + arms["A3_hflip"]) / 2),
      "A4 averages identity and flip in probability space")
check(np.allclose(arms["A7_full_avg"],
                  (arms["A0_baseline"] + arms["A3_hflip"] + arms["A5_zoom"]) / 3),
      "A7 averages the three views in probability space")
check(not np.allclose(arms["A7_full_avg"], arms["A8_full_avg_logit"]),
      "logit-space averaging is a genuinely different operator")
check(np.allclose(arms["A2_windows_jitter3"],
                  np.mean([sigmoid(raw[f"identity/jitter{i}"]) for i in range(3)], axis=0)),
      "A2 averages the three jittered draws")
check(all(((arms[name] >= 0.0) & (arms[name] <= 1.0)).all() for name in ARM_NAMES),
      "every arm outputs probabilities")
degenerate = {key: np.zeros((7, 12)) for key in raw_keys}
same = namespace["assemble_arms"](degenerate)
check(all(np.allclose(same[name], 0.5) for name in ARM_NAMES),
      "identical logits collapse every arm onto the same predictions")

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
for outcome in ["STOP_MECHANICAL_FAILED", "PASS_TTA_PROMOTE",
                "STOP_TTA_UNDERPOWERED", "STOP_TTA_NO_GAIN"]:
    check(outcome in gate_cell, f"{outcome} is a declared outcome")
check('record["arm"] != BASELINE_ARM' in gate_cell, "the baseline cannot pass against itself")
check("null_arm_delta_zero" in gate_cell and "null_arm_never_wins" in gate_cell,
      "a null arm proves the estimator reports zero for zero")

print("== split determinism ==")
ids = [f"study{index:05d}" for index in range(4349)]


def split(seed):
    rng_ = np.random.default_rng(seed)
    shuffled = list(np.asarray(ids, dtype=object)[rng_.permutation(len(ids))])
    return [str(value) for value in shuffled[:2048]]


first, second = split(3701), split(3701)
check(first == second, "the 2,048-study split is reproducible")
check(len(first) == 2048 and len(set(first)) == 2048, "the split holds 2,048 distinct studies")
check(first != split(3601)[:2048], "the E037 split is not the E036 split")

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
