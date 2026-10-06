# %% [markdown]
# # ARC Prize 2026 - Original L4x4 Candidate
#
# This notebook is written from scratch for ARC Prize 2026. It uses ideas from
# strong ARC 2025 solutions, but it does not import or copy 2025 solution code.
#
# Ideas borrowed, not code:
#
# - test-time fitting per task
# - geometric and color augmentation
# - candidate voting and rescoring
# - strict schema validation before writing `submission.json`
# - runtime-aware task batching for Kaggle
#
# This is meant to run on Kaggle with the official competition input attached.
# It is not meant to spend local compute.

# %%
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import json
import math
import os
import time

import numpy as np

try:
    import torch
except Exception:
    torch = None


START_TIME = time.time()
TIME_LIMIT_SECONDS = 11.6 * 60 * 60
MAX_TASK_SECONDS = 120
USE_PUBLIC_EVAL = False
WRITE_SUBMISSION = True

KAGGLE_DATA_DIR = Path("/kaggle/input/arc-prize-2026-arc-agi-2")
LOCAL_DATA_DIR = Path("../data/raw")
DATA_DIR = KAGGLE_DATA_DIR if KAGGLE_DATA_DIR.exists() else LOCAL_DATA_DIR

TEST_CHALLENGES_PATH = DATA_DIR / "arc-agi_test_challenges.json"
EVAL_CHALLENGES_PATH = DATA_DIR / "arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS_PATH = DATA_DIR / "arc-agi_evaluation_solutions.json"

print("DATA_DIR:", DATA_DIR)
print("KAGGLE_IS_COMPETITION_RERUN:", os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
if torch is not None:
    print("torch:", torch.__version__)
    print("cuda devices:", torch.cuda.device_count())


# %% [markdown]
# ## IO and Validation

# %%
Grid = list[list[int]]
Task = dict[str, list[dict[str, Grid]]]


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def grid_to_tuple(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in grid)


def tuple_to_grid(grid: tuple[tuple[int, ...], ...]) -> Grid:
    return [[int(v) for v in row] for row in grid]


def to_array(grid: Grid) -> np.ndarray:
    return np.array(grid, dtype=np.int16)


def to_grid(arr: np.ndarray) -> Grid:
    return np.asarray(arr, dtype=np.int16).astype(int).tolist()


def is_valid_grid(grid: Grid) -> bool:
    if not isinstance(grid, list) or not grid:
        return False
    if not all(isinstance(row, list) and row for row in grid):
        return False
    width = len(grid[0])
    if not all(len(row) == width for row in grid):
        return False
    if len(grid) > 30 or width > 30:
        return False
    return all(isinstance(v, int) and 0 <= v <= 9 for row in grid for v in row)


def validate_submission(challenges: dict[str, Task], submission: dict) -> list[str]:
    errors = []
    for task_id, task in challenges.items():
        if task_id not in submission:
            errors.append(f"{task_id}: missing task")
            continue
        if len(submission[task_id]) != len(task["test"]):
            errors.append(f"{task_id}: wrong number of test predictions")
            continue
        for idx, attempts in enumerate(submission[task_id]):
            if set(attempts) != {"attempt_1", "attempt_2"}:
                errors.append(f"{task_id}/{idx}: wrong attempt keys")
                continue
            for name in ("attempt_1", "attempt_2"):
                if not is_valid_grid(attempts[name]):
                    errors.append(f"{task_id}/{idx}: invalid {name}")
    extra = sorted(set(submission) - set(challenges))
    if extra:
        errors.append(f"extra tasks: {extra[:5]}")
    return errors


def score_submission(challenges: dict[str, Task], solutions: dict[str, list[Grid]], submission: dict) -> dict:
    errors = validate_submission(challenges, submission)
    if errors:
        return {"score": None, "hits": 0, "total": 0, "errors": errors[:20]}
    hits = 0
    total = 0
    missed = []
    for task_id, expected_outputs in solutions.items():
        task_hit = 0
        for idx, expected in enumerate(expected_outputs):
            attempts = submission[task_id][idx]
            expected_key = grid_to_tuple(expected)
            ok = (
                grid_to_tuple(attempts["attempt_1"]) == expected_key
                or grid_to_tuple(attempts["attempt_2"]) == expected_key
            )
            hits += int(ok)
            task_hit += int(ok)
            total += 1
        if task_hit < len(expected_outputs):
            missed.append(task_id)
    return {
        "score": hits / total if total else 0.0,
        "hits": hits,
        "total": total,
        "missed_head": missed[:20],
        "errors": [],
    }


# %% [markdown]
# ## Grid Augmentations
#
# We fit solvers under several geometric views, invert predictions back, and
# vote. This keeps the 2025 "augmentation scoring" idea without using their code.

# %%
def g_identity(a: np.ndarray) -> np.ndarray:
    return a


def g_rot90(a: np.ndarray) -> np.ndarray:
    return np.rot90(a, 1)


def g_rot180(a: np.ndarray) -> np.ndarray:
    return np.rot90(a, 2)


def g_rot270(a: np.ndarray) -> np.ndarray:
    return np.rot90(a, 3)


def g_flip_lr(a: np.ndarray) -> np.ndarray:
    return np.fliplr(a)


def g_flip_ud(a: np.ndarray) -> np.ndarray:
    return np.flipud(a)


def g_transpose(a: np.ndarray) -> np.ndarray:
    return a.T


def g_anti_transpose(a: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(a).T)


GEOMETRY: list[tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = [
    ("id", g_identity, g_identity),
    ("rot90", g_rot90, g_rot270),
    ("rot180", g_rot180, g_rot180),
    ("rot270", g_rot270, g_rot90),
    ("flip_lr", g_flip_lr, g_flip_lr),
    ("flip_ud", g_flip_ud, g_flip_ud),
    ("transpose", g_transpose, g_transpose),
    ("anti_transpose", g_anti_transpose, g_anti_transpose),
]


def remap_grid(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[int(mapping.get(int(v), int(v))) for v in row] for row in grid]


def invert_mapping(mapping: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in mapping.items()}


def color_permutations_for_task(task: Task) -> list[tuple[str, dict[int, int]]]:
    colors = sorted({int(v) for pair in task["train"] for grid in (pair["input"], pair["output"]) for row in grid for v in row})
    if not colors:
        return [("color_id", {i: i for i in range(10)})]
    identity = {i: i for i in range(10)}
    ordered = colors + [c for c in range(10) if c not in colors]
    compact = {c: i for i, c in enumerate(ordered)}
    reverse_used = {c: (len(colors) - 1 - i) if c in colors else c for i, c in enumerate(colors)}
    return [
        ("color_id", identity),
        ("color_compact", compact),
        ("color_reverse_used", reverse_used),
    ]


def augment_task(task: Task, geom_fn: Callable[[np.ndarray], np.ndarray], color_map: dict[int, int]) -> Task:
    out = {"train": [], "test": []}
    for pair in task["train"]:
        out["train"].append({
            "input": remap_grid(to_grid(geom_fn(to_array(pair["input"]))), color_map),
            "output": remap_grid(to_grid(geom_fn(to_array(pair["output"]))), color_map),
        })
    for case in task["test"]:
        out["test"].append({"input": remap_grid(to_grid(geom_fn(to_array(case["input"]))), color_map)})
    return out


def invert_prediction(grid: Grid, inv_geom_fn: Callable[[np.ndarray], np.ndarray], color_map: dict[int, int]) -> Grid:
    inv_color = invert_mapping(color_map)
    uncolored = remap_grid(grid, inv_color)
    return to_grid(inv_geom_fn(to_array(uncolored)))


# %% [markdown]
# ## Solver Fitting
#
# Each solver is fitted only from the demonstration pairs of the current task.
# This is the test-time adaptation lane.

# %%
@dataclass
class Candidate:
    grid: Grid
    score: float
    source: str


@dataclass
class Solver:
    name: str
    predict: Callable[[Grid], Grid]
    complexity: float = 1.0


def same_grid(a: Grid, b: Grid) -> bool:
    return grid_to_tuple(a) == grid_to_tuple(b)


def most_common_color(grid: Grid) -> int:
    return Counter(int(v) for row in grid for v in row).most_common(1)[0][0]


def solid_grid(height: int, width: int, color: int) -> Grid:
    return [[int(color) for _ in range(width)] for _ in range(height)]


def crop_non_background(grid: Grid) -> Grid:
    arr = to_array(grid)
    bg = most_common_color(grid)
    mask = arr != bg
    if not mask.any():
        return grid
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return to_grid(arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1])


def bbox_mask(arr: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    mask = arr != bg
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def pad_to_shape(grid: Grid, shape: tuple[int, int], color: int) -> Grid:
    arr = to_array(grid)
    h, w = shape
    out = np.full((h, w), int(color), dtype=np.int16)
    hh = min(h, arr.shape[0])
    ww = min(w, arr.shape[1])
    out[:hh, :ww] = arr[:hh, :ww]
    return to_grid(out)


def detect_color_map(train_pairs: list[dict[str, Grid]], geom_fn: Callable[[np.ndarray], np.ndarray] = g_identity) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for pair in train_pairs:
        src = to_grid(geom_fn(to_array(pair["input"])))
        dst = pair["output"]
        if np.shape(src) != np.shape(dst):
            return None
        for a, b in zip(np.ravel(src), np.ravel(dst)):
            a = int(a)
            b = int(b)
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping if mapping else None


def detect_repeat_scale(train_pairs: list[dict[str, Grid]]) -> tuple[int, int] | None:
    scale = None
    for pair in train_pairs:
        inp = to_array(pair["input"])
        out = to_array(pair["output"])
        if out.shape[0] % inp.shape[0] or out.shape[1] % inp.shape[1]:
            return None
        current = (out.shape[0] // inp.shape[0], out.shape[1] // inp.shape[1])
        if current[0] < 1 or current[1] < 1:
            return None
        if not np.array_equal(np.kron(inp, np.ones(current, dtype=np.int16)), out):
            return None
        if scale is not None and scale != current:
            return None
        scale = current
    return scale


def detect_tile_scale(train_pairs: list[dict[str, Grid]]) -> tuple[int, int] | None:
    scale = None
    for pair in train_pairs:
        inp = to_array(pair["input"])
        out = to_array(pair["output"])
        if out.shape[0] % inp.shape[0] or out.shape[1] % inp.shape[1]:
            return None
        current = (out.shape[0] // inp.shape[0], out.shape[1] // inp.shape[1])
        if current[0] < 1 or current[1] < 1:
            return None
        if not np.array_equal(np.tile(inp, current), out):
            return None
        if scale is not None and scale != current:
            return None
        scale = current
    return scale


def detect_shape_delta(train_pairs: list[dict[str, Grid]]) -> tuple[int, int] | None:
    delta = None
    for pair in train_pairs:
        inp = to_array(pair["input"])
        out = to_array(pair["output"])
        current = (out.shape[0] - inp.shape[0], out.shape[1] - inp.shape[1])
        if delta is not None and delta != current:
            return None
        delta = current
    return delta


def constant_output_solver(train_pairs: list[dict[str, Grid]]) -> Solver | None:
    first = train_pairs[0]["output"]
    if all(same_grid(pair["output"], first) for pair in train_pairs):
        return Solver("constant_output", lambda _grid, out=first: out, 0.1)
    return None


def build_exact_solvers(task: Task) -> list[Solver]:
    train_pairs = task["train"]
    solvers: list[Solver] = []

    const_solver = constant_output_solver(train_pairs)
    if const_solver is not None:
        solvers.append(const_solver)

    for name, geom_fn, _inv in GEOMETRY:
        if all(same_grid(to_grid(geom_fn(to_array(pair["input"]))), pair["output"]) for pair in train_pairs):
            solvers.append(Solver(f"geom:{name}", lambda grid, fn=geom_fn: to_grid(fn(to_array(grid))), 0.8))

        mapping = detect_color_map(train_pairs, geom_fn)
        if mapping is not None:
            solvers.append(Solver(
                f"geom_color:{name}",
                lambda grid, fn=geom_fn, mp=mapping: remap_grid(to_grid(fn(to_array(grid))), mp),
                1.2,
            ))

    if all(same_grid(crop_non_background(pair["input"]), pair["output"]) for pair in train_pairs):
        solvers.append(Solver("crop_non_background", crop_non_background, 1.0))

    repeat_scale = detect_repeat_scale(train_pairs)
    if repeat_scale is not None:
        solvers.append(Solver(
            f"repeat:{repeat_scale}",
            lambda grid, scale=repeat_scale: to_grid(np.kron(to_array(grid), np.ones(scale, dtype=np.int16))),
            1.4,
        ))

    tile_scale = detect_tile_scale(train_pairs)
    if tile_scale is not None:
        solvers.append(Solver(
            f"tile:{tile_scale}",
            lambda grid, scale=tile_scale: to_grid(np.tile(to_array(grid), scale)),
            1.4,
        ))

    delta = detect_shape_delta(train_pairs)
    if delta is not None:
        out_colors = [most_common_color(pair["output"]) for pair in train_pairs]
        color = Counter(out_colors).most_common(1)[0][0]
        def predict_delta(grid: Grid, delta=delta, color=color) -> Grid:
            arr = to_array(grid)
            shape = (max(1, min(30, arr.shape[0] + delta[0])), max(1, min(30, arr.shape[1] + delta[1])))
            return pad_to_shape(grid, shape, color)
        if all(same_grid(predict_delta(pair["input"]), pair["output"]) for pair in train_pairs):
            solvers.append(Solver(f"shape_delta:{delta}", predict_delta, 1.5))

    return solvers


def output_shape_priors(task: Task, test_input: Grid) -> list[tuple[int, int]]:
    priors = []
    out_shapes = [tuple(to_array(pair["output"]).shape) for pair in task["train"]]
    in_shapes = [tuple(to_array(pair["input"]).shape) for pair in task["train"]]
    test_shape = tuple(to_array(test_input).shape)

    if len(set(out_shapes)) == 1:
        priors.append(out_shapes[0])
    if all(inp == out for inp, out in zip(in_shapes, out_shapes)):
        priors.append(test_shape)
    delta = detect_shape_delta(task["train"])
    if delta is not None:
        priors.append((max(1, min(30, test_shape[0] + delta[0])), max(1, min(30, test_shape[1] + delta[1]))))
    scale = detect_repeat_scale(task["train"]) or detect_tile_scale(task["train"])
    if scale is not None:
        priors.append((max(1, min(30, test_shape[0] * scale[0])), max(1, min(30, test_shape[1] * scale[1]))))

    seen = set()
    out = []
    for shape in priors:
        if shape not in seen and 1 <= shape[0] <= 30 and 1 <= shape[1] <= 30:
            seen.add(shape)
            out.append(shape)
    return out


def fallback_candidates(task: Task, test_input: Grid) -> list[Candidate]:
    arr = to_array(test_input)
    bg = most_common_color(test_input)
    cands = [
        Candidate(test_input, 0.02, "fallback:identity"),
        Candidate(solid_grid(arr.shape[0], arr.shape[1], bg), 0.01, "fallback:solid_bg"),
    ]
    for shape in output_shape_priors(task, test_input):
        cands.append(Candidate(solid_grid(shape[0], shape[1], bg), 0.015, f"fallback:shape_bg:{shape}"))
        cands.append(Candidate(pad_to_shape(test_input, shape, bg), 0.018, f"fallback:pad:{shape}"))
    cropped = crop_non_background(test_input)
    if is_valid_grid(cropped):
        cands.append(Candidate(cropped, 0.025, "fallback:crop"))
    return cands


# %% [markdown]
# ## Relation Matrix Candidates
#
# These candidates learn a categorical relation of the form
# `output[r, c] <- color_map(input[f(r, c)])`. This catches ARC patterns such
# as tiled, mirrored, shifted, and scaled copies where ordinary correlation is
# too numeric and weak for categorical grid colors.

# %%
RelationMapper = Callable[[int, int, int, int, int, int], tuple[int, int]]


def relation_mappers() -> list[tuple[str, RelationMapper]]:
    def tile(r: int, c: int, ih: int, iw: int, _oh: int, _ow: int) -> tuple[int, int]:
        return r % ih, c % iw

    def scale(r: int, c: int, ih: int, iw: int, oh: int, ow: int) -> tuple[int, int]:
        return min(ih - 1, r * ih // oh), min(iw - 1, c * iw // ow)

    def row_block_shift_x(r: int, c: int, ih: int, iw: int, _oh: int, _ow: int) -> tuple[int, int]:
        return r % ih, (c + (r // ih)) % iw

    def row_block_shift_x_neg(r: int, c: int, ih: int, iw: int, _oh: int, _ow: int) -> tuple[int, int]:
        return r % ih, (c - (r // ih)) % iw

    def col_block_shift_y(r: int, c: int, ih: int, iw: int, _oh: int, _ow: int) -> tuple[int, int]:
        return (r + (c // iw)) % ih, c % iw

    def col_block_shift_y_neg(r: int, c: int, ih: int, iw: int, _oh: int, _ow: int) -> tuple[int, int]:
        return (r - (c // iw)) % ih, c % iw

    def checker_block_shift(r: int, c: int, ih: int, iw: int, _oh: int, _ow: int) -> tuple[int, int]:
        return (r + (c // iw)) % ih, (c + (r // ih)) % iw

    def mirror_tile_x(r: int, c: int, ih: int, iw: int, _oh: int, _ow: int) -> tuple[int, int]:
        sr, sc = r % ih, c % iw
        if (r // ih) % 2:
            sc = iw - 1 - sc
        return sr, sc

    def mirror_tile_y(r: int, c: int, ih: int, iw: int, _oh: int, _ow: int) -> tuple[int, int]:
        sr, sc = r % ih, c % iw
        if (c // iw) % 2:
            sr = ih - 1 - sr
        return sr, sc

    return [
        ("rel:tile", tile),
        ("rel:scale", scale),
        ("rel:row_block_shift_x", row_block_shift_x),
        ("rel:row_block_shift_x_neg", row_block_shift_x_neg),
        ("rel:col_block_shift_y", col_block_shift_y),
        ("rel:col_block_shift_y_neg", col_block_shift_y_neg),
        ("rel:checker_block_shift", checker_block_shift),
        ("rel:mirror_tile_x", mirror_tile_x),
        ("rel:mirror_tile_y", mirror_tile_y),
    ]


def in_bounds(row: int, col: int, height: int, width: int) -> bool:
    return 0 <= row < height and 0 <= col < width


def learn_relation_matrix(task: Task, mapper: RelationMapper) -> tuple[float, dict[int, int]] | None:
    matrix: dict[int, Counter] = defaultdict(Counter)
    total = 0

    for pair in task["train"]:
        inp = pair["input"]
        out = pair["output"]
        ih, iw = tuple(to_array(inp).shape)
        oh, ow = tuple(to_array(out).shape)
        if ih <= 0 or iw <= 0 or oh <= 0 or ow <= 0:
            return None
        for r in range(oh):
            for c in range(ow):
                sr, sc = mapper(r, c, ih, iw, oh, ow)
                if not in_bounds(sr, sc, ih, iw):
                    return None
                matrix[int(inp[sr][sc])][int(out[r][c])] += 1
                total += 1

    if total == 0 or not matrix:
        return None

    color_map: dict[int, int] = {}
    explained = 0
    for src_color, counts in matrix.items():
        dst_color, count = counts.most_common(1)[0]
        color_map[int(src_color)] = int(dst_color)
        explained += int(count)

    return explained / total, color_map


def apply_relation_matrix(
    test_input: Grid,
    out_shape: tuple[int, int],
    mapper: RelationMapper,
    color_map: dict[int, int],
) -> Grid:
    ih, iw = tuple(to_array(test_input).shape)
    oh, ow = out_shape
    out: Grid = []
    for r in range(oh):
        row = []
        for c in range(ow):
            sr, sc = mapper(r, c, ih, iw, oh, ow)
            if not in_bounds(sr, sc, ih, iw):
                return []
            value = int(test_input[sr][sc])
            row.append(int(color_map.get(value, value)))
        out.append(row)
    return out


def relation_matrix_candidates(task: Task, test_index: int, min_score: float = 0.995) -> list[Candidate]:
    test_input = task["test"][test_index]["input"]
    candidates: list[Candidate] = []

    for out_shape in output_shape_priors(task, test_input):
        for mapper_name, mapper in relation_mappers():
            learned = learn_relation_matrix(task, mapper)
            if learned is None:
                continue
            score, color_map = learned
            if score < min_score:
                continue
            grid = apply_relation_matrix(test_input, out_shape, mapper, color_map)
            if is_valid_grid(grid):
                candidates.append(Candidate(grid, score, f"{mapper_name}:shape={out_shape}:colors={color_map}"))

    return candidates


def relation_matrix_candidates_aug(task: Task, test_index: int, min_score: float = 0.995) -> list[Candidate]:
    bucket: dict[tuple[tuple[int, ...], ...], Candidate] = {}

    for geom_name, geom_fn, inv_geom_fn in GEOMETRY:
        aug_task = augment_task(task, geom_fn, {i: i for i in range(10)})
        for cand in relation_matrix_candidates(aug_task, test_index, min_score=min_score):
            try:
                restored = to_grid(inv_geom_fn(to_array(cand.grid)))
                add_candidate(bucket, Candidate(restored, cand.score, f"aug:{geom_name}:{cand.source}"))
            except Exception:
                pass

    return sorted(bucket.values(), key=lambda x: (-x.score, len(x.source), np.prod(np.shape(x.grid))))


# %% [markdown]
# ## Candidate Generation and Voting

# %%
def add_candidate(bucket: dict[tuple[tuple[int, ...], ...], Candidate], candidate: Candidate) -> None:
    if not is_valid_grid(candidate.grid):
        return
    key = grid_to_tuple(candidate.grid)
    if key in bucket:
        bucket[key].score += candidate.score
        bucket[key].source += f";{candidate.source}"
    else:
        bucket[key] = candidate


def generate_for_augmented_task(task: Task, test_index: int, prefix: str) -> list[Candidate]:
    solvers = build_exact_solvers(task)
    test_input = task["test"][test_index]["input"]
    candidates: list[Candidate] = []
    for solver in solvers:
        try:
            pred = solver.predict(test_input)
            score = max(0.05, 1.0 / (1.0 + solver.complexity))
            candidates.append(Candidate(pred, score, f"{prefix}:{solver.name}"))
        except Exception as exc:
            _ = exc
    candidates.extend(fallback_candidates(task, test_input))
    return candidates


def generate_candidates(task: Task, test_index: int) -> list[Candidate]:
    bucket: dict[tuple[tuple[int, ...], ...], Candidate] = {}

    # High-confidence symbolic relation candidates should outrank weak LLM or
    # fallback guesses when they exactly explain the train pairs.
    for idx, cand in enumerate(relation_matrix_candidates_aug(task, test_index, min_score=0.995)[:10]):
        boost = 4.0 if cand.score >= 0.999 else 2.0
        add_candidate(bucket, Candidate(cand.grid, boost * cand.score - 0.01 * idx, cand.source))

    for geom_name, geom_fn, inv_geom_fn in GEOMETRY:
        for color_name, color_map in color_permutations_for_task(task):
            aug = augment_task(task, geom_fn, color_map)
            prefix = f"{geom_name}/{color_name}"
            for cand in generate_for_augmented_task(aug, test_index, prefix):
                try:
                    restored = invert_prediction(cand.grid, inv_geom_fn, color_map)
                    add_candidate(bucket, Candidate(restored, cand.score, cand.source))
                except Exception:
                    pass

    # Shape-prior boost.
    shape_priors = set(output_shape_priors(task, task["test"][test_index]["input"]))
    for cand in bucket.values():
        shape = tuple(to_array(cand.grid).shape)
        if shape in shape_priors:
            cand.score += 0.05

    return sorted(bucket.values(), key=lambda x: (-x.score, len(x.source), np.prod(np.shape(x.grid))))


def attempts_for_case(task: Task, test_index: int) -> dict[str, Grid]:
    candidates = generate_candidates(task, test_index)
    if not candidates:
        candidates = fallback_candidates(task, task["test"][test_index]["input"])

    unique = []
    seen = set()
    for cand in candidates:
        key = grid_to_tuple(cand.grid)
        if key not in seen and is_valid_grid(cand.grid):
            seen.add(key)
            unique.append(cand)
        if len(unique) >= 2:
            break

    while len(unique) < 2:
        unique.append(Candidate([[len(unique)]], 0.0, "emergency"))

    return {"attempt_1": unique[0].grid, "attempt_2": unique[1].grid}


def solve_task(task_id: str, task: Task) -> tuple[str, list[dict[str, Grid]], dict]:
    task_start = time.time()
    outputs = []
    for test_index, _case in enumerate(task["test"]):
        outputs.append(attempts_for_case(task, test_index))
        if time.time() - task_start > MAX_TASK_SECONDS:
            print(f"Task {task_id} hit per-task soft limit after test index {test_index}")
            break
    while len(outputs) < len(task["test"]):
        inp = task["test"][len(outputs)]["input"]
        bg = most_common_color(inp)
        arr = to_array(inp)
        outputs.append({
            "attempt_1": inp,
            "attempt_2": solid_grid(arr.shape[0], arr.shape[1], bg),
        })
    meta = {"seconds": round(time.time() - task_start, 3), "n_test": len(task["test"])}
    return task_id, outputs, meta


def make_submission(challenges: dict[str, Task]) -> tuple[dict, dict]:
    submission = {}
    meta = {}
    total = len(challenges)
    for idx, (task_id, task) in enumerate(challenges.items(), start=1):
        if idx % 25 == 0:
            elapsed = time.time() - START_TIME
            print(f"{idx}/{total} tasks, elapsed={elapsed/60:.1f}m")
        if time.time() - START_TIME > TIME_LIMIT_SECONDS:
            print("Global time limit hit; filling remaining tasks with fallbacks")
            break
        tid, outputs, info = solve_task(task_id, task)
        submission[tid] = outputs
        meta[tid] = info

    for task_id, task in challenges.items():
        if task_id in submission:
            continue
        outputs = []
        for case in task["test"]:
            inp = case["input"]
            arr = to_array(inp)
            bg = most_common_color(inp)
            outputs.append({
                "attempt_1": inp,
                "attempt_2": solid_grid(arr.shape[0], arr.shape[1], bg),
            })
        submission[task_id] = outputs
        meta[task_id] = {"seconds": 0.0, "fallback": True}
    return submission, meta


# %% [markdown]
# ## Optional Public Eval
#
# Leave `USE_PUBLIC_EVAL=False` for competition rerun. Turn it on only inside
# the editor when you want to inspect local behavior on the public eval split.

# %%
if USE_PUBLIC_EVAL and EVAL_CHALLENGES_PATH.exists() and EVAL_SOLUTIONS_PATH.exists():
    eval_challenges = load_json(EVAL_CHALLENGES_PATH)
    eval_solutions = load_json(EVAL_SOLUTIONS_PATH)
    eval_submission, eval_meta = make_submission(eval_challenges)
    print(score_submission(eval_challenges, eval_solutions, eval_submission))
else:
    print("Public eval skipped")


# %% [markdown]
# ## Build Competition Submission

# %%
test_challenges = load_json(TEST_CHALLENGES_PATH)
submission, run_meta = make_submission(test_challenges)
errors = validate_submission(test_challenges, submission)
print("submission tasks:", len(submission))
print("validation errors:", len(errors))
if errors:
    print(errors[:20])
    raise ValueError("Invalid submission")

if WRITE_SUBMISSION:
    save_json(submission, "submission.json")
    save_json(run_meta, "run_meta.json")
    print("Wrote submission.json and run_meta.json")
