# ARC 2026 Omni-ARC hybrid one-cell Kaggle runner
#
# Paste/run this as a Kaggle notebook code cell.
# Inputs expected:
# - Competition: arc-prize-2026-arc-agi-2
# - Model path below points to the Qwen ARC model already attached to the notebook.
# Internet must be off for competition rerun.

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from IPython.display import Image as IPyImage
    from IPython.display import display
except Exception:
    IPyImage = None
    display = None


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
)
def _resolve_data_dir():
    for c in (
        Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2"),
        Path("/kaggle/input/arc-prize-2026-arc-agi-2"),
        Path("data/raw"),
    ):
        try:
            if (c / "arc-agi_test_challenges.json").exists():
                return c
        except Exception:
            pass
    return Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")


DATA_DIR = _resolve_data_dir()
WORK_DIR = Path("/kaggle/working")
EVAL_MODE = os.environ.get("ARC_EVAL_MODE", "0") == "1"
N_SHARDS = int(os.environ.get("ARC_N_SHARDS", "4"))
GPU_IDS = os.environ.get("ARC_GPU_IDS", "0,1,2,3").split(",")[:N_SHARDS]

CHALLENGE_FILE = DATA_DIR / (
    "arc-agi_evaluation_challenges.json" if EVAL_MODE else "arc-agi_test_challenges.json"
)
SOLUTION_FILE = DATA_DIR / "arc-agi_evaluation_solutions.json"
FINAL_SUBMISSION = WORK_DIR / ("eval_submission.json" if EVAL_MODE else "submission.json")

print("MODEL_PATH =", MODEL_PATH)
print("challenge =", CHALLENGE_FILE)
print("eval mode =", EVAL_MODE)
print("shards =", N_SHARDS, "gpu ids =", GPU_IDS)


WORKER_CODE = r'''
import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# CUDA visibility must be set before importing torch/transformers. If this is
# delayed, all shard processes can load onto physical GPU0 and OOM.
if "--gpu" in sys.argv:
    _gpu_arg_i = sys.argv.index("--gpu") + 1
    if _gpu_arg_i < len(sys.argv):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[_gpu_arg_i])

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


TTT_STEPS = int(os.environ.get("ARC_TTT_STEPS", "128"))
MAX_INPUT_TOKENS = int(os.environ.get("ARC_MAX_INPUT_TOKENS", "8192"))
TTT_MAX_INPUT_TOKENS = int(os.environ.get("ARC_TTT_MAX_INPUT_TOKENS", "8192"))
MAX_NEW_TOKENS = int(os.environ.get("ARC_MAX_NEW_TOKENS", "1200"))
MAX_AUGS = int(os.environ.get("ARC_MAX_AUGS", "16"))
LORA_R = int(os.environ.get("ARC_LORA_R", "256"))
LORA_ALPHA = int(os.environ.get("ARC_LORA_ALPHA", "32"))
USE_RSLORA = os.environ.get("ARC_USE_RSLORA", "1") == "1"
LR = float(os.environ.get("ARC_TTT_LR", "5e-5"))
DFS_P = float(os.environ.get("ARC_DFS_P", "0.2"))
DFS_MAX_SCORE = -math.log(max(1e-6, DFS_P))
USE_DFS_DECODE = os.environ.get("ARC_USE_DFS", "1") == "1"
PUZZLE_TIME_LIMIT = int(os.environ.get("ARC_PUZZLE_TIME", "1200"))
SEED = int(os.environ.get("ARC_SEED", "20260430"))
TTT_LOG_EVERY = int(os.environ.get("ARC_TTT_LOG_EVERY", "30"))
DECODE_LOG_EVERY = int(os.environ.get("ARC_DECODE_LOG_EVERY", "8"))
SHOW_VIZ = os.environ.get("ARC_SHOW_VIZ", "1") == "1"
VIZ_EVERY = int(os.environ.get("ARC_VIZ_EVERY", str(DECODE_LOG_EVERY)))
VIZ_MAX_PER_CASE = int(os.environ.get("ARC_VIZ_MAX_PER_CASE", "1"))
VIZ_DIR = Path(os.environ.get("ARC_VIZ_DIR", "/kaggle/working/viz_progress"))
DECODE_EARLY_VALID = int(os.environ.get("ARC_DECODE_EARLY_VALID", "8"))
DECODE_EARLY_VOTES = int(os.environ.get("ARC_DECODE_EARLY_VOTES", "4"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]


def shape(g):
    if not isinstance(g, list) or not g or not isinstance(g[0], list):
        return (0, 0)
    return (len(g), len(g[0]))


def is_grid(g):
    if not isinstance(g, list) or not g or not all(isinstance(r, list) for r in g):
        return False
    w = len(g[0])
    if w == 0:
        return False
    if len(g) > 30 or w > 30:
        return False
    for r in g:
        if len(r) != w:
            return False
        for v in r:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    return True


def normalize_grid(g):
    if not isinstance(g, list) or not g:
        return None
    out = []
    for r in g:
        if not isinstance(r, list):
            return None
        row = []
        for v in r:
            if isinstance(v, bool):
                return None
            if isinstance(v, (int, np.integer)):
                iv = int(v)
            elif isinstance(v, float) and abs(v - round(v)) < 1e-6:
                iv = int(round(v))
            elif isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
                iv = int(v)
            else:
                return None
            row.append(iv)
        out.append(row)
    return out if is_grid(out) else None


def _grid_to_rgb(g):
    import matplotlib.colors as mcolors

    arr = np.array(g, dtype=int)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=float)
    for i, color in enumerate(ARC_COLORS):
        rgb[arr == i] = mcolors.to_rgb(color)
    return rgb


def _plot_grid(ax, g, title):
    ax.axis("off")
    if not is_grid(g):
        ax.text(0.5, 0.5, "invalid grid", ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=9)
        return
    h, w = shape(g)
    ax.imshow(_grid_to_rgb(g), interpolation="nearest")
    ax.set_title(f"{title}\n{h}x{w}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#333333", linewidth=0.25)


def save_decode_viz(task, task_id, test_index, aug_i, candidate_grid, valid_count, invalid_count, first_invalid):
    if not SHOW_VIZ:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"viz import error {task_id}: {repr(exc)}", flush=True)
        return None

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    train_examples = task.get("train", [])[:2]
    cols = max(4, len(train_examples) * 2 + 2)
    fig, axes = plt.subplots(1, cols, figsize=(2.4 * cols, 2.8), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.axis("off")

    k = 0
    for j, ex in enumerate(train_examples):
        _plot_grid(axes[k], ex["input"], f"train {j} in")
        k += 1
        _plot_grid(axes[k], ex["output"], f"train {j} out")
        k += 1

    _plot_grid(axes[k], task["test"][test_index]["input"], "test input")
    k += 1
    if is_grid(candidate_grid):
        _plot_grid(axes[k], candidate_grid, "best valid so far")
    else:
        axes[k].axis("off")
        axes[k].text(0.02, 0.78, "no valid grid yet", fontsize=11, weight="bold")
        axes[k].text(0.02, 0.58, f"valid={valid_count} invalid={invalid_count}", fontsize=10)
        if first_invalid:
            axes[k].text(0.02, 0.34, f"raw: {first_invalid[:220]}", fontsize=7, wrap=True)
        axes[k].set_title("decode state", fontsize=9)

    fig.suptitle(
        f"{task_id} test={test_index} decode {aug_i}/{MAX_AUGS}",
        fontsize=12,
    )
    fig.tight_layout()
    path = VIZ_DIR / f"{task_id}_test{test_index}_aug{aug_i}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"VIZ_FILE {path}", flush=True)
    return str(path)


def grid_key(g):
    return tuple(tuple(r) for r in g)


def most_common_color(g):
    vals = [v for r in g for v in r]
    return Counter(vals).most_common(1)[0][0] if vals else 0


def solid_grid(h, w, c):
    h = max(1, min(30, int(h)))
    w = max(1, min(30, int(w)))
    c = int(c) if isinstance(c, int) else 0
    return [[c for _ in range(w)] for _ in range(h)]


def colors_in_task(task):
    vals = set()
    for ex in task.get("train", []):
        for g in (ex["input"], ex["output"]):
            vals.update(v for r in g for v in r)
    for ex in task.get("test", []):
        vals.update(v for r in ex["input"] for v in r)
    return sorted(vals)


def apply_color(g, cmap):
    return [[cmap.get(v, v) for v in r] for r in g]


def invert_color_map(cmap):
    return {v: k for k, v in cmap.items()}


def color_maps(task):
    cols = colors_in_task(task)
    maps = []
    maps.append({i: i for i in range(10)})
    if len(cols) >= 2:
        rev = {i: i for i in range(10)}
        for a, b in zip(cols, list(reversed(cols))):
            rev[a] = b
        maps.append(rev)
        for shift in range(1, len(cols)):
            cyc = {i: i for i in range(10)}
            for idx, a in enumerate(cols):
                cyc[a] = cols[(idx + shift) % len(cols)]
            maps.append(cyc)
        rng = random.Random(SEED + 1009 * len(cols))
        for _ in range(16):
            perm = cols[:]
            rng.shuffle(perm)
            if perm == cols:
                continue
            mp = {i: i for i in range(10)}
            for a, b in zip(cols, perm):
                mp[a] = b
            maps.append(mp)
    unique = []
    seen = set()
    for mp in maps:
        key = tuple(mp[i] for i in range(10))
        if key not in seen:
            seen.add(key)
            unique.append(mp)
    return unique


def geom(g, name):
    a = np.array(g, dtype=int)
    if name == "id":
        b = a
    elif name == "rot90":
        b = np.rot90(a, 1)
    elif name == "rot180":
        b = np.rot90(a, 2)
    elif name == "rot270":
        b = np.rot90(a, 3)
    elif name == "flip_h":
        b = np.fliplr(a)
    elif name == "flip_v":
        b = np.flipud(a)
    elif name == "transpose":
        b = a.T
    else:
        b = a
    return b.tolist()


INV_GEOM = {
    "id": "id",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "transpose": "transpose",
}


GEOMS = ["id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"]


# --- SOTA port (NVARC 2025 LB33.89 baseline, minimal perfpatch) ---
# Prompt format + DFS-12 + NLL rescore + kgmon selection.
# Model qwen3_4b_grids15_sft139 was SFT'd on compact digit grids with
# <|im_start|>user / assistant markers, NOT on verbose JSON prompts.
ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "\u010a": 10,
    "<|im_end|>": 15,
}
ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15

_ARC_TOKEN_ID_CACHE = {}


def _arc_token_ids(device):
    key = str(device)
    tids = _ARC_TOKEN_ID_CACHE.get(key)
    if tids is None:
        tids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = tids
    return tids


def convert_grid_to_string(grid):
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()


class QwenFormatter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fmt_query(self, query):
        grid_input = convert_grid_to_string(query[0]["input"])
        return "<|im_start|>user\n" + grid_input + "<|im_end|><|im_start|>assistant\n"

    def fmt_reply(self, reply):
        return convert_grid_to_string(reply[0]) + "<|im_end|>"

    def fmt_train(self, train, last_is_challenge=False):
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            grid_input = convert_grid_to_string(x["input"])
            grid_output = convert_grid_to_string(x["output"])
            text += f"<|im_start|>user\n{grid_input}<|im_end|><|im_start|>assistant\n{grid_output}<|im_end|>"
        if test is not None:
            test_in = {"input": test["input"] if "input" in test else test}
            text += self.fmt_query([test_in]) + self.fmt_reply([test["output"]])
        return text

    def max_new_tokens(self):
        max_sized_reply = [[0] * 30 for _ in range(30)]
        tokens = self.tokenizer.encode(self.fmt_reply([max_sized_reply]))
        return len(tokens) + 1

    def convert_tokens_to_array(self, tokens, limit_rows=30):
        if len(tokens) < 2:
            return None
        try:
            text = self.tokenizer.decode(tokens[:-1])
        except Exception:
            return None
        try:
            lines = text.strip().split("\n")
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in lines] if len(row)]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if isinstance(array, np.ndarray) and array.ndim == 2 and all(0 < x <= 30 for x in array.shape):
                return array
        except Exception:
            pass
        return None


def _model_device(model):
    try:
        return model.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time):
    n = logits.size(0)
    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()
    suffixes = defaultdict(list)
    candidates = dict()
    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))
    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x: x[0])
    dev = logits.device
    while time.time() - start_time < 540 and time.time() < end_time:
        batch_tokens = []
        batch_scores = []
        num_alive = 0
        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000.0)
            else:
                sc, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(sc)
                num_alive += 1
        if num_alive == 0:
            break
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=dev, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=dev),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens - 1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos + 1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )
        for bid, beams in next_suffixes.items():
            for sc, suf in beams:
                suf.insert(0, batch_tokens[bid])
                suffixes[bid].append((sc, suf))
    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens_list, max_new_tokens, max_score, end_time):
    dev = _model_device(model)
    max_len = max(len(t) for t in prefix_tokens_list)
    padded = [t + [PAD_ID] * (max_len - len(t)) for t in prefix_tokens_list]
    input_ids = torch.tensor(padded, device=dev, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for bid, beams in suffixes.items():
        result.append((bid, sorted(beams, key=lambda x: x[0])))
    return result


@torch.no_grad()
def calc_scores_nll(queries, answers, tokenizer, model):
    if not queries:
        return []
    dev = _model_device(model)
    bq, ba, bt, bl = [], [], [], []
    for q, a in zip(queries, answers):
        qt = tokenizer.encode(q)
        at = tokenizer.encode(a)
        bq.append(qt)
        ba.append(at)
        bt.append(qt + at)
        bl.append(len(qt + at))
    max_len = max(bl)
    padded = [t + [PAD_ID] * (max_len - len(t)) for t in bt]
    input_ids = torch.tensor(padded, device=dev, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    blogits = outputs.logits.float()
    bnorm = torch.logsumexp(blogits, dim=-1)
    out = []
    for rid, (qt, at) in enumerate(zip(bq, ba)):
        ql, al = len(qt), len(at)
        pos = torch.arange(ql - 1, ql - 1 + al, device=dev)
        tgt = torch.tensor(at, device=dev, dtype=torch.long)
        lp = blogits[rid, pos, tgt] - bnorm[rid, pos]
        out.append(float(-lp.sum().item()))
    return out


def _hashable_grid(g):
    if isinstance(g, np.ndarray):
        g = g.tolist()
    return tuple(tuple(int(v) for v in row) for row in g)


def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = float(np.mean([float(np.mean(g["score_aug"])) for g in guesses])) if guesses else 1e9
    return inf_score - aug_score


def getter_probmul3(guesses, baseline=3.0):
    inf_score = float(np.sum([baseline - g["beam_score"] for g in guesses]))
    aug_score = float(np.mean([float(np.sum([baseline - s for s in g["score_aug"]])) for g in guesses])) if guesses else 0.0
    return inf_score + aug_score


def score_groups_kgmon(grouped):
    scored = []
    for h, lst in grouped.items():
        scored.append((getter_kgmon(lst), lst[0]["solution"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [g for _, g in scored]


def transform_task(task, geom_name, cmap, shuffle_seed=None):
    train = [
        {
            "input": apply_color(geom(ex["input"], geom_name), cmap),
            "output": apply_color(geom(ex["output"], geom_name), cmap),
        }
        for ex in task["train"]
    ]
    if shuffle_seed is not None and len(train) > 1:
        rng = random.Random(shuffle_seed)
        rng.shuffle(train)
    return {
        "train": train,
        "test": [{"input": apply_color(geom(ex["input"], geom_name), cmap)} for ex in task["test"]],
    }


def reverse_pred(g, geom_name, cmap):
    inv_cmap = invert_color_map(cmap)
    return apply_color(geom(g, INV_GEOM[geom_name]), inv_cmap)


def grid_text(g):
    h, w = shape(g)
    lines = [f"shape: {h}x{w}"]
    for i, r in enumerate(g):
        lines.append(f"{i}: " + " ".join(str(x) for x in r))
    return "\n".join(lines)


def json_grid(g):
    return json.dumps(g, separators=(",", ":"))


def make_prompt(task, test_index=0, heldout=None, include_answer=False):
    train = list(task["train"])
    if heldout is not None:
        train = [ex for i, ex in enumerate(train) if i != heldout]
        test_input = task["train"][heldout]["input"]
        answer = task["train"][heldout]["output"]
    else:
        test_input = task["test"][test_index]["input"]
        answer = None

    parts = []
    parts.append("You solve ARC grid tasks.")
    parts.append("Colors are integers 0..9. Infer the transformation from examples.")
    parts.append("Return only one valid JSON grid as a list of lists of integers.")
    for i, ex in enumerate(train):
        parts.append(f"\nExample {i} input:\n```grid\n{grid_text(ex['input'])}\n```")
        parts.append(f"Example {i} output:\n```grid\n{grid_text(ex['output'])}\n```")
    parts.append(f"\nTest input:\n```grid\n{grid_text(test_input)}\n```")
    parts.append("Output JSON grid:")
    prompt = "\n".join(parts)
    if include_answer and answer is not None:
        prompt += "\n" + json_grid(answer)
    return prompt


def find_jsonish_arrays(text):
    spans = []
    starts = [i for i, ch in enumerate(text) if ch == "["]
    for s in starts[:80]:
        depth = 0
        for j in range(s, min(len(text), s + 20000)):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    spans.append(text[s : j + 1])
                    break
    return spans


def compact_digit_grid(text, shape_priors):
    digits = [int(x) for x in re.findall(r"[0-9]", text)]
    if not digits or not shape_priors:
        return None
    priors = sorted(set(shape_priors), key=lambda s: (abs(len(digits) - s[0] * s[1]), -(s[0] * s[1])))
    for h, w in priors:
        n = h * w
        if n <= 0 or n > len(digits):
            continue
        g = [digits[i * w : (i + 1) * w] for i in range(h)]
        if is_grid(g):
            return g
    return None


def parse_grid(text, shape_priors=None):
    text = text.strip()
    for cand in find_jsonish_arrays(text):
        for loader in (json.loads, ast.literal_eval):
            try:
                g = normalize_grid(loader(cand))
                if g is not None:
                    return g
            except Exception:
                pass
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"\b[0-9]\b", line)
        if nums:
            rows.append([int(x) for x in nums])
    if rows:
        w = len(rows[0])
        rows = [r for r in rows if len(r) == w]
        g = normalize_grid(rows)
        if g is not None:
            return g
    g = compact_digit_grid(text, shape_priors)
    if g is not None:
        return g
    return None


def output_shape_priors(task, test_input):
    priors = Counter()
    for ex in task["train"]:
        ih, iw = shape(ex["input"])
        oh, ow = shape(ex["output"])
        priors[(oh, ow)] += 2
        th, tw = shape(test_input)
        if ih and iw:
            if oh % ih == 0 and ow % iw == 0:
                priors[(min(30, th * (oh // ih)), min(30, tw * (ow // iw)))] += 3
            priors[(th + (oh - ih), tw + (ow - iw))] += 1
    return {k for k, v in priors.items() if k[0] > 0 and k[1] > 0 and k[0] <= 30 and k[1] <= 30}


def candidate_family(source):
    if source.startswith("rel:"):
        return "symbolic"
    if source.startswith("llm"):
        return "llm"
    if source.startswith("fallback"):
        return "fallback"
    return source.split(":", 1)[0]


def is_solid(g):
    vals = [v for r in g for v in r]
    return bool(vals) and len(set(vals)) == 1


def relation_mappers():
    def tile(inp, out_shape):
        ih, iw = shape(inp)
        oh, ow = out_shape
        if ih == 0 or iw == 0:
            return None
        return lambda r, c: (r % ih, c % iw)

    def row_block_shift_x(inp, out_shape):
        ih, iw = shape(inp)
        if ih == 0 or iw == 0:
            return None
        return lambda r, c: (r % ih, (c + (r // ih)) % iw)

    def col_block_shift_y(inp, out_shape):
        ih, iw = shape(inp)
        if ih == 0 or iw == 0:
            return None
        return lambda r, c: ((r + (c // iw)) % ih, c % iw)

    def mirror_tile_x(inp, out_shape):
        ih, iw = shape(inp)
        if ih == 0 or iw == 0:
            return None
        return lambda r, c: (r % ih, c % iw if (r // ih) % 2 == 0 else iw - 1 - (c % iw))

    def mirror_tile_y(inp, out_shape):
        ih, iw = shape(inp)
        if ih == 0 or iw == 0:
            return None
        return lambda r, c: (r % ih if (c // iw) % 2 == 0 else ih - 1 - (r % ih), c % iw)

    def upscale(inp, out_shape):
        ih, iw = shape(inp)
        oh, ow = out_shape
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return None
        sh, sw = oh // ih, ow // iw
        return lambda r, c: (r // sh, c // sw)

    def flip_h(inp, out_shape):
        ih, iw = shape(inp)
        if (ih, iw) != tuple(out_shape) or ih == 0:
            return None
        return lambda r, c: (r, iw - 1 - c)

    def flip_v(inp, out_shape):
        ih, iw = shape(inp)
        if (ih, iw) != tuple(out_shape) or ih == 0:
            return None
        return lambda r, c: (ih - 1 - r, c)

    def rot180(inp, out_shape):
        ih, iw = shape(inp)
        if (ih, iw) != tuple(out_shape) or ih == 0:
            return None
        return lambda r, c: (ih - 1 - r, iw - 1 - c)

    return [
        ("tile", tile),
        ("row_block_shift_x", row_block_shift_x),
        ("col_block_shift_y", col_block_shift_y),
        ("mirror_tile_x", mirror_tile_x),
        ("mirror_tile_y", mirror_tile_y),
        ("upscale", upscale),
        ("flip_h", flip_h),
        ("flip_v", flip_v),
        ("rot180", rot180),
    ]


def verify_relation(task, mapper_fn, cmap):
    # Exact gate: rule must reproduce EVERY train output cell-perfect.
    try:
        for ex in task["train"]:
            pred = apply_relation(ex["input"], shape(ex["output"]), mapper_fn, cmap)
            if pred is None or pred != ex["output"]:
                return False
    except Exception:
        return False
    return True


def learn_recolor_exact(task):
    # Single global color permutation explaining all same-shape pairs.
    cmap = {}
    for ex in task["train"]:
        if shape(ex["input"]) != shape(ex["output"]):
            return None
        for ri, row in enumerate(ex["input"]):
            for ci, v in enumerate(row):
                w = ex["output"][ri][ci]
                if v in cmap and cmap[v] != w:
                    return None
                cmap[v] = w
    return cmap


def is_identity_task(task):
    try:
        return bool(task.get("train")) and all(ex["input"] == ex["output"] for ex in task["train"])
    except Exception:
        return False


def _to_py(a):
    return [[int(v) for v in row] for row in np.asarray(a).tolist()]


def connected_components(g, bg=None, diag=False):
    a = np.asarray(g, dtype=int)
    h, w = a.shape
    if bg is None:
        bg = most_common_color(g)
    seen = np.zeros((h, w), dtype=bool)
    if diag:
        neigh = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    else:
        neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    objs = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(a[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            colors = Counter()
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                colors[int(a[cr, cc])] += 1
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(a[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            objs.append({"cells": cells, "bbox": (min(rs), max(rs) + 1, min(cs), max(cs) + 1), "size": len(cells), "colors": colors})
    return objs


def _learn_transform(task, proposer):
    try:
        fn = proposer(task)
    except Exception:
        return None
    if fn is None:
        return None
    try:
        for ex in task["train"]:
            if fn(ex["input"]) != ex["output"]:
                return None
    except Exception:
        return None
    return fn


def _verified_candidate(task, test_input, name, proposer):
    fn = _learn_transform(task, proposer)
    if fn is None:
        return None
    try:
        g = fn(test_input)
    except Exception:
        return None
    if not is_grid(g):
        return None
    return {"grid": g, "score": 5.0, "source": f"dsl:{name}", "verified": True}


def _prop_filter_extreme(extreme):
    def propose(task):
        def fn(g):
            bg = most_common_color(g)
            objs = connected_components(g, bg=bg)
            if not objs:
                return None
            sizes = [o["size"] for o in objs]
            ext = max(sizes) if extreme == "largest" else min(sizes)
            if sizes.count(ext) != 1:
                return None
            tgt = next(o for o in objs if o["size"] == ext)
            keep = set(tgt["cells"])
            h, w = shape(g)
            a = np.asarray(g, dtype=int)
            out = np.full((h, w), int(bg), dtype=int)
            for (r, c) in keep:
                out[r, c] = int(a[r, c])
            return _to_py(out)
        return fn
    return propose


def _prop_recolor_by_rank():
    def propose(task):
        table = {}
        for ex in task["train"]:
            if shape(ex["input"]) != shape(ex["output"]):
                return None
            bg_in = most_common_color(ex["input"])
            bg_out = most_common_color(ex["output"])
            if "bg" in table and table["bg"] != bg_out:
                return None
            table["bg"] = bg_out
            objs = connected_components(ex["input"], bg=bg_in)
            if not objs:
                return None
            order = sorted(range(len(objs)), key=lambda i: (-objs[i]["size"], i))
            a_out = np.asarray(ex["output"], dtype=int)
            for rank, i in enumerate(order):
                cols = {int(a_out[r, c]) for (r, c) in objs[i]["cells"]}
                if len(cols) != 1:
                    return None
                col = next(iter(cols))
                if rank in table and table[rank] != col:
                    return None
                table[rank] = col
        def fn(g):
            bg = most_common_color(g)
            objs = connected_components(g, bg=bg)
            if not objs:
                return None
            if "bg" not in table:
                return None
            order = sorted(range(len(objs)), key=lambda i: (-objs[i]["size"], i))
            h, w = shape(g)
            out = [[int(table["bg"])] * w for _ in range(h)]
            for rank, i in enumerate(order):
                if rank not in table:
                    return None
                for (r, c) in objs[i]["cells"]:
                    out[r][c] = int(table[rank])
            return out
        return fn
    return propose


def _gravity(g, bg, direction):
    a = np.asarray(g, dtype=int)
    h, w = a.shape
    out = np.full((h, w), int(bg), dtype=int)
    if direction in ("down", "up"):
        for c in range(w):
            col = [int(a[r, c]) for r in range(h) if int(a[r, c]) != bg]
            if direction == "down":
                for i, v in enumerate(col):
                    out[h - len(col) + i, c] = v
            else:
                for i, v in enumerate(col):
                    out[i, c] = v
    else:
        for r in range(h):
            row = [int(a[r, c]) for c in range(w) if int(a[r, c]) != bg]
            if direction == "right":
                for i, v in enumerate(row):
                    out[r, w - len(row) + i] = v
            else:
                for i, v in enumerate(row):
                    out[r, i] = v
    return _to_py(out)


def _prop_gravity(direction):
    def propose(task):
        def fn(g):
            return _gravity(g, most_common_color(g), direction)
        return fn
    return propose


def _prop_crop():
    def propose(task):
        def fn(g):
            bg = most_common_color(g)
            a = np.asarray(g, dtype=int)
            mask = a != bg
            if not mask.any():
                return None
            rs = np.where(mask.any(axis=1))[0]
            cs = np.where(mask.any(axis=0))[0]
            return _to_py(a[rs.min():rs.max() + 1, cs.min():cs.max() + 1])
        return fn
    return propose


def _prop_sym(axis, side):
    def propose(task):
        def fn(g):
            a = np.asarray(g, dtype=int)
            h, w = a.shape
            out = a.copy()
            if axis == "h":
                for r in range(h):
                    for c in range(w):
                        if side == "left" and c >= (w + 1) // 2:
                            out[r, c] = a[r, w - 1 - c]
                        if side == "right" and c < w // 2:
                            out[r, c] = a[r, w - 1 - c]
            else:
                for r in range(h):
                    for c in range(w):
                        if side == "top" and r >= (h + 1) // 2:
                            out[r, c] = a[h - 1 - r, c]
                        if side == "bottom" and r < h // 2:
                            out[r, c] = a[h - 1 - r, c]
            return _to_py(out)
        return fn
    return propose


def _prop_count(kind, diag):
    def feat(g):
        bg = most_common_color(g)
        if kind == "ncolors":
            return len({int(v) for row in g for v in row} - {int(bg)})
        objs = connected_components(g, bg=bg, diag=diag)
        if kind == "count":
            return len(objs)
        return max([o["size"] for o in objs] or [0])
    def propose(task):
        for ex in task["train"]:
            if shape(ex["output"]) != (1, 1):
                return None
            if ex["output"][0][0] != feat(ex["input"]):
                return None
        def fn(g):
            v = feat(g)
            if not isinstance(v, int) or v < 0 or v > 9:
                return None
            return [[int(v)]]
        return fn
    return propose


def learn_relation(task, mapper_fn):
    out_shapes = [shape(ex["output"]) for ex in task["train"]]
    if not out_shapes:
        return None
    # Prefer shape rule from the first pair; reject if ratios disagree badly.
    first_ih, first_iw = shape(task["train"][0]["input"])
    first_oh, first_ow = out_shapes[0]
    if first_ih == 0 or first_iw == 0:
        return None
    ratio = (
        first_oh / first_ih if first_ih else 1,
        first_ow / first_iw if first_iw else 1,
    )
    pairs = defaultdict(Counter)
    ok = 0
    total = 0
    for ex in task["train"]:
        inp, out = ex["input"], ex["output"]
        mp = mapper_fn(inp, shape(out))
        if mp is None:
            return None
        ih, iw = shape(inp)
        for r in range(shape(out)[0]):
            for c in range(shape(out)[1]):
                sr, sc = mp(r, c)
                if sr < 0 or sr >= ih or sc < 0 or sc >= iw:
                    return None
                pairs[inp[sr][sc]][out[r][c]] += 1
                total += 1
    cmap = {}
    for k, ctr in pairs.items():
        val, cnt = ctr.most_common(1)[0]
        cmap[k] = val
        ok += cnt
    score = ok / max(1, total)
    return score, cmap, ratio


def apply_relation(test_input, out_shape, mapper_fn, cmap):
    mp = mapper_fn(test_input, out_shape)
    if mp is None:
        return None
    ih, iw = shape(test_input)
    oh, ow = out_shape
    out = []
    for r in range(oh):
        row = []
        for c in range(ow):
            sr, sc = mp(r, c)
            if sr < 0 or sr >= ih or sc < 0 or sc >= iw:
                return None
            row.append(cmap.get(test_input[sr][sc], test_input[sr][sc]))
        out.append(row)
    return out if is_grid(out) else None


def symbolic_candidates(task, test_index):
    out = []
    test_input = task["test"][test_index]["input"]
    th, tw = shape(test_input)
    if is_identity_task(task):
        out.append({"grid": [row[:] for row in test_input], "score": 5.0, "source": "rel:identity_exact", "verified": True})
    rc = learn_recolor_exact(task)
    if rc is not None:
        try:
            g = apply_color(test_input, rc)
            if is_grid(g):
                out.append({"grid": g, "score": 5.0, "source": "rel:recolor_exact", "verified": True})
        except Exception:
            pass
    for name, mapper in relation_mappers():
        learned = learn_relation(task, mapper)
        if not learned:
            continue
        score, cmap, ratio = learned
        if score < 0.995:
            continue
        oh = int(round(th * ratio[0]))
        ow = int(round(tw * ratio[1]))
        if oh <= 0 or ow <= 0 or oh > 30 or ow > 30:
            continue
        g = apply_relation(test_input, (oh, ow), mapper, cmap)
        if g is not None:
            ver = bool(verify_relation(task, mapper, cmap))
            out.append({"grid": g, "score": 5.0 if ver else 4.0 * score, "source": f"rel:{name}", "verified": ver})
    dsl_rules = [
        ("filter_largest", _prop_filter_extreme("largest")),
        ("filter_smallest", _prop_filter_extreme("smallest")),
        ("recolor_byrank", _prop_recolor_by_rank()),
        ("gravity_down", _prop_gravity("down")),
        ("gravity_up", _prop_gravity("up")),
        ("gravity_left", _prop_gravity("left")),
        ("gravity_right", _prop_gravity("right")),
        ("crop_bbox", _prop_crop()),
        ("sym_h_left", _prop_sym("h", "left")),
        ("sym_h_right", _prop_sym("h", "right")),
        ("sym_v_top", _prop_sym("v", "top")),
        ("sym_v_bottom", _prop_sym("v", "bottom")),
        ("count4", _prop_count("count", False)),
        ("count8", _prop_count("count", True)),
        ("maxsize", _prop_count("maxsize", False)),
        ("ncolors", _prop_count("ncolors", False)),
    ]
    for dsl_name, proposer in dsl_rules:
        try:
            cand = _verified_candidate(task, test_input, dsl_name, proposer)
        except Exception:
            cand = None
        if cand is not None:
            out.append(cand)
    return out


def fallback_candidates(task, test_index):
    inp = task["test"][test_index]["input"]
    h, w = shape(inp)
    bg = most_common_color(inp)
    out = [
        {"grid": inp, "score": 0.05, "source": "fallback:identity"},
        {"grid": solid_grid(h, w, bg), "score": 0.03, "source": "fallback:solid_input_bg"},
    ]
    for sh in output_shape_priors(task, inp):
        out.append({"grid": solid_grid(sh[0], sh[1], bg), "score": 0.025, "source": "fallback:solid_shape_prior"})
    return out


def cast_trainable_to_bf16(model):
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)


def reset_trainable(model, state):
    for name, p in model.named_parameters():
        if p.requires_grad and name in state:
            p.data.copy_(state[name].to(device=p.device, dtype=p.dtype))


def _neuter_broken_torchao_check():
    # Kaggle image ships torchao 0.10.0 while peft requires >0.16.0.
    # peft's dispatcher chain ALWAYS calls is_torchao_available(), which
    # RAISES (not False) on the old version -> every get_peft_model crashes.
    # SOTA is immune (Unsloth path). Neuter the check in both namespaces.
    try:
        import peft.import_utils as _piu
        try:
            _piu.is_torchao_available()
        except ImportError:
            _piu.is_torchao_available = lambda: False
            try:
                import peft.tuners.lora.torchao as _pt
                _pt.is_torchao_available = lambda: False
            except Exception:
                pass
            print("neutered broken torchao availability check", flush=True)
    except Exception as exc:
        print(f"torchao neuter skipped: {repr(exc)}", flush=True)


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    lora_kwargs = dict(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    )
    try:
        lora_kwargs["use_rslora"] = USE_RSLORA
        lora = LoraConfig(**lora_kwargs)
    except TypeError:
        lora_kwargs.pop("use_rslora", None)
        lora = LoraConfig(**lora_kwargs)
    try:
        import peft as _peftmod
        print(f"peft={getattr(_peftmod, '__version__', '?')} torch={torch.__version__} rslora_cfg={lora_kwargs.get('use_rslora', 'n/a')}", flush=True)
    except Exception:
        pass
    _neuter_broken_torchao_check()
    model = get_peft_model(model, lora)
    cast_trainable_to_bf16(model)
    init_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    return model, tokenizer, init_state


def encode_for_loss(tokenizer, prompt, answer, max_length=None):
    max_length = int(max_length or MAX_INPUT_TOKENS)
    if "<|im_start|>" in prompt:
        full = prompt + answer
    else:
        full = prompt + "\n" + answer
    enc_full = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc_full["input_ids"].cuda()
    attn = enc_full["attention_mask"].cuda()
    labels = input_ids.clone()
    try:
        ids = input_ids[0].tolist()
        user_pos = [i for i, v in enumerate(ids) if v == USER_TOKEN_ID]
        asst_pos = [i for i, v in enumerate(ids) if v == ASSISTANT_TOKEN_ID]
        starts = sorted(user_pos + asst_pos)
        ends = [i for i, v in enumerate(ids) if v == EOS_ID]
        if len(starts) >= 2 and len(ends) >= 1:
            labels[:, :] = -100
            pairs = list(zip(starts, ends[:len(starts)]))
            for j, (s, e) in enumerate(pairs):
                if s >= e:
                    continue
                if j % 2 == 1:
                    a = min(s + 2, labels.shape[1])
                    b = min(e + 1, labels.shape[1])
                    if a < b:
                        labels[:, a:b] = input_ids[:, a:b]
            if (labels != -100).any():
                return input_ids, attn, labels
    except Exception:
        pass
    enc_prompt = tokenizer(prompt + "\n", return_tensors="pt", truncation=True, max_length=max_length)
    cut = min(enc_prompt["input_ids"].shape[1], labels.shape[1] - 1)
    labels[:, :cut] = -100
    return input_ids, attn, labels


def random_aug(task):
    gname = random.choice(GEOMS)
    cmap = random.choice(color_maps(task))
    return transform_task(task, gname, cmap)


def ttt_train(model, tokenizer, task, task_id):
    if TTT_STEPS <= 0 or len(task["train"]) < 2:
        return 0
    model.config.use_cache = False
    grad_ckpt = os.environ.get("ARC_GRAD_CKPT", "0") == "1"
    if grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
            print("grad-checkpointing ON", flush=True)
        except Exception as exc:
            print(f"grad-ckpt unsupported: {repr(exc)}", flush=True)
            grad_ckpt = False
    model.train()
    formatter = QwenFormatter(tokenizer)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    warmup = max(1, TTT_STEPS // 10)
    did = 0
    t0 = time.time()
    print(f"TTT start {task_id}: {TTT_STEPS} steps lr={LR} r={LORA_R} alpha={LORA_ALPHA} rslora={USE_RSLORA}", flush=True)
    for step in range(TTT_STEPS):
        if step < warmup:
            ls = float(step + 1) / float(warmup)
        else:
            prog = float(step - warmup) / float(max(1, TTT_STEPS - warmup))
            ls = 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in optim.param_groups:
            pg["lr"] = LR * ls
        input_ids = attn = labels = loss = None
        try:
            aug_task = random_aug(task)
            heldout = random.randrange(len(aug_task["train"]))
            train_pairs = [ex for i, ex in enumerate(aug_task["train"]) if i != heldout]
            held_in = aug_task["train"][heldout]["input"]
            held_out = aug_task["train"][heldout]["output"]
            prompt = formatter.fmt_train(train_pairs) + formatter.fmt_query([{"input": held_in}])
            answer = formatter.fmt_reply([held_out])
            max_len = min(MAX_INPUT_TOKENS, TTT_MAX_INPUT_TOKENS)
            while len(train_pairs) > 1:
                try:
                    est = len(tokenizer.encode(prompt + answer))
                except Exception:
                    break
                if est <= max_len:
                    break
                train_pairs = train_pairs[1:]
                prompt = formatter.fmt_train(train_pairs) + formatter.fmt_query([{"input": held_in}])
            input_ids, attn, labels = encode_for_loss(
                tokenizer,
                prompt,
                answer,
                max_length=max_len,
            )
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                did += 1
            if step == 0 or (step + 1) % TTT_LOG_EVERY == 0 or step + 1 == TTT_STEPS:
                print(
                    f"TTT {task_id} {step+1}/{TTT_STEPS} loss={float(loss.detach().cpu()):.4f} tok={input_ids.numel()} time={time.time()-t0:.1f}s",
                    flush=True,
                )
        except torch.cuda.OutOfMemoryError as exc:
            print(f"TTT OOM {task_id}: {exc}", flush=True)
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
        finally:
            del input_ids, attn, labels, loss
    if grad_ckpt:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    model.eval()
    model.config.use_cache = True
    torch.cuda.empty_cache()
    return did


@torch.no_grad()
def generate_text(model, tokenizer, prompt, temp=0.0, max_new_tokens=None):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    enc = {k: v.cuda() for k, v in enc.items()}
    do_sample = temp > 0
    kwargs = dict(
        max_new_tokens=int(max_new_tokens or MAX_NEW_TOKENS),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        use_cache=True,
    )
    if do_sample:
        kwargs.update(dict(temperature=temp, top_p=0.95))
    gen = model.generate(**enc, **kwargs)
    new_ids = gen[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def aug_specs(task):
    specs = []
    cmaps = color_maps(task)
    s = 0
    while len(specs) < MAX_AUGS:
        added = False
        for gname in GEOMS:
            for mi, cmap in enumerate(cmaps):
                if len(specs) >= MAX_AUGS:
                    break
                shuffle_seed = SEED + 100000 * s + 997 * mi + 17 * len(specs)
                specs.append((gname, cmap, shuffle_seed, 0.0))
                added = True
            if len(specs) >= MAX_AUGS:
                break
        if not added:
            break
        s += 1
    return specs[:MAX_AUGS]


def max_new_tokens_for_case(task, test_index):
    test_input = task["test"][test_index]["input"]
    priors = output_shape_priors(task, test_input)
    if priors:
        max_cells = max(h * w for h, w in priors)
    else:
        h, w = shape(test_input)
        max_cells = h * w
    # The ARC SFT model often emits compact digit streams without EOS. If the
    # cap is too loose it keeps writing valid-looking digits until max_new,
    # making every augmentation painfully slow. Keep enough room for compact
    # grids and short JSON-ish answers, but do not let small grids ramble.
    estimate = 64 + int(max_cells * 1.35)
    return max(64, min(MAX_NEW_TOKENS, estimate))


def _parse_compact_text(raw):
    try:
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        rows = []
        for ln in lines:
            digs = [int(ch) for ch in ln if ch.isdigit()]
            if digs:
                rows.append(digs)
        if not rows:
            return None
        w = len(rows[0])
        rows = [r for r in rows if len(r) == w]
        g = normalize_grid(rows)
        return g
    except Exception:
        return None


def _llm_dfs_candidates(model, tokenizer, formatter, task, test_index, task_id, token_cap, t0):
    specs = aug_specs(task)
    puzzle_end = t0 + PUZZLE_TIME_LIMIT
    all_beams = []
    invalid = 0
    for aug_i, (gname, cmap, shuffle_seed, temp) in enumerate(specs, 1):
        if time.time() > puzzle_end:
            print(f"dfs timeout {task_id} test={test_index} aug={aug_i}/{len(specs)}", flush=True)
            break
        try:
            aug_task = transform_task(task, gname, cmap, shuffle_seed=shuffle_seed)
            prompt_text = formatter.fmt_train(aug_task["train"]) + formatter.fmt_query([aug_task["test"][test_index]])
            prefix = tokenizer.encode(prompt_text)
            budget = MAX_INPUT_TOKENS - token_cap
            if len(prefix) > budget:
                prefix = prefix[-budget:]
            with torch.inference_mode():
                dfs_res = inference_turbo_dfs(model, [prefix], token_cap, DFS_MAX_SCORE, puzzle_end)
            if not dfs_res:
                invalid += 1
                continue
            _, beams = dfs_res[0]
            if not beams:
                invalid += 1
                continue
            hit = 0
            for beam_score, toks in beams:
                arr = formatter.convert_tokens_to_array(toks)
                if arr is None:
                    continue
                try:
                    restored = reverse_pred(arr.tolist(), gname, cmap)
                except Exception:
                    continue
                if not is_grid(restored):
                    continue
                all_beams.append({"grid": restored, "beam_score": float(beam_score), "source": f"llm-dfs:{gname}", "gname": gname})
                hit += 1
            if aug_i % DECODE_LOG_EVERY == 0:
                print(f"dfs progress {task_id} test={test_index} {aug_i}/{len(specs)} beams={len(all_beams)} invalid={invalid}", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"dfs OOM {task_id}: {exc}", flush=True)
            break
        except Exception as exc:
            print(f"dfs error {task_id} aug={gname}: {repr(exc)}", flush=True)
            invalid += 1
    if not all_beams:
        return []
    grouped = {}
    for b in all_beams:
        grouped.setdefault(grid_key(b["grid"]), []).append(b)
    items = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), min(x["beam_score"] for x in kv[1])))[:20]
    rescored = []
    for h, lst in items:
        G = lst[0]["grid"]
        best_beam = min(x["beam_score"] for x in lst)
        qs, ans = [], []
        for j in range(8):
            gj = random.choice(GEOMS)
            cj = random.choice(color_maps(task))
            atrain = [{"input": apply_color(geom(ex["input"], gj), cj), "output": apply_color(geom(ex["output"], gj), cj)} for ex in task["train"]]
            rng = random.Random(SEED + 7919 * j + (hash(task_id) % 100000))
            rng.shuffle(atrain)
            atest_in = apply_color(geom(task["test"][test_index]["input"], gj), cj)
            asol = apply_color(geom(G, gj), cj)
            q = formatter.fmt_train(atrain) + formatter.fmt_query([{"input": atest_in}])
            a = formatter.fmt_reply([asol])
            try:
                if len(tokenizer.encode(q + a)) > MAX_INPUT_TOKENS:
                    continue
            except Exception:
                pass
            qs.append(q)
            ans.append(a)
        try:
            if qs:
                with torch.inference_mode():
                    aug_nlls = calc_scores_nll(qs, ans, tokenizer, model)
            else:
                aug_nlls = [5.0]
        except Exception as exc:
            print(f"calc_scores failed {task_id}: {repr(exc)}", flush=True)
            aug_nlls = [5.0] * (len(qs) if qs else 1)
        total = len(lst) - float(np.mean(aug_nlls))
        rescored.append((total, h, lst[0]["grid"], len(lst), float(np.mean(aug_nlls)), best_beam))
    rescored.sort(key=lambda x: (-x[0], x[5]))
    out = []
    for rank_i, (total, h, grid, cnt, aug_mean, best_beam) in enumerate(rescored):
        norm = 2.0 - 0.1 * rank_i + max(0.0, (DFS_MAX_SCORE - best_beam)) * 0.1
        out.append({"grid": grid, "score": float(norm), "source": f"llm-dfs:kgmon:r{rank_i}:n{cnt}", "beam_score": best_beam, "score_aug": aug_mean, "kgmon": total})
    print(f"dfs summary {task_id} test={test_index} uniq={len(rescored)} beams={len(all_beams)} invalid={invalid}", flush=True)
    return out


def _llm_greedy_candidates(model, tokenizer, formatter, task, test_index, task_id, token_cap, t0):
    out = []
    invalid = 0
    first_invalid = None
    viz_count = 0
    def maybe_viz(aug_i):
        nonlocal viz_count
        if not SHOW_VIZ or viz_count >= VIZ_MAX_PER_CASE:
            return
        if aug_i % VIZ_EVERY != 0 and aug_i != MAX_AUGS:
            return
        try:
            cg = out[-1]["grid"] if out else None
            save_decode_viz(task, task_id, test_index, aug_i, cg, len(out), invalid, first_invalid)
            viz_count += 1
        except Exception as exc:
            print(f"viz error {task_id} test={test_index} aug={aug_i}: {repr(exc)}", flush=True)
    for aug_i, (gname, cmap, shuffle_seed, temp) in enumerate(aug_specs(task), 1):
        if time.time() > t0 + PUZZLE_TIME_LIMIT:
            print(f"greedy timeout {task_id} test={test_index}", flush=True)
            break
        try:
            aug_task = transform_task(task, gname, cmap, shuffle_seed=shuffle_seed)
            prompt = formatter.fmt_train(aug_task["train"]) + formatter.fmt_query([aug_task["test"][test_index]])
            if len(tokenizer.encode(prompt)) > MAX_INPUT_TOKENS - token_cap:
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS - token_cap)
                prompt = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)
            raw = generate_text(model, tokenizer, prompt, temp=temp, max_new_tokens=token_cap)
            g = None
            try:
                toks = tokenizer.encode(raw) + [EOS_ID]
                arr = formatter.convert_tokens_to_array(toks)
                if arr is not None:
                    g = arr.tolist()
            except Exception:
                g = None
            if g is None:
                g = _parse_compact_text(raw)
            if g is None:
                aug_test_input = aug_task["test"][test_index]["input"]
                g = parse_grid(raw, output_shape_priors(aug_task, aug_test_input))
            if g is None:
                invalid += 1
                if first_invalid is None:
                    first_invalid = " ".join(raw.replace("\n", " ").split())[:180]
                if aug_i % DECODE_LOG_EVERY == 0:
                    print(f"decode progress {task_id} test={test_index} {aug_i}/{MAX_AUGS} valid={len(out)} invalid={invalid}", flush=True)
                    maybe_viz(aug_i)
                continue
            restored = reverse_pred(g, gname, cmap)
            if is_grid(restored):
                out.append({"grid": restored, "score": 1.0, "source": f"llm:{gname}:temp{temp}"})
            if aug_i % DECODE_LOG_EVERY == 0:
                print(f"decode progress {task_id} test={test_index} {aug_i}/{MAX_AUGS} valid={len(out)} invalid={invalid}", flush=True)
                maybe_viz(aug_i)
            if len(out) >= DECODE_EARLY_VALID:
                top_votes = Counter(grid_key(c["grid"]) for c in out).most_common(1)[0][1]
                if top_votes >= DECODE_EARLY_VOTES:
                    print(f"decode early stop {task_id} test={test_index} aug={aug_i}/{MAX_AUGS} valid={len(out)} top_votes={top_votes}", flush=True)
                    break
        except torch.cuda.OutOfMemoryError as exc:
            print(f"decode OOM {task_id}: {exc}", flush=True)
            break
        except Exception as exc:
            print(f"decode error {task_id} aug={gname}: {repr(exc)}", flush=True)
    if invalid and first_invalid is not None:
        print(f"decode summary {task_id} test={test_index} valid={len(out)} invalid={invalid} first_invalid={first_invalid!r}", flush=True)
    else:
        print(f"decode summary {task_id} test={test_index} valid={len(out)} invalid={invalid}", flush=True)
    return out


def llm_candidates(model, tokenizer, task, test_index, task_id):
    t0 = time.time()
    formatter = QwenFormatter(tokenizer)
    try:
        fmt_max = formatter.max_new_tokens()
    except Exception:
        fmt_max = MAX_NEW_TOKENS
    token_cap = max_new_tokens_for_case(task, test_index)
    token_cap = max(64, min(token_cap, fmt_max, MAX_NEW_TOKENS))
    print(f"decode start {task_id} test={test_index} augs={MAX_AUGS} max_new={token_cap} dfs={USE_DFS_DECODE}", flush=True)
    if USE_DFS_DECODE:
        try:
            dfs_out = _llm_dfs_candidates(model, tokenizer, formatter, task, test_index, task_id, token_cap, t0)
            if dfs_out:
                return dfs_out
            print(f"dfs empty {task_id} test={test_index}, fallback to greedy", flush=True)
        except Exception as exc:
            print(f"dfs failed {task_id} test={test_index}: {repr(exc)}, fallback to greedy", flush=True)
    return _llm_greedy_candidates(model, tokenizer, formatter, task, test_index, task_id, token_cap, t0)


def rank_candidates(task, test_index, candidates):
    inp = task["test"][test_index]["input"]
    priors = output_shape_priors(task, inp)
    train_colors = set(colors_in_task(task))
    bucket = {}
    meta = {}
    for c in candidates:
        g = normalize_grid(c.get("grid"))
        if g is None:
            continue
        k = grid_key(g)
        if k not in bucket:
            bucket[k] = g
            meta[k] = {"score": 0.0, "sources": [], "families": set(), "votes": 0}
        score = float(c.get("score", 0.0))
        src = c.get("source", "unknown")
        fam = candidate_family(src)
        score += 0.20 if shape(g) in priors else 0.0
        score += 0.10 if all(v in train_colors for r in g for v in r) else -0.10
        if is_solid(g):
            score -= 0.35
        if g == inp:
            score -= 0.20
        if fam == "symbolic":
            score += 0.30
        if fam == "llm":
            score += 0.05
        meta[k]["score"] += score
        meta[k]["sources"].append(src)
        meta[k]["families"].add(fam)
        meta[k]["votes"] += 1
    ranked = []
    for k, g in bucket.items():
        m = meta[k]
        ranked.append((m["score"] + 0.05 * math.log1p(m["votes"]), g, m))
    ranked.sort(key=lambda x: (-x[0], shape(x[1]), len(str(x[1]))))
    return ranked


def choose_two(task, test_index, candidates):
    # VERIFIED-GATE (Layer 1): an exact train-verified rule preempts attempt_1.
    # Neural stays as attempt_2 -> monotonic, can only add score.
    ver_seen = set()
    ver_list = []
    for c in candidates:
        if not c.get("verified"):
            continue
        g = normalize_grid(c.get("grid"))
        if g is None:
            continue
        k = grid_key(g)
        if k in ver_seen:
            continue
        ver_seen.add(k)
        ver_list.append((float(c.get("score", 5.0)), g, str(c.get("source", "rel:verified"))))
    if ver_list:
        ver_list.sort(key=lambda x: (-x[0], shape(x[1])))
        first_grid = ver_list[0][1]
        first_key = grid_key(first_grid)
        rest = []
        for c in candidates:
            g = normalize_grid(c.get("grid"))
            if g is None or grid_key(g) == first_key:
                continue
            rest.append(c)
        second_grid = None
        if rest:
            llm_rest = [c for c in rest if candidate_family(str(c.get("source", ""))) == "llm"]
            ranked_rest = rank_candidates(task, test_index, llm_rest or rest)
            if ranked_rest:
                second_grid = ranked_rest[0][1]
        if second_grid is None:
            fb = fallback_candidates(task, test_index)
            second_grid = fb[0]["grid"] if grid_key(fb[0]["grid"]) != first_key else fb[1]["grid"]
        print(f"select-verified test={test_index} a1={[ver_list[0][2]]} n_verified={len(ver_list)}", flush=True)
        return {"attempt_1": first_grid, "attempt_2": second_grid}
    ranked = rank_candidates(task, test_index, candidates)
    if not ranked:
        ranked = rank_candidates(task, test_index, fallback_candidates(task, test_index))
    first = ranked[0]
    first_key = grid_key(first[1])
    first_fams = first[2]["families"]
    second = None
    for cand in ranked[1:]:
        if grid_key(cand[1]) == first_key:
            continue
        if cand[2]["families"].isdisjoint(first_fams):
            second = cand
            break
    if second is None:
        for cand in ranked[1:]:
            if grid_key(cand[1]) != first_key:
                second = cand
                break
    if second is None:
        fb = fallback_candidates(task, test_index)
        second_grid = fb[0]["grid"] if grid_key(fb[0]["grid"]) != first_key else fb[1]["grid"]
        second = (0.0, second_grid, {"sources": ["fallback:emergency"], "families": {"fallback"}})
    print(
        f"select test={test_index} a1={first[2]['sources'][:2]} a2={second[2]['sources'][:2]}",
        flush=True,
    )
    return {"attempt_1": first[1], "attempt_2": second[1]}


def solve_task(model, tokenizer, init_state, task_id, task):
    t0 = time.time()
    reset_trainable(model, init_state)
    did = ttt_train(model, tokenizer, task, task_id)
    rows = []
    for test_index in range(len(task["test"])):
        cands = []
        cands.extend(symbolic_candidates(task, test_index))
        cands.extend(llm_candidates(model, tokenizer, task, test_index, task_id))
        cands.extend(fallback_candidates(task, test_index))
        rows.append(choose_two(task, test_index, cands))
    elapsed = time.time() - t0
    valid = sum(1 for row in rows for g in row.values() if is_grid(g))
    print(f"done {task_id} time={elapsed:.1f}s ttt={did} valid={valid}", flush=True)
    return rows, {"time": elapsed, "ttt": did, "valid": valid}


def fallback_task(task):
    rows = []
    for i in range(len(task["test"])):
        fb = fallback_candidates(task, i)
        rows.append({"attempt_1": fb[0]["grid"], "attempt_2": fb[1]["grid"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--challenge", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    print(f"[shard {args.shard}] gpu={args.gpu} loading data", flush=True)

    tasks_all = json.loads(Path(args.challenge).read_text())
    items = [(k, v) for idx, (k, v) in enumerate(tasks_all.items()) if idx % args.nshards == args.shard]
    try:
        max_tasks = int(os.environ.get("ARC_MAX_TASKS", "0"))
    except Exception:
        max_tasks = 0
    if max_tasks and max_tasks > 0:
        items = items[:max_tasks]
    out_path = Path(args.out)
    prog_path = out_path.with_suffix(".progress.json")

    solved = {}
    if out_path.exists():
        try:
            solved = json.loads(out_path.read_text())
        except Exception:
            solved = {}

    print(f"[shard {args.shard}] tasks={len(items)} model={args.model}", flush=True)
    model, tokenizer, init_state = load_model(args.model)
    print(f"[shard {args.shard}] PEFT ready", flush=True)

    metas = {}
    for n, (task_id, task) in enumerate(items, 1):
        if task_id in solved:
            continue
        try:
            print(f"[shard {args.shard}] start {task_id} ({n}/{len(items)})", flush=True)
            rows, meta = solve_task(model, tokenizer, init_state, task_id, task)
        except Exception as exc:
            print(f"[shard {args.shard}] ERROR {task_id}: {repr(exc)}", flush=True)
            traceback.print_exc()
            rows = fallback_task(task)
            meta = {"error": repr(exc)}
        solved[task_id] = rows
        metas[task_id] = meta
        out_path.write_text(json.dumps(solved, separators=(",", ":")))
        prog_path.write_text(json.dumps({"done": len(solved), "total": len(items), "last": task_id, "meta": metas}, indent=2))
        print(f"[shard {args.shard}] checkpoint {len(solved)}/{len(items)}: {task_id}", flush=True)

    out_path.write_text(json.dumps(solved, separators=(",", ":")))
    print(f"[shard {args.shard}] finished solved={len(solved)}", flush=True)


if __name__ == "__main__":
    main()
'''


(WORK_DIR / "arc_worker.py").write_text(WORKER_CODE)
print("wrote", WORK_DIR / "arc_worker.py")


def stream_reader(proc, name, q):
    for line in proc.stdout:
        q.put((name, line.rstrip()))
    proc.stdout.close()


def launch_shards():
    procs = []
    q = queue.Queue()
    for shard, gpu in enumerate(GPU_IDS):
        out = WORK_DIR / f"shard_{shard}.json"
        cmd = [
            sys.executable,
            str(WORK_DIR / "arc_worker.py"),
            "--gpu",
            str(gpu),
            "--shard",
            str(shard),
            "--nshards",
            str(N_SHARDS),
            "--challenge",
            str(CHALLENGE_FILE),
            "--model",
            MODEL_PATH,
            "--out",
            str(out),
        ]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(WORK_DIR),
            env=env,
        )
        procs.append((shard, proc))
        threading.Thread(target=stream_reader, args=(proc, f"shard {shard}", q), daemon=True).start()

    live = {p.pid for _, p in procs}
    last_print = time.time()
    while live:
        try:
            name, line = q.get(timeout=1)
            if line.startswith("VIZ_FILE "):
                img_path = line.split(" ", 1)[1].strip()
                print(f"[{name}] viz {img_path}")
                if display is not None and IPyImage is not None and Path(img_path).exists():
                    display(IPyImage(filename=img_path))
                continue
            print(f"[{name}] {line}")
        except queue.Empty:
            pass
        for shard, proc in procs:
            if proc.pid in live and proc.poll() is not None:
                live.remove(proc.pid)
                print(f"shard {shard} returncode {proc.returncode}")
        if time.time() - last_print > 60:
            done = 0
            for shard, _ in procs:
                p = WORK_DIR / f"shard_{shard}.json"
                if p.exists():
                    try:
                        done += len(json.loads(p.read_text()))
                    except Exception:
                        pass
            print(f"waiting... solved so far {done}")
            last_print = time.time()
    return procs


def shape(g):
    return (len(g), len(g[0]) if isinstance(g, list) and g and isinstance(g[0], list) else 0)


def is_grid(g):
    if not isinstance(g, list) or not g or not all(isinstance(r, list) for r in g):
        return False
    w = len(g[0])
    return w > 0 and len(g) <= 30 and w <= 30 and all(
        len(r) == w and all(isinstance(v, int) and 0 <= v <= 9 for v in r) for r in g
    )


def most_common_color(g):
    vals = [v for r in g for v in r]
    return max(set(vals), key=vals.count) if vals else 0


def fallback_rows(task):
    rows = []
    for case in task["test"]:
        inp = case["input"]
        h, w = shape(inp)
        bg = most_common_color(inp)
        rows.append({"attempt_1": inp, "attempt_2": [[bg for _ in range(w)] for _ in range(h)]})
    return rows


def merge_and_validate():
    tasks = json.loads(CHALLENGE_FILE.read_text())
    merged = {}
    for shard in range(N_SHARDS):
        p = WORK_DIR / f"shard_{shard}.json"
        if p.exists():
            try:
                merged.update(json.loads(p.read_text()))
            except Exception as exc:
                print("could not read", p, exc)

    missing = []
    bad = []
    for tid, task in tasks.items():
        rows = merged.get(tid)
        if not isinstance(rows, list) or len(rows) != len(task["test"]):
            missing.append(tid)
            rows = fallback_rows(task)
        fixed = []
        for i, case in enumerate(task["test"]):
            row = rows[i] if i < len(rows) and isinstance(rows[i], dict) else {}
            a1 = row.get("attempt_1")
            a2 = row.get("attempt_2")
            if not is_grid(a1):
                bad.append((tid, i, "attempt_1"))
                a1 = case["input"]
            if not is_grid(a2):
                bad.append((tid, i, "attempt_2"))
                h, w = shape(case["input"])
                bg = most_common_color(case["input"])
                a2 = [[bg for _ in range(w)] for _ in range(h)]
            fixed.append({"attempt_1": a1, "attempt_2": a2})
        merged[tid] = fixed

    ordered = {tid: merged[tid] for tid in tasks.keys()}
    FINAL_SUBMISSION.write_text(json.dumps(ordered, separators=(",", ":")))
    print("merged solved", len(ordered), "tasks ->", FINAL_SUBMISSION)
    print("missing filled", len(missing), "bad fixed", len(bad))
    if missing:
        print("missing head", missing[:10])
    if bad:
        print("bad head", bad[:10])
    return ordered


def eval_score(submission):
    if not EVAL_MODE or not SOLUTION_FILE.exists():
        return
    sol = json.loads(SOLUTION_FILE.read_text())
    hit = 0
    total = 0
    hits = []
    for tid, outs in sol.items():
        rows = submission.get(tid, [])
        for i, true in enumerate(outs):
            total += 1
            row = rows[i] if i < len(rows) else {}
            ok = row.get("attempt_1") == true or row.get("attempt_2") == true
            hit += int(ok)
            if ok:
                hits.append((tid, i))
    print("eval exact", hit, "/", total, "=", hit / total if total else 0)
    print("hits", hits[:50])


started = time.time()
procs = launch_shards()
submission = merge_and_validate()
eval_score(submission)
print("DONE final:", FINAL_SUBMISSION, "elapsed_min=", round((time.time() - started) / 60, 2))
