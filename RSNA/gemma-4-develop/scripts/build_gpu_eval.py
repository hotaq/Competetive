"""Kaggle GPU notebook that runs OUR bundle on the real stack (int4 Gemma on vLLM, host's recipe) for a few
tasks and prints each agent transcript, so we can see what happens on the scorer's hardware.

    python3 scripts/build_gpu_eval.py [--src submission] [--tasks requests_7309 fastapi_15589] [--minutes 4] [--calls 24]
    kaggle kernels push -p kernel_gpu
Install + vLLM cells are copied verbatim from the host's ryanholbrook/getting-started-gemma-4-developer-agent
(pulled into ref/nb_getting-started-gemma-4-developer-agent).
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST = next((ROOT / "ref/nb_getting-started-gemma-4-developer-agent").glob("*.ipynb"))

EVAL = '''import asyncio, concurrent.futures, json, time
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator

def run_sync(fn, *a, **k):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(fn(*a, **k))).result()

TASK_IDS = {task_ids}
limits, gen_constraints = build_submission_limits()
eval_config = EvalConfig(
    tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots', results_dir=WORKING_DIR / 'results',
    submission_dir=AGENT_DIR, models=models, sandbox='subprocess',
    timeout_seconds=300, max_time_minutes={minutes}, max_tool_calls={calls}, max_turns=None,
    limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                    token_threshold=14336, event_retention_size=5),
    graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'),
    wheels_dir=DATA_DIR / 'wheels', verbose=False,
)
evaluator = Evaluator(eval_config)
rows = []
for i, task in enumerate([t for t in tasks if t.instance_id in TASK_IDS], 1):
    t0 = time.time()
    r = run_sync(evaluator.evaluate_task, task=task, task_index=i, total_tasks=len(TASK_IDS))
    rows.append(dict(id=task.instance_id, resolved=r.resolved, exit=r.test_exit_code, patch=len(r.agent_patch or ''),
                     tool_calls=r.tool_calls, secs=round(time.time() - t0), error=(r.error or '')[:300]))
    print(json.dumps(rows[-1]), flush=True)
(WORKING_DIR / 'gpu_eval.json').write_text(json.dumps(rows, indent=1))
'''

SHOW = '''import re
for log in sorted((WORKING_DIR / 'results' / 'logs').glob('*.log')):
    text = re.sub(r'\\x1b\\[[0-9;]*m', '', log.read_text(errors='ignore'))
    print('#' * 20, log.name, f'({len(text)} chars)')
    print(text[-6000:])
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="submission")
    ap.add_argument("--tasks", nargs="+", default=["requests_7309", "fastapi_15589"])
    ap.add_argument("--minutes", type=float, default=4)
    ap.add_argument("--calls", type=int, default=24)
    ap.add_argument("--slug", default="gemma4-gpu-eval")
    args = ap.parse_args()

    host = [''.join(c["source"]) for c in json.load(open(HOST))["cells"] if c["cell_type"] == "code"]
    src = ROOT / args.src
    files = {p.relative_to(src).as_posix(): p.read_text() for p in sorted(src.rglob("*"))
             if p.is_file() and not p.name.startswith(".") and "__pycache__" not in p.parts}
    bundle = f'''FILES = {json.dumps(files, indent=1)}
from pathlib import Path
from swegemma.models import load_tasks
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORKING_DIR = Path('/kaggle/working')
AGENT_DIR = WORKING_DIR / 'agent'
for rel, text in FILES.items():
    (AGENT_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
    (AGENT_DIR / rel).write_text(text)
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
tasks = load_tasks(TASKS_PATH)
print(len(FILES), 'bundle files;', len(tasks), 'tasks')
'''
    cells = [host[0], bundle, host[3],
             EVAL.format(task_ids=set(args.tasks), minutes=args.minutes, calls=args.calls), SHOW]
    out = ROOT / "kernel_gpu"
    out.mkdir(exist_ok=True)
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c}
                    for c in cells]}
    (out / f"{args.slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.load(open(HOST.parent / "kernel-metadata.json"))
    meta.update(id=f"chinnaphatkhun/{args.slug}", title=args.slug, code_file=f"{args.slug}.ipynb",
                is_private=True, keywords=[])
    for k in ("id_no", "docker_image"):
        meta.pop(k, None)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote kernel_gpu/{args.slug}.ipynb: {len(files)} bundle files, tasks={args.tasks}, "
          f"{args.minutes} min / {args.calls} calls, machine={meta['machine_shape']}")


if __name__ == "__main__":
    main()
