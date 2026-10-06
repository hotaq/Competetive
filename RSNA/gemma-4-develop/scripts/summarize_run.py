"""Per-task table + tool-error counts + analyzer localization accuracy for local runs.

    python3 scripts/summarize_run.py results/v2_sed [results/baseline_v1 ...]

ctx = largest prompt of any turn (scorer limit: 32,768 tokens). $ = cost at ~$0.14/M in, $0.40/M out.
loc= scores each code_analyzer LOCATION against the gold patch: F = right file,
L = its line range overlaps a gold hunk, - = wrong file.
"""

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = {t["instance_id"]: t["patch"] for t in map(json.loads, open(ROOT / "data/tasks.jsonl"))}


def tool_errors(steps) -> collections.Counter:
    """Count failed tool calls as 'tool:ErrorType' (CommandError is often just grep finding nothing)."""
    errs = collections.Counter()
    for step in steps:
        content = str((step.get("observation") or {}).get("content", ""))
        m = re.search(r'"error_type": "(\w+)"', content)
        if m and step.get("tool_calls"):
            errs[f"{step['tool_calls'][0]['function_name']}:{m.group(1)}"] += 1
    return errs


def gold_hunks(patch: str) -> dict[str, list[tuple[int, int]]]:
    """file -> [(start, end)] of changed lines in the original file."""
    hunks, f = collections.defaultdict(list), None
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            f = line[6:].strip()
        elif line.startswith("@@") and f:
            a, n = re.match(r"@@ -(\d+)(?:,(\d+))?", line).groups()
            hunks[f].append((int(a), int(a) + int(n or 1)))
    return hunks


def localization(steps, patch: str) -> str:
    """Score every LOCATION the analyzer reported to the coder."""
    hunks, marks = gold_hunks(patch), []
    for step in steps:  # the analyzer's report is its own agent step whose message holds LOCATION:
        text = str(step.get("message", ""))
        if step.get("source") == "agent" and "LOCATION:" in text:
            m = re.search(r"LOCATION:\s*`?([\w./-]+\.py)(?::(\d+)(?:-(\d+))?)?", text)
            if not m:
                marks.append("?")
                continue
            path, a, b = m.group(1).removeprefix("/workspace/").lstrip("./"), m.group(2), m.group(3)
            if path not in hunks:
                marks.append("-")
            elif a and any(int(a) <= e and int(b or a) >= s for s, e in hunks[path]):
                marks.append("L")
            else:
                marks.append("F")
    return "".join(marks) or "none"


for run in map(Path, sys.argv[1:]):
    rows = [json.loads(l) for l in open(run / "task_results.jsonl")]
    total = collections.Counter()
    print(f"\n== {run.name}: {sum(r['resolved'] for r in rows)}/{len(rows)} resolved")
    for r in sorted(rows, key=lambda r: r["instance_id"]):
        trace = run / "traces" / f"trace_{r['instance_id']}.json"
        tr = json.loads(trace.read_text()) if trace.exists() else {}
        steps, fm = tr.get("steps", []), tr.get("final_metrics") or {}
        ctx = max([(st.get("metrics") or {}).get("prompt_tokens", 0) for st in steps] or [0])
        cost = fm.get("total_prompt_tokens", 0) * 0.14e-6 + fm.get("total_completion_tokens", 0) * 0.40e-6
        errs = tool_errors(steps)
        total += errs
        print(f"  {r['instance_id']:15s} {'PASS' if r['resolved'] else 'fail'}  calls={r.get('tool_calls', 0):3}"
              f"  {r['duration_seconds']:5.0f}s  patch={r.get('agent_patch_size', 0):5}B  ctx={ctx:6,}  ${cost:.3f}"
              f"  loc={localization(steps, GOLD.get(r['instance_id'], ''))}  errors={dict(errs)}"
              + (f"  ERR={r['error'][:60]}" if r.get("error") else ""))
    print(f"  tool errors total: {dict(total)}")
