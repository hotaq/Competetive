"""Localization benchmark for `ct.py find` (no model calls): does the issue text rank the code the gold patch
changes in the top 10?  python3 scripts/bench_find.py [BUNDLE]   (uses every snapshot in data/snapshots)

sym@10  = a function/class the gold patch edits is among the top-10 source results
file@10 = a file the gold patch edits appears among the top-10 source results
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
bundle = sys.argv[1] if len(sys.argv) > 1 else "submission"
spec = importlib.util.spec_from_file_location("ct", ROOT / bundle / "skills/code-tools/scripts/ct.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)
tasks = {t["instance_id"]: t for t in map(json.loads, open(ROOT / "data/tasks.jsonl"))}


def gold_symbols(patch, repo_root):
    """(file, qualname) of every class/function whose line range overlaps a gold hunk."""
    hits, f = set(), None
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            f = line[6:].strip()
        elif line.startswith("@@") and f and f.endswith(".py"):
            a, n = re.match(r"@@ -(\d+)(?:,(\d+))?", line).groups()
            a, b = int(a), int(a) + int(n or 1)
            tree = ct.parse(os.path.join(repo_root, f)) if os.path.exists(os.path.join(repo_root, f)) else None
            for q, node in ct.symbols(tree) if tree else []:
                if ct.start(node) <= b and node.end_lineno >= a:
                    hits.add((f, q))
    return hits


rows = []
for snap in sorted((ROOT / "data/snapshots").glob("*.tgz")):
    iid = snap.name[:-4]
    t = tasks[iid]
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(snap) as tf:
            tf.extractall(tmp, filter="data")
        ct.ROOT = tmp
        if os.path.exists("/tmp/ct_index.json"):
            os.remove("/tmp/ct_index.json")
        issue = re.sub(r"<!--.*?-->", "", t["problem_statement"], flags=re.S)
        t0 = time.time()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ct.find(issue)
        secs = time.time() - t0
        src = buf.getvalue().split("tests:")[0]
        found = re.findall(r"^\s+(\S+?):\d+-\d+\s+(\S+)", src, re.M)
        gold = gold_symbols(t["patch"], tmp)
        gold_files = {f for f, _ in gold} | set(re.findall(r"^\+\+\+ b/(\S+\.py)", t["patch"], re.M))
        sym = any((f, q) in gold for f, q in found)
        fil = any(f in gold_files for f, _ in found)
        rows.append((iid, sym, fil, secs))
        print(f"{iid:15s} sym@10={'Y' if sym else '-'} file@10={'Y' if fil else '-'}  index+query {secs:4.1f}s", flush=True)
os.path.exists("/tmp/ct_index.json") and os.remove("/tmp/ct_index.json")
n = len(rows)
print(f"\n{n} tasks: sym@10 {sum(r[1] for r in rows)}/{n}, file@10 {sum(r[2] for r in rows)}/{n}, "
      f"median time {sorted(r[3] for r in rows)[n // 2]:.1f}s")
