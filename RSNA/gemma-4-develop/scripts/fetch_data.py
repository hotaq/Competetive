"""Download what `swegemma eval` needs for a set of tasks into data/ (the full dataset is 22 GB).

    python3 scripts/fetch_data.py $(cat dev_tasks.txt)

Graphs/embeddings are hard-linked pairs upstream (task-named + commit-named) and the zip
download leaves one side empty; the harness looks up the commit-named file first, so both
names get the readable copy.
"""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMP = "gemma-4-developer-agent"
DATA = Path(__file__).resolve().parent.parent / "data"
SIZES = {r[0]: int(r[1]) for r in csv.reader(open(DATA / "_files.txt")) if len(r) > 1 and r[1].isdigit()}


def fetch(rel: str) -> Path:
    dest = DATA / rel
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if SIZES[rel] == 0:  # the empty half of a hard-linked pair; kaggle refuses to serve it
        dest.touch()
        return dest
    subprocess.run(["kaggle", "competitions", "download", COMP, "-f", rel, "-p", str(dest.parent), "-q"],
                   check=True, capture_output=True)
    z = dest.with_name(dest.name + ".zip")
    if z.exists():  # kaggle zips some files
        with zipfile.ZipFile(z) as f:
            f.extractall(dest.parent)
        z.unlink()
    return dest


def main(ids: list[str]) -> None:
    listing = SIZES
    tasks = {t["instance_id"]: t for t in map(json.loads, open(DATA / "tasks.jsonl"))}

    def fetch_task(iid: str) -> None:
        t = tasks[iid]
        fetch(f"snapshots/{iid}.tgz")
        short = t["repo"].split("/")[-1]
        for kind, ext in [("graphs", ".json"), ("embeddings", ".npz")]:
            names = [f"{kind}/{iid}{ext}", f"{kind}/{short}_{t['base_commit']}{ext}"]
            names += [f"{kind}/{o}{ext}" for o, ot in tasks.items() if ot["base_commit"] == t["base_commit"] and o != iid]
            paths = [fetch(n) for n in names if n in listing]
            good = max(paths, key=lambda p: p.stat().st_size)
            for p in paths:
                if p.stat().st_size < 100 and good.stat().st_size >= 100:
                    shutil.copyfile(good, p)
        print(iid, "ok", flush=True)

    # Each kaggle call is ~10 s of overhead, so run them in parallel. Tasks sharing a base
    # commit may copy the same graph file twice; the content is identical, so that's harmless.
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, [n for n in listing if n.startswith(("wheels/", "sandbox/"))]))
        list(ex.map(fetch_task, ids))

    # Test deps the public wheels lack (fastapi tests import dirty_equals; pydantic 2.13
    # needs typing_inspection; newer fastapi tests use inline_snapshot). Without them, fastapi tasks fail collection even with the gold patch.
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                    "--python-version", "3.13", "--platform", "manylinux2014_x86_64", "-q",
                    "-d", str(DATA / "wheels"), "dirty-equals", "typing-inspection", "inline-snapshot", "asttokens", "executing"], check=True)
    # The harness caches unpacked wheels here and never rebuilds, so drop it after any change.
    shutil.rmtree(Path(tempfile.gettempdir()) / "swegemma_sp_cache_v8", ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
