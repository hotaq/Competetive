"""Keep only tasks the local harness can grade: gold patch passes AND no patch fails with exit 1
(tests failed) or 137 (killed: the unfixed bug hangs/eats memory). Exit 2/3 = collection or
internal error, i.e. a broken environment, not a failing test.

    python3 scripts/validate_tasks.py <task_id ...>     # writes the valid ones to dev_tasks.txt
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(name: str, env: dict, ids: list[str]) -> dict:
    subprocess.run([str(ROOT / "scripts/eval_local.sh"), name, *ids], env={**os.environ, **env},
                   check=True, stdout=subprocess.DEVNULL)
    return {d["instance_id"]: d for d in map(json.loads, open(ROOT / f"results/{name}/task_results.jsonl"))}


def main(ids: list[str]) -> None:
    gold = run("check_gold", {"GOLD": "1"}, ids)
    nopatch = run("check_nopatch", {"EXTRA_ARGS": "--skip-agent-patch"}, ids)
    valid = []
    for iid in sorted(ids):
        g, n = gold.get(iid, {}), nopatch.get(iid, {})
        ok = g.get("resolved") is True and n.get("resolved") is False and n.get("test_exit_code") in (1, 137)
        print(f"{iid:16s} gold={g.get('resolved')!s:5} (exit {g.get('test_exit_code')})  "
              f"nopatch={n.get('resolved')!s:5} (exit {n.get('test_exit_code')})  {'VALID' if ok else ''}")
        valid += [iid] * ok
    (ROOT / "dev_tasks.txt").write_text("\n".join(valid) + "\n")
    print(f"{len(valid)}/{len(ids)} valid -> dev_tasks.txt")


if __name__ == "__main__":
    main(sys.argv[1:])
