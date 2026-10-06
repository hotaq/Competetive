"""Split the frozen 328-episode index (sorted by end_time) into SELECT = older 164, CONFIRM = newer 164.
Halves come from the frozen index, not from whichever files exist; missing files are reported and skipped."""
import json, os
from pathlib import Path
F = Path("analysis/final_round_20260927")
idx = json.load(open(F / "fresh_index.json"))
half = len(idx) // 2
out = {}
for name, part in (("select", idx[:half]), ("confirm", idx[half:])):
    d = F / name
    d.mkdir(exist_ok=True)
    have = [x for x in part if (F / "replays" / f"ep{x[1]}.json").exists()]
    for x in have:
        link = d / f"ep{x[1]}.json"
        if not link.exists():
            os.symlink(os.path.abspath(F / "replays" / f"ep{x[1]}.json"), link)
    out[name] = [x[1] for x in have]
    print(name, "frozen", len(part), "present", len(have), part[0][0][:16], "->", part[-1][0][:16])
json.dump(out, open(F / "split.json", "w"))
