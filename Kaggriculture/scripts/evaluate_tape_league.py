# ruff: noqa: E501
"""Play every tape in a pool, verbatim, against a fixed opponent set; keep per-(tape, opponent) aggregates.

Screen for a new base lineage: which recorded agents, played open-loop, beat the current router
(`--router`) on fresh seeds? Tapes are processed in chunks so memory stays bounded; only sums are
kept (games, points, our money, opponent money), which is all a ranking needs. Seats are both
played for protocol parity even though tape-vs-tape games are seat-symmetric.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggriculture_rs import ShopRouter, Tape, play_batch

ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}


def pool_tapes(tar_path: Path) -> list[tuple[str, dict[str, Any]]]:
    out = []
    if tar_path.is_dir():  # harvest_tape_pool.py output: <dir>/tapes/*.json
        for path in sorted((tar_path / "tapes").glob("*.json")):
            out.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
        return out
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            name = member.name
            if not name.endswith(".json") or "/tapes/" not in name or "/._" in name:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            payload = json.load(handle)
            out.append((name.split("/")[-1][:-5], payload))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="tape_pool tar.gz, or a harvest directory with tapes/")
    parser.add_argument("--opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--router", type=Path, default=None, help="routes.json reference arm (played as 'ROUTER')")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=120)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tapes = pool_tapes(args.pool)
    opp_names = [p.stem for p in args.opponents]
    opp_players = [Tape(json.dumps(json.loads(p.read_text(encoding="utf-8"))["actions"][1:720])) for p in args.opponents]
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    meta = {name: {k: v for k, v in payload.items() if k != "actions"} for name, payload in tapes}
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0, 0.0]))

    def run(players: list[Any], names: list[str]) -> None:
        offset = len(players)
        allp = players + opp_players
        jobs = [(seed, seat, i, offset + o) for i in range(len(players)) for o in range(len(opp_players)) for seed in seeds for seat in (0, 1)]
        results = play_batch(json.dumps(ENGINE_CONFIG), allp, jobs, args.threads)
        for (seed, seat, i, o), r in zip(jobs, results, strict=True):
            cell = agg[names[i]][opp_names[o - offset]]
            cell[0] += 1
            cell[1] += r["point"]
            cell[2] += r["money"][seat]
            cell[3] += r["money"][1 - seat]

    started = time.perf_counter()
    if args.router is not None:
        run([ShopRouter(args.router.read_text(encoding="utf-8"), 2)], ["ROUTER"])
        print(f"reference router done in {time.perf_counter() - started:.0f}s", flush=True)
    total = len(tapes)
    for start in range(0, total, args.chunk):
        chunk = tapes[start:start + args.chunk]
        players = [Tape(json.dumps(payload["actions"][1:720])) for _, payload in chunk]
        run(players, [name for name, _ in chunk])
        done = min(start + args.chunk, total)
        elapsed = time.perf_counter() - started
        print(f"  {done}/{total} tapes  {elapsed / 60:.1f} min  eta {elapsed / done * (total - done) / 60:.1f} min", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"seeds": [seeds.start, seeds.stop - 1], "opponents": opp_names, "meta": meta,
                                           "aggregates": {t: dict(v) for t, v in agg.items()}}) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
