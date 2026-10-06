"""Score the V95 holdout: the fitted shop-pair table against the best single tape.

The pair a game reveals is not a property of the seed -- over the full 1,390-tape pool it
moves with the tape being played. It is constant across the 40 shortlisted tapes, which is
what makes the lookup well defined here, and this script re-checks that on every game rather
than assuming it: the tape the table selects must reveal the same pair the lookup used, or
the row is flagged.

Both arms therefore come out of one pass per (seed, opponent, seat): play the best single
tape, read the pair off that game, look the winner up, and play the winner only when it
differs. Games where the table falls back to the pooled best cost nothing extra.

Read once. See `analysis/v95_shop_pair_router_20260830/PREREGISTRATION.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from concurrent.futures import ProcessPoolExecutor

import evaluate_shop_router as engine

_TABLE: dict[str, str] = {}
_BEST = ""


def _init(paths: dict[str, str], table: dict[str, str], best: str) -> None:
    engine._init(paths, (best, *table.values()))
    _TABLE.update(table)
    globals()["_BEST"] = best


def _work(job: tuple[str, int, int]) -> dict[str, Any]:
    opponent, seed, seat = job
    single = engine.play(_BEST, opponent, seed, seat)
    pair = f"{single['shop1']}|{single['shop2']}"
    winner = _TABLE.get(pair, _BEST)
    if winner == _BEST:
        routed, mismatch = single, False
    else:
        routed = engine.play(winner, opponent, seed, seat)
        mismatch = (routed["shop1"], routed["shop2"]) != (single["shop1"], single["shop2"])
    return {
        "opponent": opponent, "seed": seed, "seat": seat, "pair": pair, "winner": winner,
        "branched": winner != _BEST, "pair_mismatch": mismatch,
        "single_point": single["point"], "single_margin": single["margin"],
        "routed_point": routed["point"], "routed_margin": routed["margin"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    fitted = json.loads(args.table.read_text(encoding="utf-8"))
    best = fitted["best_overall"]
    table = {key: cell["winner"] for key, cell in fitted["cells"].items()}
    tapes = {best, *table.values()}

    def key(path: Path) -> str:
        return f"{path.parent.parent.name}/{path.stem}"

    paths = {key(p): str(p) for p in args.pool.glob("*.json")
             if not p.name.startswith("._")}
    opponents = [key(p) for p in args.opponents]
    paths.update({key(p): str(p) for p in args.opponents})
    missing = [t for t in tapes if t not in paths]
    assert not missing, f"table names the pool does not hold: {missing}"

    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    jobs = [(o, s, seat) for o in opponents for s in seeds for seat in (0, 1)]
    print(f"{len(tapes)} distinct tapes, {len(opponents)} opponents, {len(seeds)} seeds "
          f"x 2 seats = {len(jobs):,} matchups on {args.workers} workers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.jsonl")
    rows: list[dict[str, Any]] = []
    finished: set[tuple[str, int, int]] = set()
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            finished.add((row["opponent"], row["seed"], row["seat"]))
        jobs = [j for j in jobs if j not in finished]
        print(f"resuming: {len(finished):,} recorded, {len(jobs):,} left")

    with partial.open("a", encoding="utf-8") as sink, \
            ProcessPoolExecutor(args.workers, initializer=_init,
                                initargs=(paths, table, best)) as pool:
        for done, row in enumerate(pool.map(_work, jobs, chunksize=4), 1):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            if done % 200 == 0:
                sink.flush()
                print(f"  {done}/{len(jobs)}", flush=True)

    args.output.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    played = len(rows) + sum(1 for r in rows if r["branched"])
    print(f"wrote {len(rows):,} matchups ({played:,} games) to {args.output}")


if __name__ == "__main__":
    main()
