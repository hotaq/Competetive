"""Harvest compact action tapes for leaderboard opponents from their public replays.

Each replay is ~32 MB, so a replay is downloaded, reduced to the target team's action
sequence, and discarded before the next one. Only the compact tape is kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from download_top1_replays import (
    COMPETITION,
    completed_public_episodes,
    plain,
    replay_bytes,
    scoring_submission,
)
from kaggle.api.kaggle_api_extended import KaggleApi


def leaderboard(api: KaggleApi) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        api.competition_leaderboard_download(COMPETITION, tmp, quiet=True)
        archive = next(Path(tmp).glob("*.zip"))
        with zipfile.ZipFile(archive) as bundle:
            name = bundle.namelist()[0]
            rows = list(csv.DictReader(bundle.read(name).decode("utf-8-sig").splitlines()))
    return [
        {"team_id": int(r["TeamId"]), "team_name": r["TeamName"],
         "score": float(r["Score"]), "rank": int(r["Rank"])}
        for r in rows
    ]


def team_seat(replay: dict[str, Any], team_name: str) -> int | None:
    names = replay.get("info", {}).get("TeamNames", [])
    wanted = " ".join(team_name.casefold().split())
    for seat, name in enumerate(names):
        if " ".join(str(name).casefold().split()) == wanted:
            return seat
    return None


def harvest(api: KaggleApi, leader: dict[str, Any], episodes: int, out_root: Path) -> dict:
    submission = scoring_submission(api, leader)
    listed = completed_public_episodes(api, submission["submission_id"])[:episodes]
    slug = "".join(c if c.isalnum() else "_" for c in leader["team_name"]).strip("_").lower()
    out = out_root / f"rank{leader['rank']:03d}_{slug}"
    (out / "tapes").mkdir(parents=True, exist_ok=True)

    tapes = []
    for row in listed:
        replay = json.loads(replay_bytes(api, row["id"]))
        seat = team_seat(replay, leader["team_name"])
        if seat is None:
            continue
        actions = [step[seat]["action"] for step in replay["steps"]]
        payload = {
            "team_name": leader["team_name"], "team_id": leader["team_id"],
            "submission_id": submission["submission_id"], "episode": row["id"],
            "seed": replay["info"].get("seed"), "seat": seat,
            "final_money": replay["rewards"][seat],
            "opponent": replay["info"]["TeamNames"][1 - seat],
            "opponent_money": replay["rewards"][1 - seat],
            "length": len(actions), "actions": actions,
        }
        path = out / "tapes" / f"ep{row['id']}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        tapes.append((row["id"], payload, [json.dumps(a, sort_keys=True) for a in actions]))
        del replay

    pairs = []
    for (ia, _, xa), (ib, _, xb) in itertools.combinations(tapes, 2):
        n = min(len(xa), len(xb))
        same = sum(1 for i in range(n) if xa[i] == xb[i])
        first = next((i for i in range(n) if xa[i] != xb[i]), None)
        pairs.append({"a": ia, "b": ib, "identity": same / n, "first_difference": first})

    summary = {
        "rank": leader["rank"], "team_name": leader["team_name"],
        "score": leader["score"], "submission_id": submission["submission_id"],
        "tapes": [
            {"episode": e, "seat": p["seat"], "seed": p["seed"],
             "final_money": p["final_money"], "opponent": p["opponent"],
             "sha256": hashlib.sha256(
                 json.dumps(p["actions"], separators=(",", ":")).encode()).hexdigest()}
            for e, p, _ in tapes
        ],
        "pairwise": pairs,
        "min_identity": min((x["identity"] for x in pairs), default=None),
        "max_identity": max((x["identity"] for x in pairs), default=None),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teams", type=int, default=5)
    parser.add_argument("--episodes-per-team", type=int, default=3)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    board = leaderboard(api)[: args.teams]
    results = []
    for leader in board:
        summary = harvest(api, leader, args.episodes_per_team, args.output)
        results.append(summary)
        lo, hi = summary["min_identity"], summary["max_identity"]
        span = f"{lo:.1%}-{hi:.1%}" if lo is not None else "n/a"
        money = [t["final_money"] for t in summary["tapes"]]
        print(f"rank {summary['rank']:>2}  {summary['team_name']:<20} score {summary['score']:>7}"
              f"  tapes {len(summary['tapes'])}  identity {span:>14}"
              f"  money {min(money):,.0f}-{max(money):,.0f}")
    (args.output / "index.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
