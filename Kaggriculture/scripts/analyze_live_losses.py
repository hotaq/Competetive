"""Break down a live submission's episodes by opponent strength, and list the losses.

Win rate on its own is not interpretable: it is dominated by who the matchmaker
sent. Every episode here is joined to the opponent's current leaderboard score so
results can be read per rating band.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "kaggriculture"
TEAM_ID = 16621481
BANDS = ((0, 1500), (1500, 2000), (2000, 2500), (2500, 3100))


def leaderboard(api: KaggleApi) -> dict[int, dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        api.competition_leaderboard_download(COMPETITION, tmp, quiet=True)
        archive = next(Path(tmp).glob("*.zip"))
        with zipfile.ZipFile(archive) as bundle:
            name = bundle.namelist()[0]
            rows = list(csv.DictReader(bundle.read(name).decode("utf-8-sig").splitlines()))
    return {
        int(row["TeamId"]): {
            "rank": int(row["Rank"]),
            "score": float(row["Score"]),
            "name": row["TeamName"],
        }
        for row in rows
    }


def episodes(api: KaggleApi, submission_id: int, board: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for episode in api.competition_list_episodes(submission_id) or []:
        if episode.state.name != "COMPLETED":
            continue
        agents = list(episode.agents)
        mine = [a for a in agents if a.team_id == TEAM_ID]
        other = [a for a in agents if a.team_id != TEAM_ID]
        if not mine or not other or mine[0].reward is None or other[0].reward is None:
            continue
        opponent = other[0]
        entry = board.get(opponent.team_id, {})
        rows.append({
            "episode": episode.id,
            "end_time": str(episode.end_time),
            "our_money": float(mine[0].reward),
            "opponent_money": float(opponent.reward),
            "margin": float(mine[0].reward) - float(opponent.reward),
            "opponent_team": opponent.team_name or entry.get("name"),
            "opponent_score": entry.get("score"),
            "opponent_rank": entry.get("rank"),
        })
    rows.sort(key=lambda row: row["end_time"])
    return rows


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rated = [row for row in rows if row["opponent_score"] is not None]
    bands = []
    for low, high in BANDS:
        bucket = [row for row in rated if low <= row["opponent_score"] < high]
        if not bucket:
            continue
        bands.append({
            "band": f"{low}-{high}",
            "episodes": len(bucket),
            "wins": sum(row["margin"] > 0 for row in bucket),
            "win_rate": sum(row["margin"] > 0 for row in bucket) / len(bucket),
            "median_margin": st.median(row["margin"] for row in bucket),
            "median_our_money": st.median(row["our_money"] for row in bucket),
            "median_opponent_money": st.median(row["opponent_money"] for row in bucket),
        })
    losses = sorted((row for row in rows if row["margin"] <= 0), key=lambda row: row["margin"])
    return {
        "episodes": len(rows),
        "wins": sum(row["margin"] > 0 for row in rows),
        "median_opponent_score": st.median(row["opponent_score"] for row in rated) if rated else None,
        "bands": bands,
        "losses": losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    board = leaderboard(api)

    full: dict[str, Any] = {}
    for submission_id in args.submission:
        rows = episodes(api, submission_id, board)
        summary = report(rows)
        full[str(submission_id)] = {"summary": summary, "episodes": rows}

        print(f"\n=== submission {submission_id} — {summary['wins']}/{summary['episodes']} wins, "
              f"median opponent score {summary['median_opponent_score']}")
        print(f"{'opponent band':>14}{'eps':>5}{'win':>7}{'med margin':>12}{'our money':>11}{'opp money':>11}")
        for band in summary["bands"]:
            print(f"{band['band']:>14}{band['episodes']:>5}{band['win_rate']:>6.0%}"
                  f"{band['median_margin']:>12,.0f}{band['median_our_money']:>11,.0f}"
                  f"{band['median_opponent_money']:>11,.0f}")
        print(f"  losses ({len(summary['losses'])}):")
        for row in summary["losses"]:
            score = row["opponent_score"]
            print(f"    {row['end_time'][:16]}  vs {str(row['opponent_team'])[:24]:24s}"
                  f" score {score if score is None else f'{score:7.1f}'}"
                  f"  us {row['our_money']:>8,.0f}  them {row['opponent_money']:>8,.0f}"
                  f"  margin {row['margin']:>9,.0f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
