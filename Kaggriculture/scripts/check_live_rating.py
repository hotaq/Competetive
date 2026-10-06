"""Report live Kaggle standing: active submissions, per-agent episode results, and rank."""

from __future__ import annotations

import argparse
import csv
import statistics as st
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "kaggriculture"

# Our team. `boatlee` (16641169) is a different team and a competitor; the Kaggle API
# happily returns another team's submissions, so always filter on this id.
TEAM_ID = 16621481
TEAM_NAME = "zero waste"

# Benchmarks worth watching each time: the competitor whose tape we recovered, and the
# rank-3 open-loop agent used as our local reference.
WATCH = ("boatlee", "Yusuke Hayashi")


def field(agent: Any, *names: str) -> Any:
    for name in names:
        value = agent.get(name) if isinstance(agent, dict) else getattr(agent, name, None)
        if value is not None:
            return value
    return None


def episode_stats(api: KaggleApi, submission_id: int) -> dict[str, Any] | None:
    rows = []
    for episode in api.competition_list_episodes(submission_id) or []:
        if episode.state.name != "COMPLETED":
            continue
        agents = list(episode.agents)
        mine = [a for a in agents if field(a, "team_id", "teamId") == TEAM_ID]
        other = [a for a in agents if a not in mine]
        if not mine or not other:
            continue
        ours, theirs = field(mine[0], "reward"), field(other[0], "reward")
        if ours is None or theirs is None:
            continue
        rows.append((episode.end_time, ours, theirs))
    if not rows:
        return None
    rows.sort()
    wins = sum(1 for _, o, t in rows if o > t)
    recent = rows[-25:]
    return {
        "episodes": len(rows),
        "win_rate": wins / len(rows),
        "recent_win_rate": sum(1 for _, o, t in recent if o > t) / len(recent),
        "money": st.mean(o for _, o, _ in rows),
        "opponent_money": st.mean(t for _, _, t in rows),
        "last_episode": rows[-1][0],
    }


def leaderboard_line(api: KaggleApi) -> str:
    """Rank needs the full standings; the paged view only returns the top page."""
    with tempfile.TemporaryDirectory() as tmp:
        api.competition_leaderboard_download(COMPETITION, tmp, quiet=True)
        archive = next(Path(tmp).glob("*.zip"))
        with zipfile.ZipFile(archive) as bundle:
            name = bundle.namelist()[0]
            rows = list(csv.DictReader(bundle.read(name).decode("utf-8-sig").splitlines()))
    ours = next((r for r in rows if int(r["TeamId"]) == TEAM_ID), None)
    if ours is None:
        return f"leaderboard: team {TEAM_ID} not found"
    cut = {int(r["Rank"]): r["Score"] for r in rows if int(r["Rank"]) in (1, 10, 20)}
    lines = [f"{TEAM_NAME}: rank {ours['Rank']} / {len(rows)}  score {ours['Score']}   "
             f"| top1 {cut.get(1)}  top10 {cut.get(10)}  top20 {cut.get(20)}"]
    for name in WATCH:
        other = next((r for r in rows if r["TeamName"] == name), None)
        if other is not None:
            gap = float(other["Score"]) - float(ours["Score"])
            lines.append(f"  watch  {name:<16} rank {other['Rank']:>4}  score {other['Score']:>7}"
                         f"  ({gap:+.1f} vs us)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-episodes", action="store_true")
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    print(leaderboard_line(api))

    print(f"\n{'submission':>10}  {'score':>7}  {'submitted':<17}  {'eps':>4}  "
          f"{'win':>6}  {'last25':>7}  {'money':>8}  {'opp':>8}")
    for submission in api.competition_team_submissions(TEAM_ID):
        line = (f"{submission.id:>10}  {submission.public_score:>7}  "
                f"{submission.date_submitted:%Y-%m-%d %H:%M}")
        stats = None if args.skip_episodes else episode_stats(api, submission.id)
        if stats:
            line += (f"  {stats['episodes']:>4}  {stats['win_rate']:>5.0%}  "
                     f"{stats['recent_win_rate']:>6.0%}  {stats['money']:>8,.0f}  "
                     f"{stats['opponent_money']:>8,.0f}")
        print(line)


if __name__ == "__main__":
    main()
