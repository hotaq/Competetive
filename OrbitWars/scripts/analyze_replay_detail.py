#!/usr/bin/env python3
"""Detailed Orbit Wars replay forensics.

Reads a Kaggle/lab replay JSON and prints a markdown report with:
- score / production / planet-count timeline
- owner-change timeline
- launch log with approximate target, distance, and ETA
- likely launch that caused each capture

The target inference is geometric and intentionally dependency-free.  It uses
the launch angle at the observed turn and picks the first planet intersected by
that ray in the current frame.  Moving/orbiting targets can make the target a
near miss; those rows are marked with lower confidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


BOARD = 100.0
LAUNCH_CLEARANCE = 0.1
MAX_RAY = BOARD * 2.0
MAX_SPEED = 6.0


def fleet_speed(ships: int) -> float:
    ships = max(1, int(ships))
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def angle_delta(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def planet_dict(row: list[Any]) -> dict[str, Any]:
    return {
        "id": int(row[0]),
        "owner": int(row[1]),
        "x": float(row[2]),
        "y": float(row[3]),
        "radius": float(row[4]),
        "ships": int(row[5]),
        "production": int(row[6]),
    }


def fleet_dict(row: list[Any]) -> dict[str, Any]:
    return {
        "id": int(row[0]),
        "owner": int(row[1]),
        "x": float(row[2]),
        "y": float(row[3]),
        "angle": float(row[4]),
        "from_planet_id": int(row[5]),
        "ships": int(row[6]),
    }


def planets(obs: dict[str, Any]) -> list[dict[str, Any]]:
    return [planet_dict(p) for p in (obs.get("planets") or [])]


def fleets(obs: dict[str, Any]) -> list[dict[str, Any]]:
    return [fleet_dict(f) for f in (obs.get("fleets") or [])]


def action_rows(step: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for player, entry in enumerate(step):
        for move in entry.get("action") or []:
            if len(move) >= 3:
                rows.append([player, int(move[0]), float(move[1]), int(move[2])])
    return rows


def obs_at(data: dict[str, Any], idx: int) -> dict[str, Any]:
    """Return a populated observation for the frame.

    Most replays have identical board observations for each player, but step 0
    can be empty.  Prefer the first agent observation that has planets.
    """
    for entry in data["steps"][idx]:
        obs = entry.get("observation") or {}
        if obs.get("planets"):
            return obs
    return data["steps"][idx][0].get("observation") or {}


def first_last(data: dict[str, Any]) -> tuple[int, int]:
    populated = [
        i for i in range(len(data["steps"])) if (obs_at(data, i).get("planets") or [])
    ]
    if not populated:
        return 0, len(data["steps"]) - 1
    return populated[0], populated[-1]


def score(obs: dict[str, Any], nplayers: int) -> tuple[list[int], list[int], list[int]]:
    totals = [0.0] * nplayers
    production = [0.0] * nplayers
    counts = [0] * nplayers
    for p in planets(obs):
        owner = p["owner"]
        if 0 <= owner < nplayers:
            totals[owner] += p["ships"]
            production[owner] += p["production"]
            counts[owner] += 1
    for f in fleets(obs):
        owner = f["owner"]
        if 0 <= owner < nplayers:
            totals[owner] += f["ships"]
    return [int(x) for x in totals], [int(x) for x in production], counts


def infer_target(
    planet_by_id: dict[int, dict[str, Any]],
    from_id: int,
    angle: float,
    ships: int,
) -> dict[str, Any]:
    src = planet_by_id.get(from_id)
    if src is None:
        return {"target_id": None, "confidence": "missing_source"}

    ux, uy = math.cos(angle), math.sin(angle)
    sx = src["x"] + ux * (src["radius"] + LAUNCH_CLEARANCE)
    sy = src["y"] + uy * (src["radius"] + LAUNCH_CLEARANCE)
    ray_end_x = sx + ux * MAX_RAY
    ray_end_y = sy + uy * MAX_RAY

    best_hit: tuple[float, dict[str, Any], float] | None = None
    best_angle: tuple[float, float, dict[str, Any], float] | None = None

    for p in planet_by_id.values():
        if p["id"] == from_id:
            continue
        vx, vy = p["x"] - sx, p["y"] - sy
        projection = vx * ux + vy * uy
        if projection <= 0:
            continue
        closest_x = sx + ux * projection
        closest_y = sy + uy * projection
        perpendicular = dist(p["x"], p["y"], closest_x, closest_y)
        forward_dist = max(0.0, projection - p["radius"])

        if perpendicular <= p["radius"] + 0.75:
            if best_hit is None or forward_dist < best_hit[0]:
                best_hit = (forward_dist, p, perpendicular)

        bearing = math.atan2(p["y"] - sy, p["x"] - sx)
        ad = angle_delta(angle, bearing)
        if ad <= 0.22:
            key = ad * 40.0 + forward_dist / 25.0
            if best_angle is None or key < best_angle[0]:
                best_angle = (key, forward_dist, p, ad)

    chosen: tuple[float, dict[str, Any], float, str] | None = None
    if best_hit is not None:
        chosen = (best_hit[0], best_hit[1], best_hit[2], "ray")
    elif best_angle is not None:
        chosen = (best_angle[1], best_angle[2], best_angle[3], "angle")

    if chosen is None:
        return {
            "target_id": None,
            "target_owner": None,
            "target_ships": None,
            "target_prod": None,
            "distance": None,
            "eta": None,
            "confidence": "unknown",
            "ray_end_x": ray_end_x,
            "ray_end_y": ray_end_y,
        }

    distance, target, miss, confidence = chosen
    eta = math.ceil(distance / fleet_speed(ships))
    return {
        "target_id": target["id"],
        "target_owner": target["owner"],
        "target_ships": target["ships"],
        "target_prod": target["production"],
        "distance": round(distance, 1),
        "eta": int(eta),
        "confidence": confidence,
        "miss": round(miss, 3),
    }


def build_launch_log(data: dict[str, Any]) -> list[dict[str, Any]]:
    first, last = first_last(data)
    rows: list[dict[str, Any]] = []
    for idx in range(first, last + 1):
        # Kaggle replay rows store the action that produced the current frame,
        # so use the previous populated frame as the decision state.  Without
        # this, sources look "overdrawn" because ships have already launched.
        decision_idx = max(first, idx - 1)
        obs = obs_at(data, decision_idx)
        pmap = {p["id"]: p for p in planets(obs)}
        for player, from_id, angle, ships in action_rows(data["steps"][idx]):
            src = pmap.get(from_id)
            target = infer_target(pmap, from_id, angle, ships)
            rows.append(
                {
                    "step": idx,
                    "decision_step": decision_idx,
                    "player": player,
                    "from_id": from_id,
                    "from_ships": src["ships"] if src else None,
                    "from_prod": src["production"] if src else None,
                    "ships": ships,
                    "angle": round(angle, 4),
                    **target,
                }
            )
    return rows


def owner_changes(data: dict[str, Any]) -> list[dict[str, Any]]:
    first, last = first_last(data)
    changes: list[dict[str, Any]] = []
    prev = {p["id"]: p for p in planets(obs_at(data, first))}
    for idx in range(first + 1, last + 1):
        cur = {p["id"]: p for p in planets(obs_at(data, idx))}
        for pid, p in cur.items():
            old = prev.get(pid)
            if old is not None and old["owner"] != p["owner"]:
                changes.append(
                    {
                        "step": idx,
                        "planet_id": pid,
                        "old_owner": old["owner"],
                        "new_owner": p["owner"],
                        "ships_after": p["ships"],
                        "prod": p["production"],
                        "x": round(p["x"], 1),
                        "y": round(p["y"], 1),
                    }
                )
        prev = cur
    return changes


def attach_likely_causes(
    changes: list[dict[str, Any]],
    launches: list[dict[str, Any]],
) -> None:
    by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for launch in launches:
        tid = launch.get("target_id")
        if tid is not None:
            by_target[int(tid)].append(launch)
    for change in changes:
        candidates = []
        for launch in by_target.get(int(change["planet_id"]), []):
            eta = launch.get("eta")
            if eta is None:
                continue
            arrival_error = abs((int(change["step"]) - int(launch["step"])) - int(eta))
            if launch["player"] == change["new_owner"] and arrival_error <= 5:
                candidates.append((arrival_error, -int(launch["ships"]), launch))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]["step"], item[2]["from_id"]))
        if candidates:
            change["cause"] = candidates[0][2]


def print_report(data: dict[str, Any], launches: list[dict[str, Any]], changes: list[dict[str, Any]], args: argparse.Namespace) -> None:
    nplayers = len(data["steps"][0])
    first, last = first_last(data)
    info = data.get("info") or {}
    print(f"# Replay detail: {info.get('EpisodeId') or Path(args.replay).name}")
    print()
    print(f"- file: `{args.replay}`")
    print(f"- players: {nplayers}")
    print(f"- rewards: {data.get('rewards')}")
    print(f"- statuses: {data.get('statuses')}")
    if info.get("TeamNames"):
        print(f"- teams: {info.get('TeamNames')}")
    print(f"- frames: {first}..{last} ({len(data['steps'])} raw steps)")
    print()

    obs = obs_at(data, last)
    totals, prod, counts = score(obs, nplayers)
    print("## Final observed")
    for p in range(nplayers):
        print(f"- p{p}: total={totals[p]} prod={prod[p]} planets={counts[p]}")
    print()

    print("## Timeline")
    timeline_steps = sorted(set([first, last] + list(range(first, last + 1, args.timeline_step))))
    print("| step | " + " | ".join(f"p{p} total/prod/pl" for p in range(nplayers)) + " |")
    print("|---:|" + "|".join(["---:"] * nplayers) + "|")
    for idx in timeline_steps:
        totals, prod, counts = score(obs_at(data, idx), nplayers)
        cells = [f"{totals[p]}/{prod[p]}/{counts[p]}" for p in range(nplayers)]
        print(f"| {idx} | " + " | ".join(cells) + " |")
    print()

    print("## Launch summary")
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in launches:
        by_player[int(row["player"])].append(row)
    for p in range(nplayers):
        rows = by_player.get(p, [])
        first_launch = rows[0]["step"] if rows else None
        total_ships = sum(int(r["ships"]) for r in rows)
        big = sorted(rows, key=lambda r: int(r["ships"]), reverse=True)[:5]
        big_s = ", ".join(
            f"s{r['step']} {r['ships']} p{r['from_id']}→p{r.get('target_id')} eta{r.get('eta')}"
            for r in big
        )
        print(f"- p{p}: launches={len(rows)} first={first_launch} ships={total_ships}; biggest: {big_s}")
    print()

    print(f"## Important launch log (ships >= {args.min_ships}, first {args.max_action_step} steps)")
    print("| step | player | from | send | target | target owner/ships/prod | dist | eta | conf |")
    print("|---:|---:|---:|---:|---:|---|---:|---:|---|")
    shown = 0
    for r in launches:
        if r["step"] > args.max_action_step:
            continue
        if int(r["ships"]) < args.min_ships:
            continue
        print(
            f"| {r['step']} | p{r['player']} | p{r['from_id']}({r.get('from_ships')}) | "
            f"{r['ships']} | p{r.get('target_id')} | "
            f"{r.get('target_owner')}/{r.get('target_ships')}/{r.get('target_prod')} | "
            f"{r.get('distance')} | {r.get('eta')} | {r.get('confidence')} |"
        )
        shown += 1
        if shown >= args.max_actions:
            print(f"| ... | ... | ... | ... | ... | truncated at {args.max_actions} rows | ... | ... | ... |")
            break
    print()

    print("## Owner changes")
    print("| step | planet | old→new | ships_after | prod | likely launch |")
    print("|---:|---:|---|---:|---:|---|")
    for c in changes[: args.max_changes]:
        cause = c.get("cause")
        if cause:
            cause_s = (
                f"p{cause['player']} s{cause['step']} "
                f"p{cause['from_id']}→p{cause.get('target_id')} "
                f"send{cause['ships']} eta{cause.get('eta')}"
            )
        else:
            cause_s = ""
        print(
            f"| {c['step']} | p{c['planet_id']} | {c['old_owner']}→{c['new_owner']} | "
            f"{c['ships_after']} | {c['prod']} | {cause_s} |"
        )
    if len(changes) > args.max_changes:
        print(f"| ... | ... | ... | ... | ... | {len(changes) - args.max_changes} more |")
    print()

    high_losses: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for c in changes:
        if c["prod"] >= args.high_prod and c["old_owner"] >= 0:
            high_losses[int(c["old_owner"])].append(c)
    print(f"## High-production losses (prod >= {args.high_prod})")
    for p in range(nplayers):
        rows = high_losses.get(p, [])
        if not rows:
            print(f"- p{p}: none")
            continue
        top = "; ".join(
            f"s{r['step']} p{r['planet_id']} prod{r['prod']} to p{r['new_owner']} after={r['ships_after']}"
            for r in rows[:12]
        )
        print(f"- p{p}: {len(rows)} losses — {top}")


def write_csv(path: Path, launches: list[dict[str, Any]], changes: list[dict[str, Any]]) -> None:
    launch_path = path.with_suffix(".launches.csv")
    change_path = path.with_suffix(".changes.csv")
    if launches:
        with launch_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in launches for k in r}))
            writer.writeheader()
            writer.writerows(launches)
    if changes:
        flat_changes: list[dict[str, Any]] = []
        for c in changes:
            row = dict(c)
            cause = row.pop("cause", None)
            if cause:
                for k, v in cause.items():
                    row[f"cause_{k}"] = v
            flat_changes.append(row)
        with change_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in flat_changes for k in r}))
            writer.writeheader()
            writer.writerows(flat_changes)
    print(f"\nCSV written: {launch_path} {change_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", help="Kaggle/lab replay JSON")
    p.add_argument("--min-ships", type=int, default=18)
    p.add_argument("--max-action-step", type=int, default=140)
    p.add_argument("--max-actions", type=int, default=120)
    p.add_argument("--max-changes", type=int, default=120)
    p.add_argument("--timeline-step", type=int, default=20)
    p.add_argument("--high-prod", type=int, default=3)
    p.add_argument("--csv-prefix", help="write launch/change CSVs with this output prefix")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.replay).read_text())
    launches = build_launch_log(data)
    changes = owner_changes(data)
    attach_likely_causes(changes, launches)
    print_report(data, launches, changes, args)
    if args.csv_prefix:
        write_csv(Path(args.csv_prefix), launches, changes)


if __name__ == "__main__":
    main()
