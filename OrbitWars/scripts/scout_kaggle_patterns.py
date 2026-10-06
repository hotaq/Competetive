#!/usr/bin/env python3
"""Scout Orbit Wars Kaggle replays and classify opponent patterns.

This is an offline pre-submit tool: it can fetch the newest episodes for a
submission id via orbit-wars-lab's scraper, then summarize each opponent's
opening style and the counters our bot should encode as general in-game rules.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / ".omx/research/upstream/orbit-wars-lab"
REPLAYS_ROOT = LAB_DIR / "replays"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(LAB_DIR))

import analyze_replay_detail as ard  # noqa: E402


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_time(value: str | None) -> str:
    return value or ""


def _episode_id(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        return _safe_int(stem.split("_", 1)[1])
    return 0


def load_metadata(submission_id: int, replays_root: Path) -> dict[int, dict[str, Any]]:
    meta_path = replays_root / "kaggle" / str(submission_id) / "_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        rows = json.loads(meta_path.read_text())
    except Exception:
        return {}
    result = {}
    for row in rows if isinstance(rows, list) else []:
        eid = _safe_int(row.get("id"))
        if eid:
            result[eid] = row
    return result


def maybe_fetch(submission_id: int, count: int, replays_root: Path) -> None:
    if count <= 0:
        return
    from orbit_wars_app import kaggle_scraper  # type: ignore

    job = kaggle_scraper.scrape_submission(
        submission_id=submission_id,
        count=count,
        replays_root=replays_root,
    )
    if job.status != "completed":
        raise SystemExit(f"scrape failed: {job.error or job.status}")


def replay_paths(submission_id: int, replays_root: Path, limit: int) -> list[Path]:
    base = replays_root / "kaggle" / str(submission_id)
    paths = sorted(
        base.glob("episode_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return paths[:limit] if limit > 0 else paths


def infer_my_index(meta: dict[str, Any] | None, submission_id: int, data: dict[str, Any]) -> int | None:
    if meta:
        for idx, agent in enumerate(meta.get("agents") or []):
            if _safe_int(agent.get("submissionId")) == submission_id:
                return _safe_int(agent.get("index"), idx)
    # Fallback: if exactly one team is named hotaq, use it.  This is only a
    # local convenience and is not used by the submitted bot.
    names = data.get("info", {}).get("TeamNames") or []
    hotaq = [i for i, name in enumerate(names) if str(name).lower() == "hotaq"]
    return hotaq[0] if len(hotaq) == 1 else None


def base_planet_count(data: dict[str, Any]) -> int:
    first, _ = ard.first_last(data)
    obs = ard.obs_at(data, first)
    comet_ids = set(obs.get("comet_planet_ids") or [])
    initial = obs.get("initial_planets") or []
    if initial:
        return len([p for p in initial if int(p[0]) not in comet_ids])
    return len([p for p in obs.get("planets", []) if int(p[0]) not in comet_ids])


def comet_ids_seen(data: dict[str, Any]) -> set[int]:
    first, last = ard.first_last(data)
    ids: set[int] = set()
    for idx in range(first, last + 1):
        obs = ard.obs_at(data, idx)
        ids.update(int(x) for x in (obs.get("comet_planet_ids") or []))
    return ids


def owner_changes(data: dict[str, Any]) -> list[dict[str, Any]]:
    first, last = ard.first_last(data)
    changes: list[dict[str, Any]] = []
    prev: dict[int, dict[str, Any]] = {}
    for idx in range(first, last + 1):
        obs = ard.obs_at(data, idx)
        cur = {p["id"]: p for p in ard.planets(obs)}
        if prev:
            for pid, p in cur.items():
                old = prev.get(pid)
                if old and int(old["owner"]) != int(p["owner"]):
                    changes.append(
                        {
                            "step": idx,
                            "planet_id": pid,
                            "old_owner": int(old["owner"]),
                            "new_owner": int(p["owner"]),
                            "ships": int(p["ships"]),
                            "production": int(p["production"]),
                        }
                    )
        prev = cur
    return changes


def timeline_at(data: dict[str, Any], nplayers: int, checkpoints: tuple[int, ...]) -> dict[int, tuple[list[int], list[int], list[int]]]:
    first, last = ard.first_last(data)
    result = {}
    for cp in checkpoints:
        idx = min(last, max(first, cp))
        result[cp] = ard.score(ard.obs_at(data, idx), nplayers)
    return result


def classify_player(
    player: int,
    data: dict[str, Any],
    launches: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    nplayers: int,
    my_idx: int | None,
) -> dict[str, Any]:
    comet_ids = comet_ids_seen(data)
    my_launches = [r for r in launches if int(r["player"]) == player]
    early = [r for r in my_launches if int(r["step"]) <= 80]
    mid = [r for r in my_launches if 80 < int(r["step"]) <= 180]
    all_targeted = [r for r in my_launches if r.get("target_id") is not None]

    def is_hostile(row: dict[str, Any]) -> bool:
        owner = row.get("target_owner")
        return owner not in (None, -1, player)

    def is_neutral(row: dict[str, Any]) -> bool:
        return row.get("target_owner") == -1

    def is_friendly(row: dict[str, Any]) -> bool:
        return row.get("target_owner") == player

    early_hostile = [r for r in early if is_hostile(r)]
    early_neutral = [r for r in early if is_neutral(r)]
    early_friendly = [r for r in early if is_friendly(r)]
    early_long = [r for r in early if (r.get("eta") or 0) >= 24]
    early_heavy_neutral = [
        r for r in early_neutral if (r.get("target_ships") or 0) >= 50
    ]
    comet_launches = [r for r in all_targeted if int(r.get("target_id") or -999) in comet_ids]
    home_raids = [
        r for r in early_hostile
        if (r.get("target_prod") or 0) <= 1 and (r.get("eta") or 999) <= 35
    ]

    caps = [c for c in changes if int(c["new_owner"]) == player]
    early_caps = [c for c in caps if int(c["step"]) <= 100]
    early_neutral_caps = [c for c in early_caps if int(c["old_owner"]) == -1]
    early_hostile_caps = [c for c in early_caps if int(c["old_owner"]) not in (-1, player)]

    fresh_recaps = 0
    last_change_by_planet: dict[int, dict[str, Any]] = {}
    for change in sorted(changes, key=lambda c: int(c["step"])):
        prev = last_change_by_planet.get(int(change["planet_id"]))
        if (
            int(change["new_owner"]) == player
            and prev
            and int(prev["new_owner"]) not in (-1, player)
            and int(change["step"]) - int(prev["step"]) <= 18
            and int(change["production"]) >= 2
        ):
            fresh_recaps += 1
        last_change_by_planet[int(change["planet_id"])] = change

    etas = [float(r["eta"]) for r in early if r.get("eta") is not None]
    dists = [float(r["distance"]) for r in early if r.get("distance") is not None]
    sends = [int(r["ships"]) for r in early]
    first_launch = min([int(r["step"]) for r in my_launches], default=None)

    final_totals, final_prod, final_counts = ard.score(ard.obs_at(data, ard.first_last(data)[1]), nplayers)
    checkpoints = timeline_at(data, nplayers, (30, 60, 90, 150))

    labels: list[str] = []
    bpc = base_planet_count(data)
    if nplayers >= 4 and bpc <= 24:
        labels.append("scarce_4p")
    if early and len(early_neutral_caps) >= 3 and (statistics.mean(etas) if etas else 99) <= 16:
        labels.append("nearby_expander")
    if early and len(early_hostile) / max(1, len(early)) >= 0.30:
        labels.append("hostile_rusher")
    if early and len(early_long) / max(1, len(early)) >= 0.28:
        labels.append("long_crossmap")
    if len(early_heavy_neutral) >= 2:
        labels.append("heavy_neutral_chaser")
    if len(home_raids) >= 1:
        labels.append("home_raider")
    if len(comet_launches) / max(1, len(all_targeted)) >= 0.18:
        labels.append("comet_chaser")
    if fresh_recaps >= 2:
        labels.append("fresh_recapture_sniper")
    if len(early) <= 3 and checkpoints[90][1][player] >= 10:
        labels.append("low_action_boomer")
    if not labels:
        labels.append("balanced")

    counters = recommended_counters(labels)

    return {
        "player": player,
        "labels": labels,
        "counters": counters,
        "first_launch": first_launch,
        "early_launches": len(early),
        "mid_launches": len(mid),
        "early_ships_launched": sum(sends),
        "avg_early_eta": round(statistics.mean(etas), 2) if etas else None,
        "avg_early_dist": round(statistics.mean(dists), 2) if dists else None,
        "hostile_launch_ratio_80": round(len(early_hostile) / max(1, len(early)), 3),
        "neutral_launch_ratio_80": round(len(early_neutral) / max(1, len(early)), 3),
        "friendly_launch_ratio_80": round(len(early_friendly) / max(1, len(early)), 3),
        "long_eta_ratio_80": round(len(early_long) / max(1, len(early)), 3),
        "heavy_neutral_launches_80": len(early_heavy_neutral),
        "home_raids_80": len(home_raids),
        "comet_launches": len(comet_launches),
        "fresh_recaptures": fresh_recaps,
        "caps_100": len(early_caps),
        "neutral_caps_100": len(early_neutral_caps),
        "hostile_caps_100": len(early_hostile_caps),
        "prod_60": checkpoints[60][1][player],
        "planets_60": checkpoints[60][2][player],
        "prod_90": checkpoints[90][1][player],
        "planets_90": checkpoints[90][2][player],
        "final_total": final_totals[player],
        "final_prod": final_prod[player],
        "final_planets": final_counts[player],
        "is_us": my_idx == player if my_idx is not None else False,
    }


def recommended_counters(labels: list[str]) -> list[str]:
    counters: list[str] = []
    labelset = set(labels)
    if "scarce_4p" in labelset:
        counters.append("scarce map: avoid long early wars; prioritize compact high-prod ring")
    if "nearby_expander" in labelset:
        counters.append("race close neutrals first; then attack exposed ring anchors, not far homes")
    if "hostile_rusher" in labelset or "home_raider" in labelset:
        counters.append("raise early home/frontier reserve; punish their fresh captures")
    if "long_crossmap" in labelset:
        counters.append("do not chase cross-map; reinforce target and counter-attack source while fleet in transit")
    if "heavy_neutral_chaser" in labelset:
        counters.append("let them spend into high-ship neutrals; snipe after capture with hold margin")
    if "comet_chaser" in labelset:
        counters.append("discount comet bait; attack planets emptied by comet launches")
    if "fresh_recapture_sniper" in labelset:
        counters.append("capture with bigger surplus/second wave; keep prod>=3 anchors guarded")
    if "low_action_boomer" in labelset:
        counters.append("greedy expand early, then pressure before their late stack")
    if not counters:
        counters.append("balanced: use default planner")
    return counters


def analyze_submission(submission_id: int, limit: int, fetch_count: int, out_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    maybe_fetch(submission_id, fetch_count, REPLAYS_ROOT)
    meta = load_metadata(submission_id, REPLAYS_ROOT)
    paths = replay_paths(submission_id, REPLAYS_ROOT, limit)
    rows = []
    one_v_one_losses = []
    opponent_rollup: dict[str, dict[str, Any]] = {}

    for path in paths:
        data = json.loads(path.read_text())
        eid = _episode_id(path)
        ep_meta = meta.get(eid)
        nplayers = len(data.get("steps", [[]])[0]) if data.get("steps") else 0
        if nplayers <= 0:
            continue
        my_idx = infer_my_index(ep_meta, submission_id, data)
        names = data.get("info", {}).get("TeamNames") or [f"p{i}" for i in range(nplayers)]
        launches = ard.build_launch_log(data)
        changes = owner_changes(data)
        rewards = data.get("rewards") or [None] * nplayers
        winner_idx = None
        if rewards and any(r == 1 for r in rewards):
            winner_idx = list(rewards).index(1)
        profiles = [classify_player(i, data, launches, changes, nplayers, my_idx) for i in range(nplayers)]
        for prof in profiles:
            prof["name"] = names[prof["player"]] if prof["player"] < len(names) else f"p{prof['player']}"
            prof["reward"] = rewards[prof["player"]] if prof["player"] < len(rewards) else None
        rows.append(
            {
                "episode_id": eid,
                "endTime": _parse_time(ep_meta.get("endTime") if ep_meta else None),
                "type": ep_meta.get("type") if ep_meta else None,
                "nplayers": nplayers,
                "base_planets": base_planet_count(data),
                "my_idx": my_idx,
                "winner_idx": winner_idx,
                "winner_name": names[winner_idx] if winner_idx is not None and winner_idx < len(names) else None,
                "profiles": profiles,
            }
        )
        if nplayers == 2 and my_idx is not None and 0 <= my_idx < 2:
            my_prof = profiles[my_idx]
            my_reward = my_prof.get("reward")
            if my_reward != 1:
                opp_idx = 1 - my_idx
                opp_prof = profiles[opp_idx]
                my_high_losses = [
                    c for c in changes
                    if int(c["old_owner"]) == my_idx and int(c["production"]) >= 3
                ]
                opp_high_losses = [
                    c for c in changes
                    if int(c["old_owner"]) == opp_idx and int(c["production"]) >= 3
                ]
                checkpoints = timeline_at(data, nplayers, (30, 60, 90, 120, 150, 180, 220, 260))
                collapse_step = None
                for cp, (_totals, prod, counts) in checkpoints.items():
                    if prod[opp_idx] - prod[my_idx] >= 10 or counts[opp_idx] - counts[my_idx] >= 4:
                        collapse_step = cp
                        break
                notes = []
                if len(my_high_losses) - len(opp_high_losses) >= 4:
                    notes.append("high_prod_anchor_bleed")
                if opp_prof["fresh_recaptures"] > my_prof["fresh_recaptures"]:
                    notes.append("opponent_recaptures_more")
                if opp_prof["early_ships_launched"] > my_prof["early_ships_launched"] * 1.18:
                    notes.append("opponent_higher_opening_tempo")
                if opp_prof["long_eta_ratio_80"] >= 0.28:
                    notes.append("long_crossmap_pressure")
                if not notes:
                    notes.append("tempo_midgame_loss")
                one_v_one_losses.append({
                    "episode_id": eid,
                    "endTime": _parse_time(ep_meta.get("endTime") if ep_meta else None),
                    "opponent": names[opp_idx] if opp_idx < len(names) else f"p{opp_idx}",
                    "winner": names[winner_idx] if winner_idx is not None and winner_idx < len(names) else None,
                    "my_player": my_idx,
                    "opponent_player": opp_idx,
                    "my_labels": my_prof["labels"],
                    "opponent_labels": opp_prof["labels"],
                    "my_prod90": my_prof["prod_90"],
                    "opp_prod90": opp_prof["prod_90"],
                    "my_final": my_prof["final_total"],
                    "opp_final": opp_prof["final_total"],
                    "my_early_launches": my_prof["early_launches"],
                    "opp_early_launches": opp_prof["early_launches"],
                    "my_early_ships": my_prof["early_ships_launched"],
                    "opp_early_ships": opp_prof["early_ships_launched"],
                    "my_high_prod_losses": len(my_high_losses),
                    "opp_high_prod_losses": len(opp_high_losses),
                    "collapse_step": collapse_step,
                    "notes": notes,
                    "recommended_counter": recommended_counters(list(set(opp_prof["labels"] + notes))),
                })
        for prof in profiles:
            if prof.get("is_us"):
                continue
            name = str(prof["name"])
            roll = opponent_rollup.setdefault(name, {"episodes": 0, "labels": Counter(), "counters": Counter(), "wins": 0, "losses_vs_us": 0, "features": defaultdict(list)})
            roll["episodes"] += 1
            roll["labels"].update(prof["labels"])
            roll["counters"].update(prof["counters"])
            if prof.get("reward") == 1:
                roll["wins"] += 1
            if prof.get("reward") == -1:
                roll["losses_vs_us"] += 1
            for key in ["early_launches", "hostile_launch_ratio_80", "long_eta_ratio_80", "neutral_caps_100", "hostile_caps_100", "prod_90", "planets_90", "final_total"]:
                val = prof.get(key)
                if isinstance(val, (int, float)):
                    roll["features"][key].append(val)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "submission_id": submission_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "replays_root": str(REPLAYS_ROOT.relative_to(ROOT)),
        "episodes": rows,
        "one_v_one_losses": one_v_one_losses,
        "opponents": {},
    }
    for name, roll in opponent_rollup.items():
        payload["opponents"][name] = {
            "episodes": roll["episodes"],
            "wins": roll["wins"],
            "losses": roll["losses_vs_us"],
            "labels": dict(roll["labels"].most_common()),
            "top_counters": [x for x, _ in roll["counters"].most_common(5)],
            "features_avg": {
                key: round(statistics.mean(vals), 3)
                for key, vals in roll["features"].items()
                if vals
            },
        }

    json_path = out_dir / f"{submission_id}.pattern_scout.json"
    md_path = out_dir / f"{submission_id}.pattern_scout.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path, payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Orbit Wars Pattern Scout — submission {payload['submission_id']}",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Opponent rollup",
        "",
        "| opponent | eps | wins | common labels | counter | avg early launches | avg prod90 |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for name, opp in sorted(payload["opponents"].items(), key=lambda kv: (-kv[1]["wins"], kv[0].lower())):
        labels = ", ".join(f"{k}×{v}" for k, v in list(opp["labels"].items())[:4])
        counters = "; ".join(opp["top_counters"][:2])
        feats = opp.get("features_avg", {})
        lines.append(
            f"| {name} | {opp['episodes']} | {opp['wins']} | {labels} | {counters} | {feats.get('early_launches', '')} | {feats.get('prod_90', '')} |"
        )
    if payload.get("one_v_one_losses"):
        lines += [
            "",
            "## 1v1 losses to study first",
            "",
            "| episode | opponent | final us/opp | prod90 us/opp | early ships us/opp | high-prod losses us/opp | collapse | notes | counter |",
            "|---:|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for loss in payload["one_v_one_losses"]:
            notes = ", ".join(loss.get("notes") or [])
            counters = "; ".join((loss.get("recommended_counter") or [])[:2])
            lines.append(
                f"| {loss['episode_id']} | {loss['opponent']} | "
                f"{loss['my_final']}/{loss['opp_final']} | "
                f"{loss['my_prod90']}/{loss['opp_prod90']} | "
                f"{loss['my_early_ships']}/{loss['opp_early_ships']} | "
                f"{loss['my_high_prod_losses']}/{loss['opp_high_prod_losses']} | "
                f"{loss.get('collapse_step') or ''} | {notes} | {counters} |"
            )
    lines += ["", "## Episode detail", ""]
    for ep in payload["episodes"]:
        lines.append(
            f"### Episode {ep['episode_id']} — {ep['nplayers']}p, base_planets={ep['base_planets']}, winner={ep['winner_name']}"
        )
        if ep.get("endTime"):
            lines.append(f"- endTime: {ep['endTime']}")
        lines.append("| p | name | us | reward | labels | early L | host% | long% | caps100 | prod90/pl90 | final |")
        lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for prof in ep["profiles"]:
            labels = ", ".join(prof["labels"])
            caps = f"{prof['neutral_caps_100']}N/{prof['hostile_caps_100']}H"
            prodpl = f"{prof['prod_90']}/{prof['planets_90']}"
            lines.append(
                f"| p{prof['player']} | {prof['name']} | {'Y' if prof['is_us'] else ''} | {prof['reward']} | {labels} | {prof['early_launches']} | {prof['hostile_launch_ratio_80']} | {prof['long_eta_ratio_80']} | {caps} | {prodpl} | {prof['final_total']} |"
            )
        lines.append("")
    lines += [
        "## How to translate to agent counters",
        "",
        "- Runtime submitted bot cannot read Kaggle names/submission ids during a match, so use this scout to choose **general behavior detectors**.",
        "- Good counters to encode: scarce_4p compact opening, heavy-neutral snipe-after-capture, comet-chaser exposed-source punish, hostile-rusher early reserve.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id", type=int)
    parser.add_argument("--fetch-count", type=int, default=0, help="download up to N missing latest episodes first")
    parser.add_argument("--limit", type=int, default=30, help="analyze newest N local replays; 0=all")
    parser.add_argument("--out-dir", type=Path, default=ROOT / ".omx/research/opponent_profiles")
    args = parser.parse_args()
    json_path, md_path, payload = analyze_submission(args.submission_id, args.limit, args.fetch_count, args.out_dir)
    print(f"json={json_path}")
    print(f"report={md_path}")
    print(f"episodes={len(payload['episodes'])} opponents={len(payload['opponents'])}")
    top = sorted(payload["opponents"].items(), key=lambda kv: (-kv[1]["wins"], kv[0].lower()))[:5]
    for name, opp in top:
        print(f"- {name}: wins={opp['wins']} labels={opp['labels']} counter={opp['top_counters'][:2]}")


if __name__ == "__main__":
    main()
