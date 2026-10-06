"""Ask whether observable state at step 360 predicts how Q3 turns out.

Every previous selection experiment branched at step 26, 120 or 145 and failed on
identifiability. `analysis/live_losses_20260830/REPORT.md` locates the decision in
steps 360-540, where Q3 market state is observed rather than forecast. This script
replays live tapes to parity, reads the state at step 360, and tests whether any
observable separates the games where our fixed 400-unit sell schedule realises ~26
per unit from the ones where it realises ~136.

Replay convention: in a Kaggle replay, `steps[i].observation` is the state *after*
`steps[i].action`, so the action taken from state i is `actions[i + 1]`. Replaying
without that offset does not reproduce the episode.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import normalize_legacy_action
from kaggriculture_rs import Engine

ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "EGG", "MILK", "WOOL", "FERTILIZER")
MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
# Arity from ACTION_SPACE.md. PICKUP and PLACE take an optional count defaulting to 1.
NO_ARG_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER", "HARVEST",
              "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE",
              "COLLECT_FERTILIZER")
DECISION_STEP = 360
SNAPSHOT_STEPS = (120, 180, 240, 300, 360)
Q3_END = 540
PARITY_TOLERANCE = 0.02


def clean(action: Any) -> dict[str, Any]:
    """Drop commands the live environment treats as a silent no-op.

    Recorded tapes contain orders the engine binding rejects outright — `NOOP` and
    `PASS` used as market operations, and non-integer quantities. Live, these are
    ignored, so they are stripped here rather than allowed to abort a replay.
    """
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def quantity_ok(order: list[Any]) -> bool:
        if len(order) < 3 or not isinstance(order[1], str):
            return False
        quantity = order[2]
        if isinstance(quantity, str) and quantity.lstrip("-").isdigit():
            quantity = int(quantity)
        return isinstance(quantity, int) and quantity > 0

    orders = []
    for order in action.get("market") or []:
        if not (isinstance(order, list) and order and isinstance(order[0], str)):
            continue
        if order[0] in ("HIRE", "BUY_LAND"):
            orders.append([order[0]])
        elif order[0] in MARKET_OPS and quantity_ok(order):
            orders.append([order[0], order[1], int(order[2])])

    def unit(command: Any) -> list[Any]:
        """Coerce to the documented arity; anything else becomes the live no-op."""
        if not (isinstance(command, list) and command and isinstance(command[0], str)):
            return ["PASS"]
        operation, *arguments = command
        coerced: list[Any] = []
        for argument in arguments:
            if isinstance(argument, str) and argument.lstrip("-").isdigit():
                coerced.append(int(argument))
            elif isinstance(argument, (str, int)):
                coerced.append(argument)
            else:
                return ["PASS"]
        if operation in NO_ARG_OPS:
            return [operation]
        if operation == "PLANT":
            return [operation, coerced[0]] if coerced and isinstance(coerced[0], str) else ["PASS"]
        if operation in ("PICKUP", "PLACE"):
            if not coerced or not isinstance(coerced[0], str):
                return ["PASS"]
            count = coerced[1] if len(coerced) > 1 and isinstance(coerced[1], int) else 1
            return [operation, coerced[0], count]
        return ["PASS"]

    return {
        "farmer": unit(action.get("farmer")),
        "hands": [unit(command) for command in action.get("hands") or []],
        "market": orders,
    }


def replay(seed: int, seat: int, ours: list[Any], theirs: list[Any]) -> dict[str, Any]:
    engine = Engine({"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005})
    snapshots: dict[int, dict[str, float]] = {}
    money = {"ours": [], "theirs": []}
    for step in range(719):
        state_money = [farm["money"] for farm in engine.state["farms"]]
        money["ours"].append(state_money[seat])
        money["theirs"].append(state_money[1 - seat])
        if step in SNAPSHOT_STEPS:
            snapshots[step] = features(engine.observation(seat), seat)
        index = step + 1
        if index >= len(ours) or index >= len(theirs):
            break
        actions: list[Any] = [None, None]
        actions[seat] = engine.sanitize_action(
            seat, normalize_legacy_action(clean(ours[index])))
        actions[1 - seat] = engine.sanitize_action(
            1 - seat, normalize_legacy_action(clean(theirs[index])))
        engine.step(actions)
    final = [farm["money"] for farm in engine.state["farms"]]
    return {"snapshots": snapshots, "money": money,
            "final_ours": final[seat], "final_theirs": final[1 - seat]}


def features(observation: dict[str, Any], seat: int) -> dict[str, float]:
    market = observation["market"]
    private = observation["private"]
    farm = observation["farms"][seat]
    other = observation["farms"][1 - seat]
    tiles = farm["tiles"]
    plants = [t for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PLANT"]
    animals = [t for row in tiles for t in row
               if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and t.get("animal")]
    out: dict[str, float] = {}
    for item in ITEMS:
        out[f"price_{item}"] = float(market["prices"].get(item, 0))
        out[f"stock_{item}"] = float(market["inventory"].get(item, 0))
        out[f"shed_{item}"] = float(private["shed"].get(item, 0))
    out["price_basket_sellable"] = sum(
        float(market["prices"].get(i, 0)) for i in ("STRAWBERRY", "MILK", "WOOL", "FERTILIZER"))
    out["shed_basket_sellable"] = sum(
        float(private["shed"].get(i, 0)) for i in ("STRAWBERRY", "MILK", "WOOL", "FERTILIZER"))
    out["market_stock_total"] = sum(float(market["inventory"].get(i, 0)) for i in ITEMS)
    out["our_money"] = float(farm["money"])
    out["their_money"] = float(other["money"])
    out["money_lead"] = out["our_money"] - out["their_money"]
    out["our_hands"] = float(len(farm["hands"]))
    out["their_hands"] = float(len(other["hands"]))
    out["our_plants"] = float(len(plants))
    out["our_animals"] = float(len(animals))
    out["our_quadrants"] = float(len(farm["unlocked_quadrants"]))
    out["their_quadrants"] = float(len(other["unlocked_quadrants"]))
    return out


def q3_units(actions: list[Any]) -> int:
    total = 0
    for action in actions[DECISION_STEP + 1:Q3_END + 1]:
        if not isinstance(action, dict):
            continue
        for order in action.get("market") or []:
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                total += int(order[2])
    return total


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def load_pairs(root: Path) -> list[dict[str, Any]]:
    """Accept both tape layouts: one file holding both seats, or `_ours`/`_opp` pairs."""
    pairs = []
    for path in sorted(root.glob("ep*.json")):
        if path.name.endswith("_opp.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name.endswith("_ours.json"):
            opponent_path = path.with_name(path.name.replace("_ours.json", "_opp.json"))
            if not opponent_path.exists():
                continue
            opponent = json.loads(opponent_path.read_text(encoding="utf-8"))
            pairs.append({
                "episode": payload["episode"], "seed": payload["seed"],
                "seat": payload["seat"],
                "ours": payload["actions"], "theirs": opponent["actions"],
                "live_ours": payload["final_money"],
                "live_theirs": opponent["final_money"],
            })
        elif isinstance(payload.get("actions"), dict):
            pairs.append({
                "episode": payload["episode"], "seed": payload["seed"],
                "seat": payload["our_seat"],
                "ours": payload["actions"]["ours"], "theirs": payload["actions"]["theirs"],
                "live_ours": payload["our_money_final"],
                "live_theirs": payload["opponent_money_final"],
            })
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs: list[dict[str, Any]] = []
    for root in args.tapes_dir:
        pairs.extend(load_pairs(root))

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for pair in pairs:
        result = replay(pair["seed"], pair["seat"], pair["ours"], pair["theirs"])
        drift = abs(result["final_ours"] - pair["live_ours"]) / max(abs(pair["live_ours"]), 1)
        if drift > PARITY_TOLERANCE or len(result["snapshots"]) < len(SNAPSHOT_STEPS):
            rejected.append({"episode": pair["episode"], "parity_drift": drift})
            continue
        money = result["money"]
        gain = money["ours"][Q3_END] - money["ours"][DECISION_STEP]
        units = q3_units(pair["ours"])
        rows.append({
            "episode": pair["episode"], "seed": pair["seed"],
            "parity_drift": drift,
            "q3_units": units,
            "q3_gain": gain,
            "q3_price_per_unit": gain / units if units else 0.0,
            "final_margin": pair["live_ours"] - pair["live_theirs"],
            "snapshots": result["snapshots"],
        })

    print(f"episodes replayed to parity: {len(rows)}  rejected: {len(rejected)}")
    for entry in rejected:
        print(f"  rejected ep{entry['episode']} parity drift {entry['parity_drift']:.1%}")
    if len(rows) < 8:
        print("too few parity episodes to run the test")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"rows": rows, "rejected": rejected}, indent=2))
        return

    ppu = [row["q3_price_per_unit"] for row in rows]
    margin = [float(row["final_margin"]) for row in rows]
    names = sorted(rows[0]["snapshots"][SNAPSHOT_STEPS[0]])

    ranked: dict[int, list[dict[str, Any]]] = {}
    for step in SNAPSHOT_STEPS:
        entries = []
        for name in names:
            xs = [row["snapshots"][step][name] for row in rows]
            entries.append({"feature": name,
                            "rho_ppu": spearman(xs, ppu),
                            "rho_margin": spearman(xs, margin)})
        entries.sort(key=lambda entry: -abs(entry["rho_ppu"]))
        ranked[step] = entries

    print(f"\nQ3 price per unit: min {min(ppu):.0f} median {st.median(ppu):.0f} max {max(ppu):.0f}")
    print(f"\nhow early is the Q3 outcome visible? best |rho| at each snapshot step")
    print(f"{'step':>6}{'best feature vs ppu':>26}{'rho':>8}"
          f"{'best feature vs margin':>28}{'rho':>8}")
    for step in SNAPSHOT_STEPS:
        best_ppu = ranked[step][0]
        best_margin = max(ranked[step], key=lambda entry: abs(entry["rho_margin"]))
        print(f"{step:>6}{best_ppu['feature']:>26}{best_ppu['rho_ppu']:>8.2f}"
              f"{best_margin['feature']:>28}{best_margin['rho_margin']:>8.2f}")

    tracked = ("price_STRAWBERRY", "price_MILK", "price_FERTILIZER", "market_stock_total")
    print(f"\nsame observable tracked across steps (rho vs Q3 price per unit)")
    print(f"{'feature':>20}" + "".join(f"{step:>8}" for step in SNAPSHOT_STEPS))
    for name in tracked:
        cells = []
        for step in SNAPSHOT_STEPS:
            entry = next(e for e in ranked[step] if e["feature"] == name)
            cells.append(f"{entry['rho_ppu']:>8.2f}")
        print(f"{name:>20}" + "".join(cells))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"episodes": len(rows), "rejected": rejected,
         "correlations": {str(k): v for k, v in ranked.items()},
         "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
