"""Mine tomato play from the frozen fresh tape pool (v105, analysis only).

Reads artifacts/tape_pool_20260922_top10/tapes/*.json + manifest.json.
Tape schema: actions[720] of {"farmer": [...], "hands": [[...], ...],
"market": [[...], ...]}.

Observability limits (documented, not worked around):
- PLANT/HARVEST/WATER/DIG/FERTILIZE are BARE unit actions (no crop arg;
  crop resolves by tile position in engine state). Per-crop plant/tile
  counts are NOT recoverable from action tapes. We report total PLANT
  counts as labour context only.
- SELL entries are ["SELL", crop, qty] with NO price. Sale day/qty are
  exact; price/revenue are NOT observable from tapes (manifest gives
  final_money per tape only).
- HIRE entries are ["HIRE"]. Per-day hire counts are exact; which hands
  water/harvest tomato tiles is NOT attributable.

Day convention: day = turn // 24 (matches top10_learning REPORT).
Tomato rule constants (rust/kaggriculture-engine/src/rules.rs + engine.rs):
  seed 50, first_yield_day 8, interval 1, max_yield 4, ongoing=true;
  daily production 2 if (watered that day AND fertilized) else 1;
  tile expires after 4th production (then weed).
"""
import json
import glob
import os
import collections

POOL = "artifacts/tape_pool_20260922_top10/tapes"
MANIFEST = "artifacts/tape_pool_20260922_top10/manifest.json"


def main():
    manifest = json.load(open(MANIFEST))["tapes"]
    files = sorted(glob.glob(os.path.join(POOL, "*.json")))
    assert len(files) == 92, f"pool frozen at 92, found {len(files)}"

    per_tape = []
    pool_seed_buys = collections.Counter()  # day -> units
    pool_sell_qty = collections.Counter()
    for f in files:
        tid = os.path.basename(f)[:-5]
        d = json.load(open(f))
        acts = d["actions"]
        assert len(acts) == 720, tid
        m = manifest[tid]
        seed_events = []  # (turn, day, qty)
        sell_events = []  # (turn, day, qty)
        plants = 0
        harvests = 0
        waters = 0
        digs = 0
        hires_by_day = collections.Counter()
        for t, a in enumerate(acts):
            for mo in a.get("market", []):
                if not mo:
                    continue
                if mo[0] == "BUY_SEED" and len(mo) > 2 and mo[1] == "TOMATO":
                    seed_events.append((t, t // 24, mo[2]))
                    pool_seed_buys[t // 24] += mo[2]
                elif mo[0] == "SELL" and len(mo) > 2 and mo[1] == "TOMATO":
                    sell_events.append((t, t // 24, mo[2]))
                    pool_sell_qty[t // 24] += mo[2]
                elif mo[0] == "HIRE":
                    hires_by_day[t // 24] += 1
            fa = a.get("farmer", [])
            if fa:
                if fa[0] == "PLANT":
                    plants += 1
                elif fa[0] == "HARVEST":
                    harvests += 1
                elif fa[0] == "WATER":
                    waters += 1
                elif fa[0] == "DIG":
                    digs += 1
            for h in a.get("hands", []):
                if h and h[0] == "PLANT":
                    plants += 1
                elif h and h[0] == "HARVEST":
                    harvests += 1
                elif h and h[0] == "WATER":
                    waters += 1
                elif h and h[0] == "DIG":
                    digs += 1
        tot_seed = sum(q for _, _, q in seed_events)
        tot_sell = sum(q for _, _, q in sell_events)
        per_tape.append({
            "tape": tid,
            "team": m["team_name"],
            "final_money": m["final_money"],
            "opp": m["opponent"],
            "opp_money": m["opponent_money"],
            "seed_events": seed_events,
            "total_seeds": tot_seed,
            "first_buy": seed_events[0] if seed_events else None,
            "sell_events": sell_events,
            "total_sold": tot_sell,
            "first_sale": sell_events[0] if sell_events else None,
            "plants_all_crops": plants,
            "harvests": harvests,
            "waters": waters,
            "digs": digs,
            "hires_by_day": dict(sorted(hires_by_day.items())),
            "hires_total": sum(hires_by_day.values()),
        })

    buyers = [p for p in per_tape if p["total_seeds"] > 0]
    nonbuy = [p for p in per_tape if p["total_seeds"] == 0]
    sellers = [p for p in per_tape if p["total_sold"] > 0]

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        return {"n": n, "mean": sum(xs) / n if n else 0,
                "median": xs[n // 2] if n else 0,
                "min": xs[0] if n else 0, "max": xs[-1] if n else 0}

    fb_days = [b["first_buy"][1] for b in buyers]
    fb_turns = [b["first_buy"][0] for b in buyers]
    print("== POOL ==")
    print(f"tapes=92 buyers={len(buyers)} ({100*len(buyers)/92:.1f}%) "
          f"sellers={len(sellers)}")
    print("total_seeds stats (buyers only):", stats([b["total_seeds"] for b in buyers]))
    print("total_seeds stats (all 92):", stats([p["total_seeds"] for p in per_tape]))
    print("first-buy-day stats (buyers):", stats(fb_days))
    print("first-buy-turn stats (buyers):", stats(fb_turns))
    print("seed volume by phase: d0-10=%d d11-19=%d d20-29=%d" % (
        sum(q for d, q in pool_seed_buys.items() if d <= 10),
        sum(q for d, q in pool_seed_buys.items() if 11 <= d <= 19),
        sum(q for d, q in pool_seed_buys.items() if d >= 20)))
    print("seed units by day:", dict(sorted(pool_seed_buys.items())))
    print("sell qty by day:", dict(sorted(pool_sell_qty.items())))
    fs_days = [s["first_sale"][1] for s in sellers]
    print("first-sale-day stats (sellers):", stats(fs_days) if fs_days else "none")
    print("total_sold stats (sellers):", stats([s["total_sold"] for s in sellers]) if sellers else "none")

    # team aggregates
    by_team = collections.defaultdict(list)
    for p in per_tape:
        by_team[p["team"]].append(p)
    print("\n== BY TEAM (mean money | n | tom_mean | tom_med | first_day_mean | sold_mean) ==")
    for team, ps in sorted(by_team.items(), key=lambda kv: -sum(p["final_money"] for p in kv[1]) / len(kv[1])):
        mm = sum(p["final_money"] for p in ps) / len(ps)
        ts = sorted(p["total_seeds"] for p in ps)
        fb = [p["first_buy"][1] for p in ps if p["first_buy"]]
        ss = sum(p["total_sold"] for p in ps) / len(ps)
        print(f"{team} {mm:.0f} | {len(ps)} | {sum(ts)/len(ts):.1f} | {ts[len(ts)//2]} | "
              f"{(sum(fb)/len(fb) if fb else float('nan')):.1f} | {ss:.1f}")

    # money: buyers vs non-buyers
    if buyers and nonbuy:
        mb = sum(p["final_money"] for p in buyers) / len(buyers)
        mn = sum(p["final_money"] for p in nonbuy) / len(nonbuy)
        print(f"\nbuyers mean money={mb:.0f} (n={len(buyers)}) vs non-buyers={mn:.0f} (n={len(nonbuy)})")

    # labour: hires/day distribution + tomato-window hires for buyers
    print("\n== LABOUR ==")
    all_hires_day = collections.Counter()
    for p in per_tape:
        for d, c in p["hires_by_day"].items():
            all_hires_day[d] += c
    for d in range(30):
        print(f"day {d}: total_hires={all_hires_day[d]} avg/tape={all_hires_day[d]/92:.2f}")
    print("\nhires_total stats (all):", stats([p["hires_total"] for p in per_tape]))
    # tomato window: first-buy-day .. first-sale-day (or 29 if never sold)
    win_hires = []
    for b in buyers:
        d0 = b["first_buy"][1]
        d1 = b["first_sale"][1] if b["first_sale"] else 29
        win_hires.append(sum(c for d, c in b["hires_by_day"].items() if d0 <= d <= d1))
    print("buyer window-hires stats:", stats(win_hires) if win_hires else "none")

    # per-tape table (compact, for the report)
    print("\n== PER-TAPE (tape | team | money | seeds | firstbuy d/t | planting span | sold | firstsale d | hires_tot) ==")
    for p in sorted(per_tape, key=lambda p: -p["final_money"]):
        se = p["seed_events"]
        span = f"{se[0][1]}-{se[-1][1]}" if se else "-"
        fb = f"{p['first_buy'][1]}/{p['first_buy'][0]}" if p["first_buy"] else "-"
        fs = f"{p['first_sale'][1]}/{p['first_sale'][0]}" if p["first_sale"] else "-"
        print(f"{p['tape']} | {p['team']} | {p['final_money']:.0f} | {p['total_seeds']} | "
              f"{fb} | {span} | {p['total_sold']} | {fs} | {p['hires_total']}")

    # sell split: qty>=100 endgame liquidation spam vs qty<100 candidate grown/flip
    for p in per_tape:
        small = [(t, d, q) for (t, d, q) in p["sell_events"] if q < 100]
        p["small_sells"] = small
        p["small_sold_total"] = sum(q for _, _, q in small)
        p["small_sold_d19_28"] = sum(q for _, d, q in small if 19 <= d <= 28)
        p["small_sold_d0_15"] = sum(q for _, d, q in small if d <= 15)
        p["spam_sold"] = p["total_sold"] - p["small_sold_total"]
    print("pool SELL TOMATO: total=%d small(<100)=%d spam(>=100)=%d" % (
        sum(p["total_sold"] for p in per_tape),
        sum(p["small_sold_total"] for p in per_tape),
        sum(p["spam_sold"] for p in per_tape)))
    sq = collections.Counter()
    for p in per_tape:
        for (_, d, q) in p["small_sells"]:
            sq[d] += q
    print("small-sell qty by day:", dict(sorted(sq.items())))
    print("small_sold_d19_28 stats (buyers):",
          stats([p["small_sold_d19_28"] for p in buyers]))
    json.dump(per_tape, open("artifacts/tape_pool_20260922_top10/v105_tomato_per_tape.json", "w"), indent=1)
    print("\nwrote artifacts/tape_pool_20260922_top10/v105_tomato_per_tape.json")


if __name__ == "__main__":
    main()
