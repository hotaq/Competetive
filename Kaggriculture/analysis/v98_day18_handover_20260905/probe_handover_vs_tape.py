import json, sys, statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path("/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Kaggriculture")
def run(args):
    seed, opp_name = args
    from kaggriculture_rs import Engine
    from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1
    from kaggriculture_rl.policy.late_game_manager import LateGameHandover, LateGameManager
    router = ShopRouterV1.from_file(ROOT / "artifacts/shop_router_v2/routes.json")
    opp = json.load(open(ROOT / f"analysis/v97_shop_router_selection_20260905/opponents/{opp_name}.json"))["actions"][1:720]
    out = {"seed": seed, "opp": opp_name}
    for label in ("tape", "mgr"):
        pol = router.agent if label == "tape" else LateGameHandover(router.agent, LateGameManager(), start_step=432).agent
        e = Engine({"episodeSteps": 720, "seed": seed})
        for t in range(719):
            a = pol(e.observation(0), None); e.step([a, opp[t]])
            if t == 431:
                f = e.state["farms"][0]; tiles = f["tiles"]
                out["d18"] = {"money": f["money"], "plants": sum(1 for r in tiles for x in r if isinstance(x, dict) and x.get("kind") == "PLANT"),
                              "animals": sum(1 for r in tiles for x in r if isinstance(x, dict) and x.get("animal")),
                              "weeds": sum(1 for r in tiles for x in r if isinstance(x, dict) and x.get("kind") == "WEED"), "quadrants": len(f["unlocked_quadrants"])}
        m = [x["money"] for x in e.state["farms"]]; out[label] = {"us": m[0], "opp": m[1], "gain": m[0] - out["d18"]["money"]}
    return out
if __name__ == "__main__":
    SP = sys.argv[1]
    jobs = [(s, o) for o in ("kenjo1209__ep102449287__seat1", "subramanya_n__ep102288474__seat1") for s in range(16800001, 16800061)]
    with ProcessPoolExecutor(8) as pool: rows = list(pool.map(run, jobs))
    json.dump(rows, open(f"{SP}/rescue_probe.json", "w"))
    rows.sort(key=lambda r: r["tape"]["gain"])
    print(f"{'seed':>9} {'opp':<10} {'d18$':>7} {'plants':>6} {'anim':>4} {'weeds':>5} | {'tape gain':>9} {'mgr gain':>9} {'diff':>8} | tape mgr")
    for r in rows[:15] + rows[-5:]:
        print(f"{r['seed']:>9} {r['opp'][:10]:<10} {r['d18']['money']:>7,.0f} {r['d18']['plants']:>6} {r['d18']['animals']:>4} {r['d18']['weeds']:>5} | {r['tape']['gain']:>+9,.0f} {r['mgr']['gain']:>+9,.0f} {r['mgr']['gain']-r['tape']['gain']:>+8,.0f} | {'W' if r['tape']['us']>r['tape']['opp'] else 'L'}    {'W' if r['mgr']['us']>r['mgr']['opp'] else 'L'}")
    tg = [r["tape"]["gain"] for r in rows]; mg = [r["mgr"]["gain"] for r in rows]
    print(f"\nn={len(rows)} tape gain median {st.median(tg):,.0f} (p10 {sorted(tg)[len(tg)//10]:,.0f}, p90 {sorted(tg)[9*len(tg)//10]:,.0f}) | mgr gain median {st.median(mg):,.0f} (p10 {sorted(mg)[len(mg)//10]:,.0f}, p90 {sorted(mg)[9*len(mg)//10]:,.0f})")
    print("tape wins", sum(1 for r in rows if r["tape"]["us"] > r["tape"]["opp"]), "mgr wins", sum(1 for r in rows if r["mgr"]["us"] > r["mgr"]["opp"]), "mgr better gain in", sum(1 for r in rows if r["mgr"]["gain"] > r["tape"]["gain"]))
    for thr in (35, 45, 50):
        low = [r for r in rows if r["d18"]["plants"] < thr]
        if low: print(f"plants<{thr} at d18: {len(low)} games; tape gain median {st.median(r['tape']['gain'] for r in low):,.0f} vs mgr {st.median(r['mgr']['gain'] for r in low):,.0f}")
