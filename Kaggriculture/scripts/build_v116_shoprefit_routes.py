"""V116: build a fresh shop-router (Yusuke method) on the Sept-22 winner tapes.

Base route = yannik. Each named donor tape is verified (720 actions, exact
action keys) and checked for canonical prefix identity through the branch
steps (v97 `canonical` rule). Splicing uses `splice()` semantics and ONLY
where the prefix is identical through the branch step -- never across
divergent farms (v106 chimeras die). Cells with no eligible donor stay at
incumbent 0. If ANY donor is eligible the script refuses to tune blind
(exit 2): a fit screen on fit seeds must pick the cells first.

Outputs: artifacts/shop_router_v116/routes.json + analysis dir fit.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import normalize_legacy_action

from kaggriculture_rl.policy.shop_router_v1 import splice

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = ROOT / "analysis" / "refresh_20260923" / (
    "base_tape_routes_yannik_schiffner__ep111952809__seat0.json"
)
POOL = ROOT / "artifacts" / "tape_pool_20260922_top10" / "tapes"
OUT_ROUTES = ROOT / "artifacts" / "shop_router_v116" / "routes.json"
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
DONORS = [
    "mtmr_s1__ep111947463__seat0",
    "unknown_mother_goose__ep111950709__seat0",
    "third_farm_club__ep111957345__seat0",
    "majkel1337__ep111958811__seat1",
    "m___m___p___q__ep111946154__seat1",
    "otter_vibe__ep111947463__seat1",
    "dsm__ep111948590__seat1",
    "mtmr_s1__ep111956211__seat0",
    "dsm__ep111956627__seat0",
    "otter_vibe__ep111955452__seat0",
]


def canonical(actions: list[Any]) -> list[dict[str, Any]]:
    return [normalize_legacy_action(a) for a in actions]


def load_donor_route(stem: str) -> list[dict[str, Any]]:
    payload = json.loads((POOL / f"{stem}.json").read_text(encoding="utf-8"))
    actions = payload["actions"]
    assert len(actions) == 720, f"{stem}: length {len(actions)} != 720"
    assert all(set(a) == {"farmer", "hands", "market"} for a in actions), f"{stem}: bad keys"
    return actions[1:720]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_ROUTES)
    parser.add_argument("--fit-output", type=Path, required=True)
    args = parser.parse_args()

    base_data = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    first_step = int(base_data["first_branch_step"])
    second_step = int(base_data["second_branch_step"])
    base = base_data["routes"][0]
    assert len(base) == 719, f"base length {len(base)}"
    base_canon = canonical(base)
    # splice() semantics self-check: identical prefix splices to the whole route.
    assert splice(base_canon, base_canon, first_step) == base_canon
    assert splice(base_canon, base_canon, second_step) == base_canon

    donor_routes = {stem: load_donor_route(stem) for stem in DONORS}
    donor_canon = {stem: canonical(route) for stem, route in donor_routes.items()}
    eligible = {
        stem: {
            "first": donor_canon[stem][:first_step] == base_canon[:first_step],
            "second": donor_canon[stem][:second_step] == base_canon[:second_step],
        }
        for stem in DONORS
    }
    print(json.dumps(eligible, indent=1))
    usable = [stem for stem, flag in eligible.items() if flag["first"]]
    if usable:
        print(f"eligible donors exist {usable}: run a fit screen before building tables; refusing")
        raise SystemExit(2)

    first_routes = {shop: 0 for shop in SHOPS}
    second_routes = {f"{a}__{b}": 0 for a in SHOPS for b in SHOPS}
    route_blob = {
        "first_branch_step": first_step,
        "second_branch_step": second_step,
        "first_routes": first_routes,
        "second_routes": second_routes,
        "routes": [base],
        "route_names": list(base_data.get("route_names", ["yannik_base"])),
        "route_sha256": [hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()],
        "schema": "shop_router_v4/base_tape",
        "source": {"v116": "fresh Yusuke-method refit on Sept-22 winners; base = yannik",
                   "base": str(BASE_PATH.relative_to(ROOT)),
                   "eligibility": "no named donor shares the canonical prefix; tables all -> 0"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(route_blob) + "\n", encoding="utf-8")
    identity = {}
    for stem in DONORS:
        hits = sum(1 for i in range(first_step) if donor_canon[stem][i] == base_canon[i])
        identity[stem] = hits / first_step
    fit_note = {"eligible_donors": [], "cells_changed": [], "donor_identity72": identity}
    args.fit_output.parent.mkdir(parents=True, exist_ok=True)
    args.fit_output.write_text(json.dumps(fit_note, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} (1 route, identity tables) + {args.fit_output}")


if __name__ == "__main__":
    main()
