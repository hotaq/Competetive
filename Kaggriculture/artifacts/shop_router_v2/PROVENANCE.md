# Provenance: shop_router_v2 (V97 fitted two-shop table)

- `donors.json` — 142 distinct 719-action routes that share the shop router prefix (route 0 of
  `artifacts/shop_router_v1/yusuke_routes.json` to step 144, or route 0 to 72 + route 1 to 144),
  built by `scripts/build_shop_router_donors.py` from: the 10 Yusuke notebook routes, the
  harvested Yusuke episodes in `artifacts/opponent_tapes_20260829/`, and `artifacts/tape_pool_20260830.tar.gz`
  (public Kaggle replays of ringbearer, yukino, Boredom, Umataro Tenma, Hai Dang, Milan Leonard).
  Actions are canonicalised (explicit PICKUP/PLACE counts); provenance per donor is inside the file.
- `routes.json` (sha 0e9e005496ccc284) — the table fitted in `analysis/v97_shop_router_selection_20260905/`
  (screen seeds 16100001-16100500, train seeds 16200001-16201500, 9 tape opponents, Yusuke's
  no-regression rule). 13 routes: yusuke_route_0..9, yusuke_hayashi__ep102574290 (the seat-0 tape of
  live submission 56016840), ringbearer__ep103201048, yukino__ep103199121. 12 of 64 cells differ from
  the v1 table; `first_routes` no longer branches on YARN_STORE.
- Same schema as v1, so `ShopRouterV1.from_file(path)`, `kaggriculture_rs.ShopRouter` and
  `scripts/build_shop_router_v1_submission.py --routes` consume it unchanged.
- Licence: Yusuke's hand-written code is Apache-2.0; all route data derives from public competition
  episodes and remains subject to the Kaggle competition rules. Review before redistribution.
- Built bundle: `submissions/shop_router_v2_standalone/` (main.py sha f98ae37d6c246ca5, tar.gz sha 8300f4455afd8ec5),
  verified against the official env in `analysis/v97_shop_router_selection_20260905/REPORT.md`.
