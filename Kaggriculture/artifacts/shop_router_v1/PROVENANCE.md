# Provenance: shop_router_v1 route data

- `yusuke_routes.json` holds the 10 fixed 719-action routes and the two lookup tables
  (`first_routes`, 8 keys; `second_routes`, 64 keys; branch steps 72 and 144) decoded from
  cell 7 (`_ROUTES_DATA`, base85 + zlib) of the public Kaggle notebook
  `yhay81/public-match-history-router-rating-2929-aug-30` by Yusuke Hayashi, harvested in
  commit `d9adff6` under `artifacts/public_notebooks_20260831/`. The notebook SHA-256 is
  recorded inside the JSON under `source.notebook_sha256`; per-route SHA-256 under `route_sha256`.
- Route 0 is the base (Milan Leonard episode 102248386 per the notebook); route 1 branches at
  step 73 (first shop YARN_STORE); routes 2-9 branch at step 144-150 (second shop). 23 of the
  64 ordered shop pairs point to a specialist route, the other 41 keep the first-shop route.
- The notebook releases its hand-written code under Apache License 2.0 and states that the
  episode data remains subject to the Kaggle competition rules. Review before redistribution.
- Decoded 2026-09-05. Project code lives in `src/kaggriculture_rl/policy/shop_router_v1.py`;
  the router logic is re-implemented there, not copied.
- Relation to our own harvest: route 0 is 95.5% identical to the Yusuke base tape in
  `artifacts/opponent_tapes_20260829/rank003_yusuke_hayashi/` (sha `d22b7b54`), route 2 to
  `25f0696b` (95.3%) and to our live seat-0 tape `26e35645` / ep102574290 (87.1%), route 5 to
  `7b49c14c` (95.7%). Same lineage; the live agent of 2026-08-29 was the "fuller ShopForge CR2"
  the notebook mentions.
