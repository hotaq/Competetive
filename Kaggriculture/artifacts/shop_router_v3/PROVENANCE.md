# Provenance: shop_router_v3 (CemBas lineage)

- `donors.json` — 13 tapes of team **CemBas** harvested 2026-09-05 from public Kaggle replays
  (`artifacts/tape_pool_20260905`, `scripts/harvest_tape_pool.py --teams 20`), all sharing the
  first 144 canonical actions of `cembas__ep105807822__seat0` (episode 105807822, seat 0, vs peikopon).
- `base_tape_routes.json` — that single tape in the shop_router schema (all cells -> route 0), i.e. a
  verbatim replay. This is what `submissions/cembas_base_tape_standalone/` plays.
- `routes.json` — the V100 fitted table (8 cells switch to other CemBas suffixes); +0.011 over the
  base on holdout, below the pre-registered +0.02 bar, kept as an alternative.
- The Kaggle host stated in the competition discussion (topics 737788/738837, cited in 739273) that
  building submissions from public replays is allowed. The recorded actions are the CemBas team's
  work; this repository adds only the routing/replay shell. Review before redistribution.
