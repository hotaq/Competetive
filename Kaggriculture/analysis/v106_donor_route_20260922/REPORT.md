# V106 — donor-route transplant screen: OVERALL FAIL, no bundle (22 ก.ย.)

Gates frozen in `PREREGISTRATION.md` (+ Amendment A1, written before any
screen seed was played). Data: `screen.json` (4800 rows),
`routes_donor.json` (attempted router construction, not the candidate),
`gates.json`. Verdict: **OVERALL FAIL — E0 pass, G1/G2/G3/LABOUR fail. No
bundle built, nothing under `submissions/` touched.** Screen seeds
18600101–18600300 are burnt (never tune on them).

## Result

| arm | point | mean margin | median margin |
|---|---|---|---|
| base (omar tape, Rust ShopRouter) | 0.8679 | +29,722 | +19,792 |
| donor (third_farm_club tape, verbatim replay) | 0.3927 | −772 | −7,158 |

Paired (opponent, seed, seat) n=2400: mean money diff **−30,493**, 95% CI
[−33,297, −27,708].

| gate | result |
|---|---|
| E0-legality (amended: tape verified + 24/24 scratch games complete) | **PASS** |
| G1 (mean diff > 0, CI excludes 0) | **FAIL** (−30,493, CI far below 0) |
| G2 (point diff ≥ −0.01) | **FAIL** (−0.4752) |
| G3 (every opp mean diff ≥ −200) | **FAIL** (best binghua −18,280; worst ymg_aq −71,204) |
| LABOUR (window hires 101 ≤ buyer median 99 ∧ E0) | **FAIL** (101 > 99; E0 half passes) |

Seat symmetry: seat0 −30,405 / seat1 −30,582 — loses on both seats equally.
Per-opponent donor point: binghua 0.52 (only winning matchup), ebi 0.37,
thunder 0.355, arda 0.32, ymg_aq 0.649, sida_zuo 0.1425. Base point per
opponent 0.69–1.00.

## Mechanism notes (measured, not guessed)

1. **The donor program does not transfer.** Verbatim replay of a 160,916 home
   tape averages −772 margin off home seeds — a ~−30k paired deficit to omar.
   Same shape as v102 (−112k): a foreign program measured on omar's panel
   loses, and the loss concentrates on specific matchups (ymg_aq −71k here,
   −174k in v104; sida_zuo −37k). The tomato revenue the donor realizes at
   home (~43 units days 19–28) does not survive the move.
2. **Not destroyed, just beaten.** Donor point 0.39 (vs 0.00 for the v102/v104
   candidates) with positive standalone margins vs binghua (+11,967), ebi
   (+2,380), ymg_aq (+25,324). This is a competitive tape losing to a better
   route, not a degenerate one — consistent with v105 §1.4 (corr −0.138):
   tomato presence is not a dose anyone can copy; execution + matchup context
   is everything.
3. **Amendment-A1 construction finding (pre-screen, scratch seeds).** The
   proposal's router embedding is unimplementable for foreign openings:
   `PyShopRouter::index` hardwires routes[0] for steps 0–71
   (`rust/kaggriculture-python/src/fastplay.rs`), so route 1 with an alien
   prefix plays an omar-opening/donor-body chimera (measured 15,822 vs tape
   replay 77,892 on ebi/18600001). The cited `build_shop_router_donors.py`
   path only admits prefix-matching donors; this donor diverges at turn 0.
   Any future router embedding needs a splice design + its own prereg — and
   per this screen's G1, there is no evidence it would be worth it.

## v105 caution — disposition

corr(total_seeds, final_money) = −0.138 and the zero-tomato pool-max tape are
consistent with this outcome: no seed-count knob was touched (single donor
frozen at prereg, 10 seeds as-harvested), and the LABOUR gate guarded the
execution gap rather than ignoring it — it failed the transplant on crew
cost (101 > 99) before any game was scored, and the screen confirmed the
transfer fails on money too.

## Deliverables (no bundle per decision rule)

- `scripts/screen_v106_donor_route.py` (build/parity/screen/gates; ruff clean)
- `analysis/v106_donor_route_20260922/`: `PREREGISTRATION.md` (+A1),
  `opponents/` (6 frozen copies, shas match v102), `routes_donor.json`
  (attempted construction, superseded), `screen.json` (4800 rows),
  `gates.json`, this report.

## Limits / follow-ups (each needs a new prereg; these seeds are burnt)

- Donor-suffix splice onto the omar prefix (the only router-embeddable form)
  is a new hypothesis, not a fix — and G1 here gives no reason to expect it
  to pass.
- Donor wins 39% of games with matchup structure (binghua 0.52); a
  shop-conditioned selection rule cannot be keyed from frozen data and must
  not be fit on these seeds.
- Opponents are open-loop tapes; closed-loop opponents untested.
