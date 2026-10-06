# Pre-deadline refresh runbook (frozen 2026-09-22, fire ~Sept 28-29)

Goal: replace the decaying live arms before the **2026-09-30 23:59 UTC** deadline by
repeating the v101 tape-league pipeline (stages from
`analysis/v101_tape_league_20260919/PREREGISTRATION.md`) on a freshly harvested
top-10 pool. Team "zero waste". Live now: `omar_base_tape_standalone`
(submission **56361130**, public ~1466 and falling).

Rules for this file: every compute/submit command below is copy-paste in order.
`kaggle competitions submit` commands are **documented, NOT executed** (read-only
CLI only until the human fires). No step below modifies `src/`, `tests/`,
`submissions/` (except creating ONE new bundle dir), `rust/`, or `bindings/`.
Nothing is committed by this runbook.

Conventions: run everything from the repo root. Python 3.13 via `uv run python`,
scripts run as files with bare imports (`sys.path` handling is inside the scripts).
`OUT=analysis/refresh_20260928` is the fire-day analysis dir (dated for the day you
fire; the dry-run in §8 used `analysis/refresh_runbook_20260922/`).

## 0. Pre-flight (read-only, ~1 min)

```bash
kaggle competitions leaderboard kaggriculture -s | head -15
kaggle competitions submissions kaggriculture | head -5
ls artifacts/tape_pool_20260922_top10/tapes | wc -l   # 92 expected (frozen Sept-22 pool)
```

Go/no-go: CLI authed and leaderboard returns; deadline still 2026-09-30 23:59 UTC.

## 1. Harvest — re-target the CURRENT top-10 (~5-10 min)

No code change needed to re-target: `scripts/harvest_top10_pool.py` reads the live
leaderboard itself (`--teams 10`) and takes the 6 most-recent public episodes per
team, both seats, same tape schema as the Sept-22 run.

```bash
uv run python scripts/harvest_top10_pool.py --output artifacts/tape_pool_20260928_top10 --teams 10 --episodes-per-team 6
```

Inputs: live leaderboard + public replay downloads (~32 MB each, discarded after
reduction). Outputs: `artifacts/tape_pool_20260928_top10/{tapes/*.json,
manifest.json}` + a `HARVEST_REPORT.md` you write from the manifest (leaderboard
snapshot table, pool contents, no-replay teams, sanitize check — copy the shape of
`artifacts/tape_pool_20260922_top10/HARVEST_REPORT.md`).

Expected duration: ~2 s/episode download+reduce (+0.75 s request delay); Sept-22 run
did 46 episodes in a few minutes; budget **5-10 min** for ~40-60 episodes.

Go/no-go: success criterion from the Sept-22 run is >= 1 tape for most of the top-10
(Sept-22: 10/10, 92 tapes / 46 episodes); every tape `length == 720` with action keys
exactly `{farmer, hands, market}`; `manifest.json` logs `dropped`,
`teams_with_no_replays`, `own_team_seats_skipped` ("zero waste" seats are skipped,
never harvested). ABORT the refresh if > 3 of the top-10 yield zero replays (pool
too thin — wait ~12 h and re-harvest).

## 2. Tape league screen (~1-2 min compute)

Screen every pool tape verbatim against the FROZEN V100 opponent set (same 10 tapes
v101 used, so points are comparable to `analysis/v101_tape_league_20260919/`) with
the current live route as the ROUTER reference arm. Pool stays frozen (do NOT
re-harvest between screen and holdout).

First stage the frozen opponents (one-time; they live inside two existing tarballs):

```bash
mkdir -p $OUT/opponents && cd /tmp && rm -rf opp_stage && mkdir opp_stage && cd opp_stage \
&& tar -xzf "$OLDPWD/artifacts/tape_pool_20260905.tar.gz" --strip-components=2 \
  "tape_pool_20260905/tapes/ai是我的豆包__ep105803010__seat0.json" \
  "tape_pool_20260905/tapes/andrey_tikhomirov__ep105796705__seat0.json" \
  "tape_pool_20260905/tapes/baran_akyıldız__ep105805663__seat1.json" \
  "tape_pool_20260905/tapes/giulio_ravasio__ep105784996__seat1.json" \
  "tape_pool_20260905/tapes/keiz__ep105793106__seat0.json" \
  "tape_pool_20260905/tapes/lagrangianlocomotive__ep105793292__seat0.json" \
  "tape_pool_20260905/tapes/peikopon__ep105807824__seat0.json" \
  "tape_pool_20260905/tapes/yuk__ep105790347__seat0.json" \
&& tar -xzf "$OLDPWD/artifacts/tape_pool_20260830.tar.gz" --strip-components=2 \
  "tape_pool_20260830/tapes/yukino__ep103129686__seat0.json" \
  "tape_pool_20260830/tapes/a_poor_vul__ep102980038__seat1.json" \
&& cp *.json "$OLDPWD/$OUT/opponents/" && cd "$OLDPWD" && ls $OUT/opponents | wc -l  # 10
```

Then the screen (use FRESH seeds — never reuse 181xxxxx/184xxxxx ranges):

```bash
uv run python scripts/evaluate_tape_league.py \
  --pool artifacts/tape_pool_20260928_top10 \
  --opponents $OUT/opponents/*.json \
  --router artifacts/shop_router_v4/base_tape_routes.json \
  --seed-start 18500101 --seed-count 100 \
  --output $OUT/screen.json
```

Inputs: frozen pool + 10 frozen opponents + live routes file as ROUTER.
Output: `$OUT/screen.json` (`seeds`, `opponents`, `meta`, `aggregates`).

Expected duration: **~1 min** (dry-run §8: 186,000 games in 57 s wall, ~3,260
games/s). Correction to the v101 "~25 min" claim: the screen compute itself is
minutes on this machine; 25 min plausibly covered harvest downloads + ranking
analysis on slower hardware. Budget 5 min incl. overhead.

Go/no-go: `aggregates` holds every pool tape + `ROUTER` with 2,000 games
(10 opp x 100 seeds x 2 seats); then keep the best tape per team and take the top
12 teams forward (same cut as v101).

## 3. Round-robin best-per-team (~1 min compute)

Build a 13-tape pool (12 shortlisted tapes + the live omar tape) and play every pair,
seeds 182xxxxx-pattern fresh (v101 used 18200001-18200200; use a fresh range):

```bash
mkdir -p $OUT/roundrobin_pool/tapes
for t in <12 shortlisted tape stems> omar_althobaiti__ep110807162__seat1; do :; done  # pick from $OUT/screen.json
```

Copy the 12 winning tape files plus the live tape
(`artifacts/tape_pool_20260919` extraction of `omar_althobaiti__ep110807162__seat1`;
get it from the v101 pool tarball, NOT a new download) into
`$OUT/roundrobin_pool/tapes/`, then:

```bash
uv run python scripts/evaluate_tape_league.py \
  --pool $OUT/roundrobin_pool \
  --opponents $OUT/roundrobin_pool/tapes/*.json \
  --seed-start 18500201 --seed-count 200 \
  --output $OUT/round_robin.json
```

(No `--router` here: every arm is a pool tape; ROUTER reference already measured in
§2. Tape-vs-tape games are seat-symmetric but both seats are still played for
protocol parity.)

Expected duration: 13 x 13 x 200 x 2 = 67,600 games, **< 1 min** at dry-run
throughput.

Go/no-go / picks: W = highest pooled point among the 12 (excluding the live tape);
W2 = runner-up from a DIFFERENT team whose prefix 0-72 canonical-action identity
with W is < 0.90 (hedge must be a different lineage — v101 had 0.96 field
identity). Lineage check snippet:

```bash
uv run python -c "
import json
a = json.load(open('<W tape>'))['actions'][1:73]
b = json.load(open('<W2 tape>'))['actions'][1:73]
print('prefix-72 identity:', sum(x == y for x, y in zip(a, b)) / 72)
"
```

ABORT (keep omar in both final slots) if no tape beats the live ROUTER point from §2.

## 4. Holdout (~1 min compute + analysis)

Arms W, W2, and live vs 10 holdout opponents = remaining shortlist tapes (exclude W
and W2), fresh seeds (v101: 18300001-18301000; use a fresh 18500301-based range):

```bash
uv run python scripts/evaluate_shop_router_fast.py \
  --tape W=<path-to-W-tape> --tape W2=<path-to-W2-tape> --tape live=<path-to-live-omar-tape> \
  --opponents <10 holdout tape files> \
  --routes artifacts/shop_router_v4/base_tape_routes.json \
  --seed-start 18500301 --seed-count 1000 \
  --output $OUT/holdout.json
```

(`--routes` only sets the base/one/two/live router arms; our arms come from
`--tape`, one file per arm. Row schema matches v101 `holdout.json`: arm, opponent,
seed, seat, money, margin, point, shop1/shop2.)

Expected duration: 3 x 10 x 1000 x 2 = 60,000 games, **~1 min** compute; budget
30 min with the gate analysis below.

Go/no-go (frozen v101 gates,ตัดสินแยก per arm):
- G1: point(W) - point(live) >= +0.20 with a 95% seed-bootstrap CI excluding 0.
- G2: point(W) >= 0.50 pooled on holdout.
- G1'/G2': same bars for W2, judged independently (v101: W2 passed G1, failed G2
  0.437 -> NOT built; slot 2 stayed on the incumbent).
- G3: bundle plays in official `kaggle_environments` with money matching
  `play_fast` on both seats (§5).
- G4 (strength gate, added 2026-09-23 after the yannik lesson: 8-0 vs weak
  opponents rated only 1165.6): at least 3 of the 10 holdout opponents must be
  tapes from currently top-500 teams (check names against the live leaderboard
  at fire time; the Sept-22 pool's per-tape `final_money`/team fields help), and
  point(W) on that strong subset alone must be >= 0.50. Farming 1400-2000s pays
  Elo pennies; only wins vs strong opponents move the rating.

## 5. Build bundle (~2 min)

Convert the winning tape to a single-route file (same schema as
`artifacts/shop_router_v4/base_tape_routes.json`: `routes=[719 actions]`,
`first_branch_step=72`, `second_branch_step=144`, all branch tables pointing at
route 0), then build with the standard script:

```bash
uv run python -c "
import json, hashlib
base = json.load(open('artifacts/shop_router_v4/base_tape_routes.json'))
tape = json.load(open('<W tape file>'))
route = tape['actions'][1:720]
assert len(route) == 719 and all(set(a) == {'farmer','hands','market'} for a in route)
base['routes'] = [route]
base['route_names'] = ['<W tape stem>']
base['route_sha256'] = [hashlib.sha256(json.dumps(route, sort_keys=True, separators=(',',':')).encode()).hexdigest()]
base['source'] = {'tape': '<W tape stem>', 'evidence': '$OUT'}
json.dump(base, open('$OUT/base_tape_routes_<W stem>.json','w'), separators=(',',':'), sort_keys=True)
print('wrote routes file, file-sha16:', hashlib.sha256(open('$OUT/base_tape_routes_<W stem>.json','rb').read()).hexdigest()[:16])
"
uv run python scripts/build_shop_router_v1_submission.py \
  --routes $OUT/base_tape_routes_<W stem>.json \
  --output-dir submissions/<new_name_standalone> \
  --name <new_name_standalone> \
  --evidence $OUT/
```

The build script automatically runs the equivalence check (bundle `agent()` vs
reference on 3 seeds x 160 steps x 2 seats) and writes `candidate.json`. Then G3
in the official env, both seats:

```bash
uv run python scripts/smoke_kaggle_raw_submission.py submissions/<new_name_standalone>/submission.tar.gz --candidate-seat 0 --output $OUT/g3_seat0.json
uv run python scripts/smoke_kaggle_raw_submission.py submissions/<new_name_standalone>/submission.tar.gz --candidate-seat 1 --output $OUT/g3_seat1.json
```

Expected duration: build+equivalence **~1-2 min**; each official-env full game is
much slower than Rust play_batch — budget **~10-20 min per seat**.

Go/no-go: `candidate.json` status `standalone_validated`; archive contains ONLY
`main.py`; G3 passes iff both seats report `statuses == ["DONE", "DONE"]`, zero
invalid-action-key steps, and rewards match the Rust holdout money for the same
seed (exact match — any mismatch aborts the fire).

## 6. Submit (DOCUMENTED — NOT EXECUTED in dry-run; human fires)

```bash
kaggle competitions submit -c kaggriculture -f submissions/<new_name_standalone>/submission.tar.gz -m "<W tape stem> refresh <YYYY-MM-DD>: holdout <point> vs live, G1+G2+G3 pass, evidence $OUT"
```

Two-slot hedge rule (final team score = the BETTER of the 2 selected submissions):
- Slot A (fallback): submission **56361130** (`omar_base_tape_standalone`) stays
  selected no matter what. Never deselect both incumbents in one move.
- Slot B (challenger): the new bundle from §5, submitted only if it passed
  G1+G2+G3. If W2 also passed all three gates, it may replace a dead third arm per
  the daily submission budget — never at the cost of Slot A until the challenger
  has a COMPLETE status and a public score above omar's.
- If only W passes: Slot B = W bundle; the second final slot remains omar.
- If neither passes: submit NOTHING; omar rides to the deadline (a stale known
  arm beats an unproven one under newest-economy-wins).

## 7. Verify (read-only, minutes to hours)

```bash
kaggle competitions submissions kaggriculture | head -5
kaggle competitions leaderboard kaggriculture -s | head -15
```

Go/no-go: new submission reaches `SubmissionStatus.COMPLETE`; public score posts
(usually within hours — this is why §9 fires early); score > omar's current public
(~1466 and decaying) before selecting it as a final submission. Keep polling the
leaderboard; top-1 churns daily (DSM 3121.0 on Sept-22).

## 8. Dry-run evidence (2026-09-22, league-screen portion only)

Executed exactly the §2 stage on the EXISTING frozen pool
(`artifacts/tape_pool_20260922_top10/`, 92 tapes — NO re-harvest, NO new downloads;
opponents extracted from existing tarballs to /tmp, repo untouched except the two
deliverable files):

```bash
uv run python scripts/evaluate_tape_league.py \
  --pool artifacts/tape_pool_20260922_top10 \
  --opponents /tmp/refresh_opp/*.json \
  --router artifacts/shop_router_v4/base_tape_routes.json \
  --seed-start 18400101 --seed-count 100 \
  --output analysis/refresh_runbook_20260922/dryrun.json
```

Record (wall-clock via `date +%s` around the call):
- Wall time **57 s** for **186,000 games** (92 tapes + ROUTER x 10 opp x 100 fresh
  seeds x 2 seats) => **~3,260 games/s**. A 2-seed probe (3,720 games) ran in 2 s
  first, confirming throughput before the full run.
- Output `dryrun.json` (101 KB): `seeds=[18400101, 18400200]`, 10 opponents,
  93 aggregates (92 tapes + ROUTER), ROUTER arm complete with 2,000 games.
- Best-per-team pooled point on this panel (ROUTER/live-omar reference 0.9035 —
  highest on the board, as expected: these are the stale V100 opponents omar was
  selected against; the fire-day screen uses the same panel for comparability, and
  §3 picks only tapes that beat live):

| pooled point | team | best tape |
|---:|---|---|
| 0.8250 | Yannik Schiffner | yannik_schiffner__ep111952809__seat0 |
| 0.7140 | mtmr_s1 | mtmr_s1__ep111955057__seat0 |
| 0.6790 | KawattaTaido | kawattataido__ep111931862__seat1 |
| 0.6370 | Otter Vibe | otter_vibe__ep111939633__seat0 |
| 0.5665 | THIRD FARM CLUB | third_farm_club__ep111946168__seat0 |
| 0.5365 | Kaggledew Valley | kaggledew_valley__ep111943076__seat0 |
| 0.5035 | SpaTaro | spataro__ep111934253__seat0 |
| 0.4890 | Majkel1337 | majkel1337__ep111933074__seat0 |
| 0.4450 | M & M & P & Q | m___m___p___q__ep111934253__seat1 |
| 0.4345 | THUNDER THUNDER | thunder_thunder__ep111943074__seat1 |
| 0.4140 | 吃白饭的大肥鱼 | 吃白饭的大肥鱼__ep111931862__seat0 |
| 0.4125 | QQ | qq__ep111953944__seat0 |

- Slowness/failures: none. No errors, no retries, no repo files touched besides
  `analysis/refresh_runbook_20260922/dryrun.json` (+ this runbook). Seeds
  18400101-18400200 are fresh (never played before: v101 used 181/182/183xxxxx,
  v106 used 186xxxxx) — no seed-reuse leakage.
- NOT dry-run (documented only): harvest download, round-robin, holdout, bundle
  build, submit, verify-against-live-Kaggle. The round-robin/holdout commands reuse
  the same two verified scripts (`evaluate_tape_league.py` verified here,
  `evaluate_shop_router_fast.py` sharing its `play_batch` engine), so residual risk
  is arg-wiring, mitigated by the go/no-go row-count checks in §3-§4.

## 9. Freshness note — pool decay and the fire window

- Meta: newest-economy-wins, leaderboard turning over in days. Sept-19 pool's
  lineage decayed to ~1300 by fire time; live omar (56361130, submitted Sept-19)
  now scores public **1465.9** (submissions list, Sept-22) and is still falling,
  while top-1 moved DSM 3113.7 -> 3121.0 in the same window. Expect any pool to be
  stale **2-3 days** after harvest — a Sept-22 pool fired on Sept-28 would be
  ~6 days old, i.e. dead on arrival.
- Fire window: re-harvest NO earlier than ~48 h before firing, and fire
  **Sept-28 daytime UTC through Sept-29 ~12:00 UTC at the latest**. Rationale:
  Kaggle scoring posts hours after submit, and one retry buffer (failed G3, bad
  score, daily submission limit) must fit before the **Sept-30 23:59 UTC**
  deadline. Firing on Sept-30 itself leaves no recovery margin — treat Sept-29
  midday as the hard stop.
- Minimum viable fire (if short on time): harvest (§1) + screen (§2) + build the
  single best screened tape that clears G1/G2 on a 200-seed holdout + G3 + submit
  to Slot B. Skip the W2 hedge only if time forces it — never skip G3.
