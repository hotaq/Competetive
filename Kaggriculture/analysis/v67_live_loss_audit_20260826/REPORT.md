# V67 live public replay loss audit (28 games)

Date: 2026-08-26  
Scope: read-only diagnosis of submission `55797143` public replays. No agent change, holdout access, promotion, or upload.

## Evidence inputs

- Replay manifest: `replays/submission_55797143_public_refresh_after22_20260826/manifest.json`
  - SHA256: `9c02fcec98fb1f2de14dbe7d87e79f301dfd65ab4da286856d1efedc8e449829`
- Uploaded-action parity: `replays/submission_55797143_public_refresh_after22_20260826/v67_uploaded_action_parity.json`
  - SHA256: `f3efa8d3188de670b7b4facb03ac5dc57ff29b307908e5ea656b38630d5e5abd`
  - Exact: 28/28 episodes, 20,132/20,132 compared action steps.
- Local V67 standalone:
  - `submissions/meta_v67_quantity_conserving_front_run_standalone/main.py`
  - SHA256: `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`

## Outcome summary

- All 28: 24 wins, 4 losses, win rate 85.714%, mean margin +23,050.79, worst -21,061.
- Prior 22: 21 wins, 1 loss, mean margin +29,611.91.
- Newest 6: 3 wins, 3 losses, mean margin -1,006.67.
- Seat 0: 18/20 wins; seat 1: 6/8 wins. This is a weak small-sample seat signal, not proof of seat causality.
- Opponent rank is absent from these artifacts; no Top-1-to-20 claim is supported.

## Common macro signature

All four losses selected V59 route `base`. The 20 base-route wins and all four base-route losses share:

- land unlocks at steps 161 and 253;
- terminal land 3 and hands 9;
- terminal composition COW 11, SHEEP 4, WHEAT plots 12;
- terminal carried inventory 0 and shed 0.

The shared route/execution signature rules out an opening, land schedule, no-action, or terminal inventory-stranding explanation.

Mean base-route margins show the first cohort-level divergence is late:

| Step | Day/hour | Base-win mean margin | Loss mean margin |
|---:|:---:|---:|---:|
| 120 | 5/0 | -339.9 | -228.5 |
| 360 | 15/0 | -644.0 | +475.8 |
| 432 | 18/0 | +9,014.7 | +4,519.5 |
| 480 | 20/0 | +10,531.7 | +437.8 |
| 576 | 24/0 | +16,334.9 | -2,832.5 |
| 719 | 29/23 | +25,243.2 | -8,967.2 |

During steps 360-575, base wins add candidate/opponent cash +49,977.9/+32,748.3; losses add +46,362.0/+49,351.0. During steps 576-719, base wins add +26,334.7/+17,426.3; losses add +21,476.2/+27,611.0.

## Market-regime signature

The loss cohort ends in a wool-scarce, milk-glutted shared market:

| Step | Base-win mean MILK | Base-win mean WOOL | Loss mean MILK | Loss mean WOOL |
|---:|---:|---:|---:|---:|
| 360 | 180.9 | 29.9 | 163.8 | 166.2 |
| 480 | 74.0 | 31.9 | 19.8 | 161.0 |
| 576 | 53.4 | 47.8 | 9.2 | 179.5 |
| 648 | 47.8 | 58.9 | 1.5 | 224.5 |
| 718 | 42.5 | 59.6 | 1.0 | 231.0 |

Average shop multiplicity per base-route episode is YARN_STORE 1.75 losses vs 0.35 wins and BRUNCH_SPOT 2.25 vs 1.15. With only four losses this is association, not a sufficient rule: V67 also wins a YARN_STORE x4 control (`100143759`).

Despite the price inversion, cumulative requested V67 sells remain nearly fixed. At step 719, the base-win median requested quantities `[WHEAT, STRAWBERRY, MILK, WOOL]` are `[195, 295, 263, 127]`. Three losses request `[195, 289-295, 263, 127]`; episode `100187229` requests `[234, 298, 273, 136]`. Requested quantities are not guaranteed fills because the engine can stop an active order when an economic commit fails.

## Base-win checkpoint median

Notation: `M=C/O/margin`, `P=[WHEAT,STRAWBERRY,MILK,WOOL]`, `S=` cumulative requested SELL quantities through the preceding transition, `BW=` cumulative `BUY_PRODUCT WHEAT`.

| Step | Day/hour | M | P | S | BW |
|---:|:---:|:---|:---|:---|---:|
| 120 | 5/0 | 12/319.5/-308.5 | 33/147/179/215 | 0/0/0/0 | 25 |
| 360 | 15/0 | 12,253.5/15,853/-3,917.5 | 41.5/206.5/206.5/1 | 6/6/12/47 | 82 |
| 480 | 20/0 | 40,867.5/31,028/+8,562 | 46/197.5/60/1 | 6/80/103/74 | 103 |
| 576 | 24/0 | 64,680/43,563/+6,654.5 | 48.5/21/13/1 | 6/206/165/96 | 103 |
| 719 | 29/23 | 81,775/55,828/+12,767.5 | 46.5/11.5/1/1 | 195/295/263/127 | 103 |

## Four losses

### 100159725 - seat 0 vs wqlun06

- Replay SHA256: `2016357995951b4ed48f5ffd54d70c9ad9d1b224adff7a33943c3f8e592fd243`
- Final 74,625 vs 81,484; margin -6,859.
- First permanently negative step 487 (day 20/hour 7).
- Checkpoints:
  - 120: M 10/359/-349; P 33/139/179/215; S 0/0/0/0; BW 25.
  - 360: M 13,955/9,771/+4,184; P 42/196/122/215; S 6/6/12/47; BW 82.
  - 480: M 34,888/33,797/+1,091; P 46/174/1/200; S 6/76/103/74; BW 103.
  - 576: M 52,918/54,552/-1,634; P 49/1/3/227; S 6/206/165/96; BW 103.
  - 719: M 74,625/81,484/-6,859; P 42/51/1/224; S 195/289/263/127; BW 103.

### 100184942 - seat 1 vs Jasper Turnidge

- Replay SHA256: `753383925b79cdcebbb8214c67afaf99fa5ff6d3337ecceebce5d4de4be9dd07`
- Final 88,777 vs 94,247; margin -5,470.
- First permanently negative step 577 (day 24/hour 1).
- Checkpoints:
  - 120: M 15/126/-111; P 32/139/179/215; S 0/0/0/0; BW 25.
  - 360: M 14,184/20,191/-6,007; P 39/196/212/217; S 6/6/12/47; BW 82.
  - 480: M 47,876/48,397/-521; P 40/181/76/209; S 6/80/103/74; BW 103.
  - 576: M 69,893/69,602/+291; P 41/1/22/167; S 6/212/165/96; BW 103.
  - 719: M 88,777/94,247/-5,470; P 30/20/1/221; S 195/295/263/127; BW 103.

### 100187229 - seat 0 vs Ayaan Ghosh

- Replay SHA256: `0fc0e6ee2ee30f900f37781bb150aba2ac685b59ce49e70b12d961e536f0f24c`
- Final 72,129 vs 93,190; margin -21,061.
- First permanently negative step 457 (day 19/hour 1).
- Checkpoints:
  - 120: M 23/128/-105; P 32/139/179/215; S 0/0/0/0; BW 25.
  - 360: M 14,732/13,167/+1,565; P 40/187/122/232; S 6/5/12/48; BW 84.
  - 480: M 39,531/43,940/-4,409; P 42/170/1/234; S 6/85/106/83; BW 105.
  - 576: M 54,684/65,473/-10,789; P 43/1/9/236; S 6/197/173/105; BW 105.
  - 719: M 72,129/93,190/-21,061; P 39/1/1/239; S 234/298/273/136; BW 105.

### 100196432 - seat 1 vs bangqun ZHAO

- Replay SHA256: `756ad37232656018ec1c66bd0ad37244b77652ce5d7381aaa5214aca6406fe20`
- Final 91,247 vs 93,726; margin -2,479.
- First permanently negative step 590 (day 24/hour 14).
- Checkpoints:
  - 120: M 10/359/-349; P 33/155/179/215; S 0/0/0/0; BW 25.
  - 360: M 12,095/9,934/+2,161; P 42/216/199/1; S 6/6/12/47; BW 82.
  - 480: M 35,429/29,839/+5,590; P 46/214/1/1; S 6/76/103/74; BW 103.
  - 576: M 63,378/62,576/+802; P 49/116/3/88; S 6/206/165/96; BW 103.
  - 719: M 91,247/93,726/-2,479; P 41/135/1/227; S 195/289/263/127; BW 103.

## Opponent-policy family evidence

- `wqlun06` (`100159725`) and `newbyy` (`100168881`) have 96.9% exact opponent-action equality and an exact 233-step prefix, with V67 margins -6,859 and +18,079. Both place V67 in seat 0.
- `bangqun ZHAO` (`100196432`) shares a 150-step prefix and 64.7% exact actions with `wqlun06`, but loses by only -2,479.
- `Jasper Turnidge` (`100184942`) has 58.8%-71.2% exact-action similarity to winning controls `sd`, `Loi Tran Bao`, and `RhoenSprudel`; the corresponding V67 winning margins are +8,857, +4,593, and +9,765.

Inference: failures cluster by opponent-policy-family x shared market regime, rather than username or seat alone.

## Diagnostic counterfactuals on loss4 only

The replay-tape opponent seat was held fixed and the candidate occupied its original V67 seat. These results are diagnostic and cannot support promotion.

| Candidate | Margins by episode 100159725 / 100184942 / 100187229 / 100196432 | Wins | Mean | Worst |
|:---|:---|---:|---:|---:|
| V59 | -6,814 / -5,432 / -20,288 / -2,437 | 0/4 | -8,742.75 | -20,288 |
| V67 all-items | -6,859 / -5,470 / -21,061 / -2,479 | 0/4 | -8,967.25 | -21,061 |
| MELON front-run | -6,814 / -5,432 / -20,288 / -2,437 | 0/4 | -8,742.75 | -20,288 |
| MILK front-run | -6,834 / -5,471 / -20,390 / -2,372 | 0/4 | -8,766.75 | -20,390 |
| STRAWBERRY front-run | -6,656 / -5,444 / -20,951 / -2,442 | 0/4 | -8,873.25 | -20,951 |
| WOOL front-run | -6,833 / -5,419 / -20,298 / -2,521 | 0/4 | -8,767.75 | -20,298 |

V59 improves every observed loss over V67 by +45, +38, +773, and +42 margin, but rescues none. Therefore sale front-running is a small regression in this failure regime, not the macro cause.

## Smallest causal hypothesis and gate

V59 locks its suffix at step 120, when only one shop is visible. The base trace buys COW 1 at step 97, then COW 1/4/2/1/2 at steps 155/158/169/193/249. By the time the wool/milk regime becomes discriminating, the portfolio is already cow-heavy.

Smallest causal intervention to test:

1. preserve livestock optionality by delaying part of the COW x4 at 158 and COW x2 at 169 until shop 3 / steps 216-240;
2. select deferred cow vs sheep capacity from observed shop demand plus a persistent WOOL/MILK price ratio;
3. disable one-step front-running only in a detected low-price/unstable regime;
4. leave movement, land unlocks, and legality unchanged.

Run the exact development-only 8-case gate in `gate8_manifest_suggestion.json`. Passing it is only a causal screen. It does not authorize promotion or upload.

Evaluation orientation is explicit in that manifest: the candidate must occupy `agent_seat`, while only `opponent_tape_seat` is replayed. Using the source manifest's `top1_seat` directly reverses the counterfactual and is invalid for this gate.

## Limits

- Public replays and all eight proposed gate cases are consumed development evidence.
- Replay-tape counterfactuals hold opponent actions fixed and are not live strategic-response proof.
- Scheduled market quantities are not filled-quantity evidence.
- Four losses are too few for a stable statistical meta estimate.
- TokenSave MCP was unavailable and `.tokensave/tokensave.db` was stale (55 files, newest source 2026-08-16), so current artifacts required direct inspection.
