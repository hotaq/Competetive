# Money Runway (มันนี่ รันเวย์) — "รู้ก่อนเงินหมด" — Forward-looking cash-flow intelligence that tells every First Jobber how many days of money they have left, days before the money runs out.

> Source: workflow `kbtg_gap_research` (run 2026-09-07, all 11 agents ok, no failures). Raw agent output preserved below. `[needs verification]` flags, if any, are the agent's own — verify before pitching.


# Money Runway (มันนี่ รันเวย์) — "รู้ก่อนเงินหมด" — Forward-looking cash-flow intelligence that tells every First Jobber how many days of money they have left, days before the money runs out.

**One-line pitch:** K PLUS's first predictive money feature: a per-user, uncertainty-aware forecast of end-of-month balance ("At your pace: ~฿1,200 left on the 31st — ฿800 below your safe buffer"), with what-if simulation and pre-payday early warnings, built entirely from transaction data K PLUS already holds.

---

## Problem (with evidence from the gap + sources)

**The defining pain.** "เงินเดือนหมดก่อนสิ้นเดือน" — salary gone before month-end — is the single most-reported complaint of Thai first jobbers (first-jobbers research §9, pain #1 and #5: mid-month crunch). The current K PLUS experience (and the entire Thai market — SCB EASY, Krungsri AT Agenda, GXS TH, KTB NEXT, ttb touch) is **backward-looking only**: statements, static monthly spend summaries, manual goal pockets. Nothing answers the only question a First Jobber actually has: *"will I make it to payday?"* (first-jobbers research §7, pain #1; competitor scan — zero predictive cash-flow features in the Thai market).

**Consequence chain.** Because overspending is discovered *after* it happens, the failure mode lands at day 25–30 of every month: overdrafts, payday-loan apps (Thai digital lending apps are notoriously predatory [needs verification — cite BOT/AMLO enforcement actions on illegal lending apps at pitch time]), and — critically for this hackathon's fraud lens — heightened scam vulnerability at month-end (financially stressed users are the prime target for "fast money" investment scams and mule-account recruitment; social-engineering scams dominate Thai fraud losses [needs verification — pull current Royal Thai Police cyber-crime / Thai Bankers' Association loss figures before quoting numbers]).

**Global proof the mechanic works, local absence.** Monzo's "left to spend" / Salary Sorter and Douugh's payday autopilot demonstrate the engagement and behavior-change value of forward-looking cash positioning — neither exists in Thailand (competitor store, whitespace gap #3). Manual expense trackers die within weeks from abandonment; the only sustainable surface is the bank app the user already opens daily.

**The gap in one sentence:** K PLUS holds all the raw signal (full transaction logs, recurring salary credits with fixed Thai paydays, recurring-payment patterns) and renders none of it into a trajectory. That is a pure Data Science gap — not a product, not a security gap — which is exactly why this is the #1 ranked pain × data-availability × Track-2 fit in the first-jobbers synthesis.

## Target User Insight

**Persona: "New," 24, Bangkok, salary ฿22,000 credited on the 28th of each month.**

- First steady income, first full financial autonomy, no learned internal model of her own cash-flow rhythm — she has never lived through 12 of her own pay cycles, so her "gut forecast" is untrained.
- Spending is rhythmic but invisible to her: rent due the 1st, phone bill the 10th, ~฿1,500/week food delivery, Songkran and payday-week splurges.
- She does not want a budgeting *chore* (abandonment killed every tracker app she tried). She wants an **answer, pushed to her, before it's too late** — the same way she consumes weather and traffic forecasts.
- Behavioral framing: she responds to *loss aversion and concrete near-dates* ("฿800 short in 11 days"), not to abstract monthly percentages. This is the design principle for every surface of the feature.

**The insight that makes this winnable:** salaried Thai First Jobbers are the *easiest forecasting population in retail banking* — fixed paydays, recurring bills, stable category habits — yet nobody forecasts for them. High model accuracy is achievable on this segment, so the DS story is credible, not aspirational.

## Data / ML Approach

### Data sources (plausible K PLUS / bank data)

| Source | Fields used | Status |
|---|---|---|
| Core-banking transaction logs | daily debits/credits, amount, timestamp, channel, counterparty descriptor/PromptPay proxy | Almost certainly available (DS toolbox data inventory) |
| Payroll credit events | recurring salary credits from corporate payroll accounts → inferred payday & income | Almost certainly available; enables First Jobber identification (payroll-receiving + age 22–28 + account age <3yr) |
| KYC | age, occupation, declared income band, region | Almost certainly available |
| Product portfolio | deposit balances, credit-card/installment servicing within KBank | Available (NCB cross-bank debt optional; flagged as extension) |
| Calendar | Thai holidays, Songkran, pay-day clustering, long weekends | Public data |
| App telemetry | card impressions, what-if interactions, nudge responses | Generated by the feature itself → feeds the learning loop |

Closed-world caveat acknowledged up front: e-wallet spend (TrueMoney, Rabbit Line Pay) is invisible to the bank; this is stated as a known bias with a consented data-sharing extension path (Thai MyData/PDPA frameworks [needs verification — current adoption state]).

### Features & signals

Engineered per user × per day, aligned to **pay-cycle time** (day 1 = payday), not calendar time:

- **Pay-cycle features:** days-since/to-payday; payday distance to Thai holidays (the "payday + weekend/Songkran cluster" spike is a uniquely Thai, high-signal feature); month-length (28 vs 31 days materially changes runway for paycheck-to-paycheck users).
- **Income features:** payday regularity (std of credit dates), salary trend, detected side-hustle income, inbound family transfers.
- **Obligation features (from recurring-payment detection):** detected bills by date & amount (rent, phone, subscriptions, installments) — pre-committed outflows that make the forecast *mechanically* accurate, not just statistically.
- **Burn features:** trailing 7/14/28-day burn rate per category; weekend vs weekday burn; night-time spend share; payday-spike index (spend in days 0–3 after credit ÷ baseline).
- **Behavior state:** category-mix drift, new-merchant rate, current balance, days-of-buffer remaining.

### Model / technique (specific)

**Layer 0 — Salary & recurring-obligation classifiers.** Rule + classifier hybrid on transaction streams: LightGBM on descriptor TF-IDF (Thai+English code-switched) + amount/date regularity features to label *salary credit*, *recurring bill*, *subscription*, *family transfer*. Recurrence detection via amount/date-cycle matching with edit tolerance (LSTM sequence matcher as upgrade path). This layer converts the forecast from "curve fitting" to "known fixed events + stochastic discretionary spend."

**Layer 1 — Per-user baselines (cold-start backbone).** Seasonal-naive and day-of-pay-cycle average spend profile per user. Cheap, explainable, surprisingly strong for fixed-payday salaried users — and it is the **production fallback** whenever Layer 2 can't run.

**Layer 2 — Pooled global forecaster.** **LightGBM gradient boosting trained across users** (global model, per-row = user × day), because individual users — especially First Jobbers with <6 months of history — lack data for per-user deep models. Trained to predict **daily net-flow quantiles** (q10/q50/q90) via **quantile regression with pinball loss** — this gives honest 80% prediction intervals instead of false-precision point estimates, which is the single most important honesty device in the product ("you'll finish between ฿500 and ฿3,000" is more useful and more trustworthy than "฿1,473").

**Layer 2b — Classical per-user track.** SARIMA / ETS on daily net-flow with day-of-month seasonality for users with ≥9 months of clean history, blended with Layer 2 via inverse-pinball-loss weights (per-user model selection, evaluated on a rolling origin).

**Layer 3 — Monte-Carlo runway simulation.** Sample from the predicted conditional daily-flow quantiles → full distribution of day-by-day balance to next payday → outputs: P(balance < 0 before payday), P(balance < safe buffer), median days-of-money-left. This is what turns a regression output into the user-facing sentence "you run out around the 24th (1-in-5 chance by the 21st)."

**Cold-start handling (explicit, judges will ask):**
- **<1 pay cycle of history:** pooled segment model conditioned on KYC (age band, declared income band, occupation) — a "typical First Jobber at your income" trajectory, clearly labeled to the user as an estimate that improves each week.
- **1–3 cycles:** Layer 2 global model with user-specific fixed events (payday, detected bills) already personalized; discretionary burn shrinks toward pooled prior as history grows (empirical-Bayes shrinkage).
- **≥3 cycles:** Layer 2b blend kicks in.

**What-if counterfactual engine (the demo star).** For "skip 3 late-night Grab orders," don't re-run the model — apply the counterfactual as a deterministic delta on the sampled trajectories: identify the top-k discretionary categories by forecast contribution, let the user toggle them, re-simulate. Milliseconds, explainable, and it makes the intervention *feel* causal.

### Personalization logic

- All forecasts expressed in **pay-cycle time** and in the user's own units (days of money, baht-to-payday), never percentages.
- **Safe buffer** is personalized, not fixed: default = max(7 days of median discretionary spend, ฿500); adapts if the user has detected upcoming obligations (rent due on the 1st).
- Message framing per segment (light-touch, one split): **Payday-splurger** profile (high payday-spike index) gets pre-payday warnings on *day 0–3 after credit* ("you've spent 40% of discretionary budget in 3 days — at this pace…"); steady burners get a single mid-cycle check-in. Nudge timing itself is learned later via contextual bandits — out of scope for the demo, named as roadmap.
- Language: Thai-first copy with concrete-loss framing; the number is the hero of the card.

### Evaluation metrics & success measurement

**Offline (pay-cycle-aligned backtesting — never random time splits):**
- **MAE / RMSE** of end-of-cycle balance (in ฿ and as % of monthly income), reported per history-length cohort (0–1, 1–3, 3+ cycles) — showing the cold-start degradation curve is itself a credibility signal.
- **Pinball loss** at q10/q50/q90; **empirical coverage of the 80% interval** (target 78–82% — coverage calibration is checked, not assumed).
- **Runway classification metrics:** does the model flag "will breach buffer" — PR-AUC (rare-ish positive class), plus lead time distribution (how many days before the breach the alert fires — this is the *product* metric of the model).
- Baseline honesty: always benchmark against seasonal-naive; if the ML doesn't beat it by a margin, say so — judges reward this.

**Online (causal, not correlational):**
- **A/B test, user-level randomization:** treatment = Runway card + early warnings + one-click auto-save suggestion on payday; control = current K PLUS summary.
- Primary: **buffer-days at payday** (days-of-expenses remaining) at 8/12 weeks; secondary: end-of-month overdraft/negative-balance rate, auto-save adoption, savings-rate lift.
- Guardrails: opt-out rate, notification fatigue (dismiss rate), app sentiment.
- Uplift modeling (Qini) on the experiment data to learn *who* the feature helps — feeds the targeting roadmap.

## How It Helps First Jobbers Decide Better

1. **Time-shifts the decision.** Today, discovery of overspending happens at day 27. Runway moves it to day 8–12 — when course-correction is still cheap (skip two deliveries, delay a purchase) instead of expensive (payday loan, overdraft, or falling for a "fast money" scam).
2. **Replaces guilt with a number.** A forecast is not a scolding; "฿800 below buffer with 11 days left" is information. Loss-framed, concrete, dated — the framing the target user responds to.
3. **Makes saving mechanically easy.** The what-if engine converts insight into a one-tap action: "Move ฿500 to your savings pocket on payday and you still finish above buffer." (Default-framed auto-save is the strongest replicated behavioral-econ effect; roadmap after the core card.)
4. **Month-end scam exposure shrinks as a side effect.** The month-end financially-stressed state is scam-vulnerability ground state; reducing its frequency reduces exposure — a data-driven prevention effect, achieved with zero security controls, fully Track-2-compliant.
5. **Compounds into habit.** Each pay cycle, the model gets better and the user gets a visible trajectory ("3 cycles ago you ended at −฿900; last cycle +฿600") — the engagement loop tracker apps never achieved, delivered inside the app already opened daily.

## UX Sketch (brief: key screens / moments inside K PLUS)

1. **The Runway Card (home screen, above statements):** big number = days of money / projected end-of-cycle balance with 80% band; a sparkline of projected balance to payday with buffer line; status color (green / amber / red) only on breach probability, not balance size.
   - *"ถึงวันเงินเดือน คุณจะเหลือประมาณ ฿1,200 (ช่วง ฿500–฿3,000) — ต่ำกว่าเส้นปลอดภัย ฿800"*
2. **The Early Warning push (the money moment):** fires 5–10 days before predicted breach, once per cycle max (fatigue-guarded): *"อีก 11 วันถึงเงินเดือน แต่ตามจังหวะนี้ เงินจะหมดก่อนถึงวันที่ 24 — ลองดู 3 ทางเลือก"* → deep-links to What-if.
3. **What-if Simulator:** top forecast drivers as tappable chips (Grab ฿1,100, Shopee ฿890, weekend dining…) — toggle off, watch the runway number move live. One-tap "Save the difference ฿500 on payday."
4. **Post-payday check-in (day 3):** "You spent 35% of discretionary in 3 days (your usual: 22%). Still on track — barely." The teachable moment for Payday-splurger profiles.
5. **Cycle Review (payday +1):** forecast vs actual ("we predicted ฿1,200; you finished ฿1,650 — you beat your own trend") — builds trust in the model and shows the user their own trajectory.

## Why This Wins for Track 2

- **It is the purest possible Track-2 answer:** first-party data → predictive model → better decision. No new UI paradigm, no security control, no external dependency. The value *is* the intelligence.
- **Full-stack DS depth on one feature:** the pipeline demonstrably chains §1 (recurring-obligation detection) → §2 (quantile forecasting) → simulation → §5 (experimentation) of the DS toolbox — the exact architecture the toolbox report identifies as the highest-leverage chain — while staying scoped enough to actually build.
- **Methodological rigor as a differentiator:** pay-cycle-aligned backtesting, pinball loss, interval coverage, cold-start cohorts, seasonal-naive benchmarking, A/B with causal framing. Most hackathon "predictive" features have none of these; showing the eval design *is* the pitch.
- **Ranked #1 pain × data × fit** in the team's own first-jobbers research, with the mechanic proven globally (Monzo, Douugh) and absent in all six Thai market competitors — so "why now / why here / why us" answers itself.
- **Honest uncertainty as product craft:** the 80% interval converts the classic ML weakness (imperfect accuracy) into a trust feature. Judges from a bank will recognize that as production maturity.

## Risks & Limitations (PDPA/privacy, data availability, cold start, model risk)

- **PDPA (Thailand, effective 2022):** profiling and automated forecasting of personal financial data requires consent/purpose-limitation compliance [needs verification — cite the specific PDPA provisions on profiling and automated decision-making before the pitch]. Mitigation: opt-in feature, onboarding consent screen, forecasts never shared externally, no automated decisions against the user (advice only). Compliance-aware design is a credibility asset with bank judges.
- **Closed-world blind spot:** spend in e-wallets (TrueMoney, Rabbit Line Pay) and other banks is invisible → forecasts systematically overstate runway for wallet-heavy users. Mitigation: honest in-app disclosure ("based on your K PLUS accounts"), error feedback loop (forecast misses nudge users to link consented data — Thai MyData/open-banking path [needs verification — adoption state]).
- **Cold start is real but bounded:** new-to-bank users get segment-prior forecasts; expect visibly worse accuracy in cycle 0–1. Mitigated by design (label uncertainty, improve weekly) — and honestly presented as a known cohort with a measured degradation curve rather than hidden.
- **Model risk & trust:** a confidently wrong forecast destroys the feature ("boy who cried wolf" — one bad miss early kills adoption). Mitigations: 80% intervals by construction; alert frequency caps; cycle-review screen showing forecast-vs-actual to calibrate user trust; rollback to seasonal-naive baseline per user if pinball loss degrades on rolling-origin checks.
- **PromptPay descriptor noise:** merchant metadata on QR transfers is often just a registered payee name [needs verification], degrading categorization. Mitigation: Thai-NLP classifier; fixed-obligation detection leans on amount/date regularity rather than merchant labels (more robust).
- **Behavioral backfire risk:** anxiety-inducing alerts could drive disengagement (the "sleeping dogs" problem from uplift literature). Mitigation: guardrail metrics (opt-out, dismiss rate) in the A/B; loss-framing capped; early-warning max 1/cycle.
- **Attribution limits:** proving scam-exposure reduction from reduced month-end stress is hard to attribute causally; present as a hypothesized second-order effect backed by the A/B's overdraft/buffer metrics, not as a claimed fraud KPI.

## Hackathon Demo Plan (what a small team can actually build & show)

**Scope for a 3–4 person team, ~48h: build the intelligence, mock the bank surface.**

1. **Dataset:** synthesize realistic First Jobber transaction logs (fixed paydays 25th–5th, rent/phone/subscriptions, delivery-app noise, Songkran & payday spikes) for ~500 virtual users × 12 months; or use a public personal-finance dataset (e.g., a transactions dataset on Kaggle) relabeled to Thai context. State clearly that it's simulated.
2. **Pipeline (Python, ~2 days):**
   - Recurring-bill/salary detector (rules + LightGBM) — a few hours, high demo value.
   - Global LightGBM quantile regressor (q10/q50/q90) on pay-cycle-aligned features + seasonal-naive baseline; **pay-cycle-aligned rolling-origin backtest** with MAE, pinball loss, coverage table per cohort — this eval table is a core slide.
   - Monte-Carlo simulator for runway sentence generation + what-if deltas.
   - (Stretch) ETS/SARIMA blend for long-history synthetic users.
3. **Demo surface:** a K PLUS-styled phone mock (Figma screen or simple web app) showing the four moments: Runway Card → Early Warning push → What-if chips moving the number live → Cycle Review with forecast vs actual. Drive the mock live from the real model outputs — this is what separates "DS demo" from "slide demo."
4. **The money moment (90 seconds):** user taps "skip 3 late-night Grab orders" → runway jumps from "runs out the 24th" to "survives to payday with ฿700 above buffer" → one-tap "auto-save ฿500 on payday." Show the model's uncertainty band honestly on screen.
5. **Slide assets:** eval table (MAE/pinball/coverage vs seasonal-naive, per cold-start cohort), architecture diagram of the chain (detection → quantile forecast → simulation → intervention → A/B), and the A/B design for the online metric (buffer-days at 12 weeks).
6. **What NOT to attempt in 48h:** LLM assistant layer, contextual bandit timing, graph analytics, real PDPA/legal review — mention each as a one-line roadmap item to show scope discipline.

## Sources

**Internal research (team-produced, cite by section in the pitch):**
- First Jobbers Research Report — §7 (pain #1: backward-looking only), §9 (pain #1, #5: salary-gone-before-month-end, mid-month crunch), segment persona data.
- DS Techniques Reference Report — §1 categorization & recurring-payment detection; §2 cash-flow forecasting methods, uncertainty requirement, pinball-loss/coverage evaluation, cold-start/pooled-model guidance; §5 experimentation & uplift; §8 wellness-score integration; data inventory table ("data a Thai bank plausibly already has").
- Competitor Scan — Thai market (K PLUS, SCB EASY, Krungsri, GXS TH, KTB NEXT, ttb touch): zero predictive cash-flow features; whitespace gap #3.

**External methodology anchors (standard, citable without web verification):**
- Quantile regression / pinball loss — Koenker & Bassett (1978); standard forecasting-interval evaluation practice.
- SARIMA / ETS — Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (otexts.com/fpp3) — including rolling-origin evaluation methodology.
- LightGBM/XGBoost gradient boosting — Ke et al. (2017), Chen & Guestrin (2016).
- Monzo "left to spend" / Salary Sorter; Douugh payday autopilot — product mechanics as global proof-of-concept (monzo.com, douugh.com) [verify current feature descriptions at pitch time].
- Default effects / commitment devices in behavioral econ — Thaler & Benartzi "Save More Tomorrow" (2004); uplift/Qini — Radcliffe (2007).

**Flagged for pre-pitch verification [needs verification]:** Thai scam-loss statistics (Royal Thai Police cyber-crime / Thai Bankers' Association); predatory lending-app enforcement (BOT/AMLO); PromptPay descriptor metadata richness; typical Thai payroll cycles; PDPA profiling/automated-decision provisions; Thai MyData/open-banking adoption state; K PLUS user count (KBank IR / SET filings).