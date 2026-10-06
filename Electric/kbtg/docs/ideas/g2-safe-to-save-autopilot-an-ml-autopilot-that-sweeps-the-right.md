# Safe-to-Save Autopilot — "เงินเก็บอัตโนมัติ ที่ปรับให้พอดีกับชีวิตคุณ" — An ML autopilot that sweeps the *right* amount to savings the moment salary lands, sized by each user's forecast cash flow and delivered with segment-matched behavioral framing — so First Jobbers finally save without deciding to.

> Source: workflow `kbtg_gap_research` (run 2026-09-07, all 11 agents ok, no failures). Raw agent output preserved below. `[needs verification]` flags, if any, are the agent's own — verify before pitching.


# Safe-to-Save Autopilot — "เงินเก็บอัตโนมัติ ที่ปรับให้พอดีกับชีวิตคุณ" — An ML autopilot that sweeps the *right* amount to savings the moment salary lands, sized by each user's forecast cash flow and delivered with segment-matched behavioral framing — so First Jobbers finally save without deciding to.

## Problem (with evidence from the gap + sources)

Thai financial literacy is **high but saving behavior is not** — the classic knowing–doing gap:

- BOT/NESDC 2022 financial literacy survey: Thai adults scored ~71.4%, **above** the ~60.5% OECD average, yet saving behavior, budgeting discipline, and emergency buffers lag badly [needs verification — exact 2022 BOT/NESDC survey figures].
- Most young Thai households cannot survive a 3-month income shock; emergency-buffer adequacy is a weak pillar of the same survey [needs verification].
- First Jobbers practice "save what's left" — and nothing is left. Present bias (hyperbolic discounting) means every raise is absorbed by lifestyle creep, so savings never materialize despite above-average knowledge.
- K PLUS's existing savings surfaces (K-Pookpigian-style pockets, K-Saving products) are **passive containers**: no automation, no affordability awareness, no commitment device, no adaptive sizing [needs verification — exact current K PLUS savings-feature capabilities].
- The mechanics that demonstrably work with young users globally — Plum's affordability-aware auto-save, Acorns' invisible round-ups, Cleo's behavioral coach, Douugh's autopilot — have **no Thai-market equivalent** (Thai-market whitespace; competitor scan g1/#1, #2).
- Generic financial education is wasted on this segment: their *knowledge* is already above average; their *behavior at the moment of choice* is what fails.

**The gap:** nobody in the Thai market converts intelligence into action *at the moment money arrives* (salary credit). Payday is the single highest-leverage decision moment — income lands, willpower is highest relative to the month, and the transaction is a machine-detectable event. The solution must be an adaptive, automated saving intelligence, not another container or another course.

## Target User Insight

**Persona: "Ploy", 24, first job, ฿28,000/month salary credited on the 27th.**

- She *knows* she should save 10%; by day 10 her balance tells a different story. Money is spent on Grab Eats, Shopee, Netflix/Disney+/Spotify, and nights out — small, frequent, frictionless.
- She has never once manually transferred money to savings *before* spending — manual saving requires a decision, and decisions are exactly what present bias destroys.
- Her income is regular (fixed payday) but her spending is spiky (Songkran, birthday months, year-end). A static "save ฿3,000 every month" fails in the spiky months, then she cancels the whole thing after one overdraft — **commitment devices fail when they're brittle**.
- She responds to streaks and progress (Duolingo/Apple Watch generation), hates being scolded by an app, and distrusts anything that feels like the bank upselling a product.

**Insight:** First Jobbers don't need more motivation — they need the decision to be *made for them, correctly sized, and resilient to bad months*. The right amount to save is not a rule of thumb; it's a **forecasting problem**.

## Data / ML Approach

### Data sources (plausible K PLUS / bank data)

| Source | Fields used | Availability |
|---|---|---|
| Transaction logs (core banking) | Every debit/credit: timestamp, amount, channel (PromptPay QR, mobile transfer, debit card, ATM), merchant descriptor / counterparty proxy, MCC where present | Almost certainly available |
| Salary-credit events | Recurring inbound credits from corporate payroll accounts (amount, date cadence, payer corporate ID) | Almost certainly available — this is the **automatic First Jobber detector**: age 22–28 (KYC) + regular payroll credit + account age < 3 years |
| Deposit/product data | Balances across accounts, savings pockets, fixed deposits; overdraft events | Available |
| Recurring-payment detections | Subscriptions, rent, bills (from transaction enrichment layer — g1/§1 of DS toolbox) | Builds on existing K PLUS categorization [needs verification] |
| App telemetry | Feature usage, nudge impressions/clicks, opt-out events, session timing | Available |
| External spend (TrueMoney, Rabbit Line Pay, other banks) | **Not available** — closed-world blind spot; acknowledge and optionally propose consented PDPA-based data sharing [needs verification — Thai open banking / MyData status] |

### Features & signals

**Cash-flow forecast features (per user, daily granularity):**
- Payday distance (days since/to salary credit), payday amount trend & stability (CV of last 6 salary credits)
- Committed-expense forecast: detected recurring payments due in next 30 days (rent, installments, subscriptions, phone)
- Recent burn rate (7/14/30-day rolling net outflow by category), discretionary vs. committed split
- Day-of-week, day-of-month, Thai calendar effects (Songkran, New Year, payday-adjacent clustering — uniquely Thai and highly predictable spikes)
- Balance level, buffer-days (current balance ÷ trailing 30-day expense rate), overdraft history
- User tenure & history length (drives cold-start model routing, below)

**Behavioral segmentation features (for clustering):**
- Payday-spike index (spend in days 0–3 post-payday ÷ trailing monthly average)
- Night-time spend share (22:00–05:00)
- Subscription count & leak ratio (recurring small outflows ÷ income)
- Savings rate (net accumulation ÷ income, trailing 3 months)
- Delivery/e-commerce concentration, new-merchant rate
- Income volatility

**Nudge-response / uplift features:**
- Prior nudge impressions, opens, actions, opt-outs; feature-engagement depth; streak history; segment label; forecast headroom at payday.

### Model / technique

**1. Income-event & First Jobber detection (lightweight):**
Recurring-credit classifier on inbound transactions — XGBoost over credit cadence regularity, amount stability, payer account type (corporate payroll), plus rules; near-trivial to make high-precision. Bonus/13th-month/windfall detection = credit > k × trailing salary median on a one-off cadence.

**2. Cash-flow forecasting with uncertainty (the sizing brain):**
- **Per-user point forecast** of day-30 net position: gradient boosting (LightGBM) on the features above, with day-of-month seasonality — a global model with per-user feature encoding (catboost-style user embeddings or per-user bias terms), which beats per-user ARIMA at this history length and shares strength across users.
- **Uncertainty via quantile regression**: train the same model at quantiles τ ∈ {0.1, 0.5, 0.9} with pinball loss → **safe-to-save = q₀.₁(projected day-30 balance) − safety buffer**, floored at 0. This is the Plum-style "affordability-aware" mechanic done properly: we sweep only money that survives a pessimistic forecast.
- Baselines to show judges: seasonal-naive day-of-month profile per user, and ETS/SARIMA — LightGBM quantile model must beat them on pinball loss to justify itself.
- **Backtesting discipline:** pay-cycle-aligned, time-based splits only — train on months 1–9, test on months 10–12 per user; never random splits.

**3. Behavioral segmentation:**
k-means / GMM on the segmentation features (standardized, with GMM preferred for soft assignments and BIC-selected k). Expected archetypes: *Payday Splurger, Subscription Leaker, Paycheck-to-Paycheck, Disciplined Starter, Windfall-Rider*. Use soft cluster probabilities downstream so edge users get blended framing.

**4. Nudge-response uplift modeling (target only persuadeables):**
- Two-model uplift (T = auto-save prompt with framing variant, C = no prompt) or **causal forest** on experiment data; features = segment, engagement, forecast headroom, tenure.
- Select by **Qini/AUUC**; suppress nudges for predicted "sure things" (would save anyway) and **"sleeping dogs"** (respond negatively — measured via opt-out lift). This is the anti-spam brain.

**5. Cold-start handling (explicit design, judges will probe):**
- < 3 months history: rule-based safe tier — sweep a conservative flat % (e.g., 3% of salary credit) with a low cap (฿300), plus "shadow mode": run the forecast model, log what it *would* have swept, compare to actual end-of-month balance, until 3 pay cycles of validation before live sweeping.
- 3–6 months: pooled segment-level model (cluster-conditional quantile forecast) — borrow strength from the archetype.
- > 6 months: full personalized quantile model.
- Every sweep sized by the pessimistic quantile, never the median — a structurally conservative system that fails toward *not* taking money.

**6. Windfall auto-split:** on detected bonus/13th-month credit, propose a pre-registered split (default: 50% save / 50% spend) using the same quantile headroom logic on the *total* month.

**7. Experimentation engine:** A/B/n randomization at user level; treatment arms = {no autopilot, autopilot default-off prompt, autopilot default-on with one-tap decline}; framing variants per segment. Multi-armed bandit (Thompson sampling) for send-time and framing once minimum detectable effects are established.

### Personalization logic

- **Amount:** per-user quantile-forecast safe-to-save, dynamic monthly — ฿1,800 in a quiet month, ฿400 in Songkran month, ฿9,000 when the bonus lands. Never a static number.
- **Timing:** sweep executes automatically on salary-credit detection (T+0), before spending can claim the money — the moment of capability, not the moment of asking.
- **Framing by segment (from §3 clusters):**
  - *Habit-formers / streak-responsive*: streak gamification — "3-month streak 🔥 ฿5,400 banked," milestone badges, streak-freeze for one bad month (resilience, not brittleness).
  - *Loss-averse*: loss-framing — "Last month ฿1,240 leaked to subscriptions & delivery. This payday, we locked ฿1,500 before it could."
  - *Paycheck-to-paycheck (low headroom)*: no sweep pressure — switch to micro-mechanics (round-ups: PromptPay ฿126 → ฿174, ฿48 swept) plus leak-reduction nudges; grow the sweep as forecast headroom grows.
- **Product routing:** swept funds land in a user-designated K PLUS savings pocket / e-savings / fixed deposit tier chosen by horizon preference — the intelligence layer, not a product upsell.

### Evaluation metrics & success measurement

| Layer | Offline metrics | Online / causal metrics |
|---|---|---|
| Salary-credit & windfall detection | Precision/recall vs. labeled payroll credits; F1 ≥ 0.98 target (high precision required — a false "bonus" sweep is a trust-destroyer) | User correction rate |
| Cash-flow forecast (quantile) | Pinball loss @ τ = 0.1/0.5/0.9; **empirical coverage of the 80% interval** (target 78–82%); MAE on day-30 balance (฿ and % of monthly income); vs. seasonal-naive & SARIMA baselines, pay-cycle-aligned splits | % of months where post-sweep balance stayed above safety buffer (the "did we take too much" metric) |
| Segmentation | Silhouette, GMM BIC, cluster stability month-over-month (ARI) | Segment-conditional nudge response rates |
| Uplift / targeting | **Qini coefficient / AUUC** on held-out experiment data | Conversion lift vs. control; targeting efficiency (sweep activations per prompt sent) |
| Autopilot overall | — | **A/B lift on savings rate & buffer days at 4/8/12 weeks** (primary); sweep-retention (autopsy survival curves); opt-out / complaint rate as **guardrails**; % users who ever hit overdraft *because of* a sweep (must be ~0; auto-refund + model update if it happens) |

**Success definition (causal, not correlational):** treatment arm shows ≥ X pp lift in 3-month savings rate and ≥ 2 additional buffer-days vs. control at 8 weeks, with opt-out rate below guardrail. Every number shown to judges is an experiment readout, not a vanity metric.

## How It Helps First Jobbers Decide Better

1. **It removes the decision, correctly.** The #1 failure of "just save 10%" is that it demands willpower at the worst moment. The autopilot executes at salary credit — money is saved *before* lifestyle creep can touch it, with a default-on commitment device (opt-out in one tap).
2. **It's sized to reality, not to a rule.** Quantile-forecast sizing means the commitment never breaks: in a heavy month it sweeps ฿400 instead of ฿3,000, so the user never cancels — the system survives the exact months that kill static standing orders.
3. **It converts data into a visible decision aid.** "At your pace you'll end the month ฿800 below your buffer — a ฿1,500 payday sweep keeps you above it with 95% confidence" turns a forecast into a concrete, checkable claim, building the habit of thinking in *projected* money.
4. **It teaches through doing, not lecturing.** Loss-framed leak reports ("฿1,240 leaked to delivery & subscriptions") deliver financial-education content personalized to *their own* numbers — the segment that ignores generic education responds to their own data.
5. **Windfalls become wealth by default.** Bonus/13th-month money — the biggest present-bias trap for young earners — gets a pre-registered 50/50 split proposed the second it lands.

## UX Sketch (key screens / moments inside K PLUS)

**Moment 1 — Salary credit lands (push, T+0):**
> "เงินเดือนเข้าแล้ว 💰 We've set aside ฿1,350 for you (safe even in your heaviest month). [See why] [Adjust] — Tap to undo anytime."
One-tap undo = the commitment device's pressure valve.

**Screen — "Why this amount" (transparency = trust):**
Forecast chart: projected day-30 balance curve with the 80% interval band; committed expenses ahead; the sweep shown keeping the pessimistic path above the safety buffer line. This screen is the pitch to regulators and judges in one image: *deterministic numbers, pessimistic quantile, visible logic.*

**Screen — Safe-to-Save home (weekly check-in):**
Streak flame, total banked, buffer-days gauge, this month's leak report ("Delivery +38% vs. last month — want a ฿200/payday top-up?"). Trajectory framing: "Buffer: 6 → 11 days this quarter."

**Moment 2 — Windfall detected:**
> "That's ฿45,000 — looks like a bonus 🎉 Pre-plan: ฿22,500 saved, ฿22,500 yours. [Confirm split]"

**Moment 3 — Low-headroom user (paycheck-to-paycheck):**
No sweep pressure; round-ups card + leak-reduction suggestion. The system explicitly *backs off* — restraint as personalization.

## Why This Wins for Track 2

1. **Full closed learning loop, presentable as a system:** detect salary credit → forecast cash flow with uncertainty → size the sweep → select & frame via uplift → **measure causal lift with A/B at 4/8/12 weeks** → retrain. Most hackathon pitches show a model; this shows a *learning system with an experiment engine* — exactly what the gap synthesis flags as a judging differentiator.
2. **Technically specific and honest:** quantile regression with pinball loss and coverage checks, uplift with Qini/AUUC and sleeping-dog suppression, GMM soft segmentation, explicit cold-start tiers with shadow-mode validation, pay-cycle-aligned backtesting. Every claim has a metric.
3. **Solves the exact gap:** the knowing–doing gap is a *moment-of-choice* problem; this is a moment-of-choice solution. Education is explicitly not the mechanism — intelligence is.
4. **Fails safe by construction:** pessimistic-quantile sizing, auto-refund on overdraft, opt-out guardrails — model-risk thinking that bank judges live with daily.
5. **Whitespace-backed:** proven global mechanics (Plum, Acorns, Douugh) with no Thai equivalent; not a hypothetical behavior-change bet.

## Risks & Limitations

- **PDPA (Thailand, effective 2022):** profiling + automated sweeping = consent, purpose limitation, and transparency obligations. Mitigations: explicit opt-in consent for the autopilot, the "Why this amount" screen as transparency-by-design, transaction data processed within the existing banking relationship purpose [needs verification — cite specific PDPA sections on profiling/automated decisions at build time].
- **A wrong sweep is worse than no sweep:** an over-aggressive sweep causing an overdraft destroys trust. Structural mitigation: q₀.₁ sizing, per-sweep caps in early tenure, auto-refund + incident-triggered model rollback; online metric "% sweeps causing overdraft" as a hard guardrail.
- **Closed-world blind spot:** external spend (TrueMoney, other banks) is invisible → forecasts overestimate headroom. Mitigations: conservative quantile absorbs some of this; balance actuals feed back within days; propose consented external-data sharing long-term [needs verification — Thai open banking/MyData status].
- **Cold start:** new-to-bank First Jobbers have < 3 months history. Handled by tiered design (rules → pooled segment model → personalized model) with shadow mode; be honest that the first 3 cycles are deliberately boring.
- **PromptPay descriptor poverty:** merchant metadata on QR payments is thin, hurting categorization → Thai-NLP enrichment layer needed upstream (g1 dependency) [needs verification — descriptor richness varies by acquirer].
- **Attribution & novelty decay:** A/B lift is cleanest in the first quarters; habit formation vs. novelty effect requires the 12-week streak-survival readout, which is longer than a hackathon — present as the pilot's phased design.
- **Uplift models need experiment data before they exist:** at launch, targeting is segment-based; uplift targeting activates after the first experiment wave. State this sequencing honestly — it's a maturity roadmap, not a flaw.

## Hackathon Demo Plan

Small team, 4–6 weeks, synthetic-but-realistic data:

**Data:** generate a synthetic First Jobber population (n ≈ 5,000, 12 months of daily transactions) with scripted behavioral archetypes (splurgers, leakers, steady savers), Thai calendar spikes (Songkran, payday clustering), PromptPay-style noisy descriptors. Optionally seed with any public transaction-simulation approach; label generation logic is known, so **offline metrics are computable against ground truth** — a huge demo advantage over teams with no labels.

**Build (split 3 ways):**
1. **ML core (2 people):** salary-credit detector → LightGBM quantile forecast (τ = 0.1/0.5/0.9) with pinball-loss evaluation vs. seasonal-naive baseline; GMM segmentation. Show the coverage calibration plot and the pinball-loss comparison table live — judges' table stakes.
2. **Uplift + experiment simulation (1 person):** simulate a past A/B on the synthetic population, train a causal forest, show the **Qini curve** and the persuadeable/sure-thing/sleeping-dog split. This slide alone separates Track 2 contenders.
3. **Demo app (1–2 people):** Figma-to-clickable or thin React mock of the three moments — payday push, "Why this amount" forecast band chart, windfall split, streak home. Feed the mock from the *real* model outputs (run inference on one synthetic user live), so the demo shows the actual pipeline, not hardcoded numbers.

**The 5-minute story:** (1) the knowing–doing gap stat slide; (2) live demo of one payday with the forecast band and the sized sweep; (3) the learning-loop diagram with A/B lift readout and guardrails; (4) coverage/pinball/Qini metrics table; (5) cold-start & PDPA honesty slide — closing on risk maturity, which bank judges remember.

**Deliberately out of demo scope:** live LLM copy generation (mention as roadmap), graph analytics (other gaps), real production integration.

## Sources

*No live web search was available in this environment; per the task method, claims rest on the provided gap synthesis and DS toolbox report plus domain knowledge, with unverified specifics flagged inline.*

1. **BOT/NESDC 2022 Financial Literacy Survey** — Thai literacy ~71.4% vs. OECD average ~60.5%; weak saving/emergency-buffer behavior. [needs verification — pull exact survey title, figures, and URL from BOT/NESDC before pitch]
2. **Gap synthesis, first-jobbers §3 & §9 (#2)** — "save what's left," present bias, lifestyle creep on raises, 3-month-shock vulnerability.
3. **Competitor scan (g1, gaps #1, #2)** — Plum (affordability-aware auto-save), Acorns (round-ups), Cleo (behavioral coach), Douugh (autopilot); no Thai-market equivalent; K PLUS savings features are passive containers. [needs verification — current K PLUS feature set]
4. **DS toolbox report §2 (quantile/pinball-loss forecasting), §4 (GMM/k-means segmentation, uplift/Qini), §5 (experimentation, commitment devices, guardrails)** — methodological anchors.
5. **Methodology anchors (standard, citable without Thai-specific verification):** quantile regression / pinball loss (Koenker & Bassett, 1978); uplift modeling & Qini (Radcliffe, 2007; Jaskowski & Jaroszewicz, 2012); causal forests (Wager & Athey, 2018); GMM clustering; LightGBM (Ke et al., 2017); default effects / commitment devices (Thaler & Benartzi, "Save More Tomorrow," 2004).
6. **K PLUS scale (~19–20M users)** — [needs verification — KBank latest annual report / SET filing, kasikornbank.com IR].
7. **Thai payroll cycles (fixed end-of-month / 25th–5th paydays)** — [needs verification].
8. **PDPA (Thailand, effective June 2022)** — [needs verification — cite specific sections on profiling and automated decision-making if challenged].