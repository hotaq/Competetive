# Debt X-Ray (หนี้เอกซเรย์) — See everything you owe, ranked by what actually hurts you

> Source: workflow `kbtg_gap_research` (run 2026-09-07, all 11 agents ok, no failures). Raw agent output preserved below. `[needs verification]` flags, if any, are the agent's own — verify before pitching.


# Debt X-Ray (หนี้เอกซเรย์) — See everything you owe, ranked by what actually hurts you

One-line pitch: An intelligence layer inside K PLUS that auto-detects every installment obligation — including BNPL legs KBank can see — computes a private total debt-service picture and a personal delinquency-risk score, then prescribes the payoff order that minimizes damage.

---

## Problem (with evidence from the gap + sources)

First Jobbers' signature debt is **BNPL stacked across non-bank apps** (SPayLater, Atome, LINE installments) on top of credit cards, personal loans, and กยศ./กรอ. student loans. Because these obligations live in separate apps with separate billing dates, no single view exists of what they owe in total:

- **Fragmentation blindness:** K PLUS sees only its own legs of each payment (the PromptPay/card debits to SPayLater or Atome), never the obligation itself. The user sees a "฿1,050 payment to SHOPEEPAY" — not "installment 4 of 10 on a ฿10,500 purchase."
- **Minimum-payment trap:** BNPL is perceived as "free money" (0% interest framing), so users take on more legs than their cash flow supports. Evidence from the gap synthesis: SLF (Student Loan Fund) repayment data shows **early-career delinquency is predictable from cash-flow patterns** — the first loan interaction often goes delinquent from poor cash-flow planning, not unwillingness to pay.
- **Regulation is institutional, not customer-facing:** BOT personal-loan licensing of BNPL operators (2024–25) **[needs verification — exact circular and effective dates]** protects at the licensing level; nothing gives the *customer* a total-burden view or cost comparison.
- **Shame suppresses help-seeking:** nothing in the market offers a private, judgement-free "what do I actually owe and what does it cost" view. No Thai app aggregates or models installment burden (competitor-store whitespace gap #6: subscription/BNPL leakage).

The core Track-2 problem framing: this is a **data-integration + aggregation intelligence problem** — recurring-payment classification, obligation reconstruction, and delinquency-risk prediction. Classic supervised/unsupervised DS, no new data collection required beyond what the bank already holds.

## Target User Insight

**Persona:** "Aim," 24, first salaried job (฿22–28K/month), Bangkok. Has 3 BNPL legs (Shopee, Atome, LINE Pay later), 1 credit card (minimum payments), กยศ. loan, rent, phone bill, ~10 subscriptions. Pays everything through K PLUS because that's where her salary lands.

**Key insight:** Aim's delinquency risk isn't willingness — it's **structure**. She doesn't know her true monthly debt-service ratio (DSR), doesn't know which obligation damages her most if she misses it (BNPL late fees compound fast; กยศ. has deferment options; credit card minimums are a trap), and can't see collision weeks (two installment due dates + rent on the same week). When money runs short, she makes the decision with zero information — and shame means she asks no one. The product's job is to be the **judgement-free analyst** she's too embarrassed to hire.

**Behavioral anchor:** the first BNPL miss is the highest-leverage intervention moment — a just-in-time, forward-looking nudge at that moment ("if you pay Atome first this month, you avoid ฿350 in late fees and your risk score stays flat") beats any generic financial-literacy content.

## Data / ML Approach

### Data sources (plausible K PLUS / bank data)

1. **Transaction logs** — every debit with timestamp, amount, channel (PromptPay QR, inter-bank transfer, direct debit, card), counterparty proxy/display name. This is where BNPL legs appear as recurring debits to SPayLater/Atome/Rabbit/Fulfill/etc.
2. **Internal loan & card servicing records** — KBank installment loans, credit-card statements (minimum due, revolving balance, interest rate), payment history. Labels for delinquency modeling.
3. **Detected recurring-payment registry** (built by this solution) — reconstructed obligations from transaction streams.
4. **KYC / payroll metadata** — age, occupation, account age, salary credit cadence (payroll-receiving + age 22–28 + account age < 3y = First Jobber heuristic).
5. **Optional enrichment: NCB credit-bureau data** for cross-bank debt — **[needs verification — NCB permissible-purpose scope for non-credit-use customer-facing analytics; likely requires consented purpose; design must work without it]**. Present as Phase-2, not a dependency.
6. **Calendar features** — Thai paydays, due-date clustering, Songkran/holiday spend spikes.

### Features & signals

**For obligation reconstruction (the classifier layer):**
- Descriptor strings (code-switched Thai/English: "SHOPEEPAY", "ATOME TH", "โอนไป บ.เทอร์ม", MCC where available)
- Recurrence structure: amount stability (±small tolerance), inter-payment interval regularity, day-of-month alignment, duration of pattern persistence, sequence position estimate ( installment n of ~m inferable from amount + start date)
- Counterparty proxy identity + known BNPL-operator registry (dictionary seed)
- Autopay vs manual payment pattern (direct-debit vs user-initiated same-amount transfers)
- Context features: did a large purchase at a matching merchant precede the installment sequence?

**For delinquency-risk scoring:**
- **DSR features:** total detected monthly debt service ÷ detected monthly income; per-obligation weight; count of active legs
- **Cash-flow features:** end-of-month projected balance (from a quantile cash-flow forecast — §2 of toolbox), buffer days (days of expenses in balance), payday-spike index, income volatility
- **Collision features:** bills due within same 7-day window ÷ weekly income; overdue-detection (missed recurrence — a detected installment that didn't arrive on schedule is the single strongest early signal)
- **Behavioral:** minimum-payment behavior on card (revolving vs full payoff), night-time spending, new-BNPL-leg origination rate (taking a 4th leg while 3 are active)
- **History:** prior overdrafts, prior late fees, account age

### Model / technique (specific)

**Layer 1 — Obligation detection (recurring-payment classifier):**
- Staged architecture: **rules/dictionary on known BNPL operators** (high precision, covers SPayLater/Atome etc.) → **XGBoost on engineered recurrence features** for unknown counterparties → **Thai-language transformer (WangchanBERTa / Typhoon-family)** on descriptor strings to classify obligation type when descriptors are noisy. Rules handle the head of the distribution; the transformer handles the tail (exactly the progression in the DS techniques reference §1).
- Recurrence scoring: interval-regularity score = variance of inter-payment gaps normalized by expected cycle; combined with amount-stability into a per-series "is-installment" probability, thresholded with hysteresis (start requiring strong evidence; stop on 2 consecutive misses).
- Sequence-position estimation: fit expected term length from amount ÷ observed purchase size (a ฿10,500 purchase with ฿1,050 monthly debits ⇒ 10 months ⇒ "installment 4 of ~10, ends Sep").

**Layer 2 — Delinquency-risk model:**
- **Gradient boosting (LightGBM)** survival-style model: predict P(30+ DPD on any obligation in next 90 days), trained on labels from internal loan/card performance plus **detected-missed-recurrence events** as proxy labels (SLF-pattern insight: early-career cash flow predicts delinquency).
- Baseline: penalized logistic regression for explainability comparison.
- **Calibration:** isotonic calibration — mandatory because the score is shown to users and drives thresholds (Brier score + reliability curves in eval).
- **Cold-start handling:** users with <6 months history get a **pooled First-Jobber cohort model** (segment-level priors: median DSR, typical leg counts) with wide confidence bands shown honestly; individual model takes over as history accumulates. Explicitly name the pooled-to-personal transition design.
- **Time-based train/test splits only** (pay-cycle-aligned); never random splits.

**Layer 3 — Payoff-order optimizer (not really ML — deterministic optimization on ML outputs):**
- Avalanche ordering by effective interest/fee cost per obligation (card revolving ~16–20% APR **[needs verification — current KBank card rates]**, BNPL late-fee effective annualized cost can exceed card APR, กยศ. low/fixed and deferrable), constrained by minimum dues and due-date calendar; outputs "pay this ฿500 here first" — an impact-ranked plan. This layer is what makes the model *decision-useful* rather than diagnostic-only.

### Personalization logic

- Risk score and messaging tuned by segment (from a light clustering pass: "structured struggler" vs "payday splurger with legs" vs "over-committed multi-leg"); framing varies (loss-framing for high-DSR users, capability-framing for near-miss users).
- **Just-in-time triggers:** (a) first detected missed installment → triage moment; (b) new BNPL-leg origination detected while DSR > 35% → pre-commitment insight ("this leg takes you to 41% DSR; your end-of-month projection drops below buffer"); (c) collision week ahead → reorder suggestion.
- Contextual bandit over nudge variants (§5 toolbox) to learn send-time and framing per user; uplift targeting so only persuadeable users get nudged (avoid sleeping-dog backlash on a shame-sensitive topic).

### Evaluation metrics & success measurement

| Layer | Offline metrics | Online / product metrics |
|---|---|---|
| Obligation detection | Precision/recall vs. hand-labeled installment series (target ≥0.9 precision on high-stakes "miss detection"); coverage (% of recurring series classified); macro-F1 on type classification | User confirmation/correction rate; false-obligation complaint rate |
| Delinquency risk | AUC-ROC + **PR-AUC** (rare outcome); calibration (Brier, reliability curve); lead time of missed-recurrence signal vs. actual default event | Incidence of first-miss in coached vs. control cohort |
| Payoff plan | Plan quality vs. brute-force optimum (cost avoided simulation) | Adoption rate; measured late-fee/interest cost reduction over 8–12 weeks (A/B, causal) |
| Product | — | DSR trajectory lift, buffer days lift, opt-out rate (guardrail — shame-sensitive topic means opt-out is the key health metric), engagement with triage moments |

**Causal framing:** all impact claims via A/B with holdout — "treatment cohort reduced incurred late fees by X% vs. control over 8 weeks," never correlational.

## How It Helps First Jobbers Decide Better

1. **Visibility:** one screen shows total obligations across every BNPL/app/card/student loan leg KBank can detect — ending fragmentation blindness ("You have 6 active installments totaling ฿4,850/month = 22% of your income").
2. **Honest risk perception:** a calibrated personal risk score replaces the "free money" BNPL illusion with a number they understand — before the first miss, not after.
3. **Decision compression:** when money is short, the app answers the actual question ("which one do I pay first?") with a cost-ranked plan — replacing panic-driven choices with optimization.
4. **Pre-commitment intelligence:** the DSR impact preview at new-leg origination converts an invisible future cost into a visible present tradeoff — the highest-leverage prevention moment.
5. **No shame path:** everything private, in-app, no human contact required — removing the help-seeking barrier identified in the user pain.

## UX Sketch (key screens / moments inside K PLUS)

1. **"หนี้ของฉัน" (My Debts) home card:** total monthly service ฿4,850 · 6 legs · DSR 22% 🟡 · "1 collision week this month."
2. **X-Ray screen:** timeline of all detected obligations — each leg with remaining installments, ends-on date, effective cost, detected-from marker ("auto-detected from your Shopee payments — correct?").
3. **Risk score moment:** "Your miss-risk for the next 90 days: LOW→ELEVATED. Driver: Atome leg 3 + projected ฿900 end-of-month shortfall."
4. **Just-in-time triage (first detected miss):** calm, judgement-free — "We noticed the Atome payment didn't go out. Here's the cheapest order to fix this week. Tapping below shows the plan."
5. **New-leg preview (pre-commitment):** inline insight when a new recurring debit pattern starts: "New installment detected. This takes you to 27% DSR."
6. **Payoff plan:** ranked list with cost-avoided quantified per choice.

## Why This Wins for Track 2

- **The intelligence IS the product.** Remove the models (recurrence classifier, transformer descriptor classifier, delinquency model, optimizer) and nothing remains — this is the purest Track-2 fit in the gap set.
- **Full-stack DS demonstration:** the architecture chains the toolbox's core progression — rules → XGBoost → Thai-language transformers on noisy code-switched data — exactly what DS judges score as applied skill, not just product sense.
- **Realistic data story:** every input is data a bank already holds (transaction legs, servicing records); no fantasy external-data dependency. The NCB enrichment is a scoped Phase-2 with an honest [needs verification] flag — showing data-scoping discipline judges reward.
- **Labeled-supervised ML with measurable ground truth:** delinquency prediction has real labels, real metrics (PR-AUC, calibration, lead time), and a causal online measurement plan — demonstrable rigor, not a heuristic scorecard.
- **Answers a whitespace competitors left open:** no Thai app aggregates installment burden (gap #6); first-mover on a segment KBank explicitly cares about.

## Risks & Limitations

- **PDPA/privacy:** profiling debt position is sensitive. Requires explicit consent flow, purpose limitation (score shown to the user, not used for cross-sell without separate consent), and on-device or masked processing where feasible **[needs verification — specific PDPA sections on profiling/automated decisions if challenged]**. The judgement-free framing demands the score never be shown to third parties (e.g., employer products) — state this as a design principle.
- **Closed-world blind spot:** K PLUS sees only its own legs. BNPL paid from TrueMoney wallet or direct auto-debit from another bank is invisible — the total-burden view is a **lower bound**, and the UX must say so ("based on payments through K PLUS"). Optional consented open-banking/MyData enrichment **[needs verification — Thai open-banking adoption status]**.
- **Detection errors:** a false obligation alarms a shame-sensitive user; a missed obligation defeats the purpose. Mitigation: high-precision thresholds + one-tap correction + "detected from X — correct?" confirmation loop feeding retraining.
- **Cold start:** <6 months history users get pooled cohort estimates with wide bands — honest uncertainty display, not false precision.
- **Model risk / governance:** a risk score shown to users is a regulated-adjacent artifact; needs model-risk documentation, calibration monitoring, drift detection, and human-escalation path for distressed users (link to KBank debt-restructuring contact) — mention as production gating, not hackathon blocker.
- **NCB scope uncertainty:** bureau enrichment for customer-facing wellness is likely not a permissible purpose without consent — designed as optional, consent-gated Phase 2.

## Hackathon Demo Plan (small team, buildable & showable)

**Data:** synthesize a realistic First Jobber transaction dataset (3 personas × 12 months: salary credits, 5–6 recurring legs incl. BNPL with code-switched descriptors, subscriptions, collisions, one deliberate first-miss trajectory). Optionally mask a small hand-labeled public-style set.

**Build (3–4 days for a team of 3–4):**
1. Obligation detector: dictionary rules + XGBoost recurrence features on the synthetic data; show detection of a Shopee installment series including "installment 4 of ~10."
2. Missed-recurrence alarm + LightGBM 90-day risk score with calibration plot; logistic baseline for comparison.
3. Deterministic payoff-order optimizer (avalanche with due-date constraints) — the demo's payoff moment.
4. K PLUS-style mock UI: home card → X-Ray timeline → triage moment → plan. Figma or lightweight web mock; 3–4 screens max.
5. **Demo script (5 min):** "Aim's" month — app detects a new Atome leg (DSR preview), then the Shopee payment doesn't arrive (first-miss trigger), risk score shifts color, app prescribes the cheapest fix with quantified cost-avoided. End on the model card: precision/recall, calibration curve, A/B design for the online metric.

**Stretch (if time):** Thai transformer (WangchanBERTa) classifier on noisy descriptors to show the rules→GBM→transformer progression, and an LLM-narrated explanation of the plan (grounded, deterministic numbers) as the Thai-language shell.

## Sources

Per task method — web search unavailable in this environment; methodology anchors are standard citable works, Thai-specific facts flagged:

1. **Gap synthesis g3** (internal research): SLF delinquency predictability from early-career cash flow; BOT BNPL/personal-loan licensing 2024–25 **[needs verification — circular number/date]**; NCB bureau access scope **[needs verification]**; competitor-store whitespace #6 (no Thai app aggregates installment burden).
2. **DS techniques reference report** (internal): §1 transaction categorization (rules → XGBoost → Thai transformers; WangchanBERTa: Udomcharoenchaikit et al., 2021; Typhoon: SCB 10X **[needs verification — latest release]**); §2 quantile forecasting / pinball loss; §4 PR-AUC/calibration/Qini uplift; §5 contextual bandits & default effects; §8 wellness scoring (CFPB Financial Well-Being Scale — consumerfinance.gov).
3. **Methodology anchors:** XGBoost (Chen & Guestrin, 2016); LightGBM (Ke et al., 2017); isotonic calibration (Platt/Barlow); Isolation Forest (Liu et al., 2008) — only if anomaly framing used; survival/competing-risk basics for 90-day DPD prediction.
4. **Thai-context facts needing pre-pitch verification:** KBank credit-card APR range (kbank.com product pages); BOT BNPL licensing details (bot.or.th); NCB permissible-purpose rules (ncb.co.th); PDPA profiling provisions (pdpc.or.th); K PLUS existing spending-categorization feature name (kplus / KBank annual report, kasikornbank.com IR); Thai BNPL market size (Krungsri Research / Kasikorn Research Center reports — Kasikorn Research (kresearch) is a natural in-house citation **[needs verification — locate kresearch BNPL report]**).
5. **[needs verification — all quantitative Thai market figures]** — no unsourced numbers quoted in this document by design; every specific figure above is either flagged or derived from the internal gap/technique reports.