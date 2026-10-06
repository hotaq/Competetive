# Account Guardian (ยามเฝ้าบัญชี) — "รู้ก่อน ป้องกันก่อน ก่อนบัญชีคุณจะกลายเป็นบัญชีม้า" — A data-driven early-warning system that detects when a young account's behavior is drifting toward money-mule patterns and intervenes with plain-Thai education and financial-stress relief *before* the account is abused, not after it's frozen.

> Source: workflow `kbtg_gap_research` (run 2026-09-07, all 11 agents ok, no failures). Raw agent output preserved below. `[needs verification]` flags, if any, are the agent's own — verify before pitching.


# Account Guardian (ยามเฝ้าบัญชี) — "รู้ก่อน ป้องกันก่อน ก่อนบัญชีคุณจะกลายเป็นบัญชีม้า" — A data-driven early-warning system that detects when a young account's behavior is drifting toward money-mule patterns and intervenes with plain-Thai education and financial-stress relief *before* the account is abused, not after it's frozen.

---

## Problem (with evidence from the gap + sources)

**Money-mule account abuse among Thai youth is rising fast, and every existing countermeasure is institution-side — the young person finds out last.**

- Youth mule accounts rose **70.3%**, from **3,371 in January 2025 to 5,741 in March 2026**, tied to **more than ฿185 million in losses** (Nation Thailand, citing the BOT-linked scam-fraud reporting; source: nationthailand.com/news/general/40069312).
- Interdiction today happens only after the fact and at the institutional level: **~1.18M accounts frozen by the DES (Digital Economy and Society Ministry)** and **1.8–1.9M accounts closed via the CFR (Continuous Fraud Fighting framework / Thai Bankers' Association's shared mule-list mechanism)** [needs verification — exact split between DES freezing and CFR closure figures; numbers from the same Nation Thailand report and DES press releases].
- Recruitment mechanics target exactly our segment: syndicates approach cash-strapped young people via social media with "earn easy money by renting out your bank account" offers (scam-fraud report §2). Many recruits don't know that under the **2023 Emergency Decree on Measures for Prevention and Suppression of Cybercrime** (พ.ร.ก. ป้องกันและปราบปรามอาชญากรรมทางคอมพิวเตอร์), renting/selling your account is a **criminal offence** — a life-derailing record, not just a frozen account (scam-fraud report §3).
- The gap synthesis (whitespace gap #5) is explicit: **nothing detects the *transition into* mule-like behavior at the individual level and educates before abuse.** All existing intelligence targets institutions (shared mule lists, network interdiction), not the potential recruit.

**The Track-2 reframing:** this is not a security-control problem — it is a *prediction + persuasion* problem. Can we predict the mule transition from behavioral graph signals, and can we measurably change the outcome with education and stress relief? That is a Data Science problem end to end.

## Target User Insight

**Persona:** "Bank" — 24, first job, ฿22,000/month salary, account opened at university, 8 months into employment. By day 27 of each month his balance hovers near zero. A Telegram post offers ฿3,000 to "let a company use your bank account for transactions." He doesn't think of himself as a criminal; he thinks of himself as broke.

Three insights drive the design:

1. **The motive is financial stress, the mechanism is ignorance.** First Jobbers join mule networks because of end-of-month cash shortfalls (motive) and because they don't know renting an account is a crime under the 2023 Emergency Decree (mechanism). A preventive product must attack *both*: intelligence predicts *who* and *when*; education removes the ignorance; stress relief removes the motive (this links to the cash-flow forecasting and nudge capability — the "g2" stress-relief direction in the gap).
2. **Mule transition is visible in the data before it's visible to anyone else.** Confirmed-mule behavior has a recognizable signature — fan-in from unrelated senders, rapid pass-through, dormancy-then-burst activation, transfers clustered just under interbank limits (BOT imposes age- and channel-conditioned transfer caps on mobile banking accounts [needs verification — exact BOT cap schedule, e.g., limits for accounts opened via non-face-to-face e-KYC; cite the BOT circular at pitch time]). The *recruitment phase* — the account starting to receive test transfers, the user's own financial trajectory deteriorating — precedes the confirmed-mule label by days to weeks. That window is the product.
3. **First Jobbers trust in-app signals more than calls from "the bank" (which scammers imitate).** A calm, plain-Thai, in-app message referencing *their own actual transaction pattern* ("you received 14 transfers from strangers totaling ฿48,000 in 3 days") is credible in a way a generic warning SMS is not.

## Data / ML Approach

### Data sources (plausible K PLUS / KBank data)

| Source | Fields used | Availability |
|---|---|---|
| Core transaction log (within-bank legs) | timestamped debits/credits, counterparty account ID, channel (PromptPay QR / ID transfer / ATM / branch), amount | Almost certainly available |
| Interbank PromptPay/ITMX legs | payee proxy (phone/national ID), display name, bank code, amount, timestamp — KBank sees its own customer's side | Available, but only one leg (acknowledge honestly) |
| Shared mule-account lists (Thai Bankers' Association / CFR mechanism) | confirmed-mule account identifiers — **the label source** | Likely available to KBank as a TBA member [needs verification — field-level scope and refresh cadence of the shared list] |
| KYC | age, occupation (Thai KYC captures occupation — "employee"), income band, account age | Almost certainly available |
| App/device telemetry | device fingerprint, OS, login cadence, session timing; new-device registrations | Almost certainly available |
| Account status events | freeze/closure orders received from DES/CFR | Available via fraud ops |
| Cash-flow forecast output (from the §2 forecasting capability) | predicted end-of-month shortfall — the *stress* input to the susceptibility score | Model output, built in-house |

**Honest scoping note for judges:** the full interbank transfer graph is not visible to any single bank; the graph model operates on the **within-KBank graph + shared mule-list lookups on counterparties**, exactly as flagged in the DS techniques report (§6). This is a real constraint, stated up front — not hidden.

### Features & signals

Structured into the four behavioral phases of mule recruitment (this framing itself demonstrates problem understanding):

**A. Personal-context susceptibility features (the recruit profile):**
- Age band (18–24 / 25–28), account age (months since opening), "first-salary account" flag (payroll credit detected within last 12 months + age ≤ 28)
- Financial stress: predicted end-of-month shortfall (฿ and days-of-expenses), buffer days (balance ÷ typical daily spend), income volatility, recent overdraft/fee events, debt-service ratio (within-bank installments + card)
- Prior near-misses: count of past anomaly flags (new-payee large transfers, urgent-sequence transfers) that self-resolved

**B. Onset features (recruitment/testing phase — small, probing):**
- First-time-inbound from never-seen counterparties, small amounts (฿100–2,000 test transfers)
- Rise in *distinct new inbound counterparties per week* (fan-in velocity, not volume — the discriminating early signal)
- Counterparty-side lookup hits: inbound senders already on the shared mule list or with mule-like topology

**C. Transition features (the mule signature):**
- **Fan-in:** distinct unrelated inbound senders per rolling 7 days; sender concentration (HHI) collapsing toward zero
- **Pass-through layering:** inbound→outbound time lag distribution (median lag < 1 hour), retention ratio (balance kept ≈ 0%), out-hop count
- **Dormancy-then-burst:** low-activity baseline for 30+ days then sudden activation (z-score of daily transaction count vs. personal 6-month baseline)
- **Near-cap clustering:** transfers binned just under BOT/mobile-banking limits (e.g., amounts in [cap−5%, cap]) [needs verification — exact cap values], heavy interbank outflow immediately after intrabank inflow
- **Device-graph fingerprints:** new device linked to account, device shared across multiple accounts (shared-device edge in graph), rapid payee-list additions

**D. Graph-structural features (per-account, computed on the within-bank transfer graph):**
- In-degree / out-degree ratio, in-degree trend slope
- Betweenness/PageRank of the account within its transaction community (funnel-hub centrality)
- Community (Louvain/Leiden) composition: fraction of the account's ego-network already confirmed-mule
- node2vec / GraphSAGE embedding distance to known-mule embedding clusters (distance-to-centroid of mule class in embedding space)

### Model / technique

**Two models, two jobs — keep them separate (judges reward this precision):**

**Model 1: Mule-transition risk model (supervised, the detection engine).**
- **Task:** predict P(account becomes confirmed-mule within next 30 days), trained on confirmed-mule labels from the shared list + DES freeze events, with a **time-decay label window** (an account's 30 days *before* its confirmation date is a positive-labeled window — this trains the model on the *pre-transition* signature, which is the whole point).
- **Architecture:** (a) **GraphSAGE** (inductive GNN) on the within-bank transfer graph producing 128-d node embeddings — inductive so new/young accounts embed without retraining (built-in cold-start mitigation at the graph level); (b) **LightGBM** classifier on concatenated [graph embedding ‖ behavioral features A–D], chosen for tabular robustness, class-imbalance handling (`scale_pos_weight` / focal loss), and native SHAP explainability.
- **Temporal sequence component:** for dormancy-then-burst and lag patterns, a lightweight per-user temporal feature bank (rolling z-scores, lag distributions) rather than a full sequence model — justified at hackathon scale; a Transformer/TPP head is the stated v2 path.
- **Training discipline:** strict **temporal split** (train on Jan–Dec 2024 labels, validate on Q1 2025, test on Q2 2025 — never random splits); positive class is rare (thousands of confirmed cases among millions of accounts), so **PR-AUC** is the headline metric, not ROC-AUC.
- **Cold start (accounts < 6 months history — most First Jobbers):** pooled segment model — behavioral z-scores computed against an **age-band × account-age cohort baseline** instead of personal history, plus the inductive GNN embedding (needs no personal history). Explicitly call out that models trained on all customers underperform on young, short-history users, so cohort-pooled baselines are a first-class design choice, not a fallback.

**Model 2: Susceptibility score (composite, the targeting engine).**
- A calibrated **logistic regression / monotonic GBM** on personal-context features (A) + prior anomaly history, predicting *propensity to be recruited* — trained as a proxy task on *near-miss* accounts (those that showed onset features but never transitioned) and confirmed mules, since direct "was approached by a syndicate" labels don't exist. Framed honestly as a **proxy-labeled screening score** (0–100, isotonic-calibrated), whose job is *targeting of preventive education*, never accusation.
- **Uplift layer:** education and stress-relief offers are targeted with a **two-model uplift estimator** (T-learner, causal-forest as v2) on users segmented by susceptibility × stress — so we only contact *persuadeables* and avoid alarming the unconcerned ("sleeping dogs" effect). Evaluation via **Qini coefficient / AUUC**.

**Explainability:** every alert message is generated from **SHAP top-3 contributing features**, translated into plain-Thai behavioral language by a template system (deterministic templates + optional Thai LLM narration with grounding — the "deterministic core, LLM shell" pattern). The user sees *their own data* ("คุณได้รับเงินโอนจากคนแปลกหน้า 14 ราย ใน 3 วัน"), never a score or a model name.

### Personalization logic

Three-tier intervention ladder keyed to (transition risk × susceptibility × stress):

| Tier | Trigger | Intervention |
|---|---|---|
| **Green (pre-emptive)** | High susceptibility + high predicted end-of-month shortfall, no onset signals | Silent stress relief: payday auto-save suggestion, budget nudge, access to low-cost credit alternatives — *remove the motive before the syndicate arrives* |
| **Yellow (onset)** | Onset features detected (new-counterparty fan-in velocity rising) | Soft-touch in-app card: "พฤติกรรมบัญชีคุณกำลังเปลี่ยน — รู้ไหมว่าให้คนอื่นใช้บัญชีเป็นความผิดอาญา" + 60-second interactive explainer of the 2023 Emergency Decree liability (fines + imprisonment) + one-tap "I've been approached" self-report |
| **Orange (transition)** | Transition + graph features firing (pass-through signature, mule-community proximity) | Hard educational interruption at the next outbound transfer: full-screen self-check, transaction-pattern recap from their own data, helpline, explicit plain-Thai statement of personal criminal liability — and an offer to freeze out-transfers *at the user's own request* (user-initiated, therefore education + user control, not a bank-side security control) |

Message tone and channel are personalized by segment (from the §4 segmentation capability): streak-gamification framing for habit-formers, loss-framing for loss-averse profiles, urgent-framing only at Orange.

### Evaluation metrics & success measurement

**Offline (Model 1 — mule transition):**
- **PR-AUC** and Precision@k at operational alert budgets (e.g., precision at top 0.1% of young accounts — the realistic alert volume a prevention team can handle)
- **Recall@30d** against confirmed-mule labels; **lead time** distribution (days between first Orange flag and the confirmed-mule date — the metric that proves "prevention," target: median ≥ 7 days)
- Calibration (Brier, reliability curves) — mandatory since thresholds drive user-facing actions
- FP rate per user per month (< 0.5% — alert-fatigue guardrail), fairness check across income bands
- Graph component: community purity vs. confirmed labels; hub-detection hit rate

**Offline (Model 2 — susceptibility):** PR-AUC vs. near-miss proxy labels; **Qini/AUUC** for the uplift layer; cluster stability.

**Online / product (the causal evidence):**
- **Preventive-contact effectiveness:** mule-confirmation rate in treatment vs. holdout cohort at 90/180 days (the gap synthesis's own suggested evaluation) — measured as a proper randomized experiment
- Self-report rate ("I've been approached") per Yellow-tier contact — an early leading indicator
- Stress-relief uptake and end-of-month shortfall reduction in Green tier (links to the savings outcome)
- Score/flag stability month-over-month; opt-out and complaint rates as guardrails

## How It Helps First Jobbers Decide Better

1. **It converts invisible, catastrophic risk into a visible, actionable decision.** Today the First Jobber's first "data point" is a frozen account or a police summons. Guardian surfaces the risk *at the decision moment* — with their own transaction pattern as evidence — so "should I let this 'company' use my account?" becomes an informed decision backed by the actual criminal-liability facts.
2. **It removes the motive, not just warns about the mechanism.** The susceptibility model identifies *why* they're a target (predicted shortfall) and the Green tier actually fixes that reason — payday auto-saves, spending nudges, cheaper credit alternatives. Prevention through prosperity, which is also the strongest savings-behavior story for the money-management half of the challenge.
3. **It teaches the law in 60 seconds instead of a courtroom.** Most recruits don't know the 2023 Emergency Decree makes account rental a crime; a personalized, evidence-backed in-app explainer at the right moment is cheap education with asymmetric value (฿3,000 rental fee vs. criminal record).
4. **It protects the counterparty side too:** the pre-transfer signal for a First Jobber *sending* money into a mule-topology account doubles as scam education (the same graph intelligence, two educational uses — no blocking rules involved).

## UX Sketch (key screens / moments inside K PLUS)

- **Guardian Home Card (subtle, always-on):** a small shield icon with a status color inside the existing K PLUS home. Green default: "บัญชีคุณปลอดภัย" + this month's stress forecast ("เหลือพอถึงสิ้นเดือน ✓ / ขาด ~฿1,200 วันที่ 28").
- **Yellow moment (in-app card, non-intrusive):** "สังเกตเห็นการเปลี่ยนแปลงในบัญชีคุณ" → 3 facts from their own data → 60-second interactive card on the 2023 Emergency Decree → "มีคนมาชวนให้ใช้บัญชีไหม?" self-report button → one-tap stress-relief alternatives.
- **Orange moment (full-screen at next transfer attempt):** plain-Thai recap: "3 วันนี้คุณรับเงินจากคนแปลกหน้า 14 ราย ฿48,000 และโอนออกเกือบทั้งหมดภายใน 1 ชั่วโมง" → "การให้ผู้อื่นใช้บัญชีของคุณมีโทษตามกฎหมาย (พ.ร.ก. 2566)" → choices: [ปรึกษาฝ่ายกันโกง 24 ชม.] [หยุดโอนออกชั่วคราว — คุณสั่งเอง] [ดำเนินการต่อ แต่รับทราบความเสี่ยง].
- **Post-event debrief:** if the user self-reports an approach or an anomaly self-resolves, a short "what almost happened" story screen — the teachable moment that builds the scam-resilience pillar of their financial-wellness score.

## Why This Wins for Track 2

1. **It is the DS showpiece the gap synthesis itself called out:** graph analytics (GNN, community detection, centrality) + supervised rare-event modeling with confirmed-mule labels + uplift-targeted intervention — "the DS 'showpiece' that stays Track-2 compliant by educating rather than blocking." No other competitor is likely to bring graph ML *and* causal targeting to the mule problem.
2. **Every layer is a named, citable technique with a realistic data story:** GraphSAGE + LightGBM + SHAP + isotonic calibration + T-learner uplift + cohort-pooled cold start. Judges scoring "application of Data Science skills" see depth at every layer, plus the maturity to know *why* (PR-AUC over ROC-AUC for rare classes; temporal splits; calibration before user-facing thresholds).
3. **The evaluation is causal, not correlational:** the headline claim — "preventive contact reduces mule confirmations" — is designed as a randomized experiment with Qini-based targeting, exactly the experimental discipline the DS techniques report identifies as where teams under-deliver.
4. **The framing shows problem understanding, not just tech:** the four-phase behavioral model (susceptibility → onset → transition → confirmation) demonstrates that the team understood the *recruitment mechanics*, and the two-sided design (motive + ignorance) shows understanding of the First Jobber. Track-2 compliance is explicit: intelligence educates and the user decides; no blocking, no auth controls.

## Risks & Limitations

- **PDPA / privacy (Thailand's Personal Data Protection Act, effective 2022):** behavioral profiling for risk scoring is sensitive. Mitigations: explicit purpose declaration in consent flow, on-device display of only the user's own data, no sharing of individual risk scores externally, retention limits, DPIA before production. Profiling of *potential* criminality is ethically fraught — the design deliberately keeps the score internal, shows evidence-not-scores, and makes all consequential actions user-initiated [needs verification — specific PDPA sections on profiling/automated decisions if challenged by judges].
- **Label quality & availability:** confirmed-mule lists are the ground truth, but they contain the *caught* population — survivorship bias (sophisticated mules that evade freezing are negative-labeled). Mitigation: PU-learning framing (treat unlabeled as unknown, not negative) is the honest v2 path; state it.
- **Graph blindness beyond KBank:** only one leg of interbank transfers is visible; fan-in from other banks is partially observable. Mitigation: scope to within-bank + shared-list lookups; acknowledge as a stated limitation (also true for every competitor, and honesty scores points).
- **Cold start:** most First Jobbers have < 2 years of history. Mitigated by cohort-pooled baselines and inductive GNN embeddings — but early-life model precision will be lower; the Yellow tier's soft touch is calibrated for this uncertainty.
- **False-positive harm:** wrongly implying a young user is becoming a criminal is reputational and emotional damage. Mitigations: Orange tier requires *multiple* independent signal families (graph + behavioral + temporal) to fire; alert-fatigue guardrail < 0.5%/user-month; human-review option before hardest language; complaint-rate guardrail in the experiment.
- **Model risk governance:** a model that influences fraud-related customer contact needs model-risk documentation (validation, monitoring, drift detection) — name it to bank judges; it signals maturity.

## Hackathon Demo Plan (what a small team can actually build & show)

**Team of 3–4, ~2 days of build:**

1. **Synthetic data generator (0.5 day):** Python script generating ~100k accounts × 6 months of transactions: 70% normal First Jobbers (payroll cadence, subscription leaks, payday spikes), 20% stressed-but-clean (low buffer), 5% near-miss, 5% synthetic mules implementing the real signature (fan-in from random senders, < 1h pass-through, dormancy-then-burst, near-cap clustering, shared-device edges). Generator doubles as the honest answer to "where do you get labels?"
2. **Graph + model pipeline (1 day):** NetworkX/StellarGraph graph construction → GraphSAGE embeddings → LightGBM on [embeddings ‖ features] → temporal split → **show the PR-AUC / lead-time chart live**. Show SHAP explanation for one flagged account — this is the demo money-shot: "the model caught this account 11 days before its synthetic confirmation."
3. **Susceptibility + uplift (0.5 day):** proxy-task susceptibility score; simulated A/B on the synthetic population showing mule-confirmation rate 5.0% (control) → 1.8% (treated) with a Qini curve — the causal-evidence slide.
4. **UX mock (0.5 day):** Figma/clickable mock of the three K PLUS moments (Green card, Yellow card, Orange full-screen with plain-Thai Decree explainer). All text pre-written in Thai.
5. **Narrative assets:** one system-architecture diagram, one four-phase behavioral-model diagram, the honest-limitations slide (interbank blindness, PDPA, survivorship bias + PU-learning v2).
6. **Optional flex if time allows:** template-generated Thai explanation of a flagged account, grounded in the SHAP features — "deterministic core, LLM shell."

**What the judges see in 5 minutes:** a live model catching mule transitions before confirmation, a measured preventive-contact effect, a real product surface in Thai, and a limitations slide that reads like a bank's own model-risk team wrote it.

## Sources

- **Nation Thailand — youth mule account statistics** (+70.3%, 3,371 → 5,741 accounts Jan 2025 → Mar 2026; >฿185M losses; DES ~1.18M frozen accounts; CFR account closures): nationthailand.com/news/general/40069312 (as cited in the team's scam-fraud report §1)
- **2023 Royal Emergency Decree on Measures for Prevention and Suppression of Cybercrime (พ.ร.ก. ป้องกันและปราบปรามอาชญากรรมทางคอมพิวเตอร์ พ.ศ. 2566)** — criminal liability for renting/selling bank accounts [needs verification — cite the Royal Gazette text or an official English summary at pitch time]
- **Team gap synthesis & DS techniques reference report** (g5 gap point; §6 graph analytics; §4 propensity/uplift; §5 experimentation; evaluation-metrics table; Thai bank data inventory)
- **Methodology anchors (no Thai-specific verification needed):** GraphSAGE (Hamilton et al., NeurIPS 2017); LightGBM (Ke et al., NeurIPS 2017); Isolation Forest (Liu et al., 2008); Louvain/Leiden community detection; SHAP (Lundberg & Lee, 2017); uplift/Qini (Radcliffe, 2007); PU-learning (Elkan & Noto, 2008)
- **[Needs verification before pitch]:** BOT age/channel-conditioned mobile-banking transfer caps (exact schedule and circular reference); Thai Bankers' Association shared mule-list data fields and cadence; DES/CFR figure split; PDPA profiling provisions; KBank internal data availability for shared-mule-list lookups