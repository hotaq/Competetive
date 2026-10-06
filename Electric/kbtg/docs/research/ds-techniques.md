# Research: ds-techniques

> Source: workflow `kbtg_gap_research` (run 2026-09-07, all 11 agents ok, no failures). Raw agent output preserved below. `[needs verification]` flags, if any, are the agent's own — verify before pitching.


Web search is effectively unavailable (search engines and bank sites block scraping from this environment). Per the task method, I'll use reliable domain knowledge and flag uncertain facts with **[needs verification]**. Here is the research output:

---

# Data Science Toolbox for a Thai Retail Bank (KBTG / K PLUS) — First Jobbers Track 2 Research

**Context recap.** K PLUS is KBank's mobile banking app (one of Thailand's largest, ~20M+ registered users **[needs verification — KBank reports cite ~19–20M K PLUS users in recent years; check latest annual report]**). The design target is First Jobbers (ages ~22–28, first years of full-time income): a segment with (a) rising but unstable income, (b) newly acquired financial autonomy, (c) high digital/first-jobber scam victimization via social engineering, and (d) low savings discipline. Track 2 requires the solution's **core value to come from data/analytics/intelligence**, not from UI or security controls.

Below is the applicable DS toolbox. For each technique: what it does, maturity, data it needs, and a concrete First Jobber application. Then: evaluation metrics and the data inventory a Thai bank plausibly already has.

---

## 1. Transaction Categorization & Enrichment

**What it does.** Maps raw transaction records (merchant descriptor strings, transfer counterparties, channel codes) into human-readable spending categories (food delivery, transit, entertainment, "sent to a friend", subscription, etc.) and optionally enriches with merchant metadata (brand, logo, location, recurring-vs-one-off flag). This is the **foundational layer** — almost every other capability (forecasting, wellness scoring, anomaly detection, nudges) depends on clean categories.

**Maturity.** Very high. This is the most commercially proven retail-banking DS application globally (Mint, Revolut, Cleo, YNAB, and Thai apps like MyMo/MeSpend, plus K PLUS's own existing "My Wallet"/spending-summary features **[needs verification — K PLUS already shows transaction categorization in-app; verify the exact feature name and how it's built]**). Mature methods:

- **Rules + dictionary matching** on merchant MCC codes (ISO 18245) and Thai merchant name strings — fast, cheap, explainable; Thai banks already use this.
- **Classical ML text classification** (TF-IDF/BoW + logistic regression, SVM, XGBoost) on descriptors — strong baseline for Thai+English mixed merchant strings.
- **Thai-language NLP embeddings + transformer classifiers** (e.g., PhoBERT-Thai / WangchanBERT / Typhoon-family models **[needs verification — current SOTA Thai LMs change rapidly; cite at demo time]**) for noisy, code-switched descriptors like "GRAB EAT", "ล้างรถ", "SHOPEE TH".
- **LSTM/sequence or entity-resolution models** for recurring-payment detection (subscriptions, gym, phone bills) — key for First Jobbers who leak money via ~10+ small subscriptions.

**Data it needs.** Raw transaction logs (descriptor strings, MCC where available, amount, timestamp, channel), merchant directory (Banks and card networks have MCC; merchants transacting via QR PromptPay have less structured metadata — a known Thai pain point since PromptPay descriptors are often just the registered payee name **[needs verification — PromptPay metadata richness varies by acquirer]**).

**First Jobber use case.** Auto-detect "invisible leaks": categorize then aggregate small recurring subscriptions and delivery-app spend; show "Your Grab/Shopee/Netflix spend grew 40% vs. last month" — the input to nudge engine (§5) and wellness score (§8). Also enables "income vs. spending" classification: detecting salary credits, side-hustle income, and family transfers (money sent home is a real First Jobber behavior in Thailand — remittances to parents **[needs verification — quantify from bank data at build time]**).

---

## 2. Cash-Flow Forecasting (End-of-Month Balance Prediction)

**What it does.** Predicts future account balance / net cash position at a horizon (e.g., day-30 balance) per customer, plus income and expense forecasts. Two granularities:

- **Individual-level:** per-account time-series forecasting (salary cadence, rent day, bill cycles).
- **Segment/portfolio level:** aggregated behavior for product decisions (who will be cash-strapped by day 25).

**Maturity.** High in research (energy/load forecasting literature transfers directly); moderately deployed in fintech apps (e.g., Cleo, Monzo's "left to spend", PFM tools). Banks typically ship simplified versions (rolling averages, "typical month" profiles) rather than full ML forecasting — so a proper ML forecast is a **differentiator but not moonshot risk**.

**Methods, in rising complexity:**

- Baseline: seasonal-naive / day-of-month average profile per user (surprisingly strong for salaried First Jobbers with regular paydays — Thai payroll is commonly paid on a fixed date, e.g., end-of-month or the 5th **[needs verification — typical Thai payroll cycles: 25th–5th; verify]**).
- Classical: exponential smoothing (ETS), SARIMA on daily net-flow with day-of-month seasonality.
- Gradient boosting on engineered features (XGBoost/LightGBM): payday distance, upcoming bills detected from recurring-payment model, recent burn rate, weekday/holiday effects (Thai holidays and Songkran/Payday clustering cause massive predictable spikes in spending — a uniquely Thai feature).
- Deep: sequence models (LSTM/Temporal CNN/Temporal Fusion Transformer) or global deep models across users; Transformers work well but need lots of users and careful backtesting.

**Data it needs.** 6–24 months of daily transaction history per user; payday metadata (or inferred via recurring salary credit classifier); recurring-payment detections (§1); calendar/holiday features.

**First Jobber use case.** **"Projected end-of-month balance"** card: "At your current pace you'll have ฿1,200 left on the 31st — that's ฿800 below your safe buffer." This is the single highest-leverage intelligence feature for First Jobbers: it converts raw data into a *forward-looking decision aid* and drives savings nudges ("move ฿500 out on payday, and you still end the month above your buffer"). Also powers risk flags: predicted negative-balance users → proactive micro-savings or overdraft-warning interventions.

**Key subtlety.** Forecast must include **uncertainty** (prediction intervals), not just a point estimate — "you'll end the month between ฿500 and ฿3,000 (80% interval)" is honest and more useful than a false-precision number. Quantile regression / pinball loss methods fit this well.

---

## 3. Anomaly Detection on Transactions (Unsupervised Methods)

**What it does.** Flags transactions or behavior that deviates from a customer's learned normal pattern. Two distinct applications (keep them separate in your pitch — judges reward precision here):

- **Fraud/scam lens (Track-2-compliant):** *risk scoring and detection signals* that feed alerts/education — NOT blocking rules or auth controls (which Track 2 excludes).
- **Wellness lens:** anomalous *spending* (unusually large entertainment splurge right after payday — "payday binge" detection).

**Maturity.** High; standard in card-fraud stacks (Visa/Mastercard networks and Thai banks all run velocity/anomaly rules). Unsupervised anomaly detection is well established.

**Methods:**

- **Rules/velocity features:** count/amount thresholds per time window, new-payee-first-time-large-transfer — cheap, explainable, catches the "grandma scam" pattern where victim suddenly sends a large sum to a never-before-seen personal account.
- **Statistical / unsupervised:** Isolation Forest, Local Outlier Factor, One-Class SVM, autoencoders on transaction embeddings; per-user density modeling (Gaussian on amount×hour×channel).
- **Behavioral sequence models:** LSTM/Transformer next-transaction predictors; anomaly = low likelihood of the observed sequence (predict-then-score). This is the modern approach and handles "unusual *sequence*" not just "unusual single event."
- **Peer-group (cohort) anomaly:** compare user's behavior against their segment (age band, income band) — "your delivery spending is 3× the median for people with your income profile."

**Data it needs.** Transaction stream with timestamps/amounts/channels/payee identifiers; per-user history to build baselines; peer cohort labels (from §4).

**First Jobber use case (data-driven scam lens, no security controls).** A **pre-transfer behavioral risk score**: unsupervised model flags that this transfer (new counterparty, amount = 40% of balance, initiated 3 minutes after an unusual inbound call-pattern or rapid sequence of transfers, account recently added) is anomalous vs. the user's 12-month profile → app surfaces an *educational interruption and self-check* ("This doesn't look like your usual pattern — here are the 3 scam patterns that look exactly like this; take 30 seconds") rather than a block. This is squarely "data-driven risk reduction through intelligence," not a security control. Also flag "money-mule-like" receiving patterns (many small inbound from many senders, rapid out) on the *counterparty* side — bridges to §6.

**Thai context note.** Thailand's scam crisis is dominated by social-engineering (pig-butchering/investment scams, fake police, parcel-scam calls) where the *victim authorizes* the transfer — so classic card-fraud models (stolen credentials) underperform; the discriminating signal is **behavioral deviation from the victim's own baseline**, which is exactly what unsupervised per-user models give you. Thailand reportedly loses tens of billions of baht per year to scams, with victims frequently transferring willingly via PromptPay **[needs verification — exact figures: Royal Thai Police cyber-crime data and BOT/AMLA reports; do not quote numbers without pulling the current source]**.

---

## 4. Propensity & Segmentation Models (Who Can Save, Who Is at Risk)

**What it does.** Two families:

- **Segmentation (unsupervised):** cluster customers into behavioral archetypes — e.g., "Disciplined saver," "Payday splurger," "Paycheck-to-paycheck," "Subscription leaker," "High scam exposure." Methods: k-means/GMM on behavioral features, hierarchical clustering, or embeddings + HDBSCAN; RFM-style feature engineering generalized to banking.
- **Propensity (supervised):** predict probability of an outcome — *will maintain a savings streak*, *will respond to a nudge*, *is likely to become a fraud victim*, *will churn from the app*, *will take a deposit product*. Methods: logistic regression (baseline, explainable), gradient boosting (XGBoost/LightGBM — the workhorse), calibrated with isotonic/Platt; uplift modeling (two-model or causal forests) to predict *who responds to treatment*, which is strictly better than plain propensity when targeting nudges.

**Maturity.** Very high — core retail-bank marketing analytics for two decades. Uplift/causal modeling is more advanced but well-documented. Not a risk area.

**Data it needs.**

- For segmentation: engineered per-user features (savings rate, income volatility, category mix, payday spending spike index, night-time spending, new-merchant rate, subscription count).
- For "who can save" propensity: outcome labels — e.g., did balance grow over 3 months, did they open/use a savings pocket — which a bank with deposit products already has.
- For "fraud-risk propensity" (Track 2's data-driven fraud lens): labels from reported-scam cases (Thai banks have victim-case data via their fraud operations and the national anti-online-scam infrastructure — Thailand's 1212 OCC / Online Complaint Center and asset-freezing mechanisms generate victim case records **[needs verification — confirm what KBank/KTB share via the Thai Bankers' Association data-sharing scheme on mule accounts and victims]**), plus features: age, account age, transaction sensitivity to urgency patterns, first-time counterparty rate, prior near-miss anomalies.

**First Jobber use case.** Personalize the intervention: "Payday splurgers" get timing-targeted auto-save prompts on salary credit; "at-risk-of-scam" users (high scores on behavioral susceptibility features) get in-app education *before* exposure, and stricter pre-transfer checks in §3. This is the **targeting brain** that makes nudges (§5) efficient instead of spammy.

---

## 5. Behavioral Nudges, Personalization & Experimentation

**What it does.** The decision layer: converts model outputs (§1–§4, §8) into timed, personalized interventions grounded in behavioral economics — and *measures* whether they work with proper experimentation.

**Maturity.** High in fintech product practice; the underlying behavioral-econ principles (hyperbolic discounting, mental accounting, framing, defaults, commitment devices) are well validated. The **experimental discipline** (A/B testing, uplift measurement) is where most teams under-deliver — doing it properly is a judging differentiator.

**Key techniques:**

- **Timing intelligence:** nudges at the *moment of capability* — payday (income lands), pre-payday (spending decision), post-anomaly (teachable moment for scam education). ML picks the send-time per user (contextual bandits / Thompson sampling over nudge variants).
- **Message personalization:** framing chosen per segment (loss-framing for loss-averse segments, streak-gamification for habit-formers); LLM-generated, tone-controlled copy (§7).
- **Experimentation infra:** A/B/n tests, switchback/geo tests where user-level randomization is hard; metrics guardrails (engagement, opt-out rate, savings-rate lift); **uplift modeling** to target only persuadeable users (avoid "sleeping dogs" who react badly to nudges).
- **Commitment devices:** "Save ฿1,000 on every payday" auto-sweep defaults — the classic highest-impact nudge (default effects are among the strongest replicated results in behavioral econ).

**Data it needs.** App event logs (impressions, clicks, feature usage), nudge delivery logs, outcome metrics over time, segment labels (§4).

**First Jobber use case.** The full loop: forecast (§2) says user will dip below buffer → propensity model (§4) says this user is nudge-responsive → on next payday, an auto-save suggestion appears with personalized framing → experiment engine measures 8-week savings-rate lift vs. control. Present this loop to judges as a *learning system*, not a set of popups.

---

## 6. Graph Analytics for Mule-Account Networks

**What it does.** Models accounts and counterparties as a graph (nodes = accounts/merchants/devices; edges = transfers, shared devices/IPs, shared KYC attributes) and detects suspicious **network structures**: money-mule rings, funnel accounts, scam syndicate cash-out chains. Methods: community detection (Louvain/Leiden), centrality measures (PageRank, betweenness) to find funnel hubs, graph embeddings (node2vec, GraphSAGE) + GNN link/role prediction, and motif/shape queries (fan-in/fan-out patterns: many-to-one cash-in then rapid many-out).

**Maturity.** High in AML/fraud practice at large banks globally (standard in AML transaction monitoring modernization); moderately mature in Thailand — Thai banks and the Thai Bankers' Association have been building shared mule-account data exchange following regulatory pressure (the Anti-Money Laundering Office and BOT have pushed inter-bank mule data sharing since ~2023–2024 **[needs verification — confirm current state of the shared mule-account database among Thai banks]**).

**Data it needs.** Transfer graph across accounts (within-bank at minimum; inter-bank via PromptPay/ITMX records the bank sees only its own side — a real limitation to acknowledge), device/IP fingerprints, KYC attributes, historical confirmed-mule labels from fraud operations.

**First Jobber use case (two-sided).**

1. **Protect First Jobbers as mule-recruitment targets:** young, financially stressed, newly banked adults are prime targets for "easy money — rent your bank account" schemes. A model flagging accounts whose behavior starts resembling mule patterns (many small inbound from strangers, rapid outflow, dormant-then-sudden activation) can trigger *preventive education* ("Your account is being used in a way that risks legal liability under Thai law — renting out your account is a crime") before the account is abused. This is data-driven **prevention via intelligence**, not a security control.
2. **Pre-transfer counterparty risk:** when a First Jobber is about to transfer to a new payee, a graph-based "counterparty risk" signal (that account shows mule-like topology) informs the educational interruption from §3.

**Note for scope discipline:** graph analytics is the most impressive Track-2 flex but the most data-dependent; if inter-bank data isn't available, present it as "works on within-KBank graph + shared mule-list lookups" honestly.

---

## 7. LLM-Based Financial Assistants (Grounding, Guardrails, Thai Language)

**What it does.** A conversational interface where the user asks money questions in Thai ("เดือนนี้เงินเหลือเท่าไหร่", "ทำไมค่าใช้จ่ายเดือนนี้เยอะกว่าเดือนก่อน", "สะสมยังไงให้ได้บ้าน") and gets answers grounded in **their own transaction data** and the bank's models (§1–§4, §8 outputs served to the LLM as structured context).

**Maturity.** Rapidly maturing; every major bank is piloting LLM assistants. Risk areas are well known: hallucination (mitigated by grounding/RAG + refusal scope), latency/cost, and compliance. Thai-language capability is now solid: OpenAI-class models handle Thai well, and strong domestic options exist (SCB's Typhoon family, OP ThaiGPT **[needs verification — current names/versions of open Thai LLMs; Typhoon (SCB 10X) is the best-known; verify latest release]**) — being able to run on-prem matters for banking data residency.

**Architecture essentials to show judges:**

- **Grounding (RAG/context injection):** the assistant never computes from memory — balances, forecasts, category breakdowns come from structured model APIs, and the LLM only *narrates* them. Function-calling / tool-use pattern.
- **Guardrails:** scope restriction (only the user's own data + approved financial-education content), no product advice outside approved sets, no "get-rich" content, PII masking, output classifiers, refusal templates; audit logging.
- **Deterministic core, LLM shell:** numbers from deterministic systems; LLM handles language, tone, explanation — a framing judges and regulators both like.
- **Safety for the scam context:** the assistant is a *trusted channel* — a First Jobber who gets a suspicious "police call" can ask the assistant "is this a known scam pattern?" against a curated, retrieval-backed scam-pattern knowledge base (this is data-driven fraud defense via intelligence, not a security control).

**Data it needs.** All of §1–§4 outputs as tool APIs; a curated Thai financial-education/scam-pattern corpus (Thai authorities publish scam-pattern advisories — the Royal Thai Police cybercrime unit and BOT issue public warnings **[needs verification — collect specific advisory URLs at build time]**); conversation logs for evaluation.

**First Jobber use case.** The conversational surface of the whole system: asks questions in natural Thai, receives grounded answers ("You have ฿4,200 left; at your current pace you'll finish the month with ~฿900; your biggest jump was food delivery, +38%. Want to try a ฿200/payday auto-save?") — the LLM is the **presentation and persuasion layer** on top of the DS stack, which keeps the solution Track-2-centered on intelligence rather than chat UI.

---

## 8. Financial Wellness Scoring

**What it does.** A single composite score (and sub-scores) summarizing a customer's financial health: income stability, savings rate, burn discipline, buffer/emergency-fund adequacy, debt-service burden, and (new for this hackathon) **scam-resilience**. Methods range from rules-based composite indices (weighted feature aggregation — like the CFPB's Financial Well-Being Scale methodology, which is a validated survey-based approach) to ML-learned weights, or supervised scoring trained on downstream outcomes (default, distress, savings growth).

**Maturity.** Medium-high. PFM apps ship "financial health scores" routinely; academically validated scales exist (CFPB FWB scale; in banking, products like Empower/Mint scores). Composite scores are part craft (feature selection/weights) part ML — the craft part needs defensible rationale (that's a *feature* for a hackathon: explainability).

**Design guidance:**

- **Sub-scores matter more than one number:** Income Stability (variance/trend of salary credits), Savings Rate (net accumulation / income), Buffer Adequacy (days-of-expenses in balance), Spending Discipline (payday-spike index, category concentration), Debt Load (installment/credit-service ratio — KBank sees installment loans and credit-card payments), **Scam Resilience** (composite of §3 anomaly history, §6 exposure, education interaction).
- **Trajectory over snapshot:** "your score went 58→64 this quarter, driven by buffer days 6→11" — progress framing is itself a nudge (§5) and sustains engagement.
- **Ground each subscore in a model output**, so the score is an *integration layer* over the whole DS stack, not a standalone heuristic — this makes the score itself Track-2 material.

**Data it needs.** Everything above: transaction history, income credits, recurring payments, deposit balances, loan/credit servicing records (within the bank), model outputs from §2/§3/§6.

**First Jobber use case.** The headline metric of the product: a gamified, trajectory-based wellness score with scam-resilience as a first-class pillar — the thing First Jobbers check weekly, the thing nudges move, and the thing the experiment engine optimizes.

---

## Realistic Evaluation Metrics (per capability)

| Capability | Offline metrics | Online / product metrics |
|---|---|---|
| **Categorization (§1)** | Macro/micro F1 per category (macro matters — rare categories like "gambling" or "loan repayment" are the high-stakes ones); per-class precision/recall; coverage (% auto-categorized) | User correction rate; time-to-correct |
| **Cash-flow forecast (§2)** | MAE/RMSE on day-30 balance (in baht, and as % of monthly income); **quantile/pinball loss and empirical coverage of prediction intervals**; backtest across pay-cycle-aligned splits (never random splits on time series) | % users whose end-of-month outcome beats forecast after intervention (the causal measure) |
| **Anomaly detection (§3)** | Precision@k on flagged cases; recall against labeled victim cases (where available); false-positive rate per user per month (this decides whether users keep the app); alert fatigue metrics | Confirmation rate of surfaced warnings; scam-loss avoidance (hard to attribute — use victim-case rates in test cohorts) |
| **Propensity/segmentation (§4)** | AUC-ROC / PR-AUC (PR-AUC for rare outcomes like fraud victimization); calibration (Brier score, reliability curves — crucial if score is shown to users or drives thresholds); cluster stability/silhouette for segmentation; **Qini/AUUC for uplift models** | Conversion lift vs. control; targeting efficiency (response per nudge sent) |
| **Nudges/experiments (§5)** | — | **A/B lift on savings rate and buffer days at 4/8/12 weeks**; engagement, opt-out/uninstall guardrails; long-run habit retention (streak survival curves) |
| **Graph/mule (§6)** | Precision/recall vs. confirmed-mule labels; community purity; hub-detection hit rate | Preventive-contact effectiveness; counterparty-risk precision on live transfers |
| **LLM assistant (§7)** | Grounding faithfulness (answer vs. retrieved facts — RAGAS-style eval); refusal accuracy on out-of-scope prompts; Thai fluency; hallucination rate on numeric questions (should be ~0 by construction); latency/cost per query | Task completion, retention, escalation-to-human rate, complaint rate |
| **Wellness score (§8)** | Subscore-outcome correlation (score vs. subsequent distress/savings growth — predictive validity); score stability (month-over-month volatility should be low absent real changes) | Score trajectory lift in treatment vs. control; user comprehension (survey) |

**Cross-cutting evaluation principles to name in the pitch:** time-based train/test splits (no leakage); cohort evaluation on First Jobbers specifically (models trained on all customers systematically underperform on young, short-history users — short history is itself a modeling constraint worth calling out: use segment-level/pooled models for users with <6 months of history); fairness checks across income bands; and always frame online impact as **causal** (experiment), not correlational.

---

## Data a Thai Bank Plausibly Already Has (the inventory your solution can assume)

**Almost certainly available (core banking + app):**

- **Transaction logs:** every debit/credit with timestamp, amount, channel (branch/ATM/mobile app/PromptPay QR/transfer), counterparty account or merchant descriptor; within-bank transfers have full counterparty identity; inter-bank PromptPay transfers see the payee's proxy (phone/national ID) and display name.
- **Account data:** account age, product portfolio (savings, e-savings, fixed deposit, credit card, installment/personal loans), balances, overdraft events, interest/fee events.
- **App event telemetry:** logins, screens viewed, feature usage, device model/OS, session timing — the raw material for engagement and nudge targeting.
- **KYC:** age, occupation (Thai KYC captures occupation — "employee/company" categories help identify First Jobbers directly), income band (declared at account/loan opening), address region, national ID-linked identity.
- **Payday structure:** recurring salary credits from corporate payroll accounts — enables automatic First Jobber identification (payroll-receiving + age 22–28 + account age < 3 years is a clean heuristic).

**Likely available at a bank of KBank's maturity:**

- **Credit bureau access** (National Credit Bureau/NCB): debt obligations across all banks — crucial for debt-burden subscores (permissible-purpose constraints apply **[needs verification — NCB access scope for non-credit use cases]**).
- **Card network data** for credit-card MCC categorization.
- **Fraud/victim case records:** internal fraud ops, plus shared mule-account lists and victim-case data via the Thai Bankers' Association inter-bank data-sharing scheme established under regulatory pressure to fight online scams **[needs verification — confirm scheme scope, data fields, and recency]**.
- **Existing model outputs:** credit scores, marketing propensity scores, existing in-app categorization (if K PLUS already categorizes spend, your solution builds on/extends it).

**Probably NOT available / constraints to acknowledge honestly:**

- Full inter-bank transfer graph (each bank sees only its leg of PromptPay/ITMX transfers — graph analytics §6 must be scoped to within-bank + shared mule lists).
- Merchant richness on PromptPay QR (descriptors are often just registered payee names — categorization §1 needs Thai-NLP muscle for exactly this reason).
- External data (e-spend outside the bank, e-wallets like TrueMoney/Rabbit Line Pay) — First Jobbers split spend across e-wallets; a "closed-world" assumption is a known blind spot worth flagging, and optionally proposing consented open-banking data sharing (Thailand's Data Sharing platform/MyData and PDPA-consent frameworks are the enabler **[needs verification — current state of Thai open banking/MyData adoption for personal financial data]**).
- PDPA (Thailand's Personal Data Protection Act, effective 2022) governs all of this: user consent, purpose limitation, sensitive-data handling — a compliance-aware design is a credibility signal for bank judges **[needs verification — cite the specific PDPA sections relevant to profiling and automated decision-making if challenged]**.

---

## Sourcing note (for the team)

No web search was available in this environment; the above is domain knowledge with **[needs verification]** flags on every specific figure or claim requiring a current citation. Before the pitch, verify and cite: (1) K PLUS user counts from KBank's latest annual report/SET filing (IR site: kasikornbank.com investor relations); (2) Thai scam-loss statistics from Royal Thai Police cybercrime reports / BOT press releases / Thai Bankers' Association statements; (3) the inter-bank mule-account data-sharing scheme status; (4) current Thai LLM releases (Typhoon/SCB 10X); (5) PDPA and NCB scope details. Methodology anchors that don't need Thai-specific verification: CFPB Financial Well-Being Scale (consumerfinance.gov), uplift/Qini literature, quantile-loss forecasting, Isolation Forest (Liu et al. 2008), Louvain/Leiden community detection — all standard citable methods.

**Bottom line for solution design:** the strongest Track-2 architecture chains the layers — *categorization (§1) → forecasting (§2) + anomaly (§3) → targeting (§4) → intervention + experimentation (§5) → wellness score (§8) as the user-facing integration, LLM assistant (§7) as the Thai-language interface, and graph analytics (§6) as the scam-prevention intelligence edge*. That chain is mature-technology end-to-end (low execution risk), personally meaningful for First Jobbers, and demonstrates the exact DS depth the judges score.