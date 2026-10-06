# GornOon (ก่อนโอน — "Know Before You Transfer") — A personalized, pre-transfer deviation score that turns the moment of a scam transfer into a 30-second intelligent self-check, instead of a generic "this may be a scam" banner you've already learned to ignore.

> Source: workflow `kbtg_gap_research` (run 2026-09-07, all 11 agents ok, no failures). Raw agent output preserved below. `[needs verification]` flags, if any, are the agent's own — verify before pitching.


# GornOon (ก่อนโอน — "Know Before You Transfer") — A personalized, pre-transfer deviation score that turns the moment of a scam transfer into a 30-second intelligent self-check, instead of a generic "this may be a scam" banner you've already learned to ignore.

## Problem (with evidence from the gap + sources)

Thai anti-scam machinery — the Central Fraud Registry, AOC 1441 (the Royal Thai Police's 24/7 online scam complaint center), DE-fence, BOT's 2023 transfer-control rules — is **institution-centric, rule/threshold-based, reactive, and post-transfer** (BOT, bot.or.th/en/news-and-media/news/news-20230309.html). The customer-facing layer inside apps like K PLUS is a generic warning banner with no personalization and no use of the individual's behavioral baseline — the equivalent of a smoke detector that says "fire is dangerous" instead of "your kitchen is on fire."

Meanwhile the victimization data is unambiguous about *who* bears the cost:

- Ages **21–30 are the most frequently victimized age group** in Thai Police/ACSC weekly scam data (nationthailand.com/news/general/40069932).
- Average loss per victim is **over ฿52,000** (Thai Chamber of Commerce, nationthailand.com/news/general/40069582) — roughly **1–3 months of a first salary**, financially catastrophic for a First Jobber with a realistic ฿2,000–6,000 monthly surplus. One authorized transfer erases months of saving progress.
- Investment scams — the costliest category at roughly **~30.7% of weekly losses [needs verification — figure from team synthesis of ACSC weekly reports; re-pull the current week before the pitch]** — are precisely the type engineered for first-jobber aspiration ("start investing with ฿5,000").

The core insight: in Thailand's scam crisis, the victim **authorizes** the transfer. Credential-fraud models don't fire because there is no stolen credential — the discriminating signal is *behavioral deviation from the victim's own baseline*, combined with *who is receiving the money*. No Thai bank currently computes either signal into a personalized, pre-transfer, customer-facing decision aid. Revolut's first-time-payee warnings prove the mechanic works — but nothing like it exists in Thailand (competitor scan, whitespace gap #5). That's our gap, and it's squarely a Data Science gap, not a UX or security gap.

## Target User Insight

**"Ploy", 24, brand analyst, ฿28,000/month.** Salary lands on the 27th. She auto-saves ฿3,000/payday into a K PLUS e-savings pocket and is proud of her ฿18,000 goal toward a MacBook. At 22:40 on a Sunday, a "SEC-authorized investment advisor" she's been chatting with on LINE for 3 weeks tells her the arbitrage window closes tonight. She adds the payee, transfers ฿15,000 — 83% of her liquid balance — to a personal account she's never paid before. K PLUS shows the generic banner. She's seen it 200 times. She taps confirm.

Three facts about Ploy that current solutions ignore:

1. **Her baseline is unusually learnable.** First Jobbers have salary-anchored inflows and category-stable outflows — the *ideal* segment for per-user baselines. Deviation is therefore high-signal, unlike for irregular-income customers where anomaly models drown in noise.
2. **The scam steals more than money.** The same transfer that empties her account also breaks her savings trajectory and kills goal engagement — she abandons the goal and the app's money-management features. Scam prevention *is* money management for this segment.
3. **At the moment of transfer she is under social-engineering pressure and deciding alone.** What she needs isn't a block (she'd complete the transfer by another rail) — it's a personalized, credible interruption that breaks the scammer's urgency script with her *own* data.

## Data / ML Approach

The product is a single fused **Pre-Transfer Deviation Score (0–100, calibrated probability)** computed in <300ms at confirm-screen time, from three independent model families.

### Data sources (plausible K PLUS / bank data)

- **Transaction stream:** 12–24 months of per-user history — timestamps, amounts, channel, counterparty identity (full identity for within-KBank transfers; proxy + display name for inter-bank PromptPay/ITMX legs — a known visibility limit to acknowledge honestly).
- **Payee graph:** within-KBank transfer graph (nodes = accounts, edges = transfers), plus shared mule-account lists from the Thai Bankers' Association inter-bank data-sharing scheme [needs verification — confirm current scheme scope and field-level access at build time].
- **Labels:** internal fraud-ops case records + victim case records via AOC 1441 / the inter-bank complaint scheme [needs verification — field availability and latency of victim-case data sharing].
- **Savings/goal telemetry:** auto-saver enrollment, sweep streaks, goal balance trajectory, goal-abandonment events (K PLUS product data).
- **KYC:** age, occupation, income band — First Jobber identification (payroll-receiving + age 22–28 + account age < 3 years is a clean heuristic).
- **Reference corpus:** licensed-entity registry for payee-name matching (SEC/BOT licensed firms, official names) [needs verification — data access route for the registry and refresh cadence].

### Features & signals

**A. Personal deviation features (per-user baseline, the "self anomaly" family):**
- First-time-payee flag + amount-vs-user-transfer-median ratio (z-score of log-amount against user's own distribution of personal transfers)
- Timing: transfer initiated 01:00–05:00, or within N minutes of payee-add (velocity from add→confirm)
- **Threshold-avoidance signature:** repeated transfers clustered just under known friction thresholds (~฿49,000 under the biometric/review trigger) — a strong coercion fingerprint
- **Escalation pattern:** 2+ new payees within 72h with rising amounts (classic pig-butchering cash-out ladder)
- **Savings-trajectory break:** auto-saver streak interrupted + savings pocket out-transfer (a uniquely KBank-able feature — the victim's own good habits are the anomaly signal)
- Account-age-weighted novelty: new payee for a 6-month account ≠ new payee for a 6-year account

**B. Counterparty graph features (the "who receives" family):**
- Fan-in: distinct senders / distinct sender-peer-groups into the payee in trailing 30/90 days
- Pass-through ratio: outflow within 24h of inflow / inflow
- Dormancy-burst: payee account dormant > 6 months, then sudden activity burst
- Graph embedding (node2vec or GraphSAGE over the within-KBank transfer graph) → distance between payee embedding and confirmed-mule cluster centroids; shared community (Louvain/Leiden) with known mule accounts
- Mule-list membership / mule-list proximity lookup

**C. Payee identity-plausibility features (the "impersonation" family):**
- Fuzzy Thai/English name matching of payee display name against licensed-entity registry — "หลักทรัพย์..." that is *not* a licensed broker's official name
- Personal-account flag: receiving account is a personal account while the conversation context implies an institution (detectable from B+C jointly)
- Name-vs-behavior contradiction: entity-looking name + mule-topology graph behavior

### Model / technique

Three layers, one calibrated fusion:

1. **Personal anomaly scorer:** per-user probabilistic baseline — a two-part design: (a) engineered deviation features (above) scored by a **pooled LightGBM** trained on labeled victim cases + confirmed-benign transfers, and (b) an unsupervised per-user density/sequence model (Isolation Forest per cohort, or an LSTM/Transformer next-transaction predictor where anomaly = low sequence likelihood) as a cold-start-independent backstop. Ensemble both; the unsupervised leg catches novel patterns with no labels, the supervised leg optimizes precision on known ones.
2. **Counterparty graph scorer:** GraphSAGE link/role model over the within-KBank graph, trained on confirmed-mule labels; motif queries (fan-in→rapid-out, dormancy-burst) as deterministic high-precision rules layered on top.
3. **Fusion meta-model:** gradient boosting over the three scorers' outputs + context features, **calibrated (isotonic)** so the surfaced number is an honest probability, not a raw score. SHAP-based per-feature attribution drives the personalized explanation ("never paid this person before · 71% of your balance · 23:40").

**Cold-start handling (explicit, because First Jobbers have short histories):**
- < 3 months history: pooled/cohort baselines — "typical First Jobber" segment profiles built from users with matching KYC + payroll signature; score leans on counterparty graph + entity-matching legs, which are user-history-independent.
- 3–12 months: hybrid — cohort priors shrink as personal history accumulates (empirical-Bayes-style blending).
- Payee-side cold start is the *strength*: a brand-new payee is fully scorable via the graph and name-matching legs even on the user's first day.

**Training discipline:** time-based train/test splits only (no leakage across the victim-label window); class imbalance via PR-AUC-oriented optimization and focal loss alternatives; per-segment evaluation on First Jobbers specifically (models trained on all customers systematically underperform on young, short-history users).

### Personalization logic

The score doesn't just decide *whether* to interrupt — it decides **what the interruption says**, matched by an unsupervised segment model (k-means/GMM on behavioral features) + a lightweight scam-type classifier (which of the top-3 scam families this transfer's signature most resembles):
- Signature ≈ investment/pig-butchering → self-check content targets "advisor" narratives, shows the licensed-entity mismatch
- Signature ≈ fake-police/authority → content targets authority scams, offers the AOC 1212/1441 verification path
- Signature ≈ threshold-avoidance/escalation → shows the user's own transfer ladder ("3 new people, 4 days, rising amounts — this is a known cash-out pattern")
- Severity tiers: low score → silent log; medium → inline nudge; high → 30-second self-check + 10-minute cool-off timer; extreme (mule-graph + savings-drain + night + first payee) → cool-off + checklist + direct AOC contact info. **Never a block** — decision support, not a security control (Track 2 carve-out).

### Evaluation metrics & success measurement

- **Offline:** precision@k on flagged transfers (ops can only review so much); **recall vs. labeled victim cases** (AOC 1441 / internal case records); **false-positive rate per user per month — the alert-fatigue budget that decides retention** (design target: < 1 unnecessary interruption per user per 3 months); calibration (Brier score, reliability curves — mandatory since the number drives user-facing tiers); PR-AUC over time-decayed test windows; segment-sliced metrics for First Jobbers specifically.
- **Online (causal, not correlational):** A/B test the friction moment — treated vs. control cohorts, measure **victim-case rate reduction** (from shared victim-case records), transfer-completion-rate delta post-intervention, and guardrails: abandonment of *legitimate* transfers, opt-out rate, complaint rate. Secondary: scam-resilience sub-score trajectory (linking to the team's wellness-score idea if the portfolio merges).

## How It Helps First Jobbers Decide Better

It moves intelligence to the exact moment a catastrophic decision is being made, and arms the user with the three things a scammer's script is designed to strip away: **time** (cool-off timer breaks urgency), **self-knowledge** ("this is 83% of your balance and breaks your 4-month saving streak — in your own data, you've never done this"), and **verification paths** (licensed-entity mismatch, AOC 1212/1441 contact, "hang up and call back on the official number"). For a segment losing an average of >฿52,000 per incident — months of surplus in one tap — preventing a single authorized transfer does more for a First Jobber's savings trajectory than a year of budgeting nudges. The savings-trajectory-break feature closes the loop: the product protects the savings habit the money-management track is trying to build.

## UX Sketch (brief: key screens / moments inside K PLUS)

1. **Confirm-screen moment (the core):** transfer review screen shows a soft amber panel with the personalized deviation explanation — three bullet points from SHAP-top features, written in plain Thai: "คุณไม่เคยโอนให้บัญชีนี้มาก่อน · จำนวน 71% ของยอดเงินคงเหลือ · เวลา 23:42 น." Buttons: **[โอนต่อ (Continue)] / [เช็กก่อน 30 วินาที (30-second self-check)]**. Continue stays one tap — never a block.
2. **30-second self-check:** three swipeable cards showing the top-3 scam patterns that most resemble *this specific transfer's signature* (from the scam-type classifier), ending with a live verification: "ชื่อบัญชีนี้ไม่ตรงกับบริษัทที่ขึ้นทะเบียนกับ ก.ล.ต." where entity-matching fires.
3. **Cool-off timer (high-severity only):** "พัก 10 นาที แล้วค่อยคุยกับคนที่คุณไว้ใจ" — 10-minute gentle lock with one-tap "call someone" affordance and AOC 1441 contact.
4. **Post-decision learning:** if the user proceeds, a next-day follow-up: "ยอดโอนเมื่อคืนเป็นอย่างไรบ้าง?" feeding the label loop; if they cancelled, a +1 to their in-app **scam-resilience** progress (habit reward, not shame).
5. **Silent mode:** low scores never interrupt — the model logs and learns.

## Why This Wins for Track 2

- **It is the carve-out, executed literally:** predictive risk scoring, anomaly detection, behavioral modeling, and graph intelligence that *informs the customer* — education and decision support, never a security control. The bot commands zero blocking authority; every intervention is data + persuasion.
- **Full DS stack in one feature:** supervised scoring (LightGBM + calibration), unsupervised per-user baselines (Isolation Forest / sequence likelihood), graph learning (GraphSAGE, Louvain, motif rules), entity resolution (Thai fuzzy name matching), and causal online evaluation (A/B with victim-case-rate endpoints). That's breadth *and* depth for DS judges.
- **Segment-model fit is the clever part:** judges will notice that we didn't apply generic anomaly detection — we exploited the fact that First Jobbers' salary-anchored baselines make personal deviation unusually high-signal, and that savings-trajectory-as-fraud-feature links the two hackathon goals in one feature.
- **Honest engineering:** we scope the graph to within-KBank + shared mule lists, handle cold start explicitly, and put alert-fatigue FPR on the same pedestal as recall — the exact trade-off literacy that separates practitioners from slideware.
- **Whitespace with a proven mechanic:** Revolut's transfer-time first-time-payee warnings validate the UX pattern exists nowhere in Thailand — first-mover for K PLUS, grounded in mature techniques (low execution risk, high differentiation).

## Risks & Limitations

- **PDPA/privacy:** profiling individual behavior and scoring counterparties the user interacts with is personal-data processing — requires clear consent basis, purpose limitation, and transparency about what the score uses (showing SHAP-style reasons is both UX and compliance win). Counterparty scoring processes another person's data in the context of the user's transaction — needs legal review of the legitimate-interest/consent framing [needs verification — PDPA sections on profiling and automated decision-making; cite precisely if challenged].
- **Data availability:** the inter-bank victim-case and mule-label pipelines are the model's label lifeline — if latency or field access is limited, supervised legs degrade; mitigation is the unsupervised/graph/motif legs, which need no victim labels.
- **Graph visibility ceiling:** KBank sees only its own leg of inter-bank PromptPay transfers — counterparty risk for payees at other banks relies on shared mule lists + proxy/display-name matching, which is weaker. State this openly; it is a structural industry limit, not a design flaw.
- **Adversarial drift:** scammers adapt (amount structuring below thresholds is exactly why threshold-avoidance is a feature, but it will be gamed too). Requires continuous retraining + drift monitoring on score distributions.
- **Cold start for brand-new First Jobbers:** mitigated by cohort priors + payee-side scoring, but personal-deviation precision genuinely is lower in the first 3 months — communicate confidence honestly (suppression of high-uncertainty alerts protects the alert-fatigue budget).
- **False-positive cost asymmetry:** interrupting a legitimate rent transfer erodes trust; the severity-tier design and the <1/3-months FP budget exist precisely for this. Still the #1 product risk — measure it in the A/B guardrails.
- **Model risk governance:** a calibrated probability driving user-facing friction needs model-risk documentation, monitoring, and fallback to rules if the model degrades — a credibility point with bank judges, worth one slide.

## Hackathon Demo Plan (what a small team can actually build & show)

**Scope: a 6-minute live demo of the decision moment + the model behind it.**

1. **Synthetic data generator (1 day):** simulate 5,000 First Jobber profiles (salary cadence, category-stable spend, savings sweeps) + inject scripted scam narratives (pig-butchering ladder with ฿49,000 threshold avoidance, fake-police night transfer, savings-drain) + mule subgraph (fan-in/pass-through/dormancy-burst) within a generated transfer graph. Fully PDPA-safe since it's synthetic.
2. **Real model pipeline (2–3 days):** LightGBM deviation scorer + Isolation Forest per-user baselines + networkx/igraph motif rules and node2vec embeddings + simple Levenshtein/matched-name entity check — fused and isotonic-calibrated. Offline evaluation dashboard showing precision@k, recall vs. synthetic victim labels, FPR/user/month, and calibration curve. This dashboard is a judge magnet — show the actual numbers.
3. **In-app mock (2 days):** Figma-to-React prototype of the K PLUS confirm screen with the amber panel, 30-second self-check (3 personalized scam-pattern cards), and cool-off timer — driven live by the model API on the synthetic user "Ploy."
4. **Demo script:** play the scam timeline at 10× speed → Ploy initiates the transfer → live score computation with per-feature attribution appearing → she enters the self-check → sees the entity-name mismatch → cancels → next-day follow-up. Then flip to the evaluation dashboard: "here's precision, recall, and the false-positive budget."
5. **Stretch (if time):** A/B simulation — replay the synthetic population with/without intervention and show projected victim-case-rate reduction; one slide on scaling from synthetic to real KBank data (the label pipeline, the graph scope, PDPA path).

## Sources

- ACSC weekly scam data, ages 21–30 most victimized — nationthailand.com/news/general/40069932 (Nation Thailand)
- Thai Chamber of Commerce: average scam loss >฿52,000 — nationthailand.com/news/general/40069582 (Nation Thailand)
- Bank of Thailand 2023 transfer/suspension rules (institution-centric, post-transfer machinery) — bot.or.th/en/news-and-media/news/news-20230309.html (Bank of Thailand)
- Team synthesis §5 signal families (first-time payee, threshold avoidance, velocity/escalation, savings-trajectory break) and §4 institutional gap analysis
- DS techniques reference §3 (anomaly detection), §4 (propensity/segmentation), §5 (nudges/experimentation), §6 (graph analytics for mule networks), §8 (wellness scoring)
- Competitor scan: Revolut transfer-time first-time-payee warnings; Thailand whitespace gap #5
- Methodology anchors needing no Thai verification: Isolation Forest (Liu et al., 2008), LightGBM (Ke et al., 2017), GraphSAGE (Hamilton et al., 2017), node2vec (Grover & Leskovec, 2016), Louvain community detection (Blondel et al., 2008), isotonic calibration (Platt/Barlow)
- **[needs verification] flags:** ~30.7% investment-scam share of weekly losses; AOC 1441 victim-case data sharing scope/latency; Thai Bankers' Association mule-list scheme fields; SEC/BOT licensed-entity registry access route; PDPA profiling/automated-decision sections; K PLUS existing bank-side detection (FraudBuster) capabilities. All were marked in-line above; pull current citations before final pitch.