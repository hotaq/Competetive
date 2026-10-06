# Research: scam-fraud

> Source: workflow `kbtg_gap_research` (run 2026-09-07, all 11 agents ok, no failures). Raw agent output preserved below. `[needs verification]` flags, if any, are the agent's own — verify before pitching.


# Scam/Fraud Landscape in Thailand — Impact on First Jobbers (Ages ~22–28)
**Research deliverable for KBTG Hackathon Track 2 (Data Science & Intelligence) — scam/fraud lens**

---

## 1. Headline statistics (with sources)

**Scale of the problem:**
- **739,000+ online crime cases reported, damages exceeding 77 billion baht** between March 2022 and November 2024 — Royal Thai Police (RTP), reported Dec 22, 2024. (Source: The Pattaya News, https://thepattayanews.com/2024/12/23/thai-police-reveal-shocking-online-crime-statistics-issue-fraud-warning/)
- **~60 billion baht lost to online fraud in 2024**; Thai citizens were targeted by **38 million fraudulent phone calls and 130 million deceptive SMS messages** in 2024 (a five-year high); RTP reported **over 400,000 online fraud cases in Q1 2024 alone**. (Source: The Nation Thailand, https://www.nationthailand.com/news/general/40048240)
- Office of the Ombudsman: **more than 642,000 reported cases and 71.6 billion baht in damages between 2022 and 2024**. (Source: Pattaya Mail, https://www.pattayamail.com/thailandnews/thai-ombudsman-calls-for-systemic-action-as-cybercrime-losses-top-71-billion-baht-532933)
- **First half of 2026: ~170,000 online fraud cases, losses approaching 9 billion baht, average loss >52,000 baht per case** — Thailand Consumers Council (TCC). Facebook was the leading scam channel (**>61% of cases**, incl. fake pages, fraudulent investment ads, impersonation); LINE had fewer cases but **similar total losses**. (Source: The Nation Thailand, https://www.nationthailand.com/news/general/40069582)
- Recent monthly trend: **December 2025 saw 31,198 cases (~1,006/day) with reported losses of ~1.79 billion baht (~58 million baht/day)** — down from 33,624 cases in Dec 2024. (Source: The Nation Thailand, https://www.nationthailand.com/news/general/40062307)
- The **Cyber Crime Investigation Bureau records roughly 700 cases of online fraud/scams daily** (per TDRI/Safer Internet Lab research report). (Source: Safer Internet Lab, "Online Fraud and Scams in Thailand," https://saferinternetlab.org/wp-content/uploads/2025/05/Online-Fraud-and-Scams-in-Thailand.pdf)

**Young-adult-specific data (critical for our target user):**
- Police (Anti Cyber Scam Center, Thai Police Online): **women and people aged 21–30 are the groups most frequently victimized** by online scams; in the week of Aug 9–15, 2026 alone there were **6,060 complaints with ~260 million baht in losses**; Bangkok led all provinces. Goods/services fraud was most common by count (**77.3% of complaints**), but **financial/investment scams caused the greatest monetary damage (~30.7% of losses, ~80 million baht that week)**. (Source: The Nation Thailand, https://www.nationthailand.com/news/general/40069932)
- **Youth mule accounts rose 70.3%** (3,371 in Jan 2025 → 5,741 in Mar 2026), linked to reported losses of **more than 185 million baht** (Thai Police Online data, Oct 2025–May 2026) — Bank of Thailand. (Source: The Nation Thailand, https://www.nationthailand.com/news/general/40069312)
- **ETDA 2023 survey: 43% of Thai respondents were reluctant to conduct online transactions due to fraud concerns, and 29% reduced mobile banking usage** after exposure to scam attempts. (Source: ETDA 2023 Survey on Online Transaction Behavior, cited in Safer Internet Lab report, https://saferinternetlab.org/wp-content/uploads/2025/05/Online-Fraud-and-Scams-in-Thailand.pdf)
- "More young people being taken in by scams" (Bangkok Post headline, ~2025) — details [needs verification]. A similar pattern is confirmed regionally: "Tech-savvy young adults biggest losers to online scams in 2025" (Royal Malaysia Police/Bukit Aman, The Star headline) [needs verification].

**Enforcement/interdiction scale:**
- **1,183,326 mule bank accounts frozen** between Nov 1, 2023 and Jan 30, 2026 (Digital Economy and Society Ministry). (Source: The Nation Thailand, https://www.nationthailand.com/news/general/40062307)
- Thai banks **collectively closed 1.8–1.9 million mule accounts by mid-2024** via the Central Fraud Registry. (Source: Safer Internet Lab report, https://saferinternetlab.org/wp-content/uploads/2025/05/Online-Fraud-and-Scams-in-Thailand.pdf)
- AOC 1441 (Anti-Online Scam Operation Center): **~1.46 million victim calls received and ~518,000 scammer-linked accounts frozen between Nov 2023 and Feb 2025**; officials credit the hotline with cutting daily scam damage by 36–44%. (Note: RTP also reported 537,431 accounts taken down by Feb 2025 — figures differ slightly between agencies.) (Source: Safer Internet Lab report, same URL)
- **304,631 URLs blocked in four months** (Oct 2025–Feb 2026), of which 259,385 related to online gambling; **23,057 "ghost SIM" numbers with ~2.4 million abnormal outgoing calls detected in Dec 2025 alone**. (Source: The Nation Thailand, https://www.nationthailand.com/news/general/40062307)
- Google–MDES partnership blocked **6.6 million high-risk app installation attempts** between April 2024 and April 2025. (Source: Safer Internet Lab report, citing Bangkok Post)

---

## 2. Scam types hitting Thai young adults / first jobbers

| Scam type | Mechanics | Why it hits first jobbers |
|---|---|---|
| **Investment scams / fake trading ("call-center gang")** | Social-media ads + LINE/Facebook "mentors" promising high returns; victims transfer to personal or frequently-changed corporate accounts. Largest category by monetary damage (~30.7% of weekly losses per ACSC). | First jobbers have their first disposable income and FOMO on wealth-building; many are first-time investors with no baseline for what a legitimate offer looks like. Police explicitly warn that being asked to transfer investment funds to a **personal account** is a red flag. (Source: The Nation, 40069932) |
| **Romance scams ("pig-butchering")** | Fake profiles build trust → "gifts" → investment coaching → transfers. RTP lists romance + investment scams as a top-4 category. (Source: Pattaya News, Dec 2024) | Young singles on dating/social apps; scammers exploit loneliness and inexperience with relationships + money combined. |
| **Fake job ads / job scams** | Fake job offers via social media and deceptive ads — cited by the Safer Internet Lab as a top delivery channel; worst cases lure victims across borders into forced-scam compounds ("hell in Cambodia" cases; 7,000+ people rescued/detained from Myawaddy scam centers in the early-2025 Thai–Chinese crackdown). (Sources: Safer Internet Lab report; Thai PBS "Online Job Hunt Ends in 'Hell in Cambodia'" headline [needs verification]) | First jobbers are actively job-hunting or seeking side income; "easy high-pay work" ads target exactly this cohort. |
| **Mule account recruitment** | Young people paid (often a few thousand baht) to open accounts or sell/rent existing ones to gangs; "premium grade" mule accounts are sold commercially (Bangkok Post headline, Jun 2026 [needs verification]). Youth mule accounts +70.3% (BOT, above). Under-18s specifically targeted; BOT now caps youth transfers at 3,000–10,000 baht/day by age. (Source: The Nation, 40069312) | Cash-strapped first jobbers/students respond to "earn easy money with your bank account" ads; many don't know it's a criminal offence under the 2023 Emergency Decree. |
| **Smishing / phishing / fake SMS** | 130 million deceptive SMS in 2024; fake delivery/lottery/promotion links; bots steal OTPs and login credentials. Banks are now banned from sending links in SMS/email at all (BOT rule). (Sources: The Nation 40048240; BOT press release, Mar 2023) | Digital-native users tap links reflexively; young users manage everything on one phone. |
| **Impersonation of officials / police ("social-engineered transfers")** | Caller poses as telecom staff → transfers to fake "police" → threatens victim with a money-laundering case → orders transfers to a fake "AMLO account." Real case (Aug 2026): a **first-year university student lost 3.6 million baht in a single day**, including deceiving her father into transferring 2.7 million more, and handing cash to a fake officer at a hotel. (Source: The Nation Thailand, https://www.nationthailand.com/news/general/40070378) | Young adults show deference to authority, panic under legal threats, and lack the experience to know police never demand transfers. |
| **AI-enabled scams / deepfakes** | Voice cloning, face-swap video calls, deepfake "police officers" (190+ victims in one case); an influencer lost 4M+ baht to an AI face-altering video-call scam; even the PM was targeted by an AI voice-clone. A majority of surveyed Thais said more than half of scam messages they received appeared AI-generated (Global Anti-Scam Alliance survey). (Source: Safer Internet Lab report) | Gen Z trusts video/voice authenticity; deepfakes defeat the "video call to verify" habit. |

---

## 3. Why first jobbers (22–28) are specifically vulnerable — synthesis

1. **They are the statistically most-victimized age band**: police data puts **ages 21–30 as the most frequently victimized group** (ACSC, Aug 2026).
2. **First real money + zero financial scar tissue**: a first salary creates the first meaningful bank balance, but no learned defenses — no experience with investment norms, contracts, or how banks/police actually behave.
3. **Channel exposure**: their lives run on Facebook, LINE, TikTok — the exact channels carrying >61% of scam cases (TCC).
4. **Mule recruitment sweet spot**: gangs target the young and cash-hungry (youth mule accounts +70.3%); renting out an account for quick money feels victimless but is a criminal offence.
5. **Job-market pressure**: job-hunting, side-hustle, and investment content is their daily feed — all three top scam categories (investment, fake jobs, goods/services) map directly onto their aspirations.
6. **Thin buffers**: with little savings, a single scam (average loss >52,000 baht ≈ 1–3 months of a first-jobber salary) is financially catastrophic and derails any saving habit.
7. **AI-era asymmetry**: AI-generated messages are now indistinguishable from real marketing; young users' "I'm tech-savvy so I'm safe" bias works against them.

---

## 4. How Thai banks/regulators currently detect and warn

**Bank of Thailand (regulatory floor, since March 2023):**
- **Prevent**: banks banned from sending links via SMS/email or requesting credentials via social media; **one username + one device per mobile banking account**; every transaction triggers a pre-notification; **biometric (face) verification for transfers >50,000 baht per transaction** or when raising daily limits above 50,000; daily limits set by **customer risk profile**. (Source: BOT press release, https://www.bot.or.th/en/news-and-media/news/news-20230309.html)
- **Detect**: BOT sets criteria for detecting/monitoring suspicious transactions; banks must run **near-real-time detection systems that can temporarily freeze suspicious transactions** and report to AMLO. (Same source)
- **Respond**: mandatory **24/7 fraud hotlines** at every bank; Cybercrime Emergency Decree (B.E. 2566/2023) lets banks freeze accounts **72 hours (victim-reported) then 7 days**, and freeze proactively 7 days on their own suspicion. (Source: Safer Internet Lab report, citing Tilleke & Gibbins)

**Cross-institution infrastructure:**
- **Central Fraud Registry (CFR)** — Thai Bankers' Association, mid-2024: shared fraudulent-account data across all banks (this is how 1.8–1.9M mule accounts were closed).
- **AOC 1441 hotline** (MDES + police + AMLO + BOT + NBTC + SEC): 24/7, 100 lines, ~10-minute case-to-freeze workflow; feeds on CFR + telecom data. (Source: Safer Internet Lab report)
- **DE-fence platform** (MDES, launched late 2025): moves from reactive to **predictive** — AI + big-data monitoring of suspicious financial activity, scam websites, and transaction behavior, integrated with AOC 1441. (Source: Safer Internet Lab report)
- **2026 additions**: BOT **age-based daily transfer caps (3,000–10,000 baht) for youth accounts** (banks by Sep 2026); **location services required for mobile-banking transactions**; SIM ownership capped at 5 numbers/person with KYC; DBD requires in-person directors + 3 months of bank statements to curb **corporate mule accounts**. (Sources: The Nation 40069312, 40062307)
- **KBank-specific**: KBTG's **AI-driven face-liveness detection** was recognized by Global Finance for financial-innovation excellence (kasikornbank.com, headline via Google News [needs verification]); K PLUS includes in-app scam advisories/warnings for common scam patterns [needs verification]. Notably, KBank (Thailand's most profitable bank) was ordered to pay a beauty-queen victim ~1M baht damages over an AI-assisted 4M baht scam (VnExpress/Bangkok Post headlines, Jun 2026 [needs verification]) — showing courts are beginning to assign liability when banks' warnings/filters fail.

**The gap**: all of this is institution-centric, rule/threshold-based, and reactive-first. Warning UX is generic ("this may be a scam"), signals are siloed per-bank, and there is **no personalized, behavioral, pre-transfer intelligence delivered to the individual customer** — which is exactly the Track 2 opportunity.

---

## 5. Data signals that enable detection / early warning (Track 2 core)

These are the signal families a K PLUS data-science feature could use (analytical synthesis from the sourced landscape + standard AML/fraud-analytics practice — feasibility marked accordingly):

**A. Transfer-context anomaly signals (available inside K PLUS today)**
- **First-time payee + amount-vs-baseline**: transfer to a never-seen payee whose amount is a large multiple of the user's median transfer, or a large share of their balance/savings — the canonical APP-scam signature (works well for first jobbers because their normal pattern is small, regular, salary-anchored → clean baseline).
- **Time-of-day deviation**: transfers at 01:00–05:00, or immediately after a burst of new-app downloads/SMS activity — scam victims are commonly kept on the phone into the night (see the 3.6M baht student case, executed in a single day).
- **Threshold-avoidance patterns**: amounts just under the 50,000-baht biometric trigger (e.g., repeated 49,000s) — scammers coach victims to do this.
- **Velocity & repetition**: multiple transfers in minutes to different new payees (victim "splitting" money under instructions), or escalating amounts over hours.

**B. Personal behavioral-deviation signals (the money-management angle — our track's sweet spot)**
- **Savings-trajectory break**: a user who has auto-saved X% of every salary for months suddenly drains savings mid-cycle and transfers out — deviation from their own learned financial rhythm is a high-precision scam cue *and* a financial-wellness cue simultaneously.
- **Salary-anchored profiling**: first jobbers have predictable inflows (salary date/amount) and category-stable outflows; deviation scores are unusually reliable for this segment.
- **Goal abandonment**: canceled saving goals + immediate out-transfer shortly after new social-media/ad exposure.

**C. Graph signals for mule accounts (the Track 2 showpiece)**
- **Fan-in signature**: account receiving small-to-medium transfers from many unrelated senders in a short window — this is how the 1.18M frozen mule accounts were identified ("abnormal transactions", per DES Minister).
- **Pass-through / layering**: rapid in→out with minimal balance retention, often within minutes; funds hop 2–4 accounts before withdrawal (drives AOC's 10-minute freeze race).
- **Hub/community detection**: one account bridging otherwise disconnected sender/receiver communities; shared device IDs, IPs, phone numbers, or Wi-Fi fingerprints across many "unrelated" accounts (device-graph signals — the same fingerprinting KBang/KBTG already uses for liveness/KYC).
- **Dormancy burst**: account idle for months → sudden activity spike (classic rented-mule pattern).
- **Youth-account overlays**: new accounts for ages 15–25 with immediate cross-bank transfers near the new caps — the BOT's own finding that youth mule accounts grew 70.3% makes an age-conditioned mule-risk model defensible and data-backed.
- **Cross-bank federation**: the CFR means mule-risk scores can be cross-bank, not just K PLUS-internal.

**D. Channel/telecom & content signals (context feeds)**
- Ghost-SIM traffic patterns (23,057 numbers / 2.4M abnormal calls in one month — precedent for anomaly-based telecom flags).
- URL/domain reputation feeds (304,631 URLs blocked in 4 months) — if a customer clicked a now-blocked domain before transferring, that's a strong pre-scam feature.
- AI-generated message detection (GASA finding: majority say >half of scam messages look AI-generated) — text-model features for smishing bait.

**E. Victim-side predictive signals (the "scam susceptibility score")**
- Segment (age 21–30, female = highest-reporting groups per ACSC), first salary account age, savings balance vs. peers, prior exposure to blocked URLs, first-time investor behavior (searching/engaging with investment content then transferring to a personal-named account) → composite **pre-transfer risk score** that can trigger a smart, personalized friction moment (cool-off timer, checklist, "this account name doesn't match the company you're investing with") instead of generic warnings.

**F. External verification join-keys (compliance-friendly)**
- **Payee-name vs. licensed-entity matching** — police/SEC guidance already instructs checking that the receiving account name matches the authorized company (SEC Check First); automating this check is a legitimate, sourced detection idea. (Source: The Nation, 40069932)

---

## 6. Implication for our Track 2 pitch (one paragraph)

Thailand's anti-scam machinery (CFR, AOC 1441, DE-fence) is fast but **post-transfer and institution-facing**; the customer still makes the fatal decision alone. The sourced data shows the highest-volume victim segment is exactly our user (ages 21–30, women most-hit), the costliest scam category is investment fraud (first-jobber aspiration), losses average >52,000 baht/case (1–3 months of a first salary), and the youth segment is also the fastest-growing mule pool (+70.3%). A K PLUS intelligence layer that fuses **personal behavioral-deviation signals with graph-based payee-risk signals into a pre-transfer warning and a "mule-resistance" nudge** is simultaneously (a) money-management protection (savings trajectory as a fraud feature), (b) data-science-centered, and (c) squarely inside what regulators (BOT/CFR) are already pushing banks to do — but personalized for the first jobber, which no one currently does.

---

## 7. Source list

1. The Pattaya News — RTP online crime statistics (Dec 2024): https://thepattayanews.com/2024/12/23/thai-police-reveal-shocking-online-crime-statistics-issue-fraud-warning/
2. The Nation Thailand — 60B baht 2024 losses / 38M calls / 130M SMS (Apr 2025): https://www.nationthailand.com/news/general/40048240
3. Pattaya Mail — Ombudsman 71.6B baht / 642,000 cases (Jan 2026): https://www.pattayamail.com/thailandnews/thai-ombudsman-calls-for-systemic-action-as-cybercrime-losses-top-71-billion-baht-532933
4. The Nation Thailand — ACSC weekly: women & ages 21–30 most victimized (Aug 2026): https://www.nationthailand.com/news/general/40069932
5. The Nation Thailand — TCC: ~9B baht H1-2026 losses, Facebook 61%+ (Aug 2026): https://www.nationthailand.com/news/general/40069582
6. The Nation Thailand — BOT youth transfer caps, +70.3% youth mule accounts (Aug 2026): https://www.nationthailand.com/news/general/40069312
7. The Nation Thailand — DES Ministry: 1.18M mule accounts frozen, Dec 2025 losses 1.79B baht, ghost SIMs (Feb 2026): https://www.nationthailand.com/news/general/40062307
8. The Nation Thailand — Call-centre gang tricks university student out of 3.6M baht (Aug 2026): https://www.nationthailand.com/news/general/40070378
9. Safer Internet Lab / TDRI research report — "Online Fraud and Scams in Thailand" (May 2025): https://saferinternetlab.org/wp-content/uploads/2025/05/Online-Fraud-and-Scams-in-Thailand.pdf (covers CCIB daily case counts, AOC 1441 call/freeze volumes, 1.8–1.9M mule accounts closed, ETDA survey, BOT measures, Cybercrime Decree freeze mechanics, DE-fence, deepfake cases, Thai–China crackdown)
10. Bank of Thailand press release — additional anti-fraud measures (Mar 2023): https://www.bot.or.th/en/news-and-media/news/news-20230309.html
11. OpenGov Asia — Thailand tightens digital transfer limits for youth accounts (Jul 2026): https://www.opengovasia.com/thailand-tightens-digital-transfer-limits-for-youth-accounts/
12. KBank/KBTG face-liveness recognition (headline via Google News, kasikornbank.com) [needs verification]; KBang beauty-queen damages case (VnExpress/Bangkok Post headlines, Jun 2026) [needs verification]; "premium grade" mule account sales (Bangkok Post headline, Jun 2026) [needs verification].

**Caveats for the team**: 2025–2026 Nation Thailand figures were fetched directly from their site and are reliable; aggregate multi-year totals differ between agencies (60B vs 63B vs 71.6B vs 77B baht for overlapping periods) because of different counting windows and whether smishing-attempt values are included — always cite the specific agency and window. AOC freeze counts also vary by report date (517,954 vs 537,431). The "21–30 most victimized" figure comes from ACSC weekly reporting; request the full-year age distribution from Thai Police Online public dashboards before putting it on a slide.