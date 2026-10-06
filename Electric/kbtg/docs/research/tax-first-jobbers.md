# Research: Tax × Thai First Jobbers (Track 2 lens)

> **Curator note (compiled 2026-09-09).** This file was **not** produced by the `kbtg_gap_research` workflow — it was compiled from targeted web search after the team decided to add a tax angle. Every claim below carries a URL in §9. `[needs verification]` marks items the search did not confirm to a primary source; check against rd.go.th before putting them on a slide.

---

## 1. Why tax is a First-Jobber problem specifically

Tax is the one financial obligation a First Jobber meets **for the first time with zero prior practice** and **no chance to learn gradually**:

- The first salary is also the first ทวิ 50, the first ประกันสังคม deduction, the first ค่าลดหย่อน decision, the first ภ.ง.ด.91.
- The filing threshold is crossed **silently**. A salaried single filer must file once employment income exceeds **฿120,000/year** — i.e. a salary of about **฿10,000/month**, which is below the fresh-graduate median. Practically every First Jobber is legally required to file from month one, and nothing in their life tells them so.
- Withholding hides the problem. Because the employer withholds, the First Jobber *feels* compliant. Two failure modes follow:
  1. **Never filing** → forfeits any refund of over-withheld tax (this is money they are owed, not a penalty they avoid).
  2. **Under-claiming deductions** → pays tax that was legally avoidable, because nobody told them in *October* what to do; they find out in *March* when it is too late.
- Side income makes it worse. Freelance/creator income (มาตรา 40(2)/40(8)) is typically withheld at 3% and **must** be self-reported; 39.1% of surveyed Thais did not know hiring/freelance income counts as reportable income (§3).

**The structural point for Track 2:** the tax mistakes First Jobbers make are almost entirely **timing and information** mistakes, not willingness mistakes. Timing + information on top of transaction data is a data-science problem.

---

## 2. The rules that matter (tax year 2568, filed Jan–Apr 2569)

### 2.1 Progressive brackets (เงินได้สุทธิ)

| เงินได้สุทธิ (บาท) | อัตรา |
|---|---|
| 0 – 150,000 | ยกเว้น |
| 150,001 – 300,000 | 5% |
| 300,001 – 500,000 | 10% |
| 500,001 – 750,000 | 15% |
| 750,001 – 1,000,000 | 20% |
| 1,000,001 – 2,000,000 | 25% |
| 2,000,001 – 5,000,000 | 30% |
| 5,000,001 ขึ้นไป | 35% |

**Critical for our segment:** a First Jobber on ฿22,000/month (฿264,000/year) with only the standard expense + personal + SSO deductions lands at roughly ฿264,000 − 100,000 − 60,000 − 9,000 = **฿95,000 net → ฿0 tax**. Most First Jobbers sit in the **0% or 5% bracket**. This single fact reshapes the whole solution (see §8).

### 2.2 Expense deduction (เงินได้ 40(1) เงินเดือน)

50% of income, **capped at ฿100,000**. Fully mechanical — no user input needed.

### 2.3 Allowances relevant to a First Jobber

| ค่าลดหย่อน | เพดาน (ปีภาษี 2568) | Bank can see it? |
|---|---|---|
| ส่วนตัว | 60,000 | Automatic, no data needed |
| ประกันสังคม (ม.33) | สูงสุด 9,000 | Inferable from net-salary gap / payroll |
| ประกันชีวิต + สุขภาพ + สะสมทรัพย์ | รวมไม่เกิน 100,000 (สุขภาพตนเองมีเพดานย่อย 25,000 `[needs verification]`) | Yes, if premium is paid from a KBank account |
| ประกันสุขภาพบิดามารดา | 15,000 `[needs verification]` | Yes, if paid from a KBank account |
| RMF + PVD + กบข. + ประกันบำนาญ | รวมกันไม่เกิน **30% ของเงินได้พึงประเมิน** และไม่เกิน **500,000** | Yes for RMF bought via K PLUS; PVD from payroll |
| **Thai ESG** | วงเงิน**แยกต่างหาก** ไม่เกิน 300,000 | Yes for funds bought via K PLUS |
| **Thai ESGX** | สับเปลี่ยนจาก LTF ไม่เกิน 500,000, ลดหย่อนปีแรกไม่เกิน 300,000, ส่วนที่เหลือทยอยปีละ 50,000 | Yes |
| **SSF** | **หมดสิทธิลดหย่อนแล้วตั้งแต่ปีภาษี 2568** | — |
| ดอกเบี้ยเงินกู้ที่อยู่อาศัย | ตามจริง ไม่เกิน 100,000 | Yes for a KBank mortgage |
| Easy E-Receipt 2.0 | ไม่เกิน 50,000 (30,000 ร้าน VAT + 20,000 OTOP/วิสาหกิจชุมชน) ซื้อ 16 ม.ค. – 28 ก.พ. 2568 | Partially — bank sees the spend, not the e-Tax Invoice |
| เที่ยวดี มีคืน (ที่พัก/ร้านอาหาร) | 20,000 `[needs verification — exact 2568 conditions]` | Partially |
| บริจาคทั่วไป | ไม่เกิน 10% ของเงินได้หลังหักค่าใช้จ่ายและค่าลดหย่อน | Yes via e-Donation QR |
| บริจาคการศึกษา/กีฬา/รพ.รัฐ | **2 เท่า** ของยอดบริจาค (ภายในเพดาน 10%) | Yes via e-Donation QR |
| ค่าฝากครรภ์และคลอดบุตร | ไม่เกิน 60,000 | Yes if paid from a KBank account |

### 2.4 Filing thresholds

| กรณี | ต้องยื่นเมื่อเงินได้เกิน |
|---|---|
| เงินเดือนอย่างเดียว 40(1), โสด | 120,000 / ปี |
| เงินเดือนอย่างเดียว 40(1), สมรส | 220,000 / ปี (รวมกัน) |
| เงินได้ประเภทอื่น (freelance ฯลฯ) | 60,000 / ปี |

Salaried-only → **ภ.ง.ด.91**. Any other income type → **ภ.ง.ด.90**.

### 2.5 Deadlines & penalties

- Paper filing: **31 มีนาคม**. e-Filing: **8 เมษายน** (an 8-day MOF internet-filing extension, in force for returns due between 1 Feb 2024 and 31 Jan 2027).
- Failure to file on time: criminal fine **not exceeding ฿2,000**.
- If RD later finds tax was underpaid: **เบี้ยปรับ 1–2 เท่า** of the tax due plus **เงินเพิ่ม 1.5% per month**.

### 2.6 The **Tax Consent** trap (the highest-value, lowest-effort finding in this file)

Buying an RMF / Thai ESG fund is **not enough** to get the deduction. The investor must separately instruct the AMC / insurer / bank to transmit the purchase data to the Revenue Department — for KAsset this is a one-time consent at `k-invest.kasikornbankgroup.com/TaxConsent/`. **If consent is not given, the deduction is lost entirely**, and the same requirement covers mortgage interest, life/health insurance premiums, parents' health insurance, and pension insurance. The instruction must reach the institution **by the end of the tax year**.

This is a pure information failure with a **deterministic** detection rule: the bank knows both (a) that the customer bought the fund and (b) whether consent exists. Nobody surfaces it proactively.

---

## 3. Evidence: filing is low, and tax knowledge is lower

### 3.1 Revenue Department / macro statistics

| Metric | Value | Year |
|---|---|---|
| ผู้ยื่นแบบ ภ.ง.ด.90/91 | **12.57 ล้านคน** | ปีภาษี 2567 |
| ผู้เสียภาษีจริง | **5.14 ล้านคน** (~฿280,000 ล้าน) | ปีภาษี 2567 |
| ผู้ยื่นแต่ได้รับยกเว้น | **7.43 ล้านคน** (≈59% ของผู้ยื่น) | ปีภาษี 2567 |
| แบบที่ยื่น | 11,997,139 แบบ (+3.42% YoY) | ปีภาษี 2566 |
| แบบที่ **ขอคืนภาษี** | **4,280,939 แบบ = 35.68%** (+7.6% YoY) | ปีภาษี 2566 |
| คืนภาษีแล้ว (ณ 27 พ.ค. 2567) | 3.6 ล้านแบบ / **฿36,600 ล้าน** | ปีภาษี 2566 |
| กำลังแรงงานไทย | ~40.2 ล้านคน (Q1/2567) | 2567 |

Two derived figures worth putting on a slide (state clearly that they are **our arithmetic**, not RD publications):

- **Average refund ≈ ฿36,600M ÷ 3.6M ≈ ฿10,100 per refunding filer.** For a First Jobber on ฿22,000/month this is roughly **half a month's salary**.
- **More than 1 in 3 returns are filed to get money back, and ~59% of all filers owe nothing.** So the dominant reason to file, for our segment, is **refund and record**, not payment — the exact opposite of how tax is framed to young people.

### 3.2 The knowledge survey (NESDC × SAB) — the strongest evidence in this file

Survey by สศช. with ศูนย์วิจัยเพื่อการพัฒนาสังคมและธุรกิจ (SAB): **n = 3,846, age 25+, 14 provinces**, on tax year 2565 behaviour.

| Finding | % |
|---|---|
| ยื่นแบบ | 35.7% |
| **ไม่ยื่นแบบ** | **64.3%** — of which **50.5% were legally required to file** |
| ความรู้เรื่องภาษีระดับ**ต่ำหรือไม่มีเลย** | **57.9%** |
| **ไม่รู้ว่า "ยื่นแบบ ≠ ต้องเสียภาษี"** | **65.6%** |
| ไม่รู้ว่าเงินได้สุทธิ ≤ ฿150,000 ได้รับยกเว้น | 53.6% |
| ระบุไม่ได้ว่าเงินได้ประเภทใดต้องนำมายื่น | 63.2% |
| **ไม่รู้ว่ารายได้รับจ้าง/ฟรีแลนซ์ต้องนำมายื่น** | **39.1%** |
| ในกลุ่มที่เข้าเกณฑ์ต้องยื่นแต่ไม่ยื่น: ความรู้ต่ำ/ไม่มี | 74.9% |
| ในกลุ่มที่มองว่าระบบไม่เป็นธรรม: ไม่รู้ว่าภาษีเป็นขั้นบันได | 80.4% |

Among filers, **60% were Gen Y**, with average income ฿27,827/month.

### 3.3 Honest limitations of this evidence — read before pitching

1. **The survey covers age 25+, not First Jobbers specifically.** Non-filers in it skew Gen X / Boomer / informal-sector (avg ฿12,115/month), not salaried graduates. It is **not** direct proof that First Jobbers don't file.
2. What it **does** support, and what we should claim: (a) tax knowledge in Thailand is systemically low; (b) the specific misconception **"filing means paying"** is held by a **majority** (65.6%) — and that misconception is exactly what stops a 0%-bracket First Jobber from filing to reclaim withheld tax; (c) freelance-income reporting is widely misunderstood (39.1%), and side income is disproportionately a young-worker phenomenon.
3. **No published figure exists for "people entitled to a refund who never claimed it."** RD does not release it. Do **not** invent one. Frame the opportunity as *addressable* (35.68% of filers claim refunds; a segment that never files claims ฿0) rather than as a measured loss.
4. Fresh-graduate side-hustle prevalence in Thailand is **not** well-sourced. Krungsri Research's Gen Z Finance Survey (n=426, aged 18–30, Feb–Mar 2568) is the closest Thai instrument; international figures (e.g. Harris Poll's 57% of US Gen Z with a side hustle) are **not** transferable. `[needs verification]`

---

## 4. What K PLUS / KBank already does — and where it stops

**Does today:**
- Sells RMF and Thai ESG funds in-app (KAsset lists 38 tax-saving funds purchasable via K PLUS and K-My Funds).
- Provides Tax Consent as a **separate web flow** (`k-invest.kasikornbankgroup.com/TaxConsent/`) and tax documents via K-Cyber Invest e-Documents.
- Publishes tax explainer content (e.g. MAKE by KBank articles on filing thresholds).

**Does not do:**
- Tell the customer their **projected tax position** at any point during the year.
- Tell them **whether they are required to file** based on their own income.
- Tell them **which deductions they are already eligible for** from their own transactions.
- Tell them they bought a tax fund but **never gave Tax Consent** — the deterministic own-goal in §2.6.
- Size a fund purchase against **what they can actually afford** (see §8).

The existing surface is **product distribution timed to December**, not intelligence timed to the customer.

---

## 5. Competitive landscape

| Player | What it does | Gap it leaves |
|---|---|---|
| **iTAX** (private, iOS/Android, free + IAP) | Full PIT calculator, planning mode, per-user deduction shortlist, tax-status summary, iTAX shop, supports Easy E-Receipt / Thai ESG | **Requires manual entry.** Has no view of the user's actual transactions, income trajectory, or cash flow. Cannot know whether the user can afford the ฿100,000 it suggests. Android users report crashes/slowness. |
| **RD Smart Tax / RD e-Filing** | Official filing channel | Filing mechanics only — no projection, no planning, no personalised prompts. Reactive by design. |
| **MAKE by KBank** | Tax explainer content | Content, not computation. Not tied to the user's numbers. |
| **Thai banks generally** | Sell tax funds in Q4 | Product push. No bank computes a customer's tax position. |
| **Global benchmarks** | US: Intuit/Credit Karma refund estimation from linked accounts. UK: Monzo/Starling do **not** do tax for PAYE (PAYE removes the need). | **The Thai system requires self-filing even for pure salary, which the UK/US-PAYE analogy does not cover — so the Thai opportunity is genuinely larger than the global benchmark suggests.** |

**Bottom line: no Thai bank turns its own transaction data into a tax position.** The one player who does tax well (iTAX) has no data; the players who have data (banks) do no tax.

---

## 6. Data signals inside a bank that map to tax objects

| Tax object | Bank signal | Confidence |
|---|---|---|
| เงินได้ 40(1) | recurring payroll credit, amount + date regularity | High (payroll customers: exact gross + withholding) |
| Gross income (non-payroll customers) | **inverse solve** from net credit + SSO formula + withholding formula | Medium — a real modelling problem, see idea g9 |
| เงินได้ 40(2)/40(8) | irregular credits from non-payroll counterparties, platform payouts, 3%-withheld amounts | Medium — needs Thai NLP on counterparty descriptors |
| ประกันสังคม ม.33 | payroll deduction / net-gross gap | High for payroll customers |
| PVD | payroll deduction line | High for payroll customers |
| เบี้ยประกันชีวิต/สุขภาพ | recurring debit to a known insurer | High (merchant identity is unambiguous) |
| RMF / Thai ESG | in-app fund purchase | **Exact** — first-party |
| Tax Consent status | KAsset/KSecurities consent record | **Exact** — first-party |
| ดอกเบี้ยบ้าน | KBank mortgage servicing record | Exact for KBank mortgages |
| บริจาค | e-Donation QR payment | High |
| Easy E-Receipt–eligible spend | card/QR spend at VAT-registered merchants | Medium — bank sees the spend, not the e-Tax Invoice |
| **Blind spots** | other-bank salary, cash income, other employers' ทวิ 50, spouse income, non-KBank insurers/AMCs, e-wallets | **Must be disclosed. Any deduction total we compute is a lower bound; any income total is possibly incomplete.** |

---

## 7. Where tax connects to the team's existing ideas

- **g1 Money Runway** — same forecasting engine, longer horizon. Runway forecasts to next payday; a tax projection forecasts to 31 December. The features (payday detection, income regularity, obligation detection) are shared.
- **g2 Safe-to-Save Autopilot** — a year-end tax-fund purchase *is* a savings sweep with a legal deadline and a legal cap. The affordability logic is identical; only the constraint set changes.
- **g7 Gen-Z Money Map (not yet written up)** — the Thai-NLP categorization layer is the same one that identifies insurers, donations, and platform payouts.

Tax is therefore **not a new stack** for a team that has already chosen g1 — it is a second decision surface on top of the same engine, with a deadline that creates urgency the runway card lacks.

---

## 8. Implications for a Track 2 pitch

1. **Frame it as refund + compliance, not "save tax."** Most First Jobbers are in the 0–5% bracket. The honest value is *"you're owed money and you don't know it"* and *"you're legally required to file and you don't know it"* — not *"buy funds to save tax."*
2. **The 5%-bracket fact is a trust weapon.** At a 5% marginal rate, ฿10,000 into an RMF saves ฿500 in tax and locks the money until age 55. A model that **tells the user not to buy** is (a) correct, (b) unique for a bank-built tool, and (c) directly pre-empts the "this is just a fund sales funnel" objection judges will raise. Build the "don't buy" recommendation deliberately, and show it in the demo.
3. **Affordability is the moat.** iTAX can compute the optimal deduction; only the bank knows whether the user can pay for it without going negative before payday. That is the one thing no competitor can copy.
4. **Tax Consent is the free win.** Deterministic, first-party, high-value, zero model risk. Demo it first — it costs nothing to believe.
5. **Uncertainty must be shown.** Income projection in September carries real error. Show intervals ("ประมาณ ฿1,200–฿3,400"), never a single confident number, and never call it tax advice.

---

## 9. Source list

**Statistics**
- ThaiPublica (ส.ค. 2569) — ฐานภาษีบุคคล 12.57 ล้านคน / จ่ายจริง 5.14 ล้านคน / ยกเว้น 7.43 ล้านคน: https://thaipublica.org/2026/08/thai-personal-income-tax-stats-5m-taxpayers-280b-baht-7m-exempt/
- The Standard — ผลยื่นแบบปีภาษี 2566, ขอคืน 4,280,939 แบบ (35.68%), คืนแล้ว ฿36,600 ล้าน: https://thestandard.co/thais-seek-more-tax-refunds-2023/
- Thai PBS Policy Watch — สศช.×SAB survey, ความรู้ภาษีคนไทย (n=3,846, อายุ 25+): https://policywatch.thaipbs.or.th/article/government-23
- Thairath — เพราะอะไรคนไทยจำนวนมากไม่ยื่นแบบ (64.3% ไม่ยื่น / 50.5% เข้าเกณฑ์): https://www.thairath.co.th/money/economics/thai_economics/2790576
- สำนักงบประมาณของรัฐสภา, เอกสารวิชาการ 2/2569 — โครงสร้าง PIT + ผู้ยื่น 11.88 ล้านคน vs กำลังแรงงาน 38–40 ล้าน: https://web.parliament.go.th/view/82/News_Detail/2569/1663/TH-TH
- กรมสรรพากร — สถิติการยื่นแบบภาษีเงินได้บุคคลธรรมดา: https://www.rd.go.th/62748.html และสถิติ e-Filing: https://www.rd.go.th/67140.html

**Rules & rates**
- Krungthai — วิธีคำนวณภาษีเงินได้บุคคลธรรมดา 2568: https://krungthai.com/th/financial-partner/learn-financial/1963
- SET (TSI) — อัปเดตสิทธิลดหย่อนภาษี 2568: https://www.set.or.th/th/education-research/education/happymoney/knowledge/article/137-tsi-update-thailand-tax-deduction-2568
- SET InvestNow — ทางเลือกลดหย่อนภาษี 2568 (RMF/Thai ESG/Thai ESGX, SSF สิ้นสุด): https://www.setinvestnow.com/th/knowledge/article/664-rinjai-investment-options-with-tax-ltf-thaiesgx-deduction-benefits-in-2025
- Thairath — โค้งสุดท้ายลดหย่อนภาษี 2568: พลาด 10 เรื่องใหญ่ (รวมประเด็นแจ้งความประสงค์ส่งข้อมูลให้สรรพากรภายในสิ้นปี): https://www.thairath.co.th/money/personal_finance/financial_planning/2903397
- Krungsri the COACH — First Jobber เงินเดือนแค่นี้ต้องยื่นภาษีไหม: https://www.krungsri.com/th/krungsri-the-coach/taxes/tax-knowledge/first-jobber-tax-filing
- KTC — ภ.ง.ด.90 vs ภ.ง.ด.91 และบทลงโทษ: https://www.ktc.co.th/article/knowledge/salary-man/pnd-90
- FRANK Legal & Tax — 2026 tax return filing deadlines (31 Mar / 8 Apr): https://franklegaltax.com/2026-tax-return-filing-in-thailand/
- Bangkok Post — Thai tax filing 2026, April 8 deadline: https://www.bangkokpost.com/business/general/3210459/thai-tax-filing-2026-what-expats-should-clarify-before-the-april-8-deadline
- Nishimura & Asahi — eight-day e-filing extension (1 Feb 2024 – 31 Jan 2027): https://www.nishimura.com/en/knowledge/publications/further-eight-day-extension-for-e-filing-tax-returns-and-online-tax-payments-in-thailand

**KBank / competitors**
- KAsset — กองทุนลดหย่อนภาษี RMF/ThaiESG (38 กองทุน, ซื้อผ่าน K PLUS): https://www.kasikornasset.com/kasset/th/mutual-fund/taxsaving/pages/taxsaving.aspx
- KSecurities — Tax Consent (แจ้งความประสงค์ส่งข้อมูลให้กรมสรรพากร): https://www.kasikornsecurities.com/th/startinvesting/tax_consent
- K-Cyber Invest e-Documents: https://k-invest.kasikornbankgroup.com/iPortal/EDocuments/default.aspx?lang=TH
- iTAX Pro — ฟีเจอร์แอปคำนวณ/วางแผนภาษี: https://www.itax.in.th/pedia/itax-pro/
- MAKE by KBank — รายได้เท่าไหร่ไม่ต้องยื่นภาษี: https://makebykbank.kbtg.tech/en/articles/tax-exemption

**Segment context**
- ThaiPublica — Krungsri Research Gen Z Finance Survey (n=426, อายุ 18–30, ก.พ.–มี.ค. 2568): https://thaipublica.org/2025/06/krungsri-research-gen-z-finance-survey/
