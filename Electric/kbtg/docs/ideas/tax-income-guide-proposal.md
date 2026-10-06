# Income-to-Tax Guide — Understand Your Income, Know Your Next Step

**KBTG Kampus Hackathon 2026 | Track 2: Data Science & Intelligence**

Draft proposal, 9 September 2026. Working feature name; research concept, not an implemented or validated product.

## Problem & Target User Insight

**“เริ่มมีรายได้ แต่ไม่เข้าใจว่ารายได้ที่ตัวเองได้รับเกี่ยวข้องกับภาษีอย่างไร”**

Our target users are people beginning to earn through employment, freelance work, selling goods, content creation, or a combination of these activities. Our working hypothesis is that they struggle to connect general tax information to their own income: which receipts need attention, what supporting documents to keep, and what information is needed to understand their filing obligations. This hypothesis will be validated through user interviews; we do not claim measured prevalence among First Jobbers.

## Proposed Feature & Decision Support

Income-to-Tax Guide is a proposed K PLUS feature that connects incoming transactions to plain-language tax explanations and a concrete next step. With user permission, it highlights possible income, asks the user to confirm its source, and explains the relevant considerations using tax-year-specific official guidance.

For example, a hypothetical ฿5,000 incoming transfer prompts the user to confirm whether it is design-work income, a transfer between their own accounts, or another type of receipt. If it is work income, the feature asks about the activity and available withholding documentation before explaining what evidence to retain and what remains unknown. It does not infer a legal income category from the transfer description alone.

The user can decide which receipts need further review, which documents to obtain, and when to check filing requirements through the official channel. Every explanation distinguishes confirmed facts, model suggestions, and missing information.

## Data / ML Approach

Assumed inputs are permissioned transaction amounts, dates, available descriptions, recurrence patterns, and user-confirmed income sources. Access to actual bank data is unconfirmed; an initial demo will use clearly labelled synthetic transactions and reviewed scenarios.

A rules baseline will be compared with a supervised classifier using text and transaction-pattern features to suggest broad receipt groups. The system will ask for confirmation when confidence is low and support multiple income sources per person. A tax-year-versioned rule layer maps confirmed context to reviewed guidance; any language model only explains that guidance, with source links, rather than inventing tax rules or deciding liability.

## Prototype & Evaluation

The prototype covers incoming-transaction review, source confirmation, and a personalised explanation card with an evidence checklist. Scenarios include salary, freelance work, sales receipts, mixed income, and own-account transfers. Actual filing, tax-payment forecasts, savings, and investment recommendations are outside this prototype.

Evaluate classification with macro-F1, non-income receipts incorrectly flagged as income, and precision at the chosen confidence threshold. Separate users across training and evaluation, and report synthetic results only as prototype evidence. In user testing, compare the feature with a static tax explainer on correct next-step selection, comprehension, and completion time. Review explanations for source support and appropriate handling of insufficient information.

## Risks & Limitations

Bank inflows do not establish taxable income, gross earnings, withholding, or complete annual income. Users must be able to correct suggestions and add missing context. Guidance requires tax-year maintenance and qualified review. Existing services such as My Tax Account already support tax-data preparation; our proposed value is helping inexperienced earners interpret their own receipts, a benefit still to be tested.

**Research context:** [Tax research](../research/tax-first-jobbers.md). Its broad market and demographic claims remain provisional; this proposal does not rely on them as established facts. Official reference: [Revenue Department — My Tax Account](https://www.rd.go.th/63908.html).
