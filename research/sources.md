# Research sources

Collected 17 Oct 2026. Purpose: ground FraudShield's fraud scenarios and their weights in real data instead of guesses.

## How `case_log.csv` is built

- **31 rows** in total:
  - **R01-R16**: modus operandi from the RBI booklet *BE(A)WARE*, compiled by the RBI Ombudsman from reported frauds and complaints. These describe *methods*, not single incidents.
  - **N01-N14**: specific cases reported in the news. Amounts and details are as reported and have not been checked against police records.
  - **F01**: a first-hand family case, anonymised. Keep it out of the public README beyond one anonymous sentence.
- The column `bank_visible_steps_INFERRED` is **inference**. Public reports describe how a fraud worked, almost never which signals the bank saw. Treat that column as a hypothesis to judge, not a fact.
- **Verification task:** open 5 random rows' sources and check them against the log. If any row is wrong, re-check all rows.

## Statistics (for scenario weights)

### India, all reported cyber fraud, 2025 (I4C data as reported)

- 28.15 lakh cases; Rs 22,495 crore lost (2024: 22.68 lakh cases, Rs 22,845 crore).
- **Share of money lost:**
  - Investment fraud ~76%
  - Digital arrest ~9%
  - Sextortion ~4%
  - E-commerce ~3%
  - Credit card ~7%
- **Share of cases:**
  - Investment fraud 35%
  - Sextortion 19%
  - Digital arrest 6%
- Note: phishing and OTP frauds are many complaints but small amounts.
- Sources:
  - [ThePrint: Indians lost Rs 22,495 crore to cyber fraud in 2025](https://theprint.in/india/cybercrime-saw-24-spike-in-2025-indians-lost-rs-22495-crore-mainly-in-investment-scams/2859930/)
  - [CyberPeace: The Data Behind India&#39;s Digital Fraud Surge](https://cyberpeace.org/resources/blogs/the-data-behind-indias-digital-fraud-surge)
  - [PIB: Curbing Cyber Frauds in Digital India (Oct 2025)](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/oct/doc2025108660701.pdf)

### Banks only, FY 2024-25 (RBI data as reported)

- Card and internet frauds are **most fraud cases** (reported as ~56% to ~67% depending on the counting basis) but **a small share of the amount** (~1.5% to ~7%). In FY 2023-24 they were 85.3% of cases.
- The figures differ between outlets because of different reporting bases. Use the RBI *Report on Trend and Progress of Banking in India 2024-25* for the official numbers before quoting.
- Sources:
  - [Medianama: card/internet fraud 66% of cases in FY25](https://www.medianama.com/2025/12/223-cards-internet-fraud-66-in-fy25-decline-banking-rbi/)
  - [Business Standard: bank fraud amount triples in FY25](https://www.business-standard.com/finance/news/bank-fraud-amount-triples-in-fy25-despite-drop-in-number-of-cases-rbi-125052900696_1.html)

## What the numbers mean for the generator

- **Weight by case count, not money.** The generator creates transactions, so case shares are the right base. Money shares are useful for the cost-based threshold in Phase 5 (missing an investment-scam payment costs far more than missing a small OTP fraud).
- **Investment and task scams** are a large share of cases and have a distinctive *escalation over weeks* shape (rows N05-N07, N12).
- **Sextortion** is 19% of cases but was ruled out of scope (it does not start from normal everyday activity). That is a scope decision to state in the README.
- **The I4C categories don't map 1:1 to the scenarios.** "Phishing/OTP fraud" covers several rows (R01, R02, R08, R15). The mapping from categories to scenario rows is a design decision to make and document.

## Primary source

- [RBI: BE(A)WARE, a booklet on modus operandi of financial fraudsters](<https://www.rbi.org.in/commonman/Upload/english/Content/PDFs/English%20BEAWARE.pdf>)
