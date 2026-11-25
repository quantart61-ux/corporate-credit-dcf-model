# Corporate Credit Underwriting, Covenant Stress-Testing & DCF Valuation Model

An institutional-grade corporate finance modeling and credit risk underwriting framework implemented in Python. Built to bridge **Commercial/Corporate Banking Credit Analysis** (leverage capacity, Debt Service Coverage Ratio, maintenance covenant stress-testing, and Altman Z-Score diagnostics) with **Equity Research Valuation** (Unlevered Discounted Cash Flow, CAPM/WACC derivation, and multi-variable sensitivity matrices).

---

## 1. Executive Summary & Case Study Parameters

The model evaluates **Apex Industrial Systems Inc. (NYSE: APEX)**, a representative US GAAP mid-cap industrial manufacturing borrower carrying a **$780.0M Senior Secured Term Loan B** facility ($660.0M Net Debt) alongside **$1,200.0M** in baseline FY2025 revenue and **$264.0M** in baseline EBITDA (22.0% margin).

### Core Analytical Outputs
- **Weighted Average Cost of Capital (WACC):** **8.80%** (derived via CAPM with $R_f = 4.25\%$, $\beta_L = 1.20$, $\text{ERP} = 5.25\%$, and an after-tax cost of debt of $4.75\%$).
- **Intrinsic Valuation (Unlevered DCF):** Implied Enterprise Value of **$2,975.27M** and Implied Equity Value of **$2,315.27M** (**$46.31/share** vs. $36.00 current market price, representing **+28.6% upside**).
- **Solvency & Credit Cushion:** Baseline **Altman Z-Score of 2.90** (upper Grey Zone), **2.50x Net Leverage**, and **1.94x DSCR**. Under a **Mild Recession (-10% Rev, +50 bps)** shock with sticky fixed overhead (35% fixed cost structure), the borrower remains compliant across all maintenance covenants (**3.22x Net Leverage**, **3.89x Interest Coverage**, **1.43x DSCR**). However, it breaches all three covenants under both **Stagflation (-15% Rev, +200 bps)** (**3.77x Net Leverage**, **2.72x Interest Coverage**, **1.11x DSCR**) and **Severe Downside (-20% Rev, +150 bps)** (**4.53x Net Leverage**, **2.41x Interest Coverage**, **0.94x DSCR**).

---

## 2. Equity Research Valuation Module (Unlevered DCF)

### Methodology
1. **Unlevered Free Cash Flow ($\text{FCFF}$):**
   $$\text{FCFF}_t = \text{EBIT}_t \times (1 - T) + \text{D\&A}_t - \text{CapEx}_t - \Delta \text{NWC}_t$$
2. **Weighted Average Cost of Capital ($\text{WACC}$):**
   $$\text{WACC} = \left(\frac{E}{D + E} \times R_e\right) + \left(\frac{D}{D + E} \times R_d \times (1 - T)\right)$$
3. **Terminal Value (Gordon Growth Model):**
   $$\text{TV}_5 = \frac{\text{FCFF}_5 \times (1 + g)}{\text{WACC} - g}$$

### 5-Year Operating & FCFF Projection ($ Millions)

| Period | Revenue | EBITDA | EBIT | NOPAT | D&A | CapEx | $\Delta$ NWC | FCFF | PV of FCFF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FY2026E** | $1,272.00 | $282.38 | $231.50 | $175.94 | $50.88 | $57.24 | $10.80 | **$158.78** | $145.94 |
| **FY2027E** | $1,341.96 | $301.94 | $248.26 | $188.68 | $53.68 | $59.05 | $10.49 | **$172.82** | $146.00 |
| **FY2028E** | $1,406.37 | $320.65 | $264.40 | $200.94 | $56.25 | $60.47 | $9.66 | **$187.06** | $145.26 |
| **FY2029E** | $1,465.44 | $337.05 | $278.43 | $211.61 | $58.62 | $61.55 | $8.86 | **$199.82** | $142.62 |
| **FY2030E** | $1,516.73 | $348.85 | $288.18 | $219.02 | $60.67 | $60.67 | $7.69 | **$211.32** | $138.63 |

### Implied Share Price Sensitivity Matrix (WACC vs. Terminal Growth $g$)

| Discount Rate / TGR | $g = 1.50\%$ | $g = 2.00\%$ | **$g = 2.50\%$ (Base)** | $g = 3.00\%$ | $g = 3.50\%$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WACC 7.80%** | $48.38 | $52.67 | $57.76 | $63.92 | $71.52 |
| **WACC 8.30%** | $43.74 | $47.33 | $51.54 | $56.54 | $62.59 |
| **WACC 8.80% (Base)** | $39.74 | $42.78 | **$46.31** | $50.44 | $55.35 |
| **WACC 9.30%** | $36.25 | $38.86 | $41.84 | $45.30 | $49.36 |
| **WACC 9.80%** | $33.19 | $35.44 | $37.99 | $40.93 | $44.32 |

---

## 3. Corporate Banking Underwriting & Covenant Stress-Test Module

### Credit Facility Covenants & Cash Flow Coverage Definitions
To evaluate downside credit protection for senior secured lenders, the engine models operating leverage (35% fixed overhead, 65% variable operating costs) alongside floating-rate interest shocks and tests compliance against three standard maintenance covenants:
- **Net Leverage Ratio:** $\frac{\text{Total Debt} - \text{Cash}}{\text{EBITDA}} \le 3.50\text{x}$
- **Interest Coverage Ratio:** $\frac{\text{EBITDA}}{\text{Cash Interest Expense}} \ge 3.00\text{x}$
- **Debt Service Coverage Ratio (DSCR):** $\frac{\text{EBITDA} - \text{CapEx} - \text{Cash Taxes}}{\text{Cash Interest} + \text{Mandatory Amortization}} \ge 1.25\text{x}$

### Downside Stress-Test Results

| Stress Scenario | Revenue | EBITDA | Eff. Rate | Net Leverage ($\le 3.50\text{x}$) | Interest Coverage ($\ge 3.00\text{x}$) | DSCR ($\ge 1.25\text{x}$) | Covenant Compliance Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base Case (0% Rev, +0 bps)** | $1,200M | $264M | 6.25% | **2.50x** | **5.42x** | **1.94x** | **PASS (Compliant)** |
| **Mild Recession (-10% Rev, +50 bps)** | $1,080M | $205M | 6.75% | **3.22x** | **3.89x** | **1.43x** | **PASS (Compliant)** |
| **Stagflation (-15% Rev, +200 bps)** | $1,020M | $175M | 8.25% | **3.77x** | **2.72x** | **1.11x** | **BREACH (Net Leverage, Int. Coverage, DSCR)** |
| **Severe Downside (-20% Rev, +150 bps)** | $960M | $146M | 7.75% | **4.53x** | **2.41x** | **0.94x** | **BREACH (Net Leverage, Int. Coverage, DSCR)** |

---

## 4. Quantitative Financial Reasoning Audit Rubric

In addition to deterministic modeling, this repository codifies six verification guardrails used to audit financial calculations and LLM-generated financial analyses:

| Audit Dimension | Common Analytical Failure Mode | Enforced Verification Rule |
| :--- | :--- | :--- |
| **Cash Flow / Discount Rate Alignment** | Discounting Levered Cash Flow (FCFE) with WACC, or Unlevered Cash Flow (FCFF) with Cost of Equity. | Pair **FCFF** strictly with **WACC**; pair **FCFE** strictly with **Cost of Equity ($R_e$)**. |
| **Working Capital Sign Convention** | Adding increases in Operating NWC to Free Cash Flow instead of deducting cash tied up in receivables/inventory. | Subtract $\Delta \text{NWC} = \text{NWC}_t - \text{NWC}_{t-1}$ and exclude cash and short-term debt from NWC. |
| **Capital Structure Weights** | Using book value of equity instead of market capitalization when computing WACC weights. | Weight equity by **Market Value ($P \times \text{Shares}$)** and debt by market/carrying principal value. |
| **Terminal Growth Sanity Check** | Setting perpetual growth rate $g \ge \text{WACC}$ or exceeding long-run nominal GDP growth. | Enforce strict boundary $g < \text{WACC}$ and anchor $g$ to long-term inflation/GDP expectations (1.5%–3.5%). |
| **Operating Leverage in Stress Tests** | Assuming EBITDA margins stay constant when revenue drops during a recession scenario. | Decompose fixed vs. variable OpEx so EBITDA compresses faster than top-line revenue under downside shocks. |
| **DSCR Cash Flow Definition** | Dividing raw EBITDA by debt service without deducting maintenance CapEx and cash taxes. | Compute **CFADS** ($\text{EBITDA} - \text{CapEx} - \text{Cash Taxes}$) prior to dividing by Interest + Mandatory Principal. |

---

## 5. Covenant Breakeven & Debt Deleveraging

The **Covenant Breakeven Solver** (`find_covenant_breakeven_shocks`) uses bisection to solve the revenue contraction at which Net Leverage first reaches its 3.50x maximum and DSCR reaches its 1.25x minimum. Revenue sensitivity flows through the 35% fixed / 65% variable operating-cost structure, so fixed overhead remains sticky as sales fall. The solver reports the implied revenue and EBITDA dollar threshold for each covenant; DSCR uses CFADS (EBITDA less CapEx and cash taxes) over interest plus mandatory amortization.

The **5-Year Debt Deleveraging Schedule** (`project_credit_deleveraging`) compares the Base Case with a persistent Mild Recession Stress. Each year it calculates interest on beginning debt, mandatory amortization equal to 5% of original principal (capped at remaining debt), and an excess-CFADS cash sweep after interest and mandatory amortization. Debt repayments are floored at the outstanding balance, so ending debt cannot fall below $0.00M. The schedule reports ending net debt, net leverage, and DSCR for each year.

The Altman Z-Score calculation uses the case-study balance-sheet inputs of $1,625M total assets, $180M working capital, $450M retained earnings, and $900M total liabilities.

---

## 6. How to Run

```bash
pip install -r requirements.txt
python credit_dcf_model.py
python -m pytest -v
```
