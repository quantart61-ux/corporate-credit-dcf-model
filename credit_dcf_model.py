"""
Corporate Credit Underwriting, Covenant Stress-Testing & DCF Valuation Engine
Author: Huynh Vi
Framework: US GAAP Corporate Banking Credit Risk & Institutional Equity Research
"""

import pandas as pd


def run_corporate_finance_model() -> None:
    # ==========================================================================
    # 1. BASELINE FINANCIALS & PRO FORMA CAPITAL STRUCTURE ($ MILLIONS)
    # Benchmark: Caterpillar Inc. (NYSE: CAT) - ME&T Industrial Segment Baseline
    # Credit Case: Pro Forma Senior Secured Acquisition Financing Facility
    # ==========================================================================
    company = {
        "name": "Caterpillar Inc. (NYSE: CAT) - ME&T Industrial Pro Forma Case",
        "revenue_base": 61200.0,
        "ebitda_margin_base": 0.235,      # 23.5% ME&T Operating EBITDA Margin
        "da_pct_rev": 0.030,              # 3.0% Depreciation & Amortization
        "capex_pct_rev_base": 0.032,      # 3.2% Maintenance & Growth CapEx
        "nwc_pct_rev": 0.160,             # 16.0% Operating Net Working Capital
        "tax_rate": 0.225,                # 22.5% Effective US GAAP Tax Rate
        # Pro Forma Senior Secured Credit Facility & Liquidity
        "total_debt": 38500.0,            # Senior Secured Term Loan B (Floating SOFR + Spread)
        "cash": 4500.0,                   # Unrestricted ME&T Cash & Equivalents
        "interest_rate": 0.0625,          # 6.25% Baseline All-In Floating Coupon
        "mandatory_amort_pct": 0.050,     # 5.0% Mandatory Annual Principal Amortization
        # Equity Market Data (ME&T Implied Standalone Equity)
        "shares_outstanding": 485.0,      # Diluted Shares (Millions)
        "current_share_price": 340.00,    # Reference Market Price ($/share)
        # Balance Sheet Items (for Altman Z-Score)
        "total_assets": 86500.0,
        "working_capital": 11200.0,
        "retained_earnings": 38400.0,
        "total_liabilities": 62800.0,
    }

    wacc_inputs = {
        "rf": 0.0425,                     # 10Y US Treasury Yield
        "erp": 0.0525,                    # Implied Equity Risk Premium
        "beta_levered": 1.15,             # Industrial Sector Levered Beta
        "pretax_cost_of_debt": 0.0625,    # Pre-Tax Cost of Senior Debt
        "terminal_growth": 0.0250,        # Long-Term Nominal GDP Anchor
    }

    # ==========================================================================
    # 2. COST OF CAPITAL (CAPM & WACC) + AUDIT ASSERTIONS
    # ==========================================================================
    cost_of_equity = wacc_inputs["rf"] + (wacc_inputs["beta_levered"] * wacc_inputs["erp"])
    after_tax_cost_of_debt = wacc_inputs["pretax_cost_of_debt"] * (1.0 - company["tax_rate"])

    market_cap = company["shares_outstanding"] * company["current_share_price"]
    total_capital = market_cap + company["total_debt"]
    weight_equity = market_cap / total_capital
    weight_debt = company["total_debt"] / total_capital

    wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)

    # Automated Financial Reasoning Guardrails (Audit Assertions)
    assert abs((weight_equity + weight_debt) - 1.0) < 1e-9, (
        "AUDIT VIOLATION: Capital structure weights must sum to 100%."
    )
    assert wacc > 0.0, "AUDIT VIOLATION: WACC must be strictly positive."
    assert wacc_inputs["terminal_growth"] < wacc, (
        "AUDIT VIOLATION: Terminal growth rate cannot equal or exceed WACC."
    )

    print("=" * 92)
    print(f"INSTITUTIONAL CREDIT & VALUATION REPORT: {company['name']}")
    print("=" * 92)
    print(f"Cost of Equity (Ke):      {cost_of_equity:.2%}  |  After-Tax Cost of Debt (Kd): {after_tax_cost_of_debt:.2%}")
    print(f"Market Capital Structure: {weight_equity:.2%} Equity / {weight_debt:.2%} Debt")
    print(f"Computed WACC:            {wacc:.2%}  |  Terminal Growth Rate (g):    {wacc_inputs['terminal_growth']:.2%}\n")

    # ==========================================================================
    # 3. 5-YEAR OPERATING PROJECTION & UNLEVERED DCF (FCFF)
    # ==========================================================================
    years = ["FY2026E", "FY2027E", "FY2028E", "FY2029E", "FY2030E"]
    rev_growth = [0.050, 0.045, 0.040, 0.035, 0.030]
    ebitda_margins = [0.236, 0.238, 0.240, 0.240, 0.240]
    capex_pcts = [0.032, 0.032, 0.031, 0.031, 0.030]

    projection_rows = []
    prev_rev = company["revenue_base"]
    prev_nwc = prev_rev * company["nwc_pct_rev"]

    for i, yr in enumerate(years):
        rev = prev_rev * (1.0 + rev_growth[i])
        ebitda = rev * ebitda_margins[i]
        da = rev * company["da_pct_rev"]
        ebit = ebitda - da
        nopat = ebit * (1.0 - company["tax_rate"])
        capex = rev * capex_pcts[i]
        nwc = rev * company["nwc_pct_rev"]
        delta_nwc = nwc - prev_nwc

        # Audit Assertion: Growing revenue with constant NWC% must generate a positive NWC outflow
        if rev > prev_rev:
            assert delta_nwc > 0.0, "AUDIT VIOLATION: NWC expansion sign convention error."

        # Unlevered Free Cash Flow (FCFF = NOPAT + D&A - CapEx - Change in NWC)
        fcff = nopat + da - capex - delta_nwc
        discount_factor = 1.0 / ((1.0 + wacc) ** (i + 1))
        pv_fcff = fcff * discount_factor

        projection_rows.append({
            "Period": yr,
            "Revenue": rev,
            "EBITDA": ebitda,
            "EBIT": ebit,
            "NOPAT": nopat,
            "D&A": da,
            "CapEx": capex,
            "Delta_NWC": delta_nwc,
            "FCFF": fcff,
            "PV_FCFF": pv_fcff,
        })
        prev_rev = rev
        prev_nwc = nwc

    dcf_df = pd.DataFrame(projection_rows)
    print("--- [PART 1A] 5-YEAR UNLEVERED FREE CASH FLOW (FCFF) PROJECTION ($M) ---")
    print(dcf_df.round(2).to_string(index=False))

    # Valuation Bridge (Enterprise Value -> Implied Equity Value -> Share Price)
    sum_pv_fcff = dcf_df["PV_FCFF"].sum()
    terminal_fcff = dcf_df.iloc[-1]["FCFF"] * (1.0 + wacc_inputs["terminal_growth"])
    terminal_value = terminal_fcff / (wacc - wacc_inputs["terminal_growth"])
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** len(years))

    # Institutional Cross-Check: Implied Exit EV / FY2030E EBITDA Multiple
    terminal_ebitda = dcf_df.iloc[-1]["EBITDA"]
    implied_exit_multiple = terminal_value / terminal_ebitda
    assert 4.0 <= implied_exit_multiple <= 20.0, (
        f"AUDIT VIOLATION: Implied Exit EV/EBITDA ({implied_exit_multiple:.2f}x) outside industrial sanity band."
    )

    enterprise_value = sum_pv_fcff + pv_terminal_value
    net_debt = company["total_debt"] - company["cash"]
    implied_equity_value = enterprise_value - net_debt
    implied_share_price = implied_equity_value / company["shares_outstanding"]
    implied_upside = (implied_share_price / company["current_share_price"]) - 1.0

    print("\n--- [PART 1B] ENTERPRISE & EQUITY VALUATION BRIDGE ---")
    print(f"Cumulative PV of 5Y FCFF:       ${sum_pv_fcff:,.2f}M")
    print(f"PV of Terminal Value (Gordon):  ${pv_terminal_value:,.2f}M ({pv_terminal_value / enterprise_value:.1%} of EV)")
    print(f"Implied Exit EV/EBITDA (FY30E): {implied_exit_multiple:.2f}x (Cross-Check)")
    print(f"Implied Enterprise Value (EV):  ${enterprise_value:,.2f}M")
    print(f"Less: Net Debt (${company['total_debt']:,.0f}M - ${company['cash']:,.0f}M): ${net_debt:,.2f}M")
    print(f"Implied Equity Value:           ${implied_equity_value:,.2f}M")
    print(f"Implied Share Price:            ${implied_share_price:,.2f} (Current: ${company['current_share_price']:.2f} | Upside: {implied_upside:+.1%})\n")

    # ==========================================================================
    # 4. VALUATION SENSITIVITY MATRIX (WACC vs. TERMINAL GROWTH RATE)
    # ==========================================================================
    wacc_grid = [wacc - 0.010, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.010]
    tgr_grid = [0.015, 0.020, 0.025, 0.030, 0.035]

    sens_df = pd.DataFrame(
        index=[f"WACC {w:.2%}" for w in wacc_grid],
        columns=[f"g = {g:.2%}" for g in tgr_grid],
    )

    for w in wacc_grid:
        pv_stage_1 = sum(
            dcf_df.iloc[t]["FCFF"] / ((1.0 + w) ** (t + 1)) for t in range(len(years))
        )
        for g in tgr_grid:
            tv = (dcf_df.iloc[-1]["FCFF"] * (1.0 + g)) / (w - g)
            pv_tv = tv / ((1.0 + w) ** len(years))
            eq_val = pv_stage_1 + pv_tv - net_debt
            sens_df.loc[f"WACC {w:.2%}", f"g = {g:.2%}"] = f"${eq_val / company['shares_outstanding']:,.2f}"

    print("--- [PART 1C] IMPLIED SHARE PRICE SENSITIVITY MATRIX (WACC vs. g) ---")
    print(sens_df.to_string())

    # ==========================================================================
    # 5. SOLVENCY DIAGNOSTICS: ALTMAN Z-SCORE (PUBLIC MANUFACTURING MODEL)
    # ==========================================================================
    ebit_base = company["revenue_base"] * (company["ebitda_margin_base"] - company["da_pct_rev"])
    x1 = company["working_capital"] / company["total_assets"]
    x2 = company["retained_earnings"] / company["total_assets"]
    x3 = ebit_base / company["total_assets"]
    x4 = market_cap / company["total_liabilities"]
    x5 = company["revenue_base"] / company["total_assets"]

    z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)
    if z_score > 2.99:
        z_classification = "Safe Zone (> 2.99)"
    elif z_score >= 1.81:
        z_classification = "Grey Zone (1.81 - 2.99) - Moderate Credit Watch"
    else:
        z_classification = "Distress Zone (< 1.81) - High Default Risk"

    print("\n--- [PART 2A] ALTMAN Z-SCORE SOLVENCY DIAGNOSTIC (PRO FORMA) ---")
    print(f"X1 (WC/TA): {x1:.3f} | X2 (RE/TA): {x2:.3f} | X3 (EBIT/TA): {x3:.3f} | X4 (MVE/TL): {x4:.3f} | X5 (Sales/TA): {x5:.3f}")
    print(f"Composite Altman Z-Score: {z_score:.2f} -> {z_classification}\n")

    # ==========================================================================
    # 6. CORPORATE BANKING UNDERWRITING & COVENANT STRESS-TEST ENGINE
    # ==========================================================================
    # Standard Senior Secured Credit Facility Maintenance Covenants:
    # - Maximum Net Leverage (Net Debt / EBITDA) <= 3.50x
    # - Minimum Interest Coverage (EBITDA / Cash Interest) >= 3.00x
    # - Minimum DSCR ((EBITDA - CapEx - Cash Taxes) / Total Debt Service) >= 1.25x
    covenants = {
        "max_net_leverage": 3.50,
        "min_interest_cov": 3.00,
        "min_dscr": 1.25,
    }

    # Cost Structure Decomposition (Operating Leverage Modeling):
    # 65% Variable Operating Costs, 35% Sticky Fixed Overhead
    base_opex = company["revenue_base"] * (1.0 - company["ebitda_margin_base"])
    fixed_opex = 0.35 * base_opex
    variable_cost_ratio = (0.65 * base_opex) / company["revenue_base"]

    stress_scenarios = [
        {"Scenario": "Base Case (0% Rev, +0 bps)", "Rev_Shock": 0.00, "Rate_Shock": 0.0000, "CapEx_Flex": 1.00},
        {"Scenario": "Mild Recession (-10% Rev, +50 bps)", "Rev_Shock": -0.10, "Rate_Shock": 0.0050, "CapEx_Flex": 0.90},
        {"Scenario": "Stagflation (-15% Rev, +200 bps)", "Rev_Shock": -0.15, "Rate_Shock": 0.0200, "CapEx_Flex": 0.85},
        {"Scenario": "Severe Downside (-20% Rev, +150 bps)", "Rev_Shock": -0.20, "Rate_Shock": 0.0150, "CapEx_Flex": 0.80},
    ]

    mandatory_amortization = company["total_debt"] * company["mandatory_amort_pct"]
    stress_rows = []

    for sc in stress_scenarios:
        rev = company["revenue_base"] * (1.0 + sc["Rev_Shock"])
        ebitda = (rev * (1.0 - variable_cost_ratio)) - fixed_opex
        da = company["revenue_base"] * company["da_pct_rev"]  # D&A fixed on installed PP&E
        ebit = ebitda - da

        effective_rate = company["interest_rate"] + sc["Rate_Shock"]
        interest_expense = company["total_debt"] * effective_rate
        total_debt_service = interest_expense + mandatory_amortization

        ebt = ebit - interest_expense
        cash_taxes = max(0.0, ebt * company["tax_rate"])
        capex = (company["revenue_base"] * company["capex_pct_rev_base"]) * sc["CapEx_Flex"]
        cfads = ebitda - capex - cash_taxes  # Cash Flow Available for Debt Service

        breaches = []
        if ebitda <= 0.0:
            net_lev_str = "N/M (Neg. EBITDA)"
            int_cov_str = "N/M (Neg. EBITDA)"
            breaches.extend(["Net Leverage", "Int. Coverage"])
        else:
            net_leverage = net_debt / ebitda
            interest_coverage = ebitda / interest_expense
            net_lev_str = f"{net_leverage:.2f}x"
            int_cov_str = f"{interest_coverage:.2f}x"
            if net_leverage > covenants["max_net_leverage"]:
                breaches.append("Net Leverage")
            if interest_coverage < covenants["min_interest_cov"]:
                breaches.append("Int. Coverage")

        dscr = cfads / total_debt_service
        if dscr < covenants["min_dscr"]:
            breaches.append("DSCR")

        status = "PASS (Compliant)" if not breaches else f"BREACH ({', '.join(breaches)})"

        stress_rows.append({
            "Scenario": sc["Scenario"],
            "Revenue": f"${rev:,.0f}M",
            "EBITDA": f"${ebitda:,.0f}M",
            "Eff_Rate": f"{effective_rate:.2%}",
            "Net_Lev": net_lev_str,
            "Int_Cov": int_cov_str,
            "DSCR": f"{dscr:.2f}x",
            "Covenant_Status": status,
        })

    stress_df = pd.DataFrame(stress_rows)
    print("--- [PART 2B] COMMERCIAL BANKING COVENANT STRESS-TEST MATRIX ---")
    print("Maintenance Covenants: Max Net Lev <= 3.50x | Min Int. Cov >= 3.00x | Min DSCR >= 1.25x")
    print(stress_df.to_string(index=False))
    print("=" * 92)


if __name__ == "__main__":
    run_corporate_finance_model()
