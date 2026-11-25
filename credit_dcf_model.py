"""
Corporate Credit Underwriting, Covenant Stress-Testing & DCF Valuation Engine

The model uses the Apex Industrial Systems (APEX) case-study assumptions
documented in README.md. Monetary figures are in USD millions unless noted.
"""

import pandas as pd


def get_baseline_inputs() -> dict:
    """Return a fresh copy of the documented APEX operating and market inputs."""
    company = {
        "name": "Apex Industrial Systems Inc. (NYSE: APEX)",
        "revenue_base": 1200.0,
        "ebitda_margin_base": 0.220,
        "da_pct_rev": 0.040,
        "capex_pct_rev_base": 0.045,
        "nwc_pct_rev": 0.150,
        "tax_rate": 0.240,
        "total_debt": 780.0,
        "cash": 120.0,
        "interest_rate": 0.0625,
        "mandatory_amort_pct": 0.050,
        "shares_outstanding": 50.0,
        "current_share_price": 36.00,
        "total_assets": 1625.0,
        "working_capital": 180.0,
        "retained_earnings": 450.0,
        "total_liabilities": 900.0,
    }
    wacc_inputs = {
        "rf": 0.0425,
        "erp": 0.0525,
        "beta_levered": 1.20,
        "pretax_cost_of_debt": 0.0625,
        "terminal_growth": 0.0250,
    }
    return {"company": company, "wacc": wacc_inputs}


def calculate_wacc(company: dict | None = None, wacc_inputs: dict | None = None) -> float:
    """Calculate market-value-weighted WACC using the model's tax convention."""
    inputs = get_baseline_inputs()
    company = company or inputs["company"]
    wacc_inputs = wacc_inputs or inputs["wacc"]

    cost_of_equity = wacc_inputs["rf"] + wacc_inputs["beta_levered"] * wacc_inputs["erp"]
    after_tax_cost_of_debt = wacc_inputs["pretax_cost_of_debt"] * (1.0 - company["tax_rate"])
    market_cap = company["shares_outstanding"] * company["current_share_price"]
    total_capital = market_cap + company["total_debt"]
    weight_equity = market_cap / total_capital
    weight_debt = company["total_debt"] / total_capital
    wacc = weight_equity * cost_of_equity + weight_debt * after_tax_cost_of_debt

    assert abs(weight_equity + weight_debt - 1.0) < 1e-9, (
        "AUDIT VIOLATION: Capital structure weights must sum to 100%."
    )
    if wacc <= 0.0:
        raise ValueError("WACC must be strictly positive.")
    return wacc


def build_dcf_projection(company: dict | None = None) -> pd.DataFrame:
    """Build the five-year FCFF projection with the documented NWC convention."""
    if company is None:
        company = get_baseline_inputs()["company"]

    years = ["FY2026E", "FY2027E", "FY2028E", "FY2029E", "FY2030E"]
    revenue_growth = [0.060, 0.055, 0.048, 0.042, 0.035]
    ebitda_margins = [0.222, 0.225, 0.228, 0.230, 0.230]
    capex_percentages = [0.045, 0.044, 0.043, 0.042, 0.040]
    projection_rows = []
    previous_revenue = company["revenue_base"]
    previous_nwc = previous_revenue * company["nwc_pct_rev"]

    for period, growth, ebitda_margin, capex_pct in zip(
        years, revenue_growth, ebitda_margins, capex_percentages
    ):
        revenue = previous_revenue * (1.0 + growth)
        ebitda = revenue * ebitda_margin
        da = revenue * company["da_pct_rev"]
        ebit = ebitda - da
        nopat = ebit * (1.0 - company["tax_rate"])
        capex = revenue * capex_pct
        nwc = revenue * company["nwc_pct_rev"]
        delta_nwc = nwc - previous_nwc

        if revenue > previous_revenue:
            assert delta_nwc > 0.0, "AUDIT VIOLATION: NWC expansion sign convention error."

        fcff = nopat + da - capex - delta_nwc
        projection_rows.append(
            {
                "Period": period,
                "Revenue": revenue,
                "EBITDA": ebitda,
                "EBIT": ebit,
                "NOPAT": nopat,
                "D&A": da,
                "CapEx": capex,
                "Delta_NWC": delta_nwc,
                "FCFF": fcff,
            }
        )
        previous_revenue = revenue
        previous_nwc = nwc

    return pd.DataFrame(projection_rows)


def calculate_valuation(
    company: dict | None = None, wacc_inputs: dict | None = None
) -> dict:
    """Return the DCF projection and enterprise-to-equity valuation bridge."""
    inputs = get_baseline_inputs()
    company = company or inputs["company"]
    wacc_inputs = wacc_inputs or inputs["wacc"]
    wacc = calculate_wacc(company, wacc_inputs)
    terminal_growth = wacc_inputs["terminal_growth"]
    if terminal_growth >= wacc:
        raise ValueError("Terminal growth rate must be strictly below WACC.")

    projection = build_dcf_projection(company)
    projection = projection.copy()
    projection["PV_FCFF"] = [
        row["FCFF"] / (1.0 + wacc) ** (index + 1)
        for index, (_, row) in enumerate(projection.iterrows())
    ]
    pv_explicit_fcff = projection["PV_FCFF"].sum()
    terminal_fcff = projection.iloc[-1]["FCFF"] * (1.0 + terminal_growth)
    terminal_value = terminal_fcff / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / (1.0 + wacc) ** len(projection)
    enterprise_value = pv_explicit_fcff + pv_terminal_value

    market_cap = company["shares_outstanding"] * company["current_share_price"]
    net_debt = company["total_debt"] - company["cash"]
    equity_value = enterprise_value - net_debt
    share_price = equity_value / company["shares_outstanding"]
    terminal_ebitda = projection.iloc[-1]["EBITDA"]
    exit_multiple = terminal_value / terminal_ebitda
    if not 4.0 <= exit_multiple <= 20.0:
        raise ValueError(
            f"Implied Exit EV/EBITDA ({exit_multiple:.2f}x) outside industrial sanity band."
        )

    return {
        "wacc": wacc,
        "projection": projection,
        "pv_explicit_fcff": pv_explicit_fcff,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "net_debt": net_debt,
        "equity_value": equity_value,
        "share_price": share_price,
        "implied_upside": share_price / company["current_share_price"] - 1.0,
        "exit_multiple": exit_multiple,
        "market_cap": market_cap,
    }


def calculate_altman_z_score(company: dict | None = None) -> float:
    """Calculate the public manufacturing Altman Z-Score."""
    if company is None:
        company = get_baseline_inputs()["company"]
    ebit = company["revenue_base"] * (
        company["ebitda_margin_base"] - company["da_pct_rev"]
    )
    x1 = company["working_capital"] / company["total_assets"]
    x2 = company["retained_earnings"] / company["total_assets"]
    x3 = ebit / company["total_assets"]
    x4 = (
        company["shares_outstanding"]
        * company["current_share_price"]
        / company["total_liabilities"]
    )
    x5 = company["revenue_base"] / company["total_assets"]
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + x5


def calculate_credit_metrics(
    revenue_shock: float = 0.0,
    rate_shock: float = 0.0,
    capex_flex: float = 1.0,
    company: dict | None = None,
    covenants: dict | None = None,
) -> dict:
    """Calculate operating leverage, CFADS, debt service, and covenant status."""
    inputs = get_baseline_inputs()
    company = company or inputs["company"]
    covenants = covenants or {
        "max_net_leverage": 3.50,
        "min_interest_cov": 3.00,
        "min_dscr": 1.25,
    }
    if capex_flex < 0:
        raise ValueError("CapEx flexibility must be non-negative.")

    base_opex = company["revenue_base"] * (1.0 - company["ebitda_margin_base"])
    fixed_opex = 0.35 * base_opex
    variable_cost_ratio = 0.65 * base_opex / company["revenue_base"]
    revenue = company["revenue_base"] * (1.0 + revenue_shock)
    ebitda = revenue * (1.0 - variable_cost_ratio) - fixed_opex
    da = company["revenue_base"] * company["da_pct_rev"]
    ebit = ebitda - da
    effective_rate = company["interest_rate"] + rate_shock
    interest_expense = company["total_debt"] * effective_rate
    mandatory_amortization = company["total_debt"] * company["mandatory_amort_pct"]
    total_debt_service = interest_expense + mandatory_amortization
    cash_taxes = max(0.0, (ebit - interest_expense) * company["tax_rate"])
    capex = (
        company["revenue_base"] * company["capex_pct_rev_base"] * capex_flex
    )
    cfads = ebitda - capex - cash_taxes
    net_debt = company["total_debt"] - company["cash"]
    net_leverage = net_debt / ebitda if ebitda > 0.0 else float("inf")
    interest_coverage = (
        ebitda / interest_expense if interest_expense > 0.0 else float("inf")
    )
    dscr = cfads / total_debt_service if total_debt_service > 0.0 else float("inf")

    breaches = []
    if ebitda <= 0.0 or net_leverage > covenants["max_net_leverage"]:
        breaches.append("Net Leverage")
    if ebitda <= 0.0 or interest_coverage < covenants["min_interest_cov"]:
        breaches.append("Int. Coverage")
    if dscr < covenants["min_dscr"]:
        breaches.append("DSCR")

    return {
        "revenue": revenue,
        "revenue_shock": revenue_shock,
        "ebitda": ebitda,
        "ebitda_margin": ebitda / revenue if revenue > 0.0 else float("nan"),
        "da": da,
        "ebit": ebit,
        "effective_rate": effective_rate,
        "interest_expense": interest_expense,
        "mandatory_amortization": mandatory_amortization,
        "total_debt_service": total_debt_service,
        "cash_taxes": cash_taxes,
        "capex": capex,
        "cfads": cfads,
        "net_debt": net_debt,
        "net_leverage": net_leverage,
        "interest_coverage": interest_coverage,
        "dscr": dscr,
        "breaches": breaches,
        "status": "PASS (Compliant)" if not breaches else f"BREACH ({', '.join(breaches)})",
    }


def _solve_monotonic_breakeven(metric, target: float) -> float:
    """Bisect a monotonic covenant metric over a downside revenue-shock range."""
    low, high = -0.99, 0.50
    low_value = metric(low) - target
    high_value = metric(high) - target
    if low_value * high_value > 0:
        raise ValueError("Covenant breakeven is outside the supported shock range.")
    for _ in range(100):
        middle = (low + high) / 2.0
        middle_value = metric(middle) - target
        if abs(middle_value) < 1e-12:
            return middle
        if low_value * middle_value <= 0:
            high = middle
            high_value = middle_value
        else:
            low = middle
            low_value = middle_value
    return (low + high) / 2.0


def find_covenant_breakeven_shocks(
    company: dict | None = None, covenants: dict | None = None
) -> pd.DataFrame:
    """Solve the revenue contraction at which leverage and DSCR first breach."""
    inputs = get_baseline_inputs()
    company = company or inputs["company"]
    covenants = covenants or {
        "max_net_leverage": 3.50,
        "min_interest_cov": 3.00,
        "min_dscr": 1.25,
    }
    leverage_shock = _solve_monotonic_breakeven(
        lambda shock: calculate_credit_metrics(
            revenue_shock=shock, company=company, covenants=covenants
        )["net_leverage"],
        covenants["max_net_leverage"],
    )
    dscr_shock = _solve_monotonic_breakeven(
        lambda shock: calculate_credit_metrics(
            revenue_shock=shock, company=company, covenants=covenants
        )["dscr"],
        covenants["min_dscr"],
    )

    rows = []
    for name, shock in (
        ("Maximum Net Leverage", leverage_shock),
        ("Minimum DSCR", dscr_shock),
    ):
        metrics = calculate_credit_metrics(
            revenue_shock=shock, company=company, covenants=covenants
        )
        rows.append(
            {
                "Covenant": name,
                "Revenue_Shock": shock,
                "Revenue": metrics["revenue"],
                "EBITDA": metrics["ebitda"],
                "Net_Leverage": metrics["net_leverage"],
                "DSCR": metrics["dscr"],
            }
        )
    return pd.DataFrame(rows)


def project_credit_deleveraging(
    company: dict | None = None, years: int = 5
) -> pd.DataFrame:
    """Project scheduled amortization and excess-CFADS cash sweep by scenario."""
    if years <= 0:
        raise ValueError("Projection horizon must be a positive number of years.")
    if company is None:
        company = get_baseline_inputs()["company"]

    scenarios = [
        {
            "Scenario": "Base Case",
            "Revenue_Shock": 0.0,
            "Rate_Shock": 0.0,
            "CapEx_Flex": 1.0,
        },
        {
            "Scenario": "Recession Stress",
            "Revenue_Shock": -0.10,
            "Rate_Shock": 0.005,
            "CapEx_Flex": 0.90,
        },
    ]
    rows = []
    original_debt = company["total_debt"]
    annual_mandatory = original_debt * company["mandatory_amort_pct"]
    base_opex = company["revenue_base"] * (1.0 - company["ebitda_margin_base"])
    fixed_opex = 0.35 * base_opex
    variable_cost_ratio = 0.65 * base_opex / company["revenue_base"]
    da = company["revenue_base"] * company["da_pct_rev"]

    for scenario in scenarios:
        debt_balance = original_debt
        for year in range(1, years + 1):
            revenue = company["revenue_base"] * (1.0 + scenario["Revenue_Shock"])
            ebitda = revenue * (1.0 - variable_cost_ratio) - fixed_opex
            interest = debt_balance * (company["interest_rate"] + scenario["Rate_Shock"])
            cash_taxes = max(0.0, (ebitda - da - interest) * company["tax_rate"])
            capex = (
                company["revenue_base"]
                * company["capex_pct_rev_base"]
                * scenario["CapEx_Flex"]
            )
            cfads = ebitda - capex - cash_taxes
            mandatory = min(debt_balance, annual_mandatory)
            debt_after_mandatory = debt_balance - mandatory
            cash_sweep = min(
                debt_after_mandatory,
                max(0.0, cfads - interest - mandatory),
            )
            ending_debt = max(0.0, debt_after_mandatory - cash_sweep)
            total_debt_service = interest + mandatory
            ending_net_debt = ending_debt - company["cash"]

            rows.append(
                {
                    "Scenario": scenario["Scenario"],
                    "Year": year,
                    "Beginning_Debt": debt_balance,
                    "EBITDA": ebitda,
                    "CFADS": cfads,
                    "Interest": interest,
                    "Mandatory_Amortization": mandatory,
                    "Cash_Sweep": cash_sweep,
                    "Ending_Debt": ending_debt,
                    "Ending_Net_Debt": ending_net_debt,
                    "Net_Leverage": (
                        ending_net_debt / ebitda if ebitda > 0.0 else float("inf")
                    ),
                    "DSCR": (
                        cfads / total_debt_service
                        if total_debt_service > 0.0
                        else float("inf")
                    ),
                }
            )
            debt_balance = ending_debt
    return pd.DataFrame(rows)


def run_corporate_finance_model() -> None:
    """Print the APEX valuation, solvency, covenant, and deleveraging report."""
    inputs = get_baseline_inputs()
    company = inputs["company"]
    wacc_inputs = inputs["wacc"]
    valuation = calculate_valuation(company, wacc_inputs)
    projection = valuation["projection"]

    print("=" * 92)
    print(f"INSTITUTIONAL CREDIT & VALUATION REPORT: {company['name']}")
    print("=" * 92)
    market_cap = company["shares_outstanding"] * company["current_share_price"]
    total_capital = market_cap + company["total_debt"]
    weight_equity = market_cap / total_capital
    weight_debt = company["total_debt"] / total_capital
    cost_of_equity = wacc_inputs["rf"] + wacc_inputs["beta_levered"] * wacc_inputs["erp"]
    after_tax_cost_of_debt = wacc_inputs["pretax_cost_of_debt"] * (1 - company["tax_rate"])
    print(
        f"Cost of Equity (Ke):      {cost_of_equity:.2%}  |  "
        f"After-Tax Cost of Debt (Kd): {after_tax_cost_of_debt:.2%}"
    )
    print(f"Market Capital Structure: {weight_equity:.2%} Equity / {weight_debt:.2%} Debt")
    print(
        f"Computed WACC:            {valuation['wacc']:.2%}  |  "
        f"Terminal Growth Rate (g):    {wacc_inputs['terminal_growth']:.2%}\n"
    )

    print("--- [PART 1A] 5-YEAR UNLEVERED FREE CASH FLOW (FCFF) PROJECTION ($M) ---")
    print(projection.round(2).to_string(index=False))
    print("\n--- [PART 1B] ENTERPRISE & EQUITY VALUATION BRIDGE ---")
    print(f"Cumulative PV of 5Y FCFF:       ${valuation['pv_explicit_fcff']:,.2f}M")
    print(
        f"PV of Terminal Value (Gordon):  ${valuation['pv_terminal_value']:,.2f}M "
        f"({valuation['pv_terminal_value'] / valuation['enterprise_value']:.1%} of EV)"
    )
    print(f"Implied Exit EV/EBITDA (FY30E): {valuation['exit_multiple']:.2f}x (Cross-Check)")
    print(f"Implied Enterprise Value (EV):  ${valuation['enterprise_value']:,.2f}M")
    print(
        f"Less: Net Debt (${company['total_debt']:,.0f}M - ${company['cash']:,.0f}M): "
        f"${valuation['net_debt']:,.2f}M"
    )
    print(f"Implied Equity Value:           ${valuation['equity_value']:,.2f}M")
    print(
        f"Implied Share Price:            ${valuation['share_price']:,.2f} "
        f"(Current: ${company['current_share_price']:.2f} | "
        f"Upside: {valuation['implied_upside']:+.1%})\n"
    )

    print("--- [PART 1C] IMPLIED SHARE PRICE SENSITIVITY MATRIX (WACC vs. g) ---")
    wacc_grid = [valuation["wacc"] + offset for offset in (-0.01, -0.005, 0, 0.005, 0.01)]
    growth_grid = [0.015, 0.020, 0.025, 0.030, 0.035]
    sensitivity = pd.DataFrame(
        index=[f"WACC {rate:.2%}" for rate in wacc_grid],
        columns=[f"g = {growth:.2%}" for growth in growth_grid],
    )
    for rate in wacc_grid:
        pv_stage = sum(
            row["FCFF"] / (1.0 + rate) ** (index + 1)
            for index, (_, row) in enumerate(projection.iterrows())
        )
        for growth in growth_grid:
            if growth >= rate:
                sensitivity.loc[f"WACC {rate:.2%}", f"g = {growth:.2%}"] = "N/M"
                continue
            terminal_value = projection.iloc[-1]["FCFF"] * (1 + growth) / (rate - growth)
            pv_terminal = terminal_value / (1 + rate) ** len(projection)
            equity_value = pv_stage + pv_terminal - valuation["net_debt"]
            sensitivity.loc[f"WACC {rate:.2%}", f"g = {growth:.2%}"] = (
                f"${equity_value / company['shares_outstanding']:,.2f}"
            )
    print(sensitivity.to_string())

    z_score = calculate_altman_z_score(company)
    if z_score > 2.99:
        z_classification = "Safe Zone (> 2.99)"
    elif z_score >= 1.81:
        z_classification = "Grey Zone (1.81 - 2.99) - Moderate Credit Watch"
    else:
        z_classification = "Distress Zone (< 1.81) - High Default Risk"
    print("\n--- [PART 2A] ALTMAN Z-SCORE SOLVENCY DIAGNOSTIC (PRO FORMA) ---")
    print(f"Composite Altman Z-Score: {z_score:.2f} -> {z_classification}\n")

    scenarios = [
        ("Base Case (0% Rev, +0 bps)", 0.00, 0.0000, 1.00),
        ("Mild Recession (-10% Rev, +50 bps)", -0.10, 0.0050, 0.90),
        ("Stagflation (-15% Rev, +200 bps)", -0.15, 0.0200, 0.85),
        ("Severe Downside (-20% Rev, +150 bps)", -0.20, 0.0150, 0.80),
    ]
    stress_rows = []
    for name, revenue_shock, rate_shock, capex_flex in scenarios:
        metrics = calculate_credit_metrics(
            revenue_shock, rate_shock, capex_flex, company
        )
        stress_rows.append(
            {
                "Scenario": name,
                "Revenue": f"${metrics['revenue']:,.0f}M",
                "EBITDA": f"${metrics['ebitda']:,.0f}M",
                "Eff_Rate": f"{metrics['effective_rate']:.2%}",
                "Net_Lev": f"{metrics['net_leverage']:.2f}x",
                "Int_Cov": f"{metrics['interest_coverage']:.2f}x",
                "DSCR": f"{metrics['dscr']:.2f}x",
                "Covenant_Status": metrics["status"],
            }
        )
    print("--- [PART 2B] COMMERCIAL BANKING COVENANT STRESS-TEST MATRIX ---")
    print("Maintenance Covenants: Max Net Lev <= 3.50x | Min Int. Cov >= 3.00x | Min DSCR >= 1.25x")
    print(pd.DataFrame(stress_rows).to_string(index=False))

    breakevens = find_covenant_breakeven_shocks(company)
    breakeven_display = breakevens.copy()
    breakeven_display["Revenue_Shock"] = breakeven_display["Revenue_Shock"].map(
        lambda value: f"{value:.2%}"
    )
    for column in ("Revenue", "EBITDA"):
        breakeven_display[column] = breakeven_display[column].map(
            lambda value: f"${value:,.2f}M"
        )
    breakeven_display["Net_Leverage"] = breakeven_display["Net_Leverage"].map(
        lambda value: f"{value:.3f}x"
    )
    breakeven_display["DSCR"] = breakeven_display["DSCR"].map(
        lambda value: f"{value:.3f}x"
    )
    print("\n--- [PART 2C] COVENANT BREAKEVEN REVENUE SHOCKS ---")
    print(breakeven_display.to_string(index=False))

    deleveraging = project_credit_deleveraging(company)
    deleveraging_display = deleveraging[
        [
            "Scenario",
            "Year",
            "Mandatory_Amortization",
            "Cash_Sweep",
        "Ending_Debt",
            "Ending_Net_Debt",
            "Net_Leverage",
            "DSCR",
        ]
    ].copy()
    for column in (
    "Mandatory_Amortization",
    "Cash_Sweep",
    "Ending_Debt",
    "Ending_Net_Debt",
    ):
        deleveraging_display[column] = deleveraging_display[column].map(
            lambda value: f"${value:,.2f}M"
        )
    for column in ("Net_Leverage", "DSCR"):
        deleveraging_display[column] = deleveraging_display[column].map(
            lambda value: f"{value:.2f}x"
        )
    print("\n--- [PART 2D] 5-YEAR DEBT AMORTIZATION & CASH SWEEP ($M) ---")
    print(deleveraging_display.to_string(index=False))
    print("=" * 92)


if __name__ == "__main__":
    run_corporate_finance_model()
