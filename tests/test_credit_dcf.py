import pytest

from credit_dcf_model import (
    build_dcf_projection,
    calculate_altman_z_score,
    calculate_credit_metrics,
    calculate_valuation,
    calculate_wacc,
    find_covenant_breakeven_shocks,
    get_baseline_inputs,
    project_credit_deleveraging,
)


def test_baseline_valuation_and_credit_outputs():
    inputs = get_baseline_inputs()
    company = inputs["company"]
    valuation = calculate_valuation(company, inputs["wacc"])
    credit = calculate_credit_metrics(company=company)

    assert calculate_wacc(company, inputs["wacc"]) == pytest.approx(0.0880, abs=0.0001)
    assert valuation["enterprise_value"] == pytest.approx(2975.27, abs=0.01)
    assert valuation["equity_value"] == pytest.approx(2315.27, abs=0.01)
    assert valuation["share_price"] == pytest.approx(46.31, abs=0.01)
    assert credit["net_leverage"] == pytest.approx(2.50, abs=0.01)
    assert credit["dscr"] == pytest.approx(1.94, abs=0.01)
    assert calculate_altman_z_score(company) == pytest.approx(2.90, abs=0.01)


def test_dcf_and_working_capital_invariants():
    inputs = get_baseline_inputs()
    company = inputs["company"]
    higher_nwc_company = {**company, "nwc_pct_rev": company["nwc_pct_rev"] + 0.01}
    baseline_fcff = build_dcf_projection(company)["FCFF"].sum()
    higher_nwc_fcff = build_dcf_projection(higher_nwc_company)["FCFF"].sum()

    assert higher_nwc_fcff < baseline_fcff

    invalid_wacc_inputs = {
        **inputs["wacc"],
        "terminal_growth": calculate_wacc(company, inputs["wacc"]),
    }
    with pytest.raises(ValueError, match="strictly below WACC"):
        calculate_valuation(company, invalid_wacc_inputs)

    valuation = calculate_valuation(company, inputs["wacc"])
    assert valuation["enterprise_value"] == pytest.approx(
        valuation["pv_explicit_fcff"] + valuation["pv_terminal_value"], abs=1e-4
    )


def test_operating_leverage_and_cfads_debt_service_definition():
    contracted = calculate_credit_metrics(revenue_shock=-0.10)
    baseline = calculate_credit_metrics()

    assert contracted["ebitda_margin"] < 0.22
    assert baseline["dscr"] == pytest.approx(1.94, abs=0.01)
    assert baseline["interest_coverage"] == pytest.approx(5.42, abs=0.01)
    assert baseline["dscr"] != pytest.approx(baseline["interest_coverage"])
    assert baseline["dscr"] == pytest.approx(
        baseline["cfads"] / baseline["total_debt_service"], abs=1e-12
    )


def test_covenant_breakevens_reproduce_thresholds_and_stress_statuses():
    inputs = get_baseline_inputs()
    company = inputs["company"]
    breakevens = find_covenant_breakeven_shocks(company).set_index("Covenant")

    leverage_shock = breakevens.loc["Maximum Net Leverage", "Revenue_Shock"]
    leverage_metrics = calculate_credit_metrics(
        revenue_shock=leverage_shock, company=company
    )
    assert leverage_metrics["net_leverage"] == pytest.approx(3.50, abs=1e-3)

    dscr_shock = breakevens.loc["Minimum DSCR", "Revenue_Shock"]
    dscr_metrics = calculate_credit_metrics(revenue_shock=dscr_shock, company=company)
    assert dscr_metrics["dscr"] == pytest.approx(1.25, abs=1e-3)
    assert breakevens.loc["Maximum Net Leverage", "EBITDA"] == pytest.approx(
        (company["total_debt"] - company["cash"]) / 3.50, abs=1e-6
    )

    mild = calculate_credit_metrics(-0.10, 0.005, 0.90, company)
    stagflation = calculate_credit_metrics(-0.15, 0.020, 0.85, company)
    severe = calculate_credit_metrics(-0.20, 0.015, 0.80, company)
    assert mild["status"] == "PASS (Compliant)"
    assert stagflation["status"] == "BREACH (Net Leverage, Int. Coverage, DSCR)"
    assert severe["status"] == "BREACH (Net Leverage, Int. Coverage, DSCR)"


def test_five_year_deleveraging_has_bounded_debt_and_valid_metrics():
    schedule = project_credit_deleveraging()

    assert len(schedule) == 10
    assert set(schedule["Scenario"]) == {"Base Case", "Recession Stress"}
    assert (schedule["Ending_Debt"] >= 0.0).all()
    assert (schedule["Cash_Sweep"] >= 0.0).all()
    assert (
        schedule["Cash_Sweep"]
        <= schedule["Beginning_Debt"] - schedule["Mandatory_Amortization"] + 1e-10
    ).all()
    assert (schedule["Ending_Debt"] <= schedule["Beginning_Debt"]).all()
    assert schedule.groupby("Scenario")["Year"].nunique().eq(5).all()
    assert (schedule["DSCR"] > 0.0).all()
