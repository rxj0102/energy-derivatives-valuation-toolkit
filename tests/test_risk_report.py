import numpy as np

from energy_toolkit.battery_storage_optionality import BatterySpec
from energy_toolkit.generation_asset_valuation import TollingAgreementSpec
from energy_toolkit.risk_report import (
    SwapHedgeSpec,
    compute_var_cvar,
    format_report_text,
    generate_risk_report,
    value_swap_hedge,
)


def test_cvar_is_never_less_than_var_on_random_sample():
    rng = np.random.default_rng(0)
    pnl = rng.normal(1000, 300, size=5000)
    metrics = compute_var_cvar(pnl, confidence_level=0.95)
    assert metrics["CVaR"] >= metrics["VaR"] - 1e-9


def test_var_matches_analytic_quantile_for_normal_pnl():
    from scipy.stats import norm

    rng = np.random.default_rng(1)
    mu, sigma = 0.0, 100.0
    pnl = rng.normal(mu, sigma, size=2_000_000)
    metrics = compute_var_cvar(pnl, confidence_level=0.95, relative_to_mean=True)
    analytic_var = norm.ppf(0.95) * sigma
    assert abs(metrics["VaR"] - analytic_var) < 1.0


def test_swap_hedge_short_power_gains_when_price_below_fixed(small_scenario):
    spec_short = SwapHedgeSpec(fixed_price=1000.0, notional_mwh_per_day=10.0, position="short_power")
    result = value_swap_hedge(small_scenario, spec_short, r=0.03)
    # fixed price set far above simulated spot -> short power position should
    # be solidly profitable in expectation
    assert result["mean_value"] > 0


def test_generate_risk_report_end_to_end_structure(small_scenario):
    tolling_spec = TollingAgreementSpec(capacity_mw=50, max_starts_per_month=4, min_run_days=1)
    battery_spec = BatterySpec(power_mw=10, energy_capacity_mwh=40, round_trip_efficiency=0.85)
    swap_spec = SwapHedgeSpec(fixed_price=55.0, notional_mwh_per_day=40.0, position="short_power")

    report = generate_risk_report(small_scenario, tolling_spec, battery_spec, swap_spec, r=0.04)
    table = report["var_cvar_table"]
    assert set(table["book"].unique()) == {"tolling_agreement", "battery_storage", "swap_hedge", "portfolio"}
    assert (table["CVaR"] >= table["VaR"] - 1e-6).all()
    assert report["n_paths"] == small_scenario.n_paths

    text = format_report_text(report)
    assert "SYNTHETIC PORTFOLIO RISK REPORT" in text
    assert "Portfolio total" in text


def test_portfolio_pnl_equals_sum_of_book_pnls(small_scenario):
    tolling_spec = TollingAgreementSpec(capacity_mw=50, max_starts_per_month=4, min_run_days=1)
    battery_spec = BatterySpec(power_mw=10, energy_capacity_mwh=40, round_trip_efficiency=0.85)
    swap_spec = SwapHedgeSpec(fixed_price=55.0, notional_mwh_per_day=40.0, position="short_power")
    report = generate_risk_report(small_scenario, tolling_spec, battery_spec, swap_spec, r=0.04)
    books = report["books"]
    expected = books["tolling"]["path_values"] + books["battery"]["path_values"] + books["swap"]["path_values"]
    np.testing.assert_allclose(books["portfolio_path_values"], expected)
