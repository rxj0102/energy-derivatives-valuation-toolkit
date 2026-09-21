import numpy as np
import pandas as pd

from energy_toolkit.forward_curve import (
    analytic_forward_curve,
    compare_curves,
    compute_monthly_seasonal_shape,
    historical_shape_forward_curve,
)
from energy_toolkit.power_price_model import GasPriceParams, PowerPriceParams


def test_monthly_seasonal_shape_has_unit_mean(medium_scenario):
    df = medium_scenario.to_frame(0)
    shape = compute_monthly_seasonal_shape(df)
    assert len(shape) == 12
    assert abs(shape.mean() - 1.0) < 1e-9


def test_historical_shape_curve_length_and_positivity(medium_scenario):
    df = medium_scenario.to_frame(0)
    val_date = df.index[400]
    curve = historical_shape_forward_curve(df, val_date, horizon_months=12)
    assert len(curve) == 12
    assert (curve["forward_price"] > 0).all()
    # delivery months should be strictly increasing and start after valuation date
    assert curve.index.is_monotonic_increasing
    assert curve.index[0] > val_date


def test_analytic_curve_reverts_to_seasonal_mean_at_long_horizon():
    gas_params = GasPriceParams()
    power_params = PowerPriceParams()
    curve = analytic_forward_curve(
        gas_params, power_params, current_gas_price=8.0, current_margin_state=40.0,
        valuation_date="2021-01-01", horizon_months=120,
    )
    # far out, the OU state should have decayed to ~0, leaving forward price
    # close to heat_rate * long_run_gas_mean + seasonal margin at that month
    long_horizon_price = curve["forward_price"].iloc[-1]
    near_horizon_price = curve["forward_price"].iloc[0]
    # the elevated starting state (40 above baseline) should have decayed
    # much more by month 120 than by month 1
    assert abs(long_horizon_price - power_params.margin_base - power_params.heat_rate * gas_params.long_run_mean) < \
        abs(near_horizon_price - power_params.margin_base - power_params.heat_rate * gas_params.long_run_mean)


def test_compare_curves_aligns_on_delivery_month():
    idx = pd.date_range("2022-01-01", periods=6, freq="MS")
    curve_a = pd.DataFrame({"forward_price": np.arange(6) + 50.0}, index=idx)
    curve_b = pd.DataFrame({"forward_price": np.arange(6) + 45.0}, index=idx)
    comparison = compare_curves(curve_a, curve_b)
    assert len(comparison) == 6
    np.testing.assert_allclose(comparison["abs_diff"], 5.0)
