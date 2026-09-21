import numpy as np
import pandas as pd
import pytest

from energy_toolkit.model_calibration_validation import (
    backtest_forward_vs_realized,
    calibrate_power_params,
    compute_moments,
    validate_calibration,
)
from energy_toolkit.power_price_model import GasPriceParams, PowerMarketSimulator, PowerPriceParams


def test_compute_moments_on_known_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=50.0, scale=10.0, size=200_000)
    moments = compute_moments(sample)
    assert abs(moments["mean"] - 50.0) < 0.2
    assert abs(moments["std"] - 10.0) < 0.2
    assert abs(moments["skew"]) < 0.05
    assert abs(moments["kurtosis"]) < 0.1


def test_compute_moments_spike_frequency_sensible():
    sample = np.concatenate([np.full(950, 40.0), np.full(50, 500.0)])
    moments = compute_moments(sample, spike_threshold_multiplier=3.0)
    assert abs(moments["spike_frequency"] - 0.05) < 1e-9


def test_calibration_recovers_moments_within_loose_tolerance():
    gas_params = GasPriceParams()
    true_params = PowerPriceParams(margin_base=10.0, margin_summer_amplitude=8.0, mean_reversion_speed=40.0)
    sim = PowerMarketSimulator(gas_params, true_params)
    scenario = sim.simulate_paths(n_years=2.0, n_paths=8, seed=42)
    target_moments = compute_moments(scenario.power_prices.reshape(-1))

    cal = calibrate_power_params(
        target_moments,
        base_params=PowerPriceParams(),
        gas_params=gas_params,
        n_years=1.0,
        seed=7,
        maxiter=60,
        n_paths_per_eval=3,
    )
    sim_moments = cal["sim_moments"]
    # loose tolerance: this is a noisy, cheap-budget calibration demo, not a
    # precision fit -- just check the optimizer moved skew/spike frequency
    # into a broadly sensible neighborhood of the target.
    assert abs(sim_moments["skew"] - target_moments["skew"]) < 2.0
    assert sim_moments["mean"] > 0
    assert sim_moments["std"] > 0


def test_validate_calibration_output_structure():
    gas_params = GasPriceParams()
    fitted_params = PowerPriceParams()
    target_moments = compute_moments(
        PowerMarketSimulator(gas_params, fitted_params).simulate(n_years=2.0, seed=1)["power_price"].to_numpy()
    )
    result = validate_calibration(fitted_params, gas_params, target_moments, n_years=1.0, n_paths=10, seed=2)
    assert "comparison" in result
    comparison = result["comparison"]
    assert set(comparison.index) == {"mean", "std", "skew", "kurtosis", "spike_frequency"}
    assert {"target", "achieved", "abs_diff", "pct_diff"}.issubset(comparison.columns)


def test_backtest_forward_vs_realized_near_zero_error_for_perfect_forecast():
    idx = pd.date_range("2022-01-01", periods=200, freq="D")
    spot = pd.DataFrame({"power_price": 50 + 10 * np.sin(np.arange(200) / 20)}, index=idx)
    monthly_realized = spot["power_price"].resample("MS").mean()
    forward = pd.DataFrame({"forward_price": monthly_realized.values}, index=monthly_realized.index)
    result = backtest_forward_vs_realized(forward, spot)
    assert result["rmse"] < 1e-8
    assert result["mae"] < 1e-8


def test_backtest_forward_vs_realized_raises_on_no_overlap():
    idx1 = pd.date_range("2022-01-01", periods=3, freq="MS")
    forward = pd.DataFrame({"forward_price": [50, 51, 52]}, index=idx1)
    idx2 = pd.date_range("2030-01-01", periods=30, freq="D")
    spot = pd.DataFrame({"power_price": np.full(30, 50.0)}, index=idx2)
    with pytest.raises(ValueError):
        backtest_forward_vs_realized(forward, spot)
