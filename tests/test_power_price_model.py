import numpy as np

from energy_toolkit.power_price_model import (
    PowerMarketSimulator,
    PowerPriceParams,
    simulate_diurnal_hourly_prices,
)


def test_simulate_paths_shapes():
    sim = PowerMarketSimulator()
    scenario = sim.simulate_paths(n_years=2.0, n_paths=10, dt_days=1.0, seed=1)
    expected_steps = int(round(2.0 * 365.25))
    assert scenario.power_prices.shape == (10, expected_steps)
    assert scenario.gas_prices.shape == (10, expected_steps)
    assert scenario.jump_flags.shape == (10, expected_steps)
    assert len(scenario.dates) == expected_steps


def test_price_floor_respected():
    power_params = PowerPriceParams(price_floor=0.5)
    sim = PowerMarketSimulator(power_params=power_params)
    scenario = sim.simulate_paths(n_years=5.0, n_paths=50, seed=3)
    assert scenario.power_prices.min() >= power_params.price_floor - 1e-9


def test_reproducibility_with_seed():
    sim = PowerMarketSimulator()
    a = sim.simulate_paths(n_years=1.0, n_paths=5, seed=42)
    b = sim.simulate_paths(n_years=1.0, n_paths=5, seed=42)
    np.testing.assert_array_equal(a.power_prices, b.power_prices)
    np.testing.assert_array_equal(a.gas_prices, b.gas_prices)


def test_gas_and_power_shocks_are_positively_correlated(medium_scenario):
    """Raw price levels are dominated by (largely independent) jump risk and
    by gas/power having different seasonal phases, which can mask the
    underlying shock correlation. Isolate it by looking at day-over-day
    changes on non-jump days, which is what ``price_gas_correlation``
    actually controls."""
    gas_returns = np.diff(np.log(medium_scenario.gas_prices), axis=1)
    power_changes = np.diff(medium_scenario.power_prices, axis=1)
    non_jump = ~medium_scenario.jump_flags[:, 1:]
    corr = np.corrcoef(gas_returns[non_jump], power_changes[non_jump])[0, 1]
    assert corr > 0.1


def test_summer_jump_intensity_exceeds_winter(medium_scenario):
    months = medium_scenario.dates.month
    summer_mask = np.isin(months, [6, 7, 8])
    winter_mask = np.isin(months, [12, 1, 2])
    summer_jump_rate = medium_scenario.jump_flags[:, summer_mask].mean()
    winter_jump_rate = medium_scenario.jump_flags[:, winter_mask].mean()
    assert summer_jump_rate > winter_jump_rate


def test_power_price_series_is_right_skewed(medium_scenario):
    from scipy.stats import skew

    prices = medium_scenario.power_prices.reshape(-1)
    assert skew(prices) > 0.5


def test_mean_reversion_autocorrelation_decays_with_lag():
    """A mean-reverting process should show autocorrelation that decays
    toward zero as lag increases (unlike a random walk, whose
    autocorrelation stays near 1). De-seasonalize crudely with a short
    rolling-mean subtraction so the annual seasonal cycle doesn't
    contaminate the lag structure."""
    power_params = PowerPriceParams(mean_reversion_speed=80.0, jump_base_intensity=0.0,
                                     jump_summer_intensity_multiplier=1.0)
    sim = PowerMarketSimulator(power_params=power_params)
    scenario = sim.simulate_paths(n_years=5.0, n_paths=1, seed=11)
    prices = scenario.power_prices[0]
    detrended = prices - np.convolve(prices, np.ones(31) / 31, mode="same")

    def autocorr(x, lag):
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]

    short_lag_corr = autocorr(detrended, 1)
    long_lag_corr = autocorr(detrended, 90)
    assert short_lag_corr > 0.5
    assert abs(long_lag_corr) < 0.15
    assert short_lag_corr > long_lag_corr


def test_diurnal_hourly_expansion_preserves_daily_mean():
    daily = np.array([40.0, 60.0, 25.0])
    hourly = simulate_diurnal_hourly_prices(daily, noise_std_frac=0.0, seed=7)
    hourly = hourly.reshape(3, 24)
    np.testing.assert_allclose(hourly.mean(axis=1), daily, rtol=1e-6)
