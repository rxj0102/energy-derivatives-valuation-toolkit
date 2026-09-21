import numpy as np

from energy_toolkit.battery_storage_optionality import (
    BatterySpec,
    resolve_soc_grid,
    value_battery_storage,
)
from energy_toolkit.power_price_model import simulate_diurnal_hourly_prices


def test_soc_grid_resolution_matches_power_and_capacity():
    spec = BatterySpec(power_mw=25.0, energy_capacity_mwh=100.0)
    n_levels, step = resolve_soc_grid(spec, dt_hours=1.0)
    assert n_levels == 5
    assert abs(step * (n_levels - 1) - spec.energy_capacity_mwh) < 1e-9


def test_constant_price_gives_zero_value_with_no_om_cost():
    spec = BatterySpec(power_mw=10, energy_capacity_mwh=40, round_trip_efficiency=0.9, variable_om_cost=0.0)
    const_price = np.full((30, 20), 42.0)
    result = value_battery_storage(const_price, spec, r=0.03, dt_hours=24.0)
    assert abs(result["mean_value"]) < 1e-6


def test_soc_path_stays_within_physical_bounds():
    rng = np.random.default_rng(3)
    daily = 40 + 20 * rng.standard_normal((15, 30))
    daily = np.clip(daily, 5, None)
    spec = BatterySpec(power_mw=5, energy_capacity_mwh=20, round_trip_efficiency=0.85)
    result = value_battery_storage(daily, spec, r=0.03, dt_hours=24.0)
    assert result["soc_path"].min() >= -1e-9
    assert result["soc_path"].max() <= spec.energy_capacity_mwh + 1e-9


def test_more_volatile_prices_increase_storage_value():
    rng = np.random.default_rng(4)
    n_paths, n_days = 60, 30
    base = 40 + 5 * rng.standard_normal((n_paths, n_days))
    base = np.clip(base, 5, None)
    volatile = 40 + 25 * rng.standard_normal((n_paths, n_days))
    volatile = np.clip(volatile, 5, None)

    spec = BatterySpec(power_mw=5, energy_capacity_mwh=20, round_trip_efficiency=0.9, variable_om_cost=0.2)
    low_vol_value = value_battery_storage(base, spec, r=0.03, dt_hours=24.0)["mean_value"]
    high_vol_value = value_battery_storage(volatile, spec, r=0.03, dt_hours=24.0)["mean_value"]
    assert high_vol_value > low_vol_value


def test_higher_round_trip_efficiency_never_decreases_value():
    rng = np.random.default_rng(6)
    daily = np.clip(40 + 20 * rng.standard_normal((50, 40)), 5, None)
    spec_lo = BatterySpec(power_mw=5, energy_capacity_mwh=20, round_trip_efficiency=0.6, variable_om_cost=0.3)
    spec_hi = BatterySpec(power_mw=5, energy_capacity_mwh=20, round_trip_efficiency=1.0, variable_om_cost=0.3)
    val_lo = value_battery_storage(daily, spec_lo, r=0.03, dt_hours=24.0)["mean_value"]
    val_hi = value_battery_storage(daily, spec_hi, r=0.03, dt_hours=24.0)["mean_value"]
    assert val_hi >= val_lo


def test_lsm_value_stabilizes_as_path_count_increases(small_scenario):
    spec = BatterySpec(power_mw=25, energy_capacity_mwh=100, round_trip_efficiency=0.85, variable_om_cost=0.5)
    daily = small_scenario.power_prices
    small_result = value_battery_storage(daily[:20], spec, r=0.03, dt_hours=24.0)
    large_result = value_battery_storage(daily, spec, r=0.03, dt_hours=24.0)
    # standard error should shrink materially with ~6x more paths
    assert large_result["std_error"] < small_result["std_error"]


def test_hourly_intraday_arbitrage_value_positive(small_scenario):
    daily = small_scenario.power_prices[:20, :10]
    hourly = np.array([simulate_diurnal_hourly_prices(daily[i], seed=100 + i) for i in range(daily.shape[0])])
    spec = BatterySpec(power_mw=25, energy_capacity_mwh=100, round_trip_efficiency=0.85, variable_om_cost=0.5)
    result = value_battery_storage(hourly, spec, r=0.03, dt_hours=1.0)
    assert result["mean_value"] > 0
