"""Calibration and validation of the power price model's parameters.

Since this toolkit has no real market data to calibrate against (see the
disclaimer in ``power_price_model.py``), calibration here targets a set of
*statistical moments* that a real desk would typically match a spot model
to: mean level, volatility, skewness (power prices are strongly
right-skewed due to spikes), and spike frequency (the fraction of days
priced above some multiple of the typical level). The "target" moments in
this module's tests and examples are themselves computed from a simulated
"market" price series, standing in for a real historical dataset — this
demonstrates the calibration *procedure*, not a fit to any genuine market.

Calibration uses ``scipy.optimize.minimize`` on an objective built from a
single fixed-seed simulation per evaluation, which keeps the (inherently
stochastic) objective function deterministic and therefore tractable for a
derivative-free optimizer.

Validation compares simulated vs. target moments out of sample (fresh
seeds, pooled across multiple paths for stable estimates) and backtests a
bootstrapped forward curve against subsequently realized spot prices.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from .power_price_model import GasPriceParams, PowerMarketSimulator, PowerPriceParams

DEFAULT_FREE_PARAMS: Tuple[str, ...] = (
    "margin_base",
    "margin_summer_amplitude",
    "mean_reversion_speed",
    "volatility",
    "jump_base_intensity",
    "jump_mean",
)

DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "margin_base": (1.0, 40.0),
    "margin_summer_amplitude": (0.0, 40.0),
    "mean_reversion_speed": (5.0, 200.0),
    "volatility": (0.1, 3.0),
    "jump_base_intensity": (0.5, 40.0),
    "jump_mean": (5.0, 200.0),
}


def compute_moments(prices: np.ndarray, spike_threshold_multiplier: float = 3.0) -> Dict[str, float]:
    """Summary statistics used both as calibration targets and validation checks."""
    prices = np.asarray(prices)
    median = float(np.median(prices))
    return {
        "mean": float(np.mean(prices)),
        "std": float(np.std(prices, ddof=1)),
        "skew": float(stats.skew(prices)),
        "kurtosis": float(stats.kurtosis(prices)),
        "spike_frequency": float(np.mean(prices > spike_threshold_multiplier * median)),
    }


def _params_from_vector(
    x: Sequence[float], base_params: PowerPriceParams, free_names: Sequence[str]
) -> PowerPriceParams:
    return replace(base_params, **dict(zip(free_names, x)))


def _simulate_moments(
    power_params: PowerPriceParams,
    gas_params: GasPriceParams,
    n_years: float,
    seed: int,
    spike_threshold_multiplier: float,
    n_paths: int = 5,
) -> Dict[str, float]:
    """Pool a handful of fixed-seed paths to reduce single-path sampling noise
    in the (otherwise stochastic) calibration objective, while keeping the
    objective deterministic across optimizer iterations for a given seed."""
    sim = PowerMarketSimulator(gas_params=gas_params, power_params=power_params)
    scenario = sim.simulate_paths(n_years=n_years, n_paths=n_paths, seed=seed)
    return compute_moments(scenario.power_prices.reshape(-1), spike_threshold_multiplier)


def _weighted_squared_error(sim_moments: Dict[str, float], target_moments: Dict[str, float]) -> float:
    error = 0.0
    for key, target in target_moments.items():
        sim_value = sim_moments[key]
        scale = target if abs(target) > 1e-8 else 1.0
        error += ((sim_value - target) / scale) ** 2
    return error


def calibration_objective(
    x: Sequence[float],
    base_params: PowerPriceParams,
    free_names: Sequence[str],
    gas_params: GasPriceParams,
    target_moments: Dict[str, float],
    n_years: float,
    seed: int,
    spike_threshold_multiplier: float,
    n_paths: int = 5,
) -> float:
    params = _params_from_vector(x, base_params, free_names)
    sim_moments = _simulate_moments(params, gas_params, n_years, seed, spike_threshold_multiplier, n_paths)
    return _weighted_squared_error(sim_moments, target_moments)


def calibrate_power_params(
    target_moments: Dict[str, float],
    base_params: Optional[PowerPriceParams] = None,
    gas_params: Optional[GasPriceParams] = None,
    free_param_names: Sequence[str] = DEFAULT_FREE_PARAMS,
    n_years: float = 5.0,
    seed: int = 123,
    method: str = "Nelder-Mead",
    maxiter: int = 300,
    spike_threshold_multiplier: float = 3.0,
    n_paths_per_eval: int = 5,
) -> Dict:
    """Fit a subset of ``PowerPriceParams`` fields to match ``target_moments``.

    Returns a dict with the fitted parameters, the raw ``OptimizeResult``,
    and the simulated moments achieved at the optimum (all from the same
    fixed-seed, pooled-path simulation used inside the objective).
    """
    base_params = base_params or PowerPriceParams()
    gas_params = gas_params or GasPriceParams()
    x0 = np.array([getattr(base_params, name) for name in free_param_names], dtype=float)

    result = minimize(
        calibration_objective,
        x0,
        args=(
            base_params,
            free_param_names,
            gas_params,
            target_moments,
            n_years,
            seed,
            spike_threshold_multiplier,
            n_paths_per_eval,
        ),
        method=method,
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6} if method == "Nelder-Mead" else {"maxiter": maxiter},
    )

    fitted_params = _params_from_vector(result.x, base_params, free_param_names)
    sim_moments = _simulate_moments(
        fitted_params, gas_params, n_years, seed, spike_threshold_multiplier, n_paths_per_eval
    )

    return {
        "fitted_params": fitted_params,
        "result": result,
        "sim_moments": sim_moments,
        "target_moments": target_moments,
    }


def validate_calibration(
    fitted_params: PowerPriceParams,
    gas_params: GasPriceParams,
    target_moments: Dict[str, float],
    n_years: float = 5.0,
    n_paths: int = 50,
    seed: int = 999,
    spike_threshold_multiplier: float = 3.0,
) -> Dict:
    """Out-of-sample validation: pool many fresh paths and compare moments to target."""
    sim = PowerMarketSimulator(gas_params=gas_params, power_params=fitted_params)
    scenario = sim.simulate_paths(n_years=n_years, n_paths=n_paths, seed=seed)
    pooled = scenario.power_prices.reshape(-1)
    achieved = compute_moments(pooled, spike_threshold_multiplier)

    comparison = pd.DataFrame(
        {
            "target": pd.Series(target_moments),
            "achieved": pd.Series(achieved),
        }
    )
    comparison["abs_diff"] = comparison["achieved"] - comparison["target"]
    comparison["pct_diff"] = comparison["abs_diff"] / comparison["target"].replace(0, np.nan)

    return {"achieved_moments": achieved, "comparison": comparison, "scenario": scenario}


def backtest_forward_vs_realized(
    forward_curve: pd.DataFrame,
    realized_spot: pd.DataFrame,
    forward_col: str = "forward_price",
    spot_col: str = "power_price",
) -> Dict[str, float]:
    """Compare a forward curve's monthly prices against subsequently realized
    monthly-average spot prices.

    Parameters
    ----------
    forward_curve:
        DataFrame indexed by delivery month (month-start Timestamp), as
        produced by ``forward_curve.historical_shape_forward_curve`` or
        ``forward_curve.analytic_forward_curve``.
    realized_spot:
        Daily spot price DataFrame (DatetimeIndex) covering (at least) the
        forward curve's delivery months.
    """
    realized_monthly = realized_spot[spot_col].resample("MS").mean()
    aligned = forward_curve[[forward_col]].join(realized_monthly.rename("realized"), how="inner")
    if aligned.empty:
        raise ValueError("No overlapping months between forward_curve and realized_spot")

    errors = aligned[forward_col] - aligned["realized"]
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    mape = float(np.mean(np.abs(errors) / aligned["realized"]))

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "mape": mape,
        "n_months": int(len(aligned)),
        "detail": aligned,
    }
