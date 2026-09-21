"""Forward curve construction for the synthetic power market.

Two complementary construction methods are provided:

1. ``historical_shape_forward_curve`` — the classic desk-practitioner
   "bootstrap" approach: normalize a historical (here, simulated) monthly
   seasonal shape and anchor it to a current price level. This is exactly
   the kind of bootstrap cQuant/Zema-style tools perform when building a
   forward curve from limited market quotes plus a historical shape.

2. ``analytic_forward_curve`` — a semi-analytical, model-implied curve
   using the closed-form conditional expectation of the mean-reverting
   jump-diffusion spot process. Since jump intensity and the mean level
   are time-varying (seasonal), the jump contribution is evaluated by
   discretized variation-of-constants (an exact expression for constant
   coefficients between grid points), not a generic simulation average.

The two curves are independent constructions and are expected to differ
somewhat; ``compare_curves`` cross-checks them, which is a useful sanity
check on both the simulator's internal consistency and the bootstrap
methodology.

DISCLAIMER: these are curves over a synthetic spot process, not real
tradeable market quotes. No market price of risk / risk premium is
applied — the forward curve is built under the real-world (physical)
measure, a simplifying assumption clearly called out here since it is not
generally true in real commodity markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .power_price_model import (
    DAYS_PER_YEAR,
    GasPriceParams,
    PowerPriceParams,
    gas_seasonal_log_mean,
    jump_intensity_per_year,
    power_margin_seasonal_mean,
)


def compute_monthly_seasonal_shape(spot_df: pd.DataFrame, price_col: str = "power_price") -> pd.Series:
    """Average price by calendar month, normalized to a unit mean.

    Returns a length-12 Series indexed 1..12 (Jan..Dec) of multiplicative
    seasonal factors.
    """
    monthly_avg = spot_df.groupby(spot_df.index.month)[price_col].mean()
    monthly_avg = monthly_avg.reindex(range(1, 13))
    shape = monthly_avg / monthly_avg.mean()
    shape.index.name = "month"
    return shape


def historical_shape_forward_curve(
    spot_df: pd.DataFrame,
    valuation_date,
    horizon_months: int = 36,
    anchor_window_days: int = 30,
    price_col: str = "power_price",
) -> pd.DataFrame:
    """Bootstrap a monthly forward curve from a historical/simulated shape.

    The curve is built as ``anchor_price * seasonal_shape[month]``, where
    ``anchor_price`` is the trailing ``anchor_window_days`` average spot
    price as of ``valuation_date`` (a proxy for "current market
    information") and ``seasonal_shape`` is the normalized average monthly
    price pattern estimated from ``spot_df``.

    Parameters
    ----------
    spot_df:
        DataFrame with a DatetimeIndex and a price column, e.g. as produced
        by ``PriceScenario.to_frame``.
    valuation_date:
        Date as of which the curve is built. Must be within ``spot_df``'s
        index range.
    horizon_months:
        Number of forward monthly buckets to construct.
    anchor_window_days:
        Trailing window used to estimate the current price level.

    Returns
    -------
    DataFrame indexed by delivery month (Timestamp, month start) with a
    single column ``forward_price``.
    """
    valuation_date = pd.Timestamp(valuation_date)
    shape = compute_monthly_seasonal_shape(spot_df, price_col=price_col)

    history = spot_df.loc[:valuation_date]
    if history.empty:
        raise ValueError("valuation_date is before the start of spot_df")
    anchor_price = history[price_col].tail(anchor_window_days).mean()

    delivery_months = pd.date_range(
        start=valuation_date.to_period("M").to_timestamp(), periods=horizon_months + 1, freq="MS"
    )[1:]
    forward_prices = [anchor_price * shape.loc[m.month] for m in delivery_months]

    return pd.DataFrame({"delivery_month": delivery_months, "forward_price": forward_prices}).set_index(
        "delivery_month"
    )


def _expected_gas_price_path(
    t_years: np.ndarray, gas_params: GasPriceParams, x0: float
) -> np.ndarray:
    """Closed-form E[G_t] using the lognormal OU transition (Jensen correction)."""
    kappa = gas_params.mean_reversion_speed
    doy = 1 + (t_years * DAYS_PER_YEAR) % DAYS_PER_YEAR
    g_bar_t = gas_seasonal_log_mean(doy, gas_params)
    mean_x = x0 * np.exp(-kappa * t_years)
    var_x = (gas_params.volatility**2) / (2 * kappa) * (1 - np.exp(-2 * kappa * t_years))
    return np.exp(g_bar_t + mean_x + 0.5 * var_x)


def _expected_margin_path(
    t_years: np.ndarray, power_params: PowerPriceParams, y0: float, dt_fine: float = 1.0 / DAYS_PER_YEAR
) -> np.ndarray:
    """Semi-analytic E[Y_t] via discretized variation-of-constants.

    dY = -kappa*Y*dt + dJ (deterministic ODE for the expectation, since
    E[dW] = 0); the jump term contributes its mean intensity * mean size,
    decaying via the same mean-reversion kernel from the time it occurred.
    """
    kappa = power_params.mean_reversion_speed
    max_t = t_years.max()
    n_fine = max(int(np.ceil(max_t / dt_fine)) + 1, 2)
    fine_t = np.linspace(0.0, max_t, n_fine)
    fine_dt = fine_t[1] - fine_t[0]
    doy_fine = 1 + (fine_t * DAYS_PER_YEAR) % DAYS_PER_YEAR
    lam_fine = jump_intensity_per_year(doy_fine, power_params)

    expected_y = np.empty(n_fine)
    y = y0
    expected_y[0] = y
    for i in range(1, n_fine):
        jump_contrib = lam_fine[i - 1] * fine_dt * power_params.jump_mean
        y = y * np.exp(-kappa * fine_dt) + jump_contrib
        expected_y[i] = y

    return np.interp(t_years, fine_t, expected_y)


def analytic_forward_curve(
    gas_params: GasPriceParams,
    power_params: PowerPriceParams,
    current_gas_price: float,
    current_margin_state: float,
    valuation_date,
    horizon_months: int = 36,
) -> pd.DataFrame:
    """Model-implied forward curve via closed-form / semi-analytic expectations.

    Parameters
    ----------
    current_gas_price:
        Spot gas price as of ``valuation_date``.
    current_margin_state:
        Current value of the margin OU state ``Y_0`` (i.e. deviation of the
        margin from its seasonal mean) as of ``valuation_date``.
    """
    valuation_date = pd.Timestamp(valuation_date)
    delivery_months = pd.date_range(
        start=valuation_date.to_period("M").to_timestamp(), periods=horizon_months + 1, freq="MS"
    )[1:]
    # value at the mid-point of each delivery month
    mid_dates = delivery_months + pd.Timedelta(days=15)
    t_years = (mid_dates - valuation_date).days.to_numpy() / DAYS_PER_YEAR

    doy = mid_dates.dayofyear.to_numpy(dtype=float)
    x0 = np.log(current_gas_price) - gas_seasonal_log_mean(np.array([1.0]), gas_params)[0]
    expected_gas = _expected_gas_price_path(t_years, gas_params, x0)
    expected_margin_state = _expected_margin_path(t_years, power_params, current_margin_state)
    m_bar = power_margin_seasonal_mean(doy, power_params)
    expected_power = power_params.heat_rate * expected_gas + (m_bar + expected_margin_state)
    expected_power = np.clip(expected_power, power_params.price_floor, None)

    return pd.DataFrame(
        {
            "delivery_month": delivery_months,
            "forward_price": expected_power,
            "expected_gas_price": expected_gas,
        }
    ).set_index("delivery_month")


def compare_curves(curve_a: pd.DataFrame, curve_b: pd.DataFrame) -> pd.DataFrame:
    """Align two forward curves by delivery month and report their spread."""
    merged = curve_a[["forward_price"]].join(
        curve_b[["forward_price"]], how="inner", lsuffix="_a", rsuffix="_b"
    )
    merged["abs_diff"] = merged["forward_price_a"] - merged["forward_price_b"]
    merged["pct_diff"] = merged["abs_diff"] / merged["forward_price_b"]
    return merged
