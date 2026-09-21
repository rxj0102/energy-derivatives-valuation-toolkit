"""Valuation of a synthetic gas-fired power plant operated as a tolling
agreement.

A tolling agreement gives its holder the *right* (not obligation) to run a
generator, converting fuel into power, each day, in exchange for paying the
plant's variable operating cost. Economically this is a strip of daily
spark-spread call options:

    daily payoff = max(power_price - heat_rate * gas_price - variable_om, 0) * capacity_mw * hours

If the plant could be started and stopped freely with no restrictions, its
value would just be the sum of those daily option payoffs (the
"unconstrained intrinsic value", computed here as a baseline). Real plants
cannot do that: there is a limited number of times per month the unit can be
started (``max_starts_per_month``, reflecting wear-and-tear / maintenance
economics) and, once started, it must run for a minimum number of days
before it can shut down again (``min_run_days``, a simple stand-in for
ramp / minimum-up-time constraints).

Optimal dispatch under those constraints is solved exactly, month by month,
with a small dynamic program: choose up to ``max_starts_per_month``
non-overlapping ``min_run_days``-length "runs" within the month to maximize
total discounted spread revenue net of start costs. This is the "optimal
dispatch decision simulated each day" real-options valuation: it is run
independently along each Monte Carlo price path, and the plant's value is
the discounted expected value across paths.

DISCLAIMER: synthetic data / educational purposes only, see
``power_price_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .power_price_model import PriceScenario


@dataclass
class TollingAgreementSpec:
    """Operating characteristics of the tolled gas plant."""

    capacity_mw: float = 100.0
    heat_rate: float = 7.5  # MMBtu/MWh
    variable_om_cost: float = 2.0  # $/MWh
    max_starts_per_month: int = 8
    min_run_days: int = 1  # simplified ramp / minimum-up-time constraint
    start_cost: float = 5_000.0  # $ per start, covering wear/maintenance economics
    hours_per_day: float = 24.0


def daily_spark_spread(power_prices: np.ndarray, gas_prices: np.ndarray, heat_rate: float, variable_om_cost: float) -> np.ndarray:
    """Daily spark spread ($/MWh): power - heat_rate*gas - variable O&M."""
    return power_prices - heat_rate * gas_prices - variable_om_cost


def _discounted_window_sums(
    discounted_spread_dollars: np.ndarray, min_run_days: int, start_cost_discounted: np.ndarray
) -> np.ndarray:
    """Vectorized computation of the discounted net value of a run starting on
    each day of a month (length ``min_run_days``), via a prefix-sum
    convolution. Returns an array of length ``n - min_run_days + 1``
    (or shorter, padded with -inf where a full run does not fit).
    """
    n = discounted_spread_dollars.shape[0]
    prefix = np.concatenate(([0.0], np.cumsum(discounted_spread_dollars)))
    n_valid = n - min_run_days + 1
    window_sums = np.full(n, -np.inf)
    if n_valid > 0:
        window_sums[:n_valid] = prefix[min_run_days : min_run_days + n_valid] - prefix[:n_valid]
        window_sums[:n_valid] -= start_cost_discounted[:n_valid]
    return window_sums


def optimal_monthly_dispatch(
    spread: np.ndarray,
    disc_factors: np.ndarray,
    capacity_mw: float,
    hours_per_day: float,
    max_starts: int,
    min_run_days: int,
    start_cost: float,
) -> Tuple[np.ndarray, float]:
    """Exact optimal dispatch for a single calendar month via dynamic programming.

    Chooses up to ``max_starts`` non-overlapping runs of length
    ``min_run_days`` (each run must be dispatched in full once started) to
    maximize total discounted net revenue.

    Returns
    -------
    (dispatch_mask, value): boolean array marking dispatched days, and the
    total discounted value of the optimal policy for this month.
    """
    n = spread.shape[0]
    dollar_spread = spread * capacity_mw * hours_per_day * disc_factors
    start_cost_discounted = start_cost * disc_factors
    window_value = _discounted_window_sums(dollar_spread, min_run_days, start_cost_discounted)

    K = max_starts
    # dp[d, k] = best achievable value from day d onward with k starts remaining
    dp = np.zeros((n + 1, K + 1))
    for d in range(n - 1, -1, -1):
        dp[d, 0] = dp[d + 1, 0]
        for k in range(1, K + 1):
            skip_value = dp[d + 1, k]
            if d + min_run_days <= n and np.isfinite(window_value[d]):
                start_value = window_value[d] + dp[d + min_run_days, k - 1]
            else:
                start_value = -np.inf
            dp[d, k] = max(skip_value, start_value)

    # forward reconstruction
    mask = np.zeros(n, dtype=bool)
    d, k = 0, K
    while d < n and k > 0:
        skip_value = dp[d + 1, k]
        start_value = (
            window_value[d] + dp[d + min_run_days, k - 1]
            if d + min_run_days <= n and np.isfinite(window_value[d])
            else -np.inf
        )
        if start_value >= skip_value and start_value > -np.inf:
            mask[d : d + min_run_days] = True
            d += min_run_days
            k -= 1
        else:
            d += 1

    return mask, float(dp[0, K])


def value_tolling_agreement(
    scenario: PriceScenario, spec: TollingAgreementSpec, r: float
) -> dict:
    """Monte Carlo value of the tolling agreement across all paths in ``scenario``.

    For each path, spread is computed daily, split by calendar month, and
    the exact optimal-dispatch DP is applied within each month. Path values
    are then averaged (with a standard error) to give the constrained
    agreement value, alongside an unconstrained baseline (sum of all
    positive daily spreads, i.e. a plain strip of daily spark-spread calls
    with no start/ramp restrictions) for comparison.
    """
    n_paths = scenario.n_paths
    dt = scenario.dt_years
    t_years = np.arange(scenario.n_steps) * dt
    disc_factors = np.exp(-r * t_years)

    periods = pd.PeriodIndex(scenario.dates, freq="M")
    month_boundaries = []
    start = 0
    for i in range(1, len(periods) + 1):
        if i == len(periods) or periods[i] != periods[start]:
            month_boundaries.append((start, i))
            start = i

    constrained_values = np.zeros(n_paths)
    unconstrained_values = np.zeros(n_paths)
    n_starts_used = np.zeros(n_paths)

    for p in range(n_paths):
        spread = daily_spark_spread(
            scenario.power_prices[p], scenario.gas_prices[p], spec.heat_rate, spec.variable_om_cost
        )
        unconstrained_values[p] = np.sum(
            np.maximum(spread, 0.0) * spec.capacity_mw * spec.hours_per_day * disc_factors
        )

        total_value = 0.0
        starts = 0
        for (m_start, m_end) in month_boundaries:
            mask, month_value = optimal_monthly_dispatch(
                spread[m_start:m_end],
                disc_factors[m_start:m_end],
                spec.capacity_mw,
                spec.hours_per_day,
                spec.max_starts_per_month,
                spec.min_run_days,
                spec.start_cost,
            )
            total_value += month_value
            # count starts: transitions from False->True within the mask
            starts += int(np.sum((mask.astype(int) - np.concatenate(([0], mask[:-1].astype(int)))) == 1))
        constrained_values[p] = total_value
        n_starts_used[p] = starts

    mean_value = float(constrained_values.mean())
    std_error = float(constrained_values.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0

    return {
        "mean_value": mean_value,
        "std_error": std_error,
        "path_values": constrained_values,
        "unconstrained_intrinsic_value": float(unconstrained_values.mean()),
        "mean_starts_per_path": float(n_starts_used.mean()),
        "n_months": len(month_boundaries),
    }
