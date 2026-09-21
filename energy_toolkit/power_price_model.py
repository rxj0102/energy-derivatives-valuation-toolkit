"""Synthetic wholesale power and natural gas price simulation.

DISCLAIMER
----------
Every price series produced by this module is entirely synthetic. Parameters
are chosen to be *qualitatively* realistic (strong seasonality, mean
reversion, summer-heavy spike risk, a plausible power/gas relationship) but
are NOT calibrated to any real ISO, hub, or vendor data feed. Nothing here
should be used for trading, hedging, or financial decision-making.

Modeling approach
------------------
Natural gas (e.g. a Henry-Hub-like index) is modeled as a seasonal,
mean-reverting diffusion in log-price (an Ornstein-Uhlenbeck process applied
to the deviation of log-price from a seasonal mean).

Wholesale power price is decomposed as

    P_t = HeatRate * G_t + M_t

where ``G_t`` is the simulated gas price and ``M_t`` is a "scarcity margin"
(the premium the marginal price-setting generator earns above its fuel
cost, covering variable O&M, imperfect competition, transmission
congestion, and weather-driven demand spikes). This mirrors how power
desks actually think about spot formation, and it gives the model a
built-in, economically coherent link between power and gas that the
spark-spread pricing modules downstream rely on.

The margin itself is a seasonal mean-reverting jump-diffusion:

    M_t   = m_bar(t) + Y_t
    dY_t  = -kappa * Y_t * dt + sigma * dW_t + dJ_t

``m_bar(t)`` has a summer (AC-load) hump and a smaller winter (heating)
hump. ``dJ_t`` is a compound Poisson jump process whose intensity is
elevated in summer months, representing heat-wave / generator-outage spike
risk. Jump sizes are drawn i.i.d. Normal and are added directly to the
margin (i.e. jumps are additive in $/MWh space, not multiplicative), which
is the standard way to reproduce the large, mostly-upward $/MWh spikes seen
in real power markets without generating absurd multiplicative blow-ups.

Gas and power shocks are correlated (``price_gas_correlation``) since fuel
cost surprises and demand surprises are not independent in reality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


# --------------------------------------------------------------------------
# Parameter containers
# --------------------------------------------------------------------------


@dataclass
class GasPriceParams:
    """Parameters for the natural gas (Henry-Hub-like) price process.

    Modeled as an Ornstein-Uhlenbeck process on the log-price deviation from
    a seasonal log-mean (i.e. a seasonal mean-reverting GBM-like process).
    """

    initial_price: float = 3.50  # $/MMBtu
    long_run_mean: float = 3.50  # $/MMBtu
    mean_reversion_speed: float = 4.0  # kappa, annualized
    volatility: float = 0.55  # annualized vol of the log-deviation process
    seasonal_amplitude: float = 0.18  # fractional seasonal swing in log-price
    seasonal_phase_day: float = 15.0  # day-of-year of seasonal peak (winter heating)


@dataclass
class PowerPriceParams:
    """Parameters for the wholesale power price process.

    Power price = heat_rate * gas_price + scarcity margin, where the margin
    is a seasonal mean-reverting jump-diffusion process.
    """

    heat_rate: float = 7.5  # MMBtu/MWh of the marginal (price-setting) unit
    margin_base: float = 8.0  # $/MWh long-run average scarcity margin
    margin_summer_amplitude: float = 6.0  # $/MWh
    margin_summer_phase_day: float = 200.0  # ~ mid/late July
    margin_winter_amplitude: float = 3.0  # $/MWh
    margin_winter_phase_day: float = 15.0  # mid January
    mean_reversion_speed: float = 45.0  # kappa, annualized (power reverts fast, hours-days)
    volatility: float = 0.95  # annualized vol of the margin OU process
    jump_base_intensity: float = 6.0  # expected jumps/year outside summer
    jump_summer_intensity_multiplier: float = 6.0  # extra jump intensity in summer
    jump_summer_months: Tuple[int, ...] = (6, 7, 8)
    jump_mean: float = 60.0  # $/MWh, mean jump size (upward spike)
    jump_std: float = 45.0  # $/MWh
    price_gas_correlation: float = 0.55  # correlation between gas and margin shocks
    price_floor: float = 0.01  # $/MWh floor to keep prices strictly positive


@dataclass
class PriceScenario:
    """Container for one or many jointly-simulated gas/power price paths.

    All arrays share shape ``(n_paths, n_steps)`` and are aligned to
    ``dates`` (length ``n_steps``). This is the object passed between
    modules in this toolkit: forward curve construction, derivatives
    pricing, asset valuation, and risk reporting all consume it.
    """

    dates: pd.DatetimeIndex
    gas_prices: np.ndarray
    power_prices: np.ndarray
    jump_flags: np.ndarray
    dt_years: float
    gas_params: GasPriceParams
    power_params: PowerPriceParams

    @property
    def n_paths(self) -> int:
        return self.power_prices.shape[0]

    @property
    def n_steps(self) -> int:
        return self.power_prices.shape[1]

    def to_frame(self, path_index: int = 0) -> pd.DataFrame:
        """Return a single path as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "date": self.dates,
                "gas_price": self.gas_prices[path_index],
                "power_price": self.power_prices[path_index],
                "is_jump": self.jump_flags[path_index],
            }
        ).set_index("date")


# --------------------------------------------------------------------------
# Seasonal shape helpers
# --------------------------------------------------------------------------


def seasonal_component(
    day_of_year: np.ndarray, amplitude: float, phase_day: float, period_days: float = DAYS_PER_YEAR
) -> np.ndarray:
    """Cosine seasonal shape peaking at ``phase_day``."""
    return amplitude * np.cos(2.0 * np.pi * (day_of_year - phase_day) / period_days)


def gas_seasonal_log_mean(day_of_year: np.ndarray, params: GasPriceParams) -> np.ndarray:
    return np.log(params.long_run_mean) + seasonal_component(
        day_of_year, params.seasonal_amplitude, params.seasonal_phase_day
    )


def power_margin_seasonal_mean(day_of_year: np.ndarray, params: PowerPriceParams) -> np.ndarray:
    summer = seasonal_component(day_of_year, params.margin_summer_amplitude, params.margin_summer_phase_day)
    winter = seasonal_component(day_of_year, params.margin_winter_amplitude, params.margin_winter_phase_day)
    return params.margin_base + summer + winter


def jump_intensity_per_year(day_of_year: np.ndarray, params: PowerPriceParams) -> np.ndarray:
    """Annualized Poisson jump intensity, elevated in summer months."""
    months = pd.DatetimeIndex(pd.Timestamp("2001-01-01") + pd.to_timedelta(day_of_year - 1, unit="D"))
    is_summer = np.isin(np.asarray(months.month), params.jump_summer_months)
    return np.where(
        is_summer,
        params.jump_base_intensity * params.jump_summer_intensity_multiplier,
        params.jump_base_intensity,
    )


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


class PowerMarketSimulator:
    """Jointly simulates correlated natural gas and wholesale power prices."""

    def __init__(
        self,
        gas_params: Optional[GasPriceParams] = None,
        power_params: Optional[PowerPriceParams] = None,
    ) -> None:
        self.gas_params = gas_params or GasPriceParams()
        self.power_params = power_params or PowerPriceParams()

    def simulate_paths(
        self,
        n_years: float = 5.0,
        n_paths: int = 1,
        dt_days: float = 1.0,
        seed: Optional[int] = None,
        start_date: str = "2021-01-01",
    ) -> PriceScenario:
        """Vectorized simulation of ``n_paths`` correlated gas/power paths.

        Uses the exact (non-Euler) transition density of the OU process at
        each step, which is stable and unbiased regardless of step size.
        """
        rng = np.random.default_rng(seed)
        gp, pp = self.gas_params, self.power_params

        n_steps = int(round(n_years * DAYS_PER_YEAR / dt_days))
        dates = pd.date_range(start=start_date, periods=n_steps, freq=f"{int(dt_days)}D")
        day_of_year = dates.dayofyear.to_numpy(dtype=float)
        dt = dt_days / DAYS_PER_YEAR

        g_bar = gas_seasonal_log_mean(day_of_year, gp)
        m_bar = power_margin_seasonal_mean(day_of_year, pp)
        lam = jump_intensity_per_year(day_of_year, pp)

        # OU exact-transition coefficients (constant across steps for fixed dt)
        gas_decay = np.exp(-gp.mean_reversion_speed * dt)
        gas_shock_scale = gp.volatility * np.sqrt(
            (1.0 - np.exp(-2.0 * gp.mean_reversion_speed * dt)) / (2.0 * gp.mean_reversion_speed)
        )
        pow_decay = np.exp(-pp.mean_reversion_speed * dt)
        pow_shock_scale = pp.volatility * np.sqrt(
            (1.0 - np.exp(-2.0 * pp.mean_reversion_speed * dt)) / (2.0 * pp.mean_reversion_speed)
        )
        rho = pp.price_gas_correlation

        x_gas = np.full(n_paths, np.log(gp.initial_price) - g_bar[0])
        y_pow = np.zeros(n_paths)

        gas_prices = np.empty((n_paths, n_steps))
        power_prices = np.empty((n_paths, n_steps))
        jump_flags = np.zeros((n_paths, n_steps), dtype=bool)

        gas_prices[:, 0] = np.exp(g_bar[0] + x_gas)
        power_prices[:, 0] = np.clip(pp.heat_rate * gas_prices[:, 0] + (m_bar[0] + y_pow), pp.price_floor, None)

        for t in range(1, n_steps):
            z_gas = rng.standard_normal(n_paths)
            z_indep = rng.standard_normal(n_paths)
            z_pow = rho * z_gas + np.sqrt(max(1.0 - rho**2, 0.0)) * z_indep

            x_gas = x_gas * gas_decay + gas_shock_scale * z_gas
            y_pow = y_pow * pow_decay + pow_shock_scale * z_pow

            n_jumps = rng.poisson(lam[t] * dt, size=n_paths)
            has_jump = n_jumps > 0
            if has_jump.any():
                jump_noise = rng.standard_normal(n_paths)
                jump_total = np.where(
                    has_jump,
                    n_jumps * pp.jump_mean + np.sqrt(n_jumps.astype(float)) * pp.jump_std * jump_noise,
                    0.0,
                )
                y_pow = y_pow + jump_total
                jump_flags[:, t] = has_jump

            gas_prices[:, t] = np.exp(g_bar[t] + x_gas)
            power_prices[:, t] = np.clip(pp.heat_rate * gas_prices[:, t] + (m_bar[t] + y_pow), pp.price_floor, None)

        return PriceScenario(
            dates=dates,
            gas_prices=gas_prices,
            power_prices=power_prices,
            jump_flags=jump_flags,
            dt_years=dt,
            gas_params=gp,
            power_params=pp,
        )

    def simulate(
        self,
        n_years: float = 5.0,
        dt_days: float = 1.0,
        seed: Optional[int] = None,
        start_date: str = "2021-01-01",
    ) -> pd.DataFrame:
        """Convenience wrapper: simulate a single path as a tidy DataFrame."""
        scenario = self.simulate_paths(
            n_years=n_years, n_paths=1, dt_days=dt_days, seed=seed, start_date=start_date
        )
        return scenario.to_frame(0)


def simulate_diurnal_hourly_prices(
    daily_prices: np.ndarray,
    peak_multiplier: float = 1.35,
    off_peak_multiplier: float = 0.75,
    noise_std_frac: float = 0.03,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Expand a daily price series into a synthetic 24-hour intraday shape.

    A duck-curve-like diurnal multiplier (low overnight, a mild midday dip
    for solar, an evening peak) is applied to each day's price, scaled so
    that the *average* of the 24 hourly prices for a day equals that day's
    input price. Small idiosyncratic hourly noise is added. This is a
    simplifying device to give the battery storage module intraday price
    variation to arbitrage, since the core model above simulates at daily
    resolution; it is not itself a calibrated stochastic hourly model.

    Parameters
    ----------
    daily_prices:
        1-D array of daily prices, shape ``(n_days,)``.

    Returns
    -------
    np.ndarray of shape ``(n_days * 24,)``.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    # Duck-curve shape: dip mid-day, peak in the evening, low overnight.
    shape = (
        1.0
        + 0.15 * np.cos(2 * np.pi * (hours - 3) / 24)
        - 0.30 * np.exp(-((hours - 13) ** 2) / (2 * 3.0**2))
        + (peak_multiplier - 1.0) * np.exp(-((hours - 19) ** 2) / (2 * 2.0**2))
        - (1.0 - off_peak_multiplier) * np.exp(-((hours - 4) ** 2) / (2 * 3.0**2))
    )
    shape = shape / shape.mean()

    n_days = daily_prices.shape[0]
    noise = 1.0 + noise_std_frac * rng.standard_normal((n_days, 24))
    hourly = daily_prices[:, None] * shape[None, :] * noise
    hourly = hourly * (daily_prices[:, None] / hourly.mean(axis=1, keepdims=True))
    return np.clip(hourly.reshape(-1), 1e-3, None)
