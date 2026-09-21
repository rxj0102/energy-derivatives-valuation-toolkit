"""Pricing of standard energy derivatives: forwards/swaps, European options
on power price, and spark-spread options.

Closed-form pricers are cross-checked against Monte Carlo engines built on
the same distributional assumptions:

* ``black76_price`` assumes the forward price follows a driftless
  lognormal diffusion under the pricing measure (the standard commodity
  options convention). Its Monte Carlo cross-check (``monte_carlo_black76``)
  simulates that *same* lognormal-forward assumption and should converge to
  the closed-form price as the path count grows.

* ``margrabe_price`` (and ``kirk_spread_approx``) assume both legs of a
  spread are correlated lognormal forwards. Their Monte Carlo cross-checks
  simulate correlated lognormal terminal prices.

* ``mc_price_option_on_scenario`` / ``mc_spread_option_on_scenario`` instead
  price options against the *full* mean-reverting jump-diffusion spot model
  from ``power_price_model``. These will generally NOT match Black-76 /
  Margrabe, because that underlying process is not lognormal — that
  divergence is expected and is itself informative: it quantifies the model
  risk of applying a vanilla Black-76 formula to a spiky, mean-reverting
  power price.

DISCLAIMER: synthetic data / educational purposes only, see
``power_price_model.py``.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


# --------------------------------------------------------------------------
# Forwards / swaps
# --------------------------------------------------------------------------


def forward_contract_value(forward_price: float, fixed_price: float, notional_mwh: float, r: float, T: float) -> float:
    """PV of a forward/swap position that pays (forward_price - fixed_price) per MWh at T.

    Positive for a long position (receive floating/forward reference,
    pay fixed) when the market forward price exceeds the contracted fixed
    price.
    """
    payoff = (forward_price - fixed_price) * notional_mwh
    return payoff * np.exp(-r * T)


def swap_mark_to_market(fixed_price: float, forward_curve_prices: np.ndarray, notional_mwh: float, r: float, tenors: np.ndarray) -> float:
    """PV of a strip of monthly swap settlements against a forward curve."""
    payoffs = (np.asarray(forward_curve_prices) - fixed_price) * notional_mwh
    discount = np.exp(-r * np.asarray(tenors))
    return float(np.sum(payoffs * discount))


# --------------------------------------------------------------------------
# Black-76
# --------------------------------------------------------------------------


def _d1_d2(F: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black76_price(F: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call") -> float:
    """Black-76 price for a European option on a forward price F.

    Assumes F follows dF/F = sigma dW under the pricing measure (driftless
    lognormal), the standard convention for commodity/energy options priced
    off a forward curve.
    """
    d1, d2 = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    if option_type == "call":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    raise ValueError(f"Unknown option_type: {option_type}")


def black76_greeks(F: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call") -> dict:
    """Delta, gamma, vega, theta for a Black-76 option (w.r.t. the forward)."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        delta = disc * norm.cdf(d1)
    else:
        delta = -disc * norm.cdf(-d1)
    gamma = disc * pdf_d1 / (F * sigma * np.sqrt(T))
    vega = disc * F * pdf_d1 * np.sqrt(T)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


def put_call_parity_residual(call_price: float, put_price: float, F: float, K: float, T: float, r: float) -> float:
    """C - P - disc*(F - K); should be ~0 for consistent Black-76 prices."""
    disc = np.exp(-r * T)
    return call_price - put_price - disc * (F - K)


# --------------------------------------------------------------------------
# Monte Carlo: lognormal-forward cross-check for Black-76
# --------------------------------------------------------------------------


def simulate_terminal_lognormal(F0: float, sigma: float, T: float, n_paths: int, seed: Optional[int] = None) -> np.ndarray:
    """Terminal forward prices under dF/F = sigma dW (Black-76 assumption)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    return F0 * np.exp(-0.5 * sigma**2 * T + sigma * np.sqrt(T) * z)


def monte_carlo_option_price(
    terminal_prices: np.ndarray, K: float, T: float, r: float, option_type: OptionType = "call"
) -> Tuple[float, float]:
    """Discounted MC price + standard error from an array of terminal prices."""
    if option_type == "call":
        payoff = np.maximum(terminal_prices - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - terminal_prices, 0.0)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")
    disc_payoff = np.exp(-r * T) * payoff
    price = float(disc_payoff.mean())
    std_error = float(disc_payoff.std(ddof=1) / np.sqrt(len(disc_payoff)))
    return price, std_error


def monte_carlo_black76(
    F0: float, K: float, T: float, r: float, sigma: float, n_paths: int = 100_000,
    option_type: OptionType = "call", seed: Optional[int] = None, antithetic: bool = True,
) -> Tuple[float, float]:
    """Monte Carlo cross-check of Black-76 under the same lognormal-forward assumption."""
    if antithetic:
        rng = np.random.default_rng(seed)
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
        terminal = F0 * np.exp(-0.5 * sigma**2 * T + sigma * np.sqrt(T) * z)
    else:
        terminal = simulate_terminal_lognormal(F0, sigma, T, n_paths, seed)
    return monte_carlo_option_price(terminal, K, T, r, option_type)


def mc_price_option_on_scenario(
    price_paths: np.ndarray, K: float, T: float, r: float, time_index: int, option_type: OptionType = "call"
) -> Tuple[float, float]:
    """Price a European option using terminal prices from the full spot-model
    Monte Carlo scenario (``PriceScenario.power_prices``), at column ``time_index``.

    This deliberately does NOT assume lognormality; it uses the empirical
    distribution of the simulated mean-reverting jump-diffusion process.
    """
    terminal_prices = price_paths[:, time_index]
    return monte_carlo_option_price(terminal_prices, K, T, r, option_type)


# --------------------------------------------------------------------------
# Spread options: Margrabe (K=0), Kirk's approximation (K != 0), Monte Carlo
# --------------------------------------------------------------------------


def estimate_lognormal_vol(price_path: np.ndarray, dt_years: float) -> float:
    """Annualized volatility of log returns, for use as a Black-76-style sigma input."""
    log_returns = np.diff(np.log(np.clip(price_path, 1e-6, None)))
    return float(np.std(log_returns, ddof=1) / np.sqrt(dt_years))


def margrabe_price(F1: float, F2: float, sigma1: float, sigma2: float, rho: float, T: float, r: float) -> float:
    """Margrabe (1978) closed-form price of a European exchange option:
    payoff = max(S1_T - S2_T, 0), i.e. a spread option with strike K=0.
    """
    sigma_hat = np.sqrt(max(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2, 0.0))
    if sigma_hat <= 0 or T <= 0:
        return max(F1 - F2, 0.0) * np.exp(-r * T)
    d1 = (np.log(F1 / F2) + 0.5 * sigma_hat**2 * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)
    disc = np.exp(-r * T)
    return disc * (F1 * norm.cdf(d1) - F2 * norm.cdf(d2))


def kirk_spread_approx(
    F1: float, F2: float, sigma1: float, sigma2: float, rho: float, K: float, T: float, r: float,
    option_type: OptionType = "call",
) -> float:
    """Kirk's (1995) approximation for a spread call/put with non-zero strike:
    payoff = max(S1_T - S2_T - K, 0) [or the put analogue].

    Reduces to Margrabe as K -> 0.
    """
    denom = F2 + K
    if denom <= 0:
        # Degenerate case: strike so negative the "effective F2" is non-positive;
        # fall back to intrinsic-style bound.
        return max(F1 - F2 - K, 0.0) * np.exp(-r * T) if option_type == "call" else max(
            F2 + K - F1, 0.0
        ) * np.exp(-r * T)
    F2_eff = denom
    sigma2_eff = sigma2 * F2 / denom
    sigma_hat = np.sqrt(max(sigma1**2 + sigma2_eff**2 - 2 * rho * sigma1 * sigma2_eff, 0.0))
    if sigma_hat <= 0 or T <= 0:
        intrinsic = max(F1 - F2 - K, 0.0) if option_type == "call" else max(F2 + K - F1, 0.0)
        return intrinsic * np.exp(-r * T)
    d1 = (np.log(F1 / F2_eff) + 0.5 * sigma_hat**2 * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)
    disc = np.exp(-r * T)
    if option_type == "call":
        return disc * (F1 * norm.cdf(d1) - F2_eff * norm.cdf(d2))
    return disc * (F2_eff * norm.cdf(-d2) - F1 * norm.cdf(-d1))


def simulate_correlated_terminal_lognormals(
    F1_0: float, F2_0: float, sigma1: float, sigma2: float, rho: float, T: float, n_paths: int, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z_indep = rng.standard_normal(n_paths)
    z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * z_indep
    S1 = F1_0 * np.exp(-0.5 * sigma1**2 * T + sigma1 * np.sqrt(T) * z1)
    S2 = F2_0 * np.exp(-0.5 * sigma2**2 * T + sigma2 * np.sqrt(T) * z2)
    return S1, S2


def mc_spread_option_lognormal(
    F1_0: float, F2_0: float, sigma1: float, sigma2: float, rho: float, K: float, T: float, r: float,
    n_paths: int = 100_000, option_type: OptionType = "call", seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Monte Carlo price of a spread option under correlated lognormal legs
    (for cross-checking Margrabe / Kirk).
    """
    S1, S2 = simulate_correlated_terminal_lognormals(F1_0, F2_0, sigma1, sigma2, rho, T, n_paths, seed)
    spread = S1 - S2
    if option_type == "call":
        payoff = np.maximum(spread - K, 0.0)
    else:
        payoff = np.maximum(K - spread, 0.0)
    disc_payoff = np.exp(-r * T) * payoff
    price = float(disc_payoff.mean())
    se = float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))
    return price, se


def mc_spark_spread_option_on_scenario(
    power_paths: np.ndarray,
    gas_paths: np.ndarray,
    heat_rate: float,
    K: float,
    T: float,
    r: float,
    time_index: int,
    notional_mwh: float = 1.0,
    option_type: OptionType = "call",
) -> Tuple[float, float]:
    """Price a spark-spread option (payoff = max(power - heat_rate*gas - K, 0))
    at ``time_index`` using the full jump-diffusion spot model paths.
    """
    power = power_paths[:, time_index]
    gas = gas_paths[:, time_index]
    spread = power - heat_rate * gas
    if option_type == "call":
        payoff = np.maximum(spread - K, 0.0)
    else:
        payoff = np.maximum(K - spread, 0.0)
    disc_payoff = np.exp(-r * T) * payoff * notional_mwh
    price = float(disc_payoff.mean())
    se = float(disc_payoff.std(ddof=1) / np.sqrt(disc_payoff.shape[0]))
    return price, se
