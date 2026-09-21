import numpy as np

from energy_toolkit.derivatives_pricing import (
    black76_price,
    kirk_spread_approx,
    margrabe_price,
    mc_spark_spread_option_on_scenario,
    mc_spread_option_lognormal,
    monte_carlo_black76,
    put_call_parity_residual,
)


F, K, T, r, sigma = 50.0, 55.0, 1.0, 0.03, 0.6


def test_black76_put_call_parity_holds_exactly():
    call = black76_price(F, K, T, r, sigma, "call")
    put = black76_price(F, K, T, r, sigma, "put")
    residual = put_call_parity_residual(call, put, F, K, T, r)
    assert abs(residual) < 1e-8


def test_monte_carlo_black76_converges_call():
    closed_form = black76_price(F, K, T, r, sigma, "call")
    mc_price, se = monte_carlo_black76(F, K, T, r, sigma, n_paths=500_000, option_type="call", seed=1)
    assert abs(mc_price - closed_form) < 4 * se


def test_monte_carlo_black76_converges_put():
    closed_form = black76_price(F, K, T, r, sigma, "put")
    mc_price, se = monte_carlo_black76(F, K, T, r, sigma, n_paths=500_000, option_type="put", seed=2)
    assert abs(mc_price - closed_form) < 4 * se


def test_margrabe_reduces_to_black76_when_second_leg_is_deterministic():
    # If sigma2 -> 0, exchanging S1 for a (nearly) deterministic F2 is
    # equivalent to a Black-76 call on S1 struck at F2.
    F1, F2_, sigma1 = 60.0, 40.0, 0.5
    margrabe = margrabe_price(F1, F2_, sigma1, 1e-6, rho=0.0, T=T, r=r)
    black_call = black76_price(F1, F2_, T, r, sigma1, "call")
    assert abs(margrabe - black_call) < 1e-3


def test_margrabe_matches_monte_carlo():
    F1, F2_, s1, s2, rho = 60.0, 6.5, 0.8, 0.4, 0.5
    closed_form = margrabe_price(F1, F2_, s1, s2, rho, T, r)
    mc_price, se = mc_spread_option_lognormal(F1, F2_, s1, s2, rho, K=0.0, T=T, r=r, n_paths=500_000, seed=4)
    assert abs(mc_price - closed_form) < 4 * se


def test_kirk_reduces_to_margrabe_at_zero_strike():
    F1, F2_, s1, s2, rho = 60.0, 6.5, 0.8, 0.4, 0.3
    kirk = kirk_spread_approx(F1, F2_, s1, s2, rho, K=0.0, T=T, r=r)
    margrabe = margrabe_price(F1, F2_, s1, s2, rho, T, r)
    assert abs(kirk - margrabe) < 1e-8


def test_kirk_approximation_close_to_monte_carlo_nonzero_strike():
    F1, F2_, s1, s2, rho = 60.0, 6.5, 0.8, 0.4, 0.3
    K_strike = 5.0
    kirk = kirk_spread_approx(F1, F2_, s1, s2, rho, K_strike, T, r)
    mc_price, se = mc_spread_option_lognormal(F1, F2_, s1, s2, rho, K_strike, T, r, n_paths=500_000, seed=5)
    # Kirk's is an approximation (not exact for K != 0); allow a wider but
    # still tight tolerance relative to the MC standard error.
    assert abs(kirk - mc_price) < 15 * se


def test_spark_spread_mc_payoff_is_nonnegative_for_call():
    rng = np.random.default_rng(0)
    power_paths = rng.uniform(20, 200, size=(1000, 5))
    gas_paths = rng.uniform(2, 6, size=(1000, 5))
    price, se = mc_spark_spread_option_on_scenario(
        power_paths, gas_paths, heat_rate=7.5, K=10.0, T=1.0, r=0.03, time_index=4, option_type="call"
    )
    assert price >= 0
    assert se >= 0
