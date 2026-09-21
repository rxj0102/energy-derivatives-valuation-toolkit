import itertools

import numpy as np

from energy_toolkit.generation_asset_valuation import (
    TollingAgreementSpec,
    optimal_monthly_dispatch,
    value_tolling_agreement,
)


def _brute_force_dispatch(spread, disc, capacity_mw, hours_per_day, max_starts, min_run_days, start_cost):
    n = len(spread)
    possible_starts = [d for d in range(n) if d + min_run_days <= n]
    best_value = 0.0
    for k in range(0, max_starts + 1):
        for combo in itertools.combinations(possible_starts, k):
            combo_sorted = sorted(combo)
            if any(combo_sorted[i] + min_run_days > combo_sorted[i + 1] for i in range(len(combo_sorted) - 1)):
                continue  # overlapping runs, infeasible
            value = 0.0
            for start in combo_sorted:
                window = slice(start, start + min_run_days)
                value += np.sum(spread[window] * capacity_mw * hours_per_day * disc[window]) - start_cost * disc[start]
            best_value = max(best_value, value)
    return best_value


def test_dispatch_dp_matches_brute_force_small_case():
    rng = np.random.default_rng(0)
    spread = rng.normal(0, 10, size=10)
    disc = np.ones(10)
    dp_mask, dp_value = optimal_monthly_dispatch(
        spread, disc, capacity_mw=1.0, hours_per_day=1.0, max_starts=2, min_run_days=3, start_cost=2.0
    )
    brute_value = _brute_force_dispatch(spread, disc, 1.0, 1.0, max_starts=2, min_run_days=3, start_cost=2.0)
    assert abs(dp_value - brute_value) < 1e-9


def test_dispatch_respects_max_starts_budget():
    spread = np.array([10, -10, 10, -10, 10, -10, 10], dtype=float)
    disc = np.ones_like(spread)
    mask, value = optimal_monthly_dispatch(
        spread, disc, capacity_mw=1.0, hours_per_day=1.0, max_starts=1, min_run_days=1, start_cost=0.0
    )
    assert mask.sum() == 1
    assert value == 10.0


def test_dispatch_never_starts_when_all_spreads_negative():
    spread = -np.abs(np.random.default_rng(1).normal(5, 2, size=20))
    disc = np.ones_like(spread)
    mask, value = optimal_monthly_dispatch(
        spread, disc, capacity_mw=1.0, hours_per_day=1.0, max_starts=5, min_run_days=2, start_cost=0.0
    )
    assert value == 0.0
    assert not mask.any()


def test_constrained_value_never_exceeds_unconstrained_intrinsic(small_scenario):
    spec = TollingAgreementSpec(capacity_mw=50, max_starts_per_month=3, min_run_days=2)
    result = value_tolling_agreement(small_scenario, spec, r=0.04)
    assert result["mean_value"] <= result["unconstrained_intrinsic_value"] + 1e-6


def test_relaxing_start_constraint_increases_or_maintains_value(small_scenario):
    tight_spec = TollingAgreementSpec(capacity_mw=50, max_starts_per_month=1, min_run_days=3)
    loose_spec = TollingAgreementSpec(capacity_mw=50, max_starts_per_month=15, min_run_days=1)
    tight = value_tolling_agreement(small_scenario, tight_spec, r=0.04)
    loose = value_tolling_agreement(small_scenario, loose_spec, r=0.04)
    assert loose["mean_value"] >= tight["mean_value"]


def test_value_tolling_agreement_output_structure(small_scenario):
    spec = TollingAgreementSpec()
    result = value_tolling_agreement(small_scenario, spec, r=0.04)
    assert result["mean_value"] > 0
    assert result["std_error"] >= 0
    assert len(result["path_values"]) == small_scenario.n_paths
    assert result["n_months"] == 12
