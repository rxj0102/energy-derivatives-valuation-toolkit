"""Valuation of a synthetic battery storage asset as a swing/storage option.

A battery is an option to buy power when it is cheap (charge) and sell it
back when it is expensive (discharge), subject to a maximum power rating
(MW), a maximum energy capacity (MWh), and round-trip efficiency losses.
Unlike a plain vanilla option, the "right" decision at any moment depends
on the battery's current state of charge (SoC) as well as the price, and
the optimal policy has to be solved jointly across the whole horizon — this
is a multi-exercise ("swing") optimal control problem, not a single
option.

This module values it with a Least-Squares Monte Carlo (Longstaff-Schwartz)
approach adapted to storage, following the regression-based dynamic
programming method of Boogert & de Jong (2008):

1. Discretize the state of charge onto a grid with ``n_soc_levels`` nodes
   spanning ``[0, energy_capacity_mwh]``. Grid spacing defaults to the
   energy moved by charging/discharging at full power for one time step, so
   each decision period allows exactly one of three actions from any grid
   node: **charge** (move up one level), **discharge** (move down one
   level), or **hold**.
2. **Backward pass**: starting from a terminal liquidation payoff at the
   grid's SoC levels, step backward through time. At each time step and
   each SoC level, regress the (already-computed) value-to-go at the
   resulting next-period SoC level onto a polynomial basis of the current
   price, across all simulated paths. Comparing the fitted continuation
   value of each feasible action (plus its immediate cash flow) picks the
   best action and defines that node's value-to-go for the next iteration
   back. The fitted regression coefficients are the *policy*.
3. **Forward pass**: starting from an empty battery, walk forward along
   each simulated path applying the backward-fitted policy (evaluated at
   each day's realized price) and accumulating *realized* (not regressed)
   discounted cash flows. This second pass is what is reported as the
   asset's value — using realized cash flows under a regression-fitted
   policy is the standard Longstaff-Schwartz device for avoiding
   look-ahead bias in the reported value.

Efficiency losses are booked entirely on the charge side: charging by one
grid step costs ``step / round_trip_efficiency`` MWh purchased from the
grid, while discharging by one grid step sells exactly ``step`` MWh. That
convention reproduces the specified round-trip efficiency exactly over one
charge/discharge cycle and is a common, harmless simplification (the split
between charge- and discharge-side losses does not affect asset value,
only where the loss is booked).

DISCLAIMER: synthetic data / educational purposes only, see
``power_price_model.py``. The regression-based value function is itself an
approximation and can accumulate fitting error over many time steps; this
is a known characteristic of grid/regression storage models, not a defect
specific to this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

HOURS_PER_YEAR = 24 * 365.25


@dataclass
class BatterySpec:
    """Operating characteristics of the battery storage asset."""

    power_mw: float = 25.0
    energy_capacity_mwh: float = 100.0
    round_trip_efficiency: float = 0.85
    variable_om_cost: float = 0.5  # $/MWh of throughput (either direction)
    n_soc_levels: Optional[int] = None  # auto-derived from power/energy/dt if None


def resolve_soc_grid(spec: BatterySpec, dt_hours: float) -> Tuple[int, float]:
    """Determine the number of SoC grid levels and MWh spacing between them."""
    nominal_step = spec.power_mw * dt_hours
    if spec.n_soc_levels is not None:
        n_levels = spec.n_soc_levels
    else:
        n_levels = max(2, int(round(spec.energy_capacity_mwh / max(nominal_step, 1e-9))) + 1)
    step_mwh = spec.energy_capacity_mwh / (n_levels - 1)
    return n_levels, step_mwh


def _basis(price: np.ndarray) -> np.ndarray:
    """Quadratic polynomial basis in price for the LSM regression."""
    return np.column_stack([np.ones_like(price), price, price**2])


def _fit_coeffs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares fit of y ~ basis(x), robust to small/degenerate samples."""
    design = _basis(x)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs


Action = str  # "charge" | "hold" | "discharge"


def _charge_cashflow(price: np.ndarray, step_mwh: float, eta: float, om_cost: float) -> np.ndarray:
    return -(step_mwh / eta) * price - om_cost * step_mwh


def _discharge_cashflow(price: np.ndarray, step_mwh: float, om_cost: float) -> np.ndarray:
    return step_mwh * price - om_cost * step_mwh


def fit_battery_policy(
    price_paths: np.ndarray,
    spec: BatterySpec,
    r: float,
    dt_hours: float = 24.0,
) -> Dict:
    """Backward pass: fit the regression-based dispatch policy.

    Returns a dict with the fitted per-(time, level) action coefficients
    plus grid metadata, consumed by ``apply_policy_forward``.
    """
    n_paths, n_steps = price_paths.shape
    n_levels, step_mwh = resolve_soc_grid(spec, dt_hours)
    eta = spec.round_trip_efficiency
    om = spec.variable_om_cost
    disc_dt = float(np.exp(-r * dt_hours / HOURS_PER_YEAR))

    level_energy = np.arange(n_levels) * step_mwh

    T_idx = n_steps - 1  # last column: forced liquidation day, no decision
    # terminal value-to-go at each level, for every path
    V_hat = np.tile(level_energy, (n_paths, 1)) * price_paths[:, T_idx][:, None]

    # policy[t][level] = {action: (coeffs, next_level)}
    policy: List[Dict[int, Dict[Action, Tuple[np.ndarray, int]]]] = [dict() for _ in range(T_idx)]

    for t in range(T_idx - 1, -1, -1):
        price_t = price_paths[:, t]
        new_V = np.empty((n_paths, n_levels))
        for i in range(n_levels):
            candidates: Dict[Action, np.ndarray] = {}
            node_policy: Dict[Action, Tuple[np.ndarray, int]] = {}

            # hold
            coeffs_hold = _fit_coeffs(price_t, V_hat[:, i])
            cont_hold = _basis(price_t) @ coeffs_hold
            candidates["hold"] = 0.0 + disc_dt * cont_hold
            node_policy["hold"] = (coeffs_hold, i)

            if i + 1 < n_levels:
                coeffs_c = _fit_coeffs(price_t, V_hat[:, i + 1])
                cont_c = _basis(price_t) @ coeffs_c
                cf_c = _charge_cashflow(price_t, step_mwh, eta, om)
                candidates["charge"] = cf_c + disc_dt * cont_c
                node_policy["charge"] = (coeffs_c, i + 1)

            if i - 1 >= 0:
                coeffs_d = _fit_coeffs(price_t, V_hat[:, i - 1])
                cont_d = _basis(price_t) @ coeffs_d
                cf_d = _discharge_cashflow(price_t, step_mwh, om)
                candidates["discharge"] = cf_d + disc_dt * cont_d
                node_policy["discharge"] = (coeffs_d, i - 1)

            stacked = np.column_stack(list(candidates.values()))
            new_V[:, i] = stacked.max(axis=1)
            policy[t][i] = node_policy

        V_hat = new_V

    return {
        "policy": policy,
        "n_levels": n_levels,
        "step_mwh": step_mwh,
        "level_energy": level_energy,
        "disc_dt": disc_dt,
        "T_idx": T_idx,
    }


def apply_policy_forward(
    price_paths: np.ndarray,
    fitted: Dict,
    spec: BatterySpec,
) -> Dict:
    """Forward pass: apply the fitted policy path-by-path, accumulating realized cash flows."""
    n_paths, n_steps = price_paths.shape
    policy = fitted["policy"]
    step_mwh = fitted["step_mwh"]
    level_energy = fitted["level_energy"]
    disc_dt = fitted["disc_dt"]
    T_idx = fitted["T_idx"]
    eta = spec.round_trip_efficiency
    om = spec.variable_om_cost

    level_state = np.zeros(n_paths, dtype=int)
    total_value = np.zeros(n_paths)
    soc_path = np.zeros((n_paths, n_steps))
    disc_cum = 1.0

    for t in range(T_idx):
        price_t = price_paths[:, t]
        new_level_state = level_state.copy()
        step_cashflow = np.zeros(n_paths)

        for i in np.unique(level_state):
            idx = level_state == i
            node_policy = policy[t][i]
            price_sub = price_t[idx]

            best_value = np.full(idx.sum(), -np.inf)
            best_next_level = np.full(idx.sum(), i)
            best_cashflow = np.zeros(idx.sum())

            for action, (coeffs, next_level) in node_policy.items():
                if action == "hold":
                    cf = np.zeros_like(price_sub)
                elif action == "charge":
                    cf = _charge_cashflow(price_sub, step_mwh, eta, om)
                else:  # discharge
                    cf = _discharge_cashflow(price_sub, step_mwh, om)
                cont = _basis(price_sub) @ coeffs
                val = cf + disc_dt * cont
                better = val > best_value
                best_value = np.where(better, val, best_value)
                best_next_level = np.where(better, next_level, best_next_level)
                best_cashflow = np.where(better, cf, best_cashflow)

            step_cashflow[idx] = best_cashflow
            new_level_state[idx] = best_next_level

        total_value += disc_cum * step_cashflow
        soc_path[:, t] = level_energy[level_state]
        disc_cum *= disc_dt
        level_state = new_level_state

    # forced liquidation at the final day
    liquidation = level_energy[level_state] * price_paths[:, T_idx]
    total_value += disc_cum * liquidation
    soc_path[:, T_idx] = level_energy[level_state]

    return {"path_values": total_value, "soc_path": soc_path, "final_level_state": level_state}


def value_battery_storage(
    price_paths: np.ndarray,
    spec: BatterySpec,
    r: float,
    dt_hours: float = 24.0,
) -> dict:
    """End-to-end LSM valuation of the battery over ``price_paths``.

    Parameters
    ----------
    price_paths:
        Array of shape ``(n_paths, n_steps)``, e.g.
        ``PriceScenario.power_prices`` (daily) or the output of
        ``simulate_diurnal_hourly_prices`` reshaped to ``(n_paths, n_steps)``
        for intraday arbitrage.
    dt_hours:
        Hours between consecutive columns of ``price_paths`` (24 for daily,
        1 for hourly).
    """
    fitted = fit_battery_policy(price_paths, spec, r, dt_hours)
    forward = apply_policy_forward(price_paths, fitted, spec)
    path_values = forward["path_values"]
    n_paths = price_paths.shape[0]
    mean_value = float(path_values.mean())
    std_error = float(path_values.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0

    return {
        "mean_value": mean_value,
        "std_error": std_error,
        "path_values": path_values,
        "n_levels": fitted["n_levels"],
        "step_mwh": fitted["step_mwh"],
        "soc_path": forward["soc_path"],
    }
