"""Client-facing style portfolio risk report.

Combines three books valued over the *same* set of jointly-simulated price
paths so their outcomes are properly correlated when aggregated into a
portfolio distribution:

* a gas-fired tolling agreement (``generation_asset_valuation``)
* a battery storage asset (``battery_storage_optionality``)
* a fixed-for-floating power swap hedge

Because each book's Monte Carlo value is computed path-by-path on a shared
``PriceScenario``, summing path ``p``'s value across books gives a
consistent portfolio outcome for scenario ``p``, and VaR/CVaR computed from
that combined distribution correctly reflects the correlation between the
books (e.g. the tolling agreement and a short power swap naturally offset
each other in high-price scenarios).

Value-at-Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall) are
computed via historical simulation directly on the Monte Carlo distribution
of total portfolio value, expressed relative to the portfolio's expected
(mean) value — i.e. this measures the risk of the realized valuation
outcome falling short of what was expected/booked, not a single-day
trading P&L.

DISCLAIMER: synthetic data / educational purposes only, see
``power_price_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .battery_storage_optionality import BatterySpec, value_battery_storage
from .generation_asset_valuation import TollingAgreementSpec, value_tolling_agreement
from .power_price_model import PriceScenario


@dataclass
class SwapHedgeSpec:
    """A simple fixed-for-floating power swap position."""

    fixed_price: float = 50.0  # $/MWh
    notional_mwh_per_day: float = 50.0
    position: str = "short_power"  # "short_power": receive fixed, pay floating spot


def value_swap_hedge(scenario: PriceScenario, spec: SwapHedgeSpec, r: float) -> Dict:
    """Discounted per-path P&L of the swap position across all simulated paths."""
    if spec.position not in ("short_power", "long_power"):
        raise ValueError("position must be 'short_power' or 'long_power'")
    sign = 1.0 if spec.position == "short_power" else -1.0

    dt = scenario.dt_years
    t_years = np.arange(scenario.n_steps) * dt
    disc = np.exp(-r * t_years)

    payoff_per_day = sign * (spec.fixed_price - scenario.power_prices) * spec.notional_mwh_per_day
    discounted = payoff_per_day * disc[None, :]
    path_values = discounted.sum(axis=1)

    n_paths = scenario.n_paths
    return {
        "mean_value": float(path_values.mean()),
        "std_error": float(path_values.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0,
        "path_values": path_values,
    }


def compute_var_cvar(
    pnl: np.ndarray, confidence_level: float = 0.95, relative_to_mean: bool = True
) -> Dict[str, float]:
    """Historical-simulation VaR and CVaR (Expected Shortfall) from a P&L/value sample.

    ``VaR`` is the loss threshold such that losses exceed it with probability
    ``1 - confidence_level``; ``CVaR`` is the average loss in that tail.
    """
    pnl = np.asarray(pnl)
    baseline = pnl.mean() if relative_to_mean else 0.0
    losses = -(pnl - baseline)
    var = float(np.quantile(losses, confidence_level))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    return {"VaR": var, "CVaR": cvar, "confidence_level": confidence_level}


def build_portfolio_pnl(
    scenario: PriceScenario,
    tolling_spec: TollingAgreementSpec,
    battery_spec: BatterySpec,
    swap_spec: SwapHedgeSpec,
    r: float,
) -> Dict:
    """Value each book on the shared scenario and combine into a portfolio P&L distribution."""
    tolling_result = value_tolling_agreement(scenario, tolling_spec, r)
    battery_result = value_battery_storage(scenario.power_prices, battery_spec, r, dt_hours=24.0)
    swap_result = value_swap_hedge(scenario, swap_spec, r)

    portfolio_pnl = (
        tolling_result["path_values"] + battery_result["path_values"] + swap_result["path_values"]
    )

    return {
        "tolling": tolling_result,
        "battery": battery_result,
        "swap": swap_result,
        "portfolio_path_values": portfolio_pnl,
    }


def generate_risk_report(
    scenario: PriceScenario,
    tolling_spec: TollingAgreementSpec,
    battery_spec: BatterySpec,
    swap_spec: SwapHedgeSpec,
    r: float,
    confidence_levels: Sequence[float] = (0.95, 0.99),
) -> Dict:
    """Build the full portfolio risk report: per-book values plus portfolio VaR/CVaR
    at each requested confidence level (with each book's standalone VaR/CVaR
    included for a diversification-benefit comparison).
    """
    books = build_portfolio_pnl(scenario, tolling_spec, battery_spec, swap_spec, r)

    book_pnls = {
        "tolling_agreement": books["tolling"]["path_values"],
        "battery_storage": books["battery"]["path_values"],
        "swap_hedge": books["swap"]["path_values"],
        "portfolio": books["portfolio_path_values"],
    }

    rows = []
    for level in confidence_levels:
        for book_name, pnl in book_pnls.items():
            metrics = compute_var_cvar(pnl, confidence_level=level)
            rows.append(
                {
                    "book": book_name,
                    "confidence_level": level,
                    "mean_value": float(pnl.mean()),
                    "std_value": float(pnl.std(ddof=1)),
                    "VaR": metrics["VaR"],
                    "CVaR": metrics["CVaR"],
                }
            )
    var_cvar_table = pd.DataFrame(rows)

    standalone_sum_var = var_cvar_table[
        (var_cvar_table["book"] != "portfolio") & (var_cvar_table["confidence_level"] == confidence_levels[0])
    ]["VaR"].sum()
    portfolio_var = var_cvar_table[
        (var_cvar_table["book"] == "portfolio") & (var_cvar_table["confidence_level"] == confidence_levels[0])
    ]["VaR"].iloc[0]
    diversification_benefit = float(standalone_sum_var - portfolio_var)

    return {
        "books": books,
        "var_cvar_table": var_cvar_table,
        "diversification_benefit": diversification_benefit,
        "n_paths": scenario.n_paths,
    }


def format_report_text(report: Dict) -> str:
    """Render the risk report as a plain-text summary suitable for a client memo."""
    lines = []
    lines.append("=" * 72)
    lines.append("SYNTHETIC PORTFOLIO RISK REPORT (educational / demo data only)")
    lines.append("=" * 72)
    lines.append(f"Monte Carlo paths: {report['n_paths']}")
    lines.append("")

    books = report["books"]
    lines.append("Book-level expected values:")
    lines.append(f"  Tolling agreement : {books['tolling']['mean_value']:>14,.0f}  (+/- {books['tolling']['std_error']:,.0f})")
    lines.append(f"  Battery storage   : {books['battery']['mean_value']:>14,.0f}  (+/- {books['battery']['std_error']:,.0f})")
    lines.append(f"  Swap hedge        : {books['swap']['mean_value']:>14,.0f}  (+/- {books['swap']['std_error']:,.0f})")
    portfolio_mean = books["portfolio_path_values"].mean()
    lines.append(f"  Portfolio total   : {portfolio_mean:>14,.0f}")
    lines.append("")

    lines.append("VaR / CVaR (relative to expected value):")
    table = report["var_cvar_table"]
    for level in sorted(table["confidence_level"].unique()):
        lines.append(f"  -- {level:.0%} confidence --")
        sub = table[table["confidence_level"] == level]
        for _, row in sub.iterrows():
            lines.append(f"     {row['book']:<20} VaR: {row['VaR']:>14,.0f}   CVaR: {row['CVaR']:>14,.0f}")
    lines.append("")
    lines.append(f"Estimated diversification benefit (sum of standalone VaR - portfolio VaR, "
                 f"at {table['confidence_level'].min():.0%}): {report['diversification_benefit']:,.0f}")
    lines.append("=" * 72)
    return "\n".join(lines)
