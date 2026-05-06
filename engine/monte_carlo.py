"""
Monte Carlo simulation engine.

Public API:
    run(params, config_dir, seed) → dict          # for LLM tool calling
    run_simulation(params, tax_cfg, seed) → SimulationResult

Each iteration:
  1. Applies a log-normally distributed annual return to all buckets.
  2. Inflation-adjusts spending.
  3. Calculates passive income (SS, pension).
  4. Routes the spending shortfall through the withdrawal router.
  5. Computes taxes (income + capital gains + early-withdrawal penalty).
  6. Pulls taxes and healthcare from the cheapest available bucket.
  7. Records portfolio state; marks the run failed if portfolio reaches zero.

Percentile trajectories are assembled across all iterations with numpy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .aca_engine import compute_healthcare_cost
from .models import AnnualSnapshot, Buckets, SimulationParams, SimulationResult, WithdrawalBreakdown
from .rmd_engine import rmd_required
from .tax_engine import (
    compute_capital_gains_tax,
    compute_income_tax,
    compute_penalty,
    compute_ss_taxable,
    load_tax_config,
)
from .withdrawal_router import route_withdrawal

log = logging.getLogger(__name__)


# ── Single-run simulation ─────────────────────────────────────────────────────

def _simulate_one_run(
    params: SimulationParams,
    tax_cfg: dict,
    rng: np.random.Generator,
) -> list[AnnualSnapshot]:
    buckets = Buckets(
        taxable=params.taxable_balance,
        taxable_basis=params.taxable_basis,
        tax_deferred=params.tax_deferred_balance,
        tax_free=params.tax_free_balance,
        tax_deferred_spouse=params.tax_deferred_spouse_balance,
    )

    start_age = int(params.current_age)
    ages = list(range(start_age, params.plan_to_age + 1))
    # Age offset stays constant throughout the simulation
    spouse_age_offset = (
        params.spouse_current_age - params.current_age
        if params.spouse_current_age is not None else None
    )
    n_years = len(ages)

    # Pre-draw all returns for this run at once (faster than per-year calls)
    # Use a lognormal so returns can't be worse than -100%
    mu = np.log(1 + params.mean_annual_return) - 0.5 * params.return_std_dev ** 2
    sigma = params.return_std_dev
    annual_returns = rng.lognormal(mean=mu, sigma=sigma, size=n_years) - 1.0

    snapshots: list[AnnualSnapshot] = []
    exhausted = False

    for idx, age in enumerate(ages):
        if exhausted:
            snapshots.append(_zero_snapshot(age))
            continue

        # 1. Apply investment return (beginning-of-year growth)
        buckets.apply_return(annual_returns[idx])

        spouse_age = float(age) + spouse_age_offset if spouse_age_offset is not None else None

        # 2. Inflation-adjusted spending target
        inflation_factor = (1.0 + params.inflation_rate) ** idx
        spending = params.annual_spending_today * inflation_factor

        # 3. Passive income — SS computed per person by their own claim age
        ss_primary = params.social_security_annual if age >= params.social_security_start_age else 0.0
        ss_spouse_income = (
            params.ss_spouse_annual
            if spouse_age is not None and spouse_age >= params.ss_spouse_start_age
            else 0.0
        )
        ss = ss_primary + ss_spouse_income
        pension = params.pension_annual
        passive = ss + pension

        # 4. Net cash needed from portfolio (spending, not yet including taxes / healthcare)
        net_needed = max(0.0, spending - passive)

        # 5. Withdraw for spending (pass spouse_age so router can force spouse RMD)
        bd_spend = route_withdrawal(buckets, float(age), net_needed, tax_cfg, spouse_age=spouse_age)

        # 6. Compute MAGI for tax and ACA purposes
        #    MAGI = pension + tax-deferred withdrawals + realized cap gains
        #         + taxable fraction of Social Security
        other_income = pension + bd_spend.ordinary_income + bd_spend.capital_gains
        ss_taxable = compute_ss_taxable(ss, other_income, params.filing_status, tax_cfg)
        total_ordinary = pension + bd_spend.ordinary_income + ss_taxable
        magi = total_ordinary + bd_spend.capital_gains

        # 7. Healthcare — per-person ACA/Medicare routing + medical-cost inflation
        healthcare_inflation_factor = (1.0 + params.healthcare_inflation_rate) ** idx
        healthcare = compute_healthcare_cost(
            primary_age=float(age),
            spouse_age=spouse_age,
            magi=magi,
            household_size=params.household_size,
            filing_status=params.filing_status,
            tax_cfg=tax_cfg,
        ) * healthcare_inflation_factor

        # 8. Federal taxes: income tax + capital gains tax + early-withdrawal penalty
        income_tax = compute_income_tax(total_ordinary, params.filing_status, tax_cfg)
        cg_tax = compute_capital_gains_tax(
            bd_spend.capital_gains, total_ordinary, params.filing_status, tax_cfg
        )
        penalty = compute_penalty(bd_spend.penalty_base, tax_cfg)
        total_tax = income_tax + cg_tax + penalty

        # 9. Pull taxes + healthcare from portfolio (second withdrawal, same priority order)
        #    This second pass may generate a small additional tax liability; we accept
        #    that approximation rather than iterating to convergence.
        extra_needed = total_tax + healthcare
        bd_extra = route_withdrawal(buckets, float(age), extra_needed, tax_cfg, spouse_age=spouse_age)

        # 10. Record
        total_ordinary_all = total_ordinary + bd_extra.ordinary_income
        total_cg_all = bd_spend.capital_gains + bd_extra.capital_gains
        rmd_amt = bd_spend.rmd_amount  # RMD only counted in first withdrawal pass

        snap = AnnualSnapshot(
            age=age,
            portfolio_total=buckets.total(),
            taxable=buckets.taxable,
            tax_deferred=buckets.tax_deferred,
            tax_free=buckets.tax_free,
            ordinary_income=total_ordinary_all,
            capital_gains=total_cg_all,
            taxes_paid=total_tax,
            healthcare_cost=healthcare,
            rmd_amount=rmd_amt,
            spending=spending,
        )
        snapshots.append(snap)

        if buckets.total() <= 0.0:
            exhausted = True

    return snapshots


def _zero_snapshot(age: int) -> AnnualSnapshot:
    return AnnualSnapshot(
        age=age, portfolio_total=0.0, taxable=0.0, tax_deferred=0.0,
        tax_free=0.0, ordinary_income=0.0, capital_gains=0.0,
        taxes_paid=0.0, healthcare_cost=0.0, rmd_amount=0.0, spending=0.0,
    )


# ── Outer Monte Carlo loop ────────────────────────────────────────────────────

def run_simulation(
    params: SimulationParams,
    tax_cfg: dict,
    seed: int | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    n_years = params.plan_to_age - int(params.current_age) + 1
    n = params.num_iterations

    totals     = np.zeros((n, n_years))
    taxes      = np.zeros((n, n_years))
    healthcare = np.zeros((n, n_years))
    successes  = 0

    for i in range(n):
        snaps = _simulate_one_run(params, tax_cfg, rng)
        failed = False
        for yr, s in enumerate(snaps):
            totals[i, yr]     = s.portfolio_total
            taxes[i, yr]      = s.taxes_paid
            healthcare[i, yr] = s.healthcare_cost
            if s.portfolio_total <= 0.0 and not failed:
                failed = True
        if not failed:
            successes += 1

    ages = list(range(int(params.current_age), params.plan_to_age + 1))

    return SimulationResult(
        success_rate=successes / n,
        median_trajectory=np.median(totals, axis=0).tolist(),
        p10_trajectory=np.percentile(totals, 10, axis=0).tolist(),
        p90_trajectory=np.percentile(totals, 90, axis=0).tolist(),
        median_taxes=np.median(taxes, axis=0).tolist(),
        median_healthcare=np.median(healthcare, axis=0).tolist(),
        ages=ages,
        num_iterations=n,
        plan_to_age=params.plan_to_age,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run(
    params: SimulationParams | dict,
    config_dir: Path | str = Path("config"),
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Primary entry point for the LLM tool-calling interface and the Streamlit UI.

    Accepts either a SimulationParams dataclass or a plain dict (from JSON).
    Returns a JSON-serialisable dict.
    """
    if isinstance(params, dict):
        params = SimulationParams(**params)

    tax_cfg = load_tax_config(Path(config_dir))
    result = run_simulation(params, tax_cfg, seed=seed)
    return result.to_dict()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MonteFIRE Monte Carlo engine")
    parser.add_argument("--portfolio",  default="output/portfolio_state.json")
    parser.add_argument("--config",     default="config")
    parser.add_argument("--spending",   type=float, default=80_000)
    parser.add_argument("--plan-to",    type=int,   default=90,    dest="plan_to_age")
    parser.add_argument("--iterations", type=int,   default=1_000)
    parser.add_argument("--ss-annual",  type=float, default=0.0,   dest="ss_annual")
    parser.add_argument("--ss-start",   type=int,   default=67,    dest="ss_start")
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--out",        default=None, help="Write JSON result to file")
    args = parser.parse_args()

    with open(args.portfolio) as f:
        state = json.load(f)

    sim_params = SimulationParams.from_portfolio_state(state, overrides={
        "annual_spending_today": args.spending,
        "plan_to_age": args.plan_to_age,
        "num_iterations": args.iterations,
        "social_security_annual": args.ss_annual,
        "social_security_start_age": args.ss_start,
    })

    log.info(
        "Running %d iterations, age %.1f → %d, spending $%.0f/yr",
        sim_params.num_iterations, sim_params.current_age,
        sim_params.plan_to_age, sim_params.annual_spending_today,
    )

    result = run(sim_params, config_dir=args.config, seed=args.seed)
    output = json.dumps(result, indent=2)

    if args.out:
        Path(args.out).write_text(output)
        log.info("Result written to %s", args.out)
    else:
        print(output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _cli()
