"""Data models shared across all engine modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SimulationParams:
    """All inputs required by the Monte Carlo engine."""

    # ── Portfolio starting balances ───────────────────────────────────────────
    taxable_balance: float         # current market value of brokerage accounts
    taxable_basis: float           # aggregate cost basis; used for cap-gains calc
    tax_deferred_balance: float    # traditional 401k + IRA
    tax_free_balance: float        # Roth 401k + Roth IRA + HSA

    # ── Demographics ─────────────────────────────────────────────────────────
    current_age: float
    plan_to_age: int = 90
    filing_status: str = "married_filing_jointly"   # or "single"
    household_size: int = 2
    persons_covered_by_aca: int = 2  # adults receiving ACA coverage pre-65

    # ── Market assumptions ────────────────────────────────────────────────────
    mean_annual_return: float = 0.07
    return_std_dev: float = 0.15
    inflation_rate: float = 0.03

    # ── Cash flows (today's dollars; engine inflation-adjusts internally) ─────
    annual_spending_today: float = 80_000  # non-healthcare living expenses
    social_security_annual: float = 0.0
    social_security_start_age: int = 67
    pension_annual: float = 0.0            # treated as fixed nominal (not inflation-adjusted)

    # ── Simulation config ─────────────────────────────────────────────────────
    num_iterations: int = 1_000

    def total_balance(self) -> float:
        return self.taxable_balance + self.tax_deferred_balance + self.tax_free_balance

    @classmethod
    def from_portfolio_state(cls, state: dict, overrides: dict | None = None) -> "SimulationParams":
        """
        Convenience constructor: pull bucket totals from a portfolio_state.json dict,
        then apply any overrides (spending, age, etc.) from the caller.
        """
        summary = state["summary"]["by_tax_treatment"]

        # Compute aggregate cost basis from taxable holdings
        taxable_basis = 0.0
        for acct in state.get("accounts", []):
            if acct["tax_treatment"] == "taxable":
                for h in acct.get("holdings", []):
                    taxable_basis += h.get("cost_basis_usd", 0.0)

        # Pull primary member age
        members = state.get("owner", {}).get("members", [])
        primary = next((m for m in members if m["role"] == "primary"), members[0] if members else None)
        current_age = 0.0
        if primary:
            from datetime import date
            bd = date.fromisoformat(primary["birth_date"])
            current_age = (date.today() - bd).days / 365.25

        params = cls(
            taxable_balance=summary.get("taxable", 0.0),
            taxable_basis=taxable_basis,
            tax_deferred_balance=summary.get("tax_deferred", 0.0),
            tax_free_balance=summary.get("tax_free", 0.0),
            current_age=round(current_age, 2),
        )
        if overrides:
            for k, v in overrides.items():
                setattr(params, k, v)
        return params


@dataclass
class Buckets:
    """Mutable per-run account balances tracked through the simulation."""
    taxable: float
    taxable_basis: float    # tracks realized cost basis; decreases as we sell
    tax_deferred: float
    tax_free: float

    def total(self) -> float:
        return max(0.0, self.taxable + self.tax_deferred + self.tax_free)

    def clone(self) -> "Buckets":
        return Buckets(self.taxable, self.taxable_basis, self.tax_deferred, self.tax_free)

    def apply_return(self, rate: float) -> None:
        """Grow all buckets by rate. Taxable basis stays fixed (unrealized gains accumulate)."""
        self.taxable = max(0.0, self.taxable * (1.0 + rate))
        self.tax_deferred = max(0.0, self.tax_deferred * (1.0 + rate))
        self.tax_free = max(0.0, self.tax_free * (1.0 + rate))


@dataclass
class WithdrawalBreakdown:
    ordinary_income: float = 0.0   # from tax-deferred (all ordinary)
    capital_gains: float = 0.0     # realized long-term gains from taxable
    rmd_amount: float = 0.0        # subset of ordinary_income that was RMD-forced
    penalty_base: float = 0.0      # tax-deferred amount subject to 10% early penalty
    shortfall: float = 0.0         # amount requested but unavailable (portfolio empty)


@dataclass
class AnnualSnapshot:
    age: int
    portfolio_total: float
    taxable: float
    tax_deferred: float
    tax_free: float
    ordinary_income: float
    capital_gains: float
    taxes_paid: float
    healthcare_cost: float
    rmd_amount: float
    spending: float                 # inflation-adjusted spending target for this year


@dataclass
class SimulationResult:
    success_rate: float
    median_trajectory: list[float]
    p10_trajectory: list[float]
    p90_trajectory: list[float]
    median_taxes: list[float]
    median_healthcare: list[float]
    ages: list[int]
    num_iterations: int
    plan_to_age: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_rate": round(self.success_rate, 4),
            "ages": self.ages,
            "median_trajectory":  [round(v, 2) for v in self.median_trajectory],
            "p10_trajectory":     [round(v, 2) for v in self.p10_trajectory],
            "p90_trajectory":     [round(v, 2) for v in self.p90_trajectory],
            "median_taxes":       [round(v, 2) for v in self.median_taxes],
            "median_healthcare":  [round(v, 2) for v in self.median_healthcare],
            "num_iterations":     self.num_iterations,
            "plan_to_age":        self.plan_to_age,
        }
