"""
Withdrawal Router — pulls cash from account buckets in tax-optimal priority order.

Priority rules:
  Age < 59.5  → Taxable → Tax-Free → Tax-Deferred (avoid the 10% penalty)
  Age 59.5–72 → Taxable → Tax-Deferred → Tax-Free  (standard FIRE sequencing)
  Age 73+     → RMD first (forced from Tax-Deferred) then standard order above

Capital gains tracking:
  When selling taxable assets, the realized gain is proportional to the
  unrealised-gain ratio: (balance - basis) / balance.
  The cost basis is reduced by the basis-portion of each sale so the ratio
  stays accurate throughout the simulation.
"""

from __future__ import annotations

from .models import Buckets, WithdrawalBreakdown
from .rmd_engine import compute_rmd, rmd_required


def route_withdrawal(
    buckets: Buckets,
    age: float,
    needed: float,
    tax_cfg: dict,
    spouse_age: float | None = None,
) -> WithdrawalBreakdown:
    """
    Pull `needed` dollars from buckets, mutating balances in place.
    Returns a breakdown of income types generated (used by tax_engine).

    RMDs are computed per person from their own tax-deferred bucket.
    Discretionary withdrawals treat both tax-deferred buckets as one pool
    (primary's bucket first).
    """
    if needed <= 0.0:
        return WithdrawalBreakdown()

    bd = WithdrawalBreakdown()
    remaining = needed

    # ── Step 1: Force RMDs ────────────────────────────────────────────────────
    def _pull_rmd(balance_attr: str, rmd_age: int) -> float:
        bal = getattr(buckets, balance_attr)
        if bal <= 0.0:
            return 0.0
        rmd = compute_rmd(bal, rmd_age, tax_cfg)
        pulled = min(rmd, bal)
        setattr(buckets, balance_attr, bal - pulled)
        return pulled

    if rmd_required(age, tax_cfg):
        pulled = _pull_rmd("tax_deferred", int(age))
        bd.ordinary_income += pulled
        bd.rmd_amount      += pulled
        remaining = max(0.0, remaining - pulled)

    if spouse_age is not None and rmd_required(spouse_age, tax_cfg):
        pulled = _pull_rmd("tax_deferred_spouse", int(spouse_age))
        bd.ordinary_income += pulled
        bd.rmd_amount      += pulled
        remaining = max(0.0, remaining - pulled)

    # ── Step 2: Discretionary withdrawals in priority order ───────────────────
    penalty_free_age: float = tax_cfg["penalty_free_age"]
    if age < penalty_free_age:
        priority = ["taxable", "tax_free", "tax_deferred"]
    else:
        priority = ["taxable", "tax_deferred", "tax_free"]

    for bucket in priority:
        if remaining <= 0.0:
            break

        if bucket == "taxable":
            available = buckets.taxable
            pull = min(remaining, available)
            if pull > 0.0:
                gain_ratio = max(0.0, min(1.0,
                    (buckets.taxable - buckets.taxable_basis) / buckets.taxable
                )) if buckets.taxable > 0.0 else 0.0
                bd.capital_gains += pull * gain_ratio
                buckets.taxable -= pull
                buckets.taxable_basis = max(0.0, buckets.taxable_basis - pull * (1.0 - gain_ratio))
                remaining -= pull

        elif bucket == "tax_deferred":
            # Draw from primary's bucket first, then spouse's
            for attr in ("tax_deferred", "tax_deferred_spouse"):
                if remaining <= 0.0:
                    break
                available = getattr(buckets, attr)
                pull = min(remaining, available)
                if pull > 0.0:
                    setattr(buckets, attr, available - pull)
                    bd.ordinary_income += pull
                    if age < penalty_free_age:
                        bd.penalty_base += pull
                    remaining -= pull

        elif bucket == "tax_free":
            available = buckets.tax_free
            pull = min(remaining, available)
            if pull > 0.0:
                buckets.tax_free -= pull
                remaining -= pull

    bd.shortfall = max(0.0, remaining)
    return bd
