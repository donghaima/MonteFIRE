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
) -> WithdrawalBreakdown:
    """
    Pull `needed` dollars from buckets, mutating balances in place.
    Returns a breakdown of income types generated (used by tax_engine).
    """
    if needed <= 0.0:
        return WithdrawalBreakdown()

    bd = WithdrawalBreakdown()
    remaining = needed

    # ── Step 1: Force RMD from tax-deferred if age >= 73 ─────────────────────
    if rmd_required(age, tax_cfg) and buckets.tax_deferred > 0.0:
        rmd = compute_rmd(buckets.tax_deferred, int(age), tax_cfg)
        rmd_pulled = min(rmd, buckets.tax_deferred)
        buckets.tax_deferred -= rmd_pulled
        bd.ordinary_income += rmd_pulled
        bd.rmd_amount = rmd_pulled
        # RMD counts toward the year's cash need; any excess is reinvested (approximated
        # here by simply reducing the remaining need — close enough for simulation)
        remaining = max(0.0, remaining - rmd_pulled)

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
                # Gain ratio: proportion of each dollar that is long-term capital gain
                if buckets.taxable > 0.0:
                    gain_ratio = max(0.0, min(1.0,
                        (buckets.taxable - buckets.taxable_basis) / buckets.taxable
                    ))
                else:
                    gain_ratio = 0.0

                cap_gain = pull * gain_ratio
                basis_used = pull * (1.0 - gain_ratio)

                bd.capital_gains += cap_gain
                buckets.taxable -= pull
                buckets.taxable_basis = max(0.0, buckets.taxable_basis - basis_used)
                remaining -= pull

        elif bucket == "tax_deferred":
            available = buckets.tax_deferred
            pull = min(remaining, available)
            if pull > 0.0:
                buckets.tax_deferred -= pull
                bd.ordinary_income += pull
                if age < penalty_free_age:
                    bd.penalty_base += pull   # 10% penalty assessed by tax_engine
                remaining -= pull

        elif bucket == "tax_free":
            available = buckets.tax_free
            pull = min(remaining, available)
            if pull > 0.0:
                buckets.tax_free -= pull
                remaining -= pull

    bd.shortfall = max(0.0, remaining)
    return bd
