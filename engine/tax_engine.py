"""
Federal income tax and capital gains calculations.

All bracket data comes from tax_config.yaml — nothing is hardcoded here.
Functions are pure (no side effects) so they can be called freely from the
simulation loop and from tests.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_tax_config(config_dir: Path = Path("config")) -> dict:
    path = config_dir / "tax_config.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


# ── Income tax ────────────────────────────────────────────────────────────────

def compute_income_tax(
    ordinary_income: float,
    filing_status: str,
    tax_cfg: dict,
) -> float:
    """
    Progressive federal income tax on ordinary income after the standard deduction.
    Returns the gross tax liability (not effective rate).
    """
    status = tax_cfg["federal_income_tax"][filing_status]
    taxable = max(0.0, ordinary_income - status["standard_deduction"])

    tax = 0.0
    for bracket in status["brackets"]:
        floor: float = bracket["floor"]
        ceiling = bracket["ceiling"]  # None for the top bracket
        rate: float = bracket["rate"]

        if taxable <= floor:
            break

        top = ceiling if ceiling is not None else taxable
        income_in_bracket = min(taxable, top) - floor
        tax += income_in_bracket * rate

    return max(0.0, tax)


def marginal_rate(
    ordinary_income: float,
    filing_status: str,
    tax_cfg: dict,
) -> float:
    """Return the marginal federal income tax rate at the given income level."""
    status = tax_cfg["federal_income_tax"][filing_status]
    taxable = max(0.0, ordinary_income - status["standard_deduction"])

    prev_rate = 0.0
    for bracket in status["brackets"]:
        if taxable <= bracket["floor"]:
            return prev_rate
        prev_rate = bracket["rate"]

    return prev_rate


# ── Capital gains tax ─────────────────────────────────────────────────────────

def compute_capital_gains_tax(
    cap_gains: float,
    ordinary_income: float,
    filing_status: str,
    tax_cfg: dict,
) -> float:
    """
    Long-term capital gains tax.

    Gains are stacked on top of ordinary income (after standard deduction)
    for bracket placement. The first bracket is often 0%, which is the key
    FIRE planning lever: keep ordinary income low to harvest gains tax-free.
    """
    if cap_gains <= 0:
        return 0.0

    status = tax_cfg["federal_income_tax"][filing_status]
    std_deduction = status["standard_deduction"]
    taxable_ordinary = max(0.0, ordinary_income - std_deduction)

    cg_brackets = tax_cfg["long_term_capital_gains"][filing_status]

    tax = 0.0
    remaining = cap_gains
    income_floor = taxable_ordinary  # gains stack here

    for bracket in cg_brackets:
        ceiling = bracket["ceiling"]
        rate: float = bracket["rate"]

        if ceiling is None:
            tax += remaining * rate
            break

        space = max(0.0, ceiling - income_floor)
        in_bracket = min(remaining, space)
        tax += in_bracket * rate
        remaining -= in_bracket
        income_floor = max(income_floor, ceiling)

        if remaining <= 0.0:
            break

    return tax


# ── Social Security taxability ────────────────────────────────────────────────

def compute_ss_taxable(
    ss_annual: float,
    other_income: float,    # all income EXCEPT SS (ordinary + cap gains + pension)
    filing_status: str,
    tax_cfg: dict,
) -> float:
    """
    Returns the portion of Social Security benefits included in gross income.

    IRS provisional income formula:
        provisional = other_income + 0.50 * ss_annual
    Below threshold_50pct  → 0% taxable
    threshold_50pct to threshold_85pct → up to 50% taxable (sliding)
    Above threshold_85pct  → up to 85% taxable
    """
    if ss_annual <= 0:
        return 0.0

    ss_cfg = tax_cfg["social_security_taxation"][filing_status]
    provisional = other_income + 0.5 * ss_annual

    t50 = ss_cfg["threshold_50pct"]
    t85 = ss_cfg["threshold_85pct"]

    if provisional <= t50:
        return 0.0

    if provisional <= t85:
        # 50% of the amount above t50 is included, capped at 50% of benefit
        return min(0.5 * (provisional - t50), 0.5 * ss_annual)

    # Above t85: base from the 50% zone + 85% of excess above t85, capped at 85%
    base_50 = 0.5 * (t85 - t50)
    excess_85 = 0.85 * (provisional - t85)
    return min(base_50 + excess_85, 0.85 * ss_annual)


# ── Early withdrawal penalty ──────────────────────────────────────────────────

def compute_penalty(penalty_base: float, tax_cfg: dict) -> float:
    """10% additional tax on tax-deferred withdrawals before penalty_free_age."""
    return penalty_base * tax_cfg["early_withdrawal_penalty_rate"]
