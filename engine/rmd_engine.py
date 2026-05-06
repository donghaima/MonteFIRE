"""
Required Minimum Distribution calculations.

Implements IRS Uniform Lifetime Table (Publication 590-B).
All constants come from tax_config.yaml.
"""

from __future__ import annotations


def rmd_required(age: float, tax_cfg: dict) -> bool:
    return age >= tax_cfg["rmd"]["start_age"]


def distribution_period(age: int, tax_cfg: dict) -> float:
    """
    IRS Uniform Lifetime Table distribution period for the given age.
    Ages beyond the table maximum use the minimum period (shortest remaining life).
    """
    table: dict = tax_cfg["rmd"]["uniform_lifetime_table"]
    clamped = min(max(age, min(int(k) for k in table)), max(int(k) for k in table))
    return float(table[clamped])


def compute_rmd(balance: float, age: int, tax_cfg: dict) -> float:
    """
    Minimum amount that must be distributed from a tax-deferred account this year.
    Returns 0 if age is below the RMD start age.
    """
    if not rmd_required(float(age), tax_cfg):
        return 0.0
    period = distribution_period(age, tax_cfg)
    return balance / period
