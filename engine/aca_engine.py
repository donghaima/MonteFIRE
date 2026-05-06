"""
Healthcare cost estimation.

Before Medicare eligibility (age 65): ACA marketplace premiums with subsidy cliff.
At and after 65: Medicare Part B + Part D + supplemental (Medigap), with IRMAA.

All thresholds come from tax_config.yaml — nothing hardcoded.
"""

from __future__ import annotations


def _fpl(household_size: int, tax_cfg: dict) -> float:
    """Federal Poverty Level for the given household size."""
    table: dict = tax_cfg["aca"]["fpl_by_household_size"]
    # Clamp to table bounds (max key is 4 in config; larger households use that value)
    key = min(household_size, max(int(k) for k in table))
    return float(table[key])


def _contribution_pct(fpl_multiple: float, schedule: list[list]) -> float:
    """
    Interpolate the required contribution percentage from the piecewise schedule.
    schedule entries: [fpl_multiple, max_contribution_pct_of_income]
    """
    if fpl_multiple <= schedule[0][0]:
        return schedule[0][1]
    for i in range(len(schedule) - 1):
        lo_x, lo_y = schedule[i]
        hi_x, hi_y = schedule[i + 1]
        if lo_x <= fpl_multiple <= hi_x:
            t = (fpl_multiple - lo_x) / (hi_x - lo_x)
            return lo_y + t * (hi_y - lo_y)
    return schedule[-1][1]


def compute_aca_cost(
    magi: float,
    household_size: int,
    persons_covered: int,
    tax_cfg: dict,
) -> float:
    """
    Annual ACA premium cost for the household.

    Below the subsidy cliff (400% FPL): premium is capped at the contribution
    percentage of income, so you pay little or nothing.
    Above the cliff: no subsidy — you pay the full unsubsidized benchmark premium.
    This creates the sharp cliff that dominates FIRE healthcare planning.
    """
    aca_cfg = tax_cfg["aca"]
    fpl = _fpl(household_size, tax_cfg)
    cliff_income = fpl * aca_cfg["subsidy_cliff_multiple"]
    full_premium = aca_cfg["unsubsidized_annual_premium_per_person"] * persons_covered

    if magi >= cliff_income:
        return float(full_premium)

    fpl_multiple = magi / fpl if fpl > 0 else 0.0
    schedule = aca_cfg["contribution_schedule"]
    contribution_pct = _contribution_pct(fpl_multiple, schedule)

    # You pay the lesser of: (a) your expected contribution or (b) the full premium
    your_contribution = magi * contribution_pct
    return min(float(full_premium), your_contribution)


def compute_medicare_cost(
    magi: float,
    persons_covered: int,
    filing_status: str,
    tax_cfg: dict,
) -> float:
    """
    Annual Medicare cost: Part B + Part D + Medigap supplement + IRMAA surcharge.

    IRMAA applies based on income from 2 years prior; for simulation purposes
    we use the current year's MAGI as a proxy (conservative for high-income years).
    """
    mc = tax_cfg["medicare"]

    base_annual = (
        mc["part_b_monthly_per_person"] * 12
        + mc["part_d_monthly_per_person"] * 12
        + mc["supplement_annual_per_person"]
    ) * persons_covered

    # IRMAA surcharge (MFJ table only for now; single threshold is half of MFJ)
    irmaa_thresholds = mc.get("irmaa_thresholds_mfj", [])
    if filing_status == "single":
        # IRS single thresholds are roughly half of MFJ
        irmaa_thresholds = [
            {"magi_floor": t["magi_floor"] / 2, "surcharge_monthly_per_person": t["surcharge_monthly_per_person"]}
            for t in irmaa_thresholds
        ]

    surcharge_monthly = 0.0
    for tier in reversed(irmaa_thresholds):
        if magi >= tier["magi_floor"]:
            surcharge_monthly = tier["surcharge_monthly_per_person"]
            break

    irmaa_annual = surcharge_monthly * 12 * persons_covered
    return base_annual + irmaa_annual


def compute_healthcare_cost(
    age: float,
    magi: float,
    household_size: int,
    persons_covered: int,
    filing_status: str,
    tax_cfg: dict,
) -> float:
    """
    Route to ACA or Medicare based on age. Returns total annual healthcare cost.
    """
    medicare_age = tax_cfg["medicare"]["eligible_age"]
    if age >= medicare_age:
        return compute_medicare_cost(magi, persons_covered, filing_status, tax_cfg)
    return compute_aca_cost(magi, household_size, persons_covered, tax_cfg)
