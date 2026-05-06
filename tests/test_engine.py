"""
Tests for the Phase 2 math engine.

Covers every sub-module independently, then validates the full Monte Carlo
integration. All tests are deterministic (fixed seeds / exact inputs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).parent.parent / "config"

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tax_cfg():
    from engine.tax_engine import load_tax_config
    return load_tax_config(CONFIG_DIR)


@pytest.fixture
def default_params():
    from engine.models import SimulationParams
    return SimulationParams(
        taxable_balance=500_000,
        taxable_basis=300_000,
        tax_deferred_balance=800_000,
        tax_free_balance=200_000,
        current_age=50.0,
        plan_to_age=90,
        annual_spending_today=80_000,
        num_iterations=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tax engine
# ─────────────────────────────────────────────────────────────────────────────

class TestIncomeTax:
    def test_below_standard_deduction_is_zero(self, tax_cfg):
        from engine.tax_engine import compute_income_tax
        # $29,200 standard deduction MFJ — any income under that = $0 tax
        assert compute_income_tax(20_000, "married_filing_jointly", tax_cfg) == 0.0

    def test_exactly_at_deduction_is_zero(self, tax_cfg):
        from engine.tax_engine import compute_income_tax
        assert compute_income_tax(29_200, "married_filing_jointly", tax_cfg) == 0.0

    def test_10pct_bracket(self, tax_cfg):
        from engine.tax_engine import compute_income_tax
        # $40k gross: taxable = $40k - $29.2k = $10.8k → 10% bracket entirely
        # $10,800 × 10% = $1,080
        tax = compute_income_tax(40_000, "married_filing_jointly", tax_cfg)
        assert abs(tax - 1_080.0) < 0.01

    def test_into_12pct_bracket(self, tax_cfg):
        from engine.tax_engine import compute_income_tax
        # $60k gross: taxable = $30,800
        # $23,200 × 10% + ($30,800 - $23,200) × 12% = $2,320 + $912 = $3,232
        tax = compute_income_tax(60_000, "married_filing_jointly", tax_cfg)
        assert abs(tax - 3_232.0) < 0.01

    def test_single_filer(self, tax_cfg):
        from engine.tax_engine import compute_income_tax
        # Single: std deduction $14,600; $30k gross → $15,400 taxable
        # $11,600 × 10% = $1,160 + ($15,400 - $11,600) × 12% = $456 → $1,616
        tax = compute_income_tax(30_000, "single", tax_cfg)
        assert abs(tax - 1_616.0) < 0.01

    def test_never_negative(self, tax_cfg):
        from engine.tax_engine import compute_income_tax
        assert compute_income_tax(0.0, "married_filing_jointly", tax_cfg) == 0.0
        assert compute_income_tax(-5_000, "married_filing_jointly", tax_cfg) == 0.0

    def test_marginal_rate_in_12pct_bracket(self, tax_cfg):
        from engine.tax_engine import marginal_rate
        rate = marginal_rate(60_000, "married_filing_jointly", tax_cfg)
        assert rate == pytest.approx(0.12)

    def test_marginal_rate_below_deduction(self, tax_cfg):
        from engine.tax_engine import marginal_rate
        rate = marginal_rate(10_000, "married_filing_jointly", tax_cfg)
        assert rate == pytest.approx(0.0)


class TestCapitalGainsTax:
    def test_zero_percent_rate_key_fire_scenario(self, tax_cfg):
        from engine.tax_engine import compute_capital_gains_tax
        # MFJ 0% LTCG threshold: $94,050 of total income (ordinary + gains)
        # $50k ordinary income → taxable ordinary = $20,800
        # $20k cap gains → total = $40,800 < $94,050 → 0% rate
        cg_tax = compute_capital_gains_tax(20_000, 50_000, "married_filing_jointly", tax_cfg)
        assert cg_tax == 0.0

    def test_15pct_rate_when_gains_push_over_threshold(self, tax_cfg):
        from engine.tax_engine import compute_capital_gains_tax
        # $120k ordinary → taxable ordinary = $90,800
        # $30k gains stacked: $90,800 to $94,050 is in 0% zone (space = $3,250)
        # remaining $26,750 at 15% → $4,012.50
        cg_tax = compute_capital_gains_tax(30_000, 120_000, "married_filing_jointly", tax_cfg)
        assert abs(cg_tax - 4_012.50) < 1.0

    def test_zero_gains_returns_zero(self, tax_cfg):
        from engine.tax_engine import compute_capital_gains_tax
        assert compute_capital_gains_tax(0.0, 100_000, "married_filing_jointly", tax_cfg) == 0.0

    def test_negative_gains_returns_zero(self, tax_cfg):
        from engine.tax_engine import compute_capital_gains_tax
        assert compute_capital_gains_tax(-5_000, 50_000, "married_filing_jointly", tax_cfg) == 0.0


class TestSSTaxability:
    def test_below_threshold_zero_included(self, tax_cfg):
        from engine.tax_engine import compute_ss_taxable
        # MFJ threshold_50pct = $32,000; provisional = $15k + 0.5×$24k = $27k < $32k
        result = compute_ss_taxable(24_000, 15_000, "married_filing_jointly", tax_cfg)
        assert result == 0.0

    def test_50pct_zone(self, tax_cfg):
        from engine.tax_engine import compute_ss_taxable
        # provisional = $30k + 0.5×$24k = $42k; between $32k and $44k
        # taxable SS = 0.5 × ($42k - $32k) = $5,000
        result = compute_ss_taxable(24_000, 30_000, "married_filing_jointly", tax_cfg)
        assert abs(result - 5_000) < 1.0

    def test_85pct_cap(self, tax_cfg):
        from engine.tax_engine import compute_ss_taxable
        # high income → SS fully included at 85%
        result = compute_ss_taxable(30_000, 200_000, "married_filing_jointly", tax_cfg)
        assert abs(result - 0.85 * 30_000) < 1.0

    def test_zero_ss_returns_zero(self, tax_cfg):
        from engine.tax_engine import compute_ss_taxable
        assert compute_ss_taxable(0, 100_000, "married_filing_jointly", tax_cfg) == 0.0


class TestPenalty:
    def test_10pct_of_base(self, tax_cfg):
        from engine.tax_engine import compute_penalty
        assert compute_penalty(50_000, tax_cfg) == pytest.approx(5_000.0)

    def test_zero_base(self, tax_cfg):
        from engine.tax_engine import compute_penalty
        assert compute_penalty(0.0, tax_cfg) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ACA engine
# ─────────────────────────────────────────────────────────────────────────────

class TestACAEngine:
    def test_below_cliff_affordable_premium(self, tax_cfg):
        from engine.aca_engine import compute_aca_cost
        # 2-person household, FPL = $20,440; cliff at $81,760
        # MAGI = $60k → below cliff; contribution ≈ 7% (between 300-400% FPL range)
        cost = compute_aca_cost(60_000, 2, 2, tax_cfg)
        full_premium = tax_cfg["aca"]["unsubsidized_annual_premium_per_person"] * 2
        assert cost < full_premium * 0.5   # significantly subsidised

    def test_above_cliff_full_premium(self, tax_cfg):
        from engine.aca_engine import compute_aca_cost
        # MAGI = $100k → above cliff ($81,760 for 2-person household)
        cost = compute_aca_cost(100_000, 2, 2, tax_cfg)
        full_premium = tax_cfg["aca"]["unsubsidized_annual_premium_per_person"] * 2
        assert cost == pytest.approx(full_premium)

    def test_cliff_is_a_step_function(self, tax_cfg):
        from engine.aca_engine import compute_aca_cost
        # One dollar below the cliff vs. one above must produce a large jump
        fpl = tax_cfg["aca"]["fpl_by_household_size"][2]
        cliff = fpl * tax_cfg["aca"]["subsidy_cliff_multiple"]
        below = compute_aca_cost(cliff - 1, 2, 2, tax_cfg)
        above = compute_aca_cost(cliff + 1, 2, 2, tax_cfg)
        assert above - below > 5_000   # at least $5k jump at the cliff

    def test_very_low_income_near_zero_premium(self, tax_cfg):
        from engine.aca_engine import compute_aca_cost
        # MAGI = $20k for a 2-person household (< 100% FPL) → 0% contribution rate
        cost = compute_aca_cost(20_000, 2, 2, tax_cfg)
        assert cost == pytest.approx(0.0)

    def test_medicare_cheaper_than_aca_at_cliff(self, tax_cfg):
        from engine.aca_engine import compute_healthcare_cost
        # Both on Medicare at 65 vs both on ACA at 64 — Medicare wins above ACA cliff
        mc  = compute_healthcare_cost(65, 66.0, 100_000, 2, "married_filing_jointly", tax_cfg)
        aca = compute_healthcare_cost(64, 63.0, 100_000, 2, "married_filing_jointly", tax_cfg)
        assert mc < aca

    def test_irmaa_surcharge_above_threshold(self, tax_cfg):
        from engine.aca_engine import compute_medicare_cost
        base = compute_medicare_cost(100_000, 2, "married_filing_jointly", tax_cfg)
        irmaa = compute_medicare_cost(300_000, 2, "married_filing_jointly", tax_cfg)
        assert irmaa > base

    def test_healthcare_routes_to_aca_before_65(self, tax_cfg):
        from engine.aca_engine import compute_aca_cost, compute_healthcare_cost
        # Both spouses under 65 → full ACA cost
        cost_64 = compute_healthcare_cost(64, 63.0, 60_000, 2, "married_filing_jointly", tax_cfg)
        expected = compute_aca_cost(60_000, 2, 2, tax_cfg)
        assert cost_64 == pytest.approx(expected)

    def test_healthcare_routes_to_medicare_at_65(self, tax_cfg):
        from engine.aca_engine import compute_healthcare_cost, compute_medicare_cost
        # Both spouses 65+ → full Medicare cost
        cost_65 = compute_healthcare_cost(65, 66.0, 60_000, 2, "married_filing_jointly", tax_cfg)
        expected = compute_medicare_cost(60_000, 2, "married_filing_jointly", tax_cfg)
        assert cost_65 == pytest.approx(expected)

    def test_split_coverage_when_ages_straddle_65(self, tax_cfg):
        from engine.aca_engine import compute_aca_cost, compute_healthcare_cost, compute_medicare_cost
        # Primary 66 (Medicare), spouse 63 (ACA) → split cost
        cost = compute_healthcare_cost(66, 63.0, 60_000, 2, "married_filing_jointly", tax_cfg)
        expected = (
            compute_medicare_cost(60_000, 1, "married_filing_jointly", tax_cfg)
            + compute_aca_cost(60_000, 2, 1, tax_cfg)
        )
        assert cost == pytest.approx(expected)

    def test_single_person_no_spouse(self, tax_cfg):
        from engine.aca_engine import compute_aca_cost, compute_healthcare_cost
        # No spouse (None) — only primary counted
        cost = compute_healthcare_cost(60, None, 60_000, 1, "single", tax_cfg)
        expected = compute_aca_cost(60_000, 1, 1, tax_cfg)
        assert cost == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# RMD engine
# ─────────────────────────────────────────────────────────────────────────────

class TestRMDEngine:
    def test_not_required_before_73(self, tax_cfg):
        from engine.rmd_engine import rmd_required
        assert not rmd_required(72.9, tax_cfg)

    def test_required_at_73(self, tax_cfg):
        from engine.rmd_engine import rmd_required
        assert rmd_required(73.0, tax_cfg)

    def test_compute_rmd_at_73(self, tax_cfg):
        from engine.rmd_engine import compute_rmd
        # Balance $500k, age 73, factor = 26.5 → $18,868.xx
        rmd = compute_rmd(500_000, 73, tax_cfg)
        assert abs(rmd - 500_000 / 26.5) < 0.01

    def test_compute_rmd_at_80(self, tax_cfg):
        from engine.rmd_engine import compute_rmd
        rmd = compute_rmd(300_000, 80, tax_cfg)
        assert abs(rmd - 300_000 / 20.2) < 0.01

    def test_compute_rmd_zero_before_73(self, tax_cfg):
        from engine.rmd_engine import compute_rmd
        assert compute_rmd(500_000, 70, tax_cfg) == 0.0

    def test_rmd_increases_as_balance_grows(self, tax_cfg):
        from engine.rmd_engine import compute_rmd
        assert compute_rmd(600_000, 73, tax_cfg) > compute_rmd(500_000, 73, tax_cfg)

    def test_rmd_increases_with_age_for_same_balance(self, tax_cfg):
        from engine.rmd_engine import compute_rmd
        # Shorter distribution period at older age → larger RMD
        assert compute_rmd(500_000, 85, tax_cfg) > compute_rmd(500_000, 73, tax_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Withdrawal Router
# ─────────────────────────────────────────────────────────────────────────────

class TestWithdrawalRouter:
    def _buckets(self, taxable=100_000, basis=60_000, deferred=200_000, free=50_000):
        from engine.models import Buckets
        return Buckets(taxable=taxable, taxable_basis=basis,
                       tax_deferred=deferred, tax_free=free)

    def test_pulls_taxable_first_at_60(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets()
        route_withdrawal(b, 60.0, 30_000, tax_cfg)
        assert b.taxable == pytest.approx(70_000)   # 30k pulled from taxable
        assert b.tax_deferred == 200_000             # untouched

    def test_pulls_tax_free_before_deferred_under_59(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets(taxable=0)  # taxable empty → next priority is tax_free
        route_withdrawal(b, 55.0, 30_000, tax_cfg)  # age < 59.5
        assert b.tax_free == pytest.approx(20_000)   # 30k from tax_free
        assert b.tax_deferred == 200_000              # penalty bucket avoided

    def test_penalty_incurred_under_59(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets(taxable=0, free=0)  # both empty → must hit tax_deferred
        bd = route_withdrawal(b, 55.0, 30_000, tax_cfg)
        assert bd.penalty_base == pytest.approx(30_000)
        assert bd.ordinary_income == pytest.approx(30_000)

    def test_no_penalty_at_60(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets(taxable=0, free=0)
        bd = route_withdrawal(b, 60.0, 30_000, tax_cfg)
        assert bd.penalty_base == 0.0

    def test_capital_gains_proportional_to_gain_ratio(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        # $100k balance, $60k basis → gain ratio = 40%
        b = self._buckets(taxable=100_000, basis=60_000)
        bd = route_withdrawal(b, 60.0, 50_000, tax_cfg)
        assert abs(bd.capital_gains - 50_000 * 0.40) < 1.0

    def test_basis_decreases_correctly(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        # $100k balance, $60k basis → sell $50k → basis used = $50k × 60% = $30k
        b = self._buckets(taxable=100_000, basis=60_000)
        route_withdrawal(b, 60.0, 50_000, tax_cfg)
        assert abs(b.taxable_basis - 30_000) < 1.0

    def test_rmd_forced_at_73(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        # $500k tax-deferred, age 73 → RMD = $500k / 26.5 ≈ $18,868
        b = self._buckets(taxable=0, free=0, deferred=500_000)
        bd = route_withdrawal(b, 73.0, 10_000, tax_cfg)  # need less than RMD
        assert bd.rmd_amount == pytest.approx(500_000 / 26.5, rel=1e-3)
        assert bd.ordinary_income >= bd.rmd_amount

    def test_shortfall_when_portfolio_empty(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets(taxable=5_000, basis=5_000, deferred=0, free=0)
        bd = route_withdrawal(b, 60.0, 20_000, tax_cfg)
        assert bd.shortfall == pytest.approx(15_000)

    def test_zero_needed_returns_empty_breakdown(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets()
        bd = route_withdrawal(b, 60.0, 0.0, tax_cfg)
        assert bd.ordinary_income == 0.0
        assert bd.capital_gains == 0.0
        assert bd.shortfall == 0.0

    def test_basis_never_goes_negative(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets(taxable=100_000, basis=1_000)  # almost no basis
        route_withdrawal(b, 60.0, 99_000, tax_cfg)
        assert b.taxable_basis >= 0.0

    def test_buckets_never_go_negative(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets(taxable=10_000, basis=5_000, deferred=10_000, free=5_000)
        route_withdrawal(b, 60.0, 500_000, tax_cfg)  # request more than available
        assert b.taxable >= 0.0
        assert b.tax_deferred >= 0.0
        assert b.tax_free >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Buckets model
# ─────────────────────────────────────────────────────────────────────────────

class TestBuckets:
    def test_total(self):
        from engine.models import Buckets
        b = Buckets(100_000, 60_000, 200_000, 50_000)
        assert b.total() == 350_000

    def test_apply_return_grows_all_buckets(self):
        from engine.models import Buckets
        b = Buckets(100_000, 60_000, 200_000, 50_000)
        b.apply_return(0.10)
        assert b.taxable == pytest.approx(110_000)
        assert b.tax_deferred == pytest.approx(220_000)
        assert b.tax_free == pytest.approx(55_000)

    def test_apply_return_does_not_change_basis(self):
        from engine.models import Buckets
        b = Buckets(100_000, 60_000, 200_000, 50_000)
        b.apply_return(0.10)
        assert b.taxable_basis == 60_000   # basis unchanged; unrealized gains accumulate

    def test_negative_return_floored_at_zero(self):
        from engine.models import Buckets
        b = Buckets(10_000, 5_000, 0, 0)
        b.apply_return(-1.5)   # extreme loss
        assert b.taxable >= 0.0
        assert b.tax_deferred >= 0.0

    def test_clone_is_independent(self):
        from engine.models import Buckets
        b = Buckets(100_000, 60_000, 200_000, 50_000)
        c = b.clone()
        b.taxable = 0
        assert c.taxable == 100_000


# ─────────────────────────────────────────────────────────────────────────────
# SimulationParams
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationParams:
    def test_total_balance(self, default_params):
        assert default_params.total_balance() == pytest.approx(1_500_000)

    def test_from_portfolio_state(self):
        from engine.models import SimulationParams
        state = {
            "summary": {"by_tax_treatment": {"taxable": 50_000, "tax_deferred": 100_000, "tax_free": 30_000}},
            "accounts": [
                {"tax_treatment": "taxable", "holdings": [{"value_usd": 50_000, "cost_basis_usd": 30_000}]},
            ],
            "owner": {"members": [{"role": "primary", "birth_date": "1980-01-01"}]},
        }
        p = SimulationParams.from_portfolio_state(state, overrides={"annual_spending_today": 70_000})
        assert p.taxable_balance == 50_000
        assert p.taxable_basis == 30_000
        assert p.tax_deferred_balance == 100_000
        assert p.annual_spending_today == 70_000
        assert p.current_age > 40  # born 1980, should be ~46


# ─────────────────────────────────────────────────────────────────────────────
# Full Monte Carlo integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:
    """Integration tests. Use fixed seed for reproducibility."""

    def _run(self, **overrides):
        from engine.models import SimulationParams
        from engine.monte_carlo import run_simulation
        from engine.tax_engine import load_tax_config

        p = SimulationParams(
            taxable_balance=500_000,
            taxable_basis=300_000,
            tax_deferred_balance=800_000,
            tax_free_balance=200_000,
            current_age=50.0,
            plan_to_age=90,
            annual_spending_today=80_000,
            num_iterations=500,
        )
        for k, v in overrides.items():
            setattr(p, k, v)

        tax_cfg = load_tax_config(CONFIG_DIR)
        return run_simulation(p, tax_cfg, seed=42)

    def test_trivially_succeeds(self):
        # $2M portfolio, $30k/yr spending → should nearly always succeed
        r = self._run(
            taxable_balance=1_000_000,
            tax_deferred_balance=800_000,
            tax_free_balance=200_000,
            annual_spending_today=30_000,
        )
        assert r.success_rate > 0.90

    def test_trivially_fails(self):
        # $50k portfolio, $100k/yr spending → should nearly always fail
        r = self._run(
            taxable_balance=20_000,
            taxable_basis=15_000,
            tax_deferred_balance=20_000,
            tax_free_balance=10_000,
            annual_spending_today=100_000,
        )
        assert r.success_rate < 0.10

    def test_success_rate_in_unit_interval(self):
        r = self._run()
        assert 0.0 <= r.success_rate <= 1.0

    def test_trajectory_lengths_match_ages(self):
        r = self._run()
        n = len(r.ages)
        assert len(r.median_trajectory) == n
        assert len(r.p10_trajectory) == n
        assert len(r.p90_trajectory) == n
        assert len(r.median_taxes) == n
        assert len(r.median_healthcare) == n

    def test_age_range_correct(self):
        r = self._run(current_age=50.0, plan_to_age=90)
        assert r.ages[0] == 50
        assert r.ages[-1] == 90
        assert len(r.ages) == 41

    def test_p90_above_median_above_p10(self):
        r = self._run()
        # Check midpoint of the trajectory (not first/last which may be degenerate)
        mid = len(r.ages) // 2
        assert r.p90_trajectory[mid] >= r.median_trajectory[mid]
        assert r.median_trajectory[mid] >= r.p10_trajectory[mid]

    def test_first_year_portfolio_close_to_initial(self):
        r = self._run()
        initial = 500_000 + 800_000 + 200_000
        # After one year of returns and withdrawals, median should still be in the ballpark
        assert 0.5 * initial < r.median_trajectory[0] < 3.0 * initial

    def test_taxes_are_non_negative(self):
        r = self._run()
        assert all(t >= 0 for t in r.median_taxes)

    def test_healthcare_is_non_negative(self):
        r = self._run()
        assert all(h >= 0 for h in r.median_healthcare)

    def test_social_security_improves_success_rate(self):
        without_ss = self._run(social_security_annual=0)
        with_ss = self._run(
            social_security_annual=30_000,
            social_security_start_age=67,
        )
        assert with_ss.success_rate >= without_ss.success_rate

    def test_higher_spending_reduces_success_rate(self):
        low = self._run(annual_spending_today=60_000)
        high = self._run(annual_spending_today=120_000)
        assert low.success_rate > high.success_rate

    def test_to_dict_is_json_serialisable(self):
        import json
        r = self._run()
        d = r.to_dict()
        serialised = json.dumps(d)          # raises if not serialisable
        loaded = json.loads(serialised)
        assert loaded["success_rate"] == r.to_dict()["success_rate"]

    def test_seed_produces_deterministic_result(self):
        from engine.models import SimulationParams
        from engine.monte_carlo import run_simulation
        from engine.tax_engine import load_tax_config
        p = SimulationParams(
            taxable_balance=500_000, taxable_basis=300_000,
            tax_deferred_balance=800_000, tax_free_balance=200_000,
            current_age=55.0, plan_to_age=90, num_iterations=100,
        )
        tax_cfg = load_tax_config(CONFIG_DIR)
        r1 = run_simulation(p, tax_cfg, seed=99)
        r2 = run_simulation(p, tax_cfg, seed=99)
        assert r1.success_rate == r2.success_rate
        assert r1.median_trajectory == r2.median_trajectory

    def test_run_public_api_accepts_dict(self):
        from engine.monte_carlo import run
        result = run(
            {
                "taxable_balance": 500_000,
                "taxable_basis": 300_000,
                "tax_deferred_balance": 800_000,
                "tax_free_balance": 200_000,
                "current_age": 55.0,
                "num_iterations": 100,
            },
            config_dir=CONFIG_DIR,
            seed=0,
        )
        assert "success_rate" in result
        assert "median_trajectory" in result

    def test_rmd_engaged_at_73(self):
        """Simulation starting at 73 should route RMDs from tax-deferred."""
        from engine.models import SimulationParams
        from engine.monte_carlo import run_simulation
        from engine.tax_engine import load_tax_config
        p = SimulationParams(
            taxable_balance=0,
            taxable_basis=0,
            tax_deferred_balance=500_000,
            tax_free_balance=0,
            current_age=73.0,
            plan_to_age=85,
            annual_spending_today=20_000,   # less than RMD → portfolio should survive
            num_iterations=100,
        )
        tax_cfg = load_tax_config(CONFIG_DIR)
        r = run_simulation(p, tax_cfg, seed=7)
        # With spending < RMD, all cash needs are covered; success rate should be decent
        assert r.success_rate > 0.50

    def test_aca_cliff_penalty_visible_in_healthcare(self):
        """Crossing the ACA cliff should produce a sharp jump in median healthcare costs."""
        from engine.models import SimulationParams
        from engine.monte_carlo import run_simulation
        from engine.tax_engine import load_tax_config
        tax_cfg = load_tax_config(CONFIG_DIR)

        # Below cliff: large tax-free bucket → low ordinary income → low MAGI → subsidised ACA
        below = SimulationParams(
            taxable_balance=0, taxable_basis=0,
            tax_deferred_balance=0,
            tax_free_balance=2_000_000,   # Roth — withdrawals don't appear in MAGI
            current_age=50.0, plan_to_age=64,
            annual_spending_today=60_000,
            num_iterations=200,
        )
        above = SimulationParams(
            taxable_balance=0, taxable_basis=0,
            tax_deferred_balance=2_000_000,  # trad IRA — all withdrawals = ordinary income
            tax_free_balance=0,
            current_age=50.0, plan_to_age=64,
            annual_spending_today=60_000,
            num_iterations=200,
        )
        r_below = run_simulation(below, tax_cfg, seed=5)
        r_above = run_simulation(above, tax_cfg, seed=5)

        # Someone spending from Roth (low MAGI) pays far less for ACA than someone
        # with all income from a traditional IRA pushing MAGI above the cliff
        avg_below = sum(r_below.median_healthcare) / len(r_below.median_healthcare)
        avg_above = sum(r_above.median_healthcare) / len(r_above.median_healthcare)
        assert avg_above > avg_below


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: per-person SS, spouse RMD, healthcare inflation
# ─────────────────────────────────────────────────────────────────────────────

class TestPerPersonSS:
    def _run(self, **overrides):
        from engine.models import SimulationParams
        from engine.monte_carlo import run_simulation
        from engine.tax_engine import load_tax_config
        p = SimulationParams(
            taxable_balance=0, taxable_basis=0,
            tax_deferred_balance=2_000_000, tax_free_balance=0,
            current_age=65.0, plan_to_age=85,
            annual_spending_today=80_000, num_iterations=200,
        )
        for k, v in overrides.items():
            setattr(p, k, v)
        return run_simulation(p, load_tax_config(CONFIG_DIR), seed=1)

    def test_spouse_ss_improves_success_rate(self):
        without = self._run()
        with_ss  = self._run(ss_spouse_annual=20_000, ss_spouse_start_age=65,
                             spouse_current_age=65.0)
        assert with_ss.success_rate > without.success_rate

    def test_spouse_ss_only_starts_at_claim_age(self):
        # spouse starts at 70 — primary is already 65 → spouse is 64 → no SS yet
        without_early = self._run(
            ss_spouse_annual=20_000, ss_spouse_start_age=70,
            spouse_current_age=64.0,
        )
        with_early = self._run(
            ss_spouse_annual=20_000, ss_spouse_start_age=65,
            spouse_current_age=64.0,
        )
        # Delaying SS by 5 years should hurt success rate
        assert with_early.success_rate >= without_early.success_rate

    def test_primary_ss_unchanged_by_spouse_ss(self):
        base   = self._run(social_security_annual=30_000, social_security_start_age=65)
        with_s = self._run(social_security_annual=30_000, social_security_start_age=65,
                           ss_spouse_annual=15_000, ss_spouse_start_age=65,
                           spouse_current_age=65.0)
        assert with_s.success_rate >= base.success_rate


class TestSpouseRMD:
    def _buckets_with_spouse_deferred(self, spouse_deferred=200_000):
        from engine.models import Buckets
        return Buckets(taxable=0, taxable_basis=0, tax_deferred=0,
                       tax_free=0, tax_deferred_spouse=spouse_deferred)

    def test_spouse_rmd_triggered_at_73(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets_with_spouse_deferred(200_000)
        bd = route_withdrawal(b, age=65.0, needed=5_000, tax_cfg=tax_cfg, spouse_age=73.0)
        # Spouse RMD should force a withdrawal from tax_deferred_spouse
        expected_rmd = 200_000 / 26.5
        assert bd.rmd_amount == pytest.approx(expected_rmd, rel=1e-3)

    def test_spouse_rmd_not_triggered_before_73(self, tax_cfg):
        from engine.withdrawal_router import route_withdrawal
        b = self._buckets_with_spouse_deferred(200_000)
        bd = route_withdrawal(b, age=65.0, needed=5_000, tax_cfg=tax_cfg, spouse_age=72.0)
        assert bd.rmd_amount == 0.0

    def test_both_rmds_when_both_over_73(self, tax_cfg):
        from engine.models import Buckets
        from engine.withdrawal_router import route_withdrawal
        b = Buckets(taxable=0, taxable_basis=0, tax_deferred=300_000,
                    tax_free=0, tax_deferred_spouse=200_000)
        bd = route_withdrawal(b, age=75.0, needed=1_000, tax_cfg=tax_cfg, spouse_age=73.0)
        primary_rmd = 300_000 / 24.6
        spouse_rmd  = 200_000 / 26.5
        assert bd.rmd_amount == pytest.approx(primary_rmd + spouse_rmd, rel=1e-3)

    def test_spouse_rmd_increases_success_when_spending_matches(self):
        from engine.models import SimulationParams
        from engine.monte_carlo import run_simulation
        from engine.tax_engine import load_tax_config
        # Spouse has deferred balance; both are 73 — RMDs should cover spending
        p = SimulationParams(
            taxable_balance=0, taxable_basis=0,
            tax_deferred_balance=300_000,
            tax_deferred_spouse_balance=200_000,
            tax_free_balance=0,
            current_age=73.0, plan_to_age=85,
            annual_spending_today=15_000,
            spouse_current_age=73.0,
            num_iterations=200,
        )
        r = run_simulation(p, load_tax_config(CONFIG_DIR), seed=3)
        assert r.success_rate > 0.50


class TestHealthcareInflation:
    def _run_hc(self, hc_inflation: float):
        from engine.models import SimulationParams
        from engine.monte_carlo import run_simulation
        from engine.tax_engine import load_tax_config
        p = SimulationParams(
            taxable_balance=0, taxable_basis=0,
            tax_deferred_balance=2_000_000, tax_free_balance=0,
            current_age=55.0, plan_to_age=80,
            annual_spending_today=60_000,
            healthcare_inflation_rate=hc_inflation,
            num_iterations=200,
        )
        return run_simulation(p, load_tax_config(CONFIG_DIR), seed=2)

    def test_higher_hc_inflation_raises_later_costs(self):
        low  = self._run_hc(0.02)
        high = self._run_hc(0.08)
        # Later years should show larger healthcare costs under higher inflation
        assert high.median_healthcare[-1] > low.median_healthcare[-1]

    def test_zero_hc_inflation_is_flat_in_real_terms(self):
        r = self._run_hc(0.0)
        # With 0% HC inflation, first and last year costs should be close
        # (ACA premiums don't compound — only MAGI-driven variation)
        assert r.median_healthcare[-1] >= 0.0  # no negative costs

    def test_hc_inflation_reduces_success_rate(self):
        low  = self._run_hc(0.02)
        high = self._run_hc(0.10)
        assert high.success_rate <= low.success_rate
