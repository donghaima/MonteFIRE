"""Tests for the ETL parser — no network, no external state."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from etl.parser import (
    apply_transform,
    build_summary,
    load_adapters,
    load_household,
    resolve_account_type,
    resolve_asset_class,
    resolve_owner,
    run_etl,
)

CONFIG_DIR = Path(__file__).parent.parent / "config"
FIXTURES_RAW = Path(__file__).parent / "fixtures" / "raw"


# ---------------------------------------------------------------------------
# Unit: apply_transform
# ---------------------------------------------------------------------------

class TestApplyTransform:
    def test_strips_dollar_and_comma(self):
        t = {"strip_chars": "[$,]", "cast": "float", "null_values": ["--"]}
        assert apply_transform("$24,100.00", t) == 24100.0

    def test_null_sentinel_returns_none(self):
        t = {"strip_chars": "[$,]", "cast": "float", "null_values": ["--", "N/A"]}
        assert apply_transform("--", t) is None
        assert apply_transform("N/A", t) is None

    def test_empty_string_returns_none(self):
        t = {"strip_chars": "[$,]", "cast": "float", "null_values": [""]}
        assert apply_transform("", t) is None

    def test_plain_number_no_strip(self):
        t = {"cast": "float", "null_values": []}
        assert apply_transform("320.15", t) == 320.15

    def test_bad_value_returns_none(self):
        t = {"cast": "float", "null_values": []}
        assert apply_transform("N/A", t) is None


# ---------------------------------------------------------------------------
# Unit: resolve_account_type
# ---------------------------------------------------------------------------

class TestResolveAccountType:
    def setup_method(self):
        cfg = load_adapters(CONFIG_DIR)
        self.rules = cfg["adapters"]["fidelity_brokerage"]["account_type_rules"]

    def test_roth_401k(self):
        at, tt = resolve_account_type("Fidelity Roth 401k - ABC", self.rules)
        assert at == "roth_401k"
        assert tt == "tax_free"

    def test_traditional_401k(self):
        at, tt = resolve_account_type("Fidelity 401(k) - ABC", self.rules)
        assert at == "401k"
        assert tt == "tax_deferred"

    def test_roth_ira(self):
        at, tt = resolve_account_type("Fidelity Roth IRA - ABC", self.rules)
        assert at == "roth_ira"
        assert tt == "tax_free"

    def test_traditional_ira(self):
        at, tt = resolve_account_type("Fidelity IRA - ABC", self.rules)
        assert at == "ira"
        assert tt == "tax_deferred"

    def test_brokerage_fallback(self):
        at, tt = resolve_account_type("Individual - TOD - ABC", self.rules)
        assert at == "brokerage"
        assert tt == "taxable"


# ---------------------------------------------------------------------------
# Unit: resolve_asset_class
# ---------------------------------------------------------------------------

class TestResolveAssetClass:
    def setup_method(self):
        cfg = load_adapters(CONFIG_DIR)
        self.fidelity = cfg["adapters"]["fidelity_brokerage"]
        self.empower  = cfg["adapters"]["empower_retirement"]
        self.heuristics = cfg["heuristics"]

    def test_ticker_tag_wins(self):
        ac = resolve_asset_class("SPAXX", "", None, self.fidelity, self.heuristics)
        assert ac == "cash_equivalent"

    def test_asset_class_map_empower(self):
        ac = resolve_asset_class("VIIIX", "Vanguard Inst Index", "Domestic Equity", self.empower, self.heuristics)
        assert ac == "us_equity"

    def test_heuristic_fallback(self):
        ac = resolve_asset_class("FZROX", "", None, self.fidelity, self.heuristics)
        assert ac == "us_equity"

    def test_unknown_ticker_falls_to_other(self):
        ac = resolve_asset_class("ZZZZZ", "Unknown Fund", None, self.fidelity, self.heuristics)
        assert ac == "other"

    def test_bond_heuristic(self):
        ac = resolve_asset_class("BND", "", None, self.fidelity, self.heuristics)
        assert ac == "us_bond"


# ---------------------------------------------------------------------------
# Unit: resolve_owner
# ---------------------------------------------------------------------------

class TestResolveOwner:
    def setup_method(self):
        self.owner_map = load_household(CONFIG_DIR).get("account_owner_map", [])

    def test_primary_fallback(self):
        owner = resolve_owner("Z12345678", "Individual - TOD", self.owner_map)
        assert owner == "member-primary"

    def test_spouse_match(self):
        owner = resolve_owner("spouse-acct-001", "Spouse IRA", self.owner_map)
        assert owner == "member-spouse"


# ---------------------------------------------------------------------------
# Unit: build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_totals_correct(self):
        accounts = [
            {"tax_treatment": "taxable",      "balance_usd": 10_000.0, "holdings": [{"asset_class": "us_equity", "value_usd": 10_000.0}]},
            {"tax_treatment": "tax_deferred", "balance_usd": 20_000.0, "holdings": [{"asset_class": "us_bond",   "value_usd": 20_000.0}]},
            {"tax_treatment": "tax_free",     "balance_usd": 5_000.0,  "holdings": [{"asset_class": "us_equity", "value_usd": 5_000.0}]},
        ]
        s = build_summary(accounts)
        assert s["total_net_worth_usd"] == 35_000.0
        assert s["by_tax_treatment"]["taxable"]      == 10_000.0
        assert s["by_tax_treatment"]["tax_deferred"] == 20_000.0
        assert s["by_tax_treatment"]["tax_free"]     == 5_000.0
        assert s["by_asset_class"]["us_equity"]      == 15_000.0
        assert s["by_asset_class"]["us_bond"]        == 20_000.0


# ---------------------------------------------------------------------------
# Integration: full run_etl with fixture CSVs
# ---------------------------------------------------------------------------

class TestRunEtl:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "portfolio_state.json"

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _run(self, raw_dir=FIXTURES_RAW):
        run_etl(raw_dir=raw_dir, config_dir=CONFIG_DIR, out_path=self.out)
        return json.loads(self.out.read_text())

    def test_output_file_created(self):
        self._run()
        assert self.out.exists()

    def test_schema_version_present(self):
        state = self._run()
        assert state["schema_version"] == "1.0"

    def test_source_files_recorded(self):
        state = self._run()
        names = [sf["filename"] for sf in state["source_files"]]
        assert any("Portfolio_Positions" in n for n in names)
        assert any("Empower_Holdings" in n for n in names)

    def test_expected_account_count(self):
        # Fidelity fixture: 3 accounts; Empower fixture: 2 accounts
        state = self._run()
        assert len(state["accounts"]) == 5

    def test_fidelity_brokerage_account_is_taxable(self):
        state = self._run()
        brokerage = next(a for a in state["accounts"] if a["account_type"] == "brokerage")
        assert brokerage["tax_treatment"] == "taxable"

    def test_fidelity_401k_is_tax_deferred(self):
        state = self._run()
        k401 = next(a for a in state["accounts"] if a["account_type"] == "401k" and a["institution"] == "Fidelity")
        assert k401["tax_treatment"] == "tax_deferred"

    def test_roth_accounts_are_tax_free(self):
        state = self._run()
        roths = [a for a in state["accounts"] if "roth" in a["account_type"]]
        assert len(roths) >= 2
        for r in roths:
            assert r["tax_treatment"] == "tax_free"

    def test_spaxx_classified_as_cash_equivalent(self):
        state = self._run()
        brokerage = next(a for a in state["accounts"] if a["account_type"] == "brokerage")
        spaxx = next(h for h in brokerage["holdings"] if h["ticker_or_name"] == "SPAXX")
        assert spaxx["asset_class"] == "cash_equivalent"

    def test_cost_basis_only_on_taxable(self):
        state = self._run()
        for acct in state["accounts"]:
            for h in acct["holdings"]:
                if "cost_basis_usd" in h:
                    assert acct["tax_treatment"] == "taxable", (
                        f"cost_basis_usd on non-taxable account {acct['id']}"
                    )

    def test_summary_net_worth_matches_sum(self):
        state = self._run()
        expected = round(sum(a["balance_usd"] for a in state["accounts"]), 2)
        assert state["summary"]["total_net_worth_usd"] == expected

    def test_summary_by_tax_treatment_sums_to_total(self):
        state = self._run()
        tt = state["summary"]["by_tax_treatment"]
        total = round(tt["taxable"] + tt["tax_deferred"] + tt["tax_free"], 2)
        assert total == state["summary"]["total_net_worth_usd"]

    def test_owner_block_present(self):
        state = self._run()
        assert "household_id" in state["owner"]
        assert len(state["owner"]["members"]) >= 1

    def test_no_unmatched_csv_crashes(self):
        tmp_raw = self.tmp / "raw"
        tmp_raw.mkdir()
        (tmp_raw / "unknown_bank_export.csv").write_text("col1,col2\nval1,val2\n")
        state = self._run(raw_dir=tmp_raw)
        assert state["accounts"] == []

    def test_empower_stable_value_is_cash_equivalent(self):
        state = self._run()
        empower_trad = next(
            a for a in state["accounts"]
            if a["institution"] == "Empower" and a["account_type"] == "401k"
        )
        sv = next(h for h in empower_trad["holdings"] if "Stable" in h["ticker_or_name"])
        assert sv["asset_class"] == "cash_equivalent"

    def test_zero_value_rows_excluded(self):
        state = self._run()
        for acct in state["accounts"]:
            for h in acct["holdings"]:
                assert h["value_usd"] > 0
