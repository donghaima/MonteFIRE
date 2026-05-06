"""
ETL pipeline: raw bank/brokerage CSVs → portfolio_state.json

Entry point:
    python -m etl.parser [--raw-dir RAW] [--config CONFIG] [--out OUT] [--household HOUSEHOLD]

The script:
  1. Reads bank_adapters.yaml and household.yaml from --config dir.
  2. Scans --raw-dir for CSV files, matching each against adapter filename_pattern.
  3. Parses matched CSVs using the adapter's column_map, transforms, and
     account_type_rules.
  4. Resolves asset_class via ticker_tags → asset_class_map → heuristics.
  5. Groups holdings into Account objects, computing per-account balance_usd.
  6. Writes portfolio_state.json to --out, including pre-aggregated summary rollups.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"

VALID_ACCOUNT_TYPES = {
    "brokerage", "401k", "roth_401k", "ira", "roth_ira",
    "hsa", "529", "checking", "savings", "other",
}
VALID_TAX_TREATMENTS = {"taxable", "tax_deferred", "tax_free"}
VALID_ASSET_CLASSES = {
    "us_equity", "intl_equity", "us_bond", "intl_bond",
    "real_estate", "cash_equivalent", "crypto", "other",
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_adapters(config_dir: Path) -> dict:
    raw = load_yaml(config_dir / "bank_adapters.yaml")
    adapters = {a["id"]: a for a in raw["adapters"]}
    heuristics = raw.get("asset_class_heuristics", [])
    return {"adapters": adapters, "heuristics": heuristics}


def load_household(config_dir: Path) -> dict:
    return load_yaml(config_dir / "household.yaml")

# ---------------------------------------------------------------------------
# Value transforms
# ---------------------------------------------------------------------------

def apply_transform(raw_value: str, transform: dict) -> float | None:
    """Clean and cast a raw CSV cell value per the adapter transform spec."""
    if raw_value is None:
        return None

    value = str(raw_value).strip()

    null_values = transform.get("null_values", [])
    if value in null_values:
        return None

    strip_pattern = transform.get("strip_chars")
    if strip_pattern:
        value = re.sub(strip_pattern, "", value)

    value = value.strip()
    if not value:
        return None

    cast = transform.get("cast", "str")
    if cast == "float":
        try:
            return float(value.replace(",", ""))
        except ValueError:
            log.warning("Could not cast %r to float — treating as null", raw_value)
            return None

    return value

# ---------------------------------------------------------------------------
# Asset class resolution
# ---------------------------------------------------------------------------

def resolve_asset_class(
    ticker: str,
    description: str,
    asset_class_raw: str | None,
    adapter: dict,
    global_heuristics: list[dict],
) -> str:
    # 1. Per-adapter ticker_tags table (exact, case-insensitive)
    ticker_tags: dict = adapter.get("ticker_tags", {})
    if ticker:
        hit = ticker_tags.get(ticker.upper())
        if hit:
            return hit

    # 2. asset_class_map (institution-provided label, e.g. Empower)
    if asset_class_raw:
        asset_class_map: dict = adapter.get("asset_class_map", {})
        hit = asset_class_map.get(asset_class_raw.strip())
        if hit:
            return hit

    # 3. Global regex heuristics — checked against ticker then description
    probe = ticker or description or ""
    for rule in global_heuristics:
        if re.search(rule["pattern"], probe):
            return rule["asset_class"]

    return "other"

# ---------------------------------------------------------------------------
# Account type / tax treatment resolution
# ---------------------------------------------------------------------------

def resolve_account_type(account_name: str, rules: list[dict]) -> tuple[str, str]:
    """Return (account_type, tax_treatment) by testing account_name against ordered rules."""
    for rule in rules:
        if re.search(rule["match"], account_name):
            return rule["account_type"], rule["tax_treatment"]
    return "brokerage", "taxable"


def resolve_owner(account_number: str, account_name: str, owner_map: list[dict]) -> str:
    probe = f"{account_number} {account_name}".lower()
    for rule in owner_map:
        if re.search(rule["match"], probe, re.IGNORECASE):
            return rule["owner_id"]
    return "member-primary"

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _stable_account_id(institution: str, account_number: str, account_name: str) -> str:
    """Generate a deterministic slug so re-runs produce the same IDs."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{institution}-{account_number}-{account_name}".lower())
    return slug.strip("-")


def parse_csv(file_path: Path, adapter: dict) -> list[dict]:
    """
    Return a list of raw holding dicts (one per non-empty data row).
    All values are strings at this point; transforms are applied next.
    """
    import csv

    header_row_idx: int = adapter.get("header_row", 0)
    skip_footer: int = adapter.get("skip_footer_rows", 0)

    with file_path.open(newline="", encoding="utf-8-sig") as f:
        all_lines = f.readlines()

    # Strip footer
    if skip_footer:
        all_lines = all_lines[:-skip_footer]

    # The header line is at header_row_idx; data starts after it
    header_line = all_lines[header_row_idx]
    data_lines = all_lines[header_row_idx + 1:]

    reader = csv.DictReader([header_line] + data_lines)
    return [row for row in reader if any(v.strip() for v in row.values())]


def extract_holdings_from_rows(
    rows: list[dict],
    adapter: dict,
    global_heuristics: list[dict],
    owner_map: list[dict],
    as_of_date: str,
) -> dict[str, dict]:
    """
    Group CSV rows into accounts keyed by stable account ID.
    Returns {account_id: account_dict}.
    """
    col = adapter["column_map"]
    transforms = adapter.get("transforms", {})
    account_type_rules = adapter.get("account_type_rules", [])
    defaults = adapter.get("account_defaults", {})

    accounts: dict[str, dict] = {}

    for row in rows:
        # -- Extract raw cell values via column_map --------------------------
        def get(canonical: str) -> str | None:
            csv_col = col.get(canonical)
            return row.get(csv_col, "").strip() if csv_col else None

        account_name   = get("account_name") or ""
        account_number = get("account_number") or ""
        ticker         = (get("ticker") or "").upper()
        description    = get("description") or ""
        asset_class_raw = get("asset_class_raw")

        # Skip rows with no meaningful value
        raw_value = get("current_value")
        value_usd = apply_transform(raw_value, transforms.get("current_value", {}))
        if value_usd is None or value_usd == 0.0:
            continue

        # -- Account identity ------------------------------------------------
        account_id = _stable_account_id(defaults.get("institution", ""), account_number, account_name)

        if account_id not in accounts:
            account_type, tax_treatment = resolve_account_type(account_name, account_type_rules)
            owner_id = resolve_owner(account_number, account_name, owner_map)

            accounts[account_id] = {
                "id":             account_id,
                "institution":    defaults.get("institution", ""),
                "adapter_id":     defaults.get("adapter_id", adapter["id"]),
                "account_type":   account_type,
                "tax_treatment":  tax_treatment,
                "owner_id":       owner_id,
                "early_withdrawal_penalty_applies": tax_treatment == "tax_deferred",
                "as_of_date":     as_of_date,
                "balance_usd":    0.0,
                "holdings":       [],
            }

        # -- Holding ---------------------------------------------------------
        shares    = apply_transform(get("quantity"),   transforms.get("quantity", {}))
        price_usd = apply_transform(get("last_price"), transforms.get("last_price", {}))
        cost_basis = apply_transform(get("cost_basis"), transforms.get("cost_basis", {}))

        asset_class = resolve_asset_class(
            ticker, description, asset_class_raw, adapter, global_heuristics
        )

        holding: dict[str, Any] = {
            "ticker_or_name": ticker or description,
            "asset_class":    asset_class,
            "value_usd":      round(value_usd, 2),
        }
        if shares is not None:
            holding["shares"] = round(shares, 6)
        if price_usd is not None:
            holding["price_usd"] = round(price_usd, 4)
        if cost_basis is not None and accounts[account_id]["tax_treatment"] == "taxable":
            holding["cost_basis_usd"] = round(cost_basis, 2)

        accounts[account_id]["holdings"].append(holding)
        accounts[account_id]["balance_usd"] = round(
            accounts[account_id]["balance_usd"] + value_usd, 2
        )

    return accounts

# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def build_summary(accounts: list[dict]) -> dict:
    by_tax: dict[str, float] = {"taxable": 0.0, "tax_deferred": 0.0, "tax_free": 0.0}
    by_asset: dict[str, float] = {}

    for acct in accounts:
        treatment = acct["tax_treatment"]
        by_tax[treatment] = round(by_tax.get(treatment, 0.0) + acct["balance_usd"], 2)

        for h in acct["holdings"]:
            ac = h["asset_class"]
            by_asset[ac] = round(by_asset.get(ac, 0.0) + h["value_usd"], 2)

    total = round(sum(by_tax.values()), 2)
    return {
        "total_net_worth_usd": total,
        "by_tax_treatment":    by_tax,
        "by_asset_class":      by_asset,
    }

# ---------------------------------------------------------------------------
# Household member schema shape
# ---------------------------------------------------------------------------

def build_owner_block(household: dict) -> dict:
    return {
        "household_id": household["household_id"],
        "members": [
            {
                "id":         m["id"],
                "name":       m["name"],
                "birth_date": m["birth_date"],
                "role":       m["role"],
            }
            for m in household["members"]
        ],
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_etl(raw_dir: Path, config_dir: Path, out_path: Path) -> None:
    cfg       = load_adapters(config_dir)
    household = load_household(config_dir)

    adapters        = cfg["adapters"]
    global_heuristics = cfg["heuristics"]
    owner_map       = household.get("account_owner_map", [])

    all_accounts: dict[str, dict] = {}
    source_files: list[dict] = []
    as_of_date = datetime.now(timezone.utc).date().isoformat()

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        log.warning("No CSV files found in %s", raw_dir)

    for csv_path in csv_files:
        matched_adapter = None
        for adapter in adapters.values():
            if re.match(adapter["filename_pattern"], csv_path.name):
                matched_adapter = adapter
                break

        if not matched_adapter:
            log.warning("No adapter matched %s — skipping", csv_path.name)
            continue

        log.info("Parsing %s with adapter '%s'", csv_path.name, matched_adapter["id"])

        try:
            rows = parse_csv(csv_path, matched_adapter)
        except Exception as exc:
            log.error("Failed to parse %s: %s", csv_path.name, exc)
            continue

        file_accounts = extract_holdings_from_rows(
            rows, matched_adapter, global_heuristics, owner_map, as_of_date
        )

        # Merge into global account map (same account_id across files → merge holdings)
        for acct_id, acct in file_accounts.items():
            if acct_id in all_accounts:
                log.warning("Duplicate account_id %s — merging holdings", acct_id)
                all_accounts[acct_id]["holdings"].extend(acct["holdings"])
                all_accounts[acct_id]["balance_usd"] = round(
                    all_accounts[acct_id]["balance_usd"] + acct["balance_usd"], 2
                )
            else:
                all_accounts[acct_id] = acct

        source_files.append({
            "filename":    csv_path.name,
            "adapter":     matched_adapter["id"],
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "row_count":   len(rows),
        })

    accounts_list = list(all_accounts.values())
    summary = build_summary(accounts_list)

    portfolio_state = {
        "schema_version": SCHEMA_VERSION,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "source_files":   source_files,
        "owner":          build_owner_block(household),
        "accounts":       accounts_list,
        "summary":        summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(portfolio_state, f, indent=2)

    log.info(
        "Wrote %s — %d accounts, net worth $%.0f",
        out_path,
        len(accounts_list),
        summary["total_net_worth_usd"],
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MonteFIRE ETL: CSVs → portfolio_state.json")
    parser.add_argument("--raw-dir",   default="raw",            help="Directory containing raw CSV files")
    parser.add_argument("--config",    default="config",         help="Directory containing YAML config files")
    parser.add_argument("--out",       default="output/portfolio_state.json", help="Output JSON path")
    args = parser.parse_args()

    run_etl(
        raw_dir=Path(args.raw_dir),
        config_dir=Path(args.config),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
