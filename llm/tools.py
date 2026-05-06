"""
Tool definitions and execution for the LLM co-pilot.

Design rule: the LLM never does financial math.
Every quantitative answer must come from executing one of these tools,
which delegate to the deterministic engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine import SimulationParams
from engine.monte_carlo import run as _engine_run

# ── Tool schema (Ollama / OpenAI-compatible function-calling format) ──────────

TOOL_RUN_MONTE_CARLO: dict = {
    "type": "function",
    "function": {
        "name": "run_monte_carlo",
        "description": (
            "Run a Monte Carlo retirement simulation. "
            "MUST be called before answering ANY question involving success rates, "
            "portfolio values, or 'what-if' scenarios. "
            "Unspecified parameters inherit from the current session. "
            "Never estimate or guess financial outcomes — call this tool instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "annual_spending_today": {
                    "type": "number",
                    "description": "Annual non-healthcare living expenses in today's dollars (e.g. 80000).",
                },
                "mean_annual_return": {
                    "type": "number",
                    "description": "Expected nominal portfolio return as a decimal (e.g. 0.07 for 7%).",
                },
                "return_std_dev": {
                    "type": "number",
                    "description": "Annual return volatility as a decimal (e.g. 0.15 for 15%).",
                },
                "inflation_rate": {
                    "type": "number",
                    "description": "Annual inflation rate as a decimal (e.g. 0.03 for 3%).",
                },
                "plan_to_age": {
                    "type": "integer",
                    "description": "Age through which the portfolio must last (e.g. 90).",
                },
                "social_security_annual": {
                    "type": "number",
                    "description": "Combined annual Social Security / pension income in today's dollars.",
                },
                "social_security_start_age": {
                    "type": "integer",
                    "description": "Age at which social security / pension income begins.",
                },
                "taxable_balance": {
                    "type": "number",
                    "description": "Brokerage account balance override (for hypothetical scenarios only).",
                },
                "tax_deferred_balance": {
                    "type": "number",
                    "description": "Traditional 401k / IRA balance override (hypothetical only).",
                },
                "tax_free_balance": {
                    "type": "number",
                    "description": "Roth balance override (hypothetical only).",
                },
                "num_iterations": {
                    "type": "integer",
                    "description": "Monte Carlo iterations (default 500 for interactive speed).",
                },
            },
            "required": [],
        },
    },
}

TOOLS: list[dict] = [TOOL_RUN_MONTE_CARLO]


# ── Result formatter ──────────────────────────────────────────────────────────

def _compact_sim_result(result: dict) -> dict:
    """
    Distill a full simulation result into a concise dict for LLM context.
    Avoids flooding the context window with 40+ trajectory values.
    """
    ages  = result["ages"]
    med   = result["median_trajectory"]
    taxes = result["median_taxes"]
    hc    = result["median_healthcare"]

    def _at(age: int) -> int | None:
        try:
            return round(med[ages.index(age)])
        except ValueError:
            return None

    samples: dict[str, int] = {}
    for a in [ages[0], 59, 65, 73, 80, ages[-1]]:
        v = _at(a)
        if v is not None:
            samples[f"median_portfolio_at_{a}"] = v

    max_hc_jump = max(
        (abs(hc[i] - hc[i - 1]) for i in range(1, len(hc))),
        default=0,
    )

    return {
        "success_rate": round(result["success_rate"], 4),
        "plan_to_age": result["plan_to_age"],
        "num_iterations": result["num_iterations"],
        **samples,
        "first_year_taxes": round(taxes[0]) if taxes else 0,
        "first_year_healthcare": round(hc[0]) if hc else 0,
        "aca_cliff_detected": max_hc_jump > 5_000,
        "max_single_year_healthcare_jump": round(max_hc_jump),
    }


# ── Tool execution ────────────────────────────────────────────────────────────

def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    base_params: SimulationParams,
    config_dir: Path,
) -> dict[str, Any]:
    """
    Dispatch a tool call from the LLM to the deterministic engine.
    Returns a JSON-serialisable result dict.
    """
    if tool_name == "run_monte_carlo":
        return _exec_run_monte_carlo(args, base_params, config_dir)
    return {"error": f"Unknown tool: {tool_name!r}"}


def _exec_run_monte_carlo(
    args: dict[str, Any],
    base: SimulationParams,
    config_dir: Path,
) -> dict[str, Any]:
    """
    Merge LLM-supplied overrides onto the current session params and run the engine.
    Uses a fixed seed so identical queries give identical answers within a session.
    """
    params_dict: dict[str, Any] = {
        "taxable_balance":         base.taxable_balance,
        "taxable_basis":           base.taxable_basis,
        "tax_deferred_balance":    base.tax_deferred_balance,
        "tax_free_balance":        base.tax_free_balance,
        "current_age":             base.current_age,
        "plan_to_age":             base.plan_to_age,
        "filing_status":           base.filing_status,
        "household_size":          base.household_size,
        "spouse_current_age":      base.spouse_current_age,
        "mean_annual_return":      base.mean_annual_return,
        "return_std_dev":          base.return_std_dev,
        "inflation_rate":          base.inflation_rate,
        "annual_spending_today":   base.annual_spending_today,
        "social_security_annual":  base.social_security_annual,
        "social_security_start_age": base.social_security_start_age,
        "pension_annual":          base.pension_annual,
        "num_iterations":          500,   # faster for interactive use
    }
    # Apply LLM overrides (only keys that appear in SimulationParams)
    valid_keys = set(params_dict.keys())
    for k, v in args.items():
        if k in valid_keys:
            params_dict[k] = v

    try:
        full_result = _engine_run(params_dict, config_dir=config_dir, seed=42)
        return _compact_sim_result(full_result)
    except Exception as exc:
        return {"error": str(exc)}
