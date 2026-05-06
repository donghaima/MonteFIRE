"""
System prompt builder for the MonteFIRE AI co-pilot.

The prompt encodes two things:
  1. Behavioural constraints — the LLM must never invent numbers.
  2. Current context — portfolio state and last simulation result so the
     LLM can answer factual questions without calling a tool.
"""

from __future__ import annotations

from engine import SimulationParams

_PERSONA = """\
You are MonteFIRE, an expert AI financial co-pilot specialising in FIRE \
(Financially Independent, Retire Early) planning for families.
You run fully locally — no data ever leaves this machine.\
"""

_RULES = """\
## STRICT RULES — read before every response

1. **Never invent numbers.** Do not calculate, estimate, or guess \
success rates, portfolio balances, tax amounts, or healthcare costs. \
If the answer requires math, call `run_monte_carlo` first.

2. **Tool before numbers.** Any "what-if" question (spending change, \
retirement age, Social Security timing, return assumption) MUST trigger \
a `run_monte_carlo` call. Quote only numbers returned by the tool.

3. **Explain, don't recalculate.** After a tool call, interpret the \
result in plain English. Focus on: success rate meaning, key risks, \
and actionable changes.

4. **Educational answers are fine without a tool.** For conceptual \
questions (e.g. "how do Roth conversions work?", "what is an RMD?") \
answer directly — no tool call needed.\
"""

_MILESTONES = """\
## AGE MILESTONES TO REFERENCE
- **59½** — Penalty-free withdrawals from tax-deferred accounts begin.
- **62–70** — Social Security claiming window (delay = higher benefit).
- **65** — Medicare eligibility; ACA marketplace coverage ends.
- **73** — Required Minimum Distributions (RMDs) begin from tax-deferred accounts.
- **400% FPL** — ACA subsidy cliff; one dollar over can cost $10k+/yr in premiums.\
"""

_INTERPRETATION = """\
## HOW TO INTERPRET SUCCESS RATE
- ≥ 90 % — On track. Focus on optimisation (tax efficiency, Roth conversions).
- 70–89 % — Borderline. Identify levers: spending cuts, SS delay, part-time income.
- < 70 % — At risk. Quantify the gap; suggest specific changes with tool calls.\
"""


def build_system_prompt(
    portfolio_state: dict | None,
    sim_result: dict | None,
    base_params: SimulationParams | None,
) -> str:
    sections = [_PERSONA, _RULES, _MILESTONES, _INTERPRETATION]

    # ── Portfolio context ─────────────────────────────────────────────────────
    if portfolio_state:
        s  = portfolio_state["summary"]
        tt = s["by_tax_treatment"]
        nw = s["total_net_worth_usd"]
        age_str = f"{base_params.current_age:.0f}" if base_params else "unknown"
        spending_str = (
            f"${base_params.annual_spending_today:,.0f}/yr"
            if base_params else "unknown"
        )
        portfolio_block = (
            f"## CURRENT PORTFOLIO (owner age ≈ {age_str})\n"
            f"Net Worth:     ${nw:>12,.0f}\n"
            f"  Taxable:     ${tt['taxable']:>12,.0f}\n"
            f"  Tax-Deferred:${tt['tax_deferred']:>12,.0f}\n"
            f"  Tax-Free:    ${tt['tax_free']:>12,.0f}\n"
            f"Current spending target: {spending_str}"
        )
    else:
        portfolio_block = (
            "## CURRENT PORTFOLIO\n"
            "No portfolio data loaded yet. Ask the user to run the ETL pipeline."
        )
    sections.append(portfolio_block)

    # ── Last simulation context ───────────────────────────────────────────────
    if sim_result:
        sr = sim_result["success_rate"]
        pt = sim_result["plan_to_age"]
        ni = sim_result["num_iterations"]
        hc_jump = sim_result.get("max_single_year_healthcare_jump", 0)
        aca_note = (
            f"  ⚠ ACA cliff detected: up to ${hc_jump:,.0f}/yr single-year jump."
            if sim_result.get("aca_cliff_detected") else ""
        )

        samples = {
            k.replace("median_portfolio_at_", "Age "): f"${v:,.0f}"
            for k, v in sim_result.items()
            if k.startswith("median_portfolio_at_")
        }
        sample_lines = "  " + " | ".join(f"{k}: {v}" for k, v in samples.items())

        sim_block = (
            f"## LAST SIMULATION RESULT ({ni:,} iterations, plan to age {pt})\n"
            f"Success rate: {sr:.1%}\n"
            f"Median portfolio:\n{sample_lines}\n"
            f"Year-1 taxes: ${sim_result.get('first_year_taxes', 0):,.0f}  "
            f"Year-1 healthcare: ${sim_result.get('first_year_healthcare', 0):,.0f}\n"
            + (aca_note if aca_note else "")
        )
    else:
        sim_block = (
            "## LAST SIMULATION RESULT\n"
            "No simulation has been run yet. "
            "If the user asks about success rates, call run_monte_carlo."
        )
    sections.append(sim_block)

    return "\n\n".join(sections)
