"""
MonteFIRE — Local-First Family Finance Simulator
Streamlit UI — Phase 3 / LLM Co-Pilot — Phase 4

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# Page config must be the very first Streamlit call
st.set_page_config(
    page_title="MonteFIRE — FIRE Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from engine import SimulationParams  # noqa: E402 — import after set_page_config
from engine.monte_carlo import run as run_engine  # noqa: E402
from llm import chat as llm_chat, is_ollama_running, list_models  # noqa: E402
from llm.tools import _compact_sim_result  # noqa: E402

# ── Paths & constants ─────────────────────────────────────────────────────────

CONFIG_DIR      = Path("config")
PORTFOLIO_PATH  = Path("output/portfolio_state.json")
HOUSEHOLD_PATH  = Path("config/household.yaml")
USER_SETTINGS_PATH = Path("config/user_settings.json")

DEFAULT_CASH_FLOWS = pd.DataFrame([
    {"Name": "Core Living Expenses",      "Category": "Essential",     "Annual ($)": 60_000, "Active": True},
    {"Name": "Property Tax & Insurance",  "Category": "Essential",     "Annual ($)": 8_000,  "Active": True},
    {"Name": "Utilities & Subscriptions", "Category": "Essential",     "Annual ($)": 4_000,  "Active": True},
    {"Name": "Travel",                    "Category": "Discretionary", "Annual ($)": 12_000, "Active": True},
    {"Name": "Gear & Hobbies",            "Category": "Discretionary", "Annual ($)": 3_000,  "Active": True},
    {"Name": "Dining & Entertainment",    "Category": "Discretionary", "Annual ($)": 6_000,  "Active": True},
    {"Name": "Social Security (Primary)", "Category": "Income",        "Annual ($)": 28_000, "Active": False},
    {"Name": "Social Security (Spouse)",  "Category": "Income",        "Annual ($)": 18_000, "Active": False},
    {"Name": "Part-time / Consulting",    "Category": "Income",        "Annual ($)": 20_000, "Active": False},
    {"Name": "Rental Income",             "Category": "Income",        "Annual ($)":      0, "Active": False},
])

DEFAULT_SIM_PARAMS: dict = {
    "mean_annual_return":      7.0,   # stored as percent
    "return_std_dev":         15.0,
    "inflation_rate":          3.0,
    "plan_to_age":            90,
    "num_iterations":       1_000,
    "social_security_start_age": 67,
    "ss_spouse_start_age":    67,
    "filing_status":          "married_filing_jointly",
    "household_size":          2,
    "healthcare_inflation_rate": 5.0,
}

# ── User-settings persistence ────────────────────────────────────────────────

def _load_user_settings() -> tuple[pd.DataFrame, dict]:
    """Load cash flows and sim params from disk, falling back to defaults."""
    if USER_SETTINGS_PATH.exists():
        try:
            data = json.loads(USER_SETTINGS_PATH.read_text())
            cf = pd.DataFrame(data["cash_flows"])
            sp = {**DEFAULT_SIM_PARAMS, **{
                k: v for k, v in data.get("sim_params", {}).items()
                if k in DEFAULT_SIM_PARAMS
            }}
            return cf, sp
        except Exception:
            pass
    return DEFAULT_CASH_FLOWS.copy(), DEFAULT_SIM_PARAMS.copy()


def _save_user_settings() -> None:
    """Persist current cash flows and sim params to disk."""
    data = {
        "cash_flows": st.session_state.cash_flows.to_dict(orient="records"),
        "sim_params": st.session_state.sim_params,
    }
    USER_SETTINGS_PATH.write_text(json.dumps(data, indent=2, default=str))


# ── Session-state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    if "portfolio_state" not in st.session_state:
        st.session_state.portfolio_state = _load_portfolio()
    if "cash_flows" not in st.session_state or "sim_params" not in st.session_state:
        cf, sp = _load_user_settings()
        if "cash_flows" not in st.session_state:
            st.session_state.cash_flows = cf
        if "sim_params" not in st.session_state:
            st.session_state.sim_params = sp
    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_model" not in st.session_state:
        from llm.agent import DEFAULT_MODEL
        st.session_state.llm_model = DEFAULT_MODEL


def _load_portfolio() -> dict | None:
    if PORTFOLIO_PATH.exists():
        try:
            return json.loads(PORTFOLIO_PATH.read_text())
        except Exception:
            return None
    return None


# ── Simulation helpers ────────────────────────────────────────────────────────

def _build_sim_params() -> SimulationParams | None:
    ps = st.session_state.portfolio_state
    if ps is None:
        return None

    cf: pd.DataFrame = st.session_state.cash_flows
    sp: dict = st.session_state.sim_params

    active = cf[cf["Active"]]
    annual_expenses = float(active[active["Category"] != "Income"]["Annual ($)"].sum())

    income_rows = active[active["Category"] == "Income"]
    is_ss_primary = income_rows["Name"].str.contains("Primary", case=False, na=False) & \
                    income_rows["Name"].str.contains("Security|SS", case=False, na=False)
    is_ss_spouse  = income_rows["Name"].str.contains("Spouse",  case=False, na=False) & \
                    income_rows["Name"].str.contains("Security|SS", case=False, na=False)
    ss_primary_amt = float(income_rows[is_ss_primary]["Annual ($)"].sum())
    ss_spouse_amt  = float(income_rows[is_ss_spouse]["Annual ($)"].sum())
    pension_amt    = float(income_rows[~(is_ss_primary | is_ss_spouse)]["Annual ($)"].sum())

    return SimulationParams.from_portfolio_state(
        ps,
        overrides={
            "annual_spending_today":      annual_expenses,
            "social_security_annual":     ss_primary_amt,
            "social_security_start_age":  int(sp["social_security_start_age"]),
            "ss_spouse_annual":           ss_spouse_amt,
            "ss_spouse_start_age":        int(sp["ss_spouse_start_age"]),
            "pension_annual":             pension_amt,
            "mean_annual_return":         sp["mean_annual_return"] / 100.0,
            "return_std_dev":             sp["return_std_dev"] / 100.0,
            "inflation_rate":             sp["inflation_rate"] / 100.0,
            "healthcare_inflation_rate":  sp["healthcare_inflation_rate"] / 100.0,
            "plan_to_age":                int(sp["plan_to_age"]),
            "num_iterations":             int(sp["num_iterations"]),
            "filing_status":              sp["filing_status"],
            "household_size":             int(sp["household_size"]),
        },
    )


def _run_simulation() -> None:
    params = _build_sim_params()
    if params is None:
        st.error("No portfolio loaded. Drop CSV files into `raw/` and click **Run ETL Pipeline**.")
        return
    n = params.num_iterations
    with st.spinner(f"Running {n:,} Monte Carlo iterations…"):
        st.session_state.sim_result = run_engine(params, config_dir=CONFIG_DIR)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


def _trajectory_chart(result: dict) -> go.Figure:
    ages = result["ages"]
    med  = result["median_trajectory"]
    p10  = result["p10_trajectory"]
    p90  = result["p90_trajectory"]

    fig = go.Figure()

    # P10–P90 confidence band (filled polygon)
    fig.add_trace(go.Scatter(
        x=ages + ages[::-1],
        y=p90 + p10[::-1],
        fill="toself",
        fillcolor="rgba(59,130,246,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10th–90th pct",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=p10,
        line=dict(color="rgba(59,130,246,0.45)", dash="dot", width=1.5),
        name="10th pct",
        hovertemplate="Age %{x}<br>10th pct: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=p90,
        line=dict(color="rgba(59,130,246,0.45)", dash="dot", width=1.5),
        name="90th pct",
        hovertemplate="Age %{x}<br>90th pct: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=med,
        line=dict(color="rgb(59,130,246)", width=2.5),
        name="Median",
        hovertemplate="Age %{x}<br>Median: $%{y:,.0f}<extra></extra>",
    ))

    # Zero-line to mark portfolio exhaustion
    fig.add_hline(y=0, line_color="rgba(239,68,68,0.5)", line_dash="dash", line_width=1)

    fig.update_layout(
        title="Portfolio Trajectory",
        xaxis_title="Age",
        yaxis_title="Portfolio Value",
        yaxis=dict(tickprefix="$", tickformat=",.0s"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def _area_chart(ages: list, values: list, title: str, color_rgb: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=ages, y=values,
        line=dict(color=f"rgb({color_rgb})", width=2),
        fill="tozeroy",
        fillcolor=f"rgba({color_rgb},0.15)",
        hovertemplate="Age %{x}<br>" + title + ": $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Age",
        yaxis=dict(tickprefix="$", tickformat=",.0s"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=230,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Data Pipeline")

        if st.button("🔄  Reload Portfolio", use_container_width=True):
            st.session_state.portfolio_state = _load_portfolio()
            st.session_state.sim_result = None
            st.rerun()

        if st.button("⚙️  Run ETL Pipeline", use_container_width=True,
                     help="Parses CSV files from the `raw/` directory"):
            with st.spinner("Running ETL…"):
                proc = subprocess.run(
                    [sys.executable, "-m", "etl.parser"],
                    capture_output=True, text=True,
                )
            if proc.returncode == 0:
                st.session_state.portfolio_state = _load_portfolio()
                st.session_state.sim_result = None
                st.success("ETL complete.")
                st.rerun()
            else:
                st.error(proc.stderr or "ETL failed.")

        ps = st.session_state.portfolio_state
        st.divider()
        if ps:
            st.caption(f"Portfolio: {ps.get('generated_at', '')[:10]}")
            st.caption(f"Accounts: {len(ps.get('accounts', []))}")
            nw = ps['summary']['total_net_worth_usd']
            st.caption(f"Net worth: ${nw:,.0f}")
        else:
            st.caption("No portfolio loaded.")

        result = st.session_state.sim_result
        if result:
            st.divider()
            sr = result["success_rate"]
            color = "green" if sr >= 0.90 else ("orange" if sr >= 0.70 else "red")
            st.markdown(f"**Last run success rate:** :{color}[{sr:.1%}]")


# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────

def _render_dashboard() -> None:
    result = st.session_state.sim_result
    ps = st.session_state.portfolio_state

    # ── Metrics row ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    nw = ps["summary"]["total_net_worth_usd"] if ps else 0.0
    col1.metric("Net Worth", _fmt_dollar(nw))

    if result:
        sr = result["success_rate"]
        col2.metric("Success Rate", f"{sr:.1%}",
                    delta="✓ On track" if sr >= 0.90 else ("⚠ Borderline" if sr >= 0.70 else "✗ At risk"),
                    delta_color="normal" if sr >= 0.90 else ("off" if sr >= 0.70 else "inverse"))

        # Age at which median trajectory first reaches zero
        med  = result["median_trajectory"]
        ages = result["ages"]
        survives_to = next((ages[i - 1] for i, v in enumerate(med) if v <= 0), ages[-1])
        col3.metric("Median Survives To", f"Age {survives_to}")

        lifetime_tax = sum(result["median_taxes"])
        col4.metric("Est. Lifetime Taxes", _fmt_dollar(lifetime_tax))
    else:
        col2.metric("Success Rate", "—")
        col3.metric("Median Survives To", "—")
        col4.metric("Est. Lifetime Taxes", "—")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    if result:
        st.plotly_chart(_trajectory_chart(result), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _area_chart(result["ages"], result["median_taxes"],
                            "Annual Tax Burden (Median)", "239,68,68"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _area_chart(result["ages"], result["median_healthcare"],
                            "Annual Healthcare Cost (Median)", "16,185,129"),
                use_container_width=True,
            )

        # ACA cliff callout
        cf: pd.DataFrame = st.session_state.cash_flows
        hc = result["median_healthcare"]
        if len(hc) > 1:
            max_jump = max(abs(hc[i] - hc[i - 1]) for i in range(1, len(hc)))
            if max_jump > 5_000:
                st.warning(
                    f"⚠ ACA cliff detected: up to **${max_jump:,.0f}/yr** healthcare jump "
                    "in a single year. Review income sources in **Cash Flow & Rules**."
                )
    else:
        st.info(
            "No simulation has been run yet. "
            "Configure parameters in **Cash Flow & Rules**, then click **▶ Run Simulation**."
        )

    st.divider()
    if st.button("▶  Run Simulation", type="primary", use_container_width=True, key="run_btn_dash"):
        _run_simulation()
        st.rerun()


# ── Tab 2: Asset Ledger ───────────────────────────────────────────────────────

def _render_asset_ledger() -> None:
    ps = st.session_state.portfolio_state

    if ps is None:
        st.warning(
            "No portfolio loaded. Drop CSVs into `raw/` and click **Run ETL Pipeline** in the sidebar."
        )
        return

    summary = ps["summary"]
    by_tt   = summary["by_tax_treatment"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Net Worth", _fmt_dollar(summary["total_net_worth_usd"]))
    c2.metric("Taxable",         _fmt_dollar(by_tt["taxable"]))
    c3.metric("Tax-Deferred",    _fmt_dollar(by_tt["tax_deferred"]))
    c4.metric("Tax-Free",        _fmt_dollar(by_tt["tax_free"]))

    st.divider()

    # ── Asset allocation ──────────────────────────────────────────────────────
    by_ac = summary.get("by_asset_class", {})
    if by_ac:
        total = sum(by_ac.values()) or 1.0
        ac_rows = sorted(
            [{"Asset Class": k.replace("_", " ").title(),
              "Value ($)": v,
              "Weight (%)": round(v / total * 100, 1)}
             for k, v in by_ac.items()],
            key=lambda r: -r["Value ($)"],
        )
        st.subheader("Asset Allocation")
        st.dataframe(
            pd.DataFrame(ac_rows),
            column_config={
                "Value ($)":   st.column_config.NumberColumn(format="$%.0f"),
                "Weight (%)":  st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100,
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
        st.divider()

    # ── Account-level table ───────────────────────────────────────────────────
    st.subheader("Accounts")
    account_rows = [
        {
            "Institution":     a["institution"],
            "Type":            a["account_type"].replace("_", " ").upper(),
            "Tax Treatment":   a["tax_treatment"].replace("_", " ").title(),
            "Balance ($)":     a["balance_usd"],
            "Holdings":        len(a["holdings"]),
            "As of":           a["as_of_date"],
        }
        for a in ps.get("accounts", [])
    ]
    if account_rows:
        st.dataframe(
            pd.DataFrame(account_rows),
            column_config={"Balance ($)": st.column_config.NumberColumn(format="$%.0f")},
            hide_index=True,
            use_container_width=True,
        )

    # ── Per-account holdings expanders ────────────────────────────────────────
    st.subheader("Holdings Detail")
    for acct in ps.get("accounts", []):
        label = (
            f"{acct['institution']} — "
            f"{acct['account_type'].replace('_', ' ').upper()}  "
            f"(${acct['balance_usd']:,.0f})"
        )
        with st.expander(label):
            h_rows = [
                {
                    "Ticker / Name":  h["ticker_or_name"],
                    "Asset Class":    h["asset_class"].replace("_", " ").title(),
                    "Value ($)":      h["value_usd"],
                    "Shares":         h.get("shares"),
                    "Price ($)":      h.get("price_usd"),
                    "Cost Basis ($)": h.get("cost_basis_usd"),
                }
                for h in acct.get("holdings", [])
            ]
            if h_rows:
                st.dataframe(
                    pd.DataFrame(h_rows),
                    column_config={
                        "Value ($)":      st.column_config.NumberColumn(format="$%.2f"),
                        "Price ($)":      st.column_config.NumberColumn(format="$%.4f"),
                        "Cost Basis ($)": st.column_config.NumberColumn(format="$%.2f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )


# ── Tab 3: Cash Flow & Rules ──────────────────────────────────────────────────

def _render_cash_flows() -> None:
    st.subheader("Cash Flow Ledger")
    st.caption(
        "Toggle rows on/off and edit amounts. "
        "Expenses feed `annual_spending_today`; Income items feed `social_security_annual`. "
        "Changes apply on the next **Run Simulation**."
    )

    edited: pd.DataFrame = st.data_editor(
        st.session_state.cash_flows,
        column_config={
            "Name": st.column_config.TextColumn("Line Item", width="large"),
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=["Essential", "Discretionary", "Income"],
                width="medium",
                required=True,
            ),
            "Annual ($)": st.column_config.NumberColumn(
                "Annual ($)", format="$%d", min_value=0, step=500, width="medium",
            ),
            "Active": st.column_config.CheckboxColumn("Active", width="small"),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )
    if not edited.equals(st.session_state.cash_flows):
        st.session_state.cash_flows = edited
        _save_user_settings()
    else:
        st.session_state.cash_flows = edited

    # Derived totals
    active  = edited[edited["Active"]]
    expenses = float(active[active["Category"] != "Income"]["Annual ($)"].sum())
    income   = float(active[active["Category"] == "Income"]["Annual ($)"].sum())
    net      = expenses - income

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Annual Expenses", _fmt_dollar(expenses))
    c2.metric("Total Annual Income",   _fmt_dollar(income))
    c3.metric("Net from Portfolio",    _fmt_dollar(net),
              delta=f"{net / expenses * 100:.0f}% of expenses" if expenses else "—",
              delta_color="off")

    st.divider()

    # ── Simulation parameters ─────────────────────────────────────────────────
    st.subheader("Simulation Parameters")
    sp = st.session_state.sim_params

    col_l, col_r = st.columns(2)
    with col_l:
        sp["mean_annual_return"] = st.slider(
            "Mean Annual Return (%)", 3.0, 12.0, float(sp["mean_annual_return"]), 0.1,
            help="Expected nominal return across the full portfolio (pre-inflation).",
        )
        sp["return_std_dev"] = st.slider(
            "Volatility / Std Dev (%)", 5.0, 30.0, float(sp["return_std_dev"]), 0.5,
            help="Annual standard deviation of returns. 15% ≈ broadly diversified equity portfolio.",
        )
        sp["inflation_rate"] = st.slider(
            "Inflation Rate (%)", 1.0, 8.0, float(sp["inflation_rate"]), 0.1,
        )

    with col_r:
        sp["plan_to_age"] = st.slider("Plan to Age", 80, 105, int(sp["plan_to_age"]), 1)
        sp["num_iterations"] = st.select_slider(
            "Monte Carlo Iterations",
            options=[100, 250, 500, 1_000, 2_000, 5_000],
            value=min(int(sp["num_iterations"]), 5_000),
        )
        sp["social_security_start_age"] = st.slider(
            "Primary SS Claim Age", 62, 70, int(sp["social_security_start_age"]), 1,
        )
        sp["ss_spouse_start_age"] = st.slider(
            "Spouse SS Claim Age", 62, 70, int(sp.get("ss_spouse_start_age", 67)), 1,
        )

    sp["filing_status"] = st.radio(
        "Filing Status",
        ["married_filing_jointly", "single"],
        format_func=lambda x: "Married Filing Jointly" if x == "married_filing_jointly" else "Single",
        horizontal=True,
        index=0 if sp["filing_status"] == "married_filing_jointly" else 1,
    )

    c1, c2 = st.columns(2)
    sp["household_size"] = c1.number_input(
        "Household Size", 1, 8, int(sp["household_size"]), 1,
        help="Used to compute the Federal Poverty Level for ACA subsidy calculations.",
    )
    sp["healthcare_inflation_rate"] = c2.slider(
        "Healthcare Inflation (%)", 1.0, 10.0,
        float(sp.get("healthcare_inflation_rate", 5.0)), 0.5,
        help="Medical costs typically inflate at 5–7%/yr, faster than general CPI.",
    )
    st.caption("ACA vs. Medicare coverage is determined automatically from each member's age in household.yaml.")

    if sp != st.session_state.sim_params:
        st.session_state.sim_params = sp
        _save_user_settings()
    else:
        st.session_state.sim_params = sp

    # Real-time parameter summary callout
    real_return = sp["mean_annual_return"] - sp["inflation_rate"]
    _nw = (
        st.session_state.portfolio_state["summary"]["total_net_worth_usd"]
        if st.session_state.portfolio_state else 0
    )
    rule_of_thumb_wr = net / _nw * 100 if _nw else 0.0
    st.info(
        f"**Real return:** {real_return:.1f}%  ·  "
        f"**Withdrawal rate:** {rule_of_thumb_wr:.2f}%  ·  "
        f"**4% rule threshold:** ${'Yes ✓' if rule_of_thumb_wr <= 4.0 else 'No — above 4%'}"
    )

    st.divider()
    if st.button("▶  Run Simulation", type="primary", use_container_width=True, key="run_btn_cf"):
        _run_simulation()
        st.rerun()


# ── Tab 4: AI Co-Pilot ───────────────────────────────────────────────────────

def _render_ai_copilot() -> None:
    # ── Status bar ────────────────────────────────────────────────────────────
    ollama_ok = is_ollama_running()
    available = list_models() if ollama_ok else []

    col_status, col_model = st.columns([1, 2])
    with col_status:
        if ollama_ok:
            st.success("Ollama running", icon="✅")
        else:
            st.error("Ollama not running", icon="❌")

    with col_model:
        if available:
            st.session_state.llm_model = st.selectbox(
                "Model",
                available,
                index=available.index(st.session_state.llm_model)
                      if st.session_state.llm_model in available else 0,
                label_visibility="collapsed",
            )
        else:
            st.caption("`ollama serve` · then `ollama pull llama3.2` or `ollama pull qwen2.5`")

    if not ollama_ok:
        st.info(
            "Start Ollama to enable the AI co-pilot:\n\n"
            "```bash\nollama serve\n# in another terminal:\nollama pull llama3.2\n```"
        )
        return

    st.divider()

    # ── System context expander ───────────────────────────────────────────────
    result = st.session_state.sim_result
    compact = _compact_sim_result(result) if result else None
    with st.expander("📄 LLM context (system prompt snapshot)", expanded=False):
        from llm.prompts import build_system_prompt
        params = _build_sim_params()
        st.text(build_system_prompt(
            st.session_state.portfolio_state, compact, params
        ))

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("tool_called") and msg["role"] == "assistant":
                tool_label = msg.get("tool_name", "run_monte_carlo")
                st.caption(f"🔧 Called `{tool_label}` — result verified by deterministic engine")
            st.markdown(msg["content"])

    # ── Input ─────────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about your finances… e.g. 'What if I spend $90k/year?'"):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build history snapshot (exclude the message we just rendered)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
            if m["role"] in ("user", "assistant")
        ]

        params = _build_sim_params()

        # Call agent
        with st.chat_message("assistant"):
            spinner_msg = "Thinking…"
            with st.spinner(spinner_msg):
                response = llm_chat(
                    user_message=prompt,
                    history=history,
                    portfolio_state=st.session_state.portfolio_state,
                    sim_result=compact,
                    base_params=params,
                    config_dir=CONFIG_DIR,
                    model=st.session_state.llm_model,
                )

            if response.error:
                st.error(f"LLM error: {response.error}")
                reply_text = f"*(Error: {response.error})*"
            else:
                if response.tool_called:
                    st.caption(f"🔧 Called `{response.tool_name}` — result verified by deterministic engine")
                    # If a new simulation was run, update the dashboard result
                    if response.tool_result:
                        # Merge compact result back into a display-compatible format by
                        # re-running at full fidelity so Dashboard charts update too
                        if params is not None:
                            tool_args = response.tool_args or {}
                            _rerun_and_save(params, tool_args)
                st.markdown(response.text)
                reply_text = response.text

        # Persist to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply_text,
            "tool_called": response.tool_called,
            "tool_name": response.tool_name,
        })

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑  Clear chat history", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


def _rerun_and_save(base_params: SimulationParams, tool_args: dict) -> None:
    """
    Re-run the full simulation (1000 iters) with the LLM-specified overrides
    and save the result to session_state so Dashboard charts update.
    """
    params_dict = {
        "taxable_balance":        base_params.taxable_balance,
        "taxable_basis":          base_params.taxable_basis,
        "tax_deferred_balance":   base_params.tax_deferred_balance,
        "tax_free_balance":       base_params.tax_free_balance,
        "current_age":            base_params.current_age,
        "plan_to_age":            base_params.plan_to_age,
        "filing_status":              base_params.filing_status,
        "household_size":             base_params.household_size,
        "spouse_current_age":         base_params.spouse_current_age,
        "mean_annual_return":         base_params.mean_annual_return,
        "return_std_dev":             base_params.return_std_dev,
        "inflation_rate":             base_params.inflation_rate,
        "healthcare_inflation_rate":  base_params.healthcare_inflation_rate,
        "annual_spending_today":      base_params.annual_spending_today,
        "social_security_annual":     base_params.social_security_annual,
        "social_security_start_age":  base_params.social_security_start_age,
        "ss_spouse_annual":           base_params.ss_spouse_annual,
        "ss_spouse_start_age":        base_params.ss_spouse_start_age,
        "pension_annual":             base_params.pension_annual,
        "tax_deferred_spouse_balance": base_params.tax_deferred_spouse_balance,
        "num_iterations":             st.session_state.sim_params.get("num_iterations", 1_000),
    }
    valid = set(params_dict.keys())
    for k, v in tool_args.items():
        if k in valid:
            params_dict[k] = v
    try:
        st.session_state.sim_result = run_engine(params_dict, config_dir=CONFIG_DIR)
    except Exception:
        pass  # dashboard keeps previous result


# ── Tab 5: Settings ───────────────────────────────────────────────────────────

def _render_settings() -> None:
    from datetime import date

    if not HOUSEHOLD_PATH.exists():
        st.error(f"Household config not found at `{HOUSEHOLD_PATH}`.")
        return

    household = yaml.safe_load(HOUSEHOLD_PATH.read_text())
    members   = household.get("members", [])

    # ── Household members ─────────────────────────────────────────────────────
    st.subheader("Household Members")
    st.caption("Edit names and birth dates. Changes take effect after saving and re-running the ETL pipeline.")

    today = date.today()
    age_notes = []
    for m in members:
        try:
            age = (today - date.fromisoformat(m["birth_date"])).days / 365.25
            age_notes.append(f"**{m['name']}** — currently age {age:.1f}")
        except ValueError:
            age_notes.append(f"**{m['name']}** — invalid birth date")
    if age_notes:
        st.info("  ·  ".join(age_notes))

    member_df = pd.DataFrame([
        {"Name": m["name"], "Birth Date": m["birth_date"], "Role": m["role"]}
        for m in members
    ])
    edited_members = st.data_editor(
        member_df,
        column_config={
            "Name":       st.column_config.TextColumn("Name", required=True),
            "Birth Date": st.column_config.TextColumn("Birth Date (YYYY-MM-DD)", required=True),
            "Role":       st.column_config.SelectboxColumn(
                              "Role", options=["primary", "spouse"], required=True),
        },
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # ── Account ownership rules ───────────────────────────────────────────────
    st.subheader("Account Ownership Rules")
    st.caption(
        "Regex patterns matched against account names from CSV exports. "
        "First match wins — keep the catch-all `.*` row last."
    )

    id_to_name = {m["id"]: m["name"] for m in members}
    name_to_id = {m["name"]: m["id"] for m in members}
    member_names = list(name_to_id.keys())

    map_df = pd.DataFrame([
        {"Match Pattern": r["match"], "Owner": id_to_name.get(r["owner_id"], r["owner_id"])}
        for r in household.get("account_owner_map", [])
    ])
    edited_map = st.data_editor(
        map_df,
        column_config={
            "Match Pattern": st.column_config.TextColumn(
                "Match Pattern (regex)", required=True, width="large"),
            "Owner": st.column_config.SelectboxColumn(
                "Owner", options=member_names, required=True, width="medium"),
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    if st.button("💾  Save Household Settings", type="primary", use_container_width=False):
        # Validate birth dates before saving
        errors = []
        for _, row in edited_members.iterrows():
            try:
                date.fromisoformat(row["Birth Date"])
            except ValueError:
                errors.append(f"Invalid date for {row['Name']}: `{row['Birth Date']}` — use YYYY-MM-DD.")
        if errors:
            for e in errors:
                st.error(e)
            return

        # Preserve original member IDs (keyed by role)
        id_by_role = {m["role"]: m["id"] for m in members}
        household["members"] = [
            {
                "id":         id_by_role.get(row["Role"], f"member-{row['Role']}"),
                "name":       row["Name"],
                "birth_date": row["Birth Date"],
                "role":       row["Role"],
            }
            for _, row in edited_members.iterrows()
        ]
        household["account_owner_map"] = [
            {
                "match":    row["Match Pattern"],
                "owner_id": name_to_id.get(row["Owner"], row["Owner"]),
            }
            for _, row in edited_map.iterrows()
        ]

        HOUSEHOLD_PATH.write_text(
            yaml.dump(household, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )
        # Reset portfolio so ages reload from updated birth dates
        st.session_state.portfolio_state = _load_portfolio()
        st.session_state.sim_result = None
        st.success("Saved. Re-run the ETL pipeline in the sidebar to apply account ownership changes.")
        st.rerun()


# ── App entry point ───────────────────────────────────────────────────────────

_init_state()
_render_sidebar()

ps = st.session_state.portfolio_state
as_of = ps["generated_at"][:10] if ps else "—"
st.title("MonteFIRE — Family Finance Simulator")
st.caption(f"Portfolio snapshot: {as_of}  ·  All computation runs locally (air-gapped).")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Dashboard",
    "📋  Asset Ledger",
    "💰  Cash Flow & Rules",
    "🤖  AI Co-Pilot",
    "⚙️  Settings",
])

with tab1:
    _render_dashboard()
with tab2:
    _render_asset_ledger()
with tab3:
    _render_cash_flows()
with tab4:
    _render_ai_copilot()
with tab5:
    _render_settings()
