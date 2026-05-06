# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

MonteFIRE is a local-first FIRE (Financially Independent, Retire Early) planning tool. It runs a deterministic Monte Carlo retirement simulator with an AI co-pilot, entirely on-device — no data leaves the machine. The LLM interface uses Ollama (local models only).

## Commands

```bash
# Run the Streamlit UI
streamlit run app.py

# Run the ETL pipeline (bank CSVs → portfolio_state.json)
python -m etl.parser [--raw-dir RAW] [--config CONFIG] [--out OUT] [--household HOUSEHOLD]

# Run all tests
pytest

# Run a single test file
pytest tests/test_engine.py

# Run a single test
pytest tests/test_engine.py::test_name
```

## Architecture

### Data Flow

```
Raw CSVs (banks/brokers)
    ↓ etl/parser.py (YAML adapters from config/bank_adapters.yaml)
portfolio_state.json  ← single source of truth for net worth/holdings
    ↓
Streamlit UI (app.py) — sidebar inputs feed SimulationParams
    ↓
engine/monte_carlo.py — 1000 iterations, ~1s per run
    ↓
SimulationResult → UI charts (Plotly) + LLM co-pilot (compact summary)
```

### Key Modules

**`engine/`** — Pure, side-effect-free simulation math. All rules (tax brackets, ACA thresholds, RMD tables) live in `config/tax_config.yaml` — no hardcoded constants.
- `monte_carlo.py` — Runs N iterations tracking 3 account buckets (taxable, tax-deferred, tax-free) from current age to plan_to_age
- `withdrawal_router.py` — Tax-optimal bucket sequencing; age gates matter (<59.5, 59.5–72, 73+)
- `tax_engine.py` — Federal income tax, LTCG, early-withdrawal penalties, SS taxability
- `aca_engine.py` — ACA premiums with subsidy cliff; switches to Medicare at 65
- `models.py` — `SimulationParams` (inputs), `SimulationResult` (outputs), `Buckets`, `AnnualSnapshot`

**`llm/`** — AI co-pilot orchestration against Ollama.
- `agent.py` — Non-streaming tool-calling loop; delegates to engine for all numbers
- `tools.py` — Defines a single tool (`run_monte_carlo`) and compresses results for LLM context
- `prompts.py` — System prompt enforces "never invent numbers, always run the tool first"

**`etl/`** — `parser.py` ingests bank/brokerage CSVs, resolves asset classes via ticker tags + heuristics, groups into accounts, writes `portfolio_state.json`.

**`config/`** — YAML-driven: `household.yaml` (members + account ownership), `bank_adapters.yaml` (CSV column mappings per institution), `tax_config.yaml` (all numeric tax/ACA/RMD parameters).

**`app.py`** — Streamlit UI (722 LOC). Tabs: Sim, Analysis, AI Co-Pilot. Loads/saves `portfolio_state.json`, builds `SimulationParams` from sidebar, renders Plotly charts.

### Design Invariants

- The engine is purely functional — given the same `SimulationParams` and random seed, output is identical. Preserve this.
- The LLM has exactly one tool (`run_monte_carlo`). Adding tools requires updating both `llm/tools.py` schema and `llm/agent.py` dispatch.
- `portfolio_state.json` is the handoff between ETL and UI — changing its schema requires updating both sides.
- Tax/ACA/RMD constants belong in `config/tax_config.yaml`, not in Python source.
