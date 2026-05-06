# MonteFIRE

A 100% local, air-gapped FIRE (Financial Independence, Retire Early) planning tool. Combines a deterministic Monte Carlo retirement simulator with a local AI co-pilot — no data ever leaves your machine.

## Features

- **Monte Carlo simulation** — 1,000 iterations tracking taxable, tax-deferred, and tax-free buckets from current age to plan age
- **Tax-optimal withdrawal routing** — Taxable → Tax-Deferred → Tax-Free, with age-gate logic (59½ penalty, 73 RMD)
- **ACA subsidy cliff modeling** — dynamic healthcare costs based on MAGI before Medicare at 65
- **Spreadsheet-style UI** — editable cash flow grid via `st.data_editor` for instant what-if tweaks
- **Local AI co-pilot** — Ollama-powered chat that calls the simulation engine via function calling; never invents numbers
- **CSV ETL pipeline** — converts Fidelity and Empower exports to a standard internal schema via YAML adapter config

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Launch the UI
streamlit run app.py

# Run the ETL pipeline (drop CSVs in raw/ first)
python -m etl.parser

# Run tests
pytest
```

## Data Ingestion

Place bank/brokerage CSV exports in the `raw/` directory. The ETL pipeline matches files against `config/bank_adapters.yaml` and produces `output/portfolio_state.json`.

Supported institutions out of the box:
- **Fidelity** — `Portfolio_Positions_*.csv` (NetBenefits or Brokerage export)
- **Empower** — `Empower_Holdings_*.csv`

Edit `config/household.yaml` to set member birth dates and map account numbers to household members.

## AI Co-Pilot

Requires [Ollama](https://ollama.com) running locally. Pull any supported model (e.g. `ollama pull llama3`) and select it in the Co-Pilot tab. The LLM has access to one tool — `run_monte_carlo` — and is prompted to call it before answering any quantitative question.

## Usage

### Step 1 — Configure your household

Edit `config/household.yaml` with each member's birth date and role. The `account_owner_map` ties account name substrings to household members (first match wins).

### Step 2 — Import your portfolio

Download CSV exports from your brokerage and place them in the `raw/` directory:

| Institution | How to export | Expected filename pattern |
|-------------|--------------|--------------------------|
| Fidelity | Positions → Download → CSV | `Portfolio_Positions_*.csv` |
| Empower | Holdings → Export | `Empower_Holdings_*.csv` |

Then run the ETL pipeline (or click **Re-run ETL** in the sidebar):

```bash
python -m etl.parser
```

This produces `output/portfolio_state.json` and populates the Asset Ledger tab.

### Step 3 — Set simulation parameters

Use the sidebar to configure:
- **Ages** — current age and the age to simulate through
- **Return assumptions** — mean annual return and volatility
- **Inflation rate**

### Step 4 — Edit your cash flows (Tab 3)

The **Cash Flow & Rules** tab shows an editable grid of income and expense rows. Toggle rows on/off and edit amounts directly. Key rows:
- **Essential expenses** — housing, food, utilities
- **Social Security** — set your expected benefit and start age
- **Discretionary** — travel, gear, etc. (easy to toggle for what-if scenarios)

Changes take effect immediately when you click **Run Simulation**.

### Step 5 — Run the simulation (Tab 1)

Click **Run Simulation** on the Dashboard tab. The engine runs 1,000 Monte Carlo iterations and displays:
- **Success rate** — percentage of scenarios where the portfolio survives to plan age
- **Trajectory chart** — median portfolio value with p10/p90 confidence band
- **Tax & healthcare trends** — annual breakdown over the simulation horizon

### Step 6 — Ask the AI co-pilot (Tab 4)

Start [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.2
```

Select the model in the Co-Pilot tab and ask questions like:
- *"What happens if I retire 2 years earlier?"*
- *"How sensitive is my success rate to spending $10k more per year?"*
- *"Explain the healthcare cost spike at age 63."*

The co-pilot will call the simulation engine with adjusted parameters and explain the results — it never makes up numbers.

## Architecture

```
raw/*.csv  →  ETL (bank_adapters.yaml)  →  output/portfolio_state.json
                                                      ↓
                                            Streamlit UI (app.py)
                                          ┌──────────┴──────────┐
                                    Math Engine              Ollama LLM
                                  (engine/*.py)           (llm/agent.py)
```

All tax rules, ACA thresholds, and RMD tables live in `config/tax_config.yaml` — no hardcoded constants in Python source.
