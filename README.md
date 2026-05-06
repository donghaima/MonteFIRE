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
