"""
Tests for the Phase 4 LLM layer.

All tests are Ollama-free:
  - Tool schema, execution, and result formatting use only the engine.
  - agent.chat() is tested with unittest.mock patching ollama.chat.
  - System prompt builder is pure Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine import SimulationParams

CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def base_params() -> SimulationParams:
    return SimulationParams(
        taxable_balance=500_000,
        taxable_basis=300_000,
        tax_deferred_balance=800_000,
        tax_free_balance=200_000,
        current_age=50.0,
        plan_to_age=90,
        annual_spending_today=80_000,
        social_security_annual=0.0,
        social_security_start_age=67,
        num_iterations=100,
    )


@pytest.fixture(scope="module")
def sample_sim_result(base_params) -> dict:
    from llm.tools import _exec_run_monte_carlo
    return _exec_run_monte_carlo({}, base_params, CONFIG_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema
# ─────────────────────────────────────────────────────────────────────────────

class TestToolSchema:
    def test_tools_is_list(self):
        from llm.tools import TOOLS
        assert isinstance(TOOLS, list)
        assert len(TOOLS) >= 1

    def test_tool_has_required_fields(self):
        from llm.tools import TOOL_RUN_MONTE_CARLO
        t = TOOL_RUN_MONTE_CARLO
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "description" in t["function"]
        assert "parameters" in t["function"]

    def test_tool_schema_is_json_serialisable(self):
        from llm.tools import TOOLS
        serialised = json.dumps(TOOLS)
        loaded = json.loads(serialised)
        assert loaded[0]["function"]["name"] == "run_monte_carlo"

    def test_required_is_empty_list(self):
        from llm.tools import TOOL_RUN_MONTE_CARLO
        # All params are optional — the engine uses current session defaults
        assert TOOL_RUN_MONTE_CARLO["function"]["parameters"]["required"] == []

    def test_properties_include_spending(self):
        from llm.tools import TOOL_RUN_MONTE_CARLO
        props = TOOL_RUN_MONTE_CARLO["function"]["parameters"]["properties"]
        assert "annual_spending_today" in props
        assert "mean_annual_return" in props
        assert "social_security_annual" in props


# ─────────────────────────────────────────────────────────────────────────────
# Tool execution — run_monte_carlo
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteRunMonteCarlo:
    def test_returns_success_rate(self, base_params):
        from llm.tools import _exec_run_monte_carlo
        result = _exec_run_monte_carlo({}, base_params, CONFIG_DIR)
        assert "success_rate" in result
        assert 0.0 <= result["success_rate"] <= 1.0

    def test_returns_compact_not_full_trajectories(self, base_params):
        from llm.tools import _exec_run_monte_carlo
        result = _exec_run_monte_carlo({}, base_params, CONFIG_DIR)
        assert "median_trajectory" not in result   # full array stripped
        assert "ages" not in result

    def test_override_spending_changes_success_rate(self, base_params):
        from llm.tools import _exec_run_monte_carlo
        baseline  = _exec_run_monte_carlo({}, base_params, CONFIG_DIR)
        expensive = _exec_run_monte_carlo({"annual_spending_today": 200_000}, base_params, CONFIG_DIR)
        assert expensive["success_rate"] < baseline["success_rate"]

    def test_override_ignores_unknown_keys(self, base_params):
        from llm.tools import _exec_run_monte_carlo
        result = _exec_run_monte_carlo({"nonsense_param": 999}, base_params, CONFIG_DIR)
        assert "error" not in result

    def test_returns_aca_cliff_flag(self, base_params):
        from llm.tools import _exec_run_monte_carlo
        result = _exec_run_monte_carlo({}, base_params, CONFIG_DIR)
        assert "aca_cliff_detected" in result
        assert isinstance(result["aca_cliff_detected"], bool)

    def test_returns_first_year_taxes(self, base_params):
        from llm.tools import _exec_run_monte_carlo
        result = _exec_run_monte_carlo({}, base_params, CONFIG_DIR)
        assert "first_year_taxes" in result
        assert result["first_year_taxes"] >= 0

    def test_returns_median_portfolio_samples(self, base_params):
        from llm.tools import _exec_run_monte_carlo
        result = _exec_run_monte_carlo({}, base_params, CONFIG_DIR)
        portfolio_keys = [k for k in result if k.startswith("median_portfolio_at_")]
        assert len(portfolio_keys) >= 1

    def test_deterministic_with_fixed_seed(self, base_params):
        from llm.tools import _exec_run_monte_carlo
        r1 = _exec_run_monte_carlo({}, base_params, CONFIG_DIR)
        r2 = _exec_run_monte_carlo({}, base_params, CONFIG_DIR)
        assert r1["success_rate"] == r2["success_rate"]


class TestExecuteToolDispatch:
    def test_known_tool_dispatches(self, base_params):
        from llm.tools import execute_tool
        result = execute_tool("run_monte_carlo", {}, base_params, CONFIG_DIR)
        assert "success_rate" in result

    def test_unknown_tool_returns_error(self, base_params):
        from llm.tools import execute_tool
        result = execute_tool("fly_to_moon", {}, base_params, CONFIG_DIR)
        assert "error" in result


class TestCompactSimResult:
    def test_strips_trajectory_arrays(self, sample_sim_result):
        assert "median_trajectory" not in sample_sim_result
        assert "p10_trajectory" not in sample_sim_result

    def test_contains_success_rate(self, sample_sim_result):
        assert "success_rate" in sample_sim_result

    def test_aca_cliff_is_bool(self, sample_sim_result):
        assert isinstance(sample_sim_result["aca_cliff_detected"], bool)

    def test_max_hc_jump_non_negative(self, sample_sim_result):
        assert sample_sim_result["max_single_year_healthcare_jump"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemPrompt:
    def _build(self, portfolio=None, sim=None, params=None):
        from llm.prompts import build_system_prompt
        return build_system_prompt(portfolio, sim, params)

    def test_contains_rules(self):
        prompt = self._build()
        assert "STRICT RULES" in prompt
        assert "Never invent numbers" in prompt

    def test_contains_age_milestones(self):
        prompt = self._build()
        assert "59½" in prompt
        assert "73" in prompt
        assert "ACA" in prompt

    def test_contains_interpretation_guide(self):
        prompt = self._build()
        assert "90 %" in prompt or "90%" in prompt   # success rate guide

    def test_no_portfolio_placeholder(self):
        prompt = self._build(portfolio=None)
        assert "No portfolio data loaded" in prompt

    def test_no_sim_placeholder(self):
        prompt = self._build(sim=None)
        assert "No simulation has been run yet" in prompt

    def test_with_portfolio_shows_net_worth(self):
        ps = {
            "summary": {
                "total_net_worth_usd": 1_500_000,
                "by_tax_treatment": {"taxable": 500_000, "tax_deferred": 800_000, "tax_free": 200_000},
            }
        }
        prompt = self._build(portfolio=ps)
        assert "1,500,000" in prompt

    def test_with_sim_shows_success_rate(self, sample_sim_result):
        prompt = self._build(sim=sample_sim_result)
        sr_pct = f"{sample_sim_result['success_rate']:.1%}"
        assert sr_pct in prompt

    def test_aca_cliff_note_appears_when_detected(self):
        from llm.prompts import build_system_prompt
        sim = {
            "success_rate": 0.85,
            "plan_to_age": 90,
            "num_iterations": 500,
            "first_year_taxes": 5000,
            "first_year_healthcare": 8000,
            "aca_cliff_detected": True,
            "max_single_year_healthcare_jump": 15000,
        }
        prompt = build_system_prompt(None, sim, None)
        assert "ACA cliff" in prompt

    def test_aca_note_absent_when_no_cliff(self):
        from llm.prompts import build_system_prompt
        sim = {
            "success_rate": 0.92,
            "plan_to_age": 90,
            "num_iterations": 500,
            "first_year_taxes": 3000,
            "first_year_healthcare": 6000,
            "aca_cliff_detected": False,
            "max_single_year_healthcare_jump": 200,
        }
        prompt = build_system_prompt(None, sim, None)
        assert "⚠" not in prompt

    def test_prompt_is_string_and_non_empty(self):
        prompt = self._build()
        assert isinstance(prompt, str)
        assert len(prompt) > 200

    def test_prompt_has_reasonable_token_estimate(self):
        # Rough token estimate: 4 chars per token; system prompt < 2000 tokens
        prompt = self._build()
        assert len(prompt) / 4 < 2_000


# ─────────────────────────────────────────────────────────────────────────────
# Agent — mocked Ollama
# ─────────────────────────────────────────────────────────────────────────────

def _make_ollama_response(content: str, tool_calls=None):
    """Build a mock Ollama response object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    resp = MagicMock()
    resp.message = msg
    return resp


def _make_tool_call(name: str, arguments: dict):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class TestAgentChat:
    def _chat(self, user_message, mock_response, base_params, history=None):
        from llm.agent import chat
        with patch("llm.agent.ollama.chat", return_value=mock_response):
            return chat(
                user_message=user_message,
                history=history or [],
                portfolio_state=None,
                sim_result=None,
                base_params=base_params,
                config_dir=CONFIG_DIR,
            )

    def test_plain_text_response_no_tool(self, base_params):
        mock = _make_ollama_response("RMDs start at age 73 under current law.", tool_calls=None)
        resp = self._chat("When do RMDs start?", mock, base_params)
        assert resp.tool_called is False
        assert resp.tool_name is None
        assert resp.error is None

    def test_tool_called_flag_set(self, base_params):
        tool_call = _make_tool_call("run_monte_carlo", {"annual_spending_today": 90_000})

        # Pass 1: tool call response; pass 2: text response after tool
        pass1 = _make_ollama_response("", tool_calls=[tool_call])
        pass2 = _make_ollama_response("With $90k spending, your success rate is X%.", tool_calls=None)

        with patch("llm.agent.ollama.chat", side_effect=[pass1, pass2]):
            from llm.agent import chat
            resp = chat(
                user_message="What if I spend $90k?",
                history=[],
                portfolio_state=None,
                sim_result=None,
                base_params=base_params,
                config_dir=CONFIG_DIR,
            )

        assert resp.tool_called is True
        assert resp.tool_name == "run_monte_carlo"
        assert resp.tool_result is not None
        assert "success_rate" in resp.tool_result

    def test_tool_result_from_real_engine(self, base_params):
        """Tool execution goes to the real engine — verifies the plumbing is wired."""
        tool_call = _make_tool_call("run_monte_carlo", {})
        pass1 = _make_ollama_response("", tool_calls=[tool_call])
        pass2 = _make_ollama_response("Based on the simulation…", tool_calls=None)

        with patch("llm.agent.ollama.chat", side_effect=[pass1, pass2]):
            from llm.agent import chat
            resp = chat(
                user_message="What is my success rate?",
                history=[],
                portfolio_state=None,
                sim_result=None,
                base_params=base_params,
                config_dir=CONFIG_DIR,
            )

        assert resp.tool_result is not None
        assert 0.0 <= resp.tool_result["success_rate"] <= 1.0

    def test_ollama_error_returns_error_field(self, base_params):
        with patch("llm.agent.ollama.chat", side_effect=Exception("connection refused")):
            from llm.agent import chat
            resp = chat(
                user_message="hello",
                history=[],
                portfolio_state=None,
                sim_result=None,
                base_params=base_params,
                config_dir=CONFIG_DIR,
            )
        assert resp.error is not None
        assert "connection refused" in resp.error

    def test_history_filtered_to_user_assistant(self, base_params):
        """Internal metadata fields must not leak into the LLM message list."""
        history = [
            {"role": "user",      "content": "What is my net worth?"},
            {"role": "assistant", "content": "Based on the portfolio…",
             "tool_called": True, "tool_name": "run_monte_carlo"},
        ]
        captured_messages = []

        def fake_chat(model, messages, **kwargs):
            captured_messages.extend(messages)
            return _make_ollama_response("Follow-up answer.", tool_calls=None)

        with patch("llm.agent.ollama.chat", side_effect=fake_chat):
            from llm.agent import chat
            chat("Tell me more", history, None, None, base_params, CONFIG_DIR)

        llm_msgs = [m for m in captured_messages if m["role"] in ("user", "assistant")]
        for m in llm_msgs:
            assert "tool_called" not in m
            assert "tool_name" not in m


class TestAgentHelpers:
    def test_is_ollama_running_returns_bool(self):
        from llm.agent import is_ollama_running
        result = is_ollama_running()
        assert isinstance(result, bool)

    def test_list_models_returns_list(self):
        from llm.agent import list_models
        result = list_models()
        assert isinstance(result, list)

    def test_llm_history_strips_metadata(self):
        from llm.agent import _llm_history
        history = [
            {"role": "user",      "content": "Hello",   "extra_key": "x"},
            {"role": "assistant", "content": "Hi there", "tool_called": True},
            {"role": "user",      "content": "More?"},
        ]
        cleaned = _llm_history(history)
        assert len(cleaned) == 3
        for m in cleaned:
            assert set(m.keys()) == {"role", "content"}

    def test_llm_history_excludes_empty_content(self):
        from llm.agent import _llm_history
        history = [
            {"role": "assistant", "content": ""},
            {"role": "user",      "content": "Hi"},
        ]
        cleaned = _llm_history(history)
        assert len(cleaned) == 1
        assert cleaned[0]["role"] == "user"
