"""
LLM chat agent — orchestrates Ollama calls and tool execution.

Flow for each user turn:
  1. Build messages: system_prompt + filtered history + user message.
  2. Call Ollama (non-streaming) to check for tool_calls.
  3. If tool_calls present:
       a. Execute each tool against the deterministic engine.
       b. Append assistant + tool result messages.
       c. Call Ollama again (streaming) to get the final response.
  4. If no tool_calls: stream the response directly.

The LLM never sees raw trajectory arrays — only the compact summary
returned by tools.compact_sim_result().
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import ollama

from engine import SimulationParams
from llm.prompts import build_system_prompt
from llm.tools import TOOLS, execute_tool

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemma4:e4b"


# ── Public return type ────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    text: str
    tool_called: bool = False
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    tool_result: dict | None = None    # compact sim result if run_monte_carlo was called
    error: str | None = None


# ── Ollama availability ───────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/version", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    try:
        models = ollama.list()
        return sorted(m.model for m in models.models if "embed" not in m.model.lower())
    except Exception:
        return []


# ── Message helpers ───────────────────────────────────────────────────────────

def _llm_history(chat_history: list[dict]) -> list[dict]:
    """
    Strip display metadata from session history, keeping only role + content.
    Includes only user and assistant turns (not tool turns, which are rebuilt fresh).
    """
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in chat_history
        if msg["role"] in ("user", "assistant") and msg.get("content")
    ]


def _stream_text(messages: list[dict], model: str) -> Generator[str, None, None]:
    """Yield text chunks from a streaming Ollama response."""
    try:
        for chunk in ollama.chat(model=model, messages=messages, stream=True):
            text = (chunk.message.content or "") if hasattr(chunk, "message") else ""
            if text:
                yield text
    except Exception as exc:
        yield f"\n\n*(streaming error: {exc})*"


# ── Main chat function ────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[dict],
    portfolio_state: dict | None,
    sim_result: dict | None,
    base_params: SimulationParams | None,
    config_dir: Path,
    model: str = DEFAULT_MODEL,
) -> ChatResponse:
    """
    Send user_message to the LLM, execute any tool calls, return the final response.

    Args:
        user_message:    The current user input.
        history:         Previous turns (role/content only, no current message).
        portfolio_state: portfolio_state.json dict for system prompt context.
        sim_result:      Last compact simulation result dict for context.
        base_params:     Current SimulationParams used as defaults for tool calls.
        config_dir:      Path to config/ directory (passed to engine).
        model:           Ollama model name.

    Returns a ChatResponse dataclass.
    """
    system_prompt = build_system_prompt(portfolio_state, sim_result, base_params)

    messages: list[dict] = (
        [{"role": "system", "content": system_prompt}]
        + _llm_history(history)
        + [{"role": "user", "content": user_message}]
    )

    # ── Pass 1: check for tool calls (non-streaming) ──────────────────────────
    try:
        response = ollama.chat(model=model, messages=messages, tools=TOOLS)
    except Exception as exc:
        log.error("Ollama call failed: %s", exc)
        return ChatResponse(text="", error=str(exc))

    msg = response.message

    if not msg.tool_calls:
        # No tools needed — stream the response
        stream_messages = messages  # same context, just stream it
        full_text = _collect_stream(model, stream_messages, fallback=msg.content or "")
        return ChatResponse(text=full_text)

    # ── Pass 2: execute tool calls ────────────────────────────────────────────
    # Append the assistant's tool-call message
    messages.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in msg.tool_calls
        ],
    })

    tool_name: str | None = None
    tool_args: dict = {}
    tool_result: dict | None = None

    for tc in msg.tool_calls:
        tool_name = tc.function.name
        tool_args = tc.function.arguments or {}
        log.info("LLM called tool %r with args %s", tool_name, tool_args)

        if base_params is None:
            result_dict: dict = {"error": "No portfolio loaded — cannot run simulation."}
        else:
            result_dict = execute_tool(tool_name, tool_args, base_params, config_dir)

        if tool_name == "run_monte_carlo" and "error" not in result_dict:
            tool_result = result_dict

        messages.append({
            "role": "tool",
            "content": json.dumps(result_dict, indent=2),
        })

    # ── Pass 3: get final response (streaming) ────────────────────────────────
    full_text = _collect_stream(model, messages)

    return ChatResponse(
        text=full_text,
        tool_called=True,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
    )


def _collect_stream(model: str, messages: list[dict], fallback: str = "") -> str:
    """
    Stream a response from Ollama and collect into a single string.
    Falls back to `fallback` if streaming returns nothing.
    """
    try:
        chunks = []
        for chunk in ollama.chat(model=model, messages=messages, stream=True):
            text = (chunk.message.content or "") if hasattr(chunk, "message") else ""
            if text:
                chunks.append(text)
        result = "".join(chunks).strip()
        return result if result else fallback
    except Exception as exc:
        log.error("Streaming failed: %s", exc)
        return fallback or f"*(response error: {exc})*"
