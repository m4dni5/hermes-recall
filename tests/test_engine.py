"""Unit tests for recall — schema, tool definition, loop with mocked LLM.

Tests cover:
  - Schema validity
  - SESSION_SEARCH_TOOL definition structure
  - System prompt content
  - Message building
  - Loop: tool call → tool result → text answer (the happy path)
  - Loop: immediate text answer (no tool calls)
  - Loop: max iterations → forced synthesis
  - Loop: unknown tool rejection
  - Loop: malformed tool arguments
  - Import resolution
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# --------------------------------------------------------------------------- #
# Schema and tool definition
# --------------------------------------------------------------------------- #

def test_schema_valid():
    from schemas import RECALL_SCHEMA

    assert RECALL_SCHEMA["name"] == "recall"
    assert "query" in RECALL_SCHEMA["parameters"]["properties"]
    assert RECALL_SCHEMA["parameters"]["required"] == ["query"]


def test_tool_definition_valid():
    from loop import SESSION_SEARCH_TOOL

    assert SESSION_SEARCH_TOOL["type"] == "function"
    fn = SESSION_SEARCH_TOOL["function"]
    assert fn["name"] == "session_search"
    props = fn["parameters"]["properties"]
    assert "query" in props
    assert "session_id" in props
    assert "around_message_id" in props


# --------------------------------------------------------------------------- #
# Prompts and message building
# --------------------------------------------------------------------------- #

def test_system_prompt_content():
    from loop import SUB_MODEL_SYSTEM

    assert "session_search" in SUB_MODEL_SYSTEM
    assert "discovery" in SUB_MODEL_SYSTEM.lower()
    # No FINAL convention — native tool calling handles termination.
    assert "FINAL" not in SUB_MODEL_SYSTEM


def test_build_messages():
    from loop import _build_messages

    msgs = _build_messages("test query")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "test query" in msgs[1]["content"]


# --------------------------------------------------------------------------- #
# Mock helpers
# --------------------------------------------------------------------------- #

def _make_msg(content=None, tool_calls=None):
    """Build a mock chat completion message."""
    msg = SimpleNamespace()
    msg.content = content
    if tool_calls:
        msg.tool_calls = []
        for i, tc in enumerate(tool_calls):
            msg.tool_calls.append(SimpleNamespace(
                id=f"call_{i}",
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=json.dumps(tc["args"]),
                ),
            ))
    else:
        msg.tool_calls = None
    return msg


def _make_response(msg):
    """Build a mock chat completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeDB:
    """Minimal DB stub — _dispatch_session_search is mocked, not called."""
    pass


# --------------------------------------------------------------------------- #
# Loop tests
# --------------------------------------------------------------------------- #

def test_loop_tool_call_then_answer():
    """Happy path: model searches once, gets results, returns text answer."""
    from loop import run_sub_model_loop

    responses = [
        _make_response(_make_msg(
            content=None,
            tool_calls=[{"name": "session_search", "args": {"query": "test"}}],
        )),
        _make_response(_make_msg(content="The answer is 42.")),
    ]

    def fake_dispatch(args, db, session_id):
        return json.dumps({"success": True, "mode": "discover", "results": []})

    with patch("loop.call_llm", side_effect=responses) as mock_llm, \
         patch("loop._dispatch_session_search", side_effect=fake_dispatch):
        result = run_sub_model_loop("test query", FakeDB())

    assert result == "The answer is 42."
    assert mock_llm.call_count == 2
    # First call had tools.
    assert mock_llm.call_args_list[0].kwargs.get("tools") is not None
    # Second call had tools too (the loop always passes tools until
    # the model returns text).
    assert mock_llm.call_args_list[1].kwargs.get("tools") is not None


def test_loop_immediate_text_answer():
    """Model returns text immediately without any tool calls."""
    from loop import run_sub_model_loop

    responses = [
        _make_response(_make_msg(content="No search needed, I know this.")),
    ]

    with patch("loop.call_llm", side_effect=responses) as mock_llm, \
         patch("loop._dispatch_session_search") as mock_dispatch:
        result = run_sub_model_loop("test query", FakeDB())

    assert result == "No search needed, I know this."
    assert mock_llm.call_count == 1
    mock_dispatch.assert_not_called()


def test_loop_max_iterations_synthesis():
    """When max iterations is reached, the loop forces a text synthesis."""
    from loop import run_sub_model_loop

    # Model keeps making tool calls, never returns text.
    tool_response = _make_response(_make_msg(
        content=None,
        tool_calls=[{"name": "session_search", "args": {"query": "more"}}],
    ))
    synthesis_response = _make_response(_make_msg(
        content="Synthesized answer from limited data."
    ))
    # max_iterations=2 → 2 tool-call rounds + 1 synthesis call = 3 total.
    responses = [tool_response, tool_response, synthesis_response]

    def fake_dispatch(args, db, session_id):
        return json.dumps({"success": True, "mode": "discover", "results": []})

    with patch("loop.call_llm", side_effect=responses) as mock_llm, \
         patch("loop._dispatch_session_search", side_effect=fake_dispatch):
        result = run_sub_model_loop("test query", FakeDB(), max_iterations=2)

    assert result == "Synthesized answer from limited data."
    # 2 loop iterations + 1 synthesis = 3 LLM calls.
    assert mock_llm.call_count == 3
    # The last call should NOT have tools (forcing text synthesis).
    last_call = mock_llm.call_args_list[-1]
    assert last_call.kwargs.get("tools") is None


def test_loop_timeout_synthesis():
    """Wall-clock timeout breaks the loop and forces synthesis."""
    from loop import run_sub_model_loop

    tool_response = _make_response(_make_msg(
        content=None,
        tool_calls=[{"name": "session_search", "args": {"query": "slow"}}],
    ))
    synthesis_response = _make_response(_make_msg(
        content="Timed out answer."
    ))
    responses = [tool_response, synthesis_response]

    def fake_dispatch(args, db, session_id):
        return json.dumps({"success": True, "mode": "discover", "results": []})

    # Patch time.monotonic: t=0 (start), t=1 (first deadline check,
    # under timeout → iteration runs), t=100 (second deadline check,
    # over timeout → break to synthesis).
    with patch("loop.call_llm", side_effect=responses) as mock_llm, \
         patch("loop._dispatch_session_search", side_effect=fake_dispatch), \
         patch("loop.time.monotonic", side_effect=[0, 1, 100, 100, 100]):
        result = run_sub_model_loop("test query", FakeDB(), max_iterations=10)

    assert result == "Timed out answer."
    # Only 1 tool-call round + 1 synthesis = 2 calls (timeout broke
    # after the first iteration).
    assert mock_llm.call_count == 2
    last_call = mock_llm.call_args_list[-1]
    assert last_call.kwargs.get("tools") is None


def test_loop_unknown_tool_rejected():
    """Model calls a tool that isn't session_search — loop sends error back."""
    from loop import run_sub_model_loop

    responses = [
        _make_response(_make_msg(
            content=None,
            tool_calls=[{"name": "web_search", "args": {"q": "test"}}],
        )),
        _make_response(_make_msg(content="Answer after error.")),
    ]

    with patch("loop.call_llm", side_effect=responses), \
         patch("loop._dispatch_session_search") as mock_dispatch:
        result = run_sub_model_loop("test query", FakeDB())

    assert result == "Answer after error."
    # _dispatch_session_search should NOT have been called for the
    # unknown tool — the loop handles it inline.
    mock_dispatch.assert_not_called()


def test_loop_malformed_tool_arguments():
    """Malformed JSON in tool arguments doesn't crash the loop."""
    from loop import run_sub_model_loop

    # Build a tool call with invalid JSON arguments.
    bad_msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="call_0",
            type="function",
            function=SimpleNamespace(
                name="session_search",
                arguments="not valid json {{{",
            ),
        )],
    )
    responses = [
        _make_response(bad_msg),
        _make_response(_make_msg(content="Recovered from bad args.")),
    ]

    with patch("loop.call_llm", side_effect=responses), \
         patch("loop._dispatch_session_search") as mock_dispatch:
        result = run_sub_model_loop("test query", FakeDB())

    assert result == "Recovered from bad args."
    # _dispatch_session_search was called with empty args dict.
    mock_dispatch.assert_called_once()
    call_args = mock_dispatch.call_args[0]
    assert call_args[0] == {}


def test_loop_multiple_tool_calls_per_turn():
    """Model makes multiple tool calls in one turn — all are dispatched."""
    from loop import run_sub_model_loop

    responses = [
        _make_response(_make_msg(
            content=None,
            tool_calls=[
                {"name": "session_search", "args": {"query": "first"}},
                {"name": "session_search", "args": {"query": "second"}},
            ],
        )),
        _make_response(_make_msg(content="Combined answer.")),
    ]

    with patch("loop.call_llm", side_effect=responses), \
         patch("loop._dispatch_session_search") as mock_dispatch:
        result = run_sub_model_loop("test query", FakeDB())

    assert result == "Combined answer."
    assert mock_dispatch.call_count == 2


# --------------------------------------------------------------------------- #
# Import resolution
# --------------------------------------------------------------------------- #

def test_imports():
    """Verify all key imports resolve."""
    from schemas import RECALL_SCHEMA
    from loop import (
        run_sub_model_loop,
        SESSION_SEARCH_TOOL,
        SUB_MODEL_SYSTEM,
        _build_messages,
    )
    from recall_tools import execute_recall, register

    assert callable(run_sub_model_loop)
    assert callable(execute_recall)
    assert callable(register)
    assert RECALL_SCHEMA["name"] == "recall"
    assert SESSION_SEARCH_TOOL["function"]["name"] == "session_search"
    assert "session_search" in SUB_MODEL_SYSTEM
