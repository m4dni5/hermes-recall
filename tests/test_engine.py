"""Unit tests for recall — parsing, schema, imports."""

import pytest


def test_schema_valid():
    from schemas import RECALL_SCHEMA

    assert RECALL_SCHEMA["name"] == "recall"
    assert "query" in RECALL_SCHEMA["parameters"]["properties"]
    assert RECALL_SCHEMA["parameters"]["required"] == ["query"]


@pytest.mark.parametrize(
    "text,expected_tool,expected_args",
    [
        ('{"tool": "session_search", "args": {"query": "test"}}',
         "session_search", {"query": "test"}),
        ('```json\n{"tool": "session_search", "args": {"query": "fenced"}}\n```',
         "session_search", {"query": "fenced"}),
        ('```\n{"tool": "session_search", "args": {"query": "no lang"}}\n```',
         "session_search", {"query": "no lang"}),
        ('some text {"tool": "session_search", "args": {"session_id": "abc", "around_message_id": 123}} trailing',
         "session_search", {"session_id": "abc", "around_message_id": 123}),
    ],
)
def test_parse_tool_call(text, expected_tool, expected_args):
    from loop import _parse_tool_call

    result = _parse_tool_call(text)
    assert result is not None
    assert result["tool"] == expected_tool
    assert result["args"] == expected_args


def test_parse_tool_call_no_match():
    from loop import _parse_tool_call

    assert _parse_tool_call("just some text") is None
    assert _parse_tool_call("") is None
    assert _parse_tool_call('{"no_tool_key": "value"}') is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("FINAL(the answer)", "the answer"),
        ("FINAL(multi-line\nanswer here)", "multi-line\nanswer here"),
        ("some text FINAL(result) more text", "result"),
    ],
)
def test_find_final(text, expected):
    from loop import _find_final

    assert _find_final(text) == expected


def test_find_final_no_match():
    from loop import _find_final

    assert _find_final("no final here") is None
    assert _find_final("") is None


def test_imports():
    """Verify all key imports resolve."""
    from schemas import RECALL_SCHEMA
    from loop import run_sub_model_loop, _parse_tool_call, _find_final
    from tools import execute_recall, register

    assert callable(run_sub_model_loop)
    assert callable(execute_recall)
    assert callable(register)
    assert RECALL_SCHEMA["name"] == "recall"


def test_build_system_prompt():
    from loop import _build_system_prompt

    msgs = _build_system_prompt("test query")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert "session_search" in msgs[0]["content"]
    assert "FINAL" in msgs[0]["content"]


def test_build_user_prompt():
    from loop import _build_user_prompt

    msg = _build_user_prompt("test", iteration=0)
    assert msg["role"] == "user"
    assert "test" in msg["content"]

    msg2 = _build_user_prompt("test", iteration=3)
    assert msg2["role"] == "user"
    assert "Continue" in msg2["content"]