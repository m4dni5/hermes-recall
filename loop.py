"""Sub-model reasoning loop — native tool calling over session_search.

The sub-model gets session_search as a native function-calling tool via
``call_llm(tools=[...])``. The model decides when to search, the loop
dispatches the call and feeds the result back, and the model either
searches again or returns a text answer. No regex parsing, no FINAL
convention — the OpenAI tool-calling protocol handles structured
communication between the model and the loop.

Termination: the model returns a response without tool_calls → that text
is the answer. If max iterations is reached first, a final call without
tools forces a text synthesis.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 5
_MAX_LLM_TOKENS = 2048

# --------------------------------------------------------------------------- #
# Module-level imports with fallback for test environments.
#
# In production (package mode), ``from .tools import _dispatch_session_search``
# resolves within the plugin package. In test mode (bare modules on
# pythonpath=["."]), the relative import fails and the fallback
# ``from tools import _dispatch_session_search`` picks up the plugin's own
# tools.py. ``call_llm`` is patched in tests; the try/except keeps the
# module importable when the framework isn't on the path.
# --------------------------------------------------------------------------- #

try:
    from agent.auxiliary_client import call_llm  # type: ignore[import]
except ImportError:
    call_llm = None  # type: ignore[assignment]

try:
    from .recall_tools import _dispatch_session_search  # type: ignore[import]
except ImportError:
    try:
        from recall_tools import _dispatch_session_search  # type: ignore[import,no-redef]
    except ImportError:
        _dispatch_session_search = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Tool schema — what the sub-model sees via native function calling.
# --------------------------------------------------------------------------- #

SESSION_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "session_search",
        "description": (
            "Search archived conversation history. Three modes: "
            "(1) pass query for FTS5 keyword search — returns matching "
            "sessions with snippets and message windows; (2) pass "
            "session_id + around_message_id to scroll into a specific "
            "session around a message; (3) pass nothing to browse recent "
            "sessions. Use discovery first, then scroll for detail."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for across all past conversations.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID to scroll into (from a discovery result).",
                },
                "around_message_id": {
                    "type": "integer",
                    "description": "Message ID to center the scroll window on.",
                },
                "window": {
                    "type": "integer",
                    "description": "Messages on each side of anchor (1-20). Default 5.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Sessions to return from discovery (1-10). Default 3.",
                },
            },
        },
    },
}


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

SUB_MODEL_SYSTEM = (
    "You are a research sub-agent tasked with answering a question by "
    "searching archived conversation history.\n\n"
    "You have one tool: session_search. It searches past conversations.\n\n"
    "How to work:\n"
    "1. Start with discovery — search for keywords related to the question\n"
    "2. Read the results — bookends tell you what the session was about, "
    "messages give you the match in context\n"
    "3. If you need more detail, scroll deeper into a matching session "
    "using session_id + around_message_id\n"
    "4. When you have enough information, respond with your answer as "
    "plain text (no tool call)\n\n"
    "If your first search returns nothing useful, try different keywords. "
    "Answer the question directly and completely."
)


def _build_messages(query: str) -> list[dict[str, Any]]:
    """Build the initial message list: system prompt + user query."""
    return [
        {"role": "system", "content": SUB_MODEL_SYSTEM},
        {
            "role": "user",
            "content": (
                "Answer this question using archived conversation "
                f"history: {query}"
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

def run_sub_model_loop(
    query: str,
    db: Any,
    current_session_id: str = "",
    max_iterations: int = _MAX_ITERATIONS,
) -> str:
    """Run the sub-model reasoning loop with session_search as a native tool.

    The sub-model iterates: search → read results → decide (search more /
    answer). The loop terminates when the model returns text without a
    tool call, or when max iterations is reached (at which point we prompt
    for synthesis without tools, forcing a text answer).

    The ``db`` parameter is a hermes_state.SessionDB instance.
    """
    messages = _build_messages(query)

    for _ in range(max_iterations):
        response = call_llm(
            task="recall",
            main_runtime={},
            messages=messages,
            tools=[SESSION_SEARCH_TOOL],
            max_tokens=_MAX_LLM_TOKENS,
        )
        msg = response.choices[0].message
        content = (msg.content or "").strip()

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # Append the assistant message (with tool_calls) so the model
            # sees its own request in context. The OpenAI API requires the
            # assistant message to carry the tool_calls it made, and each
            # tool result to reference the matching tool_call_id.
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}

                if tool_name == "session_search":
                    result = _dispatch_session_search(
                        args, db, current_session_id)
                else:
                    result = json.dumps({
                        "error": (
                            f"Unknown tool '{tool_name}'. "
                            "You only have session_search."
                        ),
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        # No tool calls — the text response IS the answer.
        return content

    # Max iterations reached — prompt for synthesis without tools so
    # the model must return text.
    logger.info(
        "recall: sub-model reached max_iterations (%d), synthesizing",
        max_iterations,
    )
    messages.append({
        "role": "user",
        "content": (
            "You've reached the search limit. Answer the question "
            f"concisely based on what you've found: {query}"
        ),
    })
    response = call_llm(
        task="recall",
        main_runtime={},
        messages=messages,
        max_tokens=_MAX_LLM_TOKENS,
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()
