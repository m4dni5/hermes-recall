"""Sub-model reasoning loop — iterates session_search calls until FINAL(answer).

The sub-model gets session_search as its only tool. It discovers matching
sessions, scrolls into specific messages, and synthesizes an answer.
"""

import json
import re
from typing import Any, Dict, List, Optional

_MAX_ITERATIONS = 8
_MAX_LLM_TOKENS = 2048

# ---------------------------------------------------------------------------
# Sub-model system prompt
# ---------------------------------------------------------------------------

SUB_MODEL_SYSTEM = """You are a research sub-agent tasked with answering a question by searching archived conversation history.

You have access to ONE tool: session_search. This is Hermes's built-in session search tool. It has three modes:

1. DISCOVERY — session_search(query="your keywords")
   - Runs FTS5 full-text search over all past conversations
   - Returns matching sessions with message snippets, bookends (first/last 3 messages of each session), and a window of messages around each match
   - Use this FIRST to find which conversations are relevant

2. SCROLL — session_search(session_id="...", around_message_id=12345, window=10)
   - Zoom into a specific session around a specific message
   - Use this when you found a relevant match and need more context around it
   - window defaults to 5, max 20

3. BROWSE — session_search() with no args
   - Lists recent sessions (titles, previews, timestamps)
   - Use this when you're not sure what keywords to search for

HOW TO WORK:
1. Start with discovery — search for keywords related to the question
2. Read the results — bookends tell you what the session was about, messages give you the match in context
3. If you need more detail on a match, scroll deeper into that session
4. When you have enough information, emit FINAL(your complete answer)

OUTPUT FORMAT — you MUST use EXACTLY this JSON format for tool calls:

{"tool": "session_search", "args": {"query": "keywords here"}}
{"tool": "session_search", "args": {"session_id": "...", "around_message_id": 12345, "window": 10}}

When you have enough information to answer, emit:
FINAL(your complete answer here — can span multiple sentences, be thorough)

RULES:
- Write ONLY one JSON tool call OR one FINAL per response — never both
- Do NOT write code, do NOT write Python, do NOT use code blocks
- The only tool you have is session_search — do not invent other tools
- If your first search returns nothing useful, try different keywords
- Answer the question directly and completely in your FINAL"""


def _build_system_prompt(query: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SUB_MODEL_SYSTEM}]


def _build_user_prompt(query: str, iteration: int = 0) -> Dict[str, str]:
    if iteration == 0:
        return {
            "role": "user",
            "content": f'Start by searching for relevant sessions. Question: "{query}"\n\nYour next action:',
        }
    return {
        "role": "user",
        "content": f'Continue researching. Question: "{query}"\n\nYour next action:',
    }


# ---------------------------------------------------------------------------
# JSON tool-call parsing
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}\s*\}', re.DOTALL
)


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract a single JSON tool call from the sub-model's response.

    Handles both bare JSON and JSON inside ```json fences.
    """
    fence_match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    for match in _TOOL_CALL_RE.finditer(text):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    return None


def _find_final(text: str) -> Optional[str]:
    """Extract FINAL(answer) from the sub-model's response."""
    m = re.search(r"FINAL\((.+)\)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_sub_model_loop(
    query: str,
    db,
    current_session_id: str = "",
    max_iterations: int = _MAX_ITERATIONS,
) -> str:
    """Run the sub-model reasoning loop with session_search as its only tool.

    The sub-model iterates: search → read results → decide (search more / FINAL).
    The ``db`` parameter is a hermes_state.SessionDB instance.
    """
    from .tools import _dispatch_session_search

    from agent.auxiliary_client import call_llm

    messages = _build_system_prompt(query)

    for i in range(max_iterations):
        messages.append(_build_user_prompt(query, i))

        response = call_llm(
            task="rlm",
            main_runtime={},
            messages=messages,
            max_tokens=_MAX_LLM_TOKENS,
        )
        content = response.choices[0].message.content or ""
        content = content.strip()

        final = _find_final(content)
        if final:
            return final

        tool_call = _parse_tool_call(content)
        if tool_call:
            tool_name = tool_call.get("tool", "")
            if tool_name == "session_search":
                args = tool_call.get("args", {})
                result = _dispatch_session_search(args, db, current_session_id)
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"session_search result:\n\n{result}",
                })
            else:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Unknown tool '{tool_name}'. You only have session_search. Try again.",
                })
        else:
            messages.append({"role": "assistant", "content": content})

    # Max iterations — synthesize
    synthesis = (
        f"Based on your research, answer this question concisely: {query}\n\n"
        "You MUST respond with: FINAL(your complete answer)"
    )
    messages.append({"role": "user", "content": synthesis})
    response = call_llm(
        task="rlm",
        main_runtime={},
        messages=messages,
        max_tokens=_MAX_LLM_TOKENS,
    )
    content = response.choices[0].message.content or ""
    final = _find_final(content)
    return final if final else content.strip()
