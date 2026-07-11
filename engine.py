"""RLM Hermes Plugin — encapsulated session search via sub-model reasoning loop.

Single tool: rlm_search(query). The sub-model iterates:
  session_search(query="keywords")          → discover matching sessions
  session_search(session_id, around_message_id) → scroll deeper
  FINAL(answer)                             → return synthesized answer

The sub-model uses Hermes's built-in session_search as its only tool.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_ITERATIONS = 8
_MAX_LLM_TOKENS = 2048

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

RLM_SEARCH_SCHEMA = {
    "name": "rlm_search",
    "description": (
        "Search archived conversation history via an encapsulated sub-model. "
        "The sub-model explores past sessions using FTS5 search and returns "
        "a synthesized answer. Use when you need context from earlier "
        "conversations that's no longer in your active window."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question about past conversation context",
            },
        },
        "required": ["query"],
    },
}

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

_TOOL_CALL_RE = re.compile(r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}\s*\}', re.DOTALL)


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract a single JSON tool call from the sub-model's response.

    Handles both bare JSON and JSON inside ```json fences.
    """
    # Try ```json fence first
    fence_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON objects
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    return None


def _find_final(text: str) -> Optional[str]:
    """Extract FINAL(answer) from the sub-model's response."""
    m = re.search(r'FINAL\((.+)\)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_aux_model(prompt: str, max_tokens: int = 1024) -> str:
    """Call the RLM auxiliary model (auxiliary.rlm in config.yaml)."""
    from agent.auxiliary_client import call_llm

    response = call_llm(
        task="rlm",
        main_runtime={},
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content or "")


def _get_session_db(hermes_home: Optional[str] = None):
    """Open a read-only SessionDB connection to state.db."""
    try:
        from hermes_state import SessionDB, DEFAULT_DB_PATH

        db_path = Path(hermes_home) / "state.db" if hermes_home else DEFAULT_DB_PATH
        if not db_path.exists():
            logger.warning("RLM: state.db not found at %s", db_path)
            return None
        return SessionDB(db_path, read_only=True)
    except Exception as exc:
        logger.warning("RLM: Failed to open SessionDB: %s", exc)
        return None


def _dispatch_session_search(args: dict, db, current_session_id: str = "") -> str:
    """Call Hermes's session_search tool and return a readable summary."""
    from tools.session_search_tool import session_search

    try:
        result_json = session_search(
            query=args.get("query", ""),
            session_id=args.get("session_id"),
            around_message_id=args.get("around_message_id"),
            window=args.get("window", 5),
            limit=args.get("limit", 3),
            db=db,
            current_session_id=current_session_id,
        )
        result = json.loads(result_json)
        return _format_session_search_result(result, args)
    except Exception as exc:
        logger.warning("RLM: session_search dispatch failed: %s", exc)
        return json.dumps({"error": str(exc)})


def _format_session_search_result(result: dict, args: dict) -> str:
    """Format a session_search result for the sub-model to read."""
    if not result.get("success", True):
        return json.dumps(result)

    mode = result.get("mode", "discover")

    if mode == "discover":
        parts = [f"Found {result.get('count', 0)} matching sessions for: {args.get('query', '')}"]
        for i, hit in enumerate(result.get("results", [])[:5]):
            parts.append(f"\n--- Session {i+1}: {hit.get('title') or hit.get('session_id', '?')} ---")
            parts.append(f"When: {hit.get('when', 'unknown')} | Source: {hit.get('source', '?')}")
            if hit.get("snippet"):
                parts.append(f"Snippet: {hit['snippet'][:500]}")
            # Bookend start (first messages)
            bookend = hit.get("bookend_start", [])
            if bookend:
                parts.append("Opening messages:")
                for msg in bookend[:3]:
                    parts.append(f"  [{msg.get('role', '?')}] {msg.get('content', '')[:300]}")
            # Matching window
            msgs = hit.get("messages", [])
            if msgs:
                parts.append("Messages around match:")
                for msg in msgs:
                    anchor = " ▶" if msg.get("anchor") else ""
                    parts.append(f"  [{msg.get('role', '?')}{anchor}] {msg.get('content', '')[:400]}")
            # Bookend end
            bookend = hit.get("bookend_end", [])
            if bookend:
                parts.append("Closing messages:")
                for msg in bookend[:3]:
                    parts.append(f"  [{msg.get('role', '?')}] {msg.get('content', '')[:300]}")
        return "\n".join(parts)

    elif mode == "scroll":
        parts = [f"Scrolled session {result.get('session_id', '?')} around message {result.get('around_message_id')}"]
        msgs = result.get("messages", [])
        if msgs:
            for msg in msgs:
                anchor = " ▶" if msg.get("anchor") else ""
                parts.append(f"  [{msg.get('role', '?')}{anchor}] {msg.get('content', '')[:500]}")
        before = result.get("messages_before", 0)
        after = result.get("messages_after", 0)
        if before or after:
            parts.append(f"({before} messages before, {after} messages after this window)")
        return "\n".join(parts)

    elif mode == "read":
        parts = [f"Session: {result.get('session_meta', {}).get('title') or result.get('session_id', '?')}"]
        parts.append(f"Total messages: {result.get('message_count', 0)}")
        for msg in result.get("messages", []):
            parts.append(f"  [{msg.get('role', '?')}] {msg.get('content', '')[:500]}")
        return "\n".join(parts)

    elif mode == "browse":
        parts = [f"Recent sessions ({result.get('count', 0)}):"]
        for s in result.get("results", []):
            parts.append(f"  {s.get('title') or s.get('session_id', '?')} — {s.get('message_count', 0)} msgs, {s.get('started_at', '?')}")
        return "\n".join(parts)

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Sub-model loop
# ---------------------------------------------------------------------------

def run_sub_model_loop(
    query: str,
    db,
    current_session_id: str = "",
    max_iterations: int = _MAX_ITERATIONS,
) -> str:
    """Run the sub-model reasoning loop with session_search as its only tool.

    The sub-model iterates: search → read results → decide (search more / FINAL).
    """
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

        # Check for FINAL first
        final = _find_final(content)
        if final:
            return final

        # Parse tool call
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
            # No tool call and no FINAL — just reasoning, prompt for next action
            messages.append({"role": "assistant", "content": content})

    # Max iterations — synthesize from accumulated results
    synthesis_prompt = (
        f"Based on your research, answer this question concisely: {query}\n\n"
        "You MUST respond with: FINAL(your complete answer)"
    )
    messages.append({"role": "user", "content": synthesis_prompt})
    response = call_llm(
        task="rlm",
        main_runtime={},
        messages=messages,
        max_tokens=_MAX_LLM_TOKENS,
    )
    content = response.choices[0].message.content or ""
    final = _find_final(content)
    if final:
        return final
    return content.strip()


# ---------------------------------------------------------------------------
# rlm_search — the plugin tool
# ---------------------------------------------------------------------------

def execute_rlm_search(
    query: str,
    session_id: str = "",
    hermes_home: Optional[str] = None,
) -> str:
    """Execute rlm_search — the plugin's sole tool."""
    if not query.strip():
        return json.dumps({"error": "query is required"})

    db = _get_session_db(hermes_home)
    if db is None:
        return json.dumps({"error": "Session database not available"})

    try:
        answer = run_sub_model_loop(
            query=query,
            db=db,
            current_session_id=session_id,
        )
    except Exception as exc:
        logger.warning("RLM sub-model loop failed, falling back to direct aux model: %s", exc)
        try:
            # Fallback: direct FTS5 + aux model synthesis
            from tools.session_search_tool import session_search

            sr = json.loads(session_search(query=query, db=db, limit=5))
            context_parts = []
            for hit in sr.get("results", [])[:3]:
                for msg in hit.get("messages", []):
                    context_parts.append(f"[{msg.get('role', '?')}] {msg.get('content', '')[:500]}")
            context = "\n\n".join(context_parts)[:50000]
            answer = _call_aux_model(
                f"Answer using these archived messages.\n\nQuestion: {query}\n\nMessages:\n{context}",
                max_tokens=2048,
            )
        except Exception as exc2:
            return json.dumps({"error": f"Sub-model loop and fallback both failed: {exc2}"})

    return json.dumps({
        "answer": answer,
        "method": "rlm",
    })


# ---------------------------------------------------------------------------
# Tool handler (for plugin registration)
# ---------------------------------------------------------------------------

def _handle_rlm_search_tool(args: dict, **kwargs) -> str:
    """Handler for rlm_search when registered as a regular plugin tool."""
    session_id = kwargs.get("session_id", "")
    hermes_home = None
    try:
        from hermes_constants import get_hermes_home

        hermes_home = str(get_hermes_home())
    except Exception:
        pass
    return execute_rlm_search(
        query=args.get("query", ""),
        session_id=session_id,
        hermes_home=hermes_home,
    )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx):
    """Register RLM as a regular plugin tool."""
    # Register auxiliary.rlm task for sub-model calls
    if hasattr(ctx, "register_auxiliary_task"):
        ctx.register_auxiliary_task(
            key="rlm",
            display_name="RLM retrieval",
            description="Sub-model reasoning loop for archived session search",
            defaults={
                "provider": "auto",
                "model": "",
                "timeout": 120,
            },
        )
        logger.info("RLM: registered auxiliary.rlm task")

    # Register rlm_search tool
    if hasattr(ctx, "register_tool"):
        ctx.register_tool(
            name="rlm_search",
            toolset="rlm",
            schema=RLM_SEARCH_SCHEMA,
            handler=_handle_rlm_search_tool,
            description="Search archived conversation context using sub-model reasoning loop",
            emoji="🔍",
        )
    logger.info("RLM: registered rlm_search tool")
