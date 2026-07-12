"""Recall tool plumbing — session DB access, aux model fallback, plugin registration."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _call_aux_model(prompt: str, max_tokens: int = 1024) -> str:
    """Call the recall auxiliary model (auxiliary.rlm in config.yaml)."""
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
            logger.warning("recall: state.db not found at %s", db_path)
            return None
        return SessionDB(db_path, read_only=True)
    except Exception as exc:
        logger.warning("recall: Failed to open SessionDB: %s", exc)
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
        logger.warning("recall: session_search dispatch failed: %s", exc)
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
            bookend = hit.get("bookend_start", [])
            if bookend:
                parts.append("Opening messages:")
                for msg in bookend[:3]:
                    parts.append(f"  [{msg.get('role', '?')}] {msg.get('content', '')[:300]}")
            msgs = hit.get("messages", [])
            if msgs:
                parts.append("Messages around match:")
                for msg in msgs:
                    anchor = " ▶" if msg.get("anchor") else ""
                    parts.append(f"  [{msg.get('role', '?')}{anchor}] {msg.get('content', '')[:400]}")
            bookend = hit.get("bookend_end", [])
            if bookend:
                parts.append("Closing messages:")
                for msg in bookend[:3]:
                    parts.append(f"  [{msg.get('role', '?')}] {msg.get('content', '')[:300]}")
        return "\n".join(parts)

    elif mode == "scroll":
        parts = [
            f"Scrolled session {result.get('session_id', '?')} "
            f"around message {result.get('around_message_id')}"
        ]
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
        parts = [
            f"Session: {result.get('session_meta', {}).get('title') or result.get('session_id', '?')}"
        ]
        parts.append(f"Total messages: {result.get('message_count', 0)}")
        for msg in result.get("messages", []):
            parts.append(f"  [{msg.get('role', '?')}] {msg.get('content', '')[:500]}")
        return "\n".join(parts)

    elif mode == "browse":
        parts = [f"Recent sessions ({result.get('count', 0)}):"]
        for s in result.get("results", []):
            parts.append(
                f"  {s.get('title') or s.get('session_id', '?')} — "
                f"{s.get('message_count', 0)} msgs, {s.get('started_at', '?')}"
            )
        return "\n".join(parts)

    return json.dumps(result)


def execute_recall(
    query: str,
    session_id: str = "",
    hermes_home: Optional[str] = None,
) -> str:
    """Execute recall — the plugin's sole tool."""
    if not query.strip():
        return json.dumps({"error": "query is required"})

    db = _get_session_db(hermes_home)
    if db is None:
        return json.dumps({"error": "Session database not available"})

    try:
        from .loop import run_sub_model_loop

        answer = run_sub_model_loop(
            query=query,
            db=db,
            current_session_id=session_id,
        )
    except Exception as exc:
        logger.warning("recall sub-model loop failed, falling back to direct aux model: %s", exc)
        try:
            from tools.session_search_tool import session_search

            sr = json.loads(session_search(query=query, db=db, limit=5))
            context_parts = []
            for hit in sr.get("results", [])[:3]:
                for msg in hit.get("messages", []):
                    context_parts.append(
                        f"[{msg.get('role', '?')}] {msg.get('content', '')[:500]}"
                    )
            context = "\n\n".join(context_parts)[:50000]
            answer = _call_aux_model(
                f"Answer using these archived messages.\n\n"
                f"Question: {query}\n\nMessages:\n{context}",
                max_tokens=2048,
            )
        except Exception as exc2:
            return json.dumps({"error": f"Sub-model loop and fallback both failed: {exc2}"})

    return json.dumps({"answer": answer, "method": "recall"})


def _handle_recall_tool(args: dict, **kwargs) -> str:
    """Handler for recall when registered as a regular plugin tool."""
    session_id = kwargs.get("session_id", "")
    hermes_home = None
    try:
        from hermes_constants import get_hermes_home

        hermes_home = str(get_hermes_home())
    except Exception:
        pass
    return execute_recall(
        query=args.get("query", ""),
        session_id=session_id,
        hermes_home=hermes_home,
    )


def register(ctx):
    """Register recall as a regular plugin tool."""
    from .schemas import RECALL_SCHEMA

    if hasattr(ctx, "register_auxiliary_task"):
        ctx.register_auxiliary_task(
            key="rlm",
            display_name="Recall retrieval",
            description="Sub-model reasoning loop for archived session search",
            defaults={"provider": "auto", "model": "", "timeout": 120},
        )
        logger.info("recall: registered auxiliary.rlm task")

    if hasattr(ctx, "register_tool"):
        ctx.register_tool(
            name="recall",
            toolset="recall",
            schema=RECALL_SCHEMA,
            handler=_handle_recall_tool,
            description="Search past conversation history using sub-model reasoning loop",
            emoji="🧠",
        )
    logger.info("recall: registered recall tool")