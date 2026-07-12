"""Recall tool plumbing — session DB access, aux model fallback, plugin registration."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _call_aux_model(prompt: str, max_tokens: int = 1024) -> str:
    """Call the recall auxiliary model (auxiliary.recall in config.yaml)."""
    from agent.auxiliary_client import call_llm

    response = call_llm(
        task="recall",
        main_runtime={},
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content or "")


def _get_session_db(hermes_home: Optional[str] = None):
    """Open a read-only SessionDB connection to state.db.

    SessionDB with read_only=True skips _init_schema() entirely (it
    returns early to avoid taking a write lock), which means
    ``_fts_enabled`` stays ``False`` and all FTS5 searches silently
    return zero results. We probe for the FTS5 table after opening and
    set the flag manually — the table exists if a read-write SessionDB
    ever opened this DB (i.e. Hermes has run at least once).
    """
    try:
        from hermes_state import SessionDB, DEFAULT_DB_PATH

        db_path = Path(hermes_home) / "state.db" if hermes_home else DEFAULT_DB_PATH
        if not db_path.exists():
            logger.warning("recall: state.db not found at %s", db_path)
            return None
        db = SessionDB(db_path, read_only=True)
        # SessionDB(read_only=True) skips _init_schema(), leaving
        # _fts_enabled=False. Probe the FTS5 table and flip the flag
        # so search_messages() doesn't short-circuit to [].
        conn = getattr(db, "_conn", None)
        if conn is not None and not db._fts_enabled:
            try:
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='messages_fts'"
                ).fetchone()
                if row:
                    db._fts_enabled = True
                    tri = conn.execute(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type='table' AND name='messages_fts_trigram'"
                    ).fetchone()
                    db._trigram_available = tri is not None
            except Exception:
                pass
        return db
    except Exception as exc:
        logger.warning("recall: Failed to open SessionDB: %s", exc)
        return None


def _dispatch_session_search(args: dict, db, current_session_id: str = "") -> str:
    """Call Hermes's session_search tool and return the raw JSON result.

    The sub-agent uses native tool calling — it handles structured JSON
    natively, so no formatting layer is needed. The raw result includes
    tool_calls, snippets, bookends, anchors, and every field the session
    DB exposes.
    """
    from tools.session_search_tool import session_search

    try:
        return session_search(
            query=args.get("query", ""),
            session_id=args.get("session_id"),
            around_message_id=args.get("around_message_id"),
            window=args.get("window", 5),
            limit=args.get("limit", 3),
            db=db,
            current_session_id=current_session_id,
        )
    except Exception as exc:
        logger.warning("recall: session_search dispatch failed: %s", exc)
        return json.dumps({"error": str(exc)})


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


def _recall_available() -> bool:
    """Check if state.db exists so the tool can be hidden when it can't work."""
    try:
        from hermes_constants import get_hermes_home
        from pathlib import Path
        return (Path(get_hermes_home()) / "state.db").exists()
    except Exception:
        return False


def register(ctx):
    """Register recall as a regular plugin tool."""
    from .schemas import RECALL_SCHEMA

    if hasattr(ctx, "register_auxiliary_task"):
        ctx.register_auxiliary_task(
            key="recall",
            display_name="Recall retrieval",
            description="Sub-model reasoning loop for archived session search",
            defaults={"provider": "auto", "model": "", "timeout": 120},
        )
        logger.info("recall: registered auxiliary.recall task")

    if hasattr(ctx, "register_tool"):
        ctx.register_tool(
            name="recall",
            toolset="recall",
            schema=RECALL_SCHEMA,
            handler=_handle_recall_tool,
            check_fn=_recall_available,
            description="Search past conversation history using sub-model reasoning loop",
            emoji="🧠",
        )
    logger.info("recall: registered recall tool")