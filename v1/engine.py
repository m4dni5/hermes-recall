"""RLM Hermes Plugin — two modes of operation.

MODE 1: Context Engine (context.engine: "rlm" in config.yaml)
  Replaces the built-in compressor. compress() trims to tail, rlm_search
  runs the REPL-based deep dive, pre_llm_call hook auto-retrieves.

MODE 2: Regular Plugin (just install as a plugin)
  Exposes rlm_search as a tool alongside the DEFAULT compressor.
  Messages are compressed normally by the built-in ContextCompressor,
  but ALL original messages are persisted to state.db before compression.
  rlm_search queries state.db to recover them.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DEFAULT_PROTECT_LAST_N = 20
_DEFAULT_SEARCH_LIMIT = 50
_DEFAULT_MAX_AGG_TOKENS = 4096

_CONTEXT_NOTE = """[CONTEXT ARCHIVED — RLM RETRIEVAL ACTIVE]

Earlier conversation turns have been removed from your active context window
but are fully preserved in the session archive. Two retrieval mechanisms are
active:

1. AUTOMATIC: Relevant archived context is injected into each turn via a
   pre_llm_call hook. You don't need to do anything — it's already here.

2. MANUAL (rlm_search tool): For deep dives, use:
   rlm_search(query="specific question about past context")
   - scope="current" searches this conversation's history (default)
   - scope="all" searches across all conversations
   - sort="newest"/"oldest"/"relevance"
   - The REPL model writes code to process archived messages.

IMPORTANT: If you need context that isn't in the auto-retrieved section
below, call rlm_search. Don't guess or hallucinate — search the archive."""


# ===========================================================================
# Shared helpers
# ===========================================================================

def _get_session_lineage(db, session_id: str) -> List[str]:
    """Walk parent_session_id chain from session to root."""
    chain = []
    current = session_id
    seen = set()
    for _ in range(100):
        if not current or current in seen:
            break
        seen.add(current)
        chain.append(current)
        try:
            row = db._conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ?",
                (current,),
            ).fetchone()
        except Exception:
            break
        if row is None:
            break
        parent = row["parent_session_id"] if hasattr(row, "keys") else row[0]
        current = parent
    return chain


def _load_messages_from_lineage(db, session_ids: List[str]) -> str:
    """Load all messages from session lineage into a context string."""
    if not session_ids:
        return ""
    placeholders = ",".join("?" for _ in session_ids)
    with db._lock:
        rows = db._conn.execute(
            f"SELECT session_id, role, content FROM messages "
            f"WHERE session_id IN ({placeholders}) AND active = 1 ORDER BY id",
            session_ids,
        ).fetchall()

    parts = []
    for row in rows:
        sid = row["session_id"] if hasattr(row, "keys") else row[0]
        role = row["role"] if hasattr(row, "keys") else row[1]
        content = row["content"] if hasattr(row, "keys") else row[2]
        if content:
            content = db._decode_content(content)
            parts.append(f"[session:{sid} role:{role}] {content}")
    return "\n\n".join(parts)


def _get_session_db(hermes_home: Optional[Path] = None):
    """Open a read-only SessionDB connection to state.db."""
    try:
        from hermes_state import SessionDB, DEFAULT_DB_PATH
        db_path = hermes_home / "state.db" if hermes_home else DEFAULT_DB_PATH
        if not db_path.exists():
            logger.warning("RLM: state.db not found at %s", db_path)
            return None
        return SessionDB(db_path)
    except Exception as exc:
        logger.warning("RLM: Failed to open SessionDB: %s", exc)
        return None


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


def _format_message(msg: Dict[str, Any]) -> str:
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    sid = msg.get("session_id", "?")
    display = content[:500] + "..." if len(content) > 500 else content
    return f"[session:{sid} role:{role}] {display}"


# ===========================================================================
# Tool implementation (shared by both modes)
# ===========================================================================

# rlm_search tool schema
RLM_SEARCH_SCHEMA = {
    "name": "rlm_search",
    "description": (
        "Deep search of archived conversation context. Loads messages from "
        "the conversation archive and runs the RLM REPL: a model writes "
        "Python code to search, chunk, and process the data using sub-queries. "
        "Use when you need context from earlier in the conversation that's no "
        "longer in your active window. Works even after context compression."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question about archived context",
            },
            "scope": {
                "type": "string",
                "enum": ["current", "all"],
                "description": (
                    "'current' = this conversation's full history including "
                    "compressed ancestor sessions (default). "
                    "'all' = every session in the archive."
                ),
            },
        },
        "required": ["query"],
    },
}


def execute_rlm_search(
    query: str,
    scope: str = "current",
    session_id: str = "",
    hermes_home: Optional[Path] = None,
) -> str:
    """Execute rlm_search — shared by both context engine and regular plugin."""
    if not query.strip():
        return json.dumps({"error": "query is required"})

    db = _get_session_db(hermes_home)
    if db is None:
        return json.dumps({"error": "Session database not available"})

    # Load messages — full session lineage, no pre-filtering
    try:
        if scope == "current" and session_id:
            lineage = _get_session_lineage(db, session_id)
        else:
            lineage = []

        if lineage:
            context = _load_messages_from_lineage(db, lineage)
        else:
            # scope=all — load everything
            with db._lock:
                rows = db._conn.execute(
                    "SELECT session_id, role, content FROM messages "
                    "WHERE active = 1 ORDER BY id"
                ).fetchall()
            parts = []
            for row in rows:
                sid = row["session_id"] if hasattr(row, "keys") else row[0]
                role = row["role"] if hasattr(row, "keys") else row[1]
                content = row["content"] if hasattr(row, "keys") else row[2]
                if content:
                    content = db._decode_content(content)
                    parts.append(f"[session:{sid} role:{role}] {content}")
            context = "\n\n".join(parts)

        message_count = context.count("\n\n") + 1 if context else 0
    except Exception as exc:
        return json.dumps({"error": f"Failed to load messages: {exc}"})

    if not context:
        return json.dumps({"answer": "No messages found.", "results_count": 0})

    # Run the RLM REPL
    sys.path.insert(0, str(Path(__file__).parent))

    from repl import run_rlm_repl
    try:
        answer = run_rlm_repl(
            context=context, query=query,
            hermes_home=str(hermes_home) if hermes_home else None,
            session_ids=lineage if lineage else None,
        )
    except Exception as exc:
        logger.warning("RLM REPL failed, falling back to direct aux model: %s", exc)
        try:
            answer = _call_aux_model(
                f"Answer using these archived messages.\n\n"
                f"Question: {query}\n\nMessages:\n{context[:50000]}",
                max_tokens=_DEFAULT_MAX_AGG_TOKENS,
            )
        except Exception as exc2:
            return json.dumps({"error": f"REPL and fallback both failed: {exc2}"})

    return json.dumps({
        "answer": answer,
        "results_count": message_count,
        "method": "repl",
    })



# ===========================================================================
# MODE 1: Context Engine plugin
# ===========================================================================

from agent.context_engine import ContextEngine


class RLMContextEngine(ContextEngine):
    """RLM as a context engine — replaces the built-in compressor."""

    @property
    def name(self) -> str:
        return "rlm"

    def __init__(self, protect_last_n: int = _DEFAULT_PROTECT_LAST_N, **kw):
        self.protect_last_n = protect_last_n
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0
        self._session_id: str = ""
        self._hermes_home: Optional[Path] = None
        self.model = self.base_url = self.api_key = self.provider = self.api_mode = ""
        self._hook_registered = False

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        if not self.threshold_tokens:
            return False
        return (prompt_tokens or self.last_prompt_tokens) >= self.threshold_tokens

    def compress(self, messages, current_tokens=None, focus_topic=None):
        if len(messages) <= self.protect_last_n + 2:
            return messages
        head = [messages[0]] if messages and messages[0].get("role") == "system" else []
        rest = messages[len(head):]
        tail = rest[-self.protect_last_n:] if self.protect_last_n else []
        result = head + [{"role": "assistant", "content": _CONTEXT_NOTE}] + tail
        self.compression_count += 1
        logger.info("RLM compress: %d → %d", len(messages), len(result))
        return result

    def on_session_start(self, session_id, **kwargs):
        self._session_id = session_id
        hm = kwargs.get("hermes_home")
        self._hermes_home = Path(hm) if hm else None
        self._ensure_hook_registered()

    def update_model(self, model, context_length, base_url="", api_key="",
                     provider="", api_mode="", **kw):
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.provider, self.api_mode = provider, api_mode
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    def _ensure_hook_registered(self):
        if self._hook_registered:
            return
        try:
            from hermes_cli.plugins import get_plugin_manager
            get_plugin_manager()._hooks.setdefault("pre_llm_call", []).append(self._on_pre_llm_call)
            self._hook_registered = True
            logger.info("RLM: registered pre_llm_call hook")
        except Exception as exc:
            logger.debug("RLM: hook registration failed: %s", exc)

    def _on_pre_llm_call(self, session_id, user_message, conversation_history, **kwargs):
        """Auto-retrieve lightweight context via FTS5."""
        db = _get_session_db(self._hermes_home)
        if db is None:
            return None
        try:
            from hermes_state import DEFAULT_DB_PATH
            lineage = _get_session_lineage(db, self._session_id or session_id)
            context = _load_messages_from_lineage(db, lineage)
            if not context:
                return None
            # Truncate for the lightweight sub-query
            if len(context) > 100_000:
                context = context[:100_000]
            answer = _call_aux_model(
                f"Briefly answer using these archived messages.\n\n"
                f"Question: {user_message}\n\nMessages:\n{context}",
                max_tokens=512,
            )
            if answer:
                return {"context": f"Archived context (auto-retrieved):\n{answer}"}
        except Exception as exc:
            logger.debug("RLM pre_llm_call failed: %s", exc)
        return None

    def get_tool_schemas(self):
        return [RLM_SEARCH_SCHEMA]

    def handle_tool_call(self, name, args, **kwargs):
        if name != "rlm_search":
            return json.dumps({"error": f"Unknown RLM tool: {name}"})
        return execute_rlm_search(
            query=args.get("query", ""),
            scope=args.get("scope", "current"),
            session_id=self._session_id,
            hermes_home=self._hermes_home,
        )

    def get_status(self):
        status = super().get_status()
        status["engine"] = "rlm"
        status["compression_count"] = self.compression_count
        return status


# ===========================================================================
# MODE 2: Regular plugin (just the rlm_search tool)
# ===========================================================================

def _handle_rlm_search_tool(args, **kwargs):
    """Handler for rlm_search when registered as a regular plugin tool."""
    session_id = kwargs.get("session_id", "")
    hermes_home = None
    try:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
    except Exception:
        pass
    return execute_rlm_search(
        query=args.get("query", ""),
        scope=args.get("scope", "current"),
        session_id=session_id,
        hermes_home=hermes_home,
    )


# ===========================================================================
# Plugin registration (handles both modes)
# ===========================================================================

def register(ctx):
    """Register RLM as either a context engine or a regular plugin tool.

    If context.engine == "rlm" in config.yaml, registers as a context engine
    (replaces the built-in compressor). Otherwise, registers rlm_search as
    a standalone tool that works alongside whatever compressor is active.

    In both modes, registers auxiliary.rlm task so RLM has its own config
    section independent of auxiliary.compression.
    """
    # Register the RLM auxiliary task — gives us auxiliary.rlm in config.yaml
    # independent of auxiliary.compression. Falls through silently if ctx
    # doesn't support register_auxiliary_task (e.g. _EngineCollector).
    if hasattr(ctx, "register_auxiliary_task"):
        ctx.register_auxiliary_task(
            key="rlm",
            display_name="RLM retrieval",
            description="Recursive Language Model sub-queries for archived context retrieval",
            defaults={
                "provider": "auto",
                "model": "",
                "timeout": 60,
            },
        )
        logger.info("RLM: registered auxiliary.rlm task")

    # Check if we should be the context engine
    is_context_engine = False
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        ctx_cfg = cfg.get("context", {})
        is_context_engine = ctx_cfg.get("engine") == "rlm"
    except Exception:
        pass

    if is_context_engine:
        # MODE 1: Context engine — replace the built-in compressor
        engine = RLMContextEngine()
        if hasattr(ctx, "register_context_engine"):
            ctx.register_context_engine(engine)
        if hasattr(ctx, "register_hook"):
            ctx.register_hook("pre_llm_call", engine._on_pre_llm_call)
            engine._hook_registered = True
        logger.info("RLM: registered as context engine")
    else:
        # MODE 2: Regular plugin — just expose rlm_search as a tool
        if hasattr(ctx, "register_tool"):
            ctx.register_tool(
                name="rlm_search",
                toolset="rlm",
                schema=RLM_SEARCH_SCHEMA,
                handler=_handle_rlm_search_tool,
                description="Search archived conversation context using RLM retrieval",
                emoji="🔍",
            )
        logger.info("RLM: registered rlm_search tool (regular plugin mode)")
