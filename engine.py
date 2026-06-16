"""RLM Context Engine — retrieval-based context management.

Three layers:
  1. compress() trims to tail (instant, no LLM)
  2. pre_llm_call hook auto-retrieves relevant context every turn (FTS5 → sub-query → inject)
  3. rlm_search tool for explicit deep dives with custom queries

Uses Hermes's call_llm(task="compression") for sub-queries, which resolves
through auxiliary.compression.model config automatically.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_PROTECT_LAST_N = 20
_DEFAULT_SEARCH_LIMIT = 50
_DEFAULT_MAX_AGG_TOKENS = 4096

# Context note injected after compression. Tells the agent about rlm_search
# but the pre_llm_call hook handles most retrieval automatically.
_CONTEXT_NOTE = (
    "[Earlier conversation history has been archived. "
    "Context is automatically retrieved each turn. "
    "You can also use rlm_search for explicit queries — "
    "e.g. rlm_search(query=\"what did we decide about X\").]"
)


# ===========================================================================
# Shared retrieval pipeline
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
    return chain  # [current, parent, grandparent, ...]


def _search_scoped(
    db,
    query: str,
    session_ids: List[str],
    limit: int = 50,
    sort: str = "relevance",
) -> List[Dict[str, Any]]:
    """FTS5 search scoped to specific session IDs."""
    if not session_ids:
        return []

    sanitized = db._sanitize_fts5_query(query)
    if not sanitized:
        return []

    placeholders = ",".join("?" for _ in session_ids)

    if sort == "newest":
        order = "m.timestamp DESC, rank"
    elif sort == "oldest":
        order = "m.timestamp ASC, rank"
    else:
        order = "rank"

    sql = f"""
        SELECT
            m.id, m.session_id, m.role,
            snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
            m.content, m.timestamp, m.tool_name
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        WHERE messages_fts MATCH ?
          AND m.session_id IN ({placeholders})
          AND m.active = 1
        ORDER BY {order}
        LIMIT ?
    """
    params = [sanitized] + list(session_ids) + [limit]

    with db._lock:
        cursor = db._conn.execute(sql, params)
        rows = cursor.fetchall()

    results = []
    for row in rows:
        msg = dict(row)
        if "content" in msg:
            msg["content"] = db._decode_content(msg["content"])
        results.append(msg)
    return results


def _build_context_text(results: List[Dict[str, Any]]) -> str:
    """Format search results into context for the sub-query model."""
    parts = []
    for r in results:
        role = r.get("role", "unknown")
        content = r.get("content", "")
        snippet = r.get("snippet", "")
        sid = r.get("session_id", "?")
        display = snippet or (content[:500] + "..." if len(content) > 500 else content)
        parts.append(f"[session:{sid} role:{role}] {display}")
    return "\n\n".join(parts)


def _synthesize(query: str, context: str, main_runtime: Dict[str, str]) -> Optional[str]:
    """Sub-query: send context + question to cheap model for synthesis."""
    from agent.auxiliary_client import call_llm

    prompt = (
        "You are analyzing search results from a conversation archive. "
        "Below are the most relevant messages found for the user's question. "
        "Synthesize a clear, concise answer. "
        "If the results don't contain enough information, say so briefly.\n\n"
        f"Question: {query}\n\n"
        f"Relevant messages ({len(context):,} chars):\n{context}"
    )

    response = call_llm(
        task="compression",
        main_runtime=main_runtime,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_DEFAULT_MAX_AGG_TOKENS,
    )
    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content or "")


def retrieve(
    query: str,
    db,
    session_id: str,
    main_runtime: Dict[str, str],
    scope: str = "current",
    limit: int = 50,
    sort: str = "relevance",
) -> Optional[str]:
    """Full retrieval pipeline: FTS5 search → chunk → sub-query → answer.

    Returns synthesized answer string, or None if nothing found.
    """
    try:
        if scope == "current" and session_id:
            lineage = _get_session_lineage(db, session_id)
            results = _search_scoped(db, query, session_ids=lineage,
                                     limit=limit, sort=sort)
        else:
            results = db.search_messages(
                query=query, limit=limit,
                sort=sort if sort in ("newest", "oldest") else None,
            )
    except Exception as exc:
        logger.warning("RLM search failed: %s", exc)
        return None

    if not results:
        return None

    context_text = _build_context_text(results)

    try:
        return _synthesize(query, context_text, main_runtime)
    except Exception as exc:
        logger.warning("RLM sub-query failed: %s", exc)
        # Fallback: return truncated raw results
        return context_text[:4000]


# ===========================================================================
# Context Engine
# ===========================================================================

class RLMContextEngine(ContextEngine):
    """Retrieval-based context engine: archive everything, retrieve on demand."""

    @property
    def name(self) -> str:
        return "rlm"

    def __init__(
        self,
        protect_last_n: int = _DEFAULT_PROTECT_LAST_N,
        search_limit: int = _DEFAULT_SEARCH_LIMIT,
    ):
        self.protect_last_n = protect_last_n
        self.search_limit = search_limit

        # Token tracking (required by ABC)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0

        # Session binding
        self._session_id: str = ""
        self._hermes_home: Optional[Path] = None

        # Model info (populated by update_model)
        self.model: str = ""
        self.base_url: str = ""
        self.api_key: str = ""
        self.provider: str = ""
        self.api_mode: str = ""

        # Lazy SessionDB connection
        self._session_db = None

        logger.info(
            "RLM context engine initialized: protect_last_n=%d search_limit=%d",
            protect_last_n, search_limit,
        )

    # -- SessionDB (lazy) --------------------------------------------------

    def _get_session_db(self):
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state import SessionDB, DEFAULT_DB_PATH
            db_path = self._hermes_home / "state.db" if self._hermes_home else DEFAULT_DB_PATH
            if not db_path.exists():
                logger.warning("RLM: state.db not found at %s", db_path)
                return None
            self._session_db = SessionDB(db_path)
            return self._session_db
        except Exception as exc:
            logger.warning("RLM: Failed to open SessionDB: %s", exc)
            return None

    def _get_main_runtime(self) -> Dict[str, str]:
        return {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }

    # -- ABC required methods ----------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if not self.threshold_tokens:
            return False
        tokens = prompt_tokens or self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Keep system prompt + recent tail, drop middle.

        No LLM calls. Middle messages are already in state.db.
        """
        if len(messages) <= self.protect_last_n + 2:
            return messages

        head = []
        rest = messages
        if rest and rest[0].get("role") == "system":
            head = [rest[0]]
            rest = rest[1:]

        tail = rest[-self.protect_last_n:] if self.protect_last_n else []
        note_msg = {"role": "assistant", "content": _CONTEXT_NOTE}
        result = head + [note_msg] + tail
        self.compression_count += 1

        logger.info(
            "RLM compress: %d → %d (dropped %d, tail=%d)",
            len(messages), len(result), len(messages) - len(result), len(tail),
        )
        return result

    # -- Session lifecycle -------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home")
        if self._hermes_home:
            self._hermes_home = Path(self._hermes_home)
        self._session_db = None
        self._ensure_hook_registered()

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._session_db = None

    def update_model(self, model, context_length, base_url="", api_key="",
                     provider="", api_mode="", **kw) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    # -- Hook self-registration --------------------------------------------

    _hook_registered = False

    def _ensure_hook_registered(self):
        """Register pre_llm_call hook with the plugin system.

        Called from on_session_start. Idempotent — only registers once.
        Context engine plugins loaded from the directory convention don't
        go through the general plugin register() path, so we self-register
        by reaching into the PluginManager directly.
        """
        if RLMContextEngine._hook_registered:
            return
        try:
            from hermes_cli.plugins import get_plugin_manager
            manager = get_plugin_manager()
            manager._hooks.setdefault("pre_llm_call", []).append(self.on_pre_llm_call)
            RLMContextEngine._hook_registered = True
            logger.info("RLM: registered pre_llm_call hook for auto-retrieval")
        except Exception as exc:
            logger.debug("RLM: could not register pre_llm_call hook: %s", exc)

    # -- pre_llm_call hook -------------------------------------------------

    def on_pre_llm_call(self, session_id: str, user_message: str,
                        conversation_history: list, **kwargs) -> Optional[dict]:
        """Auto-retrieve archived context relevant to this turn's message."""
        db = self._get_session_db()
        if db is None:
            return None

        answer = retrieve(
            query=user_message,
            db=db,
            session_id=self._session_id or session_id,
            main_runtime=self._get_main_runtime(),
            scope="current",
            limit=self.search_limit,
        )
        if not answer:
            return None

        return {"context": f"Archived context (auto-retrieved):\n{answer}"}

    # -- Tools (explicit deep dive) ----------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [{
            "name": "rlm_search",
            "description": (
                "Search archived conversation context with a custom query. "
                "Automatically retrieves relevant context each turn — use this "
                "for targeted searches with specific terms, different scope, "
                "or time ordering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["current", "all"],
                        "description": "'current' = session lineage, 'all' = every session. Default: 'current'",
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "newest", "oldest"],
                        "description": "Sort order. Default: 'relevance'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to retrieve (default: 50)",
                    },
                },
                "required": ["query"],
            },
        }]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if name != "rlm_search":
            return json.dumps({"error": f"Unknown RLM tool: {name}"})

        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        db = self._get_session_db()
        if db is None:
            return json.dumps({"error": "Session database not available"})

        answer = retrieve(
            query=query,
            db=db,
            session_id=self._session_id,
            main_runtime=self._get_main_runtime(),
            scope=args.get("scope", "current"),
            limit=args.get("limit", self.search_limit),
            sort=args.get("sort", "relevance"),
        )

        if answer is None:
            return json.dumps({"answer": "No relevant messages found.", "results_count": 0})

        return json.dumps({"answer": answer})

    # -- Status ------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["engine"] = "rlm"
        status["protect_last_n"] = self.protect_last_n
        status["compression_count"] = self.compression_count
        status["session_id"] = self._session_id
        return status


# ===========================================================================
# Plugin registration
# ===========================================================================

def register(ctx):
    """Register the RLM context engine.

    Called by the Hermes plugin system. Handles both:
    - _EngineCollector (directory-based loading): registers engine only
    - PluginContext (general plugin system): registers engine + hook
    """
    engine = RLMContextEngine()

    # Register the context engine (works with both _EngineCollector and PluginContext)
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)

    # Register the pre_llm_call hook (only works with real PluginContext)
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_llm_call", engine.on_pre_llm_call)
        RLMContextEngine._hook_registered = True
        logger.info("RLM: registered pre_llm_call hook via register()")
