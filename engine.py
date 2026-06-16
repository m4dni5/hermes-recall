"""RLM Context Engine implementation.

Design:
  - compress() returns system prompt + context note + recent tail (no summarization)
  - rlm_search tool does FTS5 search across all archived messages, then
    chunks results and sub-queries a cheap model to synthesize an answer
  - Uses Hermes's call_llm(task="compression") for sub-queries, which
    resolves through auxiliary.compression.model config automatically
  - Creates its own read-only SessionDB connection to state.db for searching
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_PROTECT_LAST_N = 20
_DEFAULT_CHUNK_SIZE = 80_000       # chars per chunk for sub-queries
_DEFAULT_MAX_SUB_TOKENS = 2048
_DEFAULT_MAX_AGG_TOKENS = 4096
_DEFAULT_SEARCH_LIMIT = 50        # max FTS5 results per search

# The context note injected after compression tells the agent about rlm_search.
_CONTEXT_NOTE = (
    "[Earlier conversation history has been archived. "
    "Use the rlm_search tool to retrieve relevant context — "
    "e.g. rlm_search(query=\"what did we decide about X\"). "
    "The search scans all archived messages across session boundaries.]"
)


class RLMContextEngine(ContextEngine):
    """Retrieval-based context engine: archive everything, retrieve on demand."""

    # -- Identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "rlm"

    # -- Init --------------------------------------------------------------

    def __init__(
        self,
        protect_last_n: int = _DEFAULT_PROTECT_LAST_N,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        search_limit: int = _DEFAULT_SEARCH_LIMIT,
    ):
        self.protect_last_n = protect_last_n
        self.chunk_size = chunk_size
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
            "RLM context engine initialized: protect_last_n=%d chunk_size=%d search_limit=%d",
            protect_last_n, chunk_size, search_limit,
        )

    # -- SessionDB (lazy) --------------------------------------------------

    def _get_session_db(self):
        """Lazy-init a read-only SessionDB connection to state.db."""
        if self._session_db is not None:
            return self._session_db

        try:
            from hermes_state import SessionDB, DEFAULT_DB_PATH
            db_path = self._hermes_home / "state.db" if self._hermes_home else DEFAULT_DB_PATH
            if not db_path.exists():
                logger.warning("RLM: state.db not found at %s — rlm_search will be unavailable", db_path)
                return None
            self._session_db = SessionDB(db_path)
            logger.debug("RLM: SessionDB connected at %s", db_path)
            return self._session_db
        except Exception as exc:
            logger.warning("RLM: Failed to open SessionDB: %s", exc)
            return None

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
        """Keep system prompt + recent tail, inject context note, drop middle.

        No LLM calls. No summarization. Middle messages are already persisted
        to state.db by the normal flush loop — we just trim the in-memory list.
        """
        if len(messages) <= self.protect_last_n + 2:
            return messages  # nothing meaningful to trim

        # Find the system prompt (first message, usually role=system)
        head = []
        rest = messages
        if rest and rest[0].get("role") == "system":
            head = [rest[0]]
            rest = rest[1:]

        # Keep the tail
        tail = rest[-self.protect_last_n:] if self.protect_last_n else []

        # Build context note message
        note_msg = {"role": "assistant", "content": _CONTEXT_NOTE}

        result = head + [note_msg] + tail
        self.compression_count += 1

        dropped = len(messages) - len(result)
        logger.info(
            "RLM compress: %d messages → %d (dropped %d middle, kept tail of %d)",
            len(messages), len(result), dropped, len(tail),
        )
        return result

    # -- Session lifecycle -------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home")
        if self._hermes_home:
            self._hermes_home = Path(self._hermes_home)
        # Reset the lazy DB connection on session change so we pick up
        # the right profile's state.db
        self._session_db = None

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._session_db = None

    # -- Model switch support ----------------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    # -- Tools -------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "rlm_search",
                "description": (
                    "Search archived conversation context using RLM retrieval. "
                    "Queries all historical messages (not just the active window) "
                    "using full-text search, then synthesizes findings via a "
                    "sub-query model. Use this when you need to recall details "
                    "from earlier in the conversation that are no longer in your "
                    "active context window."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what to find",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["current", "all"],
                            "description": (
                                "'current' = search only the current session lineage; "
                                "'all' = search across all sessions. Default: 'current'"
                            ),
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
            },
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if name == "rlm_search":
            return self._handle_rlm_search(args, **kwargs)
        return json.dumps({"error": f"Unknown RLM tool: {name}"})

    # -- rlm_search implementation -----------------------------------------

    def _get_session_lineage(self, db) -> List[str]:
        """Walk parent_session_id chain from current session to root."""
        chain = []
        current = self._session_id
        seen = set()
        for _ in range(100):  # safety bound
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
        self,
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

    def _handle_rlm_search(self, args: Dict[str, Any], **kwargs) -> str:
        """FTS5 search → chunk → sub-query → aggregate."""
        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        scope = args.get("scope", "current")
        sort = args.get("sort", "relevance")
        limit = args.get("limit", self.search_limit)

        db = self._get_session_db()
        if db is None:
            return json.dumps({
                "error": "Session database not available. Cannot search archived context.",
            })

        # -- Step 1: FTS5 search ------------------------------------------
        try:
            if scope == "current" and self._session_id:
                # Scope to current session lineage (root → tip)
                lineage = self._get_session_lineage(db)
                results = self._search_scoped(db, query, session_ids=lineage,
                                              limit=limit, sort=sort)
            else:
                results = db.search_messages(
                    query=query, limit=limit,
                    sort=sort if sort in ("newest", "oldest") else None,
                )
        except Exception as exc:
            return json.dumps({"error": f"Search failed: {exc}"})

        if not results:
            return json.dumps({
                "answer": "No relevant messages found in the archive.",
                "results_count": 0,
            })

        # -- Step 2: Build context from results ----------------------------
        context_parts = []
        for r in results:
            role = r.get("role", "unknown")
            content = r.get("content", "")
            snippet = r.get("snippet", "")
            ts = r.get("timestamp", 0)
            sid = r.get("session_id", "?")
            # Use snippet for display, full content for sub-query
            display = snippet or (content[:500] + "..." if len(content) > 500 else content)
            context_parts.append(
                f"[session:{sid} role:{role}] {display}"
            )

        context_text = "\n\n".join(context_parts)

        # -- Step 3: Sub-query via call_llm --------------------------------
        try:
            answer = self._sub_query(query, context_text)
        except Exception as exc:
            # Fallback: return raw search results if sub-query fails
            logger.warning("RLM sub-query failed, returning raw results: %s", exc)
            return json.dumps({
                "answer": f"(Sub-query unavailable: {exc})\n\nRaw search results:\n{context_text[:4000]}",
                "results_count": len(results),
                "sub_query_failed": True,
            })

        return json.dumps({
            "answer": answer,
            "results_count": len(results),
        })

    def _sub_query(self, query: str, context: str) -> str:
        """Send context + query to a cheap model via Hermes's auxiliary routing."""
        from agent.auxiliary_client import call_llm

        prompt = (
            "You are analyzing search results from a conversation archive. "
            "The user asked a question and below are the most relevant messages found. "
            "Synthesize a clear, concise answer from these results. "
            "If the results don't contain enough information to answer, say so.\n\n"
            f"Question: {query}\n\n"
            f"Relevant messages ({len(context):,} chars):\n{context}"
        )

        response = call_llm(
            task="compression",  # uses auxiliary.compression.model config
            main_runtime={
                "model": self.model,
                "provider": self.provider,
                "base_url": self.base_url,
                "api_key": self.api_key,
            },
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_DEFAULT_MAX_AGG_TOKENS,
        )
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) else str(content or "")

    # -- Status ------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["engine"] = "rlm"
        status["protect_last_n"] = self.protect_last_n
        status["compression_count"] = self.compression_count
        status["session_id"] = self._session_id
        return status
