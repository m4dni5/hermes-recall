"""RLM Context Engine — retrieval-based context management.

Three layers:
  1. compress() trims to tail (instant, no LLM)
  2. pre_llm_call hook auto-retrieves relevant context every turn
     (FTS5 → single sub-query via call_llm → inject — lightweight, sync)
  3. rlm_search tool delegates to a child agent that runs the full RLM
     pipeline (FTS5 → chunk → sub-query per chunk via cheap model →
     synthesize — heavyweight, async)

The child agent writes Python to process archived messages, using the
auxiliary compression model (from auxiliary.compression.model in config.yaml)
for sub-queries. This mirrors the original RLM architecture: a smart root
model orchestrates, cheap sub-models do the heavy lifting.
"""

import json
import logging
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

# Post-compression context note. Must be clear enough that the agent
# understands the retrieval architecture and uses rlm_search proactively.
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
   - Results are processed by a dedicated sub-agent that chunks and
     synthesizes findings using a cheap model.

IMPORTANT: If you need context that isn't in the auto-retrieved section
below, call rlm_search. Don't guess or hallucinate — search the archive."""


# ===========================================================================
# Helpers
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


def _get_aux_model_info() -> Dict[str, str]:
    """Read auxiliary.compression model config from config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        aux = cfg.get("auxiliary", {})
        comp = aux.get("compression", {}) if isinstance(aux, dict) else {}
        return {
            "model": str(comp.get("model") or ""),
            "provider": str(comp.get("provider") or ""),
        }
    except Exception:
        return {"model": "", "provider": ""}


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
    """Lightweight retrieval: FTS5 search → single sub-query → answer.

    Used by the pre_llm_call hook. Returns synthesized answer or None.
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

    def _get_db_path(self) -> Optional[str]:
        """Return absolute path to state.db."""
        if self._hermes_home:
            p = self._hermes_home / "state.db"
            if p.exists():
                return str(p)
        try:
            from hermes_state import DEFAULT_DB_PATH
            if DEFAULT_DB_PATH.exists():
                return str(DEFAULT_DB_PATH)
        except Exception:
            pass
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
        """Register pre_llm_call hook with the plugin system."""
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

    # -- pre_llm_call hook (lightweight, sync) -----------------------------

    def on_pre_llm_call(self, session_id: str, user_message: str,
                        conversation_history: list, **kwargs) -> Optional[dict]:
        """Auto-retrieve archived context relevant to this turn's message.

        Lightweight: FTS5 search + single sub-query via call_llm.
        Runs synchronously (~1-3 seconds with a cheap model).
        """
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

    # -- Tools (delegated, async) ------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [{
            "name": "rlm_search",
            "description": (
                "Deep search of archived conversation context. Spawns a "
                "sub-agent that queries the session archive, chunks results, "
                "and synthesizes an answer using a cheap model. Use for "
                "complex queries, multi-chunk processing, or when you need "
                "to search across all sessions. Runs asynchronously — you "
                "can continue working while it processes."
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
                            "'current' = this conversation's lineage (default). "
                            "'all' = every session in the archive."
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

        scope = args.get("scope", "current")
        limit = args.get("limit", self.search_limit)
        sort = args.get("sort", "relevance")

        # Try delegated (async) path first — needs parent_agent from kwargs
        parent_agent = kwargs.get("parent_agent")
        if parent_agent is not None:
            return self._delegate_search(
                query=query, db=db, scope=scope, limit=limit, sort=sort,
                parent_agent=parent_agent,
            )

        # Fallback: lightweight synchronous retrieval (same as pre_llm_call hook)
        answer = retrieve(
            query=query,
            db=db,
            session_id=self._session_id,
            main_runtime=self._get_main_runtime(),
            scope=scope,
            limit=limit,
            sort=sort,
        )

        if answer is None:
            return json.dumps({"answer": "No relevant messages found.", "results_count": 0})

        return json.dumps({"answer": answer})

    def _build_delegation_goal(
        self,
        query: str,
        db_path: str,
        session_ids: List[str],
        limit: int,
        sort: str,
        aux_model: str,
        aux_provider: str,
    ) -> str:
        """Build a self-contained goal for the delegated rlm_search child."""

        session_clause = ""
        if session_ids:
            id_list = ", ".join(f"'{s}'" for s in session_ids)
            session_clause = (
                f"Only search these session IDs (in order from current to oldest): "
                f"[{id_list}]. Add a WHERE clause: m.session_id IN ({id_list})"
            )
        else:
            session_clause = "Search across ALL sessions (no session filter)."

        model_instruction = ""
        if aux_model:
            model_instruction = (
                f"For sub-queries on each chunk, call the OpenAI API with "
                f"model=\"{aux_model}\""
                + (f" via provider \"{aux_provider}\"" if aux_provider else "")
                + ". Use the OPENAI_API_KEY environment variable for auth."
            )
        else:
            model_instruction = (
                "No auxiliary model configured. For sub-queries, use "
                "OPENAI_API_KEY with a cheap model like gpt-4.1-nano."
            )

        return f"""You are an RLM retrieval agent. Your job is to search the
Hermes conversation archive and answer a question by processing messages
in chunks using a cheap LLM.

DATABASE: {db_path}
TABLES: messages (id, session_id, role, content, timestamp, active),
        messages_fts (FTS5 virtual table on content)

QUERY: {query}
SCOPE: {session_clause}
LIMIT: {limit} messages
SORT: {sort}

SUB-QUERY MODEL: {model_instruction}

INSTRUCTIONS:
1. Use `terminal` to run python3 and query state.db
2. Search messages_fts using FTS5 MATCH syntax for keywords from the query
3. If results are large (>5 messages), chunk them into groups of 3-5
4. For each chunk, call the cheap model with the chunk content + the query
   to extract relevant information
5. Collect all chunk results
6. If there are multiple chunk results, do one final synthesis call to
   combine them into a coherent answer
7. Your FINAL response should be the synthesized answer — this is returned
   to the parent agent and injected into the conversation

PYTHON EXAMPLE for step 2-3:
```python
import sqlite3
conn = sqlite3.connect("{db_path}")
conn.row_factory = sqlite3.Row
rows = conn.execute(\"\"\"
    SELECT m.id, m.session_id, m.role, m.content, m.timestamp
    FROM messages_fts fts
    JOIN messages m ON m.id = fts.rowid
    WHERE messages_fts MATCH ?
      AND m.active = 1
    ORDER BY rank
    LIMIT ?
\"\"\", [search_terms, {limit}]).fetchall()
# Then chunk rows and process each chunk
```

PYTHON EXAMPLE for step 4 (calling the cheap model):
```python
import os, json
from openai import OpenAI
client = OpenAI()  # uses OPENAI_API_KEY
response = client.chat.completions.create(
    model="{aux_model or 'gpt-4.1-nano'}",
    messages=[{{"role": "user", "content": f"Given these messages: {{chunk}}\\n\\nAnswer: {query}"}}],
    max_tokens=1024,
)
answer = response.choices[0].message.content
```

Be thorough. Chunk results if they're large. Don't send 50 messages to a
single sub-query call — that's what the original RLM paper avoids.
Process iteratively, like the REPL pattern."""

    def _delegate_search(
        self,
        query: str,
        db,
        scope: str,
        limit: int,
        sort: str,
        parent_agent,
    ) -> str:
        """Delegate rlm_search to a child agent via delegate_task.

        The child runs the full RLM pipeline: FTS5 search → chunk →
        sub-query cheap model per chunk → synthesize. Uses background=true
        so the parent agent can continue working.
        """
        from tools.delegate_tool import delegate_task

        # Resolve session lineage
        lineage = []
        if scope == "current" and self._session_id:
            try:
                lineage = _get_session_lineage(db, self._session_id)
            except Exception:
                lineage = [self._session_id]

        db_path = self._get_db_path()
        if not db_path:
            return json.dumps({"error": "Session database not available"})

        # Read auxiliary compression model from config
        aux = _get_aux_model_info()
        aux_model = aux.get("model", "")
        aux_provider = aux.get("provider", "")

        # Build the child's goal — fully self-contained
        goal = self._build_delegation_goal(
            query=query,
            db_path=db_path,
            session_ids=lineage,
            limit=limit,
            sort=sort,
            aux_model=aux_model,
            aux_provider=aux_provider,
        )

        # Delegate — background=true so parent isn't blocked
        result = delegate_task(
            goal=goal,
            toolsets=["terminal", "file"],
            role="leaf",
            background=True,
            parent_agent=parent_agent,
        )

        return result

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

    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)

    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_llm_call", engine.on_pre_llm_call)
        RLMContextEngine._hook_registered = True
        logger.info("RLM: registered pre_llm_call hook via register()")
