"""RLM Context Engine — retrieval-based context management.

Three layers:
  1. compress() trims to tail (instant, no LLM)
  2. pre_llm_call hook auto-retrieves relevant context every turn
     (FTS5 → single sub-query via call_llm → inject — lightweight, sync)
  3. rlm_search tool runs the full RLM pipeline inline
     (FTS5 → chunk → parallel sub-queries via cheap model → synthesize)

No delegation needed. The engine IS the sandbox — handle_tool_call is
Python code that can call call_llm multiple times, chunk results, and
run sub-queries in parallel via ThreadPoolExecutor.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_PROTECT_LAST_N = 20
_DEFAULT_SEARCH_LIMIT = 50
_DEFAULT_CHUNK_SIZE = 5          # messages per chunk for sub-queries
_DEFAULT_MAX_WORKERS = 4         # parallel sub-queries
_DEFAULT_MAX_CHUNK_TOKENS = 1024 # max tokens per chunk sub-query
_DEFAULT_MAX_AGG_TOKENS = 4096   # max tokens for final aggregation

# Post-compression context note.
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
   - Results are chunked and processed in parallel using a cheap model.

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


def _format_message(msg: Dict[str, Any]) -> str:
    """Format a single message for inclusion in a sub-query prompt."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    snippet = msg.get("snippet", "")
    sid = msg.get("session_id", "?")
    display = snippet or (content[:500] + "..." if len(content) > 500 else content)
    return f"[session:{sid} role:{role}] {display}"


def _call_aux_model(prompt: str, max_tokens: int = 1024) -> str:
    """Call the auxiliary compression model directly."""
    from agent.auxiliary_client import call_llm

    response = call_llm(
        task="compression",
        main_runtime={},  # let auxiliary_client resolve from config
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content or "")


def _query_chunk(query: str, chunk_msgs: List[Dict[str, Any]], chunk_idx: int) -> str:
    """Sub-query: ask the cheap model about one chunk of messages."""
    formatted = "\n\n".join(_format_message(m) for m in chunk_msgs)

    prompt = (
        "You are analyzing a chunk of archived conversation messages. "
        "Extract ALL information relevant to the question below. "
        "If nothing relevant is found, say 'NO_RELEVANT_INFO'.\n\n"
        f"Question: {query}\n\n"
        f"Chunk ({chunk_idx}):\n{formatted}"
    )

    return _call_aux_model(prompt, max_tokens=_DEFAULT_MAX_CHUNK_TOKENS)


def _synthesize_findings(query: str, findings: List[str]) -> str:
    """Aggregate: ask the cheap model to combine chunk findings."""
    findings_text = "\n\n---\n\n".join(
        f"Chunk {i+1}:\n{f}" for i, f in enumerate(findings)
    )

    prompt = (
        "You are synthesizing findings from multiple chunks of a conversation "
        "archive. Below are the results of analyzing each chunk for a question. "
        "Combine them into a single, coherent answer. "
        "If multiple chunks found the same information, consolidate. "
        "If contradictions exist, note them.\n\n"
        f"Question: {query}\n\n"
        f"Findings from {len(findings)} chunks:\n{findings_text}\n\n"
        "Provide a clear, complete answer:"
    )

    return _call_aux_model(prompt, max_tokens=_DEFAULT_MAX_AGG_TOKENS)


def _get_aux_model_info() -> Dict[str, str]:
    """Read auxiliary.compression model config."""
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


# ===========================================================================
# Retrieval pipelines
# ===========================================================================

def retrieve_lightweight(
    query: str,
    db,
    session_id: str,
    scope: str = "current",
    limit: int = 50,
    sort: str = "relevance",
) -> Optional[str]:
    """Lightweight retrieval: FTS5 → single sub-query → answer.

    Used by the pre_llm_call hook. One LLM call, ~1-3 seconds.
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

    formatted = "\n\n".join(_format_message(m) for m in results)

    try:
        return _call_aux_model(
            f"Answer this question using the archived messages below.\n\n"
            f"Question: {query}\n\nMessages:\n{formatted}",
            max_tokens=_DEFAULT_MAX_AGG_TOKENS,
        )
    except Exception as exc:
        logger.warning("RLM sub-query failed: %s", exc)
        return formatted[:4000]


def retrieve_full(
    query: str,
    db,
    session_id: str,
    scope: str = "current",
    limit: int = 50,
    sort: str = "relevance",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> Optional[str]:
    """Full RLM pipeline: FTS5 → chunk → parallel sub-queries → synthesize.

    Used by the rlm_search tool. Multiple LLM calls in parallel, ~3-5 seconds.
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

    # Small result set — single sub-query is fine
    if len(results) <= chunk_size:
        formatted = "\n\n".join(_format_message(m) for m in results)
        try:
            return _call_aux_model(
                f"Answer this question using the archived messages below.\n\n"
                f"Question: {query}\n\nMessages:\n{formatted}",
                max_tokens=_DEFAULT_MAX_AGG_TOKENS,
            )
        except Exception as exc:
            logger.warning("RLM sub-query failed: %s", exc)
            return formatted[:4000]

    # Large result set — chunk and process in parallel
    chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]
    logger.info("RLM: %d results → %d chunks, querying in parallel", len(results), len(chunks))

    findings = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_query_chunk, query, chunk, i): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                findings[idx] = future.result()
            except Exception as exc:
                logger.warning("RLM chunk %d failed: %s", idx, exc)
                findings[idx] = f"(chunk {idx} failed: {exc})"

    # Filter out chunks with no relevant info
    relevant = [f for f in findings if f and "NO_RELEVANT_INFO" not in f.upper().replace("_", "")]
    if not relevant:
        return "No relevant information found in the archived messages."

    # Single chunk — no need to aggregate
    if len(relevant) == 1:
        return relevant[0]

    # Multiple chunks — synthesize
    try:
        return _synthesize_findings(query, relevant)
    except Exception as exc:
        logger.warning("RLM synthesis failed: %s", exc)
        return "\n\n---\n\n".join(relevant)


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
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        max_iterations: int = 8,
        max_llm_tokens: int = 1024,
    ):
        self.protect_last_n = protect_last_n
        self.search_limit = search_limit
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.max_iterations = max_iterations
        self.max_llm_tokens = max_llm_tokens

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
            "RLM context engine: protect_last_n=%d search_limit=%d chunk_size=%d max_workers=%d",
            protect_last_n, search_limit, chunk_size, max_workers,
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
        """Keep system prompt + recent tail, drop middle."""
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

        Lightweight: FTS5 search + single sub-query. ~1-3 seconds.
        """
        db = self._get_session_db()
        if db is None:
            return None

        answer = retrieve_lightweight(
            query=user_message,
            db=db,
            session_id=self._session_id or session_id,
            scope="current",
            limit=self.search_limit,
        )
        if not answer:
            return None

        return {"context": f"Archived context (auto-retrieved):\n{answer}"}

    # -- Tools (inline RLM pipeline) ---------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [{
            "name": "rlm_search",
            "description": (
                "Deep search of archived conversation context. Runs the full "
                "RLM pipeline: searches the archive, chunks results, and "
                "processes each chunk in parallel using a cheap model, then "
                "synthesizes findings into a coherent answer. Use for complex "
                "queries or when you need to search across all sessions."
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

        # Step 1: FTS5 search to gather initial context
        try:
            if scope == "current" and self._session_id:
                lineage = _get_session_lineage(db, self._session_id)
                results = _search_scoped(db, query, session_ids=lineage,
                                         limit=limit, sort=sort)
            else:
                results = db.search_messages(
                    query=query, limit=limit,
                    sort=sort if sort in ("newest", "oldest") else None,
                )
        except Exception as exc:
            return json.dumps({"error": f"Search failed: {exc}"})

        if not results:
            return json.dumps({"answer": "No relevant messages found.", "results_count": 0})

        # Format as context string
        context = "\n\n".join(_format_message(m) for m in results)

        # Step 2: Run the RLM REPL — model writes code to process context
        from repl import run_rlm_repl

        try:
            answer = run_rlm_repl(
                context=context,
                query=query,
                max_iterations=self.max_iterations,
                max_llm_tokens=self.max_llm_tokens,
            )
        except Exception as exc:
            logger.warning("RLM REPL failed, falling back to direct synthesis: %s", exc)
            # Fallback: single sub-query
            try:
                answer = _call_aux_model(
                    f"Answer this question using the archived messages.\n\n"
                    f"Question: {query}\n\nMessages:\n{context}",
                    max_tokens=_DEFAULT_MAX_AGG_TOKENS,
                )
            except Exception as exc2:
                return json.dumps({"error": f"Both REPL and fallback failed: {exc2}"})

        return json.dumps({
            "answer": answer,
            "results_count": len(results),
            "method": "repl",
        })

    # -- Status ------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["engine"] = "rlm"
        status["protect_last_n"] = self.protect_last_n
        status["chunk_size"] = self.chunk_size
        status["max_workers"] = self.max_workers
        status["compression_count"] = self.compression_count
        status["session_id"] = self._session_id
        return status


# ===========================================================================
# Plugin registration
# ===========================================================================

def register(ctx):
    """Register the RLM context engine."""
    engine = RLMContextEngine()

    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)

    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_llm_call", engine.on_pre_llm_call)
        RLMContextEngine._hook_registered = True
        logger.info("RLM: registered pre_llm_call hook via register()")
