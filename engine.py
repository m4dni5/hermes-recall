"""RLM Hermes Plugin v2 — JSON archive + FTS5 index.

MODE 1: Context Engine (context.engine: "rlm" in config.yaml)
  compress() serializes evicted messages to JSON (lossless).
  rlm_search loads JSON into REPL with search_context() FTS5 index.

MODE 2: Regular Plugin (just install as a plugin)
  Exposes rlm_search as a tool. Loads session lineage from state.db.
"""

import json
import logging
import os
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

_CONTEXT_NOTE = """[CONTEXT ARCHIVED — rlm_search AVAILABLE]

Earlier conversation turns have been archived as structured JSON.
To access archived context, use: rlm_search(query="your question")

The REPL environment has a `messages` variable containing all archived
messages as a JSON array. Each message has: i (index), sid (session_id),
role, content. Use search_context() for FTS5 index lookup, then read
full messages from the array. Use llm_query() for semantic analysis.

IMPORTANT: If you need context from earlier in the conversation,
call rlm_search. Don't guess or hallucinate — search the archive."""

# Archive storage
_ARCHIVE_DIR = "rlm_archives"


def _get_archive_dir(hermes_home: Optional[Path] = None) -> Path:
    """Get or create the archive directory."""
    base = hermes_home or Path.home() / ".hermes"
    archive_dir = base / _ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _messages_to_json(messages: List[Dict[str, Any]]) -> List[dict]:
    """Convert raw messages to compact JSON-serializable format."""
    result = []
    for i, msg in enumerate(messages):
        entry = {
            "i": i,
            "sid": msg.get("session_id", ""),
            "role": msg.get("role", "unknown"),
            "content": msg.get("content", "") or "",
        }
        if "id" in msg:
            entry["mid"] = msg["id"]  # message DB id for FTS5 matching
        result.append(entry)
    return result


def _load_messages_from_lineage(db, session_ids: List[str]) -> List[Dict[str, Any]]:
    """Load all messages from session lineage."""
    if not session_ids:
        return []
    placeholders = ",".join("?" for _ in session_ids)
    with db._lock:
        rows = db._conn.execute(
            f"SELECT id, session_id, role, content FROM messages "
            f"WHERE session_id IN ({placeholders}) AND active = 1 ORDER BY id",
            session_ids,
        ).fetchall()
    messages = []
    for row in rows:
        mid = row["id"] if hasattr(row, "keys") else row[0]
        sid = row["session_id"] if hasattr(row, "keys") else row[1]
        role = row["role"] if hasattr(row, "keys") else row[2]
        content = row["content"] if hasattr(row, "keys") else row[3]
        if content:
            content = db._decode_content(content)
            messages.append({"id": mid, "session_id": sid, "role": role, "content": content})
    return messages


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


# ---------------------------------------------------------------------------
# Tool implementation (shared by both modes)
# ---------------------------------------------------------------------------

def _fts_hits_to_indices(fts_hits: list, messages_json: List[dict]) -> List[int]:
    """Convert FTS5 search results to indices into the messages_json array.

    Matches by message DB id (mid field).
    """
    if not fts_hits or not messages_json:
        return []
    # Build lookup: message_db_id -> json_index
    mid_lookup = {}
    for i, msg in enumerate(messages_json):
        if "mid" in msg:
            mid_lookup[msg["mid"]] = i

    indices = []
    for hit in fts_hits:
        hit_id = hit.get("id")
        if hit_id is not None and hit_id in mid_lookup:
            indices.append(mid_lookup[hit_id])
    return sorted(set(indices))


RLM_SEARCH_SCHEMA = {
    "name": "rlm_search",
    "description": (
        "Deep search of archived conversation context. Loads messages as a "
        "JSON array into a REPL environment with search_context() (FTS5 index) "
        "and llm_query() (sub-LLM). The model writes Python to query the data. "
        "Use when you need context from earlier in the conversation."
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
    cached_fts_hits: Optional[list] = None,
) -> str:
    """Execute rlm_search — shared by both context engine and regular plugin."""
    if not query.strip():
        return json.dumps({"error": "query is required"})

    db = _get_session_db(hermes_home)
    if db is None:
        return json.dumps({"error": "Session database not available"})

    # Load messages
    try:
        if scope == "current" and session_id:
            lineage = _get_session_lineage(db, session_id)
        else:
            lineage = []

        if lineage:
            raw_messages = _load_messages_from_lineage(db, lineage)
        else:
            # scope=all — load everything
            with db._lock:
                rows = db._conn.execute(
                    "SELECT id, session_id, role, content FROM messages "
                    "WHERE active = 1 ORDER BY id"
                ).fetchall()
            raw_messages = []
            for row in rows:
                mid = row["id"] if hasattr(row, "keys") else row[0]
                sid = row["session_id"] if hasattr(row, "keys") else row[1]
                role = row["role"] if hasattr(row, "keys") else row[2]
                content = row["content"] if hasattr(row, "keys") else row[3]
                if content:
                    content = db._decode_content(content)
                    raw_messages.append({"id": mid, "session_id": sid, "role": role, "content": content})

        messages_json = _messages_to_json(raw_messages)
    except Exception as exc:
        return json.dumps({"error": f"Failed to load messages: {exc}"})

    if not messages_json:
        return json.dumps({"answer": "No messages found.", "results_count": 0})

    # Run the RLM REPL
    sys.path.insert(0, str(Path(__file__).parent))

    # Convert cached FTS5 hits to JSON indices
    fts_hints = None
    if cached_fts_hits:
        fts_hints = _fts_hits_to_indices(cached_fts_hits, messages_json)

    from repl import run_rlm_repl
    try:
        answer = run_rlm_repl(
            messages_json=messages_json,
            query=query,
            hermes_home=str(hermes_home) if hermes_home else None,
            session_ids=lineage if lineage else None,
            fts_hints=fts_hints,
        )
    except Exception as exc:
        logger.warning("RLM REPL failed, falling back to direct aux model: %s", exc)
        try:
            answer = _call_aux_model(
                f"Answer using these archived messages.\n\n"
                f"Question: {query}\n\nMessages:\n{json.dumps(messages_json[:100], indent=2)[:50000]}",
                max_tokens=_DEFAULT_MAX_AGG_TOKENS,
            )
        except Exception as exc2:
            return json.dumps({"error": f"REPL and fallback both failed: {exc2}"})

    return json.dumps({
        "answer": answer,
        "results_count": len(messages_json),
        "method": "repl",
    })


# ===========================================================================
# MODE 1: Context Engine plugin
# ===========================================================================

from agent.context_engine import ContextEngine


class RLMContextEngine(ContextEngine):
    """RLM v2 as a context engine — lossless JSON archive."""

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
        # Cached FTS5 hits from pre_llm_call — passed to rlm_search REPL
        self._cached_hits: Optional[List[int]] = None
        self._cached_query: str = ""

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        if not self.threshold_tokens:
            return False
        return (prompt_tokens or self.last_prompt_tokens) >= self.threshold_tokens

    def compress(self, messages, current_tokens=None, focus_topic=None):
        """Compress by serializing evicted messages to JSON (lossless).

        Keeps system prompt + last protect_last_n messages.
        Evicted messages are written to a JSON archive file.
        """
        if len(messages) <= self.protect_last_n + 2:
            return messages

        head = [messages[0]] if messages and messages[0].get("role") == "system" else []
        rest = messages[len(head):]
        tail = rest[-self.protect_last_n:] if self.protect_last_n else []
        evicted = rest[:-self.protect_last_n] if self.protect_last_n else rest

        # Serialize evicted messages to JSON archive
        if evicted:
            try:
                archive_dir = _get_archive_dir(self._hermes_home)
                archive_path = archive_dir / f"{self._session_id}.json"

                # Load existing archive if present
                existing = []
                if archive_path.exists():
                    with open(archive_path) as f:
                        existing = json.load(f)

                # Append evicted messages (deduplicate by index)
                existing_indices = {m.get("i") for m in existing}
                new_messages = _messages_to_json(evicted)
                start_idx = max(existing_indices) + 1 if existing_indices else 0
                for j, msg in enumerate(new_messages):
                    msg["i"] = start_idx + j

                combined = existing + new_messages
                with open(archive_path, "w") as f:
                    json.dump(combined, f)

                logger.info("RLM compress: archived %d messages to %s (%d total)",
                           len(new_messages), archive_path, len(combined))
            except Exception as exc:
                logger.warning("RLM compress: archive failed: %s", exc)

        self.compression_count += 1
        result = head + [{"role": "assistant", "content": _CONTEXT_NOTE}] + tail
        logger.info("RLM compress: %d → %d", len(messages), len(result))
        return result

    def on_session_start(self, session_id, **kwargs):
        self._session_id = session_id
        hm = kwargs.get("hermes_home")
        self._hermes_home = Path(hm) if hm else None
        self._ensure_hook_registered()

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
        """Lightweight FTS5 scan on every turn. Metadata only — no LLM call.

        Injects a one-line note into the context telling the agent how many
        archived messages match the current prompt. Caches the hit indices
        so rlm_search's REPL can skip the search step.
        """
        db = _get_session_db(self._hermes_home)
        if db is None:
            return None
        try:
            results = db.search_messages(user_message, limit=10)
            if not results:
                return None

            # Cache the FTS5 hits for the REPL
            # We can't compute JSON indices here (no messages_json loaded yet),
            # but we cache the session+role keys so the REPL can match them.
            self._cached_hits = results
            self._cached_query = user_message

            count = len(results)
            preview = results[0].get("snippet", results[0].get("content", ""))[:100]
            return {
                "context": (
                    f"[RLM] {count} archived messages match this topic. "
                    f"Preview: \"{preview}\". "
                    f"Call rlm_search(query=\"...\") to access full context."
                )
            }
        except Exception as exc:
            logger.debug("RLM pre_llm_call failed: %s", exc)
        return None

    def update_model(self, model, context_length, base_url="", api_key="",
                     provider="", api_mode="", **kw):
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.provider, self.api_mode = provider, api_mode
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    def get_tool_schemas(self):
        return [RLM_SEARCH_SCHEMA]

    def handle_tool_call(self, name, args, **kwargs):
        if name != "rlm_search":
            return json.dumps({"error": f"Unknown RLM tool: {name}"})
        result = execute_rlm_search(
            query=args.get("query", ""),
            scope=args.get("scope", "current"),
            session_id=self._session_id,
            hermes_home=self._hermes_home,
            cached_fts_hits=self._cached_hits,
        )
        # Clear cache after use
        self._cached_hits = None
        self._cached_query = ""
        return result

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
    """Register RLM v2 as either a context engine or a regular plugin tool."""
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

    is_context_engine = False
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        ctx_cfg = cfg.get("context", {})
        is_context_engine = ctx_cfg.get("engine") == "rlm"
    except Exception:
        pass

    if is_context_engine:
        engine = RLMContextEngine()
        if hasattr(ctx, "register_context_engine"):
            ctx.register_context_engine(engine)
        logger.info("RLM: registered as context engine")
    else:
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
