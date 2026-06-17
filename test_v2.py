#!/usr/bin/env python3
"""Test harness for RLM v2 (JSON archive + FTS5 index).

Simulates a compression event: loads messages from state.db,
converts to JSON, runs the REPL loop.

Usage:
    cd ~/src/rlm-hermes
    python3 test_v2.py [query] [--scope all|current] [--limit N]
"""

import json
import os
import sys

# Add plugin dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_messages_from_db(hermes_home: str, session_id: str = None, scope: str = "all", limit: int = 500):
    """Load messages from state.db, simulating a compression event."""
    from hermes_state import SessionDB, DEFAULT_DB_PATH
    from pathlib import Path

    db_path = Path(hermes_home) / "state.db"
    if not db_path.exists():
        print(f"ERROR: state.db not found at {db_path}")
        sys.exit(1)

    db = SessionDB(db_path)

    if scope == "current" and session_id:
        # Load session lineage
        from engine import _get_session_lineage
        lineage = _get_session_lineage(db, session_id)
        print(f"Session lineage: {len(lineage)} sessions")
        if not lineage:
            print("No lineage found, falling back to all")
            scope = "all"

    if scope == "all":
        with db._lock:
            rows = db._conn.execute(
                "SELECT id, session_id, role, content FROM messages "
                "WHERE active = 1 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
    else:
        from engine import _load_messages_from_lineage
        raw = _load_messages_from_lineage(db, lineage)
        rows = [{"id": m["id"], "session_id": m["session_id"], "role": m["role"], "content": m["content"]} for m in raw[:limit]]

    messages = []
    for i, row in enumerate(rows):
        mid = row["id"] if hasattr(row, "keys") else row[0]
        sid = row["session_id"] if hasattr(row, "keys") else row[1]
        role = row["role"] if hasattr(row, "keys") else row[2]
        content = row["content"] if hasattr(row, "keys") else row[3]
        if content:
            content = db._decode_content(content)
            messages.append({"i": i, "mid": mid, "sid": sid, "role": role, "content": content})

    return messages


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RLM v2 test harness")
    parser.add_argument("query", nargs="?", default="What is the agent's self-chosen name and what does it mean?",
                       help="Query to run against the REPL")
    parser.add_argument("--scope", default="current", choices=["all", "current"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--session", default=None, help="Session ID for scope=current (defaults to latest)")
    parser.add_argument("--hermes-home", default=None)
    args = parser.parse_args()

    # Default hermes home
    if not args.hermes_home:
        args.hermes_home = os.path.expanduser("~/.hermes/profiles/rbw")

    # Auto-detect latest session if not specified
    if not args.session and args.scope == "current":
        from hermes_state import SessionDB
        from pathlib import Path
        db = SessionDB(Path(args.hermes_home) / "state.db")
        try:
            rows = db._conn.execute(
                "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchall()
            if rows:
                args.session = rows[0]["id"]
        except Exception:
            pass

    print(f"=== RLM v2 Test Harness ===")
    print(f"Hermes home: {args.hermes_home}")
    print(f"Scope: {args.scope}")
    print(f"Limit: {args.limit}")
    print(f"Query: {args.query}")
    print()

    # Step 1: Load messages (simulates compression event)
    print("--- Loading messages from state.db ---")
    messages_json = load_messages_from_db(
        args.hermes_home,
        session_id=args.session,
        scope=args.scope,
        limit=args.limit,
    )
    total_chars = sum(len(m.get("content", "")) for m in messages_json)
    sessions = set(m.get("sid", "") for m in messages_json)
    print(f"Loaded: {len(messages_json)} messages, {total_chars} chars, {len(sessions)} sessions")
    if args.session:
        print(f"Session: {args.session}")
    print(f"Sessions in scope: {sorted(sessions)}")

    # Step 2: Simulate pre_llm_call hook — FTS5 on the query
    print("\n--- FTS5 pre-search (simulating pre_llm_call hook) ---")
    fts_hints = None
    try:
        from hermes_state import SessionDB
        from pathlib import Path
        db = SessionDB(Path(args.hermes_home) / "state.db")
        results = db.search_messages(args.query, limit=10)
        if results:
            # Convert to indices using message IDs
            mid_lookup = {}
            for i, msg in enumerate(messages_json):
                if "mid" in msg:
                    mid_lookup[msg["mid"]] = i
            fts_hints = []
            for hit in results:
                hit_id = hit.get("id")
                if hit_id is not None and hit_id in mid_lookup:
                    fts_hints.append(mid_lookup[hit_id])
            fts_hints = sorted(set(fts_hints))
            print(f"FTS5 found {len(results)} results → {len(fts_hints)} message indices: {fts_hints}")
        else:
            print("FTS5 found no results")
    except Exception as e:
        print(f"FTS5 pre-search failed: {e}")

    # Step 3: Run the REPL
    # Build session_ids list for search_context scoping
    session_ids_for_repl = list(sessions) if args.scope == "current" else None

    print("\n--- Running REPL ---")
    from repl import run_rlm_repl
    answer = run_rlm_repl(
        messages_json=messages_json,
        query=args.query,
        hermes_home=args.hermes_home,
        session_ids=session_ids_for_repl,
        fts_hints=fts_hints,
    )
    print()
    print("=== ANSWER ===")
    print(answer)
    print()
    print(f"Log: /tmp/rlm_repl.log")


if __name__ == "__main__":
    main()
