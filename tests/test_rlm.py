#!/usr/bin/env python3
"""Test harness for recall sub-model loop.

Exercises recall against real session history from state.db.

Usage:
    cd ~/src/hermes-recall
    python3 tests/test_rlm.py "your query"
    python3 tests/test_rlm.py "your query" --session SESSION_ID

Options:
    --session ID     Session ID for lineage (default: auto-detect latest)
    --hermes-home    Path to hermes profile (default: ~/.hermes/profiles/rbw)
"""

import argparse
import os
import sys
from pathlib import Path

# Add hermes-agent to sys.path FIRST so framework imports (hermes_state,
# tools.session_search_tool) resolve before the plugin's flat modules.
_HERMES_AGENT = str(Path.home() / ".hermes" / "hermes-agent")
if _HERMES_AGENT not in sys.path:
    sys.path.insert(0, _HERMES_AGENT)

# Framework imports must resolve before the plugin dir goes on sys.path.
from hermes_state import SessionDB  # noqa: E402

# Now add the plugin dir so `from loop import ...` and
# `from recall_tools import ...` work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Recall test harness")
    parser.add_argument(
        "query",
        nargs="?",
        default="What is the agent's self-chosen name and what does it mean?",
        help="Query to run against the sub-model loop",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID for lineage (default: auto-detect latest). Pass 'all' to search all sessions.",
    )
    parser.add_argument(
        "--hermes-home",
        default=None,
        help="Path to hermes profile (default: ~/.hermes/profiles/rbw)",
    )
    args = parser.parse_args()

    # Default hermes home
    hermes_home = args.hermes_home or os.path.expanduser("~/.hermes/profiles/rbw")
    db_path = Path(hermes_home) / "state.db"

    if not db_path.exists():
        print(f"ERROR: state.db not found at {db_path}")
        sys.exit(1)

    db = SessionDB(db_path, read_only=True)
    # SessionDB(read_only=True) skips _init_schema(), leaving
    # _fts_enabled=False. Probe the FTS5 table manually.
    if not db._fts_enabled:
        row = db._conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='messages_fts'"
        ).fetchone()
        if row:
            db._fts_enabled = True

    # Auto-detect latest session if not specified
    session_id = args.session
    if session_id is None:
        try:
            rows = db._conn.execute(
                "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchall()
            if rows:
                session_id = rows[0]["id"]
        except Exception as e:
            print(f"Warning: couldn't auto-detect session: {e}")
    elif session_id == "all":
        session_id = None

    print(f"=== Recall Test Harness ===")
    print(f"Hermes home: {hermes_home}")
    print(f"Session: {session_id or 'none (all sessions)'}")
    print(f"Query: {args.query}")
    print()

    from loop import run_sub_model_loop

    answer = run_sub_model_loop(
        query=args.query,
        db=db,
        current_session_id=session_id or "",
    )

    print()
    print("=== ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
