# hermes-recall — Agent Notes

## What this is

A Hermes Agent plugin that exposes a single tool `recall(query)`. The tool
runs a sub-model reasoning loop over past conversations using Hermes's
built-in `session_search` (FTS5 over `state.db`). The sub-model discovers
matching sessions, scrolls into messages for detail, and returns a
synthesized answer. The main agent's context only sees the final answer —
not the intermediate searches, scrollback, or reasoning.

## Architecture

```
recall(query)
  │
  └─ _handle_recall_tool(args, **kwargs)
       │  kwargs["session_id"] = current session (from framework)
       │  hermes_home = get_hermes_home()
       └─ execute_recall(query, session_id, hermes_home)
            │
            ├─ _get_session_db(hermes_home)
            │    SessionDB(read_only=True) + FTS5 probe
            │
            └─ run_sub_model_loop(query, db, session_id)
                 │
                 │  call_llm(task="recall", tools=[SESSION_SEARCH_TOOL])
                 │  ↻ model calls session_search → _dispatch_session_search → tool result
                 │  → model calls again or returns text
                 │
                 └─ return answer string
```

The loop uses **native function calling** via `call_llm(tools=[...])`. The
model gets `session_search` as a structured tool definition; the loop
dispatches each call and feeds the result back as a `tool` role message.
Termination: the model returns a response without `tool_calls` — that text
is the answer. If max iterations (8) is reached, a final call without tools
forces a text synthesis.

Fallback: if the loop raises, `execute_recall` catches and falls back to a
single `session_search` + direct aux model synthesis. If that also fails, a
clean error JSON is returned.

## How it wires into Hermes

**`call_llm` from `agent.auxiliary_client.py`.** The loop calls the
framework's centralized auxiliary LLM client, which resolves provider + model
from `auxiliary.recall` config (registered via `ctx.register_auxiliary_task`
in `register()`). The `auto` provider chain and credential pooling apply
automatically. The `tools` parameter is passed through to
`client.chat.completions.create()` — no bypass needed.

**`session_search` from `tools/session_search_tool.py`.** Called directly as
a Python function, not through `ctx.dispatch_tool`. The function takes a `db`
parameter (a `SessionDB` instance) and returns a JSON string.
`_dispatch_session_search` wraps it and formats the result into readable
text for the sub-model.

**`SessionDB` from `hermes_state.py`.** Opened read-only. See the FTS5
gotcha below — the read-only constructor skips schema init, so the FTS5 flag
must be set manually.

**`get_hermes_home` from `hermes_constants.py`.** Resolves the profile-aware
home directory so `state.db` is found correctly regardless of which profile
the agent runs under.

## Plugin structure

Flat layout — required by `hermes plugins install`:

```
hermes-recall/
├── plugin.yaml         # name, version, provides_tools
├── __init__.py         # exports register()
├── schemas.py          # RECALL_SCHEMA (what the LLM sees)
├── loop.py             # sub-model loop, tool definition, system prompt
├── recall_tools.py     # session DB access, dispatch, plugin registration
├── pyproject.toml      # pytest config only (pythonpath = ["."])
└── tests/
    ├── test_engine.py  # unit tests (mocked LLM, schema, parsing)
    └── test_rlm.py     # integration harness against real state.db
```

The plugin is loaded as `hermes_plugins.recall` — the framework's plugin
loader creates a namespace package at `sys.modules["hermes_plugins"]` and
registers the plugin under it with `submodule_search_locations` pointing at
the plugin directory. Relative imports (`.recall_tools`, `.loop`, `.schemas`)
resolve within this namespace package.

**Why `recall_tools.py` instead of `tools.py`:** Hermes has a `tools/`
package (~90 modules) at its root. A flat `tools.py` in the plugin shadows
it when the plugin directory is on `sys.path` (e.g. in the test harness).
Renaming to `recall_tools.py` avoids the collision while preserving the
flat structure.

## Design decisions

**Native tool calling, not regex parsing.** The first version parsed
`{"tool": "session_search", "args": {...}}` from the model's plain-text
output using regex. This was fragile — the fence parser was dead code
(broke on nested braces), the regex assumed `"tool"` before `"args"`, and
every model formatting quirk was a silent stall. Switching to native
function calling via `call_llm(tools=[...])` eliminates the entire parsing
layer: the API returns structured `tool_calls`, the loop dispatches them,
termination is "no tool_calls in the response." The system prompt is simpler
too — no JSON format instructions, no "write ONLY one tool call" rules.

**No `FINAL(answer)` convention.** The old loop required the model to emit
`FINAL(answer)` to signal completion, parsed with a regex. With native tool
calling, the absence of `tool_calls` in a response IS the termination signal.
This is more robust and more auditable.

**`check_fn` gates on `state.db` existence.** The tool is hidden from the
model when there's no session database, rather than returning an error at
runtime. Uses `get_hermes_home()` to respect the active profile.

**`register_auxiliary_task` for model config.** The plugin registers a
`recall` auxiliary task with `defaults={"provider": "auto", "model": "",
"timeout": 120}`. Users can override in `config.yaml` under
`auxiliary.recall`. The `auto` provider chain resolves the best available
backend. The model should support function calling — most current models do.

**No bundled skill.** The tool schema description is the documentation. A
skill would carry the same content plus usage recipes; the schema is
sufficient and loads at call time.

## Gotchas

- **`SessionDB(read_only=True)` skips `_init_schema()`.** This means
  `_fts_enabled` stays `False` and `search_messages()` silently returns `[]`.
  The fix: after opening, probe `sqlite_master` for the `messages_fts` table
  and set `db._fts_enabled = True` manually. See `_get_session_db` in
  `recall_tools.py`. This is arguably a bug in `hermes_state.py` — the
  read-only constructor could probe FTS5 availability since the tables
  already exist — but the plugin handles it locally to avoid an upstream
  dependency.

- **`call_llm` is imported at module level in `loop.py`.** In test mode
  (bare modules on `pythonpath`), the `from agent.auxiliary_client import
  call_llm` fails and falls back to `None`. Tests patch `loop.call_llm`
  with a mock. In production (namespace package), the import resolves
  because Hermes is on `sys.path`.

- **`session_id` comes from `**kwargs` in the tool handler.** The framework
  passes the current session ID through `kwargs["session_id"]` when
  dispatching tool calls. This is threaded into `session_search` as
  `current_session_id` so the discovery mode excludes the current session's
  lineage (those messages are already in context).

- **The plugin does NOT own `state.db`.** It opens read-only and never
  writes. The DB is created and maintained by Hermes's own `SessionDB`
  instances. If the DB doesn't exist (fresh install), the tool is hidden
  via `check_fn`.

- **Test harness imports must resolve before the plugin dir goes on
  `sys.path`.** `test_rlm.py` adds `~/.hermes/hermes-agent` to `sys.path`
  first, imports `hermes_state`, then adds the plugin directory. Without
  this ordering, `from hermes_state import SessionDB` can fail if the
  plugin's `recall_tools.py` is picked up first (though the rename fixed
  the worst case).

## Test plan

```bash
# Unit tests (no LLM calls, no real state.db)
pytest tests/test_engine.py

# Integration harness (real state.db, real LLM)
python3 tests/test_rlm.py "your query"
python3 tests/test_rlm.py "your query" --session SESSION_ID
python3 tests/test_rlm.py "your query" --hermes-home ~/.hermes/profiles/rbw
```

The unit tests mock `call_llm` and `_dispatch_session_search` to test the
loop's control flow: tool-call-then-answer, immediate text answer,
max-iterations synthesis, unknown-tool rejection, malformed arguments,
multiple tool calls per turn.

The integration harness runs against a real `state.db` with a real LLM.
It's the only way to verify the FTS5 fix and the end-to-end dispatch path.
