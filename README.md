# hermes-rlm

Hermes Agent plugin — encapsulated session search via sub-model reasoning loop.

## What it does

`rlm_search(query)` hides the complexity of searching past conversations behind
a single tool call. Instead of the agent calling `session_search` multiple times
and reasoning over results in its own context window, the sub-model does all of
that inside a dedicated loop and returns a synthesized answer.

```
rlm_search("how does Caido auth work?")
  → sub-model calls session_search(query="Caido auth")
  → sub-model reads results, decides it needs more
  → sub-model calls session_search(query="device code flow")
  → sub-model scrolls into a matching session
  → sub-model returns FINAL(the complete answer)
→ returned to main agent as a single answer
```

The main agent's context only sees the final answer — not the intermediate
searches, not the scrolling, not the synthesis reasoning.

## Install

```bash
ln -s ~/src/hermes-rlm ~/.hermes/plugins/rlm
# Restart Hermes
```

No config changes needed. The plugin registers as a regular tool and uses
`auxiliary.rlm` for sub-model calls (auto-configured).

## Configuration (optional)

Override the sub-model:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  rlm:
    model: "gpt-4.1-nano"
    provider: "openrouter"
    timeout: 120
```

## Test harness

```bash
cd ~/src/hermes-rlm
python3 tests/test_rlm.py "your query"

# Options:
#   --session ID     (default: auto-detect latest)
#   --hermes-home    (default: ~/.hermes/profiles/rbw)
```

## Architecture

```
rlm_search(query)
  │
  └─ sub-model loop (auxiliary.rlm)
       │  tool: session_search(query="...")
       │  tool: session_search(session_id=..., around_message_id=...)
       │  term: FINAL(answer)
       │
       └─ return {answer, method: "rlm"}
```

The sub-model gets the same `session_search` interface the main agent
uses — discovery, scroll, and browse.

## Files

```
__init__.py      # exports register()
plugin.yaml      # metadata
schemas.py       # RLM_SEARCH_SCHEMA
loop.py          # sub-model loop, parsing, prompts
tools.py         # session DB access, plugin registration
pyproject.toml   # package metadata
tests/
  test_rlm.py    # test harness
  test_engine.py # unit tests
```

## Dependencies

- `call_llm(task="rlm")` for sub-model calls — routes through `auxiliary.rlm`
- `tools.session_search_tool` — Hermes's built-in session search
- `hermes_state.SessionDB` — SQLite session store with FTS5
