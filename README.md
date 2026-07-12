# hermes-recall

Hermes Agent plugin — encapsulated session search via sub-model reasoning loop.

## What it does

`recall("query")` hides the complexity of searching past conversations behind
a single tool call. Instead of the agent calling `session_search` multiple times
and reasoning over results in its own context window, the sub-model does all of
that inside a dedicated loop and returns a synthesized answer.

```
recall("how does Caido auth work?")
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
hermes plugins install m4dni5/hermes-recall
# Restart Hermes
```

Or for local dev:

```bash
ln -s ~/src/hermes-recall ~/.hermes/plugins/recall
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
cd ~/src/hermes-recall
~/.hermes/hermes-agent/venv/bin/python tests/test_rlm.py "your query"

# Options:
#   --session ID     (default: auto-detect latest)
#   --hermes-home    (default: ~/.hermes/profiles/rbw)
```

## Architecture

```
recall(query)
  │
  └─ sub-model loop (auxiliary.rlm)
       │  tool: session_search(query="...")
       │  tool: session_search(session_id=..., around_message_id=...)
       │  term: FINAL(answer)
       │
       └─ return {answer, method: "recall"}
```

The sub-model gets the same `session_search` interface the main agent
uses — discovery, scroll, and browse.

## Files

```
__init__.py      # exports register()
plugin.yaml      # metadata
schemas.py       # RECALL_SCHEMA
loop.py          # sub-model loop, parsing, prompts
tools.py         # session DB access, plugin registration
pyproject.toml   # pytest config
tests/
  test_rlm.py    # test harness
  test_engine.py # unit tests
```
