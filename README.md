# hermes-recall

Hermes Agent plugin - "TOTAL RECALL"

First versions were inspired by [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/), and applied a [REPL environment](https://github.com/alexzhang13/rlm-minimal) using Hermes' `llm_query` to programatically search over Hermes Agent session history. Testing revealed that for structured data (SQLite session DB) with already good tools (FTS5 and/or the Hermes-native session_search), the submodels were just not very creative in their solutions.

So now this plugin just encapsulates what your Hermes Agent will usually do if you ask it to search a past session: call session_search a bunch of times and reason over it. Main benefits are: the whole search-and-synthesize process stays out of your main session context, and you can designate a cheap auxiliary model to handle it.

It was a fun experiment!

Enjoy

Matt

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

The sub-model gets the same `session_search` interface the main agent
uses — discovery, scroll, and browse — via native function calling.

The main agent's context only sees the final answer — not the intermediate
searches, not the scrolling, not the synthesis reasoning.

## Install

```bash
hermes plugins install m4dni5/hermes-recall
hermes plugins enable hermes-recall
```

## Configuration (optional)

Override the sub-model:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  recall:
    model: "deepseek-v4-flash"
    provider: "nous"
    timeout: 60
```

## Files

```
__init__.py      # exports register()
plugin.yaml      # metadata
schemas.py       # RECALL_SCHEMA
loop.py          # sub-model loop, parsing, prompts
recall_tools.py  # session DB access, plugin registration
pyproject.toml   # pytest config
tests/
  test_rlm.py    # test harness
  test_engine.py # unit tests
```
