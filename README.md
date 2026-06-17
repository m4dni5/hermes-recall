# hermes-rlm

Hermes Agent context engine plugin — lossless conversation archival with recursive search.

Based on [Recursive Language Models](https://arxiv.org/abs/2512.24601) (arXiv:2512.24601v3).
Adapted from [rlm-minimal](https://github.com/alexzhang13/rlm).

## What it does

Replaces lossy context compression with structured archival. When context pressure hits:

1. **Compress** — evicted messages are serialized to JSON (no summarization, no loss)
2. **Hint** — a pre_llm_call hook runs FTS5 on the user's message and injects a metadata hint: "N messages match this topic. Call rlm_search."
3. **Search** — `rlm_search` loads the JSON archive into a REPL where the model writes Python to filter, query, and synthesize answers

The model gets a `messages` variable (JSON array) it can reason about programmatically. No token limit applies to the archive itself.

## Architecture

```
User message
  → pre_llm_call hook: FTS5 search → metadata hint (1 line in context)
  → LLM call (with last N messages + hint)
  → if rlm_search called:
      → JSON archive loaded into REPL
      → FTS5 hints pre-loaded (indices from hook)
      → model writes Python: search_context() → messages[idx] → llm_query()
      → FINAL() returns answer
```

Three layers the model uses:

| Layer | What | Speed |
|-------|------|-------|
| `search_context(query)` | FTS5 → message indices | fast |
| `messages[idx]` | Full structured data | instant |
| `llm_query(prompt)` | Sub-LLM semantic analysis | ~1-3s |

## Install

```bash
ln -s ~/src/hermes-rlm ~/.hermes/plugins/rlm
# Restart Hermes — that's it
```

No config changes needed. Works alongside the default compressor. rlm_search recovers archived history on demand.

## Context engine mode (optional)

Replace the built-in compressor entirely:

```yaml
# ~/.hermes/config.yaml
context:
  engine: "rlm"

auxiliary:
  rlm:
    model: "gpt-4.1-nano"
    provider: "openrouter"
    timeout: 60
```

```bash
ln -s ~/src/hermes-rlm ~/.hermes/hermes-agent/plugins/context_engine/rlm
hermes config set context.engine rlm
hermes gateway restart
```

## Test harness

Simulates a compression event from real session history:

```bash
cd ~/src/hermes-rlm
~/.hermes/hermes-agent/venv/bin/python test_v2.py "your query"

# Options:
#   --scope current|all   (default: current — session lineage only)
#   --limit N             (default: 500 messages)
#   --session ID          (default: auto-detect latest)
```

## Files

```
__init__.py      # exports RLMContextEngine
plugin.yaml      # metadata
engine.py        # context engine + regular plugin tool dispatch
repl.py          # REPL sandbox, prompts, code execution
test_v2.py       # test harness
v1/              # archived v1 (flat context string, pre-JSON)
```

## Dependencies

- `call_llm(task="rlm")` for sub-queries — routes through `auxiliary.rlm` config
- Reads state.db via Hermes's `SessionDB` (FTS5 full-text search)
- No external dependencies beyond what Hermes already provides