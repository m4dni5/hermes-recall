# rlm-hermes

Hermes Agent plugin — search archived conversation context using [Recursive Language Models](https://arxiv.org/abs/2512.24601v1).

Works in two modes:

## Mode 1: Regular Plugin (recommended to start)

Drop into `~/.hermes/plugins/rlm/` and restart. Gives the agent an `rlm_search` tool that works **alongside the default compressor**.

```
User: "What did we decide about the auth flow?"
  → Agent doesn't have old context (it was compressed)
  → Agent calls rlm_search(query="auth flow decision")
  → Plugin loads ALL messages from session lineage into state.db
  → REPL model writes Python to search, chunk, and process
  → Answer injected into conversation
```

**No config changes needed.** The default compressor handles context management normally. rlm_search recovers compressed history on demand.

```bash
ln -s ~/src/rlm-hermes ~/.hermes/plugins/rlm
# Restart Hermes — that's it
```

## Mode 2: Context Engine

Replace the built-in compressor entirely. compress() trims to a tail (instant, no LLM), pre_llm_call hook auto-retrieves each turn, rlm_search does deep dives.

```yaml
# ~/.hermes/config.yaml
context:
  engine: "rlm"

# Cheap model for sub-queries (important!)
auxiliary:
  compression:
    model: "gpt-4.1-nano"
    provider: "openrouter"
```

```bash
ln -s ~/src/rlm-hermes ~/.hermes/hermes-agent/plugins/context_engine/rlm
hermes config set context.engine rlm
hermes gateway restart  # or restart CLI
```

## How rlm_search works

The plugin adapts the original RLM REPL pattern:

1. **Load context** — all messages from session lineage (no pre-filtering)
2. **REPL loop** — model writes Python code to process the context:
   ```python
   # Model writes this kind of code automatically
   lines = context.split('\n\n')
   auth_lines = [l for l in lines if 'auth' in l.lower()]
   for chunk in auth_lines[:10]:
       print(llm_query(f"What's the decision? {chunk}"))
   ```
3. **Sub-queries** — `llm_query()` calls the cheap model (auxiliary.compression.model)
4. **Iterate** — model sees results, refines approach, writes more code
5. **FINAL()** — model signals completion with final answer

The context is a Python **variable**, not a prompt. The model accesses it programmatically — no token limit applies to the context itself.

## Works with default compressor

**Key insight:** the built-in compressor summarizes in-memory messages, but ALL original messages are persisted to state.db **before** compression happens. The flush loop writes each message to state.db as it's produced. When compression fires, the originals are already archived.

```
Turn 1-50: messages flushed to state.db (originals preserved)
Compress:  in-memory messages → summary + tail
Turn 51+:  new messages flushed to state.db
rlm_search: loads ALL messages from session lineage → original context restored
```

Session lineage (`parent_session_id` chains) connects compressed sessions. rlm_search walks the chain and loads everything.

## Two layers (Mode 2 only)

| Layer | Trigger | Speed | How |
|-------|---------|-------|-----|
| `pre_llm_call` hook | Every turn | ~1-3s | Loads context → single sub-query → inject |
| `rlm_search` tool | Agent-initiated | ~10-30s | Full REPL with model agency |

## Dependencies

- `repl.py` — adapted from [rlm-minimal](https://github.com/alexzhang13/rlm)
- Uses `call_llm(task="compression")` for sub-queries — routes through `auxiliary.compression.model` config
- Reads state.db via Hermes's `SessionDB`

## Files

```
__init__.py      # exports RLMContextEngine
plugin.yaml      # metadata
engine.py        # both modes: context engine + regular plugin tool
repl.py          # REPL sandbox (adapted from rlm-minimal)
README.md        # this file
```
