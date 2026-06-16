# rlm-hermes

Hermes Agent context engine plugin — replaces lossy summarization with retrieval-based context management inspired by [Recursive Language Models](https://github.com/alexzhang13/rlm) ([paper](https://arxiv.org/abs/2512.24601v1)).

## How it works

Standard Hermes compresses context by summarizing old messages (lossy, slow, uses an LLM call).

This plugin: **nothing is discarded.** All messages stay in `state.db`. The context window is trimmed to a recent tail, and archived context is **automatically retrieved** every turn via a `pre_llm_call` hook.

```
Standard:  [system] [...middle...] → summarize (LLM) → [system] [summary] [tail]
RLM:       [system] [...middle...] → drop (instant)  → [system] [note] [tail]
                                                          ↓ every turn, automatic
                                                    pre_llm_call hook:
                                                      FTS5 search state.db
                                                      sub-query cheap model
                                                      inject synthesized context
```

### Three layers

| Layer | What | When | How |
|-------|------|------|-----|
| `compress()` | Trim to tail | Context pressure | Instant — no LLM |
| `pre_llm_call` hook | Auto-retrieve relevant context | Every turn | FTS5 → sub-query → inject |
| `rlm_search` tool | Explicit deep dive | Agent-initiated | Custom query, scope, sort |

## Installation

```bash
# Symlink into Hermes plugin directory
ln -s ~/src/rlm-hermes ~/.hermes/hermes-agent/plugins/context_engine/rlm
```

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
context:
  engine: "rlm"
```

Then restart Hermes (CLI: exit and relaunch; gateway: `hermes gateway restart`).

### Auxiliary model (important)

The `rlm_search` tool uses a sub-query model to synthesize search results. Without config, it falls back to your **main model** — expensive for a retrieval task.

Set a cheap model:

```yaml
auxiliary:
  compression:
    model: "gpt-4.1-nano"        # or gemini-flash, etc.
    provider: "openrouter"        # or "nous", "auto"
```

This is the same config that controls Hermes's built-in compression summarizer. The RLM engine piggybacks on it via `call_llm(task="compression")`.

**Without this, every rlm_search call uses your main model (e.g. Claude Sonnet) for a simple synthesis task.** With a cheap model, retrieval costs ~10x less.

### Full example

```yaml
context:
  engine: "rlm"

auxiliary:
  compression:
    model: "gpt-4.1-nano"
    provider: "openrouter"

# These still apply to the main model's context window:
compression:
  threshold: 0.50              # when to trigger (0.20 for aggressive archiving)
  protect_last_n: 20           # tail messages to keep
```

## How compress() works

When the main model's context window fills up:

1. Messages are already persisted to `state.db` by the normal flush loop
2. `compress()` returns: system prompt + context note + last N messages
3. Middle messages are gone from the **active window** but still in `state.db`
4. The context note tells the agent about `rlm_search`

No LLM calls. No summarization. Instant.

## How rlm_search works

When the agent calls `rlm_search(query="what did we decide about the auth flow")`:

1. Walks the session lineage (`parent_session_id` chain) to find all ancestor sessions
2. **FTS5 search** scoped to the lineage (or all sessions with `scope: "all"`)
3. Chunks the search results
4. Sub-queries a cheap model: "given these results, answer the question"
5. Returns the synthesized answer

### Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `query` | required | Natural language query |
| `scope` | `"current"` | `"current"` = session lineage only, `"all"` = all sessions |
| `sort` | `"relevance"` | `"relevance"`, `"newest"`, `"oldest"` |
| `limit` | `50` | Max messages to retrieve |

## Session lineage with multiple compressions

With aggressive compression (e.g. 20% threshold), compressions fire more often, creating a longer chain:

```
session_A → compress → session_B → compress → session_C → compress → session_D (current)
```

`rlm_search(scope="current")` walks `D → C → B → A` and searches all of them. FTS5 across 10+ sessions is still sub-millisecond.

## Comparison with built-in compressor

| | Built-in `compressor` | RLM engine |
|-|----------------------|------------|
| Old messages | Summarized (lossy) | Archived in state.db (lossless) |
| LLM calls on compression | Yes (aux model) | No |
| Compression speed | Slow (LLM round-trip) | Instant (list slice) |
| Retrieval of old context | Not possible (summary only) | Automatic every turn + explicit tool |
| Quality of old context | Summary (may lose details) | Original messages (exact) |
| Agent must actively retrieve | No (summary injected) | No (hook auto-retrieves) |

### rlm_search (delegated, async)

For the full RLM pipeline (chunk → sub-query per chunk → synthesize), a one-line change to Hermes core is needed to pass the parent agent reference to engine tool calls:

```python
# In agent/tool_executor.py, line ~1137, change:
return agent.context_compressor.handle_tool_call(function_name, next_args, messages=messages)
# To:
return agent.context_compressor.handle_tool_call(function_name, next_args, messages=messages, parent_agent=agent)
```

Without this change, `rlm_search` falls back to lightweight synchronous retrieval (same as the hook). With it, `rlm_search` delegates to a child agent that runs the full RLM pipeline asynchronously.

## Limitations

- **FTS5 is keyword search**, not semantic. The sub-query synthesis mitigates this — the model can understand context even if exact keywords don't match.
- **Per-turn latency**: the `pre_llm_call` hook adds 1-3 seconds (aux model round-trip). With `gpt-4.1-nano` this is fast; with a slower model it adds up.
- **Per-turn cost**: every turn burns tokens on the auxiliary model. With a cheap model, this is negligible. Without `auxiliary.compression.model` set, it falls back to the main model (expensive).
- **No chunk result caching.** Every turn re-runs the full pipeline. Future: cache recent results.
- **Sub-query uses a single LLM call.** Very large result sets may exceed the aux model's window.

## Files

```
__init__.py      # exports RLMContextEngine
plugin.yaml      # plugin metadata
engine.py        # ContextEngine implementation (~350 lines)
README.md        # this file
```

## See also

- [rlm-minimal](https://github.com/alexzhang13/rlm) — original RLM implementation this is inspired by
- [Hermes Context Engine Plugin docs](https://hermes-agent.nousresearch.com/docs/developer-guide/context-engine-plugin)
- [Hermes Context Compression docs](https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching)
