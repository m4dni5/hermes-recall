# rlm-hermes

Hermes Agent context engine plugin — replaces lossy summarization with retrieval-based context management inspired by [Recursive Language Models](https://github.com/alexzhang13/rlm) ([paper](https://arxiv.org/abs/2512.24601v1)).

## How it works

Standard Hermes compresses context by summarizing old messages (lossy, slow, uses an LLM call).

This plugin: **nothing is discarded.** All messages stay in `state.db`. The context window is trimmed to a recent tail, and the agent gets an `rlm_search` tool to retrieve archived context on demand.

```
Standard:  [system] [...middle...] → summarize (LLM) → [system] [summary] [tail]
RLM:       [system] [...middle...] → drop (instant)  → [system] [note] [tail]
                                                          ↓ agent needs old context
                                                    rlm_search("what about X?")
                                                          ↓
                                                    FTS5 search state.db
                                                          ↓
                                                    sub-query cheap model
                                                          ↓
                                                    synthesized answer
```

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
| Retrieval of old context | Not possible | `rlm_search` with FTS5 + sub-query |
| Quality of old context | Summary (may lose details) | Original messages (exact) |
| Agent must actively retrieve | No (summary injected) | Yes (must call rlm_search) |

## Limitations

- **FTS5 is keyword search**, not semantic. Agent must phrase queries matching original text.
- **Sub-query uses a single LLM call.** Very large result sets may exceed the aux model's window.
- **No chunk result caching.** Every search re-runs the full pipeline.
- **Agent must know to call rlm_search.** The context note helps but isn't foolproof.

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
