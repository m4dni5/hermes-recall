# TODO

## Test with real session data

The test harness works against live state.db which is polluted with test
runs. Add a curated test fixture — a small JSON file with known messages —
so tests are deterministic.

## Error format from sub-model

The sub-model prompt says "output JSON tool calls" but some models resist
structured output. Monitor how different models (DeepSeek V4, MiMo, GPT-4.1)
handle the format and adjust parsing accordingly.

## Iteration efficiency

The sub-model loop has max 8 iterations. For simple queries, the model
should converge in 2-3. Monitor whether the prompt is aggressive enough
about "search once, analyze, answer."

## Session lineage filtering

Currently `session_search` is called with `current_session_id` for the
active session's lineage. Verify this correctly excludes the current
conversation from results (avoiding self-referential noise).

## Fallback path

The fallback calls session_search directly + aux model synthesis. Test
this path by forcing a sub-model loop failure and verifying the fallback
produces useful answers.
