# TODO

## Verify `hermes plugins install` end-to-end

Push to GitHub, run `hermes plugins install m4dni5/hermes-rlm` from a clean
profile, verify rlm_search appears in tools and works.

## Test with real session data

The test harness works against live state.db which is polluted with test
runs. Add a curated test fixture — a small JSON file with known messages —
so tests are deterministic.

## Error format from sub-model

The sub-model prompt says "output JSON tool calls" but some models resist
structured output. Monitor how different models handle the format and adjust
parsing accordingly.

## Iteration efficiency

The sub-model loop has max 8 iterations. For simple queries, the model
should converge in 2-3. Monitor whether the prompt is aggressive enough
about "search once, analyze, answer."

## Fallback path exercise

The fallback calls session_search directly + aux model synthesis. Test
this path by forcing a sub-model loop failure and verifying the fallback
produces useful answers.
