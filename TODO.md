# TODO

## Test harness: load a benign session

The test harness currently loads session lineage, which during development
is polluted with test output about the test itself. This creates recursive
noise — the model searches for "Caido FUZZ" and finds previous test runs
searching for "Caido FUZZ", not the actual feature.

**Fix:** add a `--session` option that loads a specific session by ID, and
add a curated test session with known content (e.g., the original "Read
Before Write" naming conversation from `20260528_120855_d2cf7a`). The test
harness should load that session's messages, not the latest session which is
always the current test.

Alternatively: snapshot a small, clean set of messages as a JSON fixture
file (no state.db dependency for CI).

## Convergence: reduce unnecessary iterations

The model often runs 7-12 iterations when 3-4 would suffice. It scans
messages manually instead of trusting `llm_query()` to extract from the
first hit. The system prompt could be more aggressive about "search once,
analyze the results, FINAL."

## FTS5 phrase matching

FTS5 on "Iain M. Banks" returns 0 results because the exact phrase doesn't
match. "Culture Minds" returns 42 hits. Consider adding trigram/substring
fallback, or teaching the model to break phrases into keywords for
`search_context()`.

## JSON archive growth

Each compression event appends to the JSON archive. After many sessions,
the archive could grow large. Need a pruning/rotation strategy — cap at N
messages or N MB, oldest first.

## Model-agnostic prompting

DeepSeek V4 Flash writes ` ` ` python ` ` ` or XML tool calls instead of
` ` ` repl ` ` ` blocks. MiMo v2.5 Pro follows instructions more faithfully.
The prompt currently has a "write ` ` ` repl ` ` ` blocks" rule but some
models ignore it. Options:
- Detect and handle ` ` ` python ` ` ` blocks (already tried, leads to
  more format variants)
- Accept any code fence (tried, too permissive)
- Stronger prompt guidance (current approach — works with some models)
- Runtime hint in the system prompt about the expected model

## Remove v1/ archive directory

The `v1/` directory is a snapshot of the pre-JSON implementation. Keep it
for reference during development, remove before a stable release.