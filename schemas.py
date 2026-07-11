"""RLM tool schema — rlm_search takes a query, returns a synthesized answer."""

RLM_SEARCH_SCHEMA = {
    "name": "rlm_search",
    "description": (
        "Search archived conversation history via an encapsulated sub-model. "
        "The sub-model explores past sessions using FTS5 search and returns "
        "a synthesized answer. Use when you need context from earlier "
        "conversations that's no longer in your active window."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language question about past conversation context",
            },
        },
        "required": ["query"],
    },
}
