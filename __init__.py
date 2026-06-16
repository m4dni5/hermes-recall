"""RLM Context Engine — retrieval-based context management.

Replaces lossy summarization with: archive everything, retrieve on demand.

compress() keeps the system prompt + recent tail, drops the middle.
rlm_search tool lets the agent pull old context via FTS5 + sub-query synthesis.
"""

from .engine import RLMContextEngine

__all__ = ["RLMContextEngine"]
