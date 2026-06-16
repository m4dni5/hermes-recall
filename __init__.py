"""RLM plugin — search archived conversation context via RLM retrieval.

Two modes:
- Regular plugin: exposes rlm_search tool alongside default compressor
- Context engine: replaces compressor (set context.engine: rlm in config)
"""

from .engine import RLMContextEngine, register  # noqa: F401

__all__ = ["RLMContextEngine", "register"]
