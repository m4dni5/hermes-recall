"""RLM plugin — encapsulated session search via sub-model reasoning loop.

Registers rlm_search as a regular plugin tool. The sub-model uses
Hermes's built-in session_search to explore past conversations.
"""

try:
    from .engine import register  # noqa: F401
except ImportError:
    from engine import register  # noqa: F401

__all__ = ["register"]