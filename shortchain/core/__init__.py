from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shortchain.core.message import Message, Role, ToolCall
    from shortchain.core.agent import Agent
    from shortchain.core.coagent import CoAgent
    from shortchain.core.runner import ReActRunner

__all__ = ["Message", "Role", "ToolCall", "Agent", "CoAgent", "ReActRunner"]


def __getattr__(name: str) -> object:
    import importlib

    _map = {
        "Message": ("shortchain.core.message", "Message"),
        "Role": ("shortchain.core.message", "Role"),
        "ToolCall": ("shortchain.core.message", "ToolCall"),
        "Agent": ("shortchain.core.agent", "Agent"),
        "CoAgent": ("shortchain.core.coagent", "CoAgent"),
        "ReActRunner": ("shortchain.core.runner", "ReActRunner"),
    }
    if name in _map:
        mod, attr = _map[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'shortchain.core' has no attribute {name!r}")
