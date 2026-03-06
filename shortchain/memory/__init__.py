from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shortchain.memory.short_term import ShortTermMemory
    from shortchain.memory.long_term import LongTermMemory

__all__ = ["ShortTermMemory", "LongTermMemory"]


def __getattr__(name: str) -> object:
    import importlib

    _map = {
        "ShortTermMemory": ("shortchain.memory.short_term", "ShortTermMemory"),
        "LongTermMemory": ("shortchain.memory.long_term", "LongTermMemory"),
    }
    if name in _map:
        mod, attr = _map[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'shortchain.memory' has no attribute {name!r}")
