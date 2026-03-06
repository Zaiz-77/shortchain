from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shortchain.skills.base import Skill, SkillManager

__all__ = ["Skill", "SkillManager"]


def __getattr__(name: str) -> object:
    import importlib

    if name in ("Skill", "SkillManager"):
        return getattr(importlib.import_module("shortchain.skills.base"), name)
    raise AttributeError(f"module 'shortchain.skills' has no attribute {name!r}")
