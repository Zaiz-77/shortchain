"""
ShortChain —— 轻量级 Agent 框架

懒加载设计：子模块只在首次访问时才被导入，
避免尚未实现的模块影响整包导入，也加快包的初始化速度。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "CoAgent",
    "Message",
    "Role",
    "ToolCall",
    "tool",
    "Tool",
    "FunctionTool",
    "MCPClient",
    "MCPTool",
    "Skill",
    "ShortTermMemory",
    "LongTermMemory",
]

if TYPE_CHECKING:
    # 仅供静态类型检查器使用，运行时不执行
    from shortchain.core.agent import Agent
    from shortchain.core.coagent import CoAgent
    from shortchain.core.message import Message, Role, ToolCall
    from shortchain.tools.base import tool, Tool, FunctionTool
    from shortchain.tools.mcp import MCPClient, MCPTool
    from shortchain.skills.base import Skill
    from shortchain.memory.short_term import ShortTermMemory
    from shortchain.memory.long_term import LongTermMemory


def __getattr__(name: str) -> object:
    """运行时懒加载：只在被实际访问时才导入对应模块。"""
    _map = {
        "Agent": ("shortchain.core.agent", "Agent"),
        "CoAgent": ("shortchain.core.coagent", "CoAgent"),
        "Message": ("shortchain.core.message", "Message"),
        "Role": ("shortchain.core.message", "Role"),
        "ToolCall": ("shortchain.core.message", "ToolCall"),
        "tool": ("shortchain.tools.base", "tool"),
        "Tool": ("shortchain.tools.base", "Tool"),
        "FunctionTool": ("shortchain.tools.base", "FunctionTool"),
        "MCPClient": ("shortchain.tools.mcp", "MCPClient"),
        "MCPTool": ("shortchain.tools.mcp", "MCPTool"),
        "Skill": ("shortchain.skills.base", "Skill"),
        "ShortTermMemory": ("shortchain.memory.short_term", "ShortTermMemory"),
        "LongTermMemory": ("shortchain.memory.long_term", "LongTermMemory"),
    }
    if name in _map:
        module_path, attr = _map[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'shortchain' has no attribute {name!r}")
