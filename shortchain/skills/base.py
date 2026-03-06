"""
Skill —— 可复用的能力单元。

一个 Skill 封装了：
  - instructions : 注入 Agent system prompt 的指令片段
  - tools        : 随 Skill 一起携带的工具列表

设计原则
--------
- Skill 本身无状态，可被多个 Agent 共享
- Agent 可挂载多个 Skill，框架在构建 system prompt 和工具列表时自动合并
- 每个 Agent 对挂载的 Skill 列表独立管理，互不影响

用法示例
--------
    from shortchain.skills import Skill
    from shortchain.tools import tool

    @tool
    def search_web(query: str) -> str:
        '''搜索互联网。:param query: 搜索关键词'''
        ...

    research_skill = Skill(
        name="research",
        description="赋予 Agent 网络搜索和信息综合能力",
        instructions=\"\"\"
你擅长通过搜索引擎收集信息，并将多条结果综合成清晰的答案。
搜索时优先使用精确的关键词，并注明信息来源。
\"\"\",
        tools=[search_web],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shortchain.tools.base import Tool


class Skill:
    """
    可复用的能力单元，包含指令片段与配套工具。

    Parameters
    ----------
    name:
        Skill 的唯一名称，用于标识和去重。
    description:
        对该 Skill 能力的简短描述（供人类阅读）。
    instructions:
        注入 Agent system prompt 的指令文本，描述该 Skill 的行为规范。
    tools:
        该 Skill 附带的工具列表，会被合并到 Agent 的工具集中。
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        instructions: str = "",
        tools: list["Tool"] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions.strip()
        self.tools: list["Tool"] = list(tools or [])

    def add_tool(self, tool: "Tool") -> "Skill":
        """向 Skill 追加一个工具，返回自身以支持链式调用。"""
        self.tools.append(tool)
        return self

    def remove_tool(self, tool_name: str) -> "Skill":
        """按工具名移除工具，返回自身以支持链式调用。"""
        self.tools = [t for t in self.tools if t.name != tool_name]
        return self

    def get_tool_names(self) -> list[str]:
        """返回该 Skill 携带的所有工具名称。"""
        return [t.name for t in self.tools]

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, " f"tools={self.get_tool_names()})"


# --------------------------------------------------------------------------- #
# SkillManager —— Agent 内部的 Skill 注册表
# --------------------------------------------------------------------------- #


class SkillManager:
    """
    管理单个 Agent 挂载的 Skill 集合。

    负责：
    - 维护 Skill 的注册 / 注销
    - 聚合所有 Skill 的 instructions（按注册顺序拼接）
    - 聚合所有 Skill 的工具列表（自动去重，以工具名为 key）
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}  # 有序插入，Python 3.7+ dict 保序

    # ------------------------------------------------------------------ #
    # 注册 / 注销
    # ------------------------------------------------------------------ #

    def register(self, skill: Skill) -> "SkillManager":
        """注册一个 Skill，同名覆盖。返回自身以支持链式调用。"""
        self._skills[skill.name] = skill
        return self

    def unregister(self, skill_name: str) -> "SkillManager":
        """注销一个 Skill。返回自身以支持链式调用。"""
        self._skills.pop(skill_name, None)
        return self

    def get(self, skill_name: str) -> Skill | None:
        """按名称获取 Skill。"""
        return self._skills.get(skill_name)

    def all(self) -> list[Skill]:
        """返回所有已注册的 Skill 列表（按注册顺序）。"""
        return list(self._skills.values())

    # ------------------------------------------------------------------ #
    # 聚合
    # ------------------------------------------------------------------ #

    def build_instructions(self) -> str:
        """
        将所有 Skill 的 instructions 拼接为一段文本。
        每个 Skill 用标题分隔，便于 LLM 理解各段职责。
        """
        parts: list[str] = []
        for skill in self._skills.values():
            if skill.instructions:
                parts.append(f"## Skill: {skill.name}\n{skill.instructions}")
        return "\n\n".join(parts)

    def collect_tools(self) -> list["Tool"]:
        """
        收集所有 Skill 的工具，按工具名去重（先注册的优先）。
        """
        seen: set[str] = set()
        result: list["Tool"] = []
        for skill in self._skills.values():
            for t in skill.tools:
                if t.name not in seen:
                    seen.add(t.name)
                    result.append(t)
        return result

    # ------------------------------------------------------------------ #
    # 其他
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        names = list(self._skills.keys())
        return f"SkillManager(skills={names})"
