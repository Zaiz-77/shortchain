"""
Agent —— 单个智能体。

每个 Agent 持有：
  - 自己的 LLM 配置（model / api_key / base_url）
  - 独立的工具注册表（直接注册的工具 + 来自 Skill 的工具）
  - 独立的 SkillManager
  - 独立的 ShortTermMemory
  - 可选的 LongTermMemory（按 agent_id 隔离）
  - 可选的 response_model（Pydantic BaseModel 子类）
  - 自己的 ReActRunner
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from shortchain.config import get_api_key, get_base_url, get_default_model
from shortchain.core.message import Message, Role
from shortchain.core.runner import ReActRunner
from shortchain.memory.short_term import ShortTermMemory
from shortchain.memory.long_term import LongTermMemory
from shortchain.skills.base import SkillManager, Skill
from shortchain.tools.base import Tool


class Agent:
    """
    单个 ReAct Agent。

    Parameters
    ----------
    name:
        Agent 名称，同时作为长期记忆的 agent_id。
    system_prompt:
        基础系统提示词，Skill 的 instructions 会自动追加在其后。
    model:
        使用的模型名称，默认读取 OPENAI_DEFAULT_MODEL 环境变量。
    api_key:
        OpenAI API Key，默认读取 OPENAI_API_KEY 环境变量。
    base_url:
        API base URL，默认读取 OPENAI_BASE_URL 环境变量。
    tools:
        初始工具列表。
    skills:
        初始 Skill 列表。
    response_model:
        若提供，Agent 最终输出会被解析为该 Pydantic 类的实例。
    max_iterations:
        ReAct 循环最大迭代次数，防止无限循环，默认 10。
    max_messages:
        短期记忆保留的最大消息条数，默认 50。
    enable_long_term_memory:
        是否启用长期记忆（本地文件持久化），默认 False。    tool_calling:
        是否启用 function calling（工具调用），默认 True。
        若使用的模型/端点不支持 tools 参数，请设为 False。"""

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        tools: list[Tool] | None = None,
        skills: list[Skill] | None = None,
        response_model: Type[BaseModel] | None = None,
        max_iterations: int = 10,
        max_messages: int = 50,
        enable_long_term_memory: bool = False,
        tool_calling: bool = True,
    ) -> None:
        self.name = name
        self.model = model or get_default_model()
        self.api_key = api_key or get_api_key()
        self.base_url = base_url or get_base_url()
        self.response_model = response_model
        self.max_iterations = max_iterations
        self.tool_calling = tool_calling

        # 工具注册表（直接挂载的工具，按名称去重）
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self._tools[t.name] = t

        # Skill 管理器
        self._skill_manager = SkillManager()
        for s in skills or []:
            self._skill_manager.register(s)

        # 短期记忆
        self.short_term_memory = ShortTermMemory(max_messages=max_messages)

        # 长期记忆（可选）
        self.long_term_memory: LongTermMemory | None = (
            LongTermMemory(agent_id=name) if enable_long_term_memory else None
        )

        # 构建初始 system prompt 并写入记忆
        self._base_system_prompt = system_prompt
        self._sync_system_message()

        # ReAct 执行器
        self._runner = ReActRunner(self)

    # ------------------------------------------------------------------ #
    # 对话入口
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> Any:
        """
        发送一条用户消息并执行 ReAct 循环，返回最终答案。

        若设置了 response_model，返回对应的 Pydantic 实例；
        否则返回字符串。
        """
        return self._runner.run(user_input)

    def reset(self) -> None:
        """清空短期记忆，保留 system prompt，开始新会话。"""
        self.short_term_memory.clear()
        self._sync_system_message()

    # ------------------------------------------------------------------ #
    # 工具管理
    # ------------------------------------------------------------------ #

    def add_tool(self, tool: Tool) -> "Agent":
        """向 Agent 注册一个工具，返回自身支持链式调用。"""
        self._tools[tool.name] = tool
        return self

    def remove_tool(self, tool_name: str) -> "Agent":
        """按名称移除工具。"""
        self._tools.pop(tool_name, None)
        return self

    def get_tool(self, tool_name: str) -> Tool | None:
        """按名称查找工具（先找直接注册的，再找 Skill 携带的）。"""
        if tool_name in self._tools:
            return self._tools[tool_name]
        for t in self._skill_manager.collect_tools():
            if t.name == tool_name:
                return t
        return None

    def get_all_tools(self) -> list[Tool]:
        """
        返回全部可用工具（直接注册 + Skill 携带，直接注册优先，去重）。
        """
        combined: dict[str, Tool] = {}
        # Skill 工具先放入（优先级低）
        for t in self._skill_manager.collect_tools():
            combined[t.name] = t
        # 直接注册的工具覆盖同名 Skill 工具（优先级高）
        combined.update(self._tools)
        return list(combined.values())

    # ------------------------------------------------------------------ #
    # Skill 管理
    # ------------------------------------------------------------------ #

    def add_skill(self, skill: Skill) -> "Agent":
        """挂载一个 Skill，并刷新 system prompt。"""
        self._skill_manager.register(skill)
        self._sync_system_message()
        return self

    def remove_skill(self, skill_name: str) -> "Agent":
        """卸载一个 Skill，并刷新 system prompt。"""
        self._skill_manager.unregister(skill_name)
        self._sync_system_message()
        return self

    # ------------------------------------------------------------------ #
    # 长期记忆便捷接口
    # ------------------------------------------------------------------ #

    def remember(self, key: str, value: Any) -> None:
        """存储一条长期记忆事实（需启用 enable_long_term_memory）。"""
        self._require_long_term_memory()
        self.long_term_memory.set_fact(key, value)  # type: ignore[union-attr]

    def recall(self, key: str, default: Any = None) -> Any:
        """读取一条长期记忆事实。"""
        self._require_long_term_memory()
        return self.long_term_memory.get_fact(key, default)  # type: ignore[union-attr]

    def save_summary(self, text: str) -> None:
        """保存一条对话摘要到长期记忆。"""
        self._require_long_term_memory()
        self.long_term_memory.add_summary(text)  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _build_system_prompt(self) -> str:
        """将基础 system prompt 与所有 Skill 的 instructions 拼合。"""
        parts: list[str] = []
        if self._base_system_prompt:
            parts.append(self._base_system_prompt.strip())

        skill_instructions = self._skill_manager.build_instructions()
        if skill_instructions:
            parts.append(skill_instructions)

        # 注入长期记忆摘要（最近 5 条）
        if self.long_term_memory:
            summary_text = self.long_term_memory.get_summaries_text(last_n=5)
            if summary_text:
                parts.append(f"## 历史记忆摘要\n{summary_text}")

        return "\n\n".join(parts)

    def _sync_system_message(self) -> None:
        """重建 system prompt 并更新到短期记忆的 system 消息。"""
        # 移除旧的 system 消息
        self.short_term_memory.clear_all()
        content = self._build_system_prompt()
        if content:
            self.short_term_memory.add(Message.system(content))

    def _require_long_term_memory(self) -> None:
        if self.long_term_memory is None:
            raise RuntimeError(
                f"Agent '{self.name}' 未启用长期记忆，"
                "请在初始化时设置 enable_long_term_memory=True。"
            )

    def __repr__(self) -> str:
        tools = list(self._tools.keys())
        skills = [s.name for s in self._skill_manager.all()]
        return (
            f"Agent(name={self.name!r}, model={self.model!r}, "
            f"tools={tools}, skills={skills}, "
            f"response_model={self.response_model}, "
            f"tool_calling={self.tool_calling})"
        )
