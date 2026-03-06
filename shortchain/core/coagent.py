"""
CoAgent —— 多 Agent 协作团队。

支持三种路由模式：
  auto       : 由协调 LLM 自动决定哪个 Agent 处理任务（默认）
  sequential : 按注册顺序依次执行，前一个 Agent 的输出作为下一个的输入
  manual     : 调用方通过 run(agent_name=...) 手动指定执行哪个 Agent

Agent 间通信：
  - 每次 run() 后，结果会保存在 CoAgent 的 history 里
  - sequential 模式中，上一步结果自动拼入下一步的输入
  - 可通过 broadcast(message) 向所有 Agent 的短期记忆注入同一条信息
  - 可通过 handoff(from_agent, to_agent, message) 实现点对点消息传递

用法示例
--------
    researcher = Agent(name="researcher", system_prompt="你是一个研究员...")
    writer     = Agent(name="writer",     system_prompt="你是一个写作专家...")
    reviewer   = Agent(name="reviewer",   system_prompt="你是一个审稿人...")

    # auto 路由
    team = CoAgent(name="team", agents=[researcher, writer, reviewer])
    result = team.run("帮我研究一下量子计算的最新进展")

    # sequential 流水线
    pipeline = CoAgent(name="pipeline", agents=[researcher, writer, reviewer], routing="sequential")
    result = pipeline.run("写一篇关于量子计算的文章")

    # manual 路由
    team.run("检查这段文字是否有语法错误", agent_name="reviewer")
"""

from __future__ import annotations

import json
from typing import Any, Literal, TYPE_CHECKING

from openai import OpenAI

from shortchain.config import get_api_key, get_base_url, get_default_model
from shortchain.core.message import Message

if TYPE_CHECKING:
    from shortchain.core.agent import Agent


class CoAgent:
    """
    多 Agent 协作团队。

    Parameters
    ----------
    name:
        CoAgent 团队名称。
    agents:
        初始 Agent 列表，注册顺序在 sequential 模式中即执行顺序。
    routing:
        路由模式：'auto' | 'sequential' | 'manual'，默认 'auto'。
    coordinator_prompt:
        auto 模式下协调 LLM 使用的额外系统提示，可用于描述团队目标。
    model:
        协调 LLM 使用的模型（仅 auto 模式需要），默认读取环境变量。
    api_key:
        API Key，默认读取环境变量。
    base_url:
        API base URL，默认读取环境变量。
    """

    def __init__(
        self,
        name: str,
        agents: list["Agent"] | None = None,
        routing: Literal["auto", "sequential", "manual"] = "auto",
        coordinator_prompt: str = "",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.name = name
        self.routing = routing
        self.coordinator_prompt = coordinator_prompt
        self.model = model or get_default_model()
        self._api_key = api_key or get_api_key()
        self._base_url = base_url or get_base_url()

        # 有序 Agent 注册表：{agent_name: Agent}
        self._agents: dict[str, "Agent"] = {}
        for agent in agents or []:
            self._agents[agent.name] = agent

        # 运行历史：记录每一步 (agent_name, input, output)
        self.history: list[dict[str, Any]] = []

        # 协调 LLM 客户端（auto 模式专用）
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    # ------------------------------------------------------------------ #
    # Agent 管理
    # ------------------------------------------------------------------ #

    def add_agent(self, agent: "Agent") -> "CoAgent":
        """注册一个 Agent，返回自身支持链式调用。"""
        self._agents[agent.name] = agent
        return self

    def remove_agent(self, agent_name: str) -> "CoAgent":
        """按名称注销一个 Agent。"""
        self._agents.pop(agent_name, None)
        return self

    def get_agent(self, agent_name: str) -> "Agent | None":
        """按名称获取 Agent。"""
        return self._agents.get(agent_name)

    def list_agents(self) -> list[str]:
        """返回所有已注册 Agent 的名称列表（按注册顺序）。"""
        return list(self._agents.keys())

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def run(
        self,
        user_input: str,
        agent_name: str | None = None,
    ) -> Any:
        """
        执行一次团队任务。

        Parameters
        ----------
        user_input:
            用户输入文本。
        agent_name:
            manual 模式下指定执行的 Agent 名称；
            auto / sequential 模式下忽略此参数。

        Returns
        -------
        最终结果（字符串或 Pydantic 实例，取决于所执行 Agent 的 response_model）。
        """
        if not self._agents:
            raise RuntimeError(f"CoAgent '{self.name}' 没有注册任何 Agent。")

        if self.routing == "sequential":
            return self._run_sequential(user_input)
        elif self.routing == "manual":
            return self._run_manual(user_input, agent_name)
        else:  # auto
            return self._run_auto(user_input)

    # ------------------------------------------------------------------ #
    # 路由实现
    # ------------------------------------------------------------------ #

    def _run_auto(self, user_input: str) -> Any:
        """Auto 模式：协调 LLM 决定由哪个 Agent 处理。"""
        chosen_name = self._route(user_input)
        agent = self._agents.get(chosen_name)
        if agent is None:
            # 兜底：取第一个
            agent = next(iter(self._agents.values()))
            chosen_name = agent.name

        result = agent.run(user_input)
        self._record(chosen_name, user_input, result)
        return result

    def _run_sequential(self, user_input: str) -> Any:
        """Sequential 模式：按注册顺序依次执行，前一步输出作为下一步输入。"""
        current_input = user_input
        result: Any = None

        for agent_name, agent in self._agents.items():
            result = agent.run(current_input)
            self._record(agent_name, current_input, result)
            # 将输出转为字符串作为下一个 Agent 的输入
            current_input = result if isinstance(result, str) else str(result)

        return result

    def _run_manual(self, user_input: str, agent_name: str | None) -> Any:
        """Manual 模式：调用方指定 Agent 名称。"""
        if agent_name is None:
            raise ValueError(
                "routing='manual' 时必须通过 agent_name 参数指定要执行的 Agent。"
            )
        agent = self._agents.get(agent_name)
        if agent is None:
            available = list(self._agents.keys())
            raise ValueError(f"Agent '{agent_name}' 未注册。当前可用: {available}")
        result = agent.run(user_input)
        self._record(agent_name, user_input, result)
        return result

    # ------------------------------------------------------------------ #
    # Agent 间通信
    # ------------------------------------------------------------------ #

    def broadcast(self, message: str) -> None:
        """
        向所有 Agent 的短期记忆注入同一条 user 消息。
        可用于向全团队同步背景信息或上下文。
        """
        for agent in self._agents.values():
            agent.short_term_memory.add(Message.user(message))

    def handoff(
        self, from_agent: str, to_agent: str, message: str | None = None
    ) -> Any:
        """
        将 from_agent 最近一次输出（或指定 message）作为 user 消息发送给 to_agent，
        并执行 to_agent，返回其结果。

        Parameters
        ----------
        from_agent:
            消息来源 Agent 名称。
        to_agent:
            接收消息的 Agent 名称。
        message:
            若提供，使用此消息而非 from_agent 的历史输出。
        """
        if message is None:
            # 取 from_agent 最近一次历史输出
            message = self._last_output(from_agent)
            if message is None:
                raise ValueError(
                    f"Agent '{from_agent}' 还没有历史输出，请先执行它或手动传入 message。"
                )

        target = self._agents.get(to_agent)
        if target is None:
            raise ValueError(f"Agent '{to_agent}' 未注册。")

        handoff_input = f"[来自 Agent '{from_agent}' 的输出]\n{message}"
        result = target.run(handoff_input)
        self._record(to_agent, handoff_input, result)
        return result

    # ------------------------------------------------------------------ #
    # 历史与状态
    # ------------------------------------------------------------------ #

    def clear_history(self) -> None:
        """清空团队运行历史。"""
        self.history.clear()

    def reset_all(self) -> None:
        """重置所有 Agent 的短期记忆并清空运行历史。"""
        for agent in self._agents.values():
            agent.reset()
        self.history.clear()

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #

    def _route(self, user_input: str) -> str:
        """调用协调 LLM，从已注册 Agent 中选出最合适的一个，返回其名称。"""
        agent_descs: list[str] = []
        for agent in self._agents.values():
            desc = (
                agent._base_system_prompt.strip().splitlines()[0]
                if agent._base_system_prompt
                else "general assistant"
            )
            agent_descs.append(f"- {agent.name}: {desc}")

        system = (
            "You are a team coordinator. Your job is to assign the user's task to the "
            "most suitable team member.\n"
            + (self.coordinator_prompt + "\n" if self.coordinator_prompt else "")
            + "Team members:\n"
            + "\n".join(agent_descs)
            + "\n\n"
            "Reply with the member name only (exact match). Output nothing else."
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ],
        )
        chosen = (response.choices[0].message.content or "").strip()

        # 如果 LLM 输出不在注册列表里，取最相近的或第一个
        if chosen not in self._agents:
            for name in self._agents:
                if name.lower() in chosen.lower() or chosen.lower() in name.lower():
                    return name
            return next(iter(self._agents))
        return chosen

    def _record(self, agent_name: str, input_text: str, output: Any) -> None:
        """记录一步执行历史。"""
        self.history.append(
            {
                "agent": agent_name,
                "input": input_text,
                "output": output if isinstance(output, str) else str(output),
            }
        )

    def _last_output(self, agent_name: str) -> str | None:
        """获取指定 Agent 最近一次记录的输出。"""
        for entry in reversed(self.history):
            if entry["agent"] == agent_name:
                return entry["output"]
        return None

    def __repr__(self) -> str:
        return (
            f"CoAgent(name={self.name!r}, routing={self.routing!r}, "
            f"agents={self.list_agents()})"
        )
