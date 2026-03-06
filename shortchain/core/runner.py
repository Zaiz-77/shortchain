"""
ReAct 执行器。

循环逻辑
--------
1. 将当前消息历史发给 LLM
2. 若 LLM 返回 tool_calls  → 执行工具 → 把结果追加到历史 → 回到 1
3. 若 LLM 返回纯文本        → 这是最终答案，结束循环
4. 若设置了 response_model  → 最后一步改用 parse() 接口强制结构化输出

安全限制
--------
- max_iterations : 防止无限循环，默认 10 轮
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from openai import OpenAI
from pydantic import BaseModel

from shortchain.core.message import Message, Role, ToolCall
from shortchain.tools.base import Tool, _strip_schema_titles

if TYPE_CHECKING:
    from shortchain.core.agent import Agent


class ReActRunner:
    """
    ReAct 循环执行器。每个 Agent 实例持有一个独立的 Runner。
    """

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent
        self._client = OpenAI(
            api_key=agent.api_key,
            base_url=agent.base_url,
        )

    # ------------------------------------------------------------------ #
    # 公开入口
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> Any:
        """
        运行一次对话，返回最终答复。

        若 agent 设置了 response_model，返回对应的 Pydantic 实例；
        否则返回字符串。
        """
        agent = self._agent
        memory = agent.short_term_memory

        # 把用户输入加入短期记忆
        memory.add(Message.user(user_input))

        for _ in range(agent.max_iterations):
            messages = memory.to_openai_messages()
            tools_schema = self._build_tools_schema() if agent.tool_calling else []

            response_msg, tool_calls = self._call_llm(messages, tools_schema)
            memory.add(response_msg)

            if not tool_calls:
                # LLM 不再调用工具，这是最终答复
                if agent.response_model is not None:
                    # 把当前完整对话传给结构化输出接口
                    return self._coerce_to_model(memory.to_openai_messages())
                return response_msg.content or ""

            # 执行所有工具调用，把结果追加到记忆
            for tc in tool_calls:
                result = self._invoke_tool(tc)
                memory.add(
                    Message.tool_result(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=result,
                    )
                )

        # 超过最大迭代次数，强制返回
        if agent.response_model is not None:
            return self._coerce_to_model(memory.to_openai_messages())
        last = memory.get_history()[-1]
        return last.content or ""

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _build_tools_schema(self) -> list[dict]:
        """聚合 Agent 上所有工具的 OpenAI schema。"""
        return [t.openai_schema() for t in self._agent.get_all_tools()]

    def _call_llm(
        self,
        messages: list[dict],
        tools_schema: list[dict],
    ) -> tuple[Message, list[ToolCall] | None]:
        """调用 LLM，返回 (Message, tool_calls | None)。"""
        kwargs: dict[str, Any] = {
            "model": self._agent.model,
            "messages": messages,
        }
        if tools_schema:
            kwargs["tools"] = tools_schema
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # 如果因为工具参数导致 API 报错，自动降级为不带工具重试一次
            if tools_schema and _is_tools_unsupported_error(e):
                import warnings

                warnings.warn(
                    f"[shortchain] 当前模型/端点不支持 function calling，"
                    f"已自动降级为纯文本模式。错误信息：{e}\n"
                    "建议在 Agent 初始化时设置 tool_calling=False 以消除此警告。",
                    stacklevel=3,
                )
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise

        choice = response.choices[0]
        msg = Message.from_openai_choice(choice)
        tool_calls = msg.tool_calls  # None 或 list[ToolCall]
        return msg, tool_calls

    def _call_structured(self, messages: list[dict]) -> BaseModel:
        """使用 OpenAI Structured Outputs 接口，强制返回 Pydantic 实例。"""
        response = self._client.beta.chat.completions.parse(
            model=self._agent.model,
            messages=messages,
            response_format=self._agent.response_model,
        )
        return response.choices[0].message.parsed

    def _coerce_to_model(self, messages: list[dict]) -> BaseModel:
        """
        将已完成的对话强制转换为 response_model 实例。

        优先尝试 OpenAI Structured Outputs（beta.parse），
        若模型不支持，则降级为 JSON 提示词方案：追加一条 user 消息要求
        模型只输出符合 schema 的 JSON，再解析结果。
        """
        try:
            return self._call_structured(messages)
        except Exception:
            pass

        # 降级：在对话末尾追加 JSON 格式要求
        schema = self._agent.response_model.model_json_schema()
        _strip_schema_titles(schema)
        coerce_messages = messages + [
            {
                "role": "user",
                "content": (
                    "请将上面的回答整理为符合以下 JSON Schema 的 JSON 对象，"
                    "只输出 JSON，不要有任何其他内容：\n"
                    + json.dumps(schema, ensure_ascii=False)
                ),
            }
        ]
        response = self._client.chat.completions.create(
            model=self._agent.model,
            messages=coerce_messages,
        )
        text = response.choices[0].message.content or ""
        return self._parse_response_model(text)

    def _parse_response_model(self, text: str) -> BaseModel:
        """将 LLM 返回的文本解析为 response_model 实例（兜底）。"""
        text = text.strip()
        # 去掉可能的 markdown 代码块包裹
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        try:
            return self._agent.response_model.model_validate_json(text)
        except Exception:
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return self._agent.response_model.model_validate_json(match.group())
            raise ValueError(
                f"无法将 LLM 输出解析为 {self._agent.response_model.__name__}:\n{text}"
            )

    def _invoke_tool(self, tool_call: ToolCall) -> str:
        """按工具名查找并执行工具，返回字符串结果。"""
        tool = self._agent.get_tool(tool_call.name)
        if tool is None:
            return json.dumps(
                {"error": f"工具 '{tool_call.name}' 未找到"},
                ensure_ascii=False,
            )
        try:
            return tool.run(**tool_call.arguments)
        except Exception as e:
            return json.dumps(
                {"error": f"工具 '{tool_call.name}' 执行出错: {e}"},
                ensure_ascii=False,
            )


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #


def _is_tools_unsupported_error(exc: Exception) -> bool:
    """
    判断异常是否由模型/端点不支持 function calling 导致。
    覆盖常见的 404 / 400 / 'not support' 类错误。
    """
    msg = str(exc).lower()
    keywords = ("not support", "not found", "unsupported", "tool", "function")
    status_codes = (400, 404, 422)

    # openai SDK 的 APIStatusError 带有 status_code 属性
    status_code = getattr(exc, "status_code", None)
    if status_code in status_codes:
        return any(k in msg for k in keywords)
    return False
