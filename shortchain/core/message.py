"""统一消息数据结构。"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """代表 LLM 发出的一次工具调用请求。"""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_openai(cls, raw: Any) -> "ToolCall":
        """从 openai ChatCompletionMessageToolCall 对象构建。"""
        return cls(
            id=raw.id,
            name=raw.function.name,
            arguments=json.loads(raw.function.arguments or "{}"),
        )


class Message(BaseModel):
    """框架内统一消息格式，可与 OpenAI API 互转。"""

    role: Role
    content: str | None = None
    name: str | None = None  # 用于 tool 消息的工具名
    tool_call_id: str | None = None  # tool 消息关联的调用 id
    tool_calls: list[ToolCall] | None = None  # assistant 发出的调用列表

    # ------------------------------------------------------------------ #
    # 工厂方法
    # ------------------------------------------------------------------ #

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> "Message":
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, name: str, content: str) -> "Message":
        return cls(
            role=Role.TOOL, content=content, name=name, tool_call_id=tool_call_id
        )

    # ------------------------------------------------------------------ #
    # OpenAI 格式互转
    # ------------------------------------------------------------------ #

    def to_openai_dict(self) -> dict[str, Any]:
        """转换为 OpenAI API 接受的消息字典。"""
        d: dict[str, Any] = {"role": self.role.value}

        if self.role == Role.TOOL:
            # tool 结果消息：只需 role / tool_call_id / content
            # 不包含 name 字段（Qwen 等兼容接口不支持）
            d["tool_call_id"] = self.tool_call_id or ""
            d["content"] = self.content or ""
            return d

        if self.tool_calls:
            # assistant 发出工具调用时，content 必须为字符串（即使为空）
            d["content"] = self.content if self.content is not None else ""
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
            return d

        # 普通消息（system / user / assistant 纯文本）
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name

        return d

    @classmethod
    def from_openai_choice(cls, choice: Any) -> "Message":
        """从 openai ChatCompletion choice 对象构建。"""
        msg = choice.message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [ToolCall.from_openai(tc) for tc in msg.tool_calls]
        return cls(
            role=Role(msg.role),
            content=msg.content,
            tool_calls=tool_calls,
        )
