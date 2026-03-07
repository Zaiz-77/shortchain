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
        verbose = agent.verbose

        # 把用户输入加入短期记忆
        memory.add(Message.user(user_input))

        for iteration in range(agent.max_iterations):
            if verbose:
                print(f"\n[Iteration {iteration + 1}]")

            messages = memory.to_openai_messages()
            tools_schema = self._build_tools_schema() if agent.tool_calling else []

            # 当 agent.stream=True 时，_call_llm 内部使用流式调用，
            # 最终回答的 token 会实时打印到 stdout
            response_msg, tool_calls = self._call_llm(messages, tools_schema)
            memory.add(response_msg)

            if not tool_calls:
                # LLM 不再调用工具，这是最终答复
                # （若 stream=True，内容已由 _call_llm 实时打印完毕）
                content = response_msg.content or ""

                if agent.response_model is not None:
                    try:
                        return self._parse_response_model(content)
                    except Exception:
                        return self._coerce_to_model(memory.to_openai_messages())

                if verbose and not agent.stream:
                    # stream=True 时内容已流式打印，不重复输出
                    print(f"\n[Final Answer]\n{content}")

                return content

            # ── 有工具调用 ────────────────────────────────────────────────
            # verbose: 打印 Thought（模型在工具调用前输出的内容，若有）
            if verbose and response_msg.content:
                print(f"\n[Thought]\n{response_msg.content}")

            for tc in tool_calls:
                if verbose:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    print(f"\n[Action] {tc.name}({args_str})")

                result = self._invoke_tool(tc)

                if verbose:
                    print(f"[Observation] {result}")

                memory.add(
                    Message.tool_result(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=result,
                    )
                )

        # 超过最大迭代次数，强制返回
        last_content = memory.get_history()[-1].content or ""
        if agent.response_model is not None:
            try:
                return self._parse_response_model(last_content)
            except Exception:
                return self._coerce_to_model(memory.to_openai_messages())
        return last_content

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
        """调用 LLM，返回 (Message, tool_calls | None)。

        当 agent.stream=True 时，使用流式接口并实时将 token 打印到 stdout。
        """
        kwargs: dict[str, Any] = {
            "model": self._agent.model,
            "messages": messages,
        }
        if tools_schema:
            kwargs["tools"] = tools_schema
            kwargs["tool_choice"] = "auto"
        if self._agent.stream:
            kwargs["stream"] = True

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

        if self._agent.stream:
            return self._consume_stream(response)

        choice = response.choices[0]
        msg = Message.from_openai_choice(choice)
        return msg, msg.tool_calls

    def _consume_stream(self, stream: Any) -> tuple[Message, list[ToolCall] | None]:
        """消费流式响应，实时打印 content token，累积 tool_calls，返回完整 Message。

        注意：对于工具调用响应，模型通常不输出 content，因此不会有额外的打印。
        对于最终回答，content token 会实时输出到 stdout。
        """
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}  # index → {id, name, arguments}
        printed_any = False

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 打印 content token
            if delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)
                printed_any = True

            # 累积 tool_calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx][
                                "arguments"
                            ] += tc_delta.function.arguments

        if printed_any:
            print()  # 流式内容打印完毕后换行

        content = "".join(content_parts)

        if tool_calls_acc:
            tool_calls: list[ToolCall] = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc["id"], name=tc["name"], arguments=args)
                )
            msg = Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
            return msg, tool_calls

        msg = Message(role=Role.ASSISTANT, content=content)
        return msg, None

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
                    "Reformat your previous answer as a valid JSON object that conforms "
                    "to the following JSON Schema. Output only the JSON, nothing else:\n"
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
