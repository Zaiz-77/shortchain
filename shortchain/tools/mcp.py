"""
MCP 工具适配器。

将 MCP (Model Context Protocol) Server 暴露的工具适配为 ShortChain 的 Tool，
使 Agent 可以像使用普通 @tool 一样使用 MCP 工具。

支持两种 MCP Server 连接方式：
  - stdio : 启动本地子进程（最常见，如 npx / uvx 启动的 MCP server）
  - sse   : 连接远程 SSE HTTP 端点

用法示例
--------
    import asyncio
    from shortchain.tools.mcp import MCPClient

    async def main():
        # stdio 方式
        async with MCPClient.from_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as client:
            tools = await client.get_tools()   # 返回 list[MCPTool]
            agent = Agent(name="mcp_agent", tools=tools)
            result = agent.run("列出当前目录的文件")
            print(result)

    asyncio.run(main())

注意
----
- MCPClient 是异步上下文管理器，需在 async 环境中使用
- Agent.run() 本身是同步接口；MCP 工具调用内部用 asyncio.run() 桥接
- 若已在异步环境中，使用 await client.call_tool() 直接调用
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from shortchain.tools.base import Tool


# --------------------------------------------------------------------------- #
# MCPTool —— 单个 MCP 工具的 Tool 包装
# --------------------------------------------------------------------------- #


class MCPTool(Tool):
    """
    将 MCP Server 的单个工具包装为 ShortChain Tool。

    不直接继承自 FunctionTool，因为调用方式是通过 MCP session 而非本地函数。
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        session: ClientSession,
    ) -> None:
        self.name = name
        self.description = description
        self._input_schema = input_schema
        self._session = session

    def openai_schema(self) -> dict[str, Any]:
        """直接使用 MCP Server 返回的 inputSchema 作为 parameters。"""
        parameters = dict(self._input_schema)
        parameters.pop("title", None)
        # 确保有 type 字段
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def run(self, **kwargs: Any) -> str:
        """同步执行 MCP 工具调用（内部用 asyncio 桥接）。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 已在异步环境中，用 run_coroutine_threadsafe 避免递归
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(self._async_run(**kwargs), loop)
            return future.result(timeout=60)
        else:
            return asyncio.run(self._async_run(**kwargs))

    async def _async_run(self, **kwargs: Any) -> str:
        """异步执行 MCP 工具调用。"""
        result = await self._session.call_tool(self.name, arguments=kwargs)
        # MCP 返回的 content 是 list[TextContent | ImageContent | ...]
        parts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
        return "\n".join(parts) if parts else ""

    def __repr__(self) -> str:
        return f"MCPTool(name={self.name!r})"


# --------------------------------------------------------------------------- #
# MCPClient —— 连接 MCP Server 并暴露工具列表
# --------------------------------------------------------------------------- #


class MCPClient:
    """
    MCP Server 客户端，异步上下文管理器。

    连接后调用 get_tools() 获取工具列表，可直接传给 Agent：

        async with MCPClient.from_stdio(...) as client:
            agent = Agent(name="x", tools=await client.get_tools())
    """

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    # ------------------------------------------------------------------ #
    # 工厂方法
    # ------------------------------------------------------------------ #

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> "MCPClient":
        """
        创建 stdio 连接的 MCPClient（连接本地子进程 MCP Server）。

        Parameters
        ----------
        command:
            启动 MCP Server 的命令，如 "npx" 或 "uvx"。
        args:
            命令参数列表，如 ["-y", "@modelcontextprotocol/server-filesystem", "."]。
        env:
            额外的环境变量字典。
        """
        client = cls()
        client._connect_kwargs = {
            "mode": "stdio",
            "command": command,
            "args": args or [],
            "env": env,
        }
        return client

    @classmethod
    def from_sse(cls, url: str) -> "MCPClient":
        """
        创建 SSE 连接的 MCPClient（连接远程 HTTP MCP Server）。

        Parameters
        ----------
        url:
            MCP Server 的 SSE 端点 URL。
        """
        client = cls()
        client._connect_kwargs = {"mode": "sse", "url": url}
        return client

    # ------------------------------------------------------------------ #
    # 异步上下文管理器
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "MCPClient":
        kwargs = self._connect_kwargs
        mode = kwargs["mode"]

        if mode == "stdio":
            params = StdioServerParameters(
                command=kwargs["command"],
                args=kwargs["args"],
                env=kwargs.get("env"),
            )
            transport = await self._exit_stack.enter_async_context(stdio_client(params))
        elif mode == "sse":
            transport = await self._exit_stack.enter_async_context(
                sse_client(kwargs["url"])
            )
        else:
            raise ValueError(f"不支持的 MCP 连接模式: {mode}")

        read, write = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._exit_stack.aclose()
        self._session = None

    # ------------------------------------------------------------------ #
    # 工具获取
    # ------------------------------------------------------------------ #

    async def get_tools(self) -> list[MCPTool]:
        """获取 MCP Server 提供的所有工具，转换为 MCPTool 列表。"""
        if self._session is None:
            raise RuntimeError("MCPClient 未连接，请在 async with 块内使用。")
        response = await self._session.list_tools()
        tools: list[MCPTool] = []
        for t in response.tools:
            tools.append(
                MCPTool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {},
                    session=self._session,
                )
            )
        return tools

    async def call_tool(self, name: str, **kwargs: Any) -> str:
        """直接异步调用指定工具（在 async 环境中使用）。"""
        if self._session is None:
            raise RuntimeError("MCPClient 未连接，请在 async with 块内使用。")
        tool = MCPTool(
            name=name, description="", input_schema={}, session=self._session
        )
        return await tool._async_run(**kwargs)
