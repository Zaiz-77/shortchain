"""
示例：使用 MCP 工具

演示如何将 MCP Server 的工具挂载到 Agent 上。

MCP（Model Context Protocol）允许通过标准协议连接外部工具服务。
本示例使用 @modelcontextprotocol/server-filesystem（文件系统工具）作为演示。

运行前需要安装 Node.js，然后执行：
    uv run examples/mcp_agent.py
"""

import asyncio
from shortchain import Agent
from shortchain.tools.mcp import MCPClient


async def main():
    # 连接本地文件系统 MCP Server（操作当前目录）
    async with MCPClient.from_stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    ) as client:
        # 获取 MCP Server 提供的所有工具
        mcp_tools = await client.get_tools()
        print("MCP 工具列表：", [t.name for t in mcp_tools])

        # 创建挂载了 MCP 工具的 Agent
        agent = Agent(
            name="fs_agent",
            system_prompt="你是一个文件系统助手，可以读取和管理文件。请用中文回答。",
            tools=mcp_tools,
        )
        print(agent)

        # 运行
        result = agent.run("列出当前目录下的所有文件")
        print("\n结果：")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
