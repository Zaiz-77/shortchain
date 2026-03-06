from shortchain.tools.base import tool, Tool, FunctionTool

__all__ = ["tool", "Tool", "FunctionTool", "MCPClient", "MCPTool"]


# MCPClient / MCPTool 按需导入，避免未安装 mcp 包时整包报错
def __getattr__(name: str) -> object:
    if name in ("MCPClient", "MCPTool"):
        try:
            from shortchain.tools.mcp import MCPClient, MCPTool  # noqa: F401
            import shortchain.tools.mcp as _mcp_mod

            return getattr(_mcp_mod, name)
        except ImportError as e:
            raise ImportError(
                f"使用 MCP 功能需要安装 mcp 包：uv add mcp\n原始错误：{e}"
            ) from e
    raise AttributeError(f"module 'shortchain.tools' has no attribute {name!r}")
