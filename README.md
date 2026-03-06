# ShortChain

轻量级 AI Agent 框架，基于 OpenAI API 构建，无需依赖 LangChain / LangGraph。

## 特性

- **单 Agent**：基于 ReAct 架构，支持工具调用、结构化输出、多轮对话
- **CoAgent**：多 Agent 协作团队，支持 auto / sequential / manual 三种路由
- **记忆系统**：短期记忆（会话窗口）+ 长期记忆（本地 JSON 文件持久化）
- **工具系统**：`@tool` 装饰器自动生成 schema，支持 MCP 协议外部工具
- **结构化输出**：传入 Pydantic 模型，自动将 LLM 输出解析为 Python 对象
- **兼容性**：支持 OpenAI 及所有兼容接口（Qwen、DeepSeek 等）

## 安装

```bash
# 克隆项目
git clone https://github.com/Zaiz-77/shortchain.git
cd shortchain

# 使用 uv 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key 等配置
```

## 快速开始

### 1. 单个 Agent

```python
from shortchain import Agent, tool

@tool
def get_weather(city: str) -> str:
    """查询城市天气。:param city: 城市名"""
    return f"{city}今天晴，25°C"

agent = Agent(
    name="assistant",
    system_prompt="你是一个简洁的助手。",
    tools=[get_weather],
)

result = agent.run("北京天气怎么样？")
print(result)  # "北京今天晴，25°C"
```

### 2. 结构化输出

```python
from pydantic import BaseModel
from shortchain import Agent

class CityInfo(BaseModel):
    city: str
    weather: str
    suggestion: str

agent = Agent(
    name="weather_agent",
    system_prompt="你是天气助手。",
    response_model=CityInfo,
)

info: CityInfo = agent.run("上海天气如何？")
print(info.city, info.suggestion)
```

### 3. 长期记忆

```python
from shortchain import Agent

agent = Agent(
    name="my_agent",
    system_prompt="你是一个有记忆的助手。",
    enable_long_term_memory=True,
)

agent.remember("用户名", "Alice")
agent.save_summary("用户对 Python 很感兴趣")

# 下次启动时，相同 name 的 Agent 自动加载历史记忆
print(agent.recall("用户名"))  # "Alice"
```

### 4. 多 Agent 协作

```python
from shortchain import Agent, CoAgent

researcher = Agent(name="researcher", system_prompt="你是研究员，负责收集信息。")
writer     = Agent(name="writer",     system_prompt="你是写作专家，负责撰写文章。")
reviewer   = Agent(name="reviewer",   system_prompt="你是编辑，负责审阅文章。")

# sequential 流水线
pipeline = CoAgent(
    name="content_pipeline",
    agents=[researcher, writer, reviewer],
    routing="sequential",
)
result = pipeline.run("AI 在医疗领域的应用")

# auto 路由，LLM 自动选择合适的 Agent
auto_team = CoAgent(
    name="auto_team",
    agents=[researcher, writer],
    routing="auto",
)
result = auto_team.run("帮我搜索一下量子计算的最新进展")

# manual 手动路由
result = auto_team.run("检查这段文字", agent_name="writer")

# handoff：将 researcher 的结果转交给 writer
auto_team.run("量子计算概述", agent_name="researcher")
article = auto_team.handoff(from_agent="researcher", to_agent="writer")
```

### 5. MCP 工具

```python
import asyncio
from shortchain import Agent
from shortchain.tools.mcp import MCPClient

async def main():
    async with MCPClient.from_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as client:
        tools = await client.get_tools()
        agent = Agent(name="fs_agent", system_prompt="你是文件助手。", tools=tools)
        print(agent.run("列出当前目录的文件"))

asyncio.run(main())
```

## 环境配置

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_DEFAULT_MODEL=gpt-4o

# 长期记忆存储目录（默认 .shortchain_memory/）
SHORTCHAIN_MEMORY_DIR=.shortchain_memory
```

## 项目结构

```
shortchain/
├── shortchain/
│   ├── __init__.py          # 公开 API，懒加载
│   ├── config.py            # 环境变量读取
│   ├── core/
│   │   ├── message.py       # 消息数据结构（Message / Role / ToolCall）
│   │   ├── agent.py         # 单 Agent（ReAct 架构）
│   │   ├── runner.py        # ReAct 执行循环
│   │   └── coagent.py       # 多 Agent 协作
│   ├── tools/
│   │   ├── base.py          # @tool 装饰器 / Tool 基类
│   │   └── mcp.py           # MCP 工具适配器
│   ├── memory/
│   │   ├── short_term.py    # 短期记忆（会话窗口）
│   │   └── long_term.py     # 长期记忆（本地 JSON）
│   └── skills/
│       └── base.py          # Skill / SkillManager
├── examples/
│   ├── single_agent.py      # 单 Agent 示例
│   ├── coagent_demo.py      # 多 Agent 协作示例
│   └── mcp_agent.py         # MCP 工具示例
├── .env.example
└── pyproject.toml
```

## Agent 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | Agent 唯一名称，也用作长期记忆 ID |
| `system_prompt` | `str` | 基础系统提示词 |
| `model` | `str` | 模型名称，默认读 `OPENAI_DEFAULT_MODEL` |
| `tools` | `list[Tool]` | 工具列表 |
| `skills` | `list[Skill]` | Skill 列表（附带指令片段 + 工具） |
| `response_model` | `type[BaseModel]` | 结构化输出目标类型 |
| `max_iterations` | `int` | ReAct 最大迭代次数，默认 10 |
| `max_messages` | `int` | 短期记忆窗口大小，默认 50 |
| `enable_long_term_memory` | `bool` | 是否启用长期记忆，默认 False |
| `tool_calling` | `bool` | 是否启用 function calling，默认 True |

## CoAgent 路由模式

| 模式 | 说明 |
|------|------|
| `auto` | 协调 LLM 读取各 Agent 的 system_prompt，自动选择最合适的执行 |
| `sequential` | 按注册顺序串行执行，前一个输出作为下一个的输入 |
| `manual` | `run(input, agent_name="xxx")` 手动指定执行的 Agent |

## 依赖

- `openai >= 1.0`
- `python-dotenv >= 1.0`
- `pydantic >= 2.0`
- `mcp >= 1.0`（MCP 功能可选）
