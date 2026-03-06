"""
示例：单个 Agent

演示内容：
1. 定义工具（@tool 装饰器）
2. 创建带工具的 Agent
3. 普通文本对话
4. 结构化输出（response_model）
5. 多轮对话 + 短期记忆
6. 长期记忆（跨对话持久化）

运行前确保 .env 已配置：
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...
    OPENAI_DEFAULT_MODEL=...
"""

from pydantic import BaseModel
from shortchain import Agent, tool


# ── 工具定义 ──────────────────────────────────────────────────────────────────


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息。
    :param city: 城市名称，如"北京"、"上海"
    """
    # 实际使用中替换为真实 API 调用
    return f"{city}今天晴，气温 25°C，适合出行。"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。
    :param expression: 合法的数学表达式，如 "100 * 0.85"
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算失败：{e}"


# ── 示例 1：普通文本对话 ──────────────────────────────────────────────────────


def example_basic():
    print("=" * 60)
    print("示例 1：普通文本对话（带工具）")
    print("=" * 60)

    agent = Agent(
        name="assistant",
        system_prompt="你是一个简洁高效的助手，使用中文回答。",
        tools=[get_weather, calculator],
    )

    questions = [
        "北京天气怎么样？",
        "如果机票打 85 折，原价 1200 元，需要多少钱？",
    ]
    for q in questions:
        print(f"\n用户: {q}")
        answer = agent.run(q)
        print(f"助手: {answer}")


# ── 示例 2：结构化输出 ────────────────────────────────────────────────────────


class TravelPlan(BaseModel):
    destination: str
    best_season: str
    estimated_days: int
    highlights: list[str]
    budget_cny: int


def example_structured_output():
    print("\n" + "=" * 60)
    print("示例 2：结构化输出（response_model）")
    print("=" * 60)

    agent = Agent(
        name="travel_planner",
        system_prompt="你是一个专业的旅行规划师，提供详尽的旅行建议。",
        response_model=TravelPlan,
    )

    print("\n用户: 帮我规划一个去成都的旅行")
    plan: TravelPlan = agent.run("帮我规划一个去成都的旅行")
    print(f"目的地: {plan.destination}")
    print(f"最佳季节: {plan.best_season}")
    print(f"建议天数: {plan.estimated_days} 天")
    print(f"必游亮点: {', '.join(plan.highlights)}")
    print(f"预计预算: ¥{plan.budget_cny}")


# ── 示例 3：多轮对话 ──────────────────────────────────────────────────────────


def example_multi_turn():
    print("\n" + "=" * 60)
    print("示例 3：多轮对话（短期记忆）")
    print("=" * 60)

    agent = Agent(
        name="chat_agent",
        system_prompt="你是一个友善的助手，记住对话上下文。",
    )

    turns = [
        "我叫小明，我在学 Python。",
        "我主要在学什么？",  # 测试记忆上下文
        "有什么学习建议？",
    ]
    for msg in turns:
        print(f"\n用户: {msg}")
        reply = agent.run(msg)
        print(f"助手: {reply}")

    print(f"\n当前对话消息数: {len(agent.short_term_memory.get_history())}")


# ── 示例 4：长期记忆 ──────────────────────────────────────────────────────────


def example_long_term_memory():
    print("\n" + "=" * 60)
    print("示例 4：长期记忆（文件持久化）")
    print("=" * 60)

    agent = Agent(
        name="memory_demo",
        system_prompt="你是一个会记住用户信息的助手。",
        enable_long_term_memory=True,
    )

    # 存储用户偏好
    agent.remember("用户名", "小明")
    agent.remember("偏好语言", "Python")
    agent.remember("学习目标", "成为全栈工程师")

    print("已存储的事实：", agent.long_term_memory.all_facts())

    # 保存对话摘要
    agent.save_summary("用户询问了 Python 学习路径，对 web 开发很感兴趣。")

    # 新会话重新加载（模拟程序重启）
    agent2 = Agent(
        name="memory_demo",  # 相同 agent_id 读取同一份文件
        system_prompt="你是一个会记住用户信息的助手。",
        enable_long_term_memory=True,
    )
    print("重新加载后，用户名：", agent2.recall("用户名"))
    print("历史摘要：\n", agent2.long_term_memory.get_summaries_text())


if __name__ == "__main__":
    example_basic()
    example_structured_output()
    example_multi_turn()
    example_long_term_memory()
