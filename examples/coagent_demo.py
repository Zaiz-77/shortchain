"""
示例：CoAgent 多 Agent 团队协作

演示内容：
1. manual 路由 —— 手动指定由哪个 Agent 处理
2. sequential 路由 —— 流水线：研究 → 写作 → 审阅
3. auto 路由 —— 协调 LLM 自动分配任务
4. handoff —— Agent 间点对点消息传递

运行前确保 .env 已配置：
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=...
    OPENAI_DEFAULT_MODEL=...
"""

from shortchain import Agent, CoAgent, tool


# ── 工具 ──────────────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """搜索网络获取信息。:param query: 搜索关键词"""
    # 实际使用中替换为真实搜索 API
    return f"[搜索结果] 关于{query}的最新资料：这是一个虚拟的搜索结果，用于演示。"


@tool
def get_weather(city: str) -> str:
    """查询城市天气。:param city: 城市名"""
    return f"{city}：晴，气温 22°C，适合外出。"


# ── Agent 定义 ────────────────────────────────────────────────────────────────

researcher = Agent(
    name="researcher",
    system_prompt="你是一名研究员，擅长收集信息并整理成结构化的研究报告。请用中文回答。",
    tools=[search_web],
)

writer = Agent(
    name="writer",
    system_prompt="你是一名内容写作专家，将研究资料改写为流畅、易读的文章段落。请用中文回答。",
)

reviewer = Agent(
    name="reviewer",
    system_prompt="你是一名资深编辑，负责审查文章的准确性和可读性，并给出修改建议。请用中文回答。",
)

weather_agent = Agent(
    name="weather_agent",
    system_prompt="你是天气查询专家，只回答天气相关问题。请用中文回答。",
    tools=[get_weather],
)


# ── 示例 1：manual 路由 ───────────────────────────────────────────────────────

def example_manual():
    print("=" * 60)
    print("示例 1：manual 路由")
    print("=" * 60)

    team = CoAgent(
        name="mixed_team",
        agents=[researcher, weather_agent],
        routing="manual",
    )
    print(team)

    result = team.run("北京今天天气怎么样？", agent_name="weather_agent")
    print(f"\n天气结果: {result}")

    result2 = team.run("量子计算的基本原理是什么？", agent_name="researcher")
    print(f"\n研究结果: {result2[:100]}...")

    print(f"\n执行历史条数: {len(team.history)}")


# ── 示例 2：sequential 流水线 ─────────────────────────────────────────────────

def example_sequential():
    print("\n" + "=" * 60)
    print("示例 2：sequential 流水线（研究 → 写作 → 审阅）")
    print("=" * 60)

    pipeline = CoAgent(
        name="content_pipeline",
        agents=[researcher, writer, reviewer],
        routing="sequential",
    )
    print(pipeline)

    topic = "人工智能在医疗领域的应用"
    print(f"\n主题: {topic}")
    final = pipeline.run(topic)

    print("\n流水线各步骤：")
    for step in pipeline.history:
        print(f"\n  [{step['agent']}]")
        print(f"  输出预览: {step['output'][:80]}...")

    print(f"\n最终输出（审阅意见）:\n{final}")


# ── 示例 3：auto 路由 ─────────────────────────────────────────────────────────

def example_auto():
    print("\n" + "=" * 60)
    print("示例 3：auto 路由（LLM 自动分配）")
    print("=" * 60)

    auto_team = CoAgent(
        name="auto_team",
        agents=[researcher, weather_agent],
        routing="auto",
        coordinator_prompt="根据用户问题的类型，选择合适的专家处理。",
    )
    print(auto_team)

    questions = [
        "上海现在天气如何？",
        "大语言模型的核心技术原理是什么？",
    ]
    for q in questions:
        print(f"\n用户: {q}")
        result = auto_team.run(q)
        agent_used = auto_team.history[-1]["agent"]
        print(f"[由 {agent_used} 处理]")
        print(f"回答: {result[:80]}...")


# ── 示例 4：handoff 点对点传递 ────────────────────────────────────────────────

def example_handoff():
    print("\n" + "=" * 60)
    print("示例 4：handoff（researcher → writer）")
    print("=" * 60)

    team = CoAgent(
        name="handoff_demo",
        agents=[researcher, writer],
        routing="manual",
    )

    # 先让 researcher 收集信息
    print("Step 1: researcher 收集信息")
    team.run("区块链技术的主要应用场景有哪些？", agent_name="researcher")
    research_output = team.history[-1]["output"]
    print(f"研究结果预览: {research_output[:80]}...")

    # 将结果 handoff 给 writer 改写成文章
    print("\nStep 2: handoff 给 writer 进行改写")
    article = team.handoff(from_agent="researcher", to_agent="writer")
    print(f"文章预览:\n{article[:200]}...")


if __name__ == "__main__":
    example_manual()
    example_sequential()
    example_auto()
    example_handoff()
