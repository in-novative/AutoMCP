from langgraph.graph import StateGraph, END
from src.workflow.state import AgentState
from src.server.models import TaskCategory

# --- 导入节点 (Nodes) ---
from src.agents.planner import planner_node
from src.agents.classifier import classifier_node
from src.agents.executor import executor_node
from src.agents.web_executor import web_executor_node
from src.agents.code_generator import code_generator_node
from src.agents.llm_responder import llm_responder_node
from src.agents.reflection import reflection_node

# --- 导入路由逻辑 (Edges) ---
# 确保 src/agents/router.py 中已定义这些函数
from src.agents.router import (
    router_node,       # Classifier -> Executors
    executor_router,   # Executors -> Next/Reflection
    reflection_router  # Reflection -> Retry/Replan
)

# --- 构建状态图 ---
workflow = StateGraph(AgentState)

# 1. 注册所有节点
# ------------------------------------------------------------------
workflow.add_node("planner", planner_node)
workflow.add_node("classifier", classifier_node)

# 四种执行路径
workflow.add_node("executor", executor_node)          # Local MCP
workflow.add_node("web_executor", web_executor_node)  # Web MCP
workflow.add_node("code_generator", code_generator_node) # Code-to-MCP
workflow.add_node("llm_responder", llm_responder_node)   # Pure LLM

# 错误处理与反思
workflow.add_node("reflection", reflection_node)


# 2. 定义边与流转逻辑
# ------------------------------------------------------------------

# 2.1 启动阶段
# 入口 -> 规划器
workflow.set_entry_point("planner")

# 规划器 -> 分类器 (开始处理第一个任务)
workflow.add_edge("planner", "classifier")


# 2.2 分发阶段 (Router)
# 根据 Classifier 的标签，分发到具体的执行器
workflow.add_conditional_edges(
    "classifier",
    router_node,
    {
        "executor": "executor",
        "web_executor": "web_executor",
        "code_generator": "code_generator",
        "llm_responder": "llm_responder",
        "__end__": END  # 如果没有任务了
    }
)


# 2.3 执行阶段 (Execution Loop)
# 所有执行器共享相同的后续逻辑：成功则继续，失败则反思
target_executors = ["executor", "web_executor", "code_generator", "llm_responder"]

for node_name in target_executors:
    workflow.add_conditional_edges(
        node_name,
        executor_router,
        {
            "classifier": "classifier",  # 成功: 回到分类器处理下一步
            "reflection": "reflection",  # 失败: 进入反思
            "__end__": END               # 完成: 全部任务结束
        }
    )


# 2.4 反思阶段 (Reflection Loop)
# 根据反思结果决定是微调还是重构
workflow.add_conditional_edges(
    "reflection",
    reflection_router,
    {
        "classifier": "classifier",  # Level 1: 原地重试 (Subtask Retry)
        "planner": "planner",        # Level 2: 全局重规划 (Replan)
        "__end__": END               # Final: 彻底失败
    }
)

# 3. 编译图
# ------------------------------------------------------------------
# 生成可执行的 Runnable 对象
graph = workflow.compile()