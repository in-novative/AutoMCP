from langgraph.graph import StateGraph, END
from .state import AgentState
from src.agents.planner import planner_node
from src.agents.classifier import classifier_node
from src.agents.router import router_node
from src.agents.executor import executor_node       # 假设已实现
from src.agents.reflection import reflection_node

# --- 1. 定义路由逻辑函数 ---

def executor_router(state: AgentState):
    """
    Executor 执行后的路由：
    - 成功 -> 继续下一个步骤 (Loop back to Classifier or End)
    - 失败 -> 进入反思 (Reflection)
    """
    plan = state["plan"]
    idx = state["current_step_index"]
    current_step = plan[idx]
    
    if current_step.status == "completed":
        # 如果是最后一步，结束
        if idx >= len(plan) - 1:
            return "__end__"
        else:
            # 否则，继续处理下一步
            # 注意：需要在 executor_node 中将 index + 1
            return "classifier"
            
    elif current_step.status == "failed":
        return "reflection"
        
    return "__end__" # 默认兜底

def reflection_router(state: AgentState):
    """
    Reflection 反思后的路由：
    - retry_subtask -> 回到 Classifier (因为任务描述变了，可能分类也变了)
    - replan_all -> 回到 Planner
    - fail_final -> 结束
    """
    # 这个 next_action 标记需要在 reflection_node 中设置
    action = state.get("next_action", "retry_subtask")
    
    if action == "replan":
        return "planner"
    elif action == "fail":
        return "__end__"
    else:
        return "classifier"

# --- 2. 构建图 ---

workflow = StateGraph(AgentState)

# 2.1 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("classifier", classifier_node)

# 执行层节点
workflow.add_node("executor", executor_node)          # Local MCP
# workflow.add_node("web_executor", web_executor_node) # Web MCP (占位)
# workflow.add_node("code_generator", code_generator_node) # Code-to-MCP (占位)
# workflow.add_node("llm_responder", llm_responder_node)   # Pure LLM (占位)

# 反思节点
workflow.add_node("reflection", reflection_node)

# 2.2 定义边 (Edges)

# 入口 -> Planner
workflow.set_entry_point("planner")

# Planner -> Classifier (规划完，开始分类第一步)
workflow.add_edge("planner", "classifier")

# Classifier -> [Executor / Web / Code / LLM] (通过 Router 分发)
workflow.add_conditional_edges(
    "classifier",
    router_node,
    {
        "executor": "executor",
        "web_executor": "executor",       # 暂时复用 executor
        "code_generator": "executor",     # 暂时复用 executor
        "llm_responder": "executor",      # 暂时复用 executor
        "__end__": END
    }
)

# [Executors] -> [Next Step / Reflection] (执行结果判断)
# 所有执行节点共享相同的后续逻辑
for node_name in ["executor"]: #, "web_executor", "code_generator", "llm_responder"]:
    workflow.add_conditional_edges(
        node_name,
        executor_router,
        {
            "classifier": "classifier",  # 成功，去下一步 (再次分类)
            "reflection": "reflection",  # 失败，去反思
            "__end__": END
        }
    )

# Reflection -> [Classifier / Planner / End] (反思结果判断)
workflow.add_conditional_edges(
    "reflection",
    reflection_router,
    {
        "classifier": "classifier",  # Level 1 重试：重新分类并执行当前步
        "planner": "planner",        # Level 2 重试：重新规划所有步骤
        "__end__": END               # 彻底失败
    }
)

# 3. 编译
graph = workflow.compile()