from typing import Literal
from src.workflow.state import AgentState
from src.server.models import TaskCategory

# 1. Classifier -> Executors 路由
def router_node(state: AgentState) -> Literal["executor", "web_executor", "code_generator", "llm_responder", "__end__"]:
    """
    根据 Classifier 的分类结果，决定由哪个 Executor 执行
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    
    # 边界检查：如果没有任务或已完成
    if not plan or idx >= len(plan):
        return "__end__"
        
    current_step = plan[idx]
    task_type = current_step.task_type
    
    if task_type == TaskCategory.LOCAL_MCP:
        return "executor"
    elif task_type == TaskCategory.WEB_MCP:
        return "web_executor"
    elif task_type == TaskCategory.CODE_TO_MCP:
        return "code_generator"
    elif task_type == TaskCategory.PURE_LLM:
        return "llm_responder"
    
    # 默认回退
    return "llm_responder"

# 2. Executors -> Next Step / Reflection 路由
def executor_router(state: AgentState) -> Literal["classifier", "reflection", "__end__"]:
    """
    根据执行结果状态，决定是继续下一步还是进入反思
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    
    # 注意：这里的 idx 已经在 executor 中被 +1 了（如果是成功的话）
    # 所以我们需要检查前一个步骤的状态，或者检查是否越界
    
    # 如果索引已经推进到末尾（任务全部完成）
    if idx >= len(plan):
        return "__end__"
        
    # 如果索引已经推进（Executor 成功后 idx+1），且还有未完成的任务
    # 此时 idx 指向的是 *下一步*，我们需要回到 Classifier 对下一步进行分类
    # 判断依据：idx > 0 且前一步是成功的（虽然 executor 已经处理了，但双重保险）
    if idx > 0 and plan[idx-1].status == "completed":
        return "classifier"
        
    # 如果索引没有推进（Executor 失败），说明 idx 指向的还是 *当前失败步*
    current_step = plan[idx]
    if current_step.status == "failed":
        return "reflection"
    
    # 理论上不应该走到这里，除非 executor 既没推进也没报错
    # 默认结束以防死循环
    return "__end__"

# 3. Reflection -> Retry / Replan 路由
def reflection_router(state: AgentState) -> Literal["classifier", "planner", "__end__"]:
    """
    根据反思结果，决定是原地重试还是全局重规划
    """
    action = state.get("next_action")
    
    if action == "retry_subtask":
        return "classifier" # 重新分类并执行
    elif action == "replan":
        return "planner"    # 重新规划
    elif action == "fail_final":
        return "__end__"    # 彻底失败
        
    return "__end__"