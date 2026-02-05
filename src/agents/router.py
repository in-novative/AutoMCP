from typing import Literal
from src.workflow.state import AgentState

# 定义所有可能的下游节点名称
# 这些名称必须与 graph.py 中注册的节点名称一致
TargetNode = Literal["executor", "web_executor", "code_generator", "llm_responder", "__end__"]

def router_node(state: AgentState) -> TargetNode:
    """
    LangGraph 的路由逻辑函数
    用于在 graph.add_conditional_edges 中使用
    """
    
    # 1. 边界检查：如果没有计划或已经执行完所有步骤
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    
    if not plan or idx >= len(plan):
        # 任务全部完成，结束工作流
        return "__end__"
    
    # 2. 获取当前任务的分类结果
    current_step = plan[idx]
    task_type = current_step.task_type
    
    # 3. 路由分发
    if task_type == "local_mcp":
        return "executor"         # 本地工具执行器
        
    elif task_type == "web_mcp":
        return "web_executor"     # 外部/Web 工具执行器
        
    elif task_type == "code_to_mcp":
        return "code_generator"   # 代码生成引擎
        
    elif task_type == "pure_llm":
        return "llm_responder"    # 纯 LLM 对话节点
        
    else:
        # 异常情况：未分类或未知类型
        # 策略：默认转给 LLM 处理，或者抛出异常
        return "llm_responder"