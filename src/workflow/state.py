from typing import TypedDict, Annotated, List
import operator
from src.server.models import AgentMessage, TaskStep

# 1. 定义合并策略 (Reducer)
def add_messages(old_message: List[AgentMessage], new_messages: List[AgentMessage]):
    return old_message + new_messages

class AgentState(TypedDict):
    """
    工作流状态容器
    在 LangGraph 的各个 Node 之间传递
    """
    # --- 基础上下文 ---
    # 聊天记录：使用 Annotated + add_messages 实现增量更新
    messages: Annotated[List[AgentMessage], add_messages]
    
    # --- 任务规划 ---
    # 完整的执行计划 (由 Planner 生成)
    plan: List[TaskStep]
    
    # 当前执行进度的指针 (由 Executor/Router 更新)
    current_step_index: int
    
    # --- 反思控制 ---
    # 当前任务/子任务的反思重试计数器 (由 Reflection 节点管理)
    reflection_count: int