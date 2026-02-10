import logging
from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from src.workflow.state import AgentState
from src.server.models import TaskStep, TaskStatus, AgentMessage
# TODO: Import your tool registry or manager here
# from src.tools.registry import get_available_tools 

# 设置日志
logger = logging.getLogger(__name__)

# 定义 Executor 的 System Prompt
EXECUTOR_SYSTEM_PROMPT = """
你是一个全能执行助手 (Executor)。
你的任务是利用手中的工具，精确完成用户指定的子任务。

## 执行原则
1. **专注**: 只完成当前分配的子任务，不要做多余的事。
2. **工具优先**: 尽量使用工具来获取信息或操作环境。
3. **如实报告**: 无论成功与否，都要客观返回执行结果。
4. **角色扮演**: 请扮演以下角色进行执行: {role}

## 当前任务
{description}

## 约束条件
{requirements}
"""

async def get_tools_for_step(step: TaskStep) -> List[Any]:
    """
    根据当前步骤获取可用工具列表 (支持 Local RAG 增强)
    """
    # 1. Local MCP: 使用本地 RAG 检索最相关的本地工具
    # TODO: Implement local RAG retrieval
    # query = step.description
    # relevant_tool_names = await rag_service.search_local_tools(query, top_k=5)
    # return tool_registry.get_tools(relevant_tool_names)
    return []

async def executor_node(state: AgentState):
    """
    LangGraph 节点：执行器
    负责调用工具完成当前子任务
    """
    # 1. 获取上下文
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    
    # 边界检查
    if not plan or idx >= len(plan):
        logger.warning(f"Executor called with invalid index {idx} (plan len: {len(plan)})")
        return {"messages": [AgentMessage(role="system", content="没有待执行的任务或索引越界。")]}
        
    current_step: TaskStep = plan[idx]
    
    # 标记状态为运行中
    current_step.status = TaskStatus.RUNNING
    logger.info(f"Start executing step {idx}: {current_step.description}")
    
    # 2. 准备执行环境
    # 获取相关工具
    tools = await get_tools_for_step(current_step)
    
    # 初始化 LLM (Executor 通常需要较强的推理能力)
    # TODO: Load model config from settings
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # 构造 Prompt
    system_msg = EXECUTOR_SYSTEM_PROMPT.format(
        role=current_step.role,
        description=current_step.description,
        requirements="\n- ".join(current_step.requirements)
    )
    
    # 3. 创建并运行 ReAct Agent
    # 使用 langgraph.prebuilt 快速构建一个 ReAct 循环
    # 如果没有工具，退化为普通对话
    if not tools:
        logger.warning(f"No tools available for step {idx}, falling back to pure LLM.")
    
    agent_executor = create_react_agent(llm, tools, state_modifier=system_msg)
    
    try:
        # 执行推理
        # 输入仅包含当前任务描述，避免上下文过长干扰
        # TODO: Consider adding relevant context from previous steps if needed
        result = await agent_executor.ainvoke({
            "messages": [HumanMessage(content=f"请执行此任务: {current_step.description}")]
        })
        
        # 4. 处理结果
        # 获取 Agent 最后一条回复作为结果
        last_message = result["messages"][-1]
        execution_output = last_message.content
        
        # 更新步骤状态
        current_step.result = str(execution_output)
        current_step.status = TaskStatus.COMPLETED
        current_step.error = None # 清除之前的错误
        
        logger.info(f"Step {idx} completed successfully.")
        
        # 5. 返回更新 (推进指针)
        return {
            "current_step_index": idx + 1,
            # 可选：将执行结果摘要写入主对话流
            "messages": [
                AgentMessage(role="assistant", content=f"步骤 {idx+1} 执行完成: {execution_output[:100]}...")
            ]
        }
        
    except Exception as e:
        # 6. 错误处理
        error_msg = str(e)
        logger.error(f"Step {idx} failed: {error_msg}")
        
        # 更新失败状态
        current_step.status = TaskStatus.FAILED
        current_step.error = error_msg
        
        # 注意：这里不推进指针 (index 不变)，
        # Graph 会根据 status="failed" 路由到 Reflection 节点
        return {
            "messages": [
                AgentMessage(role="system", content=f"步骤 {idx+1} 执行出错: {error_msg}")
            ]
        }