import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.state import AgentState
from src.server.models import TaskStep, TaskStatus, AgentMessage, TaskCategory

logger = logging.getLogger(__name__)

# --- Prompt 定义 ---
RESPONDER_PROMPT = """
你是一个智能助手。
请根据用户的任务要求和之前的执行上下文，生成一个清晰、准确的文本回复。

当前任务: {description}
要求: {requirements}

## 上下文信息
{context}

请直接回答用户，不要使用任何工具调用格式。
"""

async def llm_responder_node(state: AgentState):
    """
    LangGraph 节点：纯 LLM 响应器
    处理无需工具的任务 (Pure LLM)
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    
    if not plan or idx >= len(plan):
        return {"messages": [AgentMessage(role="system", content="Index out of range")]}
        
    current_step: TaskStep = plan[idx]
    
    # 类型检查
    if current_step.task_type != TaskCategory.PURE_LLM:
        logger.warning(f"LLMResponder received wrong task type: {current_step.task_type}")
        
    current_step.status = TaskStatus.RUNNING
    logger.info(f"Generating LLM response for: {current_step.description}")
    
    # 1. 构建上下文
    # 收集之前所有步骤的执行结果，以便 LLM 能够基于之前的成果进行总结
    context_lines = []
    for i in range(idx):
        prev_step = plan[i]
        if prev_step.status == TaskStatus.COMPLETED:
            context_lines.append(f"- Step {i+1} ({prev_step.description}): {prev_step.result}")
    
    context_str = "\n".join(context_lines) if context_lines else "无前序步骤结果。"
    
    try:
        # 2. 调用 LLM
        # 使用智能模型 (GPT-4o) 以获得最佳回复质量
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        chain = ChatPromptTemplate.from_template(RESPONDER_PROMPT) | llm | StrOutputParser()
        
        response_text = await chain.ainvoke({
            "description": current_step.description,
            "requirements": "\n".join(current_step.requirements),
            "context": context_str
        })
        
        # 3. 更新状态
        current_step.result = response_text
        current_step.status = TaskStatus.COMPLETED
        
        return {
            "current_step_index": idx + 1,
            "messages": [AgentMessage(role="assistant", content=response_text)]
        }
        
    except Exception as e:
        logger.error(f"LLM response generation failed: {e}")
        current_step.status = TaskStatus.FAILED
        current_step.error = str(e)
        
        return {
            "messages": [AgentMessage(role="system", content=f"Response Error: {str(e)}")]
        }