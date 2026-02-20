from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.workflow.state import AgentState
from src.server.models import TaskStatus, AgentMessage
from config.settings import settings  # 假设已添加重试配置
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# 定义反思结果枚举
class ReflectionAction(str, Enum):
    RETRY_SUBTASK = "retry_subtask"  # 原地重试
    REPLAN_ALL = "replan_all"        # 回退给 Planner
    FAIL_FINAL = "fail_final"        # 彻底放弃

# --- Prompt 定义 ---
REFLECTION_PROMPT = """
你是一个代码审计与错误分析专家。
当前子任务执行失败。请分析错误日志，并给出修复建议。

子任务: {description}
错误信息: {error}
历史尝试次数: {retry_count}

请简要分析原因，并优化该子任务的 'description' 和 'requirements' 以便下次成功。
"""

async def reflection_node(state: AgentState):
    """
    反思节点：处理执行失败的情况
    """
    plan = state["plan"]
    idx = state["current_step_index"]
    current_step = plan[idx]
    
    # 获取全局配置的阈值 (也可以从 state 中获取动态配置)
    max_sub_retries = state.get("max_subtask_retries", settings.MAX_SUBTASK_RETRIES)
    max_plan_retries = state.get("plan_retry_count", settings.MAX_PLAN_RETRIES)
    
    # --- Level 1: 子任务级反思 ---
    if current_step.retry_count < max_sub_retries:
        # 1. 调用 LLM 分析错误
        llm = ChatOpenAI(
            model=settings.DEFAULT_LLM_MODEL,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL
        )
        prompt = ChatPromptTemplate.from_template(REFLECTION_PROMPT)
        chain = prompt | llm
        
        analysis = await chain.ainvoke({
            "description": current_step.description,
            "error": current_step.error,
            "retry_count": current_step.retry_count
        })
        
        # 2. 更新子任务
        # 追加错误分析到 requirements，或者修改 description
        current_step.requirements.append(f"Previous Error: {current_step.error}")
        current_step.requirements.append(f"Fix Hint: {analysis.content}")
        
        # 3. 增加计数并重置状态
        current_step.retry_count += 1
        current_step.status = TaskStatus.PENDING # 重置为 pending 以便再次被 Executor 拾取

        return {
            "plan": plan,
            "messages": [AgentMessage(role="system", content=f"子任务失败，正在第 {current_step.retry_count} 次重试...")],
            "next_action": "retry_subtask"  # 关键：通知路由进行重试
        }
    
    # --- Level 2: 任务级反思 ---
    # 如果子任务重试次数耗尽
    elif max_plan_retries < settings.MAX_PLAN_RETRIES:
        # 1. 汇总错误上下文
        error_summary = f"Step {idx} failed after {max_sub_retries} retries. Last error: {current_step.error}"
        
        # 2. 增加任务级计数
        new_plan_retry_count = max_plan_retries + 1
        
        # 3. 触发 Re-plan
        # 我们返回一个特殊的标记或消息，让 Graph 跳转回 Planner
        return {
            "plan_retry_count": new_plan_retry_count,
            "messages": [
                AgentMessage(role="user", content=f"计划执行受阻：{error_summary}。请根据此信息重新规划剩余任务。")
            ],
            # 这里的关键是：Graph 应该配置一条从 Reflection 到 Planner 的条件边
            "next_action": "replan" 
        }
        
    # --- Final: 彻底失败 ---
    else:
        return {
            "messages": [AgentMessage(role="system", content="所有重试手段已耗尽，任务执行失败。")],
            "next_action": "fail"
        }