import json
import logging
import asyncio
import re
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.server.models import TaskStep
from src.workflow.state import AgentState
from config.settings import settings
import ollama

logger = logging.getLogger(__name__)

# --- 1. 定义数据结构 ---
class TaskCategory(str, Enum):
    LOCAL_MCP = "local_mcp"
    WEB_MCP = "web_mcp"
    CODE_TO_MCP = "code_to_mcp"
    PURE_LLM = "pure_llm"

class ClassifierOutput(BaseModel):
    category: TaskCategory
    suggested_tool: Optional[str] = Field(None, description="The name of the tool if category is local_mcp or web_mcp")

# --- 2. RAG 接口 (预留) ---
async def retrieve_tools(query: str, top_k: int = 5) -> Dict[str, List[str]]:
    """
    RAG 检索接口：根据任务描述检索最相关的工具。
    """
    # TODO: 连接 VectorDB (Chroma/Qdrant) 实现语义搜索
    # 这里仅做 Mock 返回
    return {
        "local": ["read_file: Reads a file from local disk", "write_file: Writes content to a file"],
        "web": ["google_search: Searches the web"]
    }

# --- 3. Prompt 模板 ---
FINETUNED_SYSTEM_PROMPT = """You are AutoMCP Classifier. Classify the task into: local_mcp, web_mcp, code_to_mcp, pure_llm.
Output JSON only."""

# --- 4. Classifier 节点 ---
async def classifier_node(state: AgentState):
    """
    LangGraph 节点：使用本地 Ollama 模型进行任务分类
    """
    # 获取当前需要分类的步骤
    current_step_index = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    
    if current_step_index >= len(plan):
        logger.warning("Current step index out of range")
        return {"plan": plan}
        
    current_step: TaskStep = plan[current_step_index]
    
    # 1. RAG 检索上下文
    tools_context = await retrieve_tools(current_step.description)
    local_tools_str = "\n".join(tools_context["local"])
    web_tools_str = "\n".join(tools_context["web"])
    
    # 2. 构造 Prompt
    prompt_text = f"""Task: {current_step.description}
Requirements: {", ".join(current_step.requirements)}

[Local Tools]
{local_tools_str}

[Web Tools]
{web_tools_str}

请分类此任务类型（local_mcp, web_mcp, code_to_mcp, pure_llm），以 JSON 格式输出：
{{"category": "...", "suggested_tool": "..."}}"""

    logger.info(f"Classifier using Ollama model: {settings.CLASSIFIER_MODEL}")
    
    try:
        # 3. 使用 Ollama 原生客户端
        def _call_ollama():
            return ollama.chat(
                model=settings.CLASSIFIER_MODEL,
                messages=[
                    {"role": "system", "content": FINETUNED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text}
                ],
                options={"temperature": 0}
            )
        
        # 在线程池中执行同步的 ollama 调用
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _call_ollama)
        
        # 4. 解析结果
        content = response['message']['content']
        logger.debug(f"Ollama response: {content}")
        
        # 尝试解析 JSON
        try:
            result = json.loads(content)
            output = ClassifierOutput(**result)
        except json.JSONDecodeError:
            # 如果返回的不是纯 JSON，尝试提取 JSON 部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                output = ClassifierOutput(**result)
            else:
                raise ValueError(f"Cannot parse response: {content}")
        
        # 5. 更新状态
        current_step.task_type = output.category
        current_step.tool_name = output.suggested_tool
        
        logger.info(f"Classified step '{current_step.description}' as {output.category}")
        
        return {"plan": plan}

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        # 降级策略：默认回退到 Pure LLM
        current_step.task_type = TaskCategory.PURE_LLM
        current_step.error = f"Classification Error: {str(e)}"
        return {"plan": plan}