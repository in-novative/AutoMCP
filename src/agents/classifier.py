import json
import logging
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# 假设使用 LangChain 的本地模型接口 (如 Ollama 或 vLLM)
# 如果是 HuggingFace 模型，可以使用 HuggingFacePipeline
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.server.models import TaskStep
from src.workflow.state import AgentState

# 设置日志
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
    
    Args:
        query: 任务描述
        top_k: 返回的工具数量
        
    Returns:
        Dict: {"local": ["tool_a: description...", ...], "web": ["tool_b: description...", ...]}
    """
    # TODO: 连接 VectorDB (Chroma/Qdrant) 实现语义搜索
    # 这里仅做 Mock 返回
    return {
        "local": ["read_file: Reads a file from local disk", "write_file: Writes content to a file"],
        "web": ["google_search: Searches the web"]
    }

# --- 3. 微调模型 Prompt 模板 ---
# 因为模型已经微调过，Prompt 可以非常精简
# 微调模型通常对特定的 Prompt 格式（如 Alpaca, ChatML）敏感，这里假设使用 ChatML 格式
FINETUNED_SYSTEM_PROMPT = """You are AutoMCP Classifier. Classify the task into: local_mcp, web_mcp, code_to_mcp, pure_llm.
Output JSON only."""

async def classifier_node(state: AgentState):
    """
    LangGraph 节点：使用微调后的小模型进行任务分类
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
    # 格式应严格匹配微调时的训练数据格式
    prompt = ChatPromptTemplate.from_messages([
        ("system", FINETUNED_SYSTEM_PROMPT),
        ("user", "Task: {task}\nRequirements: {requirements}\n\n[Local Tools]\n{local_tools}\n\n[Web Tools]\n{web_tools}")
    ])
    
    # 3. 加载微调模型
    # 假设模型部署在本地端口 8000 (vLLM/Ollama) 或者使用专门的 Model ID
    # model_name 对应你微调后的模型名称，如 "automcp-classifier-v1"
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1", # 指向本地推理服务
        api_key="sk-local",                  # 本地服务通常不需要真实 Key
        model="automcp-classifier-v1",       # 微调模型 ID
        temperature=0.0,                     # 分类任务必须确定性
        max_tokens=128                       # 输出很短，限制 token 提升速度
    )
    
    # 4. 构建链
    # 使用 JsonOutputParser 确保解析稳健性
    parser = JsonOutputParser(pydantic_object=ClassifierOutput)
    chain = prompt | llm | parser
    
    try:
        # 5. 执行推理
        result = await chain.ainvoke({
            "task": current_step.description,
            "requirements": ", ".join(current_step.requirements),
            "local_tools": local_tools_str,
            "web_tools": web_tools_str
        })
        
        # 验证并转换结果
        output = ClassifierOutput(**result)
        
        # 6. 更新状态
        # 注意：这里直接修改了对象引用，LangGraph 通常需要返回新的状态补丁
        # 在实际运行时，可能需要深拷贝或使用 state 更新机制
        current_step.task_type = output.category
        current_step.tool_name = output.suggested_tool
        
        logger.info(f"Classified step '{current_step.description}' as {output.category}")
        
        return {"plan": plan}

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        # 降级策略：默认回退到 Pure LLM 或抛出异常供上层处理
        current_step.task_type = TaskCategory.PURE_LLM
        current_step.error = f"Classification Error: {str(e)}"
        return {"plan": plan}