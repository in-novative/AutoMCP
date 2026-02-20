import json
import logging
import asyncio
import re
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.server.models import TaskStep
from src.workflow.state import AgentState
from src.memory.rag_local import LocalToolRAG
from src.tools.web_mcp_service import web_mcp_service
from config.settings import settings
import ollama

logger = logging.getLogger(__name__)

# 全局 RAG 实例（单例）
_tool_rag = LocalToolRAG()

# --- 1. 定义数据结构 ---
class TaskCategory(str, Enum):
    LOCAL_MCP = "local_mcp"
    WEB_MCP = "web_mcp"
    CODE_TO_MCP = "code_to_mcp"
    PURE_LLM = "pure_llm"

class ClassifierOutput(BaseModel):
    category: TaskCategory
    suggested_tool: Optional[str] = Field(None, description="The name of the tool if category is local_mcp or web_mcp")

# --- 2. RAG 接口 ---
async def retrieve_tools(query: str, top_k: int = 5) -> Dict[str, List[str]]:
    """
    RAG 检索接口：根据任务描述检索最相关的本地工具和网络 MCP Server。
    
    Args:
        query: 任务描述查询
        top_k: 返回的工具数量
        
    Returns:
        包含工具名称和描述的字典
    """
    result = {"local": [], "web": []}
    
    try:
        # 1. 使用 Local RAG 检索本地工具
        local_tool_names = await _tool_rag.search(query, top_k=top_k)
        
        if local_tool_names:
            # 构造工具描述
            for name in local_tool_names:
                result["local"].append(f"{name}: Local tool for {name.replace('_', ' ')}")
            logger.info(f"Local RAG retrieved {len(local_tool_names)} tools")
        
        # 2. 使用 Web MCP Service 检索网络 MCP Server
        web_servers = await web_mcp_service.search_servers(query, top_k=top_k)
        
        if web_servers:
            # 格式化网络 MCP Server
            result["web"] = web_mcp_service.format_for_classifier(web_servers)
            logger.info(f"Web MCP Service retrieved {len(web_servers)} servers")
        
        return result
        
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        # 降级：返回已获取的部分结果
        return result

# --- 3. Prompt 模板 ---
FINETUNED_SYSTEM_PROMPT_TEMPLATE = """你是 AutoMCP 任务分类器。请将用户任务分类为以下四种类型之一：local_mcp、web_mcp、code_to_mcp、pure_llm。

## 可用 MCP Server 列表

### 本地 MCP Server（已部署在本地）
{local_tools}

### 网络 MCP Server（可通过网络访问的第三方服务）
{web_tools}

## 分类标准（按优先级判断）

### 1. local_mcp - 本地 MCP Server 可完成
**定义**：任务可以由本地已部署的 MCP Server 直接完成
**判断标准**：
- 本地 MCP Server 列表中有工具可以完成此任务
- 任务涉及本地文件系统、本地命令、本地资源访问
**典型任务**：
- 文件操作：创建、读取、写入、删除本地文件或目录
- 系统操作：执行本地 shell 命令、查看本地系统信息
- 本地数据处理：处理本地文件、本地数据库操作
**特征**：使用 local_tools 列表中的工具即可直接完成

### 2. web_mcp - 网络 MCP Server 可完成
**定义**：任务可以由网络上已有的 MCP Server 完成
**判断标准**：
- 网络 MCP Server 列表中有工具可以完成此任务
- 任务需要调用在线 API、获取实时网络信息
**典型任务**：
- 实时信息查询：天气、新闻、股票、汇率
- 网络服务：发送邮件、社交媒体 API、云存储服务
- 在线搜索：Google、百度、必应搜索
**特征**：使用 web_tools 列表中的工具即可直接完成

### 3. code_to_mcp - 需要转换为 MCP Server
**定义**：任务需要先将 GitHub 仓库或其他代码转换为 MCP Server 后才能完成
**判断标准**：
- 本地和网络 MCP Server 都无法直接完成
- 但可以通过克隆 GitHub 仓库并转换为 MCP Server 来实现
- 任务涉及特定的第三方服务、SDK 或 API 封装
**典型任务**：
- 使用特定第三方服务：Slack、Discord、Notion、GitHub API 等
- 特定领域工具：图像处理、机器学习模型、特定数据库
- 复杂业务逻辑：需要封装现有代码库为 MCP Server
**特征**：需要 "code → MCP Server" 转换流程

### 4. pure_llm - LLM 可直接完成
**定义**：任务可以直接由 LLM 完成，无需任何 MCP Server
**判断标准**：
- 不需要调用任何外部工具或服务
- 可以是知识问答、文本生成、简单计算
- 可以包括网络搜索（但只是获取信息，不涉及 MCP Server 调用）
**典型任务**：
- 知识问答：概念解释、技术问题、历史事实
- 文本生成：写作、翻译、总结、改写
- 创意任务：头脑风暴、建议、分析
- 简单计算：数学运算、逻辑推理
**特征**：LLM 仅凭自身能力即可完成

## 分类决策流程

1. **检查 local_tools**：任务是否可由本地 MCP Server 完成？
   → 是 → **local_mcp**
   
2. **检查 web_tools**：任务是否可由网络 MCP Server 完成？
   → 是 → **web_mcp**
   
3. **判断是否可转换**：是否需要通过 GitHub 仓库转换为 MCP Server？
   → 是 → **code_to_mcp**
   
4. **默认**：以上都不是，LLM 可直接完成
   → **pure_llm**

## 输出示例

任务："在当前目录创建 test.txt 文件，写入 'Hello'"
分析：本地 MCP Server 有 write_file 工具可以完成
输出：{{"category": "local_mcp", "suggested_tool": "write_file"}}

任务："今天北京天气怎么样？"
分析：网络 MCP Server 有天气查询工具
输出：{{"category": "web_mcp", "suggested_tool": "weather_api"}}

任务："用 Slack 发送消息到 #general 频道"
分析：本地和网络 MCP Server 都没有 Slack 工具，需要转换
输出：{{"category": "code_to_mcp", "suggested_tool": null}}

任务："什么是机器学习？"
分析：LLM 可以直接回答，无需工具
输出：{{"category": "pure_llm", "suggested_tool": null}}

任务："搜索最新的 AI 新闻"
分析：网络 MCP Server 有搜索工具
输出：{{"category": "web_mcp", "suggested_tool": "web_search"}}

任务："使用 Notion API 创建页面"
分析：需要封装 Notion SDK 为 MCP Server
输出：{{"category": "code_to_mcp", "suggested_tool": null}}

## 输出格式要求

必须严格输出 JSON 格式，不要包含任何其他文字：
{{"category": "类型", "suggested_tool": "工具名或null"}}

- category：必须是 local_mcp、web_mcp、code_to_mcp、pure_llm 之一
- suggested_tool：如果任务可由现有 MCP Server 完成，填写具体工具名；否则填 null
"""

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
    local_tools_str = "\n".join(tools_context["local"]) if tools_context["local"] else "无可用本地工具"
    web_tools_str = "\n".join(tools_context["web"]) if tools_context["web"] else "无可用网络工具"

    # 2. 构造 System Prompt（包含 RAG 检索到的工具信息）
    system_prompt = FINETUNED_SYSTEM_PROMPT_TEMPLATE.format(
        local_tools=local_tools_str,
        web_tools=web_tools_str
    )

    # 3. 构造 User Prompt（仅包含任务信息）
    user_prompt = f"""Task: {current_step.description}
Requirements: {", ".join(current_step.requirements)}

请分类此任务类型（local_mcp, web_mcp, code_to_mcp, pure_llm），以 JSON 格式输出：
{{"category": "...", "suggested_tool": "..."}}"""

    logger.info(f"Classifier using Ollama model: {settings.CLASSIFIER_MODEL}")

    try:
        # 4. 使用 Ollama 原生客户端
        def _call_ollama():
            return ollama.chat(
                model=settings.CLASSIFIER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
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