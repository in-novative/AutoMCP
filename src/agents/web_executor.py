import logging
import asyncio
from typing import List, Any, Dict
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool

from src.workflow.state import AgentState
from src.server.models import TaskStep, TaskStatus, AgentMessage, TaskCategory
from config.settings import settings

logger = logging.getLogger(__name__)

# --- 1. 外部 MCP Server 注册表 (模拟 RAG 数据库) ---
# 实际项目中应存储在 ChromaDB
KNOWN_MCP_SERVERS = [
    {
        "name": "weather-server",
        "url": "http://localhost:8001/sse",
        "description": "Provides weather information for any location."
    },
    {
        "name": "github-server",
        "url": "http://localhost:8002/sse", 
        "description": "Search repositories, read issues, and manage PRs on GitHub."
    },
    {
        "name": "finance-server",
        "url": "https://mcp.finance.api/sse",
        "description": "Stock prices, currency exchange rates, and financial news."
    }
]

# --- 2. Web RAG 检索 ---
async def search_web_mcp_servers(query: str, top_k: int = 1) -> List[Dict]:
    """
    根据任务描述检索最相关的外部 MCP Server
    """
    # TODO: 使用 VectorDB 进行语义检索
    # 这里使用简单的关键词匹配作为 Mock
    query = query.lower()
    results = []
    for server in KNOWN_MCP_SERVERS:
        if any(kw in server["description"].lower() for kw in query.split()):
            results.append(server)
            
    # 默认返回第一个作为兜底
    return results[:top_k] if results else [KNOWN_MCP_SERVERS[0]]

# --- 3. 动态 MCP 客户端 ---
async def connect_and_execute(server_url: str, task_description: str) -> str:
    """
    连接远程 MCP Server 并执行任务
    """
    async with AsyncExitStack() as stack:
        # 3.1 建立 SSE 连接
        logger.info(f"Connecting to MCP Server: {server_url}")
        # 注意：这里假设远程 Server 支持 SSE 传输
        # sse_client 返回 (read_stream, write_stream)
        read_stream, write_stream = await stack.enter_async_context(
            sse_client(server_url)
        )
        
        # 3.2 初始化 Session
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        # 3.3 列出远程工具
        remote_tools_list = await session.list_tools()
        logger.info(f"Discovered {len(remote_tools_list.tools)} tools on {server_url}")
        
        # 3.4 将 MCP Tool 转换为 LangChain Tool
        langchain_tools = []
        for tool_schema in remote_tools_list.tools:
            
            async def _dynamic_tool_func(**kwargs):
                # 闭包捕获 session 和 tool_name
                result = await session.call_tool(tool_schema.name, arguments=kwargs)
                return result.content[0].text

            langchain_tools.append(Tool(
                name=tool_schema.name,
                description=tool_schema.description,
                func=None, # 同步函数为空
                coroutine=_dynamic_tool_func # 使用异步函数
            ))
            
        if not langchain_tools:
            return "Error: Remote server has no tools."
            
        # 3.5 使用 ReAct Agent 执行
        llm = ChatOpenAI(
            model=settings.DEFAULT_LLM_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL
        )
        agent = create_react_agent(llm, langchain_tools)
        
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Use the available tools to: {task_description}")]
        })
        
        return result["messages"][-1].content

# --- 4. Web Executor Node ---
async def web_executor_node(state: AgentState):
    """
    Web 执行器节点
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    
    if not plan or idx >= len(plan):
        return {"messages": [AgentMessage(role="system", content="Index out of range")]}
        
    current_step: TaskStep = plan[idx]
    
    # 双重检查类型
    if current_step.task_type != TaskCategory.WEB_MCP:
        logger.warning(f"WebExecutor received non-web task: {current_step.task_type}")
        # 可以选择直接返回，交给 Router 重新路由，或者硬着头皮做
    
    current_step.status = TaskStatus.RUNNING
    logger.info(f"WebExecutor processing: {current_step.description}")
    
    try:
        # 1. 检索最合适的 Server
        servers = await search_web_mcp_servers(current_step.description)
        if not servers:
            raise ValueError("No suitable Web MCP Server found.")
            
        target_server = servers[0]
        logger.info(f"Selected server: {target_server['name']}")
        
        # 2. 连接并执行
        # 注意：这里包含网络 I/O，可能会比较慢
        output = await connect_and_execute(target_server["url"], current_step.description)
        
        # 3. 更新状态
        current_step.result = output
        current_step.status = TaskStatus.COMPLETED
        current_step.tool_name = target_server["name"] # 记录实际使用的服务名
        
        return {
            "current_step_index": idx + 1,
            "messages": [AgentMessage(role="assistant", content=f"Web Task Completed via {target_server['name']}")]
        }
        
    except Exception as e:
        logger.error(f"Web execution failed: {e}")
        current_step.status = TaskStatus.FAILED
        current_step.error = str(e)
        
        return {
            "messages": [AgentMessage(role="system", content=f"Web Execution Error: {str(e)}")]
        }