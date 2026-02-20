import logging
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.workflow.state import AgentState
from src.server.models import TaskStep, TaskStatus, AgentMessage, TaskCategory
from src.integrations.code2mcp_client import Code2MCPClient, Code2MCPResult, MCPServiceInstance
from src.integrations.mcp_caller import TaskParser, SimpleMCPCaller, MCPToolResult
from src.tools.web.repo_search import GitHubRepoSearcher
from config.settings import settings

logger = logging.getLogger(__name__)


SEARCH_QUERY_PROMPT = """
你是一个 GitHub 搜索专家。请根据用户的任务需求，生成 3-5 个适合的 GitHub 搜索查询词。

用户任务: {task_description}

请生成简洁、有效的搜索词，优先使用常见的关键词组合。
例如：
- "python text processing"
- "python nlp library"
- "python file utilities"

以 JSON 格式返回：
{{
  "search_queries": [
    "查询词 1",
    "查询词 2",
    "查询词 3"
  ]
}}
"""


REPO_SELECTION_PROMPT = """
你是一个仓库选择专家。请根据用户的任务需求，从搜索到的仓库中选择最适合的一个。

用户任务: {task_description}

可用仓库:
{repos_list}

请选择最适合完成该任务的仓库，并以 JSON 格式返回：
{{
  "selected": true/false,
  "repo_url": "仓库的 GitHub URL",
  "reason": "选择理由"
}}
"""


TOOL_SELECTION_PROMPT = """
你是一个工具选择专家。请根据用户任务，从可用工具中选择最合适的一个，并确定调用参数。

用户任务: {task_description}

可用工具:
{tools_list}

请以 JSON 格式返回：
{{
  "tool_name": "工具名称",
  "arguments": {{
    "param1": "value1",
    "param2": "value2"
  }},
  "reason": "选择理由"
}}

如果没有明确的工具可以调用，可以返回：
{{
  "tool_name": null,
  "arguments": {{}},
  "reason": "不需要调用特定工具，直接展示转换结果"
}}
"""


async def _generate_search_queries(task_description: str, llm) -> List[str]:
    """使用 LLM 生成搜索查询词"""
    try:
        prompt = ChatPromptTemplate.from_template(SEARCH_QUERY_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        response = await chain.ainvoke({"task_description": task_description})
        
        try:
            result = json.loads(response)
            queries = result.get("search_queries", [])
            if queries:
                return queries
        except json.JSONDecodeError:
            pass
        
    except Exception as e:
        logger.warning(f"Failed to generate search queries: {e}")
    
    return []


async def _search_with_fallback(task_description: str, searcher, llm) -> List:
    """使用多种策略搜索仓库"""
    
    all_repos = []
    seen_urls = set()
    
    # 策略 1: 使用 LLM 生成的查询词
    logger.info("Strategy 1: Using LLM-generated search queries")
    search_queries = await _generate_search_queries(task_description, llm)
    
    for query in search_queries[:3]:
        try:
            logger.info(f"Searching with: {query}")
            repos = await searcher.search_repos(query, limit=3)
            for repo in repos:
                if repo.url not in seen_urls:
                    all_repos.append(repo)
                    seen_urls.add(repo.url)
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
    
    if all_repos:
        logger.info(f"Found {len(all_repos)} repos from LLM queries")
        return all_repos[:5]
    
    # 策略 2: 使用简化的关键词搜索
    logger.info("Strategy 2: Using simplified keyword search")
    keywords = ["python", "library", "tool", "util"]
    
    for keyword in keywords:
        try:
            repos = await searcher.search_repos(keyword, limit=3)
            for repo in repos:
                if repo.url not in seen_urls:
                    all_repos.append(repo)
                    seen_urls.add(repo.url)
        except Exception as e:
            logger.warning(f"Search failed for keyword '{keyword}': {e}")
    
    if all_repos:
        logger.info(f"Found {len(all_repos)} repos from keyword search")
        return all_repos[:5]
    
    # 策略 3: 使用一些流行的 Python 库作为后备
    logger.info("Strategy 3: Using fallback popular repos")
    fallback_repos = [
        "https://github.com/python/cpython",
        "https://github.com/numpy/numpy",
        "https://github.com/pandas-dev/pandas",
    ]
    
    return all_repos


async def code2mcp_executor_node(state: AgentState):
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    
    if not plan or idx >= len(plan):
        return {"messages": [AgentMessage(role="system", content="Index out of range")]}
    
    current_step: TaskStep = plan[idx]
    
    if current_step.task_type != TaskCategory.CODE_TO_MCP:
        logger.warning(f"Code2MCP Executor received wrong task type: {current_step.task_type}")
    
    current_step.status = TaskStatus.RUNNING
    logger.info(f"Starting Code2MCP execution for: {current_step.description}")
    
    try:
        github_token = settings.GITHUB_TOKEN.get_secret_value() if settings.GITHUB_TOKEN else None
        searcher = GitHubRepoSearcher(github_token)
        client = Code2MCPClient()
        
        llm = ChatOpenAI(
            model=settings.DEFAULT_LLM_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL
        )
        
        # 使用多种策略搜索仓库
        logger.info(f"Searching for repos for task: {current_step.description}")
        repos = await _search_with_fallback(current_step.description, searcher, llm)
        
        if not repos:
            error_msg = "No suitable GitHub repositories found for this task"
            current_step.status = TaskStatus.FAILED
            current_step.error = error_msg
            return {
                "messages": [AgentMessage(role="system", content=error_msg)]
            }
        
        repos_list = searcher.format_for_classifier(repos)
        logger.info(f"Found {len(repos)} potential repos")
        
        llm = ChatOpenAI(
            model=settings.DEFAULT_LLM_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL
        )
        
        selection_prompt = ChatPromptTemplate.from_template(REPO_SELECTION_PROMPT)
        chain = selection_prompt | llm | StrOutputParser()
        
        selection_response = await chain.ainvoke({
            "task_description": current_step.description,
            "repos_list": repos_list
        })
        
        try:
            selection = json.loads(selection_response)
            if not selection.get("selected", False):
                error_msg = "No suitable repository selected"
                current_step.status = TaskStatus.FAILED
                current_step.error = error_msg
                return {
                    "messages": [AgentMessage(role="system", content=error_msg)]
                }
            repo_url = selection["repo_url"]
            selection_reason = selection.get("reason", "")
        except json.JSONDecodeError:
            repo_url = repos[0].url
            selection_reason = "Default selection (first result)"
        
        logger.info(f"Selected repo: {repo_url} - {selection_reason}")
        
        conversion_result = await client.convert_repo(repo_url)
        
        if not conversion_result.success:
            error_msg = f"Code2MCP conversion failed: {conversion_result.error}"
            current_step.status = TaskStatus.FAILED
            current_step.error = error_msg
            return {
                "messages": [AgentMessage(role="system", content=error_msg)]
            }
        
        available_tools = client.get_available_tools(conversion_result.analysis)
        tools_summary = ", ".join([t["name"] for t in available_tools[:10]]) if available_tools else "No tools found"
        
        tool_call_result = None
        tool_executed = False
        
        if available_tools and conversion_result.mcp_output_dir:
            tools_list = "\n".join([f"- {t['name']}: {t['description']}" for t in available_tools])
            
            tool_selection_prompt = ChatPromptTemplate.from_template(TOOL_SELECTION_PROMPT)
            tool_chain = tool_selection_prompt | llm | StrOutputParser()
            
            tool_response = await tool_chain.ainvoke({
                "task_description": current_step.description,
                "tools_list": tools_list
            })
            
            try:
                tool_selection = json.loads(tool_response)
                tool_name = tool_selection.get("tool_name")
                
                if tool_name:
                    arguments = tool_selection.get("arguments", {})
                    logger.info(f"Calling tool: {tool_name} with args: {arguments}")
                    
                    tool_call_result = await SimpleMCPCaller.call_tool_direct(
                        conversion_result.mcp_output_dir,
                        tool_name,
                        arguments
                    )
                    tool_executed = True
                    
            except json.JSONDecodeError:
                logger.warning("Tool selection parsing failed, skipping tool execution")
        
        result_summary = []
        result_summary.append(f"Successfully converted repository to MCP service!")
        result_summary.append(f"Repo: {repo_url}")
        result_summary.append(f"Service Name: {conversion_result.service_name}")
        result_summary.append(f"Available Tools: {tools_summary}")
        
        if tool_executed and tool_call_result:
            if tool_call_result.success:
                result_summary.append(f"\nTool Execution Result:")
                result_summary.append(f"Success: {tool_call_result.result}")
            else:
                result_summary.append(f"\nTool Execution Failed:")
                result_summary.append(f"Error: {tool_call_result.error}")
        
        result_text = "\n".join(result_summary)
        
        current_step.result = result_text
        current_step.status = TaskStatus.COMPLETED
        current_step.tool_args = json.dumps({
            "repo_url": repo_url,
            "mcp_output_dir": conversion_result.mcp_output_dir,
            "service_name": conversion_result.service_name,
            "tools": available_tools,
            "tool_executed": tool_executed,
            "tool_result": {
                "success": tool_call_result.success if tool_call_result else False,
                "result": str(tool_call_result.result) if tool_call_result else None,
                "error": tool_call_result.error if tool_call_result else None
            } if tool_call_result else None
        })
        
        return {
            "current_step_index": idx + 1,
            "messages": [AgentMessage(role="assistant", content=result_text)]
        }
        
    except Exception as e:
        logger.exception("Code2MCP execution failed")
        current_step.status = TaskStatus.FAILED
        current_step.error = str(e)
        return {
            "messages": [AgentMessage(role="system", content=f"Code2MCP Error: {str(e)}")]
        }
