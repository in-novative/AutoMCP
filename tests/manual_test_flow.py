import asyncio
import logging
from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.workflow.graph import graph
from src.server.models import AgentMessage
from src.memory.rag_local import LocalToolRAG

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ManualTest")

# --- 1. 准备 Mock 工具 ---
@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file at the given path."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def read_file(path: str) -> str:
    """Reads content from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# --- 2. 初始化环境 ---
async def setup_environment():
    """初始化 RAG 和工具"""
    logger.info("Setting up test environment...")
    
    # 获取 RAG 实例
    rag = LocalToolRAG()
    
    # 注册 Mock 工具到 RAG (为了让 Classifier/Executor 能找到)
    # 注意：实际项目中这些工具应该在 Registry 中
    test_tools = [write_file, read_file]
    
    # 模拟工具注册过程：将工具描述写入 ChromaDB
    rag.index_batch([
        {
            "id": t.name,
            "content": f"{t.name}: {t.description}",
            "meta": {"type": "local", "args_schema": str(t.args)}
        } 
        for t in test_tools
    ])
    
    logger.info("Environment setup complete.")

# --- 3. 运行测试流程 ---
async def run_test_case(case_name: str, user_input: str):
    logger.info(f"\n{'='*20} Running Case: {case_name} {'='*20}")
    logger.info(f"User Input: {user_input}")
    
    # 构造初始状态
    initial_state = {
        "messages": [AgentMessage(role="user", content=user_input)],
        "plan": [],
        "current_step_index": 0,
        "reflection_count": 0
    }
    
    try:
        # 运行 Graph
        # config={"recursion_limit": 50} 防止无限循环
        final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 20})
        
        # 输出结果摘要
        print("\n--- Execution Result ---")
        plan = final_state.get("plan", [])
        
        for i, step in enumerate(plan):
            status_icon = "✅" if step.status == "completed" else "❌"
            print(f"Step {i+1} [{step.task_type}]: {step.description}")
            print(f"  Status: {status_icon} {step.status}")
            if step.result:
                print(f"  Result: {step.result[:100]}...") # 只打印前100字符
            if step.error:
                print(f"  Error: {step.error}")
                
        last_msg = final_state["messages"][-1]
        print(f"\nFinal Response: {last_msg.content}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

# --- 主入口 ---
async def main():
    await setup_environment()
    
    # 测试用例 1: Local MCP (文件操作)
    await run_test_case(
        "Local File Op", 
        "在当前目录下创建一个名为 test_hello.txt 的文件，内容是 'Hello AutoMCP!'"
    )
    
    # 测试用例 2: Pure LLM (问答)
    # await run_test_case(
    #     "Pure Chat",
    #     "请解释一下什么是 MCP 协议？"
    # )

    # 测试用例 3: Code Gen (计算)
    # await run_test_case(
    #     "Code Calculation",
    #     "计算 1 到 100 的平方和"
    # )

if __name__ == "__main__":
    asyncio.run(main())