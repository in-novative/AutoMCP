import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.messages import HumanMessage

from src.workflow.graph import graph
from src.server.models import AgentMessage
from src.memory.rag_local import LocalToolRAG
from src.tools.registry import tool_registry
from src.tools.loader import load_and_register_all_tools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Code2MCP_E2E_Test")


async def setup_environment():
    logger.info("Setting up test environment...")
    
    tool_count = load_and_register_all_tools()
    logger.info(f"Loaded {tool_count} tools")
    
    rag = LocalToolRAG()
    existing_tools = rag.collection.count()
    
    if existing_tools == 0 and tool_count > 0:
        all_tools = tool_registry.get_all()
        rag.index_batch([
            {
                "id": t.name,
                "content": f"{t.name}: {t.description}",
                "meta": {"type": "local"}
            }
            for t in all_tools
        ])
        logger.info(f"Indexed {len(all_tools)} tools to RAG")
    
    logger.info("Environment setup complete.")


async def run_test_case(case_name: str, user_input: str):
    logger.info(f"\n{'='*20} Code2MCP E2E Test: {case_name} {'='*20}")
    logger.info(f"User Input: {user_input}")
    
    initial_state = {
        "messages": [AgentMessage(role="user", content=user_input)],
        "plan": [],
        "current_step_index": 0,
        "reflection_count": 0
    }
    
    try:
        final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 30})
        
        print("\n--- Execution Result ---")
        plan = final_state.get("plan", [])
        
        for i, step in enumerate(plan):
            status_icon = "✅" if step.status == "completed" else "❌"
            print(f"\nStep {i+1} [{step.task_type}]: {step.description}")
            print(f"  Status: {status_icon} {step.status}")
            if step.result:
                print(f"  Result:\n{step.result}")
            if step.error:
                print(f"  Error: {step.error}")
                
        last_msg = final_state["messages"][-1]
        print(f"\nFinal Response:\n{last_msg.content}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


async def main():
    await setup_environment()
    
    print("\n" + "="*60)
    print("开始 Code2MCP 端到端测试")
    print("="*60)
    
    test_cases = [
        ("简单文本处理", "使用 Python 进行文本处理，统计文本中的单词数"),
        ("数学计算", "使用 sympy 进行数学计算，计算 1 到 100 的和"),
        ("数据处理", "使用 pandas 处理 CSV 数据"),
    ]
    
    for case_name, user_input in test_cases:
        await run_test_case(case_name, user_input)
        
        print("\n" + "-"*60)
        await asyncio.sleep(1)
    
    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
