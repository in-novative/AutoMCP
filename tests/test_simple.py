import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.workflow.graph import graph
from src.server.models import AgentMessage
from src.memory.rag_local import LocalToolRAG
from src.tools.registry import tool_registry
from src.tools.loader import load_and_register_all_tools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleTest")


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


async def run_simple_test():
    await setup_environment()
    
    print("\n" + "="*60)
    print("简单测试：纯 LLM 问答")
    print("="*60)
    
    user_input = "请解释一下什么是 MCP 协议？"
    
    initial_state = {
        "messages": [AgentMessage(role="user", content=user_input)],
        "plan": [],
        "current_step_index": 0,
        "reflection_count": 0
    }
    
    try:
        final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 10})
        
        print("\n--- 执行结果 ---")
        last_msg = final_state["messages"][-1]
        print(f"回答: {last_msg.content}")
        
        print("\n✅ 测试成功！")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ 测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(run_simple_test())
