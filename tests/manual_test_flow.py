import asyncio
import logging
from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.workflow.graph import graph
from src.server.models import AgentMessage
from src.memory.rag_local import LocalToolRAG
from src.tools.registry import tool_registry
from src.tools.loader import load_and_register_all_tools

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ManualTest")

# --- 1. 初始化环境 ---
async def setup_environment():
    """初始化 RAG 和工具注册表"""
    logger.info("Setting up test environment...")
    
    # 1. 自动加载并注册所有本地工具
    tool_count = load_and_register_all_tools()
    
    # 2. 索引工具到 RAG（供语义检索使用）
    rag = LocalToolRAG()
    existing_tools = rag.collection.count()
    
    if existing_tools == 0 and tool_count > 0:
        # 获取所有已注册的工具并索引
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
    else:
        logger.info(f"RAG already has {existing_tools} tools indexed, skipping indexing")
    
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
    
    print("\n" + "="*60)
    print("开始测试 Local MCP 工具")
    print("="*60)
    
    # 测试用例 1: 文件创建
    await run_test_case(
        "1. 文件创建", 
        "在tests文件夹下创建一个名为 test.txt 的文件，内容是 'Hello AutoMCP!'"
    )
    
    # 测试用例 2: 文件读取
    await run_test_case(
        "2. 文件读取",
        "读取 tests/test.txt 文件的内容"
    )
    
    # 测试用例 3: 目录浏览
    await run_test_case(
        "3. 目录浏览",
        "列出 tests 目录下的所有文件和文件夹"
    )
    
    # 测试用例 4: 系统信息
    await run_test_case(
        "4. 系统信息",
        "查看当前系统信息"
    )
    
    # 测试用例 5: 时间查询
    await run_test_case(
        "5. 时间查询",
        "现在几点了"
    )
    
    # 测试用例 6: 命令执行
    await run_test_case(
        "6. 命令执行",
        "执行命令 echo Hello World"
    )
    
    # 测试用例 7: 复合任务（多工具协作）
    await run_test_case(
        "7. 复合任务",
        "创建一个目录 test_folder，在里面创建文件 info.txt，写入当前时间"
    )
    
    # 测试用例 8: 文件存在检查
    await run_test_case(
        "8. 存在检查",
        "检查文件 tests/test.txt 是否存在"
    )
    
    # 测试用例 9: Pure LLM (问答)
    await run_test_case(
        "9. 纯问答",
        "请解释一下什么是 MCP 协议？"
    )
    
    # 测试用例 10: Code Gen (计算)
    await run_test_case(
        "10. 代码计算",
        "计算 1 到 100 的平方和"
    )
    """
    """
    print("\n" + "="*60)
    print("开始高难度复合任务测试")
    print("="*60)
    
    # 测试用例 11: 系统监控报告（多步骤协作）
    await run_test_case(
        "11. 系统监控报告",
        "请完成以下系统监控任务：1. 检查目录 'monitor_logs' 是否存在，不存在则创建 "
        "2. 获取当前系统信息 3. 获取当前时间 4. 列出当前目录下的所有文件 "
        "5. 将这些信息整合写入到 'monitor_logs/system_report.txt' 文件中 "
        "6. 读取并确认文件写入成功"
    )
    
    # 测试用例 12: 批量文件处理
    await run_test_case(
        "12. 批量文件处理",
        "请完成以下批量处理任务：1. 创建目录 'batch_test' "
        "2. 在该目录下创建 3 个文件：file1.txt、file2.txt、file3.txt "
        "3. 分别写入内容 'Content 1'、'Content 2'、'Content 3' "
        "4. 列出 batch_test 目录下的所有文件确认创建成功 "
        "5. 读取 file2.txt 的内容并显示"
    )
    
    # 测试用例 13: 环境诊断（最高难度）
    await run_test_case(
        "13. 环境诊断",
        "请执行完整的环境诊断：1. 获取系统信息并保存 "
        "2. 获取当前时间 3. 获取环境变量 PATH 的值 "
        "4. 执行命令 'python --version' 检查 Python 版本 "
        "5. 创建目录 'diagnostics' "
        "6. 将所有收集的信息写入 'diagnostics/report.txt' "
        "7. 检查 report.txt 是否存在并读取内容确认"
    )
    
    # 文本处理任务
    await run_test_case(
        "14. 文本分析",
        "统计这段文本的字数：'Hello World, this is a test message for counting words.'"
    )
    
    # 代码分析任务
    await run_test_case(
        "15. 代码检查",
        "分析以下 Python 代码：def hello():\\n    print('Hello')\\n    return True"
    )
    
    # 数据处理任务
    await run_test_case(
        "16. CSV 处理",
        "解析以下 CSV 数据：name,age\\nAlice,25\\nBob,30\\nCharlie,35"
    )
    
    # 综合任务
    await run_test_case(
        "17. 开发辅助",
        "创建一个 Python 文件 utils.py，写入一个计算阶乘的函数，然后分析代码质量"
    )
    
    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())