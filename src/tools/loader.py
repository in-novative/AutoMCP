"""
工具自动加载器
自动发现并注册 src/tools/local/ 目录下的所有工具
"""

import os
import importlib
import inspect
import logging
from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool
from src.tools.registry import tool_registry

logger = logging.getLogger(__name__)

def discover_tools() -> List[BaseTool]:
    """
    自动发现 local 目录下所有模块中的工具
    
    Returns:
        发现的所有工具列表
    """
    tools = []
    
    # 获取 local 目录路径
    current_dir = Path(__file__).parent
    local_dir = current_dir / "local"
    
    if not local_dir.exists():
        logger.warning(f"Tools directory not found: {local_dir}")
        return tools
    
    # 遍历 local 目录下的所有 .py 文件
    for file_path in local_dir.glob("*.py"):
        if file_path.name.startswith("_"):
            continue  # 跳过 __init__.py 等私有文件
        
        module_name = f"src.tools.local.{file_path.stem}"
        
        try:
            # 动态导入模块
            module = importlib.import_module(module_name)
            
            # 查找模块中的所有工具
            for name, obj in inspect.getmembers(module):
                # 检查是否是 LangChain 工具
                if isinstance(obj, BaseTool):
                    tools.append(obj)
                    logger.debug(f"Discovered tool: {obj.name} from {module_name}")
                    
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")
    
    return tools

def load_and_register_all_tools() -> int:
    """
    加载并注册所有发现的工具
    
    Returns:
        注册的工具数量
    """
    tools = discover_tools()
    
    if tools:
        tool_registry.register_many(tools)
        logger.info(f"Auto-loaded and registered {len(tools)} tools")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description[:50]}...")
    else:
        logger.warning("No tools discovered")
    
    return len(tools)


def get_tool_modules() -> List[str]:
    """
    获取所有可用的工具模块名称
    
    Returns:
        模块名称列表
    """
    current_dir = Path(__file__).parent
    local_dir = current_dir / "local"
    
    if not local_dir.exists():
        return []
    
    modules = []
    for file_path in local_dir.glob("*.py"):
        if not file_path.name.startswith("_"):
            modules.append(file_path.stem)
    
    return modules
