"""
工具注册表模块
管理所有可用的本地工具，支持动态注册和检索
"""

import logging
from typing import Dict, List, Any, Callable, Optional
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册表
    单例模式管理所有工具
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, BaseTool] = {}
        return cls._instance
    
    def register(self, tool: BaseTool) -> None:
        """
        注册一个工具
        
        Args:
            tool: LangChain BaseTool 实例
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_many(self, tools: List[BaseTool]) -> None:
        """
        批量注册工具
        
        Args:
            tools: 工具列表
        """
        for tool in tools:
            self.register(tool)
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        根据名称获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例，如果不存在返回 None
        """
        return self._tools.get(name)
    
    def get_many(self, names: List[str]) -> List[BaseTool]:
        """
        批量获取工具
        
        Args:
            names: 工具名称列表
            
        Returns:
            存在的工具实例列表
        """
        tools = []
        for name in names:
            tool = self.get(name)
            if tool:
                tools.append(tool)
            else:
                logger.warning(f"Tool not found: {name}")
        return tools
    
    def get_all(self) -> List[BaseTool]:
        """
        获取所有已注册的工具
        
        Returns:
            所有工具实例列表
        """
        return list(self._tools.values())
    
    def list_tools(self) -> List[str]:
        """
        列出所有已注册的工具名称
        
        Returns:
            工具名称列表
        """
        return list(self._tools.keys())
    
    def unregister(self, name: str) -> bool:
        """
        注销一个工具
        
        Args:
            name: 工具名称
            
        Returns:
            是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def clear(self) -> None:
        """清空所有工具"""
        self._tools.clear()
        logger.info("Cleared all tools")


# 全局单例实例
tool_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> BaseTool:
    """
    装饰器方式注册工具
    
    使用示例:
        @register_tool
        @tool
        def my_tool(x: str) -> str:
            return f"Result: {x}"
    """
    tool_registry.register(tool)
    return tool
