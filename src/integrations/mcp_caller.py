import httpx
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MCPToolCall:
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class MCPToolResult:
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskParser:
    @staticmethod
    def parse_task_description(description: str) -> tuple[str, Optional[MCPToolCall]]:
        parts = description.split('|')
        main_task = parts[0].strip()
        
        if len(parts) > 1:
            tool_part = parts[1].strip()
            if ':' in tool_part:
                tool_name, args_str = tool_part.split(':', 1)
                try:
                    args = json.loads(args_str)
                    return main_task, MCPToolCall(tool_name.strip(), args)
                except:
                    pass
        
        return main_task, None
    
    @staticmethod
    def extract_tool_hints(description: str, available_tools: List[Dict[str, str]]) -> Optional[str]:
        for tool in available_tools:
            tool_name = tool.get('name', '')
            if tool_name and tool_name.lower() in description.lower():
                return tool_name
        return None


class SimpleMCPCaller:
    @staticmethod
    async def call_tool_direct(
        mcp_output_dir: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> MCPToolResult:
        try:
            import importlib.util
            import sys
            from pathlib import Path
            
            mcp_plugin_dir = Path(mcp_output_dir) / "mcp_plugin"
            if not mcp_plugin_dir.exists():
                return MCPToolResult(
                    success=False,
                    error=f"MCP plugin directory not found: {mcp_plugin_dir}"
                )
            
            if str(mcp_plugin_dir) not in sys.path:
                sys.path.insert(0, str(mcp_plugin_dir))
            
            try:
                mcp_service_path = mcp_plugin_dir / "mcp_service.py"
                if mcp_service_path.exists():
                    spec = importlib.util.spec_from_file_location("mcp_service", mcp_service_path)
                    if spec and spec.loader:
                        mcp_service = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mcp_service)
                        
                        if hasattr(mcp_service, 'create_app'):
                            app = mcp_service.create_app()
                            if hasattr(app, '_tools'):
                                for name, tool in app._tools.items():
                                    if name == tool_name:
                                        try:
                                            result = tool.func(**arguments)
                                            return MCPToolResult(success=True, result=result)
                                        except Exception as e:
                                            return MCPToolResult(success=False, error=str(e))
            except Exception as e:
                logger.warning(f"Direct tool call failed: {e}")
            
            return MCPToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found or could not be called"
            )
            
        except Exception as e:
            logger.exception(f"Failed to call tool directly")
            return MCPToolResult(success=False, error=str(e))
