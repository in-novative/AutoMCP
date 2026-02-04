from mcp.server import Server
import mcp.types as types
from typing import Any, Dict

# 1. 初始化标准 Server
server = Server("AutoMCP")

# 2. 定义工具函数 (纯函数)
def echo(message: str) -> str:
    """一个简单的回声工具"""
    return f"AutoMCP says: {message}"

def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    import platform, sys
    return {
        "os": platform.system(),
        "python": sys.version
    }

# 3. 注册工具列表
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="echo",
            description="Echo a message",
            inputSchema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            }
        ),
        types.Tool(
            name="get_system_info",
            description="Get system info",
            inputSchema={"type": "object", "properties": {}}
        )
    ]

# 4. 注册工具调用逻辑 (手动路由)
@server.call_tool()
async def handle_call_tool(name: str, arguments: Any) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "echo":
        result = echo(arguments["message"])
        return [types.TextContent(type="text", text=result)]
    
    elif name == "get_system_info":
        result = str(get_system_info())
        return [types.TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")