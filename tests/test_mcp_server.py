import pytest
from src.core.mcp_server import mcp

# 注意：FastMCP 的工具函数在被装饰后，
# 可以直接像普通函数一样调用，也可以通过 mcp.list_tools() 查看元数据

@pytest.mark.asyncio
async def test_tool_registration():
    """测试工具是否已成功注册"""
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "echo" in tool_names
    assert "get_system_info" in tool_names
    assert "read_file_preview" in tool_names

@pytest.mark.asyncio
async def test_echo_tool():
    """测试 echo 工具逻辑"""
    # 直接调用工具函数
    result = await mcp.call_tool("echo", {"message": "Hello World"})
    # FastMCP 返回的结果通常是 list[TextContent] 或类似结构，取决于版本
    # 但如果是直接调用内部函数逻辑，我们可以模拟调用
    
    # 更简单的方式：直接 import 函数本身进行测试
    # 但由于装饰器的存在，最好的方式是通过 mcp 实例调用接口
    
    # 假设我们通过 mcp.call_tool 接口调用
    assert "Hello World" in str(result)

@pytest.mark.asyncio
async def test_system_info_tool():
    """测试系统信息工具"""
    result = await mcp.call_tool("get_system_info", {})
    # 验证返回结果包含预期字段
    assert "os" in str(result)
    assert "python_version" in str(result)

@pytest.mark.asyncio
async def test_read_file_preview(tmp_path):
    """
    测试文件读取工具
    使用 pytest 的 tmp_path fixture 创建临时文件
    """
    # 1. 创建临时文件
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("Line 1\nLine 2\nLine 3")
    
    # 2. 调用工具读取
    result = await mcp.call_tool("read_file_preview", {"path": str(p), "lines": 2})
    
    # 3. 验证结果
    content = str(result)
    assert "Line 1" in content
    assert "Line 2" in content
    assert "Line 3" not in content  # 因为只读了前2行 (如果逻辑正确的话)