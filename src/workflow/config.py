from enum import Enum


class CodeToMCPMode(str, Enum):
    """Code-to-MCP 执行模式"""
    INTERNAL = "internal"  # 使用原有的 code_generator（代码生成）
    CODE2MCP = "code2mcp"  # 使用 Code2MCP 集成（仓库转换）
    AUTO = "auto"  # 自动选择（优先 CODE2MCP，失败则回退到 INTERNAL）


# 默认配置
DEFAULT_CODE_TO_MCP_MODE = CodeToMCPMode.AUTO
