"""
文本处理工具集
提供文本分析、转换、搜索等功能
"""

import re
from typing import List
from langchain_core.tools import tool


@tool
def count_words(text: str) -> str:
    """
    统计文本的字数、行数和字符数。
    
    Args:
        text: 要统计的文本
        
    Returns:
        统计结果
    """
    lines = text.split('\n')
    words = len(text.split())
    chars = len(text)
    chars_no_space = len(text.replace(' ', '').replace('\n', ''))
    
    return f"""文本统计结果：
- 总行数: {len(lines)}
- 总词数: {words}
- 总字符数: {chars}
- 字符数（不含空格）: {chars_no_space}"""


@tool
def find_pattern(text: str, pattern: str) -> str:
    """
    在文本中搜索匹配正则表达式的内容。
    
    Args:
        text: 要搜索的文本
        pattern: 正则表达式模式
        
    Returns:
        匹配结果
    """
    try:
        matches = re.findall(pattern, text)
        if matches:
            return f"找到 {len(matches)} 个匹配:\n" + '\n'.join([f"  - {m}" for m in matches[:10]])
        else:
            return "未找到匹配内容"
    except re.error as e:
        return f"正则表达式错误: {e}"


@tool
def replace_text(text: str, old: str, new: str) -> str:
    """
    替换文本中的指定内容。
    
    Args:
        text: 原始文本
        old: 要替换的内容
        new: 替换后的内容
        
    Returns:
        替换后的文本
    """
    count = text.count(old)
    result = text.replace(old, new)
    return f"替换了 {count} 处内容:\n{result}"


@tool
def extract_urls(text: str) -> str:
    """
    从文本中提取所有 URL 链接。
    
    Args:
        text: 包含 URL 的文本
        
    Returns:
        提取的 URL 列表
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    if urls:
        return f"找到 {len(urls)} 个 URL:\n" + '\n'.join([f"  {i+1}. {url}" for i, url in enumerate(urls)])
    else:
        return "未找到 URL"


@tool
def format_json(text: str) -> str:
    """
    格式化 JSON 字符串（如果输入是有效的 JSON）。
    
    Args:
        text: JSON 字符串
        
    Returns:
        格式化后的 JSON 或错误信息
    """
    import json
    try:
        data = json.loads(text)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"JSON 格式错误: {e}"


@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    生成文本摘要（简单实现：截取前 N 个字符并添加省略号）。
    
    Args:
        text: 要摘要的文本
        max_length: 最大长度，默认 100
        
    Returns:
        文本摘要
    """
    if len(text) <= max_length:
        return text
    
    # 在句子边界处截断
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    last_space = truncated.rfind(' ')
    
    cut_point = max(last_period, last_newline, last_space)
    if cut_point > max_length * 0.5:
        return truncated[:cut_point] + "..."
    else:
        return truncated + "..."
