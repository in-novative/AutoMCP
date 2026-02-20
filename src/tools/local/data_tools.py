"""
数据处理工具集
提供 CSV、JSON 等数据的简单处理功能
"""

import json
import csv
import io
from typing import List, Dict, Any
from langchain_core.tools import tool


@tool
def parse_csv(csv_text: str) -> str:
    """
    解析 CSV 文本并显示前 5 行。
    
    Args:
        csv_text: CSV 格式的文本内容
        
    Returns:
        解析结果
    """
    try:
        lines = csv_text.strip().split('\n')
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        
        if not rows:
            return "CSV 内容为空"
        
        # 显示表头
        result = f"表头: {', '.join(rows[0])}\n"
        result += f"总行数: {len(rows)}\n"
        result += "前 5 行数据:\n"
        
        for i, row in enumerate(rows[1:6], 1):
            result += f"  {i}. {', '.join(row)}\n"
        
        return result
        
    except Exception as e:
        return f"CSV 解析错误: {e}"


@tool
def csv_to_json(csv_text: str) -> str:
    """
    将 CSV 文本转换为 JSON 格式。
    
    Args:
        csv_text: CSV 格式的文本内容
        
    Returns:
        JSON 格式的字符串
    """
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        data = list(reader)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"转换错误: {e}"


@tool
def validate_json(json_text: str) -> str:
    """
    验证 JSON 字符串是否有效。
    
    Args:
        json_text: JSON 字符串
        
    Returns:
        验证结果
    """
    try:
        data = json.loads(json_text)
        
        # 统计信息
        if isinstance(data, dict):
            return f"✓ 有效的 JSON 对象\n键数量: {len(data)}\n顶层键: {', '.join(list(data.keys())[:10])}"
        elif isinstance(data, list):
            return f"✓ 有效的 JSON 数组\n元素数量: {len(data)}"
        else:
            return f"✓ 有效的 JSON ({type(data).__name__})"
            
    except json.JSONDecodeError as e:
        return f"✗ JSON 格式错误: {e}"


@tool
def extract_json_keys(json_text: str) -> str:
    """
    提取 JSON 对象中的所有键。
    
    Args:
        json_text: JSON 字符串
        
    Returns:
        键列表
    """
    try:
        data = json.loads(json_text)
        
        def get_keys(obj, prefix=""):
            keys = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    keys.append(full_key)
                    if isinstance(v, (dict, list)):
                        keys.extend(get_keys(v, full_key))
            elif isinstance(obj, list) and obj:
                keys.extend(get_keys(obj[0], prefix))
            return keys
        
        all_keys = get_keys(data)
        return f"找到 {len(all_keys)} 个键:\n" + '\n'.join([f"  - {k}" for k in all_keys[:20]])
        
    except json.JSONDecodeError as e:
        return f"JSON 解析错误: {e}"


@tool
def calculate_statistics(numbers_text: str) -> str:
    """
    计算一组数字的统计信息（平均值、最大值、最小值、总和）。
    
    Args:
        numbers_text: 逗号或换行分隔的数字字符串，如 "1, 2, 3, 4, 5"
        
    Returns:
        统计结果
    """
    try:
        # 解析数字
        import re
        nums = [float(n) for n in re.findall(r'-?\d+\.?\d*', numbers_text)]
        
        if not nums:
            return "未找到有效的数字"
        
        count = len(nums)
        total = sum(nums)
        avg = total / count
        max_val = max(nums)
        min_val = min(nums)
        
        return f"""统计结果（共 {count} 个数字）：
- 总和: {total:.2f}
- 平均值: {avg:.2f}
- 最大值: {max_val:.2f}
- 最小值: {min_val:.2f}"""
        
    except Exception as e:
        return f"计算错误: {e}"


@tool
def filter_data_lines(text: str, keyword: str) -> str:
    """
    从文本中过滤包含特定关键词的行。
    
    Args:
        text: 原始文本
        keyword: 要搜索的关键词
        
    Returns:
        匹配的行
    """
    lines = text.split('\n')
    matching = [line for line in lines if keyword.lower() in line.lower()]
    
    if matching:
        return f"找到 {len(matching)} 行包含 '{keyword}':\n" + '\n'.join([f"  {line}" for line in matching[:20]])
    else:
        return f"未找到包含 '{keyword}' 的行"
