"""
代码开发工具集
提供代码分析、格式化、检查等功能
"""

import ast
import re
from langchain_core.tools import tool


@tool
def analyze_python_code(code: str) -> str:
    """
    分析 Python 代码，检查语法错误和基本统计信息。
    
    Args:
        code: Python 代码字符串
        
    Returns:
        代码分析报告
    """
    try:
        # 语法检查
        tree = ast.parse(code)
        
        # 统计信息
        functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        imports = len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
        lines = code.count('\n') + 1
        
        return f"""Python 代码分析结果：
✓ 语法检查通过
- 总行数: {lines}
- 函数数量: {functions}
- 类数量: {classes}
- 导入语句: {imports}"""
        
    except SyntaxError as e:
        return f"✗ 语法错误 (第 {e.lineno} 行): {e.msg}"
    except Exception as e:
        return f"✗ 分析错误: {e}"


@tool
def count_code_lines(code: str, language: str = "python") -> str:
    """
    统计代码的行数（包括代码行、注释行、空行）。
    
    Args:
        code: 代码字符串
        language: 编程语言，支持 python/javascript/java/c 等
        
    Returns:
        行数统计结果
    """
    lines = code.split('\n')
    total = len(lines)
    
    comment_patterns = {
        "python": r'^\s*#',
        "javascript": r'^\s*//',
        "java": r'^\s*//',
        "c": r'^\s*//',
    }
    
    pattern = comment_patterns.get(language, r'^\s*#')
    
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif re.match(pattern, stripped):
            comment_lines += 1
        else:
            code_lines += 1
    
    return f"""{language.title()} 代码行数统计：
- 总行数: {total}
- 代码行: {code_lines}
- 注释行: {comment_lines}
- 空行: {blank_lines}"""


@tool
def extract_functions(code: str) -> str:
    """
    从 Python 代码中提取所有函数名和类名。
    
    Args:
        code: Python 代码
        
    Returns:
        函数和类列表
    """
    try:
        tree = ast.parse(code)
        
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        result = []
        if classes:
            result.append(f"类 ({len(classes)} 个): {', '.join(classes)}")
        if functions:
            result.append(f"函数 ({len(functions)} 个): {', '.join(functions)}")
        
        return '\n'.join(result) if result else "未找到函数或类"
        
    except SyntaxError as e:
        return f"语法错误: {e}"


@tool
def check_code_style(code: str) -> str:
    """
    简单的代码风格检查（检查缩进、行长度等）。
    
    Args:
        code: 代码字符串
        
    Returns:
        风格检查报告
    """
    issues = []
    lines = code.split('\n')
    
    for i, line in enumerate(lines, 1):
        # 检查行长度
        if len(line) > 100:
            issues.append(f"第 {i} 行: 超过 100 字符")
        
        # 检查尾随空格
        if line.rstrip() != line:
            issues.append(f"第 {i} 行: 有尾随空格")
        
        # 检查 Tab 字符
        if '\t' in line:
            issues.append(f"第 {i} 行: 使用 Tab 而非空格")
    
    if issues:
        return "代码风格问题:\n" + '\n'.join([f"  - {issue}" for issue in issues[:10]])
    else:
        return "✓ 代码风格检查通过"


@tool
def generate_function_docstring(function_code: str) -> str:
    """
    为函数生成简单的文档字符串模板。
    
    Args:
        function_code: 函数代码
        
    Returns:
        带文档字符串的函数
    """
    try:
        tree = ast.parse(function_code)
        func = tree.body[0]
        
        if not isinstance(func, ast.FunctionDef):
            return "输入不是有效的函数定义"
        
        func_name = func.name
        args = [arg.arg for arg in func.args.args]
        
        docstring = f'''    """
    {func_name} 函数的简要说明。
    
    Args:
        {chr(10).join([f"{arg}: 参数说明" for arg in args])}
        
    Returns:
        返回值说明
    """'''
        
        # 在函数定义后插入文档字符串
        lines = function_code.split('\n')
        indent = len(lines[0]) - len(lines[0].lstrip())
        spaces = ' ' * (indent + 4)
        
        docstring_lines = docstring.split('\n')
        formatted_docstring = '\n'.join([spaces + line if line.strip() else line for line in docstring_lines])
        
        return f"建议的文档字符串:\n{formatted_docstring}"
        
    except Exception as e:
        return f"生成失败: {e}"
