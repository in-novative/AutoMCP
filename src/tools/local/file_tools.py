"""
文件操作工具集
提供文件读写、目录操作等功能
"""

import os
from pathlib import Path
from langchain_core.tools import tool


@tool
def write_file(path: str, content: str) -> str:
    """
    写入内容到文件。如果文件不存在则创建，存在则覆盖。
    
    Args:
        path: 文件路径
        content: 要写入的内容
        
    Returns:
        操作结果信息
    """
    try:
        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def read_file(path: str) -> str:
    """
    读取文件内容。
    
    Args:
        path: 文件路径
        
    Returns:
        文件内容或错误信息
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """
    列出目录下的所有文件和子目录。
    
    Args:
        path: 目录路径，默认为当前目录
        
    Returns:
        目录内容列表
    """
    try:
        items = os.listdir(path)
        files = [f for f in items if os.path.isfile(os.path.join(path, f))]
        dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
        
        result = f"Directory: {path}\n"
        result += f"Subdirectories ({len(dirs)}): {', '.join(dirs) if dirs else 'None'}\n"
        result += f"Files ({len(files)}): {', '.join(files) if files else 'None'}"
        return result
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def create_directory(path: str) -> str:
    """
    创建目录（包括父目录）。
    
    Args:
        path: 目录路径
        
    Returns:
        操作结果信息
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory: {path}"
    except Exception as e:
        return f"Error creating directory: {e}"


@tool
def delete_file(path: str) -> str:
    """
    删除文件。
    
    Args:
        path: 文件路径
        
    Returns:
        操作结果信息
    """
    try:
        os.remove(path)
        return f"Successfully deleted file: {path}"
    except Exception as e:
        return f"Error deleting file: {e}"


@tool
def file_exists(path: str) -> str:
    """
    检查文件或目录是否存在。
    
    Args:
        path: 路径
        
    Returns:
        存在性检查结果
    """
    exists = os.path.exists(path)
    is_file = os.path.isfile(path) if exists else False
    is_dir = os.path.isdir(path) if exists else False
    
    if exists:
        type_str = "file" if is_file else "directory" if is_dir else "unknown"
        return f"Path exists: {path} (type: {type_str})"
    else:
        return f"Path does not exist: {path}"
