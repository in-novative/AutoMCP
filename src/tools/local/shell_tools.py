"""
系统命令工具集
提供系统信息查询、命令执行等功能
"""

import platform
import subprocess
import os
import psutil
import socket
from langchain_core.tools import tool
from datetime import datetime
from typing import List


@tool
def get_system_info() -> str:
    """
    获取系统基本信息（操作系统、Python版本等）。

    Returns:
        系统信息字符串
    """
    info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
        "Current Directory": os.getcwd(),
        "User": os.getenv("USER") or os.getenv("USERNAME") or "unknown"
    }

    return "\n".join([f"{k}: {v}" for k, v in info.items()])


@tool
def get_cpu_info() -> str:
    """
    获取 CPU 详细信息（使用率、核心数、频率等）。

    Returns:
        CPU 信息字符串
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_stats = psutil.cpu_stats()

        info = f"""CPU 信息:
- 物理核心数: {psutil.cpu_count(logical=False)}
- 逻辑核心数: {cpu_count}
- 当前使用率: {cpu_percent}%
- 当前频率: {cpu_freq.current:.2f} MHz
- 最大频率: {cpu_freq.max:.2f} MHz
- 最小频率: {cpu_freq.min:.2f} MHz
- 上下文切换次数: {cpu_stats.ctx_switches}
- 中断次数: {cpu_stats.interrupts}
- 系统调用次数: {cpu_stats.syscalls}"""
        return info
    except Exception as e:
        return f"获取 CPU 信息失败: {e}"


@tool
def get_memory_info() -> str:
    """
    获取内存使用情况（总内存、可用内存、使用率等）。

    Returns:
        内存信息字符串
    """
    try:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        info = f"""内存信息:
物理内存:
- 总内存: {mem.total / (1024**3):.2f} GB
- 可用内存: {mem.available / (1024**3):.2f} GB
- 已用内存: {mem.used / (1024**3):.2f} GB
- 使用率: {mem.percent}%
- 空闲内存: {mem.free / (1024**3):.2f} GB

交换内存:
- 总交换空间: {swap.total / (1024**3):.2f} GB
- 已用交换空间: {swap.used / (1024**3):.2f} GB
- 交换空间使用率: {swap.percent}%"""
        return info
    except Exception as e:
        return f"获取内存信息失败: {e}"


@tool
def get_disk_info() -> str:
    """
    获取磁盘使用情况（各分区的总空间、已用空间、可用空间）。

    Returns:
        磁盘信息字符串
    """
    try:
        partitions = psutil.disk_partitions()
        info = "磁盘分区信息:\n"

        for part in partitions:
            try:
                usage = psutil.disk_usage(part.mountpoint)
                info += f"""
分区: {part.device}
- 挂载点: {part.mountpoint}
- 文件系统: {part.fstype}
- 总空间: {usage.total / (1024**3):.2f} GB
- 已用空间: {usage.used / (1024**3):.2f} GB
- 可用空间: {usage.free / (1024**3):.2f} GB
- 使用率: {usage.percent}%
"""
            except PermissionError:
                continue

        return info
    except Exception as e:
        return f"获取磁盘信息失败: {e}"


@tool
def get_network_info() -> str:
    """
    获取网络信息（IP地址、网络接口、连接状态等）。

    Returns:
        网络信息字符串
    """
    try:
        # 获取主机名和IP
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "无法获取"

        # 获取网络接口信息
        net_io = psutil.net_io_counters()
        net_if_addrs = psutil.net_if_addrs()

        info = f"""网络信息:
主机名: {hostname}
IP 地址: {ip_address}

网络流量统计:
- 发送字节数: {net_io.bytes_sent / (1024**2):.2f} MB
- 接收字节数: {net_io.bytes_recv / (1024**2):.2f} MB
- 发送数据包: {net_io.packets_sent}
- 接收数据包: {net_io.packets_recv}
- 发送错误: {net_io.errout}
- 接收错误: {net_io.errin}

网络接口:"""

        for interface, addrs in net_if_addrs.items():
            info += f"\n  {interface}:"
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    info += f"\n    IPv4: {addr.address}"
                elif addr.family == socket.AF_INET6:
                    info += f"\n    IPv6: {addr.address}"
                elif addr.family == psutil.AF_LINK:
                    info += f"\n    MAC: {addr.address}"

        return info
    except Exception as e:
        return f"获取网络信息失败: {e}"


@tool
def get_process_list(limit: int = 20) -> str:
    """
    获取正在运行的进程列表（按CPU使用率排序）。

    Args:
        limit: 返回的进程数量，默认20个

    Returns:
        进程列表字符串
    """
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                pinfo = proc.info
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 按CPU使用率排序
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)

        info = f"正在运行的进程（前 {limit} 个，按CPU使用率排序）:\n"
        info += f"{'PID':<10} {'名称':<25} {'CPU%':<10} {'内存%':<10} {'状态':<15}\n"
        info += "-" * 70 + "\n"

        for proc in processes[:limit]:
            info += f"{proc['pid']:<10} {proc['name'][:24]:<25} {proc['cpu_percent'] or 0:<10.1f} {proc['memory_percent'] or 0:<10.1f} {proc['status']:<15}\n"

        return info
    except Exception as e:
        return f"获取进程列表失败: {e}"


@tool
def kill_process(pid: int) -> str:
    """
    终止指定 PID 的进程。

    Args:
        pid: 进程ID

    Returns:
        操作结果
    """
    try:
        process = psutil.Process(pid)
        process_name = process.name()
        process.terminate()
        return f"已终止进程: {process_name} (PID: {pid})"
    except psutil.NoSuchProcess:
        return f"错误: 进程 {pid} 不存在"
    except psutil.AccessDenied:
        return f"错误: 没有权限终止进程 {pid}"
    except Exception as e:
        return f"终止进程失败: {e}"


@tool
def get_boot_time() -> str:
    """
    获取系统启动时间和运行时长。

    Returns:
        系统启动信息
    """
    try:
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time

        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"""系统启动信息:
- 启动时间: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}
- 运行时长: {days}天 {hours}小时 {minutes}分钟 {seconds}秒"""
    except Exception as e:
        return f"获取启动时间失败: {e}"


@tool
def get_logged_in_users() -> str:
    """
    获取当前登录的用户列表。

    Returns:
        用户信息字符串
    """
    try:
        users = psutil.users()
        if not users:
            return "当前没有用户登录"

        info = "当前登录的用户:\n"
        for user in users:
            login_time = datetime.fromtimestamp(user.started).strftime('%Y-%m-%d %H:%M:%S')
            info += f"- 用户名: {user.name}\n"
            info += f"  终端: {user.terminal}\n"
            info += f"  主机: {user.host}\n"
            info += f"  登录时间: {login_time}\n\n"

        return info
    except Exception as e:
        return f"获取用户信息失败: {e}"


@tool
def find_process_by_name(name: str) -> str:
    """
    根据名称查找进程。

    Args:
        name: 进程名称（支持部分匹配）

    Returns:
        匹配的进程列表
    """
    try:
        matching_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if name.lower() in proc.info['name'].lower():
                    matching_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not matching_processes:
            return f"未找到包含 '{name}' 的进程"

        info = f"找到 {len(matching_processes)} 个匹配的进程:\n"
        info += f"{'PID':<10} {'名称':<30} {'CPU%':<10} {'内存%':<10}\n"
        info += "-" * 60 + "\n"

        for proc in matching_processes:
            info += f"{proc['pid']:<10} {proc['name'][:29]:<30} {proc['cpu_percent'] or 0:<10.1f} {proc['memory_percent'] or 0:<10.1f}\n"

        return info
    except Exception as e:
        return f"查找进程失败: {e}"


@tool
def execute_command(command: str) -> str:
    """
    执行系统命令（谨慎使用，仅限安全命令）。
    
    Args:
        command: 要执行的命令
        
    Returns:
        命令输出或错误信息
    """
    # 安全限制：禁止危险命令
    dangerous_commands = ["rm -rf", "del /", "format", "mkfs"]
    for dangerous in dangerous_commands:
        if dangerous in command.lower():
            return f"Error: Dangerous command detected: {dangerous}"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout if result.returncode == 0 else result.stderr
        return f"Exit code: {result.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (30s)"
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def get_current_time() -> str:
    """
    获取当前日期和时间。
    
    Returns:
        当前时间字符串
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_env_variable(name: str) -> str:
    """
    获取环境变量的值。
    
    Args:
        name: 环境变量名
        
    Returns:
        环境变量值或错误信息
    """
    value = os.getenv(name)
    if value is not None:
        return f"{name}={value}"
    else:
        return f"Environment variable '{name}' not found"
