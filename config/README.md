# AutoMCP Configuration Guide

本目录包含 AutoMCP 项目的核心配置模块。

## 1. 环境变量配置 (`settings.py`)

基于 `pydantic-settings` 实现，提供类型安全的环境变量读取与验证。

### 特性
- **自动加载**: 自动读取项目根目录下的 `.env` 文件。
- **类型验证**: 自动将环境变量转换为指定的 Python 类型（如 `PORT` 转为 `int`）。
- **安全脱敏**: `SecretStr` 类型防止敏感信息（如 API Key）在日志中明文泄露。

### 如何使用

```python
from config.settings import settings

# 1. 获取基础配置
print(f"Running in {settings.ENV} mode on port {settings.PORT}")

# 2. 获取敏感信息 (需要调用 .get_secret_value())
if settings.OPENAI_API_KEY:
    api_key = settings.OPENAI_API_KEY.get_secret_value()
    # use api_key...
```

### 关键字段
| 字段名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ENV` | str | "development" | 运行环境 |
| `DEBUG` | bool | True | 调试模式开关 |
| `OPENAI_API_KEY` | SecretStr | None | OpenAI 密钥 |
| `CHROMA_DB_PATH` | str | "./data/chroma" | 向量库路径 |

---

## 2. 日志系统 (`logging.py`)

提供统一的日志格式与输出通道管理，支持文件轮转与多通道分离。

### 特性
- **多通道**: 支持控制台 (`console`) 和文件 (`file`) 同时输出。
- **自动轮转**: 单个日志文件最大 10MB，保留 5 个备份。
- **专用通道**: 提供 `execution` 专用通道，用于独立记录模型/任务执行结果。

### 如何使用

**初始化 (在 `main.py` 顶部调用)**
```python
from config.logging import setup_logging

# 必须在应用启动最早阶段调用
setup_logging()
```

**日常打日志 (General Logging)**
记录系统运行状态、错误信息等，会写入 `logs/automcp.log`。
```python
import logging

logger = logging.getLogger(__name__)  # 或者使用 "src"

logger.info("System initialized")
logger.error("Database connection failed", exc_info=True)
```

**记录执行结果 (Execution Logging)**
专门记录 Agent 的思考过程、工具调用结果等，会**独立**写入 `logs/execution.log`，保持主日志纯净。
```python
import logging

# 获取专用 logger
exec_logger = logging.getLogger("execution")

result = {"task_id": "123", "status": "success", "output": "..."}
exec_logger.info(f"Task Completed: {result}")
```

### 日志文件位置
- 系统日志: `logs/automcp.log`
- 执行日志: `logs/execution.log`
