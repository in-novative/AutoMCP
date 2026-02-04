import sys
from pathlib import Path
from config.settings import settings

# 1. 定义日志目录 (可选)
# 确保 logs 文件夹存在，避免 FileHandler 报错
LOG_DIR = Path(settings.BASE_DIR) / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 2. 定义日志配置字典
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # 非常重要：防止覆盖 uvicorn/fastapi 默认的日志器
    
    # --- 格式器 (Formatters) ---
    "formatters": {
        "standard": {
            # 格式：时间 | 日志级别 | 模块名:行号 | 消息
            "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            # 生产环境常用，方便 ELK 等工具采集 (需要 python-json-logger 库，这里仅作示意)
            "format": "%(asctime)s %(levelname)s %(message)s",
        },
    },

    # --- 处理器 (Handlers) ---
    "handlers": {
        # 控制台输出
        "console": {
            "level": "DEBUG" if settings.DEBUG else "INFO",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "standard",
        },
        # 文件输出 (按大小轮转，防止日志文件无限增长)
        "file": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "automcp.log"),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,              # 保留 5 个备份
            "formatter": "standard",
            "encoding": "utf-8",
        },
        # 模型执行结果专用日志
        "execution_file": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "execution.log"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "formatter": "standard",
            "encoding": "utf-8",
        },
    },

    # --- 日志器 (Loggers) ---
    "loggers": {
        # 根日志器 (Root Logger)：捕获所有未被特定 logger 捕获的日志
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        },
        # 项目专属 logger：可以设置更详细的级别
        "src": {  # 假设您的代码都在 src 包下
            "handlers": ["console", "file"],
            "level": "DEBUG" if settings.DEBUG else "INFO",
            "propagate": False,
        },
        # 专门用于记录模型/任务执行结果
        "execution": {
            "handlers": ["console", "execution_file"],
            "level": "INFO",
            "propagate": False,
        },
        # 第三方库降噪：例如把 http 库的繁琐日志屏蔽掉
        "httpx": {
            "handlers": ["console"],
            "level": "WARNING",
        },
    },
}

# 3. 初始化函数
import logging.config

def setup_logging():
    """
    应用日志配置
    通常在 main.py 启动时最先调用
    """
    logging.config.dictConfig(LOGGING_CONFIG)