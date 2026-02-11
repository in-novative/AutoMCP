import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

# 1. 动态获取项目根目录
# 这样可以确保数据库路径等配置始终基于项目根路径，而不是运行命令的当前路径
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    应用配置类
    自动读取环境变量和 .env 文件
    """
    
    # --- LLM Provider Config ---
    # 使用 SecretStr 类型，打印日志时会自动脱敏为 '**********'
    # Optional 表示该字段允许为空（例如只配置了 OpenAI 而没配置 Anthropic）
    OPENAI_API_KEY: Optional[SecretStr]
    OPENAI_BASE_URL: str
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    DEFAULT_LLM_MODEL: str

    # --- Classifier Config (小模型配置，用于本地微调) ---
    CLASSIFIER_MODEL: str = "qwen2.5:1.5b"           # 分类器使用的模型
    CLASSIFIER_BASE_URL: str = "http://localhost:11434/v1"  # Ollama 或其他本地服务地址
    CLASSIFIER_API_KEY: str = "ollama"               # 本地服务占位符

    # --- Embedding Config ---
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # 嵌入模型

    # --- Application Config ---
    ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 7879
    SECRET_KEY: SecretStr = SecretStr("unsafe-default-key-change-me")

    # --- Database Config ---
    # 结合 BASE_DIR 构造绝对路径
    CHROMA_DB_PATH: str = str(BASE_DIR / "data" / "chroma")
    SQLITE_DB_PATH: str = str(BASE_DIR / "data" / "automcp.db")

    # --- Reflection Config ---
    MAX_SUBTASK_RETRIES: int = 3   # Level 1 反思重试次数
    MAX_PLAN_RETRIES: int = 2      # Level 2 任务级重规划次数

    # --- Pydantic 配置 ---
    model_config = SettingsConfigDict(
        env_file=".env",          # 指定读取根目录下的 .env 文件
        env_file_encoding="utf-8",
        case_sensitive=True,      # 区分大小写（推荐 True，因为环境变量通常全大写）
        #extra="ignore"            # 忽略 .env 中存在但类中未定义的字段，防止报错
    )

# 2. 实例化并导出
# 实现单例模式，其他模块直接导入这个 settings 对象即可
settings = Settings()