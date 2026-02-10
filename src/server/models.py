from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# --- 1. 基础枚举 ---
class TaskStatus(str, Enum):
    """任务执行状态"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 正在执行
    COMPLETED = "completed"   # 执行成功
    FAILED = "failed"         # 执行失败
    SKIPPED = "skipped"       # 跳过

class AgentRole(str, Enum):
    """消息发送者角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"             # 工具执行结果

# --- 2. 消息模型 (兼容 OpenAI 格式) ---
class AgentMessage(BaseModel):
    """
    Agent 通信的基础消息单元
    兼容 LangChain/OpenAI 的 message 格式
    """
    role: AgentRole
    content: str
    name: Optional[str] = None  # 可选：发送者名称或工具名称
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # 额外元数据 (如 token usage, processing time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- 3. 任务规划模型 (Planner 输出) ---
class TaskStep(BaseModel):
    """
    单个执行步骤 (Sub Task)
    由 Planner 生成，Classifier 分类，Executor 执行
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: str = Field(..., description="执行该任务的角色，如 coder, planner, reviewer")
    description: str = Field(..., description="准确精简的任务描述")
    requirements: List[str] = Field(default_factory=list, description="任务的具体要求，如使用语言、约束条件等")
    
    # 分类器输出
    task_type: Optional[str] = Field(None, description="任务类型: local_mcp, web_mcp, code_to_mcp, llm")
    
    # 执行状态
    tool_name: Optional[str] = Field(None, description="实际使用的工具名称")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="实际使用的工具参数")
    
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None     # 执行结果摘要
    error: Optional[str] = None      # 错误信息
    
    # 反思控制
    retry_count: int = Field(0, description="当前重试/反思次数")
    max_retries: int = Field(3, description="最大允许重试次数")
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ExecutionPlan(BaseModel):
    """
    完整的执行计划
    包含多个有序的 TaskStep
    """
    task: str = Field(..., description="用户的原始任务")
    steps: List[TaskStep] = Field(default_factory=list)

    # 反思控制
    retry_count: int = Field(0, description="当前重试/反思次数")
    max_retries: int = Field(2, description="最大允许重试次数")
    
    # 计划层面的状态
    status: TaskStatus = TaskStatus.PENDING
    summary: Optional[str] = None  # 执行后的总结

# --- 4. 运行结果模型 ---
class ToolResult(BaseModel):
    """工具执行的标准化返回"""
    tool_name: str
    success: bool
    output: Any            # 工具的原始输出 (可能是 dict, str, list)
    error_message: Optional[str] = None