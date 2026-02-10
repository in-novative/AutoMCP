import logging
import ast
import sys
import io
from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.state import AgentState
from src.server.models import TaskStep, TaskStatus, AgentMessage, TaskCategory

logger = logging.getLogger(__name__)

# --- 1. Prompt 定义 ---
CODE_GEN_PROMPT = """
你是一个 Python 代码生成专家。
请编写一段高质量的、生产级的 Python 代码来解决以下任务。

任务目标: {description}
具体要求: {requirements}

## 核心约束
1. **纯代码输出**: 严禁包含 ```python 或 ``` 标记，严禁包含任何非代码的解释性文字。
2. **输出规范**: 必须将最终结果打印到标准输出 (print)，或者定义一个名为 `solve` 的无参函数并返回结果。
3. **依赖管理**: 优先使用 Python 标准库。如果必须使用第三方库，请确保它是 `requests`, `pandas`, `numpy` 等主流库，并在代码开头注释说明。
4. **健壮性**: 必须包含基本的错误处理 (try-except)，防止脚本因意外输入而直接崩溃。
5. **安全性**: 严禁编写任何具有破坏性的代码（如删除系统文件、无限循环、反向 Shell）。
"""

# --- 2. 沙箱执行环境 (抽象层) ---
class SandboxExecutor:
    """
    负责在安全隔离的环境中执行代码
    """
    def __init__(self, sandbox_type: str = "local_exec"):
        self.sandbox_type = sandbox_type

    async def run(self, code: str, timeout: int = 30) -> str:
        """
        执行代码并返回 stdout 输出
        """
        if self.sandbox_type == "docker":
            return await self._run_in_docker(code, timeout)
        elif self.sandbox_type == "gvisor":
            return await self._run_in_gvisor(code, timeout)
        else:
            return await self._run_local_unsafe(code, timeout)

    async def _run_in_docker(self, code: str, timeout: int) -> str:
        # TODO: Implement Docker container execution
        # 1. Write code to temp file
        # 2. docker run --rm -v ... python:3.10 python script.py
        # 3. Capture logs
        return "Docker sandbox not implemented yet."

    async def _run_in_gvisor(self, code: str, timeout: int) -> str:
        # TODO: Implement gVisor (runsc) execution
        return "gVisor sandbox not implemented yet."

    async def _run_local_unsafe(self, code: str, timeout: int) -> str:
        """
        [警告] 仅用于开发测试的本地执行，极不安全！
        """
        logger.warning("Running code in UNSAFE local mode!")
        
        # 捕获 stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            # 使用 ast.parse 检查语法错误
            ast.parse(code)
            
            # 创建独立的全局命名空间
            local_scope = {}
            exec(code, {}, local_scope)
            
            # 检查是否有 solve 函数
            if "solve" in local_scope and callable(local_scope["solve"]):
                result = local_scope["solve"]()
                if result is not None:
                    print(result) # 确保返回值也能被捕获
            
            output = redirected_output.getvalue()
            return output if output.strip() else "Code executed successfully (no output)."
            
        except Exception as e:
            return f"Execution Error: {str(e)}"
        finally:
            sys.stdout = old_stdout

# --- 3. Code Generator Node ---
async def code_generator_node(state: AgentState):
    """
    LangGraph 节点：代码生成器
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    
    if not plan or idx >= len(plan):
        return {"messages": [AgentMessage(role="system", content="Index out of range")]}
        
    current_step: TaskStep = plan[idx]
    
    # 类型检查
    if current_step.task_type != TaskCategory.CODE_TO_MCP:
        logger.warning(f"CodeGenerator received wrong task type: {current_step.task_type}")
    
    current_step.status = TaskStatus.RUNNING
    logger.info(f"Generating code for: {current_step.description}")
    
    try:
        # 1. 生成代码
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        chain = ChatPromptTemplate.from_template(CODE_GEN_PROMPT) | llm | StrOutputParser()
        
        generated_code = await chain.ainvoke({
            "description": current_step.description,
            "requirements": "\n".join(current_step.requirements)
        })
        
        # 清洗代码 (移除可能的 markdown 标记)
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        logger.debug(f"Generated Code:\n{generated_code}")
        
        # 2. 执行代码
        # TODO: Load sandbox config from settings
        sandbox = SandboxExecutor(sandbox_type="local_exec") 
        execution_result = await sandbox.run(generated_code)
        
        # 3. 更新状态
        current_step.result = execution_result
        current_step.status = TaskStatus.COMPLETED
        # 保存生成的代码到 tool_args 以便后续查看或 Crystallize
        current_step.tool_args = {"code": generated_code} 
        
        return {
            "current_step_index": idx + 1,
            "messages": [AgentMessage(role="assistant", content=f"Code Generated & Executed.\nResult: {execution_result[:200]}...")]
        }
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        current_step.status = TaskStatus.FAILED
        current_step.error = str(e)
        return {
            "messages": [AgentMessage(role="system", content=f"Code Error: {str(e)}")]
        }