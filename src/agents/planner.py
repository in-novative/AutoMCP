from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.server.models import TaskStep, ExecutionPlan
from src.workflow.state import AgentState

# 1. 定义输出结构 (Pydantic)
# 我们已经在 models.py 定义了 TaskStep，这里直接作为 LLM 的输出 Schema
# LangChain 的 with_structured_output 方法会自动将其转换为 Function Calling 格式

# 2. 编写 Prompt
PLANNER_SYSTEM_PROMPT = """
你是一个高级任务规划专家 (Planner)。
你的目标是将用户输入的复杂任务拆解为一系列可执行的子任务 (Sub Tasks)。

## 拆解原则
1. **原子性**: 每个子任务应该足够小，可以由单一角色完成。
2. **顺序性**: 子任务之间应有逻辑顺序。
3. **完备性**: 所有子任务的集合必须能完整覆盖用户的原始需求，不可遗漏关键步骤（如测试、文档、环境配置）。
4. **可验证性**: 每个子任务都应包含隐含的验收标准，使得执行者能够判断任务是否成功完成。
5. **最小依赖**: 尽量减少子任务之间的复杂耦合，使得任务可以并行执行或独立重试。
6. **角色分配**: 必须为每个任务指定具体的角色和擅长领域（符合任务需要），例如:
   - "你是一个资深 Python 架构师，专注于高性能后端服务开发。"
   - "你是一个全栈工程师，擅长 React 前端与 Node.js 后端集成。"
   - "你是一个安全审计专家，负责代码漏洞扫描与合规性检查。"
   - "你是一个技术文档专家，擅长编写清晰的用户手册和 API 文档。"
   (请根据任务的具体技术栈和上下文，动态生成最匹配的专业角色描述)

## 输出要求
- 必须严格遵循 JSON 格式。
- 'description' 必须准确、精简，包含具体的行动动词（如"创建"、"分析"、"重构"）。
- 'requirements' 必须列出具体的约束条件（如编程语言版本、依赖库名称、文件路径约定、禁止使用的API等）。
- 确保所有步骤的颗粒度适中，避免出现"完成整个项目"这样的大任务，也不要拆解到"写一行代码"这样的微任务。
- 如果任务涉及代码生成，必须明确指出代码应存放的目录或文件结构。

## 输出示例
```json
{
  "task": "开发一个简单的待办事项 CLI 应用",
  "steps": [
    {
      "role": "你是一个资深 Python 开发者，专注于 CLI 工具设计。",
      "description": "创建项目结构并实现基础的 CLI 命令处理逻辑",
      "requirements": [
        "使用 Click 8.x 库",
        "所有源代码存放在 src/todo_cli 目录下",
        "数据存储实现为 data/tasks.json",
        "实现 add, list, done 三个核心命令"
      ]
    },
    {
      "role": "你是一个 QA 工程师，专注于单元测试。",
      "description": "编写并执行 pytest 测试用例以验证命令功能",
      "requirements": [
        "使用 pytest 7.x",
        "测试文件存放在 tests/ 目录下",
        "确保覆盖所有命令的正常和异常路径",
        "测试覆盖率需达到 80% 以上"
      ]
    }
  ]
}
```
"""

async def planner_node(state: AgentState):
    """
    LangGraph 的规划节点
    """
    messages = state["messages"]
    user_task = messages[-1].content  # 获取用户最后一条指令
    
    # 3. 初始化 LLM 并绑定结构化输出
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(ExecutionPlan) # 强制返回 ExecutionPlan 对象
    
    # 4. 构造调用链
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("user", "{task}")
    ])
    chain = prompt | structured_llm
    
    # 5. 执行推理
    try:
        execution_plan: ExecutionPlan = await chain.ainvoke({"task": user_task})
        
        # 6. 返回更新后的状态
        # 将生成的步骤列表写入 state["plan"]
        return {
            "plan": execution_plan.steps,
            "current_step_index": 0,
            # 可选：将规划结果也作为一条消息存入历史
            "messages": [AgentMessage(role="assistant", content=f"已生成计划，包含 {len(execution_plan.steps)} 个步骤。")] 
        }
        
    except Exception as e:
        # 错误处理逻辑
        return {
            "messages": [AgentMessage(role="system", content=f"规划失败: {str(e)}")]
        }