"""状态定义模块 - W={X, Y, A, K, G, Z, H} 元组"""
from typing import TypedDict, List, Dict, Optional, Any


class MGWorkflowState(TypedDict):
    """微电网多场景工作流状态定义

    W 元组元素说明（对应论文公式(1)）:
    - X: user_input (自然语言建模描述)
    - Y: 最终输出 (M数学模型 + C可执行代码 + S求解结果 + R综合报告)
    - A: agents集合 {AKG, A1, A2, A3}
      - AKG: Knowledge Graph Orchestration Agent
      - A1: Task Interpreter Agent
      - A2: Mathematical Modeling Agent
      - A3: Code Synthesizer Agent
    - K: agent-specific knowledge graph集合 {K1, K2, K3}
      - K1: Task Interpretation Knowledge Graph
      - K2: Mathematical Modeling Knowledge Graph
      - K3: Code Synthesis Knowledge Graph
    - G: 工作流编排图 (LangGraph)
    - Z: 中间状态集合 {Z1, Z2}
      - Z1: 结构化规格说明 (scenario_requirement)
      - Z2: MEMG数学模型 (math_model)
    - H: 人工干预机制 (human_intervention)
    """

    # ===== X: 自然语言输入 =====
    user_input: str

    # ===== Z: 中间状态 =====
    scenario_requirement: Optional[Dict[str, Any]]  # Z1: 结构化场景需求
    math_model: Optional[Dict[str, Any]]            # Z2: 数学模型

    # ===== Y: 最终输出 =====
    generated_code: Optional[str]                   # C: Python代码
    final_report: Optional[str]                     # R: 完整报告
    solution_results: Optional[Dict[str, Any]]      # S: 求解结果

    # ===== K: Agent-Specific Knowledge Graphs =====
    # 论文创新：通过AKG生成三个专用知识图谱
    K1_task_interpretation: Optional[Dict[str, Any]]    # K1: 任务解释知识图谱
    K2_math_modeling: Optional[Dict[str, Any]]          # K2: 数学建模知识图谱
    K3_code_synthesis: Optional[Dict[str, Any]]         # K3: 代码合成知识图谱

    # 原有的统一KG上下文（用于向后兼容）
    knowledge_graph_context: Optional[str]          # 多路径KG检索结果
    kg_retrieval_result: Optional[Dict[str, Any]]   # KG检索详细结果（含路径信息）
    scenario_type: Optional[str]                    # 识别到的场景类型

    # AKG编排结果
    akg_orchestration_result: Optional[Dict[str, Any]]  # AKG编排日志和元数据

    # ===== 验证状态 =====
    scenario_valid: bool                            # Z1 验证是否通过
    model_valid: bool                              # Z2 验证是否通过
    code_valid: bool                               # C 验证是否通过

    # ===== 执行状态 =====
    current_step: int
    total_steps: int
    step_status: List[str]  # ["pending", "running", "completed", "error"]

    # ===== 自我审查 =====
    self_review: Dict[str, str]

    # ===== 日志和错误 =====
    execution_log: List[Dict[str, Any]]
    errors: List[str]

    # ===== 元数据 =====
    timestamp: str
    session_id: str
