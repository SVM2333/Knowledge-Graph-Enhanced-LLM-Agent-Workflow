"""LangGraph工作流构建器 - 实现 W={X,Y,A,K,G,Z,H} 元组

知识图谱增强的LLM-Agent工作流:
1. 多路径知识检索: 候选路径检索 -> 语义编码 -> 聚类选择 -> 知识上下文
2. 多Agent协作: A1(任务解释) -> A2(数学建模) -> A3(代码合成)
3. LangGraph全局编排与反馈纠正机制
"""

import time
import uuid
from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from workflow.state_schema import MGWorkflowState
from config.settings import Config


class WorkflowBuilder:
    """工作流构建器 - 基于LangGraph的全局编排与验证机制

    工作流程对应论文框架 W={X,Y,A,K,G,Z,H}:
    - X: 自然语言建模描述输入
    - A={AKG, A1, A2, A3}: 知识图谱编排Agent和三个建模Agent
    - K={K1, K2, K3}: Agent专用知识图谱
    - G: LangGraph工作流编排图
    - Z={Z1, Z2}: 中间产物(Z1=结构化规范, Z2=数学模型)
    - H: 人工干预机制
    - Y: 最终输出(数学模型M + 可执行代码C + 求解结果S + 报告R)

    执行流程: X -> K(多路径检索) -> A1 -> Z1 -> 验证 -> A2 -> Z2 -> 验证 -> A3 -> C -> 验证 -> Y
    各阶段均可通过 H (人工干预) 进行修正
    """

    def __init__(self):
        """初始化工作流"""
        self.config = Config()
        self.kg_cache: Dict[str, Any] = {}

    def create_workflow(self) -> StateGraph:
        """创建工作流图"""
        workflow = StateGraph(MGWorkflowState)

        # ========== 添加节点 ==========
        # K: 知识图谱多路径检索
        workflow.add_node("kg_multipath_retrieval", self._kg_retrieval_node)
        # A1: 场景解释器
        workflow.add_node("scenario_interpreter", self._scenario_interpreter_node)
        # Z1 验证: 场景验证器
        workflow.add_node("scenario_validator", self._scenario_validator_node)
        # A2: 数学建模Agent
        workflow.add_node("math_modeler", self._math_modeler_node)
        # Z2 验证: 模型验证器
        workflow.add_node("model_validator", self._model_validator_node)
        # A3: 代码合成Agent
        workflow.add_node("code_synthesizer", self._code_synthesizer_node)
        # C 验证: 代码验证器
        workflow.add_node("code_validator", self._code_validator_node)
        # H: 人工干预
        workflow.add_node("human_intervention", self._human_intervention_node)

        # ========== 设置入口点 ==========
        workflow.set_entry_point("kg_multipath_retrieval")

        # ========== 定义边（工作流编排图 G）==========
        # K -> A1
        workflow.add_edge("kg_multipath_retrieval", "scenario_interpreter")

        # A1 -> Z1 验证
        workflow.add_edge("scenario_interpreter", "scenario_validator")

        # Z1 验证 -> A2 / H
        workflow.add_conditional_edges(
            "scenario_validator",
            self._route_after_scenario_validation,
            {
                "math_modeler": "math_modeler",
                "human_intervention": "human_intervention"
            }
        )

        # A2 -> Z2 验证
        workflow.add_edge("math_modeler", "model_validator")

        # Z2 验证 -> A3 / H
        workflow.add_conditional_edges(
            "model_validator",
            self._route_after_model_validation,
            {
                "code_synthesizer": "code_synthesizer",
                "human_intervention": "human_intervention"
            }
        )

        # A3 -> C 验证
        workflow.add_edge("code_synthesizer", "code_validator")

        # C 验证 -> Y(END) / H
        workflow.add_conditional_edges(
            "code_validator",
            self._route_after_code_validation,
            {
                "END": END,
                "human_intervention": "human_intervention"
            }
        )

        # H -> K (重新开始或返回A1)
        workflow.add_edge("human_intervention", "kg_multipath_retrieval")

        return workflow.compile()

    # ==================== 路由函数 ====================

    def _route_after_scenario_validation(
        self,
        state: MGWorkflowState
    ) -> Literal["math_modeler", "human_intervention"]:
        """场景验证后的路由"""
        if state.get("scenario_valid", False):
            return "math_modeler"
        return "human_intervention"

    def _route_after_model_validation(
        self,
        state: MGWorkflowState
    ) -> Literal["code_synthesizer", "human_intervention"]:
        """模型验证后的路由"""
        if state.get("model_valid", False):
            return "code_synthesizer"
        return "human_intervention"

    def _route_after_code_validation(
        self,
        state: MGWorkflowState
    ) -> Literal["END", "human_intervention"]:
        """代码验证后的路由"""
        if state.get("code_valid", False):
            return "END"
        return "human_intervention"

    # ==================== 节点实现 ====================

    def _kg_retrieval_node(self, state: MGWorkflowState) -> MGWorkflowState:
        """
        K: 知识图谱多路径检索节点

        实现论文中的多路径知识检索机制:
        1. 候选路径检索 (candidate path retrieval)
        2. 语义编码 (semantic encoding via transformer)
        3. 聚类选择 (clustering-based selection)
        """
        step_idx = 0
        state["step_status"][step_idx] = "running"
        state["current_step"] = 1

        try:
            from knowledge_graph.mg_kg import get_knowledge_graph

            kg = get_knowledge_graph()
            user_input = state["user_input"]

            # 多路径检索（核心创新：语义编码+聚类选择）
            retrieval_result = kg.multi_path_retrieve(
                query=user_input,
                scenario_type=None,
                encode=True,
                max_candidates=Config.MAX_CANDIDATE_PATHS,
                top_k=Config.PATH_SELECT_TOP_K,
                max_hops=3
            )

            # 构建知识上下文字符串
            kg_context = kg.build_kg_context(
                query=user_input,
                top_k=Config.PATH_SELECT_TOP_K,
                include_all_candidates=False
            )

            state["knowledge_graph_context"] = kg_context
            state["kg_retrieval_result"] = retrieval_result

            state["step_status"][step_idx] = "completed"
            log_entry = {
                "step": 1,
                "agent": "知识图谱多路径检索",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "retrieval_stats": {
                    "total_candidates": len(retrieval_result.get("all_candidates", [])),
                    "selected_paths": len(retrieval_result.get("paths", [])),
                    "num_clusters": retrieval_result.get("num_clusters", 0),
                    "retrieval_time": round(retrieval_result.get("retrieval_time", 0), 3),
                    "encoding_time": round(retrieval_result.get("encoding_time", 0), 3),
                    "total_time": round(retrieval_result.get("total_time", 0), 3),
                }
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][step_idx] = "error"
            state["errors"].append(f"知识图谱多路径检索错误: {str(e)}")
            state["knowledge_graph_context"] = "知识图谱检索失败，使用通用建模知识。"
            log_entry = {
                "step": 1,
                "agent": "知识图谱多路径检索",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)

        return state

    def _scenario_interpreter_node(self, state: MGWorkflowState) -> MGWorkflowState:
        """A1: 场景解释器节点"""
        step_idx = 1
        state["step_status"][step_idx] = "running"
        state["current_step"] = 2

        try:
            from agents.agent1_scenario_interpreter import ScenarioInterpreterAgent
            agent = ScenarioInterpreterAgent()

            kg_context = state.get("knowledge_graph_context", "")
            user_input = state["user_input"]

            combined_input = {
                "user_input": user_input,
                "knowledge_graph_context": kg_context
            }

            result = agent.run(combined_input)
            state["scenario_requirement"] = result["output"]
            state["scenario_type"] = result["output"].get("scenario_type")
            state["self_review"]["agent1"] = result.get("self_review", "")

            state["step_status"][step_idx] = "completed"
            log_entry = {
                "step": 2,
                "agent": "场景解释器 (A1)",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenario_type": state["scenario_type"],
                "execution_time": result["metadata"]["execution_time"]
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][step_idx] = "error"
            state["errors"].append(f"场景解释器错误: {str(e)}")
            log_entry = {
                "step": 2,
                "agent": "场景解释器 (A1)",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)
            raise

        return state

    def _scenario_validator_node(self, state: MGWorkflowState) -> MGWorkflowState:
        """Z1 验证节点 - 验证结构化场景需求的完整性"""
        step_idx = 2
        state["step_status"][step_idx] = "running"
        state["current_step"] = 3

        try:
            req = state.get("scenario_requirement", {})
            validation_errors = []

            # 检查必需字段
            required_fields = ["scenario_type", "devices", "loads", "objectives", "constraints"]
            for field in required_fields:
                if field not in req or not req[field]:
                    validation_errors.append(f"缺少必需字段: {field}")

            # 检查设备-负荷匹配
            devices = req.get("devices", [])
            loads = req.get("loads", [])
            if devices and not loads:
                validation_errors.append("识别到设备但未识别到负荷")
            if loads and not devices:
                validation_errors.append("识别到负荷但未识别到设备")

            # 检查置信度
            confidence = req.get("scenario_confidence", 0.0)
            if confidence < 0.3:
                validation_errors.append(f"场景识别置信度过低: {confidence}")

            state["scenario_valid"] = len(validation_errors) == 0
            state["step_status"][step_idx] = "completed"

            log_entry = {
                "step": 3,
                "agent": "场景验证器",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "valid": state["scenario_valid"],
                "errors": validation_errors
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][step_idx] = "error"
            state["errors"].append(f"场景验证错误: {str(e)}")
            state["scenario_valid"] = False
            log_entry = {
                "step": 3,
                "agent": "场景验证器",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)

        return state

    def _math_modeler_node(self, state: MGWorkflowState) -> MGWorkflowState:
        """A2: 数学建模节点 Z1 + K -> Z2"""
        step_idx = 3
        state["step_status"][step_idx] = "running"
        state["current_step"] = 4

        try:
            from agents.agent2_math_modeler import MathModelerAgent
            agent = MathModelerAgent()

            scenario_req = state.get("scenario_requirement", {})
            kg_context = state.get("knowledge_graph_context", "")
            scenario_type = state.get("scenario_type", "")

            combined_input = {
                "scenario_requirement": scenario_req,
                "knowledge_graph_context": kg_context,
                "scenario_type": scenario_type
            }

            result = agent.run(combined_input)
            state["math_model"] = result["output"]
            state["self_review"]["agent2"] = result.get("self_review", "")

            state["step_status"][step_idx] = "completed"
            log_entry = {
                "step": 4,
                "agent": "数学建模Agent (A2)",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenario_type": scenario_type,
                "execution_time": result["metadata"]["execution_time"]
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][step_idx] = "error"
            state["errors"].append(f"数学建模错误: {str(e)}")
            log_entry = {
                "step": 4,
                "agent": "数学建模Agent (A2)",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)
            raise

        return state

    def _model_validator_node(self, state: MGWorkflowState) -> MGWorkflowState:
        """Z2 验证节点 - 验证数学模型的正确性"""
        step_idx = 4
        state["step_status"][step_idx] = "running"
        state["current_step"] = 5

        try:
            model = state.get("math_model", {})
            validation_errors = []

            # 检查顶层结构
            required_keys = ["model_name", "decision_variables", "objective_function", "constraints"]
            for key in required_keys:
                if key not in model:
                    validation_errors.append(f"缺少顶层字段: {key}")

            # 检查目标函数
            obj = model.get("objective_function", {})
            if not obj.get("expression"):
                validation_errors.append("目标函数表达式为空")

            # 检查约束完整性
            constraints = model.get("constraints", {})
            if not constraints:
                validation_errors.append("约束条件为空")

            state["model_valid"] = len(validation_errors) == 0
            state["step_status"][step_idx] = "completed"

            log_entry = {
                "step": 5,
                "agent": "模型验证器",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "valid": state["model_valid"],
                "errors": validation_errors
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][step_idx] = "error"
            state["errors"].append(f"模型验证错误: {str(e)}")
            state["model_valid"] = False
            log_entry = {
                "step": 5,
                "agent": "模型验证器",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)

        return state

    def _code_synthesizer_node(self, state: MGWorkflowState) -> MGWorkflowState:
        """A3: 代码合成节点 Z2 -> C"""
        step_idx = 5
        state["step_status"][step_idx] = "running"
        state["current_step"] = 6

        try:
            from agents.agent3_code_synthesizer import CodeSynthesizerAgent
            agent = CodeSynthesizerAgent()

            math_model = state.get("math_model", {})
            scenario_type = state.get("scenario_type", "")
            kg_context = state.get("knowledge_graph_context", "")

            combined_input = {
                "math_model": math_model,
                "scenario_type": scenario_type,
                "kg_context": kg_context
            }

            result = agent.run(combined_input)
            state["generated_code"] = result["output"]
            state["self_review"]["agent3"] = result.get("self_review", "")

            state["step_status"][step_idx] = "completed"
            log_entry = {
                "step": 6,
                "agent": "代码合成Agent (A3)",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenario_type": scenario_type,
                "execution_time": result["metadata"]["execution_time"]
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][step_idx] = "error"
            state["errors"].append(f"代码合成错误: {str(e)}")
            log_entry = {
                "step": 6,
                "agent": "代码合成Agent (A3)",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)
            raise

        return state

    def _code_validator_node(self, state: MGWorkflowState) -> MGWorkflowState:
        """C 验证节点 - 验证生成的Python代码"""
        step_idx = 6
        state["step_status"][step_idx] = "running"
        state["current_step"] = 7

        try:
            code = state.get("generated_code", "")
            math_model = state.get("math_model", {})
            validation_errors = []

            # ---- 1. 静态语法检查 ----
            required_elements = ["import", "class", "def"]
            for elem in required_elements:
                if elem not in code:
                    validation_errors.append(f"缺少必要元素: {elem}")

            open_braces = code.count("{") - code.count("}")
            open_parens = code.count("(") - code.count(")")
            open_brackets = code.count("[") - code.count("]")

            if abs(open_braces) > 0:
                validation_errors.append(f"大括号不匹配: 差{open_braces}个")
            if abs(open_parens) > 0:
                validation_errors.append(f"小括号不匹配: 差{open_parens}个")
            if abs(open_brackets) > 0:
                validation_errors.append(f"中括号不匹配: 差{open_brackets}个")

            if "Error" in code or "TODO" in code.split("\n")[-5:]:
                validation_errors.append("代码可能包含未完成的标记")

            # ---- 2. 变量一致性检查 ----
            if math_model:
                model_vars = []
                dvars = math_model.get("decision_variables", {})
                for var_group in dvars.values():
                    if isinstance(var_group, list):
                        model_vars.extend(var_group)
                    elif isinstance(var_group, str):
                        model_vars.append(var_group)

                if model_vars:
                    # 提取代码中的变量名（简单启发式）
                    code_lower = code.lower()
                    missing_vars = []
                    for var in model_vars:
                        if isinstance(var, str):
                            var_lower = var.lower()
                            # 检查是否以 P_、H_、C_ 等常见前缀出现
                            if var_lower.startswith("p_") or var_lower.startswith("h_") or var_lower.startswith("c_"):
                                if var_lower not in code_lower and var.lower() not in code_lower:
                                    missing_vars.append(var)

                    if len(missing_vars) > len(model_vars) * 0.5:
                        validation_errors.append(f"模型定义的决策变量在代码中出现率过低: {missing_vars[:3]}...")

            # ---- 3. 实际运行代码（编译 + 导入检查）----
            runtime_error = self._try_run_code(code)
            if runtime_error:
                validation_errors.append(f"代码运行错误: {runtime_error}")

            state["code_valid"] = len(validation_errors) == 0
            state["step_status"][step_idx] = "completed"

            log_entry = {
                "step": 7,
                "agent": "代码验证器",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "valid": state["code_valid"],
                "errors": validation_errors
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][step_idx] = "error"
            state["errors"].append(f"代码验证错误: {str(e)}")
            state["code_valid"] = False
            log_entry = {
                "step": 7,
                "agent": "代码验证器",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)

        return state

    def _try_run_code(self, code: str) -> Optional[str]:
        """
        尝试编译和运行代码，返回错误信息（如果有）
        仅执行 import + class 定义阶段，不实际求解
        """
        if not code or not code.strip():
            return "代码为空"

        import sys
        import io
        import traceback

        # 保存原始 stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # 编译检查
            try:
                compiled = compile(code, "<generated_code>", "exec")
            except SyntaxError as e:
                return f"语法错误 (行{e.lineno}): {e.msg}"

            # 只 import 阶段，不执行求解
            namespace: Dict[str, Any] = {"__name__": "__test__"}
            exec(compiled, namespace)

            # 检查是否定义了预期的类名
            expected_classes = ["MGOptimizer", "ScenarioData", "MicroGridOptimizer"]
            found_classes = [c for c in expected_classes if c in namespace]
            if not found_classes:
                return "未找到预期的优化器类（MGOptimizer / ScenarioData）"

            return None  # 无错误

        except ImportError as e:
            return f"导入错误（缺少依赖）: {str(e)}"
        except NameError as e:
            return f"变量未定义: {str(e)}"
        except TypeError as e:
            return f"类型错误: {str(e)}"
        except Exception as e:
            return f"运行时错误: {str(e)}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _human_intervention_node(self, state: MGWorkflowState) -> MGWorkflowState:
        """H: 人工干预节点"""
        step_idx = 7
        state["step_status"][step_idx] = "running"
        state["current_step"] = 8

        log_entry = {
            "step": 8,
            "agent": "人工干预 (H)",
            "status": "waiting",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "errors": state.get("errors", []),
            "note": "检测到自动验证失败，需要人工审查和修正"
        }
        state["execution_log"].append(log_entry)

        # 保留错误信息以便用户查看
        state["step_status"][step_idx] = "completed"
        return state

    def create_initial_state(self, user_input: str) -> MGWorkflowState:
        """创建初始状态"""
        return {
            "user_input": user_input,
            "scenario_requirement": None,
            "math_model": None,
            "generated_code": None,
            "final_report": None,
            "knowledge_graph_context": None,
            "kg_retrieval_result": None,
            "scenario_type": None,
            "scenario_valid": False,
            "model_valid": False,
            "code_valid": False,
            "current_step": 0,
            "total_steps": 8,
            "step_status": ["pending"] * 8,
            "self_review": {},
            "execution_log": [],
            "errors": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": str(uuid.uuid4())
        }
