"""Agent 2: 数学建模Agent - 根据结构化需求和知识图谱生成数学模型"""
import json
import re
from typing import Dict, Any
from agents.base_agent import BaseAgent


class MathModelerAgent(BaseAgent):
    """数学建模Agent - 将结构化场景需求转换为完整的数学模型"""

    def __init__(self):
        super().__init__("agent2_math_modeler")

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        执行数学建模

        Args:
            input_data: {
                "scenario_requirement": Dict,    # Agent1的结构化场景需求
                "knowledge_graph_context": str,   # 多路径KG检索结果
                "scenario_type": str             # 场景类型
            }

        Returns:
            包含数学模型的字典
        """
        if isinstance(input_data, dict):
            scenario_req = input_data.get("scenario_requirement", {})
            kg_context = input_data.get("knowledge_graph_context", "")
            scenario_type = input_data.get("scenario_type", "")
        else:
            raise ValueError("输入必须是字典类型")

        if not scenario_req:
            raise ValueError("场景需求不能为空")

        # 构建提示词
        prompt_input = self._build_prompt(scenario_req, kg_context, scenario_type)

        # 调用LLM
        response = self.call_llm(prompt_input, temperature=0.4)

        # 解析JSON输出
        math_model = self._parse_json_response(response)

        # 验证输出
        if not self.validate_output(math_model):
            raise ValueError("输出验证失败：数学模型格式不正确")

        return {
            "success": True,
            "output": math_model,
            "raw_response": response,
            "self_review": self._perform_self_review(math_model)
        }

    def _build_prompt(
        self,
        scenario_req: Dict,
        kg_context: str,
        scenario_type: str
    ) -> str:
        """构建提示词"""
        scenario_str = json.dumps(scenario_req, ensure_ascii=False, indent=2)
        prompt = f"""请将以下结构化的微电网场景需求转换为完整的数学模型。

场景类型: {scenario_type}

场景需求：
{scenario_str}
"""
        if kg_context:
            prompt += f"""
========================================
知识图谱多路径检索结果（建模知识参考）：
========================================
{kg_context}
========================================
请结合上述知识图谱中的设备建模知识、目标函数模板和约束条件模板，
生成与场景完全匹配的数学模型。
"""
        else:
            prompt += """
========================================
注意：当前未使用本地知识图谱，请基于您的通用数学建模知识。
请参考标准的微电网数学建模方法：
- 决策变量：设备出力、储能充放电功率、网络潮流等
- 目标函数：成本最小化 = Σ(燃料成本 + 运维成本 + 购电成本 - 售电收益)
- 约束条件：
  * 功率平衡约束：Σ发电 = Σ负荷 + Σ损耗
  * 设备容量约束：P_min ≤ P ≤ P_max
  * 爬坡约束：|P_t - P_{t-1}| ≤ R_max
  * 储能SOC约束：SOC_min ≤ SOC ≤ SOC_max
  * 网络约束：支路功率限制、电压约束等
========================================
"""
        prompt += "\n\n请严格按照JSON格式输出完整的数学模型，不要添加任何其他文字说明。"
        return prompt

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """从响应中提取JSON"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从响应中解析JSON: {response[:300]}")

    def _perform_self_review(self, model: Dict[str, Any]) -> str:
        """执行自我审查"""
        issues = []

        required_keys = ["model_name", "decision_variables", "objective_function", "constraints"]
        for key in required_keys:
            if key not in model:
                issues.append(f"缺少顶层字段: {key}")

        # 检查目标函数
        obj = model.get("objective_function", {})
        if not obj.get("expression"):
            issues.append("目标函数表达式为空")

        # 检查约束
        constraints = model.get("constraints", {})
        if not constraints:
            issues.append("约束条件为空")
        elif "power_balance" not in constraints:
            issues.append("缺少功率平衡约束")

        if issues:
            return f"审查发现问题: {'; '.join(issues)}"
        return "审查通过：数学模型结构完整且正确"

    def validate_output(self, output: Any) -> bool:
        """验证数学模型输出"""
        if not isinstance(output, dict):
            return False

        required_keys = ["model_name", "decision_variables", "objective_function", "constraints"]
        if not all(key in output for key in required_keys):
            return False

        if not output.get("objective_function", {}).get("expression"):
            return False

        return True
