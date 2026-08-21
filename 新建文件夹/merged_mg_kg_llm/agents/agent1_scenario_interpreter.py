"""Agent 1: 场景解释器 - 解析自然语言场景描述"""
import json
import re
from typing import Dict, Any
from agents.base_agent import BaseAgent


class ScenarioInterpreterAgent(BaseAgent):
    """场景解释器 - 将自然语言转换为结构化场景需求"""

    def __init__(self):
        super().__init__("agent1_scenario_interpreter")

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        执行场景解析

        Args:
            input_data: {
                "user_input": str,           # 用户原始输入
                "knowledge_graph_context": str, # 多路径KG检索结果（可选）
                "scenario_type": str          # 预判场景类型（可选）
            }

        Returns:
            包含结构化场景需求的字典
        """
        if isinstance(input_data, dict):
            user_input = input_data.get("user_input", "")
            kg_context = input_data.get("knowledge_graph_context", "")
        else:
            user_input = str(input_data)
            kg_context = ""

        if not user_input.strip():
            raise ValueError("输入不能为空")

        # 构建完整的提示词输入
        prompt_input = self._build_prompt(user_input, kg_context)

        # 调用LLM
        response = self.call_llm(prompt_input, temperature=0.3)

        # 解析JSON输出
        structured_data = self._parse_json_response(response)

        # 验证输出
        if not self.validate_output(structured_data):
            raise ValueError("输出验证失败：缺少必需字段")

        return {
            "success": True,
            "output": structured_data,
            "raw_response": response,
            "self_review": self._perform_self_review(structured_data, user_input)
        }

    def _build_prompt(self, user_input: str, kg_context: str) -> str:
        """构建提示词"""
        prompt = f"""请将以下微电网场景的自然语言描述转换为结构化的场景需求规范。

用户描述：
{user_input}
"""
        if kg_context:
            prompt += f"""
========================================
知识图谱多路径检索结果（参考知识）：
========================================
{kg_context}
========================================
请结合上述知识图谱检索结果，综合考虑设备、负荷、目标函数和约束条件进行场景解析。
"""
        else:
            prompt += """
========================================
注意：当前未使用本地知识图谱，请基于您的通用知识进行场景解析。
请参考标准的微电网建模知识，包括：
- 常见设备类型：光伏(PV)、风机(WT)、储能(ESS)、柴油发电机(DG)、热泵(HP)、燃气轮机(GT)等
- 负荷类型：电负荷、热负荷、冷负荷、气负荷
- 优化目标：运行成本最小化、碳排放最小化、可再生能源利用率最大化
- 约束条件：功率平衡、设备容量、网络约束、储能SOC等
========================================
"""
        prompt += "\n\n请严格按照JSON格式输出，不要添加任何其他文字说明。"
        return prompt

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """从响应中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试从代码块中提取
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试查找JSON对象
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从响应中解析JSON: {response[:300]}")

    def _perform_self_review(self, output: Dict[str, Any], original_input: str) -> str:
        """执行自我审查"""
        issues = []

        # 检查完整性
        required_fields = ["scenario_type", "devices", "loads", "objectives", "constraints"]
        for field in required_fields:
            if field not in output or not output[field]:
                issues.append(f"缺少字段: {field}")

        # 检查置信度
        confidence = output.get("scenario_confidence", 0.0)
        if confidence < 0.5:
            issues.append(f"置信度过低: {confidence}")

        # 检查设备-负荷匹配
        devices = output.get("devices", [])
        loads = output.get("loads", [])
        if devices and not loads:
            issues.append("识别到设备但未识别到负荷")
        if loads and not devices:
            issues.append("识别到负荷但未识别到设备")

        if issues:
            return f"审查发现问题: {'; '.join(issues)}"
        return "审查通过：场景识别完整且合理"

    def validate_output(self, output: Any) -> bool:
        """验证结构化输出"""
        if not isinstance(output, dict):
            return False

        required_keys = ["scenario_type", "devices", "loads", "objectives", "constraints"]
        return all(key in output for key in required_keys)
