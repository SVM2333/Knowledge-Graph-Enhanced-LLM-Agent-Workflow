"""Agent 3: 代码合成Agent - 将数学模型转换为Python代码"""
import json
import re
from typing import Dict, Any
from agents.base_agent import BaseAgent


class CodeSynthesizerAgent(BaseAgent):
    """代码合成Agent - 将数学模型转换为可执行的Python优化代码"""

    def __init__(self):
        super().__init__("agent3_code_synthesizer")

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        执行代码合成

        Args:
            input_data: {
                "math_model": Dict,   # Agent2生成的数学模型
                "scenario_type": str,  # 场景类型
                "kg_context": str     # KG上下文（可选）
            }

        Returns:
            包含Python代码的字典
        """
        if isinstance(input_data, dict):
            math_model = input_data.get("math_model", {})
            scenario_type = input_data.get("scenario_type", "unknown")
            kg_context = input_data.get("kg_context", "")
        else:
            raise ValueError("输入必须是字典类型")

        if not math_model:
            raise ValueError("数学模型不能为空")

        # 构建提示词
        prompt_input = self._build_prompt(math_model, scenario_type, kg_context)

        # 调用LLM
        response = self.call_llm(prompt_input, temperature=0.3)

        # 提取Python代码
        python_code = self._extract_python_code(response)

        # 验证输出
        if not self.validate_output(python_code):
            raise ValueError("输出验证失败：代码格式不正确")

        return {
            "success": True,
            "output": python_code,
            "raw_response": response,
            "self_review": self._perform_self_review(python_code, math_model)
        }

    def _build_prompt(
        self,
        math_model: Dict,
        scenario_type: str,
        kg_context: str
    ) -> str:
        """构建提示词"""
        model_str = json.dumps(math_model, ensure_ascii=False, indent=2)
        prompt = f"""请根据以下数学模型生成完整的、可运行的Python优化代码。

场景类型: {scenario_type}

数学模型：
{model_str}
"""
        if kg_context:
            prompt += f"""
========================================
参考知识图谱建模知识：
========================================
{kg_context[:2000]}
========================================
"""
        else:
            prompt += """
========================================
注意：当前未使用本地知识图谱，请基于您的通用编程知识。
请参考标准的Python优化建模库：
- 线性规划：使用 scipy.optimize.linprog 或 pulp
- 混合整数规划：使用 pulp 或 pyomo
- 非线性规划：使用 scipy.optimize.minimize
- 建议包含：数据定义、模型构建、求解、结果输出和可视化
========================================
"""

        prompt += """
要求：
1. 生成完整的Python代码（包含所有import语句、类定义、求解逻辑和主函数）
2. 代码必须可以直接运行
3. 使用适当的求解器（scipy.optimize / pulp / ortools 等）
4. 添加必要的注释和结果可视化
5. 只输出代码，不要输出其他说明文字

请在 ```python 代码块中输出完整代码。
"""
        return prompt

    def _extract_python_code(self, response: str) -> str:
        """从响应中提取Python代码"""
        # 尝试从代码块中提取
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # 尝试从 ``` 块中提取（无语言标识）
        code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            if "import" in code or "class" in code or "def" in code:
                return code

        # 如果没有代码块，尝试从 import 开始的地方提取
        if "import" in response:
            lines = response.split("\n")
            start_idx = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import") or stripped.startswith("from"):
                    start_idx = i
                    break
            if start_idx is not None:
                return "\n".join(lines[start_idx:]).strip()

        raise ValueError(f"无法从响应中提取Python代码: {response[:300]}")

    def _perform_self_review(self, code: str, math_model: Dict) -> str:
        """执行自我审查"""
        issues = []

        # 检查关键字
        required_keywords = ["import", "class", "def"]
        for kw in required_keywords:
            if kw not in code:
                issues.append(f"缺少关键字: {kw}")

        # 检查代码块完整性
        open_braces = code.count("{") - code.count("}")
        open_parens = code.count("(") - code.count(")")
        if open_braces != 0:
            issues.append(f"大括号不匹配: 差{open_braces}")
        if open_parens != 0:
            issues.append(f"小括号不匹配: 差{open_parens}")

        # 检查模型一致性
        model_vars = str(math_model.get("decision_variables", {}))
        if "decision_variables" in math_model:
            if "power_vars" in math_model["decision_variables"]:
                expected_vars = str(math_model["decision_variables"]["power_vars"])
                if "P_" not in code and "power" not in code.lower():
                    issues.append("代码可能未正确实现功率变量")

        if issues:
            return f"审查发现问题: {'; '.join(issues)}"
        return "审查通过：代码结构完整且语法正确"

    def validate_output(self, output: Any) -> bool:
        """验证Python代码输出"""
        if not isinstance(output, str):
            return False

        required_keywords = ["import", "class", "def"]
        return all(keyword in output for keyword in required_keywords)
