"""Agent基类"""
import json
import time
import re
from typing import Dict, Any, Optional, Iterator
from openai import OpenAI
from config.settings import Config


class BaseAgent:
    """Agent基类"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = Config.get_agent_prompt(agent_name)
        self.name = self.config.get("name", "Unknown Agent")
        self.description = self.config.get("description", "")
        self.system_prompt = self.config.get("system_prompt", "")

        # 检查 API Key 有效性
        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "your_api_key_here":
            raise ValueError(
                "未配置有效的 LLM API Key！\n"
                "请在 merged_mg_kg_llm/.env 文件中填入你的 API Key：\n"
                "  OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx\n"
                "  MODEL_NAME=z-ai/glm-5  （或你的实际模型名）\n\n"
                "参考 .env.example 文件创建 .env"
            )

        self.client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL
        )

        self.execution_count = 0
        self.total_time = 0.0

    def call_llm(
        self,
        user_input: str,
        temperature: float = 0.3,
        max_retries: int = None
    ) -> str:
        """调用LLM（非流式），带重试机制"""
        if max_retries is None:
            max_retries = Config.MAX_RETRIES

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=Config.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=temperature,
                    timeout=Config.TIMEOUT,
                    extra_body={"reasoning": {"enabled": True}}
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
        raise Exception(f"LLM调用失败（尝试{max_retries}次）: {str(last_error)}")

    def call_llm_stream(self, user_input: str, temperature: float = 0.3) -> Iterator[str]:
        """流式调用LLM"""
        self._last_stream_content = ""
        try:
            stream = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=temperature,
                timeout=Config.TIMEOUT,
                stream=True,
                extra_body={"reasoning": {"enabled": True}}
            )

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    if hasattr(delta, 'reasoning') and delta.reasoning:
                        yield delta.reasoning

                    if delta.content:
                        self._last_stream_content += delta.content
                        yield delta.content

        except Exception as e:
            raise Exception(f"LLM流式调用失败: {str(e)}")

    def run(self, input_data: Any, max_retries: int = None) -> Dict[str, Any]:
        """运行Agent（带重试机制）"""
        if max_retries is None:
            max_retries = Config.MAX_RETRIES

        start_time = time.time()
        last_error = None

        for attempt in range(max_retries):
            try:
                result = self.execute(input_data)
                execution_time = time.time() - start_time

                self.execution_count += 1
                self.total_time += execution_time

                result["metadata"] = {
                    "agent_name": self.name,
                    "execution_time": round(execution_time, 2),
                    "attempt": attempt + 1,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"{self.name}执行失败（尝试{max_retries}次）: {str(last_error)}")

    def validate_output(self, output: Any) -> bool:
        """验证输出（子类可重写）"""
        return output is not None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
