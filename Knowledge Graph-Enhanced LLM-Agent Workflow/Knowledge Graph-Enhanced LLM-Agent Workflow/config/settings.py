"""配置管理模块"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """全局配置类"""

    # API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "z-ai/glm-5")

    # 应用配置
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT = int(os.getenv("TIMEOUT", "120"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # 路径配置
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    KG_DIR = BASE_DIR / "knowledge_graph"
    PROMPTS_FILE = CONFIG_DIR / "prompts.yaml"
    KG_FILE = KG_DIR / "mg_scenarios.yaml"

    # 多路径知识检索配置
    ENCODER_MODEL = os.getenv("ENCODER_MODEL", "sentence-transformers/all-mpnet-base-v2")
    PATH_ENCODER_MODEL = os.getenv("PATH_ENCODER_MODEL", "./encoder/sentence-transformers/all-mpnet-base-v2")
    PATH_CLUSTER_K = int(os.getenv("PATH_CLUSTER_K", "5"))
    MAX_CANDIDATE_PATHS = int(os.getenv("MAX_CANDIDATE_PATHS", "30"))
    PATH_SELECT_TOP_K = int(os.getenv("PATH_SELECT_TOP_K", "10"))
    PATH_ENCODER_BATCH_SIZE = int(os.getenv("PATH_ENCODER_BATCH_SIZE", "64"))
    PATH_ENCODER_MAX_LEN = int(os.getenv("PATH_ENCODER_MAX_LEN", "128"))

    @classmethod
    def load_prompts(cls) -> Dict[str, Any]:
        """加载提示词配置"""
        with open(cls.PROMPTS_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def get_agent_prompt(cls, agent_name: str) -> Dict[str, str]:
        """获取指定Agent的提示词配置"""
        prompts = cls.load_prompts()
        return prompts.get(agent_name, {})

    @classmethod
    def load_knowledge_graph(cls) -> Dict[str, Any]:
        """加载知识图谱，如果文件不存在则返回空图谱结构"""
        if not cls.KG_FILE.exists():
            # 返回空知识图谱结构，允许系统使用LLM通用知识运行
            return {"scenarios": {}, "fallback_mode": True}

        try:
            with open(cls.KG_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                data["fallback_mode"] = False
                return data
        except Exception as e:
            # 文件损坏时也返回空结构
            return {"scenarios": {}, "fallback_mode": True, "error": str(e)}

    @classmethod
    def get_scenario_knowledge(cls, scenario_type: str) -> Dict[str, Any]:
        """获取指定场景类型的知识"""
        kg = cls.load_knowledge_graph()
        scenarios = kg.get("scenarios", {})
        return scenarios.get(scenario_type, {})
