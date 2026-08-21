"""Knowledge Graph Orchestration Agent (AKG) - 知识图谱编排Agent

对应论文第III节的Knowledge Graph Orchestration Agent:
- 根据不同LLM-agent的功能职责，生成检索需求
- 调用多路径知识检索模块
- 构建agent-specific知识图谱: K1(任务解释), K2(数学建模), K3(代码合成)
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import time


@dataclass
class AgentKnowledgeConfig:
    """Agent知识配置 - 预定义每个Agent的知识需求"""
    agent_id: str
    agent_name: str
    knowledge_scope: List[str]  # 需要的知识类型
    entity_types: Set[str]      # 关注的实体类型
    retrieval_focus: str        # 检索重点描述


class KnowledgeGraphOrchestrationAgent:
    """
    Knowledge Graph Orchestration Agent (AKG)

    论文对应: Fig. 3 - The operational logic of Knowledge Graph Orchestration Agent AKG

    功能:
    1. Task Definition: 预配置各Agent的知识范围和检索重点
    2. Multi-Path Retrieval: 调用外部多路径检索模块
    3. Agent-Specific KG Generation: 构建K1, K2, K3
    """

    # 预定义的Agent知识配置
    AGENT_CONFIGS = {
        "A1": AgentKnowledgeConfig(
            agent_id="A1",
            agent_name="Task Interpreter Agent",
            knowledge_scope=["semantic", "terminology", "scenario_types"],
            entity_types={"scenario", "device", "load", "objective", "constraint"},
            retrieval_focus="任务解释需要标准化术语、场景类型、设备和负荷分类、优化目标和约束类别"
        ),
        "A2": AgentKnowledgeConfig(
            agent_id="A2",
            agent_name="Mathematical Modeling Agent",
            knowledge_scope=["modeling", "formulations", "constraints"],
            entity_types={"device", "load", "objective", "constraint"},
            retrieval_focus="数学建模需要设备和负荷模型、目标函数、能源网络、耦合关系和运行约束"
        ),
        "A3": AgentKnowledgeConfig(
            agent_id="A3",
            agent_name="Code Synthesizer Agent",
            knowledge_scope=["implementation", "code_templates", "solver_interfaces"],
            entity_types={"device", "objective", "constraint"},
            retrieval_focus="代码合成需要变量映射、代码模板、求解器接口和调试规则"
        )
    }

    def __init__(self, kg_instance=None):
        """
        初始化AKG

        Args:
            kg_instance: MicroGridKnowledgeGraph实例
        """
        self.kg = kg_instance
        if self.kg is None:
            from knowledge_graph.mg_kg import get_knowledge_graph
            self.kg = get_knowledge_graph()

    def orchestrate(
        self,
        user_input: str,
        scenario_type: Optional[str] = None,
        intermediate_artifacts: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        编排知识图谱，生成三个agent-specific KG

        Args:
            user_input: 用户自然语言输入
            scenario_type: 场景类型（可选）
            intermediate_artifacts: 中间制品（如Z1结构化需求），用于动态调整检索

        Returns:
            {
                "K1": Task Interpretation Knowledge Graph,
                "K2": Mathematical Modeling Knowledge Graph,
                "K3": Code Synthesis Knowledge Graph,
                "orchestration_log": 编排日志
            }
        """
        t0 = time.time()
        orchestration_log = []

        # Step 1: 生成三个Agent的检索需求
        retrieval_requirements = self._generate_retrieval_requirements(
            user_input, scenario_type, intermediate_artifacts
        )

        # Step 2: 调用多路径检索模块
        K1_paths = self._retrieve_for_agent("A1", retrieval_requirements["A1"])
        K2_paths = self._retrieve_for_agent("A2", retrieval_requirements["A2"])
        K3_paths = self._retrieve_for_agent("A3", retrieval_requirements["A3"])

        orchestration_log.append({
            "stage": "multi_path_retrieval",
            "K1_paths": len(K1_paths),
            "K2_paths": len(K2_paths),
            "K3_paths": len(K3_paths)
        })

        # Step 3: 构建agent-specific knowledge graphs
        K1 = self._build_agent_kg("A1", K1_paths, user_input)
        K2 = self._build_agent_kg("A2", K2_paths, user_input)
        K3 = self._build_agent_kg("A3", K3_paths, user_input)

        # Step 4: 一致性检查（确保K1-K2-K3之间的语义一致性）
        consistency_check = self._check_consistency(K1, K2, K3)
        orchestration_log.append(consistency_check)

        total_time = time.time() - t0

        return {
            "K1": K1,
            "K2": K2,
            "K3": K3,
            "orchestration_log": orchestration_log,
            "orchestration_time": total_time,
            "scenario_type": scenario_type
        }

    def _generate_retrieval_requirements(
        self,
        user_input: str,
        scenario_type: Optional[str],
        intermediate_artifacts: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        生成检索需求（Task Definition模块）

        根据预定义的agent知识配置生成具体的检索需求
        """
        requirements = {}

        # A1: Task Interpreter Agent检索需求
        A1_query = user_input
        if scenario_type:
            A1_query += f" 场景类型:{scenario_type}"

        requirements["A1"] = {
            "query": A1_query,
            "focus": self.AGENT_CONFIGS["A1"].retrieval_focus,
            "entity_types": self.AGENT_CONFIGS["A1"].entity_types,
            "max_hops": 2,  # A1主要需要1-2跳的语义关联
            "top_k": 8
        }

        # A2: Mathematical Modeling Agent检索需求
        A2_query = user_input
        if intermediate_artifacts and "scenario_requirement" in intermediate_artifacts:
            # 如果有Z1，则基于Z1的设备、负荷、目标进行精准检索
            z1 = intermediate_artifacts["scenario_requirement"]
            devices = z1.get("devices", [])
            objectives = z1.get("objectives", [])
            A2_query += f" 设备:{','.join([d.get('name','') for d in devices[:3]])}"
            A2_query += f" 目标:{','.join(objectives[:2])}"

        requirements["A2"] = {
            "query": A2_query,
            "focus": self.AGENT_CONFIGS["A2"].retrieval_focus,
            "entity_types": self.AGENT_CONFIGS["A2"].entity_types,
            "max_hops": 3,  # A2需要更深的多跳路径（设备-目标-约束）
            "top_k": 10
        }

        # A3: Code Synthesizer Agent检索需求
        A3_query = user_input
        if scenario_type:
            A3_query += f" {scenario_type}场景代码实现"

        requirements["A3"] = {
            "query": A3_query,
            "focus": self.AGENT_CONFIGS["A3"].retrieval_focus,
            "entity_types": self.AGENT_CONFIGS["A3"].entity_types,
            "max_hops": 2,  # A3主要需要模型-代码映射关系
            "top_k": 6
        }

        return requirements

    def _retrieve_for_agent(self, agent_id: str, requirement: Dict[str, Any]) -> List[Any]:
        """
        为特定Agent调用多路径检索模块

        调用knowledge_graph/mg_kg.py中的multi_path_retrieve方法
        """
        result = self.kg.multi_path_retrieve(
            query=requirement["query"],
            scenario_type=None,
            encode=True,
            max_candidates=30,
            top_k=requirement["top_k"],
            max_hops=requirement["max_hops"]
        )

        return result.get("paths", [])

    def _build_agent_kg(
        self,
        agent_id: str,
        paths: List[Any],
        user_input: str
    ) -> Dict[str, Any]:
        """
        构建agent-specific knowledge graph

        论文对应: Agent-Specific Knowledge Graph Generation
        Ki = (Ei, Ri, Ti) where i ∈ {1,2,3}
        """
        config = self.AGENT_CONFIGS[agent_id]

        # 收集实体、关系和三元组
        entities = set()
        relations = set()
        triples = []

        for path in paths:
            for node in path.nodes:
                if node.node_type in config.entity_types:
                    entities.add(node.node_id)

            for i, rel in enumerate(path.relations):
                relations.add(rel)
                if i < len(path.nodes) - 1:
                    triple = (path.nodes[i].node_id, rel, path.nodes[i+1].node_id)
                    triples.append(triple)

        # 构建知识图谱上下文字符串（用于LLM提示词）
        kg_context = self._format_kg_context(agent_id, paths, config)

        return {
            "agent_id": agent_id,
            "agent_name": config.agent_name,
            "entities": list(entities),
            "relations": list(relations),
            "triples": triples,
            "paths": paths,
            "kg_context": kg_context,
            "num_paths": len(paths),
            "num_entities": len(entities),
            "num_relations": len(relations)
        }

    def _format_kg_context(
        self,
        agent_id: str,
        paths: List[Any],
        config: AgentKnowledgeConfig
    ) -> str:
        """
        格式化KG上下文为LLM友好的文本

        不同Agent有不同的格式化策略
        """
        if agent_id == "A1":
            # Task Interpreter需要术语映射和分类知识
            context = f"## Task Interpretation Knowledge Graph (K1)\n\n"
            context += f"为{config.agent_name}提供语义理解支持:\n\n"

            # 按节点类型分组
            scenario_nodes = []
            device_nodes = []
            load_nodes = []

            for path in paths:
                for node in path.nodes:
                    if node.node_type == "scenario":
                        scenario_nodes.append(node)
                    elif node.node_type == "device":
                        device_nodes.append(node)
                    elif node.node_type == "load":
                        load_nodes.append(node)

            if scenario_nodes:
                context += "### 场景类型识别\n"
                seen = set()
                for node in scenario_nodes[:5]:
                    if node.name not in seen:
                        context += f"- {node.name}: {node.description}\n"
                        seen.add(node.name)

            if device_nodes:
                context += "\n### 设备类型识别\n"
                seen = set()
                for node in device_nodes[:10]:
                    if node.name not in seen:
                        context += f"- {node.name} ({node.model_type}): {node.description}\n"
                        seen.add(node.name)

        elif agent_id == "A2":
            # Mathematical Modeling需要建模公式和约束
            context = f"## Mathematical Modeling Knowledge Graph (K2)\n\n"
            context += f"为{config.agent_name}提供建模知识:\n\n"

            for i, path in enumerate(paths[:10], 1):
                context += f"### 建模路径 {i}\n"
                context += path.to_modeling_context()
                context += "\n\n"

        elif agent_id == "A3":
            # Code Synthesizer需要代码模板和实现模式
            context = f"## Code Synthesis Knowledge Graph (K3)\n\n"
            context += f"为{config.agent_name}提供代码实现指导:\n\n"

            context += "### 模型到代码映射\n"
            for path in paths[:8]:
                for node in path.nodes:
                    if node.expression:
                        context += f"- {node.name}: `{node.expression}`\n"

        else:
            context = "Unknown agent"

        return context

    def _check_consistency(self, K1: Dict, K2: Dict, K3: Dict) -> Dict[str, Any]:
        """
        一致性检查：确保K1-K2-K3之间的语义一致性

        检查:
        1. K1识别的设备是否在K2中有对应的建模知识
        2. K2的变量定义是否在K3中有对应的代码映射
        """
        consistency_issues = []

        # 提取K1中的实体类型
        k1_entities = set(K1.get("entities", []))
        k2_entities = set(K2.get("entities", []))
        k3_entities = set(K3.get("entities", []))

        # 检查K1->K2覆盖
        k1_device_count = sum(1 for e in k1_entities if "device" in e.lower())
        k2_device_count = sum(1 for e in k2_entities if "device" in e.lower())

        if k1_device_count > 0 and k2_device_count == 0:
            consistency_issues.append("K1识别到设备但K2未检索到建模知识")

        # 检查K2->K3覆盖
        if len(k2_entities) > 0 and len(k3_entities) == 0:
            consistency_issues.append("K2有建模知识但K3未检索到代码实现知识")

        return {
            "stage": "consistency_check",
            "k1_entities": len(k1_entities),
            "k2_entities": len(k2_entities),
            "k3_entities": len(k3_entities),
            "issues": consistency_issues,
            "status": "pass" if len(consistency_issues) == 0 else "warning"
        }

    def get_kg_for_agent(self, agent_id: str, orchestration_result: Dict[str, Any]) -> str:
        """
        获取特定Agent的知识图谱上下文字符串

        Args:
            agent_id: "A1" / "A2" / "A3"
            orchestration_result: orchestrate()方法的返回结果

        Returns:
            格式化的KG上下文字符串
        """
        kg_key = f"K{agent_id[1]}"  # A1->K1, A2->K2, A3->K3
        kg_data = orchestration_result.get(kg_key, {})
        return kg_data.get("kg_context", "")
