"""微电网多场景知识图谱 - 支持多路径语义检索

融合了ELMK的多路径知识检索机制与微电网领域知识图谱:
1. 候选路径检索: 从KG中提取与建模需求相关的候选知识路径
2. 语义编码: 使用transformer编码器对路径进行语义表示
3. 聚类选择: K-Means聚类后从各语义簇中选择代表性路径
"""

import json
import re
import time
import yaml
import random
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

from config.settings import Config


@dataclass
class KGNode:
    """知识图谱节点"""
    node_id: str
    node_type: str  # device, load, objective, constraint, scenario
    name: str
    description: str = ""
    params: List[str] = field(default_factory=list)
    model_type: str = ""
    output_type: str = ""
    expression: str = ""
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """将节点转换为文本描述"""
        parts = [f"{self.name}({self.node_type})"]
        if self.description:
            parts.append(self.description)
        if self.expression:
            parts.append(f"表达式: {self.expression}")
        if self.params:
            parts.append(f"参数: {', '.join(self.params)}")
        return " | ".join(parts)


@dataclass
class KGPath:
    """知识路径 - 表示从源节点到目标节点的路径"""
    path_id: str
    nodes: List[KGNode]
    relations: List[str]  # 关系列表
    score: float = 0.0
    cluster_id: int = -1
    embedding: Optional[np.ndarray] = None

    def to_text(self) -> str:
        """将路径转换为文本描述"""
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(node.to_text())
            if i < len(self.relations):
                parts.append(f"[{self.relations[i]}]")
        return " -> ".join(parts)

    def to_modeling_context(self) -> str:
        """将路径转换为建模上下文"""
        context_parts = []
        for i, node in enumerate(self.nodes):
            if node.node_type == "device":
                context_parts.append(f"设备: {node.name}")
                if node.expression:
                    context_parts.append(f"  模型: {node.expression}")
            elif node.node_type == "load":
                context_parts.append(f"负荷: {node.name}")
            elif node.node_type == "objective":
                context_parts.append(f"目标: {node.name}")
                if node.expression:
                    context_parts.append(f"  表达式: {node.expression}")
            elif node.node_type == "constraint":
                context_parts.append(f"约束: {node.name}")
                if node.expression:
                    context_parts.append(f"  约束: {node.expression}")
            if i < len(self.relations):
                context_parts.append(f"  关联: {self.relations[i]}")
        return "\n".join(context_parts)


class MicroGridKnowledgeGraph:
    """
    微电网知识图谱 - 支持多路径检索

    继承自论文提出的KG增强LLM工作流框架，支持:
    - 多场景微电网建模知识组织
    - 多路径知识检索 (候选路径提取 + 语义编码 + 聚类选择)
    - 动态知识上下文构建
    """

    # 节点类型
    TYPE_DEVICE = "device"
    TYPE_LOAD = "load"
    TYPE_OBJECTIVE = "objective"
    TYPE_CONSTRAINT = "constraint"
    TYPE_SCENARIO = "scenario"

    # 预定义关系类型
    REL_PROVIDES = "provides"        # 设备提供能量
    REL_REQUIRES = "requires"        # 负荷需要能量
    REL_COUPLED = "coupled_with"     # 耦合关系
    REL_SUPPORTS = "supports"        # 支撑目标
    REL_SATISFIES = "satisfies"      # 满足约束
    REL_BALANCES = "balances"        # 平衡关系
    REL_CONVERTS = "converts"        # 能量转换
    REL_STORES = "stores"            # 储能关系

    def __init__(self, kg_data: Optional[Dict[str, Any]] = None):
        """
        初始化知识图谱

        Args:
            kg_data: 知识图谱数据字典，如果为None则从YAML加载
        """
        self.config = Config()
        self.kg_data = kg_data if kg_data is not None else self.config.load_knowledge_graph()

        # 检查是否为降级模式（无本地知识图谱）
        self.fallback_mode = self.kg_data.get("fallback_mode", False)

        # 节点存储
        self.nodes: Dict[str, KGNode] = {}
        self.scenarios: Dict[str, Dict[str, Any]] = {}

        # 邻接表 - 用于多跳路径检索
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # node_id -> [(neighbor_id, relation)]

        # 编码器缓存
        self._encoder = None
        self._tokenizer = None
        self._encoder_device = None

        # 构建图谱（如果有数据）
        if not self.fallback_mode:
            self._build_graph()

    def _build_graph(self):
        """从YAML数据构建图谱"""
        scenarios_data = self.kg_data.get("scenarios", {})

        for scenario_id, scenario_data in scenarios_data.items():
            self.scenarios[scenario_id] = scenario_data

            # 创建设备节点
            for device in scenario_data.get("devices", []):
                device_id = f"{scenario_id}.{device['id']}"
                node = KGNode(
                    node_id=device_id,
                    node_type=self.TYPE_DEVICE,
                    name=device.get("name", device["id"]),
                    description=device.get("note", ""),
                    params=device.get("params", []),
                    model_type=device.get("model_type", ""),
                    output_type=device.get("output_type", ""),
                    keywords=[device.get("name", ""), device["id"]] + device.get("params", []),
                    metadata={"scenario": scenario_id, "device_id": device["id"]}
                )
                self._add_node(node)

            # 创建负荷节点
            for load in scenario_data.get("loads", []):
                load_id = f"{scenario_id}.{load['id']}"
                node = KGNode(
                    node_id=load_id,
                    node_type=self.TYPE_LOAD,
                    name=load.get("name", load["id"]),
                    description=load.get("note", ""),
                    params=[f"type={load.get('type', 'unknown')}", f"unit={load.get('unit', 'kW')}"],
                    output_type=load.get("type", ""),
                    keywords=[load.get("name", ""), load["id"]],
                    metadata={"scenario": scenario_id, "load_id": load["id"]}
                )
                self._add_node(node)

            # 创建目标函数节点
            for objective in scenario_data.get("objectives", []):
                obj_id = f"{scenario_id}.{objective['id']}"
                node = KGNode(
                    node_id=obj_id,
                    node_type=self.TYPE_OBJECTIVE,
                    name=objective.get("name", objective["id"]),
                    description=objective.get("expression", ""),
                    expression=objective.get("expression", ""),
                    keywords=[objective.get("name", ""), objective["id"]],
                    metadata={
                        "scenario": scenario_id,
                        "objective_id": objective["id"],
                        "default_weight": objective.get("default_weight", 1.0)
                    }
                )
                self._add_node(node)

            # 创建约束节点
            for constraint in scenario_data.get("constraints", []):
                con_id = f"{scenario_id}.{constraint['id']}"
                node = KGNode(
                    node_id=con_id,
                    node_type=self.TYPE_CONSTRAINT,
                    name=constraint.get("id", ""),
                    description=constraint.get("note", ""),
                    expression=constraint.get("expression", ""),
                    keywords=[constraint["id"], constraint.get("type", "")],
                    metadata={
                        "scenario": scenario_id,
                        "constraint_id": constraint["id"],
                        "constraint_type": constraint.get("type", "")
                    }
                )
                self._add_node(node)

            # 构建场景内关系
            self._build_scenario_relations(scenario_id, scenario_data)

        # 跨场景通用知识
        common = self.kg_data.get("common_knowledge", {})
        self._build_common_knowledge(common)

    def _add_node(self, node: KGNode):
        """添加节点"""
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = []

    def _build_scenario_relations(self, scenario_id: str, scenario_data: Dict[str, Any]):
        """构建场景内节点关系"""
        devices = {d["id"]: d for d in scenario_data.get("devices", [])}
        loads_map = {l["id"]: l for l in scenario_data.get("loads", [])}

        # 设备->负荷关系 (功率平衡)
        for dev_id, dev in devices.items():
            dev_node_id = f"{scenario_id}.{dev_id}"
            output_types = dev.get("output_type", "").split(",")

            for load in scenario_data.get("loads", []):
                load_node_id = f"{scenario_id}.{load['id']}"
                # 根据能量类型建立关系
                if "electrical" in output_types and load.get("type") == "time_series":
                    self._add_relation(dev_node_id, load_node_id, self.REL_BALANCES)
                if "thermal" in output_types and "thermal" in load.get("name", "").lower():
                    self._add_relation(dev_node_id, load_node_id, self.REL_BALANCES)
                if "cooling" in output_types and "冷" in load.get("name", ""):
                    self._add_relation(dev_node_id, load_node_id, self.REL_BALANCES)

        # 设备->目标关系
        for obj in scenario_data.get("objectives", []):
            obj_node_id = f"{scenario_id}.{obj['id']}"
            obj_name = obj.get("name", "").lower()
            for dev_id, dev in devices.items():
                dev_node_id = f"{scenario_id}.{dev_id}"
                dev_name = dev.get("name", "").lower()
                dev_type = dev.get("model_type", "")

                if "renewable" in dev_type or "pv" in dev_id.lower() or "wt" in dev_id.lower():
                    if "可再生" in obj_name or "renewable" in obj_name:
                        self._add_relation(dev_node_id, obj_node_id, self.REL_SUPPORTS)
                if "storage" in dev_type:
                    if "储能" in obj_name or "cost" in obj_name or "economic" in obj_name:
                        self._add_relation(dev_node_id, obj_node_id, self.REL_SUPPORTS)
                if "dispatchable" in dev_type:
                    if "经济" in obj_name or "cost" in obj.get("id", "").lower():
                        self._add_relation(dev_node_id, obj_node_id, self.REL_SUPPORTS)

        # 设备->约束关系
        for con in scenario_data.get("constraints", []):
            con_node_id = f"{scenario_id}.{con['id']}"
            for dev_id, dev in devices.items():
                dev_node_id = f"{scenario_id}.{dev_id}"
                self._add_relation(dev_node_id, con_node_id, self.REL_SATISFIES)

        # 负荷->约束关系
        for con in scenario_data.get("constraints", []):
            con_node_id = f"{scenario_id}.{con['id']}"
            for load in scenario_data.get("loads", []):
                load_node_id = f"{scenario_id}.{load['id']}"
                self._add_relation(load_node_id, con_node_id, self.REL_SATISFIES)

        # 能量转换耦合 (CCHP等)
        for dev_id, dev in devices.items():
            if "cchp" in dev_id.lower() or "chp" in dev_id.lower():
                dev_node_id = f"{scenario_id}.{dev_id}"
                output_types = dev.get("output_type", "").split(",")
                for other_dev_id, other_dev in devices.items():
                    if other_dev_id == dev_id:
                        continue
                    other_node_id = f"{scenario_id}.{other_dev_id}"
                    if any(x in other_dev.get("output_type", "") for x in output_types):
                        self._add_relation(dev_node_id, other_node_id, self.REL_COUPLED)

        # 储能设备关系
        for dev_id, dev in devices.items():
            if "storage" in dev.get("model_type", "").lower() or "ees" in dev_id.lower() or "bess" in dev_id.lower() or "tes" in dev_id.lower():
                dev_node_id = f"{scenario_id}.{dev_id}"
                for load in scenario_data.get("loads", []):
                    load_node_id = f"{scenario_id}.{load['id']}"
                    self._add_relation(dev_node_id, load_node_id, self.REL_STORES)

    def _build_common_knowledge(self, common: Dict[str, Any]):
        """构建通用知识关系"""
        device_defaults = common.get("device_defaults", {})
        solver_config = common.get("solver_config", {})

        # 为所有设备节点添加默认参数
        for node_id, node in self.nodes.items():
            if node.node_type == self.TYPE_DEVICE:
                dev_id = node.metadata.get("device_id", "")
                if dev_id in device_defaults:
                    node.params.extend([f"{k}={v}" for k, v in device_defaults[dev_id].items()])

    def _add_relation(self, from_id: str, to_id: str, relation: str):
        """添加关系边"""
        if from_id in self.nodes and to_id in self.nodes:
            self.adjacency[from_id].append((to_id, relation))

    # ==================== 候选路径检索 ====================

    def _tokenize_query(self, query: str) -> Set[str]:
        """
        智能分词：同时支持中文和英文

        - 英文单词：用 \w+ 提取
        - 中文关键词匹配：优先提取完整场景词、设备词，再做字符级n-gram补漏
        """
        tokens: Set[str] = set()

        # 英文部分
        tokens.update(re.findall(r'[a-zA-Z_][\w]*', query.lower()))

        # 中文字符序列
        chinese_segments = re.findall(r'[\u4e00-\u9fff]+', query)
        for seg in chinese_segments:
            # 整段作为一个粗粒度token（优先匹配完整词）
            if len(seg) <= 4:
                tokens.add(seg)
            # 再做2-gram和3-gram补漏（只针对4字以上的片段）
            for i in range(len(seg)):
                for n in (2, 3):
                    if i + n <= len(seg):
                        tokens.add(seg[i:i + n])

        return tokens

    def retrieve_candidate_paths(
        self,
        query: str,
        scenario_type: Optional[str] = None,
        max_hops: int = 3,
        max_candidates: int = 30
    ) -> List[KGPath]:
        """
        检索候选知识路径 (多跳路径检索)

        对应ELMK的多跳子图构建方法:
        1. 根据查询识别相关实体节点
        2. 沿关系边扩展检索，跳数不超过max_hops
        3. 收集所有满足条件的路径

        Args:
            query: 自然语言查询
            scenario_type: 场景类型过滤
            max_hops: 最大跳数
            max_candidates: 最大候选路径数

        Returns:
            候选知识路径列表
        """
        # 降级模式：返回空列表
        if self.fallback_mode:
            return []

        query_tokens = self._tokenize_query(query)

        # Step 1: 识别相关节点
        relevant_nodes = self._find_relevant_nodes(query_tokens, scenario_type, query)

        if not relevant_nodes:
            relevant_nodes = list(self.nodes.values())

        # Step 2: 多跳路径扩展
        all_paths: List[KGPath] = []
        path_id_counter = 0

        for start_node in relevant_nodes[:min(len(relevant_nodes), 10)]:
            # 1-hop paths
            for neighbor, relation in self.adjacency.get(start_node.node_id, []):
                neighbor_node = self.nodes.get(neighbor)
                if neighbor_node:
                    path = KGPath(
                        path_id=f"path_{path_id_counter}",
                        nodes=[start_node, neighbor_node],
                        relations=[relation],
                        score=self._compute_path_score(query_tokens, start_node, neighbor_node, relation)
                    )
                    all_paths.append(path)
                    path_id_counter += 1

            # 2-hop paths
            if max_hops >= 2:
                for neighbor1, rel1 in self.adjacency.get(start_node.node_id, []):
                    n1_node = self.nodes.get(neighbor1)
                    if not n1_node:
                        continue
                    for neighbor2, rel2 in self.adjacency.get(neighbor1, []):
                        n2_node = self.nodes.get(neighbor2)
                        if not n2_node or neighbor2 == start_node.node_id:
                            continue
                        path = KGPath(
                            path_id=f"path_{path_id_counter}",
                            nodes=[start_node, n1_node, n2_node],
                            relations=[rel1, rel2],
                            score=self._compute_path_score_2hop(query_tokens, start_node, n1_node, n2_node, rel1, rel2)
                        )
                        all_paths.append(path)
                        path_id_counter += 1

            # 3-hop paths
            if max_hops >= 3:
                for neighbor1, rel1 in self.adjacency.get(start_node.node_id, []):
                    n1_node = self.nodes.get(neighbor1)
                    if not n1_node:
                        continue
                    for neighbor2, rel2 in self.adjacency.get(neighbor1, []):
                        n2_node = self.nodes.get(neighbor2)
                        if not n2_node or neighbor2 == start_node.node_id:
                            continue
                        for neighbor3, rel3 in self.adjacency.get(neighbor2, []):
                            n3_node = self.nodes.get(neighbor3)
                            if not n3_node or neighbor3 in [start_node.node_id, neighbor1]:
                                continue
                            path = KGPath(
                                path_id=f"path_{path_id_counter}",
                                nodes=[start_node, n1_node, n2_node, n3_node],
                                relations=[rel1, rel2, rel3],
                                score=self._compute_path_score_3hop(query_tokens, start_node, n1_node, n2_node, n3_node)
                            )
                            all_paths.append(path)
                            path_id_counter += 1

        # 去重并排序
        unique_paths = self._deduplicate_paths(all_paths)
        unique_paths.sort(key=lambda p: p.score, reverse=True)

        return unique_paths[:max_candidates]

    def _find_relevant_nodes(
        self,
        query_tokens: Set[str],
        scenario_type: Optional[str],
        query: str = ""
    ) -> List[KGNode]:
        """根据查询tokens查找相关节点，场景词触发场景boost"""
        # 场景词 -> scenario_id 映射（用于场景限定）
        scenario_keywords: Dict[str, List[str]] = {
            "agricultural": ["农业", "农村", "灌溉", "温室", "大棚", "养殖", "农户", "农作物", "沼气", "种植", "畜牧"],
            "grid_connected": ["并网", "并网型", "主电网", "购售电", "PCC"],
            "islanded": ["离网", "孤岛", "独立运行", "自主调度"],
            "multi_energy": ["多能互补", "冷热电", "CCHP", "热泵", "梯级利用"],
            "nmg": ["多微电网", "微电网群", "互联调度", "配电网", "IEEE33", "IEEE69"],
            "ev_integrated": ["电动汽车", "EV", "V2G", "充电站", "移动储能"],
            "industrial": ["工业", "园区", "重要负荷", "可中断"],
            "commercial": ["商业", "商场", "酒店", "写字楼", "需求响应"],
            "residential": ["住宅", "居民", "家庭", "小区", "别墅"],
            "multi_energy_hub": ["多能源", "综合能源", "能源Hub", "氢能", "P2G", "燃气"],
            "net_zero": ["净零", "近零能耗", "产能建筑", "被动式"],
        }

        # 检测查询中出现的场景词
        detected_scenarios: Set[str] = set()
        for sid, keywords in scenario_keywords.items():
            for kw in keywords:
                if kw in query:
                    detected_scenarios.add(sid)

        candidates = []
        for node_id, node in self.nodes.items():
            # 场景限定：如果检测到特定场景，只返回该场景的节点
            if detected_scenarios:
                if node.metadata.get("scenario") not in detected_scenarios:
                    continue
            elif scenario_type and node.metadata.get("scenario") != scenario_type:
                continue

            score = 0.0
            node_scenario = node.metadata.get("scenario", "")

            for token in query_tokens:
                if token in node.name:
                    score += 3.0
                if token in node.node_id.lower():
                    score += 2.0
                for kw in node.keywords:
                    if token in kw:
                        score += 1.5
                if node.expression and token in node.expression:
                    score += 1.0

            if score > 0:
                candidates.append((node, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]

    def _compute_path_score(
        self,
        query_tokens: Set[str],
        node1: KGNode,
        node2: KGNode,
        relation: str
    ) -> float:
        """计算路径分数 (基于节点与查询的匹配度)"""
        score = 0.0

        for token in query_tokens:
            # 节点1匹配
            if token in node1.name.lower():
                score += 2.0
            for kw in node1.keywords:
                if token in kw.lower():
                    score += 1.0

            # 节点2匹配
            if token in node2.name.lower():
                score += 2.0
            for kw in node2.keywords:
                if token in kw.lower():
                    score += 1.0

            # 关系匹配
            if token in relation.lower():
                score += 0.5

            # 节点类型权重
            if node1.node_type == self.TYPE_DEVICE and ("pv" in token or "wt" in token or "storage" in token or "cchp" in token):
                score += 1.5
            if node2.node_type == self.TYPE_OBJECTIVE and ("目标" in node2.name or "objective" in node2.name.lower()):
                score += 1.0

        return score

    def _compute_path_score_2hop(
        self,
        query_tokens: Set[str],
        n1: KGNode,
        n2: KGNode,
        n3: KGNode,
        rel1: str,
        rel2: str
    ) -> float:
        """计算2跳路径分数"""
        base = self._compute_path_score(query_tokens, n1, n2, rel1)
        # 额外考虑中间节点和末端节点
        for token in query_tokens:
            if token in n3.name.lower():
                base += 1.5
            for kw in n3.keywords:
                if token in kw.lower():
                    base += 0.8
            if token in rel2.lower():
                base += 0.3
        # 中间节点类型加权
        if n2.node_type == self.TYPE_DEVICE:
            base *= 1.2
        return base

    def _compute_path_score_3hop(
        self,
        query_tokens: Set[str],
        n1: KGNode,
        n2: KGNode,
        n3: KGNode,
        n4: KGNode
    ) -> float:
        """计算3跳路径分数"""
        base = self._compute_path_score_2hop(query_tokens, n1, n2, n3, "", "")
        for token in query_tokens:
            if token in n4.name.lower():
                base += 1.0
        return base

    def _deduplicate_paths(self, paths: List[KGPath]) -> List[KGPath]:
        """去重路径"""
        seen = set()
        unique = []
        for p in paths:
            node_ids = tuple(n.node_id for n in p.nodes)
            if node_ids not in seen:
                seen.add(node_ids)
                unique.append(p)
        return unique

    # ==================== 语义编码 ====================

    def encode_paths(
        self,
        paths: List[KGPath],
        encoder_model: Optional[str] = None,
        batch_size: int = 64,
        max_len: int = 128
    ) -> List[KGPath]:
        """
        使用Transformer编码器对路径进行语义编码

        对应ELMK的语义编码步骤:
        - 使用sentence-transformers/all-mpnet-base-v2编码路径文本
        - 得到路径的768维语义向量表示
        """
        if not paths:
            return paths

        model_path = encoder_model or self.config.PATH_ENCODER_MODEL

        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            if self._encoder is None:
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._encoder = AutoModel.from_pretrained(model_path)
                self._encoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._encoder.to(self._encoder_device)
                self._encoder.eval()

            path_texts = [p.to_text() for p in paths]
            all_embeddings = []

            with torch.inference_mode():
                for i in range(0, len(path_texts), batch_size):
                    batch_texts = path_texts[i:i + batch_size]
                    encoded = self._tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt"
                    )
                    encoded = {k: v.to(self._encoder_device) for k, v in encoded.items()}

                    try:
                        with torch.autocast(device_type=str(self._encoder_device), dtype=torch.float16):
                            outputs = self._encoder(**encoded)
                    except Exception:
                        outputs = self._encoder(**encoded)

                    # CLS向量
                    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeddings.append(cls_emb)

            embeddings = np.concatenate(all_embeddings, axis=0)

            # L2归一化 (与ELMK一致)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-12)

            for path, emb in zip(paths, embeddings):
                path.embedding = emb

        except Exception as e:
            # 如果编码器加载失败，使用随机初始化但保持框架运行
            print(f"[Warning] 语义编码器加载失败: {e}, 使用文本表示替代")
            dim = 768
            for path in paths:
                # 使用路径文本的简单哈希作为伪嵌入
                np.random.seed(hash(path.to_text()) % (2**31))
                path.embedding = np.random.randn(dim).astype(np.float32)
                path.embedding /= (np.linalg.norm(path.embedding) + 1e-12)

        return paths

    # ==================== 聚类选择 ====================

    def cluster_and_select_paths(
        self,
        paths: List[KGPath],
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        n_clusters: Optional[int] = None
    ) -> List[KGPath]:
        """
        聚类选择代表性路径

        对应ELMK的K-Means聚类选择机制:
        1. 使用K-Means对路径嵌入进行聚类
        2. 每个簇中选择与查询最相似的路径
        3. 合并所有簇的选择结果

        Args:
            paths: 已编码的路径列表
            query_embedding: 查询嵌入 (用于相似度计算)
            top_k: 最终选择的路径数
            n_clusters: 聚类数，默认sqrt(n_paths)

        Returns:
            选中的代表性路径列表
        """
        if not paths:
            return []

        # 收集嵌入
        embeddings = []
        valid_paths = []
        for p in paths:
            if p.embedding is not None:
                embeddings.append(p.embedding)
                valid_paths.append(p)

        if not valid_paths:
            return paths[:top_k]

        embeddings = np.array(embeddings)

        if query_embedding is not None:
            # 确保维度一致
            if query_embedding.shape[-1] != embeddings.shape[-1]:
                print(f"[Warning] 嵌入维度不匹配: query={query_embedding.shape}, paths={embeddings.shape}")
                return paths[:top_k]

        # 确定聚类数
        n_clusters = n_clusters or max(3, min(len(valid_paths) // 3, 10))

        if len(valid_paths) <= n_clusters:
            # 路径数少于聚类数，直接按分数排序
            valid_paths.sort(key=lambda p: p.score, reverse=True)
            return valid_paths[:top_k]

        # K-Means聚类
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            # sklearn不可用时直接按分数选择
            valid_paths.sort(key=lambda p: p.score, reverse=True)
            return valid_paths[:top_k]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # 计算每个路径与查询的相似度
        if query_embedding is not None:
            sim_scores = np.dot(embeddings, query_embedding)
        else:
            # 无查询嵌入时使用簇中心距离
            centers = kmeans.cluster_centers_
            dists = np.linalg.norm(embeddings - centers[labels], axis=1)
            sim_scores = -dists  # 距离越小越好

        # 为每个路径分配簇ID
        for path, label in zip(valid_paths, labels):
            path.cluster_id = int(label)

        # 从每个簇中选择路径
        selected_paths = []
        quota_per_cluster = top_k // n_clusters
        remainder = top_k % n_clusters

        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, p in enumerate(valid_paths) if p.cluster_id == cluster_id]
            if not cluster_indices:
                continue

            cluster_scores = [(i, sim_scores[i]) for i in cluster_indices]
            cluster_scores.sort(key=lambda x: x[1], reverse=True)

            # 分配配额
            quota = quota_per_cluster + (1 if cluster_id < remainder else 0)
            for idx, _ in cluster_scores[:quota]:
                selected_paths.append(valid_paths[idx])

        # 如果选择不足，补充分数最高的路径
        selected_ids = set(id(p) for p in selected_paths)
        remaining = [p for p in valid_paths if id(p) not in selected_ids]
        remaining.sort(key=lambda p: p.score, reverse=True)

        while len(selected_paths) < top_k and remaining:
            selected_paths.append(remaining.pop(0))

        return selected_paths

    # ==================== 多路径检索主接口 ====================

    def multi_path_retrieve(
        self,
        query: str,
        scenario_type: Optional[str] = None,
        encode: bool = True,
        max_candidates: int = 30,
        top_k: int = 10,
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """
        多路径知识检索主接口

        对应论文框架的"多路径知识检索机制":
        1. 候选路径检索 - candidate path retrieval
        2. 语义编码 - semantic encoding (可选)
        3. 聚类选择 - clustering-based selection

        Args:
            query: 自然语言建模描述
            scenario_type: 场景类型
            encode: 是否使用语义编码
            max_candidates: 最大候选路径数
            top_k: 最终选择数
            max_hops: 最大跳数

        Returns:
            {
                "paths": 选中的路径列表,
                "all_candidates": 所有候选路径,
                "query": 原始查询,
                "scenario_type": 场景类型,
                "num_clusters": 聚类数,
                "encoding_time": 编码耗时,
                "retrieval_time": 总检索耗时
            }
        """
        t0 = time.time()

        # 降级模式：无本地知识图谱，返回空结果
        if self.fallback_mode:
            return {
                "paths": [],
                "all_candidates": [],
                "query": query,
                "scenario_type": scenario_type,
                "num_clusters": 0,
                "encoding_time": 0.0,
                "retrieval_time": 0.0,
                "total_time": 0.0,
                "fallback_mode": True,
                "message": "无本地知识图谱，将使用LLM通用知识"
            }

        # Step 1: 候选路径检索
        candidate_paths = self.retrieve_candidate_paths(
            query=query,
            scenario_type=scenario_type,
            max_hops=max_hops,
            max_candidates=max_candidates
        )

        retrieval_time = time.time() - t0

        if not candidate_paths:
            total_time = time.time() - t0
            return {
                "paths": [],
                "all_candidates": [],
                "query": query,
                "scenario_type": scenario_type,
                "num_clusters": 0,
                "encoding_time": 0.0,
                "retrieval_time": retrieval_time,
                "total_time": total_time
            }

        # Step 2: 语义编码
        encoding_time = 0.0
        query_embedding = None

        if encode:
            t1 = time.time()
            candidate_paths = self.encode_paths(candidate_paths)

            # 生成查询嵌入
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                model_path = self.config.PATH_ENCODER_MODEL

                if self._encoder is None:
                    self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self._encoder = AutoModel.from_pretrained(model_path)
                    self._encoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self._encoder.to(self._encoder_device)
                    self._encoder.eval()

                with torch.inference_mode():
                    encoded_q = self._tokenizer(
                        query,
                        padding=True,
                        truncation=True,
                        max_length=self.config.PATH_ENCODER_MAX_LEN,
                        return_tensors="pt"
                    )
                    encoded_q = {k: v.to(self._encoder_device) for k, v in encoded_q.items()}
                    outputs = self._encoder(**encoded_q)
                    q_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    norms = np.linalg.norm(q_emb) + 1e-12
                    query_embedding = q_emb / norms

                encoding_time = time.time() - t1
            except Exception as e:
                print(f"[Warning] 查询编码失败: {e}")
                encoding_time = time.time() - t1

        # Step 3: 聚类选择
        n_clusters = max(3, min(len(candidate_paths) // 3, 10))
        selected_paths = self.cluster_and_select_paths(
            paths=candidate_paths,
            query_embedding=query_embedding,
            top_k=top_k,
            n_clusters=n_clusters
        )

        total_time = time.time() - t0

        return {
            "paths": selected_paths,
            "all_candidates": candidate_paths,
            "num_candidates": len(candidate_paths),
            "query": query,
            "scenario_type": scenario_type,
            "num_clusters": n_clusters,
            "encoding_time": encoding_time,
            "retrieval_time": retrieval_time,
            "total_time": total_time
        }

    # ==================== 上下文构建 ====================

    def build_kg_context(
        self,
        query: str,
        scenario_type: Optional[str] = None,
        top_k: int = 10,
        include_all_candidates: bool = False
    ) -> str:
        """
        构建知识图谱上下文字符串

        对应论文框架的"动态知识证据":
        - 将选中的多路径知识整合为LLM友好的上下文

        Args:
            query: 原始查询
            scenario_type: 场景类型
            top_k: 选择路径数
            include_all_candidates: 是否包含所有候选路径

        Returns:
            格式化的知识上下文字符串
        """
        # 降级模式：返回空上下文
        if self.fallback_mode:
            return ""

        result = self.multi_path_retrieve(
            query=query,
            scenario_type=scenario_type,
            encode=True,
            top_k=top_k
        )

        paths = result["paths"]
        all_candidates = result["all_candidates"]

        if not paths:
            return "未检索到相关建模知识，使用通用微电网建模原则。"

        context_parts = [
            f"## 知识图谱多路径检索结果 (检索到 {len(all_candidates)} 条候选路径, 选择 {len(paths)} 条代表性路径, 分 {result['num_clusters']} 个语义簇)\n"
        ]

        # 按节点类型分组展示
        device_paths = []
        load_paths = []
        objective_paths = []
        constraint_paths = []
        mixed_paths = []

        for path in paths:
            types = set(n.node_type for n in path.nodes)
            if types == {self.TYPE_DEVICE}:
                device_paths.append(path)
            elif types == {self.TYPE_LOAD}:
                load_paths.append(path)
            elif types == {self.TYPE_OBJECTIVE}:
                objective_paths.append(path)
            elif types == {self.TYPE_CONSTRAINT}:
                constraint_paths.append(path)
            else:
                mixed_paths.append(path)

        # 设备相关路径
        if device_paths:
            context_parts.append("\n### 设备建模知识\n")
            for path in device_paths:
                ctx = path.to_modeling_context()
                if ctx:
                    context_parts.append(f"- {ctx}\n")

        # 负荷相关路径
        if load_paths:
            context_parts.append("\n### 负荷建模知识\n")
            for path in load_paths:
                ctx = path.to_modeling_context()
                if ctx:
                    context_parts.append(f"- {ctx}\n")

        # 目标函数路径
        if objective_paths:
            context_parts.append("\n### 优化目标知识\n")
            for path in objective_paths:
                ctx = path.to_modeling_context()
                if ctx:
                    context_parts.append(f"- {ctx}\n")

        # 约束条件路径
        if constraint_paths:
            context_parts.append("\n### 约束建模知识\n")
            for path in constraint_paths:
                ctx = path.to_modeling_context()
                if ctx:
                    context_parts.append(f"- {ctx}\n")

        # 混合路径 (跨类型关联)
        if mixed_paths:
            context_parts.append("\n### 跨类型建模关联\n")
            for path in mixed_paths:
                node_names = [n.name for n in path.nodes]
                relations = path.relations
                rel_str = " -> ".join([f"{n}({r})" for n, r in zip(node_names[:-1], relations)])
                context_parts.append(f"- 路径: {rel_str} -> {node_names[-1]}\n")
                context_parts.append(f"  {path.to_modeling_context()}\n")

        # 检索统计信息
        context_parts.append(f"\n## 检索统计\n")
        context_parts.append(f"- 候选路径数: {len(all_candidates)}\n")
        context_parts.append(f"- 选中路径数: {len(paths)}\n")
        context_parts.append(f"- 语义簇数: {result['num_clusters']}\n")
        context_parts.append(f"- 检索耗时: {result['retrieval_time']:.3f}s, 编码耗时: {result['encoding_time']:.3f}s\n")

        # 附加所有候选路径概览
        if include_all_candidates and all_candidates:
            context_parts.append("\n## 全部候选路径概览 (按分数排序)\n")
            for i, p in enumerate(all_candidates[:15], 1):
                types = "/".join(set(n.node_type for n in p.nodes))
                names = " -> ".join(n.name for n in p.nodes)
                context_parts.append(f"{i}. [{types}] {names} (分数: {p.score:.2f})\n")

        return "".join(context_parts)

    # ==================== 场景查询接口 ====================

    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """获取场景完整知识"""
        return self.scenarios.get(scenario_id)

    def list_scenarios(self) -> List[Dict[str, str]]:
        """列出所有场景"""
        return [
            {"id": sid, "name": sdata.get("name", ""), "description": sdata.get("description", "")}
            for sid, sdata in self.scenarios.items()
        ]

    def get_scenario_paths(
        self,
        scenario_id: str,
        top_k: int = 10
    ) -> List[KGPath]:
        """获取指定场景的所有建模路径"""
        scenario_nodes = [
            node for node_id, node in self.nodes.items()
            if node.metadata.get("scenario") == scenario_id
        ]

        if not scenario_nodes:
            return []

        paths: List[KGPath] = []
        for node in scenario_nodes:
            for neighbor, relation in self.adjacency.get(node.node_id, []):
                neighbor_node = self.nodes.get(neighbor)
                if neighbor_node and neighbor_node.metadata.get("scenario") == scenario_id:
                    path = KGPath(
                        path_id=f"scenario_{scenario_id}_{node.node_id}_{neighbor}",
                        nodes=[node, neighbor_node],
                        relations=[relation],
                        score=1.0
                    )
                    paths.append(path)

        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:top_k]

    def get_common_knowledge(self) -> Dict[str, Any]:
        """获取通用知识组件"""
        return self.kg_data.get("common_knowledge", {})


# 全局单例
_kg_instance: Optional[MicroGridKnowledgeGraph] = None


def get_knowledge_graph() -> MicroGridKnowledgeGraph:
    """获取知识图谱单例"""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = MicroGridKnowledgeGraph()
    return _kg_instance
