"""Streamlit Web界面 - 知识图谱增强的LLM Agent工作流"""
import sys
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from agents import (
    ScenarioInterpreterAgent,
    MathModelerAgent,
    CodeSynthesizerAgent,
)
from knowledge_graph import get_knowledge_graph, MicroGridKnowledgeGraph
from config.settings import Config

st.set_page_config(
    page_title="KG-Enhanced LLM Agent - 微电网自动化建模",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ 知识图谱增强的LLM Agent微电网自动化建模系统")
st.markdown("**论文方法**: *Knowledge Graph-Enhanced Large Language Model Agent Workflow for Automatic Microgrid Modeling*")
st.markdown("**W = {X, Y, A, K, G, Z, H}** | 多路径知识检索 · 三Agent协同 · LangGraph编排 · 反馈纠正")

# --- 侧边栏 ---
with st.sidebar:
    st.header("📋 系统架构")
    st.markdown("""
    ### 工作流 W = {X, Y, A, K, G, Z, H}
    - **X**: 自然语言建模描述
    - **K**: 多路径知识图谱检索
        - 候选路径检索
        - 语义编码 (Transformer)
        - 聚类选择 (K-Means)
    - **A={AKG, A1, A2, A3}**: Agent集合
        - AKG: 知识图谱编排Agent
        - A1: 任务解释Agent
        - A2: 数学建模Agent
        - A3: 代码合成Agent
    - **Z**: 中间产物 (Z1=结构化规范, Z2=数学模型)
    - **G**: LangGraph全局编排
    - **H**: 人工干预机制
    - **Y**: 最终输出 (M+C+S+R)

    ### 核心创新
    1. 多路径知识检索 (1/2/3-hop)
    2. 语义编码: sentence-transformers
    3. 聚类选择: K-Means多样性采样
    4. Agent专用知识图谱 K1, K2, K3
    """)

    st.header("📊 支持场景类型")
    kg = get_knowledge_graph()

    # 检查是否处于降级模式
    if kg.fallback_mode:
        st.warning("⚠️ **降级模式**: 未检测到本地知识图谱文件")
        st.info("""
        **当前状态**: 系统将使用LLM的通用微电网建模知识继续工作

        **影响**:
        - ✅ 所有Agent可正常执行
        - ✅ 使用LLM内置的微电网领域知识
        - ❌ 无法使用本地知识图谱的多路径检索
        - ❌ 无法使用自定义场景知识

        **启用知识图谱**: 将知识图谱文件放置到 `knowledge_graph/mg_scenarios.yaml`
        """)
    else:
        st.success("✅ **知识图谱已加载**: 使用多路径检索增强")
        scenarios = kg.list_scenarios()
        for s in scenarios:
            with st.expander(f"{s['name']}"):
                st.caption(s["description"][:80] + "...")


# --- Tab ---
tab_run, tab_kg_demo, tab_examples = st.tabs(["🚀 开始生成", "🔍 知识图谱", "📝 配置示例"])

# ========== 知识图谱演示 Tab ==========
with tab_kg_demo:
    st.markdown("### 知识图谱多路径检索演示")

    demo_query = st.text_input(
        "输入建模描述",
        value="需要构建一个包含光伏、风力、储能的多能互补微电网，目标是最小化运行成本",
        help="输入您想了解的微电网建模场景，系统将展示多路径知识检索过程"
    )

    col_demo1, col_demo2 = st.columns(2)

    # 知识图谱统计信息（始终可用）
    kg_demo = get_knowledge_graph()

    with col_demo1:
        st.markdown("#### 多路径检索结果")
        if st.button("🔍 执行多路径检索", key="demo_retrieval"):
            with st.spinner("正在执行多路径知识检索..."):
                kg_demo = get_knowledge_graph()

                # 检查降级模式
                if kg_demo.fallback_mode:
                    st.warning("⚠️ 降级模式：无本地知识图谱，无法执行多路径检索")
                    st.info("💡 系统将在执行任务时使用LLM的通用知识，不依赖知识图谱检索")
                else:
                    result = kg_demo.multi_path_retrieve(
                        query=demo_query,
                        encode=True,
                        max_candidates=30,
                        top_k=10,
                        max_hops=3
                    )

                    paths = result["paths"]
                    st.success(f"检索完成！候选路径: {len(result['all_candidates'])} 条, "
                               f"选中路径: {len(paths)} 条, 语义簇: {result['num_clusters']} 个")

                    st.markdown(f"**检索耗时**: {result['retrieval_time']:.3f}s (候选) + "
                                f"{result['encoding_time']:.3f}s (编码) = {result['total_time']:.3f}s")

                    for i, path in enumerate(paths, 1):
                        types = "/".join(set(n.node_type for n in path.nodes))
                        names = " → ".join(n.name for n in path.nodes)
                    st.markdown(f"**路径 {i}** [{types}] (簇{path.cluster_id}, 分数:{path.score:.2f})")
                    st.markdown(f"  {names}")
                    with st.expander("详情"):
                        st.text(path.to_modeling_context())

    with col_demo2:
        st.markdown("#### 场景知识图谱")
        st.markdown("**节点统计**:")
        nodes_by_type = {}
        for node_id, node in kg_demo.nodes.items():
            t = node.node_type
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1
        for t, c in nodes_by_type.items():
            st.markdown(f"  - {t}: {c} 个节点")
        st.markdown(f"**场景数**: {len(kg_demo.scenarios)} 个")
        st.markdown(f"**总节点数**: {len(kg_demo.nodes)} 个")
        st.markdown(f"**总边数**: {sum(len(v) for v in kg_demo.adjacency.values())} 条")


# ========== 配置示例 Tab ==========
with tab_examples:
    st.markdown("为每个Agent提供参考示例，会作为补充上下文拼接到输入中。留空则不附加。")

    ex1, ex2, ex3 = st.tabs([
        "Agent 1: 场景解释器",
        "Agent 2: 数学建模",
        "Agent 3: 代码合成"
    ])

    with ex1:
        st.session_state.setdefault("ex_agent1", "")
        st.session_state["ex_agent1"] = st.text_area(
            "Agent 1 参考示例",
            value=st.session_state["ex_agent1"],
            height=200,
            key="input_ex1",
            placeholder="参考示例将帮助Agent更准确地解析场景..."
        )

    with ex2:
        st.session_state.setdefault("ex_agent2", "")
        st.session_state["ex_agent2"] = st.text_area(
            "Agent 2 参考示例",
            value=st.session_state["ex_agent2"],
            height=200,
            key="input_ex2",
            placeholder="数学建模的参考示例..."
        )

    with ex3:
        st.session_state.setdefault("ex_agent3", "")
        st.session_state["ex_agent3"] = st.text_area(
            "Agent 3 参考示例",
            value=st.session_state["ex_agent3"],
            height=200,
            key="input_ex3",
            placeholder="代码生成的参考示例..."
        )


# ========== 运行 Tab ==========
with tab_run:
    user_input = st.text_area(
        "📝 输入微电网场景描述",
        value="需要构建一个包含光伏、风力、CCHP和储能的多能互补微电网系统，系统运行24小时，目标是最小化运行成本同时保证供电可靠性。系统需要与主电网连接进行电力交易。",
        height=120
    )

    st.markdown("**知识图谱多路径预检索**:")
    kg_preview = get_knowledge_graph()

    # 检查降级模式
    if kg_preview.fallback_mode:
        st.info("💡 **降级模式**: 无本地知识图谱，系统将使用LLM的通用微电网建模知识")
    else:
        retrieval_preview = kg_preview.multi_path_retrieve(
            query=user_input,
            encode=False,
            max_candidates=5,
            top_k=3
        )
        if retrieval_preview["paths"]:
            for r in retrieval_preview["paths"]:
                types = "/".join(set(n.node_type for n in r.nodes))
                names = " → ".join(n.name for n in r.nodes)
                st.markdown(f"  - [{types}] {names}")
        else:
            st.markdown("  - 未匹配到特定场景，使用通用知识")

    if st.button("🚀 开始生成", type="primary"):

        # --- Step 1: 多路径KG检索 ---
        st.markdown("---")
        st.markdown("### 🔍 步骤 1/4: 知识图谱多路径检索 (K)")
        st.caption("多路径知识检索: 候选路径检索 → 语义编码 → K-Means聚类选择")
        with st.spinner("正在执行多路径知识检索..."):
            kg = get_knowledge_graph()

            # 检查降级模式
            if kg.fallback_mode:
                st.warning("⚠️ **降级模式**: 无本地知识图谱，跳过多路径检索，使用LLM通用知识")
                kg_context = ""
                kg_result = {
                    "all_candidates": [],
                    "paths": [],
                    "num_clusters": 0,
                    "total_time": 0.0
                }
            else:
                kg_result = kg.multi_path_retrieve(
                    query=user_input,
                    encode=True,
                    max_candidates=Config.MAX_CANDIDATE_PATHS,
                    top_k=Config.PATH_SELECT_TOP_K,
                    max_hops=3
                )
                kg_context = kg.build_kg_context(
                    query=user_input,
                    top_k=Config.PATH_SELECT_TOP_K,
                    include_all_candidates=False
                )

        col_kg1, col_kg2, col_kg3, col_kg4 = st.columns(4)
        with col_kg1:
            st.metric("候选路径", len(kg_result.get("all_candidates", [])))
        with col_kg2:
            st.metric("选中路径", len(kg_result.get("paths", [])))
        with col_kg3:
            st.metric("语义簇数", kg_result.get("num_clusters", 0))
        with col_kg4:
            total_t = kg_result.get("total_time", 0.0)
            st.metric("总耗时", f"{total_t:.2f}s")

        if not kg.fallback_mode:
            with st.expander("📖 知识图谱检索上下文（传给Agent）", expanded=False):
                st.text(kg_context[:3000] + ("..." if len(kg_context) > 3000 else ""))

        # --- Agent 1 ---
        st.markdown("---")
        st.markdown("### 🤖 步骤 2/4: 场景解释器 (A1)")
        st.caption("解析自然语言场景，识别设备、负荷、目标函数和约束条件...")
        try:
            agent1 = ScenarioInterpreterAgent()
            agent1_result = agent1.run({
                "user_input": user_input,
                "knowledge_graph_context": kg_context
            })
            scenario_req = agent1_result["output"]

            col_header1, col_header2, col_header3 = st.columns(3)
            with col_header1:
                st.metric("场景类型", scenario_req.get("scenario_type", "unknown"))
            with col_header2:
                st.metric("置信度", f"{scenario_req.get('scenario_confidence', 0):.2f}")
            with col_header3:
                st.metric("设备数量", len(scenario_req.get("devices", [])))

            with st.expander("📋 场景需求详情（完整JSON）", expanded=True):
                st.json(scenario_req)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**设备列表**")
                for d in scenario_req.get("devices", []):
                    st.markdown(f"  - {d}")
                st.markdown("**负荷列表**")
                for l in scenario_req.get("loads", []):
                    st.markdown(f"  - {l}")
            with col_b:
                st.markdown("**优化目标**")
                for o in scenario_req.get("objectives", []):
                    st.markdown(f"  - {o}")
                st.markdown("**约束条件**")
                for c in scenario_req.get("constraints", []):
                    st.markdown(f"  - {c}")

            st.markdown("**🔎 自我审查**: " + agent1_result.get("self_review", ""))

        except Exception as e:
            st.error(f"❌ Agent 1 错误: {str(e)}")
            st.stop()

        # --- Agent 2 ---
        st.markdown("---")
        st.markdown("### 🤖 步骤 3/4: 数学建模Agent (A2)")
        st.caption("根据场景需求和知识图谱多路径知识生成数学模型...")
        try:
            agent2 = MathModelerAgent()
            agent2_result = agent2.run({
                "scenario_requirement": scenario_req,
                "knowledge_graph_context": kg_context,
                "scenario_type": scenario_req.get("scenario_type", "")
            })
            math_model = agent2_result["output"]

            st.success(f"✅ 数学模型生成完成: **{math_model.get('model_name', 'MG_Optimization_Model')}**")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("模型名称", math_model.get("model_name", "N/A"))
            with col_m2:
                st.metric("决策变量数", len(math_model.get("decision_variables", {})))
            with col_m3:
                st.metric("约束条件数", len(math_model.get("constraints", {})))

            with st.expander("📋 数学模型详情（完整JSON）", expanded=True):
                st.json(math_model)

            obj = math_model.get("objective_function", {})
            st.markdown("#### 目标函数")
            st.markdown(f"**类型**: {obj.get('type', 'N/A')}")
            st.markdown(f"**表达式**: {obj.get('expression', 'N/A')}")

            st.markdown("#### 决策变量")
            dvars = math_model.get("decision_variables", {})
            for var_name, var_info in dvars.items():
                with st.expander(f"  `{var_name}`"):
                    st.json(var_info if isinstance(var_info, dict) else {"value": var_info})

            st.markdown("#### 约束条件")
            constraints = math_model.get("constraints", {})
            col_con1, col_con2 = st.columns(2)
            for i, (con_name, con_info) in enumerate(constraints.items()):
                col = col_con1 if i % 2 == 0 else col_con2
                with col:
                    with st.expander(f"  `{con_name}`"):
                        st.json(con_info if isinstance(con_info, dict) else {"expression": str(con_info)})

            st.markdown("**🔎 自我审查**: " + agent2_result.get("self_review", ""))

        except Exception as e:
            st.error(f"❌ Agent 2 错误: {str(e)}")
            st.stop()

        # --- Agent 3 ---
        st.markdown("---")
        st.markdown("### 🤖 步骤 4/4: 代码合成Agent (A3)")
        st.caption("将数学模型转换为可执行的Python优化代码...")
        try:
            agent3 = CodeSynthesizerAgent()
            agent3_result = agent3.run({
                "math_model": math_model,
                "scenario_type": scenario_req.get("scenario_type", "unknown"),
                "kg_context": kg_context
            })
            generated_code = agent3_result["output"]
            st.success("✅ Python代码生成完成！")

            st.metric("代码行数", len(generated_code.split("\n")))

            st.markdown("#### 📄 完整代码")
            st.code(generated_code, language="python")

            # 代码片段解析
            code_lines = generated_code.split("\n")
            st.markdown("#### 🔑 关键代码片段解析")
            snippet_tabs = st.tabs(["导入模块", "模型构建", "求解逻辑"])

            with snippet_tabs[0]:
                import_section = "\n".join([l for l in code_lines if "import" in l.lower() or "from" in l.lower()])
                st.code(import_section, language="python") if import_section else st.info("未检测到import语句")

            with snippet_tabs[1]:
                model_keywords = ["model", "problem", "optimize", "minimize", "maximize", "LpProblem", "Model"]
                model_start = -1
                for i, line in enumerate(code_lines):
                    if any(kw in line.lower() for kw in model_keywords):
                        model_start = i
                        break
                if model_start >= 0:
                    st.code("\n".join(code_lines[model_start:min(model_start + 50, len(code_lines))]), language="python")
                else:
                    st.info("未检测到显式模型定义区域")

            with snippet_tabs[2]:
                solve_keywords = ["solve", ".optimize", "result", "optimize.minimize", "pulp.value"]
                solve_start = -1
                for i, line in enumerate(code_lines):
                    if any(kw in line.lower() for kw in solve_keywords):
                        solve_start = i
                        break
                if solve_start >= 0:
                    st.code("\n".join(code_lines[solve_start:min(solve_start + 30, len(code_lines))]), language="python")
                else:
                    st.info("未检测到显式求解区域")

            st.markdown("**🔎 自我审查**: " + agent3_result.get("self_review", ""))

            # --- 导出 ---
            st.markdown("---")
            st.markdown("### 📦 导出结果")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 下载Python代码 (.py)",
                    data=generated_code,
                    file_name="mg_optimization.py",
                    mime="text/x-python",
                    key="dl_code"
                )
            with col2:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                full_report = f"""# 微电网自动化建模报告
> 生成时间: {timestamp}
> 方法: 知识图谱增强的LLM Agent工作流 (多路径知识检索)

## 用户需求 (X)
{user_input}

## 知识图谱多路径检索 (K)
- 候选路径数: {len(kg_result.get('all_candidates', []))}
- 选中路径数: {len(kg_result.get('paths', []))}
- 语义簇数: {kg_result.get('num_clusters', 0)}
- 检索耗时: {kg_result.get('total_time', 0.0):.3f}s

## 场景识别结果 (Z1)
- 场景类型: {scenario_req.get('scenario_type', 'unknown')}
- 置信度: {scenario_req.get('scenario_confidence', 0):.2f}
- 设备: {', '.join(scenario_req.get('devices', []))}
- 负荷: {', '.join(scenario_req.get('loads', []))}
- 目标: {', '.join(scenario_req.get('objectives', []))}
- 约束: {', '.join(scenario_req.get('constraints', []))}

## 数学模型 (Z2)
- 模型名称: {math_model.get('model_name', 'N/A')}
- 决策变量数: {len(math_model.get('decision_variables', {}))}
- 约束条件数: {len(math_model.get('constraints', {}))}
- 目标函数: {math_model.get('objective_function', {}).get('expression', 'N/A')}

## 生成的Python代码 (C)
```python
{generated_code}
```

---
*由知识图谱增强的LLM Agent工作流生成*
"""
                st.download_button(
                    label="📄 导出完整报告 (.md)",
                    data=full_report,
                    file_name="mg_generation_report.md",
                    mime="text/markdown",
                    key="dl_report"
                )

            # 完整JSON导出
            st.download_button(
                label="📊 导出完整JSON (场景+模型)",
                data=json.dumps({
                    "scenario_requirement": scenario_req,
                    "math_model": math_model,
                    "kg_retrieval_stats": {
                        "total_candidates": len(kg_result.get('all_candidates', [])),
                        "selected_paths": len(kg_result.get('paths', [])),
                        "num_clusters": kg_result.get('num_clusters', 0),
                        "total_time": kg_result.get('total_time', 0.0)
                    }
                }, ensure_ascii=False, indent=2),
                file_name="mg_full_output.json",
                mime="application/json",
                key="dl_json"
            )

            st.success("🎉 所有步骤执行完成！")

        except Exception as e:
            st.error(f"❌ Agent 3 错误: {str(e)}")
            st.stop()
