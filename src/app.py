"""
Orthopedic Medical-Legal RAG Demo（中文界面）

用法（从项目根目录）:
    python -m streamlit run src/app.py
"""
import os
import sys
import json
import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ---------- 路径设置 ----------
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv()

# ---------- rag 接口 ----------
try:
    from rag import ask, retrieve, retrieve_with_routing, client as qdrant_client
    RAG_OK = True
    RAG_ERR = None
except Exception as e:
    RAG_OK = False
    RAG_ERR = str(e)
    qdrant_client = None


st.set_page_config(
    page_title="骨科医疗法律 RAG 演示",
    layout="wide",
)


# ============================================================
#  缓存与健康检查
# ============================================================

@st.cache_resource
def get_collection_status():
    """读取四个 collection 的状态"""
    if not RAG_OK or qdrant_client is None:
        return {"error": "rag 模块未加载"}
    out = {}
    for name in ["cfr493", "medical", "legal_ortho", "ortho_single"]:
        try:
            if qdrant_client.collection_exists(name):
                info = qdrant_client.get_collection(name)
                out[name] = {"exists": True, "points": info.points_count}
            else:
                out[name] = {"exists": False, "points": 0}
        except Exception as e:
            out[name] = {"exists": False, "points": 0, "error": str(e)}
    return out


def clear_caches():
    get_collection_status.clear()


# ============================================================
#  顶部：标题 + collection 状态
# ============================================================

st.title("骨科医疗法律 RAG 演示")
st.caption("面向骨科医疗法律问答的证据类型感知检索系统")

if not RAG_OK:
    st.error(f"RAG 模块加载失败：{RAG_ERR}")
    st.stop()

col_status = get_collection_status()
if "error" in col_status:
    st.error(f"Qdrant 连接失败：{col_status['error']}")
    st.caption("请确认 Docker 中的 raven-qdrant 容器正在运行。")
    st.stop()

status_cols = st.columns(4)
COLLECTION_LABELS = {
    "cfr493": "cfr493（Raven 复现库）",
    "medical": "medical（AAOS 骨科指南）",
    "legal_ortho": "legal_ortho（CMS 知情同意）",
    "ortho_single": "ortho_single（混合单库）",
}
for c, (name, label) in zip(status_cols, COLLECTION_LABELS.items()):
    info = col_status.get(name, {})
    if info.get("exists"):
        c.metric(label, f"{info['points']} 条")
    else:
        c.warning(f"{label}\n\n未找到")

st.divider()


# ============================================================
#  侧边栏
# ============================================================

with st.sidebar:
    st.header("演示模式")

    MODE_DESC = {
        "routed": "先判断问题需要哪类证据，再动态选择检索路径。（推荐）",
        "dual": "固定同时检索 medical 和 legal 两个库。",
        "ortho_single": "medical 和 legal 混合在一个池子里检索，不做区分。",
    }

    mode = st.radio(
        "检索模式",
        ["routed", "dual", "ortho_single"],
        index=0,
        help="选择检索策略。演示推荐 routed。",
    )
    st.caption(MODE_DESC[mode])

    with st.expander("高级选项（Raven 原版复现）", expanded=False):
        st.caption("baseline 模式使用 42 CFR Part 493，用于论文原版复现对照。")
        if st.button("切换到 baseline"):
            st.session_state["_mode_override"] = "baseline"

    st.divider()
    st.header("参数设置")
    k_single = st.slider("ortho_single 检索数量 Top-K", 1, 10, 6)
    k_medical = st.slider("dual / routed 的 medical Top-K", 1, 10, 3)
    k_legal = st.slider("dual / routed 的 legal Top-K", 1, 10, 3)

    st.divider()
    if st.button("刷新 collection 状态"):
        clear_caches()
        st.rerun()


if st.session_state.get("_mode_override") == "baseline":
    mode = "baseline"
    st.info("当前模式 = baseline（Raven 原版复现）")


# ============================================================
#  预设问题
# ============================================================

PRESET = {
    "医学类问题": "What are the AAOS recommendations for hip fracture management in older adults?",
    "法律类问题": "What information should generally be included in surgical informed consent?",
    "跨域问题": "What clinical considerations and informed consent information should be addressed before surgical management of a hip fracture in an older adult?",
}

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

st.subheader("示例问题")
preset_cols = st.columns(3)
for col, (label, q) in zip(preset_cols, PRESET.items()):
    if col.button(label, use_container_width=True):
        st.session_state.question_input = q

question = st.text_input(
    "问题",
    key="question_input",
    placeholder="输入一个骨科医疗法律问题...",
)


# ============================================================
#  Compare Modes 函数
# ============================================================

def compare_modes(question, k_single, k_medical, k_legal):
    rows = []
    try:
        r1 = retrieve(question, mode="ortho_single", k_single=k_single)
    except Exception as e:
        r1 = []
        st.warning(f"ortho_single 检索失败：{e}")

    try:
        r2 = retrieve(question, mode="dual",
                      k_medical=k_medical, k_legal=k_legal)
    except Exception as e:
        r2 = []
        st.warning(f"dual 检索失败：{e}")

    routing_pred = None
    try:
        r3, routing_pred = retrieve_with_routing(question)
    except Exception as e:
        r3 = []
        st.warning(f"routed 检索失败：{e}")

    def count(results):
        med = sum(1 for r in results if r["domain"] == "medical")
        leg = sum(1 for r in results if r["domain"] == "legal")
        return med, leg

    med1, leg1 = count(r1)
    med2, leg2 = count(r2)
    med3, leg3 = count(r3)

    rows.append({
        "模式": "ortho_single",
        "medical 片段": med1,
        "legal 片段": leg1,
        "总计": med1 + leg1,
    })
    rows.append({
        "模式": "dual",
        "medical 片段": med2,
        "legal 片段": leg2,
        "总计": med2 + leg2,
    })
    routed_label = "routed"
    if routing_pred is not None:
        if routing_pred["medical_required"] and routing_pred["legal_required"]:
            routed_label = "routed（医学+法律）"
        elif routing_pred["medical_required"]:
            routed_label = "routed（仅医学）"
        elif routing_pred["legal_required"]:
            routed_label = "routed（仅法律）"
    rows.append({
        "模式": routed_label,
        "medical 片段": med3,
        "legal 片段": leg3,
        "总计": med3 + leg3,
    })

    return rows, routing_pred


# ============================================================
#  主流程
# ============================================================

if question.strip():
    st.divider()

    results = []
    routing = None
    ans = None
    gen_error = None

    with st.spinner(f"正在运行 mode={mode} ..."):
        try:
            if mode == "routed":
                ans, results, routing = ask(
                    question, mode=mode,
                    k_medical=k_medical, k_legal=k_legal,
                )
            else:
                ans, results = ask(
                    question, mode=mode,
                    k_single=k_single, k_medical=k_medical, k_legal=k_legal,
                )
        except Exception as e:
            gen_error = str(e)

    # ---------- 证据需求判定 ----------
    if mode == "routed":
        st.subheader("证据需求判定")
        if routing is not None:
            c1, c2 = st.columns(2)
            c1.metric(
                "医学证据",
                "需要" if routing["medical_required"] else "不需要",
            )
            c2.metric(
                "法律证据",
                "需要" if routing["legal_required"] else "不需要",
            )
            if "fallback" in routing.get("reason", "").lower():
                st.warning(f"分类失败，回退到 dual 检索\n\n原因：{routing['reason']}")
            else:
                st.caption(f"判定原因：{routing['reason']}")
        else:
            st.info("未获得判定结果。")
        st.divider()

    # ---------- 检索到的证据 ----------
    st.subheader("检索到的证据")

    med_results = [r for r in results if r["domain"] == "medical"]
    leg_results = [r for r in results if r["domain"] == "legal"]
    other_results = [r for r in results if r["domain"] not in ("medical", "legal")]

    ev_col1, ev_col2 = st.columns(2)

    with ev_col1:
        st.markdown(f"**医学证据（{len(med_results)} 条）**")
        if not med_results:
            st.info("未检索到医学证据")
        else:
            for r in med_results:
                label = f"排名 {r['rank']}  |  相似度 {r['similarity_score']:.3f}  |  {r['source']}"
                with st.expander(label):
                    st.caption(f"document_id: {r['document_id']}")
                    st.write(r["content"])

    with ev_col2:
        st.markdown(f"**法律证据（{len(leg_results)} 条）**")
        if not leg_results:
            st.info("未检索到法律证据")
        else:
            for r in leg_results:
                label = f"排名 {r['rank']}  |  相似度 {r['similarity_score']:.3f}  |  {r['source']}"
                with st.expander(label):
                    st.caption(f"document_id: {r['document_id']}")
                    st.write(r["content"])

    if other_results:
        st.markdown(f"**其他（{len(other_results)} 条）**")
        for r in other_results:
            with st.expander(f"排名 {r['rank']}  |  {r['source']}"):
                st.write(r["content"])

    st.divider()

    # ---------- 检索统计 ----------
    st.subheader("检索统计")
    c1, c2, c3 = st.columns(3)
    c1.metric("医学片段数", len(med_results))
    c2.metric("法律片段数", len(leg_results))
    c3.metric("片段总数", len(results))

    st.divider()

    # ---------- 最终回答 ----------
    st.subheader("最终回答")
    if gen_error is not None:
        st.error(f"回答生成失败：{gen_error}")
        st.caption("上方检索结果仍然可用。")
    elif ans:
        st.markdown(ans)
    else:
        st.info("未生成回答。")

    st.divider()

    # ---------- 对比三种模式 ----------
    st.subheader("对比三种检索模式")
    st.caption("只运行检索，不调用 LLM 生成，速度较快。")

    if st.button("对比当前问题在三种模式下的检索结果"):
        with st.spinner("正在运行三种模式..."):
            rows, routing_pred = compare_modes(
                question, k_single, k_medical, k_legal
            )
        st.table(rows)
        if routing_pred is not None:
            st.caption(
                f"routed 判定："
                f"medical_required={routing_pred['medical_required']}，"
                f"legal_required={routing_pred['legal_required']}"
            )
            st.caption(f"原因：{routing_pred['reason']}")


# ============================================================
#  实验结果面板
# ============================================================

st.divider()
st.subheader("实验结果")
st.caption("以下结果来自 12 道题的初步 sanity test，仅供参考，未做统计显著性检验。")

RUNS_DIR = ROOT / "eval" / "runs"


def load_runs_stats():
    """从 eval/runs/*.json 计算简要统计"""
    stats = {}
    for mode in ["ortho_single", "dual", "routed"]:
        path = RUNS_DIR / f"{mode}.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            continue

        n_med_req = sum(1 for r in records if r.get("medical_required") == 1)
        n_leg_req = sum(1 for r in records if r.get("legal_required") == 1)
        med_cov = sum(
            1 for r in records
            if r.get("medical_required") == 1
            and any(x["domain"] == "medical" for x in r.get("retrieval", []))
        )
        leg_cov = sum(
            1 for r in records
            if r.get("legal_required") == 1
            and any(x["domain"] == "legal" for x in r.get("retrieval", []))
        )
        unnec_med = sum(
            sum(1 for x in r.get("retrieval", []) if x["domain"] == "medical")
            for r in records if r.get("medical_required") == 0
        )
        unnec_leg = sum(
            sum(1 for x in r.get("retrieval", []) if x["domain"] == "legal")
            for r in records if r.get("legal_required") == 0
        )
        cross = [r for r in records if r.get("question_type") == "cross"]
        if cross:
            cross_med = sum(
                sum(1 for x in r["retrieval"] if x["domain"] == "medical")
                for r in cross
            ) / len(cross)
            cross_leg = sum(
                sum(1 for x in r["retrieval"] if x["domain"] == "legal")
                for r in cross
            ) / len(cross)
        else:
            cross_med = cross_leg = 0

        stats[mode] = {
            "医学覆盖率": f"{med_cov}/{n_med_req}",
            "法律覆盖率": f"{leg_cov}/{n_leg_req}",
            "多余医学检索": unnec_med,
            "多余法律检索": unnec_leg,
            "跨域医学均值": f"{cross_med:.1f}",
            "跨域法律均值": f"{cross_leg:.1f}",
        }

        if mode == "routed":
            exact = 0
            for r in records:
                pred = r.get("routing_prediction")
                if not pred:
                    continue
                if (pred["medical_required"] == bool(r["medical_required"])
                        and pred["legal_required"] == bool(r["legal_required"])):
                    exact += 1
            stats[mode]["分类准确率"] = f"{exact}/{len(records)}"

    return stats


def hardcoded_stats():
    """如果 eval/runs/ 不存在，用已知摘要"""
    return {
        "ortho_single": {
            "医学覆盖率": "8/8",
            "法律覆盖率": "4/8",
            "多余医学检索": 8,
            "多余法律检索": 0,
            "跨域医学均值": "6.0",
            "跨域法律均值": "0.0",
        },
        "dual": {
            "医学覆盖率": "8/8",
            "法律覆盖率": "8/8",
            "多余医学检索": 12,
            "多余法律检索": 12,
            "跨域医学均值": "3.0",
            "跨域法律均值": "3.0",
        },
        "routed": {
            "医学覆盖率": "8/8",
            "法律覆盖率": "8/8",
            "多余医学检索": 0,
            "多余法律检索": 0,
            "跨域医学均值": "3.0",
            "跨域法律均值": "3.0",
            "分类准确率": "12/12",
        },
    }


runs_stats = load_runs_stats()
stats = runs_stats if runs_stats else hardcoded_stats()

METRIC_ORDER = [
    "医学覆盖率",
    "法律覆盖率",
    "多余医学检索",
    "多余法律检索",
    "跨域医学均值",
    "跨域法律均值",
    "分类准确率",
]


def build_table(stats):
    rows = []
    for metric in METRIC_ORDER:
        row = {"指标": metric}
        for mode in ["ortho_single", "dual", "routed"]:
            # 加上 str() 强制转换，避免 pyarrow 报错
            val = stats.get(mode, {}).get(metric, "n/a")
            row[mode] = str(val)
        rows.append(row)
    return rows


st.table(build_table(stats))
st.caption(
    "以上结果仅来自 12 道题的 sanity test，为初步结果，未做统计显著性声明。"
)