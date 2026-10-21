"""
论文级别评估脚本 (Paper-ready Evaluation)
计算：Domain Coverage, Source Recall@K, Precision@K, MRR, Routing Confusion Matrix.

用法:
    python eval/run_eval.py
"""
import os
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag import retrieve, retrieve_with_routing  # noqa: E402

load_dotenv()

# ---------- 配置区 ----------
# 优先寻找论文数据集，如果没有则回退到旧的 sanity test
QUESTIONS_CSV = ROOT / "eval" / "questions_paper.csv"
if not QUESTIONS_CSV.exists():
    QUESTIONS_CSV = ROOT / "eval" / "questions.csv"
    print(f"NOTICE: questions_paper.csv not found, falling back to {QUESTIONS_CSV.name}")

RUNS_DIR = ROOT / "eval" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

K_SINGLE = 6
K_MED = 3
K_LEG = 3
MODES = ["ortho_single", "dual", "routed"]


def parse_gold_sources(s):
    """解析人工标注的 gold source ids（逗号分隔）"""
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def calc_retrieval_metrics(retrieved_items, gold_med, gold_leg, k_limit=6):
    """计算单题级别的检索指标"""
    gold_all = set(gold_med + gold_leg)
    med_req = len(gold_med) > 0
    leg_req = len(gold_leg) > 0
    
    has_med = any(x["domain"] == "medical" for x in retrieved_items)
    has_leg = any(x["domain"] == "legal" for x in retrieved_items)
    
    # 1. Domain Coverage (是否覆盖了需要的 domain)
    med_cov = 1 if (med_req and has_med) else 0
    leg_cov = 1 if (leg_req and has_leg) else 0
    
    # 2. Unnecessary Retrieval (是否引入了不需要的 domain)
    unnec_med = sum(1 for x in retrieved_items if x["domain"] == "medical" and not med_req)
    unnec_leg = sum(1 for x in retrieved_items if x["domain"] == "legal" and not leg_req)
    
    # 如果没有 Ground Truth，只返回 Coverage 指标
    if not gold_all:
        return {
            "med_cov": med_cov, "leg_cov": leg_cov,
            "unnec_med": unnec_med, "unnec_leg": unnec_leg,
            "recall": 0.0, "precision": 0.0, "mrr": 0.0
        }
        
    # 3. Retrieval Precision/Recall/MRR
    hits = 0
    mrr = 0.0
    first_hit = False
    retrieved_sources = set()
    
    for rank, item in enumerate(retrieved_items[:k_limit], 1):
        src = item["source"]
        retrieved_sources.add(src)
        if src in gold_all:
            hits += 1
            if not first_hit:
                mrr = 1.0 / rank
                first_hit = True
                
    # Source Recall@K = 召回的 distinct 正确 source / 总 gold source
    source_hits = len(retrieved_sources & gold_all)
    recall = source_hits / len(gold_all) if len(gold_all) > 0 else 0.0
    
    # Chunk Precision@K = 召回的属于 gold source 的 chunk / 实际召回的 chunk 数量
    precision = hits / len(retrieved_items) if len(retrieved_items) > 0 else 0.0
    
    return {
        "med_cov": med_cov, "leg_cov": leg_cov,
        "unnec_med": unnec_med, "unnec_leg": unnec_leg,
        "recall": recall, "precision": precision, "mrr": mrr
    }


def run_mode(questions, mode):
    records = []
    for _, row in questions.iterrows():
        qid = row["question_id"]
        q = row["question"]
        qtype = row.get("question_type", "UNKNOWN")
        
        # 如果没有人工标注列，默认为空列表
        gold_med = parse_gold_sources(row.get("gold_medical_source_ids", ""))
        gold_leg = parse_gold_sources(row.get("gold_legal_source_ids", ""))
        
        # 如果 CSV 里有 medical_required 列（0/1），则用来兜底评估 Coverage
        req_med = int(row.get("medical_required", 0))
        req_leg = int(row.get("legal_required", 0))

        routing = None
        if mode == "ortho_single":
            results = retrieve(q, mode="ortho_single", k_single=K_SINGLE)
        elif mode == "dual":
            results = retrieve(q, mode="dual", k_medical=K_MED, k_legal=K_LEG)
        elif mode == "routed":
            results, routing = retrieve_with_routing(q)
        else:
            raise ValueError(mode)

        metrics = calc_retrieval_metrics(results, gold_med, gold_leg, k_limit=K_SINGLE)
        
        rec = {
            "question_id": qid,
            "question": q,
            "question_type": qtype,
            "medical_required_gt": req_med,
            "legal_required_gt": req_leg,
            "gold_med": gold_med,
            "gold_leg": gold_leg,
            "mode": mode,
            "retrieval": results,
            "metrics": metrics
        }
        if routing is not None:
            rec["routing_prediction"] = routing
            
        records.append(rec)
    return records


def print_summary(records_by_mode):
    print("\n" + "=" * 80)
    print("PAPER-READY EVALUATION SUMMARY")
    print("=" * 80)
    
    header = f"{'Metric':<30} | {'ortho_single':>12} | {'dual':>12} | {'routed':>12}"
    print(header)
    print("-" * len(header))
    
    # 计算各模式的指标均值
    summaries = {}
    for mode, records in records_by_mode.items():
        n = len(records)
        sums = defaultdict(float)
        for r in records:
            m = r["metrics"]
            sums["med_cov"] += m["med_cov"]
            sums["leg_cov"] += m["leg_cov"]
            sums["unnec_med"] += m["unnec_med"]
            sums["unnec_leg"] += m["unnec_leg"]
            sums["recall"] += m["recall"]
            sums["precision"] += m["precision"]
            sums["mrr"] += m["mrr"]
            
        summaries[mode] = {k: v / n for k, v in sums.items()}
        
    # 打印各行指标
    metrics_to_print = [
        ("Domain Coverage (Medical)", "med_cov", "{:.1%}"),
        ("Domain Coverage (Legal)", "leg_cov", "{:.1%}"),
        ("Unnecessary Medical (count)", "unnec_med", "{:.2f}"),
        ("Unnecessary Legal (count)", "unnec_leg", "{:.2f}"),
        ("Source Recall@K", "recall", "{:.3f}"),
        ("Chunk Precision@K", "precision", "{:.3f}"),
        ("MRR", "mrr", "{:.3f}")
    ]
    
    for label, key, fmt in metrics_to_print:
        row = f"{label:<30} |"
        for mode in MODES:
            val = summaries.get(mode, {}).get(key, 0)
            row += f" {fmt.format(val):>12} |"
        print(row)

    # 打印 Routing 混淆矩阵
    if "routed" in records_by_mode:
        print("\n" + "=" * 80)
        print("ROUTING CONFUSION MATRIX (Based on Ground Truth medical_required / legal_required)")
        print("=" * 80)
        
        # TP, FP, FN, TN
        med_tp = med_fp = med_fn = med_tn = 0
        leg_tp = leg_fp = leg_fn = leg_tn = 0
        
        for r in records_by_mode["routed"]:
            pred = r.get("routing_prediction")
            if not pred: continue
            
            gt_m = r["medical_required_gt"] == 1
            gt_l = r["legal_required_gt"] == 1
            pr_m = pred["medical_required"]
            pr_l = pred["legal_required"]
            
            if gt_m and pr_m: med_tp += 1
            elif not gt_m and pr_m: med_fp += 1
            elif gt_m and not pr_m: med_fn += 1
            else: med_tn += 1
            
            if gt_l and pr_l: leg_tp += 1
            elif not gt_l and pr_l: leg_fp += 1
            elif gt_l and not pr_l: leg_fn += 1
            else: leg_tn += 1
            
        print("Medical Domain Routing:")
        print(f"  True Positive: {med_tp}  | False Positive: {med_fp}")
        print(f"  False Negative: {med_fn} | True Negative: {med_tn}")
        print(f"  Accuracy: {(med_tp + med_tn) / len(records_by_mode['routed']):.3f}")
        
        print("\nLegal Domain Routing:")
        print(f"  True Positive: {leg_tp}  | False Positive: {leg_fp}")
        print(f"  False Negative: {leg_fn} | True Negative: {leg_tn}")
        print(f"  Accuracy: {(leg_tp + leg_tn) / len(records_by_mode['routed']):.3f}")


def main():
    if not QUESTIONS_CSV.exists():
        print(f"ERROR: {QUESTIONS_CSV} not found", file=sys.stderr)
        sys.exit(1)

    questions = pd.read_csv(QUESTIONS_CSV)
    print(f"Loaded {len(questions)} questions from {QUESTIONS_CSV.name}")
    
    # 检查是否有 ground truth 标注
    if "gold_medical_source_ids" in questions.columns:
        reviewed = questions[questions.get("annotation_status", "") == "REVIEWED"]
        print(f"  - {len(reviewed)} questions are marked as REVIEWED (with ground truth).")
        if len(reviewed) == 0:
            print("  WARNING: No questions are REVIEWED. Precision/Recall/MRR will be 0.0.")
    else:
        print("  WARNING: No 'gold_medical_source_ids' column found. Only Domain Coverage will be calculated.")

    records_by_mode = {}
    for mode in MODES:
        print(f"\nEvaluating mode = {mode} ...")
        records = run_mode(questions, mode)
        records_by_mode[mode] = records
        
        # 保存到 JSON
        out_path = RUNS_DIR / f"{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  Saved detailed results to {out_path}")

    print_summary(records_by_mode)


if __name__ == "__main__":
    main()