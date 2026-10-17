"""
对 eval/questions.csv 中的问题，分别用 ortho_single / dual / routed 模式检索，
保存结构化结果到 eval/runs/，并输出对比统计。

用法:
    python eval/run_eval.py

输出:
    eval/runs/ortho_single.json
    eval/runs/dual.json
    eval/runs/routed.json
"""
import os
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag import retrieve, retrieve_with_routing  # noqa: E402

load_dotenv()

QUESTIONS_CSV = ROOT / "eval" / "questions.csv"
RUNS_DIR = ROOT / "eval" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

K_SINGLE = 6
K_MED = 3
K_LEG = 3


def run_mode(questions, mode):
    records = []
    for _, row in questions.iterrows():
        qid = row["question_id"]
        q = row["question"]
        qtype = row["question_type"]
        med_req_gt = int(row["medical_required"])
        leg_req_gt = int(row["legal_required"])

        routing = None
        if mode == "ortho_single":
            results = retrieve(q, mode="ortho_single", k_single=K_SINGLE)
        elif mode == "dual":
            results = retrieve(q, mode="dual", k_medical=K_MED, k_legal=K_LEG)
        elif mode == "routed":
            results, routing = retrieve_with_routing(q)
        else:
            raise ValueError(mode)

        rec = {
            "question_id": qid,
            "question": q,
            "question_type": qtype,
            "medical_required": med_req_gt,
            "legal_required": leg_req_gt,
            "mode": mode,
            "retrieval": results,
        }
        if routing is not None:
            rec["routing_prediction"] = routing
        records.append(rec)
    return records


# ============================================================
#  Retrieval 统计（适用于所有 mode）
# ============================================================

def print_retrieval_stats(records, mode):
    print(f"\n{'=' * 80}")
    print(f"Retrieval stats for mode={mode}")
    print(f"{'=' * 80}")

    n = len(records)
    med_covered = sum(
        1 for r in records if r["medical_required"] == 1
        and any(x["domain"] == "medical" for x in r["retrieval"])
    )
    leg_covered = sum(
        1 for r in records if r["legal_required"] == 1
        and any(x["domain"] == "legal" for x in r["retrieval"])
    )
    n_med_req = sum(1 for r in records if r["medical_required"] == 1)
    n_leg_req = sum(1 for r in records if r["legal_required"] == 1)

    print(f"  total questions: {n}")
    print(f"  Required Medical Coverage: {med_covered}/{n_med_req}")
    print(f"  Required Legal Coverage:   {leg_covered}/{n_leg_req}")

    # Unnecessary domain retrieval
    unnec_med_total = 0
    unnec_leg_total = 0
    print("\n  Unnecessary domain retrieval:")
    for r in records:
        if r["medical_required"] == 0:
            c = sum(1 for x in r["retrieval"] if x["domain"] == "medical")
            unnec_med_total += c
            if c > 0:
                print(f"    {r['question_id']} (legal-only): {c} medical chunks")
        if r["legal_required"] == 0:
            c = sum(1 for x in r["retrieval"] if x["domain"] == "legal")
            unnec_leg_total += c
            if c > 0:
                print(f"    {r['question_id']} (medical-only): {c} legal chunks")
    print(f"    total unnecessary medical: {unnec_med_total}")
    print(f"    total unnecessary legal:   {unnec_leg_total}")

    # Cross-domain balance
    print("\n  Cross-domain balance (Q009-Q012):")
    for r in records:
        if r["question_type"] == "cross":
            c = Counter(x["domain"] for x in r["retrieval"])
            print(f"    {r['question_id']}: medical={c.get('medical', 0)}  legal={c.get('legal', 0)}")

    # Mean similarity
    print("\n  Mean similarity score by domain:")
    for dom in ["medical", "legal"]:
        scores = [x["similarity_score"] for r in records
                  for x in r["retrieval"] if x["domain"] == dom]
        if scores:
            print(f"    {dom:8s} n={len(scores):3d}  mean={sum(scores) / len(scores):.4f}")


# ============================================================
#  Routing 统计（仅 routed 模式）
# ============================================================

def print_routing_stats(records):
    print(f"\n{'=' * 80}")
    print("Routing evaluation")
    print(f"{'=' * 80}")

    n = len(records)
    exact = 0
    med_correct = 0
    leg_correct = 0

    print("\n  Per-question routing:")
    for r in records:
        pred = r["routing_prediction"]
        gt_med = bool(r["medical_required"])
        gt_leg = bool(r["legal_required"])
        p_med = pred["medical_required"]
        p_leg = pred["legal_required"]

        is_exact = (p_med == gt_med) and (p_leg == gt_leg)
        if is_exact:
            exact += 1
        if p_med == gt_med:
            med_correct += 1
        if p_leg == gt_leg:
            leg_correct += 1

        flag = "OK " if is_exact else "ERR"
        print(f"    [{flag}] {r['question_id']} ({r['question_type']:7s}) "
              f"gt=(med={gt_med},leg={gt_leg}) "
              f"pred=(med={p_med},leg={p_leg})")

    print(f"\n  Routing exact-match accuracy: {exact}/{n} = {exact/n:.3f}")
    print(f"  Medical requirement accuracy: {med_correct}/{n} = {med_correct/n:.3f}")
    print(f"  Legal requirement accuracy:   {leg_correct}/{n} = {leg_correct/n:.3f}")


# ============================================================
#  三方法对比汇总
# ============================================================

def print_summary(all_stats):
    print(f"\n{'=' * 80}")
    print("SUMMARY: ortho_single vs dual vs routed")
    print(f"{'=' * 80}")
    header = f"{'metric':40s} {'ortho_single':>14s} {'dual':>14s} {'routed':>14s}"
    print(header)
    print("-" * len(header))

    def fmt(v):
        return f"{v}" if v is not None else "n/a"

    for metric in [
        "Required Medical Coverage",
        "Required Legal Coverage",
        "Unnecessary Medical Retrieval",
        "Unnecessary Legal Retrieval",
        "Cross-domain Medical Count",
        "Cross-domain Legal Count",
    ]:
        row = f"{metric:40s}"
        for mode in ["ortho_single", "dual", "routed"]:
            v = all_stats.get(mode, {}).get(metric)
            row += f" {fmt(v):>14s}"
        print(row)


def compute_stats(records):
    """计算单项统计，返回 dict"""
    stats = {}
    n_med_req = sum(1 for r in records if r["medical_required"] == 1)
    n_leg_req = sum(1 for r in records if r["legal_required"] == 1)

    med_cov = sum(
        1 for r in records if r["medical_required"] == 1
        and any(x["domain"] == "medical" for x in r["retrieval"])
    )
    leg_cov = sum(
        1 for r in records if r["legal_required"] == 1
        and any(x["domain"] == "legal" for x in r["retrieval"])
    )
    unnec_med = sum(
        sum(1 for x in r["retrieval"] if x["domain"] == "medical")
        for r in records if r["medical_required"] == 0
    )
    unnec_leg = sum(
        sum(1 for x in r["retrieval"] if x["domain"] == "legal")
        for r in records if r["legal_required"] == 0
    )

    cross_med = []
    cross_leg = []
    for r in records:
        if r["question_type"] == "cross":
            cross_med.append(sum(1 for x in r["retrieval"] if x["domain"] == "medical"))
            cross_leg.append(sum(1 for x in r["retrieval"] if x["domain"] == "legal"))

    stats["Required Medical Coverage"] = f"{med_cov}/{n_med_req}"
    stats["Required Legal Coverage"] = f"{leg_cov}/{n_leg_req}"
    stats["Unnecessary Medical Retrieval"] = str(unnec_med)
    stats["Unnecessary Legal Retrieval"] = str(unnec_leg)
    stats["Cross-domain Medical Count"] = (
        f"avg={sum(cross_med)/len(cross_med):.1f}" if cross_med else "n/a"
    )
    stats["Cross-domain Legal Count"] = (
        f"avg={sum(cross_leg)/len(cross_leg):.1f}" if cross_leg else "n/a"
    )
    return stats


def main():
    if not QUESTIONS_CSV.exists():
        print(f"ERROR: {QUESTIONS_CSV} not found", file=sys.stderr)
        sys.exit(1)

    questions = pd.read_csv(QUESTIONS_CSV)
    print(f"loaded {len(questions)} questions")

    all_stats = {}
    all_records = {}

    for mode in ["ortho_single", "dual", "routed"]:
        print(f"\n{'#' * 80}")
        print(f"# Running mode={mode}")
        print(f"{'#' * 80}")
        records = run_mode(questions, mode)
        out_path = RUNS_DIR / f"{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"wrote: {out_path}")

        print_retrieval_stats(records, mode)
        if mode == "routed":
            print_routing_stats(records)

        all_stats[mode] = compute_stats(records)
        all_records[mode] = records

    print_summary(all_stats)


if __name__ == "__main__":
    main()