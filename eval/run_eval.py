import sys
import json
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag import retrieve, retrieve_with_routing

def safe_set(s):
    if pd.isna(s) or not str(s).strip(): return set()
    return set([x.strip() for x in str(s).split(",") if x.strip()])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="paper")
    parser.add_argument("--reviewed-only", action="store_true")
    args = parser.parse_args()

    csv_path = Path("eval/questions_paper.csv")
    out_dir = Path("reports/final/retrieval_runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).fillna("")
    if args.reviewed_only:
        df = df[(df["annotation_status"] == "REVIEWED") & (df["answerable"] == 1)]
    
    print(f"Loaded {len(df)} REVIEWED & answerable questions.")
    if len(df) == 0:
        print("No questions available. Please annotate first.")
        return

    modes = ["ortho_single", "dual", "routed"]
    results = {m: [] for m in modes}

    for idx, row in df.iterrows():
        qid, q_text = row["question_id"], row["question"]
        gold_med = safe_set(row["gold_medical_source_ids"])
        gold_leg = safe_set(row["gold_legal_source_ids"])
        gold_all = gold_med | gold_leg

        print(f"Evaluating {qid}...")
        for mode in modes:
            if mode == "routed":
                retrieved, routing = retrieve_with_routing(q_text)
            else:
                retrieved = retrieve(q_text, mode=mode)
            
            # 提取 Top-K 结果
            ret_sources = [r["source"] for r in retrieved]
            ret_domains = [r["domain"] for r in retrieved]
            
            # 计算指标
            # 1. Source Recall@K
            hit_sources = set(ret_sources) & gold_all
            source_recall = len(hit_sources) / len(gold_all) if gold_all else 0
            
            # 2. MRR
            mrr = 0.0
            for rank, src in enumerate(ret_sources, 1):
                if src in gold_all:
                    mrr = 1.0 / rank
                    break
                    
            # 3. Unnecessary Retrieval
            unnec_med = sum(1 for d in ret_domains if d == "medical") if (row["medical_required"]==0 and row["legal_required"]==1) else 0
            unnec_leg = sum(1 for d in ret_domains if d == "legal") if (row["medical_required"]==1 and row["legal_required"]==0) else 0

            results[mode].append({
                "question_id": qid,
                "question_type": row["question_type"],
                "medical_required": int(row["medical_required"]),
                "legal_required": int(row["legal_required"]),
                "source_recall": source_recall,
                "mrr": mrr,
                "unnec_med": unnec_med,
                "unnec_leg": unnec_leg,
                "med_covered": 1 if (row["medical_required"]==1 and "medical" in ret_domains) else 0,
                "leg_covered": 1 if (row["legal_required"]==1 and "legal" in ret_domains) else 0,
                "retrieved_sources": ret_sources,
                "retrieved_domains": ret_domains
            })

    for mode in modes:
        with open(out_dir / f"{mode}.json", "w", encoding="utf-8") as f:
            json.dump(results[mode], f, ensure_ascii=False, indent=2)
        print(f"Saved {mode}.json ({len(results[mode])} questions)")

if __name__ == "__main__":
    main()