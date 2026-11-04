import json
from pathlib import Path
import pandas as pd

def make_tables():
    runs_dir = Path("reports/final/retrieval_runs")
    out_dir = Path("reports/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["ortho_single", "dual", "routed"]
    rows = []

    for mode in modes:
        with open(runs_dir / f"{mode}.json") as f:
            data = json.load(f)
        
        n = len(data)
        med_req_n = sum(1 for d in data if d["medical_required"] == 1)
        leg_req_n = sum(1 for d in data if d["legal_required"] == 1)
        
        med_cov = sum(d["med_covered"] for d in data) / med_req_n if med_req_n else 0
        leg_cov = sum(d["leg_covered"] for d in data) / leg_req_n if leg_req_n else 0
        recall = sum(d["source_recall"] for d in data) / n
        mrr = sum(d["mrr"] for d in data) / n
        unnec_med = sum(d["unnec_med"] for d in data) / n
        unnec_leg = sum(d["unnec_leg"] for d in data) / n

        rows.append({
            "Mode": mode,
            "Medical Coverage": f"{med_cov*100:.1f}%",
            "Legal Coverage": f"{leg_cov*100:.1f}%",
            "Source Recall@K": f"{recall:.3f}",
            "Source MRR": f"{mrr:.3f}",
            "Unnecessary Medical": f"{unnec_med:.2f}",
            "Unnecessary Legal": f"{unnec_leg:.2f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table_retrieval.csv", index=False)
    print("✅ table_retrieval.csv generated (No Chunk metrics).")

if __name__ == "__main__":
    make_tables()