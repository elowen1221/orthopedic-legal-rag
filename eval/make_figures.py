import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def make_figures():
    runs_dir = Path("reports/final/retrieval_runs")
    fig_dir = Path("reports/figures/final")
    fig_dir.mkdir(parents=True, exist_ok=True)

    modes = ["ortho_single", "dual", "routed"]
    data_all = {}
    for mode in modes:
        with open(runs_dir / f"{mode}.json") as f:
            data_all[mode] = json.load(f)

    n_questions = len(data_all["dual"]) # 取一个长度做 N

    # 计算画图数据
    med_cov = []
    leg_cov = []
    recall = []
    mrr = []
    unnec_med = []
    unnec_leg = []

    for mode in modes:
        d = data_all[mode]
        med_req = sum(1 for x in d if x["medical_required"]==1)
        leg_req = sum(1 for x in d if x["legal_required"]==1)
        
        med_cov.append((sum(x["med_covered"] for x in d) / med_req * 100) if med_req else 0)
        leg_cov.append((sum(x["leg_covered"] for x in d) / leg_req * 100) if leg_req else 0)
        recall.append(sum(x["source_recall"] for x in d) / len(d))
        mrr.append(sum(x["mrr"] for x in d) / len(d))
        unnec_med.append(sum(x["unnec_med"] for x in d) / len(d))
        unnec_leg.append(sum(x["unnec_leg"] for x in d) / len(d))

    x = np.arange(len(modes))
    width = 0.35

    # 图1：Coverage
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, med_cov, width, label='Medical', color='#4C72B0')
    ax.bar(x + width/2, leg_cov, width, label='Legal', color='#DD8452')
    ax.set_ylabel('Coverage (%)')
    ax.set_title(f'Required Domain Coverage (N={n_questions})')
    ax.set_xticks(x); ax.set_xticklabels(modes); ax.legend()
    ax.set_ylim(0, 100)
    plt.grid(axis='y', linestyle='--'); plt.tight_layout()
    plt.savefig(fig_dir / "domain_coverage.png", dpi=300); plt.close()

    # 图2：Unnecessary Retrieval
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, unnec_med, width, label='Unnecessary Medical', color='#55A868')
    ax.bar(x + width/2, unnec_leg, width, label='Unnecessary Legal', color='#C44E52')
    ax.set_ylabel('Avg Count per Question')
    ax.set_title(f'Unnecessary Retrieval (N={n_questions})')
    ax.set_xticks(x); ax.set_xticklabels(modes); ax.legend()
    plt.grid(axis='y', linestyle='--'); plt.tight_layout()
    plt.savefig(fig_dir / "unnecessary_retrieval.png", dpi=300); plt.close()

    # 图3：Recall & MRR
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, recall, width, label='Source Recall@K', color='#8172B3')
    ax.bar(x + width/2, mrr, width, label='Source MRR', color='#937860')
    ax.set_ylabel('Score')
    ax.set_title(f'Retrieval Quality (N={n_questions})')
    ax.set_xticks(x); ax.set_xticklabels(modes); ax.legend()
    ax.set_ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--'); plt.tight_layout()
    plt.savefig(fig_dir / "source_recall_mrr.png", dpi=300); plt.close()

    print(f"✅ 3 core final figures generated in {fig_dir} (N={n_questions})")

if __name__ == "__main__":
    make_figures()