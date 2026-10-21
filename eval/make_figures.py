import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

def make_figures():
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    print("Generating figures...")

    modes = ["ortho_single", "dual", "routed"]
    
    # 基于 run_eval.py 跑出的真实数据
    med_coverage = [64.4, 66.7, 66.7]
    leg_coverage = [57.8, 66.7, 66.7]
    unnec_med = [0.27, 1.00, 0.20]
    unnec_leg = [0.11, 1.00, 0.00]
    mrr = [1.000, 0.724, 0.924]
    precision = [0.681, 0.507, 0.663]
    
    x = np.arange(len(modes))
    width = 0.35

    # ==========================================
    # 图 1：Domain Coverage
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, med_coverage, width, label='Medical Coverage', color='#4C72B0')
    ax.bar(x + width/2, leg_coverage, width, label='Legal Coverage', color='#DD8452')
    
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Required Domain Coverage by Retrieval Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("reports/figures/domain_coverage.png", dpi=300)
    plt.close()

    # ==========================================
    # 图 2：Unnecessary Retrieval
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, unnec_med, width, label='Unnecessary Medical', color='#55A868')
    ax.bar(x + width/2, unnec_leg, width, label='Unnecessary Legal', color='#C44E52')
    
    ax.set_ylabel('Average Count per Question', fontsize=12)
    ax.set_title('Unnecessary Retrieval (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("reports/figures/unnecessary_retrieval.png", dpi=300)
    plt.close()

    # ==========================================
    # 图 3：MRR & Precision
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, mrr, width, label='MRR', color='#8172B3')
    ax.bar(x + width/2, precision, width, label='Chunk Precision@K', color='#937860')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Retrieval Quality: MRR and Precision', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("reports/figures/retrieval_quality.png", dpi=300)
    plt.close()

    # ==========================================
    # 图 4：Routing Confusion Matrix (无警告版)
    # ==========================================
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 混淆矩阵数据：行代表实际情况(Medical Required / Not Req)，列代表预测情况(No / Yes)
    # 原数据：TP=30, FP=3, FN=0, TN=12
    matrix = np.array([[12, 3], [0, 30]]) 
    
    cax = ax.matshow(matrix, cmap='Blues')
    fig.colorbar(cax)

    # 填充数字
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, str(z), ha='center', va='center', 
                color='white' if z > 15 else 'black', fontsize=16, fontweight='bold')
        
    # 严格先设置刻度，再设置标签，消除警告
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    
    ax.set_xticklabels(['Predicted No', 'Predicted Yes'], fontsize=11)
    ax.set_yticklabels(['Medical Not Req', 'Medical Required'], fontsize=11)
    
    ax.set_title('Routing Confusion Matrix (Medical Domain)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("reports/figures/routing_confusion_matrix.png", dpi=300)
    plt.close()

    print("✅ All 4 figures generated successfully in reports/figures/")

if __name__ == "__main__":
    make_figures()