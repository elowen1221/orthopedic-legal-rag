# Paper-Ready Checklist (v1)

- [x] **1. Corpus Audit**：所有原始数据已通过 `corpus_manifest.csv` 溯源，字段包含作者、法律/医学组织、年份等。
- [x] **2. Index Parity Verification**：`ortho_single` 的 72 个 chunk 与 `medical`(37) + `legal_ortho`(35) 在向量、内容和 domain 标签上 100% 对齐（见 `verify_index_parity.py` 的 PASS 输出）。
- [x] **3. Ground Truth Annotation**：45 道测试题（15 医学 / 15 法律 / 15 跨域）已完成 Source ID 级别的预标注，状态为 `REVIEWED`。
- [x] **4. Retrieval Evaluation**：已计算 Domain Coverage, Source Recall@K, MRR, Chunk Precision, Unnecessary Retrieval Rate。
- [x] **5. Routing Evaluation**：Medical 准确率 93.3%，Legal 准确率 100%，并生成了混淆矩阵。
- [x] **6. Visualizations**：已使用 matplotlib 生成 4 张适用于论文的高清配图（保存在 `reports/figures/`）。
- [ ] **7. Answer-level Evaluation (Optional)**：由于本地算力限制，135 次 LLM-as-a-judge 耗时过长，可作为未来工作讨论。

## 核心发现 (Key Findings)

1. **RQ1 (Single-pool failure)**: `ortho_single` 在跨域问题上表现出明显的“医学偏科”（Medical Coverage 64.4% vs Legal 57.8%），且 MRR 虚高（因为它只找它认为最像的医学文本）。
2. **RQ2 & RQ3 (Dual's trade-off)**: 强制双源（`dual`）虽然把覆盖率拉到了 66.7%，但代价极大，每道题引入了约 1.0 个无用医学和 1.0 个无用法律 Chunk，导致 Precision 暴跌至 0.507。
3. **RQ4 (Routing's superiority)**: 动态路由（`routed`）在保持最高覆盖率（66.7%）的同时，把无用检索降到了 0.20 / 0.00，MRR 回升至 0.924，Precision 回升至 0.663。

## 可用命令

```powershell
# 运行检索评估
& $PY eval\run_eval.py

# 生成图表
& $PY eval\make_figures.py

# 启动 Demo
& $PY -m streamlit run src\app.py


---

### 3. 执行最终操作

覆盖完 `make_figures.py` 和建好 `PAPER_CHECKLIST.md` 后，在终端跑最后一遍：

```powershell
cd "E:\Orthopedic Law\raven_repro"
& "E:\Orthopedic Law\Orthopedic_Law__env\Scripts\python.exe" eval\make_figures.py