import sys
import json
from pathlib import Path
import pandas as pd

# 路径设置
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag import ask, call_ollama_llm

# 严格 Judge Prompt
JUDGE_PROMPT = """You are an evaluator. Given a question, extracted evidence, and an answer.
1. Extract all medical/legal claims in the answer.
2. For each claim, check if the evidence supports it.
Output strictly JSON: {"claims": [{"claim": "...", "supported": true, "reason": "..."}]}"""

def evaluate_answers():
    csv_path = ROOT / "eval" / "questions_paper.csv"
    out_path = ROOT / "reports" / "answer_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print("Error: questions_paper.csv not found.")
        return

    df = pd.read_csv(csv_path).fillna("")
    # 只评估已经人工标注过的题目
    df = df[df["annotation_status"] == "REVIEWED"]
    
    if len(df) == 0:
        print("No REVIEWED questions found. Please annotate first.")
        return

    print(f"Loaded {len(df)} REVIEWED questions for answer evaluation.")
    results = []
    total_queries = len(df) * 3

    count = 0
    for idx, row in df.iterrows():
        qid = row["question_id"]
        q_text = row["question"]
        
        for mode in ["ortho_single", "dual", "routed"]:
            count += 1
            print(f"[{count}/{total_queries}] Evaluating QID={qid} | mode={mode} ...", end=" ", flush=True)
            
            try:
                # 适配 ask() 的返回值：routed 返回 3 个，其他返回 2 个
                if mode == "routed":
                    ans, evidence, routing = ask(q_text, mode=mode)
                else:
                    ans, evidence = ask(q_text, mode=mode)
                    routing = None

                # 构建 Evidence 文本
                ev_text = "\n".join([f"[{e['source']}] {e['content']}" for e in evidence])
                
                # 调用 LLM 裁判 (限制 prompt 长度，防止超时)
                user_prompt = f"Question: {q_text}\n\nEvidence:\n{ev_text[:4000]}\n\nAnswer:\n{ans}"
                judge_raw = call_ollama_llm(JUDGE_PROMPT, user_prompt)
                
                # 简单解析裁判结果
                judge_data = {"claims": [], "raw": judge_raw}
                if "{" in judge_raw and "}" in judge_raw:
                    try:
                        json_str = judge_raw[judge_raw.find("{"):judge_raw.rfind("}")+1]
                        judge_data = json.loads(json_str)
                    except:
                        pass

                results.append({
                    "question_id": qid,
                    "mode": mode,
                    "question": q_text,
                    "routing": routing,
                    "answer": ans,
                    "judge_output": judge_data
                })
                print("Done.")
                
            except Exception as e:
                print(f"Failed: {e}")
                results.append({
                    "question_id": qid,
                    "mode": mode,
                    "error": str(e)
                })

    # 保存结果
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Evaluation complete! Results saved to {out_path}")

if __name__ == "__main__":
    evaluate_answers()