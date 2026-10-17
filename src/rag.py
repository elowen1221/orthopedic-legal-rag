import os
import json
import re
import requests
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ---------- 四个独立 collection ----------
RAVEN_LEGAL_COLLECTION = "cfr493"        # baseline 用
MEDICAL_COLLECTION = "medical"           # dual 医学侧
ORTHO_LEGAL_COLLECTION = "legal_ortho"   # dual 法律侧
ORTHO_SINGLE_COLLECTION = "ortho_single" # single-pool baseline

emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
client = QdrantClient(url=QDRANT_URL)

SYSTEM = """You are an AI assistant for answering questions about medical-legal matters in orthopedic practice.
You are given extracted text from two sources:
- [medical] domain: orthopedic clinical guidelines and evidence
- [legal] domain: regulatory guidance on informed consent and surgical practice

Provide a professional and complete answer.
Base your answer solely on the information provided.
Do not make up answers or use sources other than the provided text.
For each assertion, cite the source section.
If the retrieved evidence is insufficient to answer fully, explicitly state what is missing.
If you don't know, say 'I am not sure.'
If the question is not about medical-legal matters, say you only answer such questions."""


# ============================================================
#  Part 0: 原生 Ollama 调用（解决 LangChain 的 404 Bug）
# ============================================================

def call_ollama_llm(system_prompt: str, user_prompt: str) -> str:
    """使用 requests 直接请求 Ollama，极其稳定，不受库版本影响"""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 512  # 限制输出长度，防止长文本生成卡死
        }
    }
    try:
        # 将 timeout 延长到 300 秒，避免长上下文生成超时
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Ollama LLM call failed: {e}")


# ============================================================
#  Part 1: 结构化输出 & 检索
# ============================================================

def _to_structured(hits, domain):
    out = []
    for i, h in enumerate(hits):
        payload = h.payload or {}
        dom = payload.get("domain", domain)
        out.append({
            "document_id": f"{dom}:{payload.get('chunk_id', h.id)}",
            "source": payload.get("source", ""),
            "domain": dom,
            "similarity_score": float(h.score),
            "rank": i + 1,
            "content": payload.get("content", ""),
        })
    return out


# ============================================================
#  Part 2: Evidence Requirement Routing 分类器
# ============================================================

ROUTING_SYSTEM = """You are a query classifier for a medical-legal RAG system.
Given a question, determine whether answering it requires:

A. medical_required: orthopedic clinical evidence, clinical guidelines, diagnoses, treatments, surgical procedures, clinical outcomes.
B. legal_required: laws, regulations, informed consent, documentation requirements, legal cases, malpractice, physician duties.

Rules:
- If the question is purely about how to treat a disease, surgery, or clinical guideline -> medical_required=true, legal_required=false.
- If the question is purely about law, court cases, informed consent rules, or malpractice -> medical_required=false, legal_required=true.
- If the question explicitly asks for BOTH clinical details and legal/consent information -> both true.
- If you are unsure, set both true.

Examples:
Q: "What are the AAOS recommendations for hip fracture management?" -> {"medical_required": true, "legal_required": false, "reason": "pure clinical guideline"}
Q: "What are the requirements for surgical informed consent?" -> {"medical_required": false, "legal_required": true, "reason": "pure legal requirement"}
Q: "What clinical considerations and informed consent should be addressed for hip fracture surgery?" -> {"medical_required": true, "legal_required": true, "reason": "cross-domain"}

Output ONLY a valid JSON object with this exact schema:
{"medical_required": true, "legal_required": false, "reason": "short explanation"}

Do NOT answer the question itself. Do NOT add markdown. Output JSON only."""


def _parse_routing_json(raw: str):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    fixed = raw.replace("'", '"').replace("True", "true").replace("False", "false")
    m = re.search(r"\{.*?\}", fixed, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def classify_evidence_requirements(question: str) -> dict:
    fallback = {
        "medical_required": True,
        "legal_required": True,
        "reason": "fallback: classification failed, default to dual",
    }

    try:
        raw = call_ollama_llm(ROUTING_SYSTEM, f"Question: {question}")
    except Exception as e:
        fallback["reason"] = f"fallback: LLM call failed ({e})"
        return fallback

    data = _parse_routing_json(raw)
    if not isinstance(data, dict):
        fallback["reason"] = f"fallback: JSON parse failed, raw={raw[:120]}"
        return fallback

    med = bool(data.get("medical_required", False))
    leg = bool(data.get("legal_required", False))
    reason = str(data.get("reason", ""))[:300]

    if not med and not leg:
        return {
            "medical_required": True,
            "legal_required": True,
            "reason": "fallback: both false from LLM, default to dual",
        }

    return {"medical_required": med, "legal_required": leg, "reason": reason}


# ============================================================
#  Part 3: 基础检索路径
# ============================================================

def retrieve_single(query, k=5):
    vec = emb.embed_query(query)
    result = client.query_points(collection_name=RAVEN_LEGAL_COLLECTION, query=vec, limit=k)
    return _to_structured(result.points, domain="legal")


def retrieve_medical(query, k=3):
    if not client.collection_exists(MEDICAL_COLLECTION): return []
    vec = emb.embed_query(query)
    result = client.query_points(collection_name=MEDICAL_COLLECTION, query=vec, limit=k)
    return _to_structured(result.points, domain="medical")


def retrieve_legal(query, k=3):
    if not client.collection_exists(ORTHO_LEGAL_COLLECTION): return []
    vec = emb.embed_query(query)
    result = client.query_points(collection_name=ORTHO_LEGAL_COLLECTION, query=vec, limit=k)
    return _to_structured(result.points, domain="legal")


def retrieve_ortho_single(query, k=6):
    if not client.collection_exists(ORTHO_SINGLE_COLLECTION): return []
    vec = emb.embed_query(query)
    result = client.query_points(collection_name=ORTHO_SINGLE_COLLECTION, query=vec, limit=k)
    return _to_structured(result.points, domain="mixed")


# ============================================================
#  Part 4: 统一检索与生成入口
# ============================================================

def retrieve(query, mode="baseline", k=5, k_medical=3, k_legal=3, k_single=6):
    if mode == "baseline":
        return retrieve_single(query, k=k)
    elif mode == "ortho_single":
        return retrieve_ortho_single(query, k=k_single)
    elif mode == "dual":
        return retrieve_medical(query, k=k_medical) + retrieve_legal(query, k=k_legal)
    else:
        raise ValueError(f"unknown mode: {mode}")


def retrieve_with_routing(query, k_single_domain=6, k_cross=3):
    routing = classify_evidence_requirements(query)
    med_req = routing["medical_required"]
    leg_req = routing["legal_required"]

    if med_req and not leg_req:
        results = retrieve_medical(query, k=k_single_domain)
    elif leg_req and not med_req:
        results = retrieve_legal(query, k=k_single_domain)
    else:
        results = retrieve_medical(query, k=k_cross) + retrieve_legal(query, k=k_cross)

    return results, routing


def build_context(results):
    return "\n\n".join(f"[{r['domain']}|{r['source']}]\n{r['content']}" for r in results)


def ask(question, mode="baseline", k=5, k_medical=3, k_legal=3, k_single=6):
    routing = None
    if mode == "routed":
        results, routing = retrieve_with_routing(question)
    else:
        results = retrieve(question, mode=mode, k=k, k_medical=k_medical, k_legal=k_legal, k_single=k_single)

    context = build_context(results)
    user_prompt = f"""<Relevant source document chunks>
{context}
</Relevant source document chunks>

<User query>
{question}
</User query>"""

    # 关键修改：即使 LLM 超时报错，也要把检索到的证据返回给前端，不让页面崩溃
    try:
        ans = call_ollama_llm(SYSTEM, user_prompt)
    except Exception as e:
        ans = f"⚠️ **回答生成超时或被中断**：{e}\n\n但上方的检索证据是完好的，请参考证据进行判断。"

    if mode == "routed":
        return ans, results, routing
    return ans, results


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    q = " ".join(sys.argv[2:]) or "Can I refer proficiency testing samples to another lab?"
    print(f"mode={mode}\nQ: {q}\n" + "=" * 80)
    
    if mode == "routed":
        ans, results, routing = ask(q, mode=mode)
        print("Routing:", json.dumps(routing, ensure_ascii=False))
    else:
        ans, results = ask(q, mode=mode)
        
    print("=" * 80 + "\n" + ans + "\n\n--- Sources ---")
    for r in results:
        print(f"  [{r['domain']}] {r['similarity_score']:.4f}  {r['source']}")