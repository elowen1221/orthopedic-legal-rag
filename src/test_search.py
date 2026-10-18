import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
client = QdrantClient(url=QDRANT_URL)

# ---------- 1. Collection 健康检查 ----------
print("=" * 80)
print("Collection check")
print("=" * 80)

COLLECTIONS = ["cfr493", "medical", "legal_ortho", "ortho_single"]
for c in COLLECTIONS:
    if client.collection_exists(c):
        info = client.get_collection(c)
        print(f"  [OK]      {c:15s} points={info.points_count}")
    else:
        print(f"  [MISSING] {c:15s} (not found)")

# ---------- 2. 三类问题 x 三种 mode ----------
questions = [
    {
        "q": "What are the AAOS recommendations for hip fracture management in older adults?",
        "type": "A. medical-only",
    },
    {
        "q": "What information should generally be included in surgical informed consent?",
        "type": "B. legal-focused",
    },
    {
        "q": "What clinical considerations and informed consent information should be addressed before surgical management of a hip fracture in an older adult?",
        "type": "C. cross-domain",
    },
]

MODES = ["ortho_single", "dual"]
K_SINGLE = 6
K_MED = 3
K_LEG = 3


def search(collection, vec, limit):
    if not client.collection_exists(collection):
        return []
    result = client.query_points(collection_name=collection, query=vec, limit=limit)
    return result.points


for item in questions:
    q = item["q"]
    print("=" * 80)
    print(f"Q ({item['type']}): {q}")
    vec = emb.embed_query(q)

    # --- ortho_single ---
    if "ortho_single" in MODES:
        print(f"\n  --- [ortho_single] Top-{K_SINGLE} ---")
        hits = search("ortho_single", vec, K_SINGLE)
        for rank, h in enumerate(hits, 1):
            dom = h.payload.get("domain", "?")
            print(f"    rank={rank}  domain={dom:8s}  score={h.score:.4f}  source={h.payload['source']}")
            print(f"      {h.payload['content'][:150].replace(chr(10), ' ')}")

    # --- dual ---
    if "dual" in MODES:
        print(f"\n  --- [dual] medical Top-{K_MED} ---")
        for rank, h in enumerate(search("medical", vec, K_MED), 1):
            print(f"    rank={rank}  domain=medical  score={h.score:.4f}  source={h.payload['source']}")
            print(f"      {h.payload['content'][:150].replace(chr(10), ' ')}")
        print(f"\n  --- [dual] legal_ortho Top-{K_LEG} ---")
        for rank, h in enumerate(search("legal_ortho", vec, K_LEG), 1):
            print(f"    rank={rank}  domain=legal    score={h.score:.4f}  source={h.payload['source']}")
            print(f"      {h.payload['content'][:150].replace(chr(10), ' ')}")

    print()