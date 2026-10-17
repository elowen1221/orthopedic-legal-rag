"""
Demo 健康检查。

用法:
    python src/demo_check.py
"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

COLLECTIONS = ["cfr493", "medical", "legal_ortho", "ortho_single"]

PASSED = []
FAILED = []


def check(name, fn):
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, str(e)
    elapsed = time.time() - t0
    if ok:
        PASSED.append(name)
        print(f"[OK]   {name:35s} {detail}  ({elapsed:.2f}s)")
    else:
        FAILED.append((name, detail))
        print(f"[FAIL] {name:35s} {detail}")


def ck_python():
    return True, f"Python {sys.version.split()[0]}"


def ck_qdrant():
    r = requests.get(f"{QDRANT_URL}/collections", timeout=5)
    r.raise_for_status()
    return True, QDRANT_URL


def ck_ollama():
    r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    return True, f"{OLLAMA_HOST} (models={len(models)})"


def ck_collection(name):
    def _fn():
        r = requests.get(f"{QDRANT_URL}/collections/{name}", timeout=5)
        if r.status_code == 404:
            return False, "not found"
        r.raise_for_status()
        d = r.json()["result"]
        return True, f"points={d['points_count']}, status={d['status']}"
    return _fn


def ck_embedding():
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": "test"},
        timeout=60,
    )
    r.raise_for_status()
    dim = len(r.json().get("embedding", []))
    if dim == 0:
        return False, "empty embedding"
    return True, f"{EMBED_MODEL} dim={dim}"


def ck_llm():
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 5},
        },
        timeout=300,
    )
    r.raise_for_status()
    return True, f"{LLM_MODEL} responded"


def ck_routed():
    from rag import retrieve_with_routing
    q = "What are the AAOS recommendations for hip fracture management in older adults?"
    results, routing = retrieve_with_routing(q)
    if not results:
        return False, "no results returned"
    return True, (
        f"{len(results)} chunks; "
        f"routing=(med={routing['medical_required']}, leg={routing['legal_required']})"
    )


def ck_fastapi():
    import fastapi
    from api import app  # noqa
    return True, f"fastapi {fastapi.__version__}"


def main():
    print("=" * 80)
    print("Demo health check")
    print("=" * 80)

    check("Python environment", ck_python)
    check("Qdrant connectivity", ck_qdrant)
    check("Ollama connectivity", ck_ollama)

    for c in COLLECTIONS:
        check(f"Collection {c}", ck_collection(c))

    check("Embedding model", ck_embedding)
    check("LLM model", ck_llm)
    check("Routed retrieval (minimal)", ck_routed)
    check("FastAPI import", ck_fastapi)

    print("=" * 80)
    if FAILED:
        print(f"DEMO NOT READY — {len(FAILED)} check(s) failed")
        for name, detail in FAILED:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print(f"DEMO READY — all {len(PASSED)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()