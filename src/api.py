"""
FastAPI 服务层。

只调用 src/rag.py 中的函数，不复制业务逻辑。

运行:
    uvicorn src.api:app --reload --port 8000
"""
import os
import sys
from pathlib import Path
from typing import List, Optional, Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# ---------- 路径设置 ----------
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv()

# ---------- 导入 rag 接口 ----------
try:
    from rag import (
        ask,
        retrieve,
        retrieve_with_routing,
        client as qdrant_client,
        EMBED_MODEL,
        LLM_MODEL,
        QDRANT_URL,
        OLLAMA_HOST,
    )
except Exception as e:
    raise RuntimeError(f"failed to import src/rag.py: {e}")


# ============================================================
#  Pydantic 模型
# ============================================================

class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["baseline", "ortho_single", "dual", "routed"] = "routed"
    k: int = 5
    k_medical: int = 3
    k_legal: int = 3
    k_single: int = 6

    @field_validator("question")
    @classmethod
    def _q_not_blank(cls, v: str):
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["baseline", "ortho_single", "dual", "routed"] = "routed"
    k: int = 5
    k_medical: int = 3
    k_legal: int = 3
    k_single: int = 6

    @field_validator("question")
    @classmethod
    def _q_not_blank(cls, v: str):
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class RetrievalItem(BaseModel):
    document_id: str
    source: str
    domain: str
    similarity_score: float
    rank: int
    content: str


class RoutingInfo(BaseModel):
    medical_required: bool
    legal_required: bool
    reason: str


class RetrieveResponse(BaseModel):
    question: str
    mode: str
    routing: Optional[RoutingInfo] = None
    retrieval: List[RetrievalItem]


class QueryResponse(BaseModel):
    question: str
    mode: str
    routing: Optional[RoutingInfo] = None
    answer: str
    evidence: List[RetrievalItem]


class HealthResponse(BaseModel):
    api: str
    qdrant: str
    ollama: str
    collections: dict
    models: dict


# ============================================================
#  FastAPI app
# ============================================================

app = FastAPI(
    title="Orthopedic Medical-Legal RAG API",
    description="Evidence-type-aware retrieval for orthopedic medical-legal QA",
    version="0.1.0",
)


# ============================================================
#  辅助函数
# ============================================================

def _check_qdrant() -> bool:
    try:
        r = requests.get(f"{QDRANT_URL}/collections", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _collection_status() -> dict:
    out = {}
    for name in ["cfr493", "medical", "legal_ortho", "ortho_single"]:
        try:
            if qdrant_client.collection_exists(name):
                info = qdrant_client.get_collection(name)
                out[name] = {"exists": True, "points_count": info.points_count}
            else:
                out[name] = {"exists": False, "points_count": 0}
        except Exception as e:
            out[name] = {"exists": False, "points_count": 0, "error": str(e)}
    return out


def _do_retrieve(req) -> tuple:
    if req.mode == "routed":
        results, routing = retrieve_with_routing(req.question)
        return results, routing
    else:
        results = retrieve(
            req.question, mode=req.mode,
            k=req.k,
            k_medical=req.k_medical,
            k_legal=req.k_legal,
            k_single=req.k_single,
        )
        return results, None


# ============================================================
#  端点
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        api="ok",
        qdrant="ok" if _check_qdrant() else "unreachable",
        ollama="ok" if _check_ollama() else "unreachable",
        collections=_collection_status(),
        models={
            "embedding": EMBED_MODEL,
            "llm": LLM_MODEL,
        },
    )


@app.get("/collections")
def get_collections():
    return _collection_status()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    try:
        results, routing = _do_retrieve(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")

    return RetrieveResponse(
        question=req.question,
        mode=req.mode,
        routing=RoutingInfo(**routing) if routing else None,
        retrieval=[RetrievalItem(**r) for r in results],
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    try:
        if req.mode == "routed":
            ans, results, routing = ask(
                req.question, mode=req.mode,
                k_medical=req.k_medical,
                k_legal=req.k_legal,
            )
        else:
            ans, results = ask(
                req.question, mode=req.mode,
                k=req.k,
                k_medical=req.k_medical,
                k_legal=req.k_legal,
                k_single=req.k_single,
            )
            routing = None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query failed: {e}")

    return QueryResponse(
        question=req.question,
        mode=req.mode,
        routing=RoutingInfo(**routing) if routing else None,
        answer=ans,
        evidence=[RetrievalItem(**r) for r in results],
    )


# ============================================================
#  全局异常处理
# ============================================================

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"internal error: {type(exc).__name__}"},
    )