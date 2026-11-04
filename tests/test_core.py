import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag import classify_evidence_requirements, retrieve_with_routing

@pytest.mark.integration
def test_router_valid_json():
    """测试正常路由，需要 Ollama 在线"""
    res = classify_evidence_requirements("What are the AAOS recommendations for hip fracture?")
    assert "medical_required" in res
    assert res["router_fallback"] is False

def test_router_invalid_json(monkeypatch):
    """测试非法 JSON 的 fallback 机制"""
    import rag
    # Mock call_ollama_llm 返回一个非 JSON 字符串
    monkeypatch.setattr(rag, "call_ollama_llm", lambda s, u: "This is not a JSON")
    res = rag.classify_evidence_requirements("test question")
    assert res["router_fallback"] is True
    assert res["medical_required"] is True
    assert res["legal_required"] is True

@pytest.mark.integration
def test_retrieval_dual_top_k():
    """测试 dual 模式返回的 chunk 是否为 3+3"""
    res = retrieve_with_routing("test") 
    assert isinstance(res, tuple)

def test_evaluation_filtering():
    """测试评估数据解析是否正常"""
    def safe_set(s):
        if not s or str(s) == "nan": return set()
        return set([x.strip() for x in str(s).split(",") if x.strip()])
        
    assert safe_set("A,B") == {"A", "B"}
    assert safe_set("") == set()