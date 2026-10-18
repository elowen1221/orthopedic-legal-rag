"""
Unit tests for evidence requirement routing.
不联网，不调用模型。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def test_invalid_mode_raises():
    from rag import retrieve
    with pytest.raises(ValueError):
        retrieve("test", mode="not_a_mode")


def test_routing_output_schema(monkeypatch):
    import rag

    fake_routing = {
        "medical_required": True,
        "legal_required": False,
        "reason": "unit test",
    }
    monkeypatch.setattr(rag, "classify_evidence_requirements",
                        lambda q: fake_routing)

    monkeypatch.setattr(rag, "retrieve_medical",
                        lambda q, k=3: [{"document_id": "medical:x",
                                          "source": "AAOS",
                                          "domain": "medical",
                                          "similarity_score": 0.9,
                                          "rank": 1,
                                          "content": "..."}])
    monkeypatch.setattr(rag, "retrieve_legal", lambda q, k=3: [])

    results, routing = rag.retrieve_with_routing("test question")
    assert set(routing.keys()) >= {"medical_required", "legal_required", "reason"}
    assert routing["medical_required"] is True
    assert routing["legal_required"] is False
    assert len(results) == 1


def test_routing_fallback_both_false(monkeypatch):
    import rag

    def fake_classify(q):
        return {
            "medical_required": True,
            "legal_required": True,
            "reason": "fallback: both false from LLM, default to dual",
        }
    monkeypatch.setattr(rag, "classify_evidence_requirements", fake_classify)
    monkeypatch.setattr(rag, "retrieve_medical",
                        lambda q, k=3: [{"document_id": "m:1", "source": "s",
                                          "domain": "medical",
                                          "similarity_score": 0.5,
                                          "rank": 1, "content": "x"}])
    monkeypatch.setattr(rag, "retrieve_legal",
                        lambda q, k=3: [{"document_id": "l:1", "source": "s",
                                          "domain": "legal",
                                          "similarity_score": 0.5,
                                          "rank": 1, "content": "y"}])

    results, routing = rag.retrieve_with_routing("test")
    assert routing["medical_required"] is True
    assert routing["legal_required"] is True
    assert "fallback" in routing["reason"].lower()
    assert len(results) == 2