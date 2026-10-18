"""
Unit tests for structured retrieval result schema.
不联网。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def test_to_structured_schema():
    from rag import _to_structured

    class FakeHit:
        id = 0
        payload = {
            "chunk_id": "medical:AAOS-0",
            "source": "AAOS_Hip_Fracture_2021",
            "content": "text",
            "domain": "medical",
        }
        score = 0.87

    out = _to_structured([FakeHit()], domain="medical")
    assert len(out) == 1
    item = out[0]
    for key in ["document_id", "source", "domain",
                "similarity_score", "rank", "content"]:
        assert key in item, f"missing key: {key}"
    assert item["domain"] == "medical"
    assert item["rank"] == 1
    assert isinstance(item["similarity_score"], float)


def test_to_structured_uses_payload_domain():
    from rag import _to_structured

    class FakeHit:
        id = 1
        payload = {
            "chunk_id": "legal:CMS-0",
            "source": "CMS",
            "content": "text",
            "domain": "legal",
        }
        score = 0.5

    out = _to_structured([FakeHit()], domain="mixed")
    assert out[0]["domain"] == "legal"


@pytest.mark.integration
def test_live_retrieval():
    pytest.importorskip("qdrant_client")
    from rag import retrieve
    try:
        results = retrieve("hip fracture", mode="dual")
    except Exception as e:
        pytest.skip(f"live retrieval unavailable: {e}")
    assert isinstance(results, list)
    if results:
        for key in ["document_id", "source", "domain",
                    "similarity_score", "rank", "content"]:
            assert key in results[0]