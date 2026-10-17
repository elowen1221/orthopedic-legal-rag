"""
CSV -> Qdrant collection "legal_ortho"

用法:
    python src/build_legal_index.py

只重建 legal_ortho，不碰 cfr493 / medical。
"""
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

COLLECTION = "legal_ortho"
INPUT_CSV = "data/legal_segments.csv"

# 保护名单：绝不允许删除
PROTECTED_COLLECTIONS = {"cfr493", "medical"}
assert COLLECTION not in PROTECTED_COLLECTIONS, \
    f"refuse to operate on protected collection: {COLLECTION}"


def _to_int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


df = pd.read_csv(INPUT_CSV).fillna("")
print("legal segments:", len(df))

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=2000,
    chunk_overlap=200,
)

chunks = []
for _, row in df.iterrows():
    source = row["Source"]
    for j, chunk in enumerate(splitter.split_text(row["Content"])):
        if chunk.strip():
            chunks.append({
                "source": source,
                "content": chunk,
                "chunk_id": f"legal:{source}-{j}",
                "title": row.get("Title", ""),
                "year": str(row.get("Year", "")),
                "document_type": row.get("Document_Type", ""),
                "page_start": _to_int(row.get("Page_Start", 0)),
                "page_end": _to_int(row.get("Page_End", 0)),
            })

print("legal chunks:", len(chunks))

emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
test_vec = emb.embed_query("test")
dim = len(test_vec)
print("embedding dim:", dim)

client = QdrantClient(url=QDRANT_URL)

# 安全检查：确认不误删别的 collection
print("\nexisting collections before build:")
for c in client.get_collections().collections:
    print(f"  - {c.name}")

if client.collection_exists(COLLECTION):
    print(f"\ndeleting existing collection: {COLLECTION}")
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
)

BATCH = 32
points = []
for i, c in enumerate(chunks):
    vec = emb.embed_query(c["content"])
    points.append(PointStruct(id=i, vector=vec, payload=c))
    if len(points) >= BATCH:
        client.upsert(collection_name=COLLECTION, points=points)
        points = []
        print(f"upserted {i+1}/{len(chunks)}")

if points:
    client.upsert(collection_name=COLLECTION, points=points)

info = client.get_collection(COLLECTION)
print("legal_ortho collection points:", info.points_count)

print("\nexisting collections after build:")
for c in client.get_collections().collections:
    ci = client.get_collection(c.name)
    print(f"  - {c.name:20s} points={ci.points_count}")

print("done")