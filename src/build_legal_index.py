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
PROTECTED_COLLECTIONS = {"cfr493", "medical"}

assert COLLECTION not in PROTECTED_COLLECTIONS

df = pd.read_csv(INPUT_CSV).fillna("")
print("legal segments:", len(df))

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=2000,
    chunk_overlap=200,
)

chunks = []
for _, row in df.iterrows():
    src_id = row.get("source_id", row.get("Source", ""))
    content = row.get("content", row.get("Content", ""))
    
    for j, chunk in enumerate(splitter.split_text(content)):
        if chunk.strip():
            chunks.append({
                "source": src_id,
                "content": chunk,
                "chunk_id": row.get("chunk_source_id", f"{src_id}-{j}"),
                "domain": "legal",  # 🌟 核心：强制写入
                "title": row.get("title", row.get("Title", "")),
                "organization_or_court": row.get("organization_or_court", row.get("Organization", "")),
                "year": str(row.get("year", row.get("Year", ""))),
                "document_type": row.get("document_type", row.get("Document_Type", "")),
                "page_start": int(row.get("page_start", row.get("Page_Start", 1))),
                "page_end": int(row.get("page_end", row.get("Page_End", 1)))
            })

print("legal chunks:", len(chunks))
if chunks:
    print("👀 检查第一条数据的 domain 字段:", chunks[0].get("domain"))

emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
dim = len(emb.embed_query("test"))

client = QdrantClient(url=QDRANT_URL)

if client.collection_exists(COLLECTION):
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

if points:
    client.upsert(collection_name=COLLECTION, points=points)

print("legal_ortho collection points:", client.get_collection(COLLECTION).points_count)
print("done")