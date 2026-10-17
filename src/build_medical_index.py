"""
建医学知识库 collection。
用法: python src/build_medical_index.py
输入: data/medical_segments.csv (列: Source, Content)
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

COLLECTION = "medical"
INPUT_CSV = "data/medical_segments.csv"

df = pd.read_csv(INPUT_CSV).fillna("")
print("medical segments:", len(df))

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=2000,
    chunk_overlap=200,
)

chunks = []
for _, row in df.iterrows():
    for j, chunk in enumerate(splitter.split_text(row["Content"])):
        if chunk.strip():
            chunks.append({
                "source": row["Source"],
                "content": chunk,
                "chunk_id": f"medical:{row['Source']}-{j}",
            })

print("medical chunks:", len(chunks))

emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
test_vec = emb.embed_query("test")
dim = len(test_vec)
print("embedding dim:", dim)

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
        print(f"upserted {i+1}/{len(chunks)}")

if points:
    client.upsert(collection_name=COLLECTION, points=points)

info = client.get_collection(COLLECTION)
print("medical collection points:", info.points_count)
print("done")