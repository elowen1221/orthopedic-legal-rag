"""
从 medical + legal_ortho 复制已有 vectors/payloads 到 ortho_single。

特点:
    - 不重新 embedding，直接复制已有 vector
    - 不重新切 chunk，chunk 与 dual 完全一致
    - payload 保留原字段，并额外写入 domain="medical"/"legal"
    - 绝不删除 cfr493 / medical / legal_ortho

用法:
    python src/build_ortho_single_index.py
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

TARGET = "ortho_single"
SOURCE_COLLECTIONS = [
    ("medical", "medical"),        # (collection_name, domain)
    ("legal_ortho", "legal"),
]
PROTECTED = {"cfr493", "medical", "legal_ortho"}

assert TARGET not in PROTECTED, f"refuse to operate on protected collection: {TARGET}"

client = QdrantClient(url=QDRANT_URL)

print("=" * 80)
print("Collections before build")
print("=" * 80)
for c in client.get_collections().collections:
    info = client.get_collection(c.name)
    print(f"  {c.name:20s} points={info.points_count}")

# ---------- 1. 从源 collection 拉取所有 points（含 vector） ----------
all_points = []
next_id = 0

for coll_name, domain in SOURCE_COLLECTIONS:
    if not client.collection_exists(coll_name):
        print(f"\nWARN: {coll_name} not found, skip")
        continue

    print(f"\nreading from {coll_name} (domain={domain})")
    offset = None
    n_read = 0
    while True:
        records, offset = client.scroll(
            collection_name=coll_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for r in records:
            payload = dict(r.payload or {})
            payload["domain"] = domain
            # 保留原始 chunk_id，domain 前缀在 _to_structured 里再加
            all_points.append(PointStruct(
                id=next_id,
                vector=r.vector,
                payload=payload,
            ))
            next_id += 1
            n_read += 1
        if offset is None:
            break
    print(f"  read {n_read} points")

print(f"\ntotal points to copy: {len(all_points)}")

if not all_points:
    print("ERROR: no points to copy", flush=True)
    raise SystemExit(1)

# ---------- 2. 重建 ortho_single ----------
if client.collection_exists(TARGET):
    print(f"\ndeleting existing {TARGET}")
    client.delete_collection(TARGET)

# 检查 vector 维度
sample_vec = all_points[0].vector
if isinstance(sample_vec, dict):
    # 命名向量场景，取第一个
    dim = len(next(iter(sample_vec.values())))
else:
    dim = len(sample_vec)
print(f"vector dim: {dim}")

client.create_collection(
    collection_name=TARGET,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
)

# ---------- 3. 批量 upsert ----------
BATCH = 32
for i in range(0, len(all_points), BATCH):
    batch = all_points[i:i + BATCH]
    client.upsert(collection_name=TARGET, points=batch)
    print(f"upserted {min(i + BATCH, len(all_points))}/{len(all_points)}")

info = client.get_collection(TARGET)
print(f"\n{TARGET} points: {info.points_count}")

# ---------- 4. 确认所有 collection 状态 ----------
print("\n" + "=" * 80)
print("Collections after build")
print("=" * 80)
for c in client.get_collections().collections:
    info = client.get_collection(c.name)
    print(f"  {c.name:20s} points={info.points_count}")

expected = sum(client.get_collection(n).points_count
               for n, _ in SOURCE_COLLECTIONS if client.collection_exists(n))
actual = client.get_collection(TARGET).points_count
print(f"\nexpected ortho_single points: {expected}")
print(f"actual   ortho_single points: {actual}")
assert actual == expected, "MISMATCH: ortho_single points != medical + legal_ortho"
print("done")