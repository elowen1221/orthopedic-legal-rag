import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag import client, MEDICAL_COLLECTION, ORTHO_LEGAL_COLLECTION, ORTHO_SINGLE_COLLECTION
import hashlib

def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def scroll_all(coll_name):
    records, _ = client.scroll(collection_name=coll_name, limit=10000, with_vectors=True, with_payload=True)
    return records

def verify():
    print("Starting Index Parity Verification...")
    med = scroll_all(MEDICAL_COLLECTION)
    leg = scroll_all(ORTHO_LEGAL_COLLECTION)
    single = scroll_all(ORTHO_SINGLE_COLLECTION)

    print(f"Point Counts: Medical={len(med)}, Legal={len(leg)}, Single={len(single)}")
    if len(med) + len(leg) != len(single):
        print("FAIL: Point count mismatch!")
        return

    # 用 content 的 hash 映射
    source_map = {}
    for r in med:
        source_map[get_hash(r.payload['content'])] = r
    for r in leg:
        source_map[get_hash(r.payload['content'])] = r

    errors = 0
    for r in single:
        h = get_hash(r.payload['content'])
        if h not in source_map:
            print(f"FAIL: Content not found in source collections.")
            errors += 1
            continue
        
        src_r = source_map[h]
        
        # 校验 vector
        if r.vector != src_r.vector:
            print(f"FAIL: Vector mismatch for chunk {r.payload.get('chunk_id')}")
            errors += 1
            continue
            
        # 校验 domain
        dom_single = r.payload.get("domain")
        dom_src = src_r.payload.get("domain")
        if dom_single != dom_src:
            print(f"FAIL: Domain mismatch! Single库={dom_single}, 源库={dom_src} (chunk: {r.payload.get('chunk_id')})")
            errors += 1
            if errors > 5: # 防止刷屏
                print("... (too many errors, stopping output)")
                break

    if errors == 0:
        print("\n>>> RESULT: PASS <<<")
    else:
        print(f"\n>>> RESULT: FAIL ({errors} errors) <<<")

if __name__ == "__main__":
    verify()