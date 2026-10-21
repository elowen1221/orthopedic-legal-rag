"""
PDF -> data/legal_segments.csv (Paper-ready with Metadata)

用法:
    python src/prepare_legal_data.py

策略:
    - 自动扫描 data/legal_raw/*.pdf
    - 每 N 页合并为一个中等粒度 segment
    - 输出标准化 metadata 字段
"""
import re
import sys
from pathlib import Path
from collections import Counter

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:
    import fitz  # 旧版本兼容

import pandas as pd

RAW_DIR = Path("data/legal_raw")
OUT_CSV = Path("data/legal_segments.csv")
PAGES_PER_SEGMENT = 3

# 文件名 stem -> 标准化 Metadata
FILE_MAP = {
    "cms_surgical_informed_consent": {
        "source_id": "CMS_Informed_Consent",
        "title": "CMS Surgical Informed Consent Guidance",
        "organization": "CMS",
        "document_type": "Regulatory Guidance",
        "jurisdiction": "US-Federal",
    },
    "bender_v._durrani": {
        "source_id": "Case_Bender_v_Durrani",
        "title": "Bender v. Durrani",
        "organization": "US Court",
        "document_type": "Case Law",
        "jurisdiction": "US",
    },
    "sobin_v._orthopaedic_sports_specialists_p.c": {
        "source_id": "Case_Sobin_v_Orthopaedic",
        "title": "Sobin v. Orthopaedic Sports Specialists",
        "organization": "US Court",
        "document_type": "Case Law",
        "jurisdiction": "US",
    },
    # 注意：如果发现 jto v02... 是扫描版没字，代码会自动跳过
}


def extract_year(stem: str) -> str:
    m = re.search(r"(\d{4})", stem)
    return m.group(1) if m else "UNKNOWN"


def detect_headers_footers(pages, threshold=0.6):
    counter = Counter()
    n = len(pages)
    for page in pages:
        seen = set()
        for line in page.split("\n"):
            ln = line.strip()
            if 2 < len(ln) < 120:
                seen.add(ln)
        for ln in seen:
            counter[ln] += 1
    cutoff = max(2, int(n * threshold))
    return {ln for ln, c in counter.items() if c >= cutoff}


def clean_page_text(text: str, drop_lines: set) -> str:
    lines = []
    for line in text.split("\n"):
        ln = line.strip()
        if ln in drop_lines:
            continue
        lines.append(ln)
    joined = "\n".join(lines)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found", file=sys.stderr)
        sys.exit(1)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: no PDF found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"found {len(pdf_files)} PDF(s)")
    all_rows = []

    for pdf_path in pdf_files:
        stem = pdf_path.stem

        meta = FILE_MAP.get(stem, {
            "source_id": f"UNKNOWN_{stem}",
            "title": stem.replace("_", " ").title(),
            "organization": "UNKNOWN",
            "document_type": "UNKNOWN",
            "jurisdiction": "UNKNOWN"
        })

        year = extract_year(stem)

        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)
        raw_pages = [doc[i].get_text("text") for i in range(n_pages)]
        doc.close()

        total_chars = sum(len(p) for p in raw_pages)
        print(f"\n[{stem}] -> {meta['source_id']}")
        print(f"  pages: {n_pages} | chars: {total_chars}")

        # 如果是扫描版（比如之前的 jto 文件），直接跳过
        if total_chars < 100 * n_pages:
            print("  WARNING: very low text density. Skipping (might be scanned).")
            continue

        drop = detect_headers_footers(raw_pages, threshold=0.6)
        cleaned = [clean_page_text(p, drop) for p in raw_pages]

        n_seg = 0
        for start in range(0, n_pages, PAGES_PER_SEGMENT):
            end = min(start + PAGES_PER_SEGMENT, n_pages)
            content = "\n\n".join(cleaned[i] for i in range(start, end) if cleaned[i])
            if not content:
                continue

            all_rows.append({
                "chunk_source_id": f"{meta['source_id']}-{n_seg}",
                "source_id": meta['source_id'],
                "domain": "legal",
                "title": meta['title'],
                "document_type": meta['document_type'],
                "organization_or_court": meta['organization'],
                "jurisdiction": meta['jurisdiction'],
                "year": year,
                "page_start": start + 1,
                "page_end": end,
                "content": content,
                # 兼容旧版建库脚本
                "Source": meta['source_id'],
                "Title": meta['title'],
                "Year": year,
                "Document_Type": meta['document_type'],
                "Page_Start": start + 1,
                "Page_End": end,
                "Content": content
            })
            n_seg += 1

        print(f"  segments: {n_seg}")

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nwrote: {OUT_CSV} (total rows: {len(df)})")


if __name__ == "__main__":
    main()