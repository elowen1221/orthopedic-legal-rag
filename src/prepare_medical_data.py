"""
PDF -> data/medical_segments.csv (Paper-ready with Metadata)

用法:
    python src/prepare_medical_data.py

策略:
    - 自动扫描 data/medical_raw/*.pdf
    - PyMuPDF 提取文本层
    - 检测并删除跨页重复的页眉/页脚
    - 每 N 页合并为一个中等粒度 segment（不做细粒度 chunk）
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

RAW_DIR = Path("data/medical_raw")
OUT_CSV = Path("data/medical_segments.csv")
PAGES_PER_SEGMENT = 4

# 文件名 stem -> 标准化 Metadata
FILE_MAP = {
    "hipfxcpg_2021": {
        "source_id": "AAOS_Hip_Fracture_2021",
        "title": "Management of Hip Fractures in Older Adults",
        "organization": "AAOS",
        "document_type": "Clinical Practice Guideline",
        "jurisdiction": "US",
    },
    "BOAST-FTLPD": {
        "source_id": "BOA_BOAST_FTLPD",
        "title": "First-time Lateral Patellar Dislocation",
        "organization": "BOA",
        "document_type": "Professional Guideline",
        "jurisdiction": "UK",
    },
    "BOASTCompartmentSyndromeSeptember-2025": {
        "source_id": "BOA_BOAST_Compartment_Syndrome",
        "title": "Compartment Syndrome",
        "organization": "BOA",
        "document_type": "Professional Guideline",
        "jurisdiction": "UK",
    }
}


def extract_year(stem: str) -> str:
    m = re.search(r"(\d{4})", stem)
    return m.group(1) if m else ""


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

        # 获取 Metadata (如果不在映射里，用默认值兜底)
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

        if total_chars < 100 * n_pages:
            print("  WARNING: very low text density, PDF may be scanned.")

        drop = detect_headers_footers(raw_pages, threshold=0.6)
        cleaned = [clean_page_text(p, drop) for p in raw_pages]

        n_seg = 0
        for start in range(0, n_pages, PAGES_PER_SEGMENT):
            end = min(start + PAGES_PER_SEGMENT, n_pages)
            content = "\n\n".join(cleaned[i] for i in range(start, end) if cleaned[i])
            if not content:
                continue
            
            # 构造统一 schema 的行
            all_rows.append({
                "chunk_source_id": f"{meta['source_id']}-{n_seg}",
                "source_id": meta['source_id'],
                "domain": "medical",
                "title": meta['title'],
                "document_type": meta['document_type'],
                "organization_or_court": meta['organization'],
                "jurisdiction": meta['jurisdiction'],
                "year": year,
                "page_start": start + 1,
                "page_end": end,
                "content": content,
                # 兼容旧版建库脚本所需字段
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