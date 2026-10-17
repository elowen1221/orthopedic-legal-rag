"""
PDF -> data/medical_segments.csv

用法:
    python src/prepare_medical_data.py

策略:
    - 自动扫描 data/medical_raw/*.pdf
    - PyMuPDF 提取文本层
    - 检测并删除跨页重复的页眉/页脚
    - 每 N 页合并为一个中等粒度 segment（不做细粒度 chunk）
    - 输出 data/medical_segments.csv (UTF-8-SIG)
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

# 文件名 stem -> (Source, Title)
# 命中则用映射，未命中则用文件名变形
FILE_MAP = {
    "hipfxcpg_2021": ("AAOS_Hip_Fracture_2021", "Management of Hip Fractures in Older Adults"),
    "hip_fracture_2021": ("AAOS_Hip_Fracture_2021", "Management of Hip Fractures in Older Adults"),
    "hip_oa_2023": ("AAOS_Hip_OA_2023", "Management of Osteoarthritis of the Hip"),
    "knee_oa_2021": ("AAOS_Knee_OA_2021", "Management of Osteoarthritis of the Knee"),
}


def filename_to_source(stem: str) -> str:
    """hip_fracture_2021 -> AAOS_Hip_Fracture_2021"""
    return "AAOS_" + "_".join(w.capitalize() for w in stem.split("_"))


def extract_year(stem: str) -> str:
    m = re.search(r"(\d{4})", stem)
    return m.group(1) if m else ""


def detect_headers_footers(pages, threshold=0.6):
    """检测在多页重复出现的短行，视为页眉/页脚"""
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

        if stem in FILE_MAP:
            source, title = FILE_MAP[stem]
        else:
            source = filename_to_source(stem)
            title = stem.replace("_", " ").title()

        year = extract_year(stem)

        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)
        raw_pages = [doc[i].get_text("text") for i in range(n_pages)]
        doc.close()

        total_chars = sum(len(p) for p in raw_pages)
        print(f"\n[{stem}]")
        print(f"  source: {source}")
        print(f"  title:  {title}")
        print(f"  year:   {year}")
        print(f"  pages:  {n_pages}")
        print(f"  extracted chars: {total_chars}")

        if total_chars < 100 * n_pages:
            print("  WARNING: very low text density, PDF may be scanned (no text layer)")

        drop = detect_headers_footers(raw_pages, threshold=0.6)
        if drop:
            print(f"  detected repeated header/footer lines: {len(drop)}")

        cleaned = [clean_page_text(p, drop) for p in raw_pages]

        n_seg = 0
        seg_chars = 0
        for start in range(0, n_pages, PAGES_PER_SEGMENT):
            end = min(start + PAGES_PER_SEGMENT, n_pages)
            content = "\n\n".join(cleaned[i] for i in range(start, end) if cleaned[i])
            if not content:
                continue
            all_rows.append({
                "Source": source,
                "Title": title,
                "Year": year,
                "Document_Type": "Clinical Practice Guideline",
                "Page_Start": start + 1,
                "Page_End": end,
                "Content": content,
            })
            n_seg += 1
            seg_chars += len(content)

        avg = seg_chars / n_seg if n_seg else 0
        print(f"  segments: {n_seg}")
        print(f"  avg segment chars: {avg:.0f}")

    df = pd.DataFrame(all_rows, columns=[
        "Source", "Title", "Year", "Document_Type",
        "Page_Start", "Page_End", "Content",
    ])

    empty = df["Content"].str.strip().eq("").sum()
    print(f"\nempty Content rows: {empty}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"wrote: {OUT_CSV}")
    print(f"total rows: {len(df)}")
    if len(df) > 0:
        print(f"avg row chars: {df['Content'].str.len().mean():.0f}")
        print(f"min row chars: {df['Content'].str.len().min()}")
        print(f"max row chars: {df['Content'].str.len().max()}")


if __name__ == "__main__":
    main()