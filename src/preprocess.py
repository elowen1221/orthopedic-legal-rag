import html2text
import re
import pandas as pd
import sqlite3
from pathlib import Path

Path("data").mkdir(exist_ok=True)

raw = Path("data/part493.html").read_text(encoding="utf-8", errors="ignore")
print("html length:", len(raw))

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True
h.body_width = 0
md = h.handle(raw)

Path("data/part493.md").write_text(md, encoding="utf-8")
print("markdown length:", len(md))

# 只匹配带 # 前缀的正文标题，排除目录表格
pattern = re.compile(r"(?m)^#{1,6}\s+§\s*493\.\d+[^\n]*")
matches = list(pattern.finditer(md))
print("section matches:", len(matches))

segments = []
for i, m in enumerate(matches):
    source = m.group(0).lstrip("#").strip()
    start = m.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
    content = md[start:end].strip()
    if content:
        segments.append({"Source": source, "Content": content})

df = pd.DataFrame(segments).drop_duplicates("Source")
df.to_csv("data/segments.csv", index=False)

conn = sqlite3.connect("data/raven.db")
df.to_sql("source_documents", conn, if_exists="replace", index=False)
conn.close()

print("segments saved:", len(df))
print(df.head())