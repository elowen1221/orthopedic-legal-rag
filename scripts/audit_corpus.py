import pandas as pd
from pathlib import Path
import hashlib

def audit():
    Path("reports").mkdir(exist_ok=True)
    med_df = pd.read_csv("data/medical_segments.csv").fillna("")
    leg_df = pd.read_csv("data/legal_segments.csv").fillna("")
    
    report = ["# Corpus Quality Report\n"]
    for name, df in [("Medical", med_df), ("Legal", leg_df)]:
        chars = df["content"].str.len()
        hashes = df["content"].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        dupes = hashes.duplicated().sum()
        
        report.append(f"## {name} Corpus")
        report.append(f"- Document count: {df['source_id'].nunique()}")
        report.append(f"- Segment count: {len(df)}")
        report.append(f"- Character distribution: Mean {chars.mean():.1f}, Min {chars.min()}, Max {chars.max()}")
        report.append(f"- Duplicate chunks: {dupes}\n")

    with open("reports/corpus_report.md", "w") as f:
        f.write("\n".join(report))
    print("Audit complete. Check reports/corpus_report.md")

if __name__ == "__main__":
    audit()