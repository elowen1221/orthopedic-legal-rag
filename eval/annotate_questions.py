import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Paper Evaluation Ground Truth Annotator")

CSV_PATH = Path("eval/questions_paper.csv")
if not CSV_PATH.exists():
    st.error("Please generate questions_paper.csv first.")
    st.stop()

df = pd.read_csv(CSV_PATH).fillna("")

# 简单用 session_state 控制当前题号
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0

idx = st.session_state.q_idx
if idx >= len(df):
    st.success("All questions annotated!")
    if st.button("Save CSV"):
        df.to_csv(CSV_PATH, index=False)
    st.stop()

row = df.iloc[idx]
st.header(f"Question {idx+1}/{len(df)}: {row['question_id']}")
st.info(f"**Type:** {row['question_type']} | **Medical Req:** {row['medical_required']} | **Legal Req:** {row['legal_required']}")
st.markdown(f"### {row['question']}")

st.write("---")
col1, col2 = st.columns(2)

# 快速查看 manifest 中的 source
manifest = pd.read_csv("data/corpus_manifest.csv")
med_sources = manifest[manifest.domain=="medical"]["source_id"].tolist()
leg_sources = manifest[manifest.domain=="legal"]["source_id"].tolist()

with col1:
    med_gold = st.multiselect("Gold Medical Sources", options=med_sources, default=str(row["gold_medical_source_ids"]).split(","))
with col2:
    leg_gold = st.multiselect("Gold Legal Sources", options=leg_sources, default=str(row["gold_legal_source_ids"]).split(","))

answerable = st.checkbox("Is this question answerable by the corpus?", value=bool(row["answerable"]))
notes = st.text_input("Annotation Notes", value=str(row["annotation_notes"]))

if st.button("Save & Next"):
    df.at[idx, "gold_medical_source_ids"] = ",".join(med_gold) if med_gold else ""
    df.at[idx, "gold_legal_source_ids"] = ",".join(leg_gold) if leg_gold else ""
    df.at[idx, "answerable"] = int(answerable)
    df.at[idx, "annotation_notes"] = notes
    df.at[idx, "annotation_status"] = "REVIEWED"
    
    df.to_csv(CSV_PATH, index=False)
    st.session_state.q_idx += 1
    st.rerun()