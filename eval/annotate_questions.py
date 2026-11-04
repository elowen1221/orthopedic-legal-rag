import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from rag import retrieve_medical, retrieve_legal
except ImportError as e:
    st.error(f"Failed to import rag: {e}")
    st.stop()

st.set_page_config(page_title="Paper Evaluation Annotator", layout="wide")

CSV_PATH = Path("eval/questions_paper.csv")

if "df" not in st.session_state:
    if not CSV_PATH.exists():
        st.error("questions_paper.csv not found.")
        st.stop()
    st.session_state.df = pd.read_csv(CSV_PATH).fillna("")

if "question_index" not in st.session_state:
    unrev = st.session_state.df[st.session_state.df["annotation_status"] != "REVIEWED"].index.tolist()
    st.session_state.question_index = unrev[0] if unrev else 0

idx = max(0, min(st.session_state.question_index, len(st.session_state.df) - 1))
row = st.session_state.df.iloc[idx]
qid = row["question_id"]

@st.cache_data(show_spinner=False)
def get_evidence(q_text, q_id):
    med_hits = retrieve_medical(q_text, k=5)
    leg_hits = retrieve_legal(q_text, k=5)
    med_hits = sorted(med_hits, key=lambda x: x.get("similarity_score", 0), reverse=True)
    leg_hits = sorted(leg_hits, key=lambda x: x.get("similarity_score", 0), reverse=True)
    return med_hits, leg_hits

st.sidebar.title("📖 Annotation Rules")
with st.sidebar.expander("How to judge Answerable", expanded=True):
    st.markdown("""
    **Select Yes ONLY IF:**
    - At least one source in the current corpus contains the **actual necessary evidence**.
    - Just being "topically related" is **NOT** enough.
    """)

with st.sidebar.expander("How to judge Gold Sources", expanded=True):
    st.markdown("""
    **Gold source** = The exact `source_id` that truly contains the necessary evidence.
    - Read the Evidence Preview on the right.
    - Click **"Use as gold..."** to automatically add it.
    """)

st.sidebar.divider()

reviewed_count = (st.session_state.df["annotation_status"] == "REVIEWED").sum()
st.sidebar.metric("Progress", f"{reviewed_count} / {len(st.session_state.df)} REVIEWED")

def go_prev():
    if st.session_state.question_index > 0:
        st.session_state.question_index -= 1

def go_next():
    if st.session_state.question_index < len(st.session_state.df) - 1:
        st.session_state.question_index += 1

st.sidebar.button("⬅️ Previous Question", on_click=go_prev, use_container_width=True)
st.sidebar.button("Next Question ➡️ (Skip)", on_click=go_next, use_container_width=True)

def update_gold_source(target_key, source_id):
    existing = [x.strip() for x in st.session_state[target_key].split(",") if x.strip()]
    if source_id not in existing:
        existing.append(source_id)
        st.session_state[target_key] = ",".join(existing)

def save_and_next():
    st.session_state.df.at[idx, "question_type"] = st.session_state[f"qtype_{qid}"]
    st.session_state.df.at[idx, "medical_required"] = st.session_state[f"med_req_{qid}"]
    st.session_state.df.at[idx, "legal_required"] = st.session_state[f"leg_req_{qid}"]
    st.session_state.df.at[idx, "answerable"] = st.session_state[f"ans_{qid}"]
    st.session_state.df.at[idx, "gold_medical_source_ids"] = st.session_state[f"med_{qid}"]
    st.session_state.df.at[idx, "gold_legal_source_ids"] = st.session_state[f"leg_{qid}"]
    st.session_state.df.at[idx, "annotation_notes"] = st.session_state[f"notes_{qid}"]
    st.session_state.df.at[idx, "annotation_status"] = "REVIEWED"
    
    st.session_state.df.to_csv(CSV_PATH, index=False)
    go_next()

if f"med_{qid}" not in st.session_state:
    st.session_state[f"qtype_{qid}"] = row["question_type"] if row["question_type"] in ["medical", "legal", "cross"] else "medical"
    st.session_state[f"med_req_{qid}"] = int(row["medical_required"])
    st.session_state[f"leg_req_{qid}"] = int(row["legal_required"])
    st.session_state[f"ans_{qid}"] = int(row["answerable"])
    st.session_state[f"med_{qid}"] = str(row["gold_medical_source_ids"])
    st.session_state[f"leg_{qid}"] = str(row["gold_legal_source_ids"])
    st.session_state[f"notes_{qid}"] = str(row["annotation_notes"])

st.title(f"Question {idx+1}/{len(st.session_state.df)}: {qid}")
st.info(f"**Question:** {row['question']}")

with st.spinner("Fetching retrieval evidence..."):
    med_hits, leg_hits = get_evidence(row["question"], qid)

col_form, col_evidence = st.columns([1, 1.5])

with col_form:
    st.subheader("📝 Annotation Form")
    
    st.selectbox("question_type", ["medical", "legal", "cross", "invalid"], key=f"qtype_{qid}")
    st.radio("Medical Required", [1, 0], key=f"med_req_{qid}", horizontal=True)
    st.radio("Legal Required", [1, 0], key=f"leg_req_{qid}", horizontal=True)
    st.radio("Answerable by Corpus?", [1, 0], key=f"ans_{qid}", horizontal=True)
    
    st.text_input("gold_medical_source_ids", key=f"med_{qid}")
    st.text_input("gold_legal_source_ids", key=f"leg_{qid}")
    st.text_input("annotation_notes", key=f"notes_{qid}")

    st.button("✅ Save & Next", on_click=save_and_next, type="primary", use_container_width=True)

with col_evidence:
    st.subheader("🔍 Evidence Preview (Top-5)")
    
    def render_evidence(hits, domain_label, target_key):
        if not hits:
            st.info(f"No {domain_label} evidence found.")
            return
            
        current_golds = [x.strip() for x in st.session_state[target_key].split(",") if x.strip()]
        
        for rank, h in enumerate(hits, 1):
            src = h.get("source", "UNKNOWN")
            score = h.get("similarity_score", 0.0)
            content = h.get("content", "")
            
            is_ai_suggested = src in current_golds
            badge = " 🏷️ *(AI suggested)*" if is_ai_suggested else ""
            
            with st.container(border=True):
                st.markdown(f"**[{domain_label}] {src}** | Score: `{score:.3f}` {badge}")
                st.caption(f"Preview: {content[:300]}...")
                
                col_btn, col_pop = st.columns([1, 1])
                with col_btn:
                    st.button(
                        f"➕ Use as gold", 
                        key=f"btn_{qid}_{domain_label}_{rank}_{src}",
                        on_click=update_gold_source,
                        args=(target_key, src)
                    )
                with col_pop:
                    with st.popover("📖 Read full chunk"):
                        st.write(content)

    st.markdown("### Medical Evidence")
    render_evidence(med_hits, "Medical", f"med_{qid}")
    
    st.divider()
    
    st.markdown("### Legal Evidence")
    render_evidence(leg_hits, "Legal", f"leg_{qid}")