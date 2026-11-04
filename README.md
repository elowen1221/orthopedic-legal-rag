Evidence-Aware Orthopedic Medical-Legal RAG

A domain-aware RAG system that dynamically retrieves orthopedic clinical and legal evidence according to question requirements.

---

Problem

Traditional RAG retrieves semantically similar chunks from a single vector pool.

For medical-legal QA, the relevant evidence may come from heterogeneous domains:

- orthopedic clinical guidelines and evidence
- regulatory and informed-consent guidance

When those domains are mixed into a single pool, one domain can dominate Top-K due to corpus size imbalance and semantic competition, leaving the other domain unrepresented.

This project studies that failure mode and evaluates three retrieval organizations on a small orthopedic medical-legal question set.

---

Method

Single Pool  →  Fixed Dual  →  Routed Dual

Mode| Retrieval organization
"ortho_single"| medical + legal in one pool, Top-6
"dual"| medical Top-3 + legal Top-3, always both
"routed"| LLM classifies required evidence types, then selects a retrieval path

Three retrieval paths are supported by the router:

- medical-only → medical Top-6
- legal-only → legal Top-6
- cross-domain → medical Top-3 + legal Top-3

---

Architecture

flowchart LR
    Q[Question] --> R[Evidence Requirement Router]
    R -->|medical-only| M[Medical Retrieval<br/>AAOS]
    R -->|legal-only| L[Legal Retrieval<br/>CMS]
    R -->|cross-domain| M
    R -->|cross-domain| L
    M --> F[Evidence Fusion]
    L --> F
    F --> G[LLM Answer]

---

Experimental Observation

Preliminary results on a 12-question sanity test
(4 medical-only, 4 legal-only, 4 cross-domain):

| Metric | ortho_single | dual | routed |
|---|---|---|---|
| Required Medical Coverage | 8/8 | 8/8 | 8/8 |
| Required Legal Coverage | 4/8 | 8/8 | 8/8 |
| Unnecessary Medical Retrieval | 4 | 12 | 3 |
| Unnecessary Legal Retrieval | 0 | 12 | 0 |
| Cross-domain Medical Avg | 6.0 | 3.0 | 3.0 |
| Cross-domain Legal Avg | 0.0 | 3.0 | 3.0 |
| Routing Exact Match | n/a | n/a | 11/12 (91.7%) |

Key observations

- Single-pool retrieval allows the medical domain to fully dominate cross-domain Top-6 ("6 medical / 0 legal").
- Fixed dual retrieval guarantees legal coverage but injects unnecessary chunks into single-domain queries.
- Routed retrieval preserves dual coverage while eliminating unnecessary retrieval on the sanity set.

«Note: This is a small, preliminary evaluation set. No statistical significance is claimed.»

---

Demo

- Streamlit UI: "src/app.py"
- FastAPI service: "src/api.py"

---

API Example

# Health
curl http://localhost:8000/health

# Retrieve only
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are the AAOS recommendations for hip fracture management?\", \"mode\": \"routed\"}"

# Full query (retrieval + generation)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What information should be discussed during surgical informed consent?\", \"mode\": \"routed\"}"

---

Quick Start (Windows PowerShell)

1. Create environment

cd "E:\Orthopedic Law\raven_repro"

# Option A: create a local venv
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1

# Option B: use an existing venv
& "E:\Orthopedic Law\Orthopedic_Law__env\Scripts\Activate.ps1"

2. Install dependencies

uv pip install -r requirements.txt

3. Configure environment

Copy-Item .env.example .env

".env" content:

QDRANT_URL=http://localhost:6333
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

4. Start Qdrant (+ Ollama via Docker)

# Make sure Docker Desktop is running
docker compose up -d
docker ps

If Ollama is already installed locally, you can start only Qdrant:

docker compose up -d qdrant

5. Pull models (only if using Docker Ollama)

docker exec -it raven-ollama ollama pull nomic-embed-text
docker exec -it raven-ollama ollama pull llama3.1:8b

6. Build indices (first time only)

python src/preprocess.py
python src/index.py

python src/prepare_medical_data.py
python src/build_medical_index.py

python src/prepare_legal_data.py
python src/build_legal_index.py

python src/build_ortho_single_index.py

7. Health check

python src/demo_check.py

Expected output:

DEMO READY — all N checks passed

8. Start API

uvicorn src.api:app --reload --port 8000

Open:

http://localhost:8000/docs

9. Start Streamlit

Open a new terminal:

streamlit run src/app.py

Open:

http://localhost:8501

---

Evaluation

eval/
├── questions.csv
├── run_eval.py
├── analyze.py
└── runs/
    ├── ortho_single.json
    ├── dual.json
    └── routed.json

Run:

python eval/run_eval.py
python eval/analyze.py

"analyze.py" outputs:

- a Markdown summary table in the console
- "eval/results_summary.csv"

---

Data

Domain| Source
medical| AAOS Clinical Practice Guideline: Management of Hip Fractures in Older Adults (2021)
legal| CMS Surgical Informed Consent Guidance

«Note: Complete copyrighted source PDFs are not included in this repository. Only segment-level derived text is used locally for development.»

---

Limitations

- Current corpus is small: 35 medical chunks and 4 legal chunks
- Legal collection is limited to a single CMS document
- Evaluation set is a 12-question sanity test designed by the author
- Routing was evaluated on this small sanity set only
- No manual ground-truth chunk annotations yet, so Recall@K is not reported
- Answer-level factuality evaluation is future work

---

Roadmap

- Larger medical and legal corpora
- Evidence sufficiency check
- Citation-level grounding
- Recall@K with manually annotated relevant chunks
- User study with orthopedic clinicians

---

License / Disclaimer

This is a research prototype.

Not for clinical use.

## Limitations & Future Work

- **Answer-level factuality evaluation** is computationally resource-intensive. Local LLM-as-a-judge inference was unstable on the available hardware. Consequently, answer-level claim-support evaluation is excluded from the primary paper metrics and is left for future supplementary analysis.