# Methodology

## 1. Corpus Construction
We constructed a domain-specific corpus comprising two distinct domains: Medical and Legal.

- **Medical Domain**: Includes clinical practice guidelines from AAOS (American Academy of Orthopaedic Surgeons) and BOA (British Orthopaedic Association).
- **Legal Domain**: Includes CMS (Centers for Medicare & Medicaid Services) informed consent guidance and court cases regarding medical malpractice.

The full provenance, jurisdiction, and copyright status of all source documents are tracked in `data/corpus_manifest.csv`.

## 2. Preprocessing and Embedding
All raw PDFs were parsed using PyMuPDF to extract plain text, removing repeated headers and footers.
- **Chunk Size**: 2000 tokens
- **Chunk Overlap**: 200 tokens
- **Embedding Model**: `nomic-embed-text` (via Ollama API, Vector Dimension: 768)

## 3. Qdrant Indexing
Four independent collections were created in Qdrant using Cosine similarity for distance measurement:
- `cfr493`: The original Raven reproduction (42 CFR Part 493).
- `medical`: Exclusively contains medical domain chunks.
- `legal_ortho`: Exclusively contains legal domain chunks.
- `ortho_single`: A single-pool baseline. To ensure a fair comparison, this collection is built by directly copying the exact vectors and payloads from `medical` and `legal_ortho` without re-embedding or re-chunking.

## 4. Retrieval Strategies
We compare three retrieval organizations for the orthopedic corpus:

### 4.1 Single-Pool Retrieval (`ortho_single`)
The system retrieves the **Top-6** most similar chunks from the mixed `ortho_single` collection, regardless of domain.

### 4.2 Fixed Dual Retrieval (`dual`)
The system unconditionally retrieves **Top-3** chunks from the `medical` collection and **Top-3** chunks from the `legal_ortho` collection, merging them into a single context of 6 chunks.

### 4.3 Evidence Requirement Routing (`routed`)
A lightweight, zero-shot LLM-based router is introduced before retrieval.
- **Router LLM**: `llama3.1:8b` (Temperature = 0).
- **Routing Logic**: The LLM evaluates the question and outputs a strict JSON structure indicating whether `medical_required` and/or `legal_required` are true.
- **Fallback Mechanism**: If the LLM fails, times out, or returns invalid JSON, the system gracefully defaults to the `dual` retrieval path `(medical_required=True, legal_required=True)`.
- **Execution**:
  - *Medical-only*: Retrieves Top-6 from `medical`.
  - *Legal-only*: Retrieves Top-6 from `legal_ortho`.
  - *Cross-domain*: Retrieves Top-3 from `medical` and Top-3 from `legal_ortho`.

## 5. Generation Setup
After retrieval, the chunks are concatenated with `[domain|source]` tags and passed to the LLM.
- **Generation LLM**: `llama3.1:8b`
- **Temperature**: 0 (deterministic)
- **Max Output Tokens**: 512 (`num_predict=512`)
- **Timeout**: 300 seconds
- **System Prompt**: Enforces strict adherence to the provided text, requires inline citations, and explicitly allows abstention if evidence is insufficient.

## 6. Evaluation Setup
Retrieval evaluations are conducted using source-level ground truth (manually annotated).
- **Domain Coverage**: Whether the retrieved chunks include at least one chunk from the required domain(s).
- **Unnecessary Retrieval**: The number of retrieved chunks belonging to a non-required domain.
- **Cross-domain Balance**: The ratio of medical to legal chunks retrieved for cross-domain questions.

## 7. Reproducibility
All experiments are fully reproducible.
- The LLM calls are made via native HTTP requests to the Ollama API to avoid framework version conflicts.
- Embedding parameters, Top-K values, and LLM temperatures are hardcoded or drawn from environment variables and remain constant across evaluations.
- The parity between `ortho_single` and the isolated domain collections is automatically verified via a dedicated script before evaluation to guarantee no data leakage or mismatch.