# RAG Pipeline Specification

## 1) Purpose
This document specifies the end-to-end Retrieval-Augmented Generation (RAG) pipeline currently implemented in `src/rag.py`, including:
- Document ingestion
- Chunking
- Embedding generation
- Vector storage and retrieval
- Answer generation
- RAG evaluation
- Spec-document generation from retrieved context

It is intended as an implementation blueprint to reproduce or extend the existing behavior.

## 2) Scope
In scope:
- CLI-first RAG execution (`python -m src.rag`)
- Workflow integration (`python -m src.workflow` with RAG options)
- Local/demo and API-backed answer generation
- Persistent ChromaDB vector store
- Lexical/model/code grading for RAG eval runs
- Artifact writing for eval and generated spec markdown

Out of scope:
- HTTP service/API wrappers for RAG
- Advanced parsing formats beyond current readers (`xlsx/xls/xlsm/json/text-like`)
- Multi-tenant storage isolation
- (Current baseline only) Reranking, hybrid retrieval, and metadata filtering

## 3) Implementation Sources
Primary code modules:
- `src/rag.py`
- `src/workflow.py`
- `src/client.py`
- `src/eval.py`

Dependencies (from `requirements.txt`):
- `anthropic`
- `python-dotenv`
- `chromadb`
- `sentence-transformers`
- `pandas`
- `openpyxl`

## 4) High-Level Architecture
Core components in `src/rag.py`:

1. `SentenceTransformerEmbedder`
   - Loads embedding model by name.
   - Produces normalized vectors for chunks and query text.

2. `ChromaVectorDB`
   - Wraps persistent Chroma client + collection.
   - Supports load, count, upsert, and similarity search.

3. `RagPipeline`
   - Orchestrates mode setup, ingestion, retrieval, answering, eval, and spec generation.

4. `RagConfig`
   - Runtime configuration for mode/model/retrieval/grader behavior.

5. CLI (`parse_args`, `main`)
   - Connects command-line options to orchestration flow.

## 5) Runtime Modes
`RagPipeline.setup()` supports:
- `auto`: uses `api` if `ANTHROPIC_API_KEY` exists, else `demo`
- `api`: requires `ANTHROPIC_API_KEY`, creates Anthropic client, resolves model via `resolve_model`
- `demo`: deterministic local behavior, no API calls

Grader mode normalization uses `normalize_grader_method` from `src/eval.py`:
- `coding` aliases to `code`
- `model` grader requires API mode

## 6) Data Contracts

### 6.1 Ingestion Input
- One or more `--document` paths.
- Supported read behavior:
  - `.xlsx/.xls/.xlsm`: sheet-aware extraction with columns and row-wise key/value formatting.
  - `.json`: pretty-printed when parseable JSON.
  - Others: UTF-8 text read with `errors="ignore"`.

### 6.2 Chunk Entity
`RagChunk` fields:
- `chunk_id: str` (format: `<document_stem>-<index>`)
- `source: str` (absolute path string)
- `text: str`

### 6.3 Vector DB Record
Stored with:
- `id`: chunk id
- `document`: chunk text
- `metadata`: `{ "source": ..., "chunk_id": ... }`
- `embedding`: normalized float vector

### 6.4 Retrieval Result
`ChromaVectorDB.search` returns list of:
- `score` (computed as `1.0 - distance`)
- `distance`
- `chunk_id`
- `source`
- `text`

### 6.5 QA Output
`RagPipeline.answer_question` returns:
- `question`
- `prompt` (RAG prompt sent to model)
- `contexts` (retrieved chunks)
- `answer`

### 6.6 Spec Generation Output
`RagPipeline.generate_spec_doc` returns:
- `spec_query`
- `implementation_target`
- `retrieved_contexts`
- `spec_markdown`

### 6.7 Eval Artifacts
Under `data/eval_runs/<run_name>/`:
- `predictions.jsonl`
- `report.json`

## 7) Processing Flow

### 7.1 Ingestion + Indexing
1. Resolve and validate document path.
2. Read and normalize source text (`read_document_text`).
3. Split text with `chunk_text`:
   - word-based sliding window
   - `step = chunk_size - chunk_overlap`
4. Build `RagChunk` list and print chunk previews.
5. Generate embeddings via sentence-transformers.
6. Upsert chunks + vectors into Chroma collection.

Validation rules:
- `chunk_size > 0`
- `chunk_overlap >= 0`
- `chunk_overlap < chunk_size`

### 7.2 Retrieval + Answering
1. Embed user question (`embed_query`).
2. Search vector DB with `top_k`.
3. Construct RAG prompt with retrieved contexts.
4. Generate answer:
   - `demo`: top-chunk-based deterministic text
   - `api`: Anthropic chat completion

Prompt policy includes:
- “Answer only from retrieved context”
- “If context insufficient, state what is missing”
- “Cite chunk ids like [1], [2] when used”

### 7.3 Spec Document Generation
1. Embed `spec_query`.
2. Retrieve `max_spec_chunks` contexts.
3. Build markdown output:
   - `demo`: local template (`build_demo_spec_doc`)
   - `api`: model-generated markdown (`build_spec_prompt`)
4. Validate non-empty output.
5. Write to `--spec-doc-path`.

`implementation_target` affects standards guidance:
- `backend`: .NET backend standards
- `frontend`: React frontend standards
- `fullstack`: both

### 7.4 RAG Evaluation
Evaluation is executed by `RagPipeline.run_eval()` and produces per-row predictions plus aggregate run metrics.

#### 7.4.1 Evaluation Inputs
- Dataset path (`--eval-dataset`) must be JSONL and loadable via `load_jsonl_dataset`.
- Required row fields: `prompt`, `answer`.
- Optional row fields:
  - `id`
  - `eval_criteria`
  - `code_grader` (used when grader is `code`/`coding`)
- Optional runtime cap: `--rows` limits evaluated records.

#### 7.4.2 Per-Row Evaluation Process
For each dataset row:
1. Resolve row id and extract `prompt`, `answer`, `eval_criteria`, `code_grader`.
2. Call `answer_question(prompt)`:
   - performs retrieval with configured `top_k`
   - constructs RAG grounding prompt
   - generates response in `demo` or `api` mode
3. Compute baseline lexical metrics:
   - `exact_match(reference, prediction)`
   - `token_f1(reference, prediction)`
   - `criteria_score(eval_criteria, prediction)` when criteria exist
4. Compute baseline weighted `overall_score` via `overall_score(...)` and derive `grade` via `score_grade(...)`.
5. Attach retrieval metadata:
   - `retrieved_chunks` count
   - `retrieved_contexts` payload (chunk ids/source/text/score)

#### 7.4.3 Grader Mode Behavior
- `lexical` (default): keeps lexical weighted score as final score.
- `model`:
  - requires API mode and active Anthropic client
  - calls `model_grader_result(...)`
  - overrides `overall_score` and `grade` using model-judge output
  - stores `model_rationale` and `raw_response` under `grading_details`
- `code` / `coding`:
  - if row has `code_grader`, executes `code_grader_result(...)`
  - uses row-level timeout override when present, else `--code-timeout-seconds`
  - overrides `overall_score` and `grade` from deterministic test pass rate
  - if no `code_grader` exists, row stays lexical-scored with rationale note

#### 7.4.4 Error Handling During Eval
- Row-level failures increment `errors` and continue to next row.
- API failures are printed via `print_api_error` where applicable.
- `--rows <= 0` raises `ValueError` before evaluation.

#### 7.4.5 Aggregate Metrics and Output Artifacts
After row evaluation:
- Aggregate metrics in `report.json`:
  - `avg_overall_score`
  - `avg_token_f1`
  - `exact_match_rate`
  - `avg_retrieved_chunks`
- Run metadata includes:
  - `run_name`, `dataset`, `mode`, `model`
  - `grader`, `grader_model`, `rows_evaluated`, `errors`
  - `vector_db_path`, `vector_db_collection`, `embedding_model`, `generated_at`
- Artifacts written to `data/eval_runs/<run_name>/`:
  - `predictions.jsonl` (row outputs + contexts + metrics)
  - `report.json` (aggregate run summary)

## 8) CLI Specification (`python -m src.rag`)
Key options:
- Ingestion/query:
  - `--document` (repeatable)
  - `--question`
  - `--chunk-size` (default `220`)
  - `--chunk-overlap` (default `40`)
  - `--top-k` (default `4`)
- Embedding/vector store:
  - `--embedding-model` (default `sentence-transformers/all-MiniLM-L6-v2`)
  - `--vector-db-path` (default `data/vector_db/chroma`)
  - `--collection-name` (default `rag_chunks`)
- Mode/model:
  - `--mode auto|api|demo` (default `auto`)
  - `--model`
  - `--fallback-model`
  - `--max-tokens` (default `350`)
- Eval:
  - `--eval-dataset`
  - `--output-dir` (default `data/eval_runs`)
  - `--run-name`
  - `--rows`
  - `--grader lexical|model|code|coding`
  - `--grader-model`
  - `--grader-fallback-model`
  - `--grader-max-tokens` (default `300`)
  - `--code-timeout-seconds` (default `3.0`)
- Spec generation:
  - `--generate-spec-doc`
  - `--spec-doc-path` (default `data/spec_docs/generated_spec.md`)
  - `--spec-query`
  - `--implementation-target backend|frontend|fullstack`
  - `--max-spec-chunks` (default `20`)

Execution behavior:
- Any combination of ingest/question/eval/spec is supported in one run.
- If none of `--question`, `--eval-dataset`, `--generate-spec-doc` are provided, CLI prints guidance and exits.

## 9) Workflow Integration (`python -m src.workflow`)
RAG options are integrated in `run_rag_steps(args)` and executed before dataset/eval workflow.

Workflow-specific flags:
- `--rag-document` (repeatable)
- `--rag-chunk-size`
- `--rag-chunk-overlap`
- `--rag-embedding-model`
- `--rag-vector-db-path`
- `--rag-collection-name`
- `--generate-spec-doc`
- `--spec-doc-path`
- `--spec-query`
- `--implementation-target`
- `--max-spec-chunks`

## 10) Error Handling and Validation
Implemented checks include:
- Missing document path -> `FileNotFoundError`
- Invalid chunk config -> `ValueError`
- API mode without key -> `RuntimeError`
- Model grader in non-API mode -> `RuntimeError`
- Non-positive code timeout -> `ValueError`
- Empty generated markdown -> `ValueError`
- API failures handled with `print_api_error` and non-zero exit in CLI paths

## 11) Non-Functional Characteristics
- Persistence: ChromaDB persists to local directory.
- Reproducibility: deterministic demo mode behavior and fixed defaults.
- Transparency: chunk previews and retrieval hit summaries printed to console.
- Extensibility: parser, embedder, and vector DB are encapsulated and swappable.

## 12) Minimal End-to-End Execution Examples

Ingest:
```bash
python -m src.rag \
  --document data/context/athlete_context.xml \
  --collection-name rag_chunks \
  --mode demo
```

Query:
```bash
python -m src.rag \
  --question "Summarize the athlete profile" \
  --collection-name rag_chunks \
  --mode demo
```

Spec generation:
```bash
python -m src.rag \
  --document data/context/requirements.xlsx \
  --generate-spec-doc \
  --implementation-target backend \
  --spec-doc-path data/spec_docs/requirements_spec.md \
  --mode demo
```

RAG eval:
```bash
python -m src.rag \
  --document data/context/athlete_context.xml \
  --eval-dataset data/datasets/rag_eval_sample.jsonl \
  --grader lexical \
  --rows 10 \
  --mode demo
```

## 13) Implementation Checklist
- [ ] Configure environment and install dependencies.
- [ ] Initialize persistent Chroma collection.
- [ ] Implement/verify document readers (spreadsheet + text/json).
- [ ] Implement validated chunking function.
- [ ] Implement embedding and upsert flow.
- [ ] Implement retrieval + grounded prompt construction.
- [ ] Implement demo and API answer generation modes.
- [ ] Implement dataset-based RAG eval with grader options.
- [ ] Implement spec generation from retrieval context.
- [ ] Write run artifacts (`predictions.jsonl`, `report.json`, spec markdown).

## 14) Next-Phase Target Architecture (High/Medium Priority)
This section defines the implementation plan for the next production-grade RAG iteration.

Priority levels:
- High: must be delivered in the first upgrade milestone.
- Medium: deliver after high-priority retrieval/evaluation stability is in place.

### 14.1 High Priority: Hybrid Retrieval
Current state uses semantic vector retrieval only. Target state combines sparse + dense retrieval with fusion.

#### 14.1.1 Retrieval Components
1. Semantic vector retrieval (existing)
   - backend: Chroma similarity search
   - candidate count: `k_dense` (default `20`)
2. BM25 keyword retrieval (new)
   - backend options: `rank_bm25` in-memory index or Tantivy/Lucene-backed index
   - candidate count: `k_sparse` (default `20`)
3. Reciprocal Rank Fusion (RRF) (new)
   - fuse ranked lists from dense + sparse searches
   - default constant: `rrf_k = 60`
   - score formula per chunk id $c$:
     $$
	ext{RRF}(c)=\sum_{m \in \{dense,sparse\}} \frac{1}{k + rank_m(c)}
     $$

#### 14.1.2 Why Required
- Improves recall for exact tokens (e.g., `Kafka`, `SQS`, API names, field names, business IDs) that embeddings may underweight.
- Preserves semantic match quality for paraphrased requirements.

#### 14.1.3 Data/Index Requirements
- Maintain both indexes from same canonical chunks.
- BM25 index must store:
  - `chunk_id`
  - tokenized text
  - metadata (`doc_id`, `doc_name`, `section`, `page`, `version`)
- Ensure deterministic lowercasing + tokenization rules shared between ingestion and query.

#### 14.1.4 CLI/Config Additions
- `--retrieval-mode dense|sparse|hybrid` (default `hybrid`)
- `--dense-k 20`
- `--sparse-k 20`
- `--rrf-k 60`

#### 14.1.5 Acceptance Criteria
- On exact-term benchmark queries, hybrid recall@20 >= dense-only recall@20.
- Retrieval response logs include per-source rank and fused rank.
- Fallback behavior: if one retriever fails, pipeline continues with the other retriever and logs degradation.

### 14.2 High Priority: Reranking
After initial retrieval, rerank candidates and pass only highest-value chunks to generation.

#### 14.2.1 Candidate Pipeline
1. Retrieve hybrid candidate pool (target 15-20).
2. Deduplicate by stable `chunk_id`.
3. Rerank candidates using one of:
   - Cross-encoder reranker (preferred; e.g., `bge-reranker-*`)
   - LLM reranking prompt (fallback)
4. Select top `final_k` chunks (default 6; range 5-8).

#### 14.2.2 Rerank Features (minimum)
- query/chunk semantic similarity
- exact term overlap score
- section title boost (if heading matches query intent)
- recency/version bias (optional)

#### 14.2.3 CLI/Config Additions
- `--rerank-enabled true|false` (default `true`)
- `--rerank-model <name>`
- `--rerank-candidates 20`
- `--final-context-k 6`

#### 14.2.4 Acceptance Criteria
- nDCG@10 and MRR improve versus non-reranked baseline on evaluation set.
- LLM prompt token count stays within configured token budget while maintaining answer quality.

### 14.3 High Priority: Better Evaluation (Retrieval + Generation Split)
Evaluation must separately score retrieval quality and answer quality.

#### 14.3.1 Retrieval Metrics (new)
- Context precision@k:
  $$
	ext{Precision@k}=\frac{|\text{retrieved}_k \cap \text{relevant}|}{k}
  $$
- Context recall@k:
  $$
	ext{Recall@k}=\frac{|\text{retrieved}_k \cap \text{relevant}|}{|\text{relevant}|}
  $$
- Optional ranking metrics: MRR, nDCG@k.

Requires labeled eval set with one or more gold relevant chunk ids per query.

#### 14.3.2 Generation Metrics (new)
- Faithfulness: answer claims supported by retrieved context.
- Answer relevance: answer addresses query intent.
- Citation correctness:
  - every citation maps to existing retrieved chunk id
  - cited chunk text supports local claim span
- Keep existing lexical metrics (`exact_match`, `token_f1`, `criteria_score`) for trend continuity.

#### 14.3.3 Evaluation Runner Changes
- Add retrieval-only mode:
  - `--eval-retrieval-only`
- Add generation-only mode using frozen retrieved contexts:
  - `--eval-generation-only`
- Add full mode:
  - `--eval-full`

#### 14.3.4 Dataset Schema Extension
Per evaluation row should support:
- `query`
- `reference_answer`
- `gold_chunk_ids` (list)
- `required_citations` (optional list of required chunk ids)
- `eval_criteria`

#### 14.3.5 Report Output Extension
`report.json` and `predictions.jsonl` must include:
- retrieval metrics block
- generation metrics block
- citation validation details
- retrieval trace (dense rank, sparse rank, fused rank, rerank score)

### 14.4 Medium Priority: Document-Aware Chunking
Replace pure word window chunking with structure-preserving chunking.

#### 14.4.1 Chunking Strategy
1. Parse document into structure units:
   - headings
   - paragraphs
   - lists
   - tables
2. Build chunks by section boundaries first.
3. Enforce token/word limits with smart split at paragraph boundaries.
4. Use small overlap only when required for continuity.

#### 14.4.2 Excel/Table-Aware Rules
- Keep header row attached to each row chunk.
- Preserve sheet name and table range metadata.
- Chunk by logical row groups rather than arbitrary word windows.

#### 14.4.3 Metadata Requirements
Each chunk must include:
- `chunk_id` (stable)
- `doc_id`
- `doc_name`
- `doc_version`
- `section_path` (e.g., `2.1 > API Contracts > CreateOrder`)
- `page_number` (nullable)
- `sheet_name` and `row_range` for spreadsheets
- `content_type` (`paragraph|heading|table|list|code`)

#### 14.4.4 Acceptance Criteria
- Heading-aware chunks improve section-targeted query recall.
- Table queries return chunks containing relevant table headers and row values.

### 14.5 Medium Priority: Production Reliability

#### 14.5.1 Document Versioning and Deletion
- Introduce `doc_id` and immutable `doc_version`.
- Support soft-delete and hard-delete by `doc_id`/`doc_version`.
- Deletion must remove both vector and BM25 entries.

#### 14.5.2 Stable Unique Chunk IDs
Use deterministic ID scheme:
`chunk_id = sha1(doc_id + doc_version + section_path + normalized_text)`.

Benefits:
- idempotent re-index
- reliable citation references
- deduplication across ingestion retries

#### 14.5.3 Retry/Timeout Policies
- Embedding, retrieval, and generation calls require bounded retries with exponential backoff.
- Timeouts per stage:
  - embedding timeout
  - retrieval timeout
  - generation timeout
  - reranker timeout
- Degrade gracefully to partial pipeline where safe.

#### 14.5.4 Token-Budget Management
- Compute available context window before generation.
- Reserve output token budget first, then fit context chunks.
- Truncate with deterministic policy:
  1. keep highest reranked chunks
  2. trim long chunks by sentence boundary
  3. never exceed model token limit

#### 14.5.5 Logging, Latency, and Cost Tracking
Capture structured telemetry per request:
- stage latencies (`ingest_ms`, `embed_ms`, `retrieve_ms`, `rerank_ms`, `generate_ms`)
- token usage (`input_tokens`, `output_tokens`)
- estimated cost
- selected model/retriever/reranker config
- degradation/fallback events

#### 14.5.6 Prompt Injection Protection
- Treat retrieved text as untrusted data.
- System prompt must explicitly ignore instructions from context documents.
- Strip/flag known injection patterns.
- Separate control instructions from context payload.

#### 14.5.7 Source Traceability
- Require citations as chunk IDs in final answer.
- Maintain citation map from chunk id -> doc/version/section/page.
- Expose trace object in output for auditing.

## 15) Requirements-Focused Spec Generation (Specific Design)
This profile optimizes spec-document generation for technical requirements documents.

### 15.1 Goal
Generate highly structured, implementation-ready requirements specs with precise source grounding and citations.

### 15.2 Retrieval Profile for Requirements
- Force `retrieval_mode=hybrid`.
- Increase candidate pool (`dense_k=25`, `sparse_k=25`, `rerank_candidates=30`).
- Final context set `final_context_k=8`.
- Boost chunks from sections matching keywords:
  - `requirements`, `api`, `schema`, `acceptance criteria`, `non-functional`, `constraints`.

### 15.3 Spec Generation Prompt Contract
Generated spec must include sections:
1. Objective
2. In Scope / Out of Scope
3. Assumptions
4. Functional Requirements
5. Non-Functional Requirements
6. Data Model / Data Dictionary
7. API Contracts and Integrations
8. Business Rules and Validation Rules
9. Edge Cases and Failure Scenarios
10. Security and Compliance Considerations
11. Observability and Operational Requirements
12. Test Strategy and Acceptance Criteria
13. Open Questions / Risks
14. Requirement-to-Source Traceability Matrix

Every requirement statement must include one or more chunk citations.

### 15.4 Requirements-Specific Output Artifacts
When `--generate-spec-doc` runs in requirements mode, write:
- `spec.md` (human-readable spec)
- `traceability.json` (requirement ID -> chunk IDs -> source metadata)
- `retrieval_debug.json` (dense/sparse/fused/rerank diagnostics)

### 15.5 CLI/Config Additions
- `--spec-profile default|requirements` (default `default`)
- `--spec-strict-citations true|false` (default `true` for requirements)
- `--spec-min-citations-per-section <int>` (default `1`)

### 15.6 Acceptance Criteria
- 100% of normative requirement bullets include valid citations.
- Traceability file resolves every cited chunk to source metadata.
- Requirements profile improves reviewer-rated completeness and actionability vs default profile.

## 16) Suggested Rollout Plan
Phase 1 (High Priority):
- Hybrid retrieval + RRF
- Reranking
- Retrieval/generation split evaluation

Phase 2 (Medium Priority):
- Document-aware chunking + metadata upgrade
- Reliability controls (versioning, deletion, retries, token budget, telemetry, injection defenses)

Phase 3 (Spec Quality):
- Requirements-specific spec profile and traceability artifacts
- Calibration of citation correctness and faithfulness thresholds

## 17) Updated Implementation Checklist
- [ ] Add BM25 index layer and query path.
- [ ] Add hybrid fusion (`RRF`) retrieval path.
- [ ] Add reranker stage and top-k context selector.
- [ ] Extend chunk schema with document/section/page metadata.
- [ ] Implement stable deterministic chunk IDs.
- [ ] Add versioned document registry and delete APIs/CLI commands.
- [ ] Extend eval datasets with `gold_chunk_ids` and citation labels.
- [ ] Add retrieval metrics (`precision@k`, `recall@k`, `MRR`, `nDCG`).
- [ ] Add generation metrics (faithfulness, relevance, citation correctness).
- [ ] Add requirements-specific spec profile + traceability outputs.
- [ ] Add telemetry for latency/tokens/cost and fallback events.
