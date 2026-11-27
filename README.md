# Claude Code Container Starter

Minimal Python container starter for experimenting with Anthropic's SDK.

## Files
- `src/app.py`: tiny runnable entrypoint
- `app.py`: compatibility launcher that delegates to `src.app`
- `requirements.txt`: Python dependencies
- `Dockerfile`: container image for the app
- `docker-compose.yml`: convenience compose config
- `.env.example`: template for local secrets

## Quick start

### Local Python
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.app
```

### Docker
```bash
docker build -t claude-code-container .
docker run --rm --env-file .env claude-code-container
```

## Environment
Set `ANTHROPIC_API_KEY` in `.env` before making API calls.

## Generate a Dataset

Create a synthetic prompt/answer dataset in `jsonl` (default):

```bash
python -m src.dataset --rows 200 --output data/datasets/synthetic_dataset.jsonl
```

For reusable prompting and grading orchestration, use the class-based workflow in `src.workflow` / `src.workflow_classes`.

## RAG Pipeline (Upload -> Chunk -> Embed -> Vector DB -> Retrieve -> Answer -> Eval)

Use `src.rag` when you want retrieval-augmented generation over uploaded local documents.

What it does:
- Upload/ingest one or more documents with `--document`
- Supports structured spreadsheet ingestion for `.xlsx/.xls/.xlsm`
- Chunk documents (word-based sliding window)
- Print each chunk preview to console for visibility
- Embed chunks with `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- Upsert vectors into persistent `ChromaDB`
- Retrieve top-k chunks for each question
- Generate an answer grounded in retrieved context (API mode) or deterministic demo answer (demo mode)
- Optionally generate a Markdown spec document from retrieved context
- Optionally run a RAG eval dataset and write run artifacts

Quick start:

```bash
# Ingest and index one document
python -m src.rag \
	--document data/context/athlete_context.xml \
	--collection-name rag_chunks \
	--mode demo

# Ask a question with retrieval
python -m src.rag \
	--question "What are the key athlete goals in the uploaded context?" \
	--collection-name rag_chunks \
	--mode demo

# Ingest + ask in one run
python -m src.rag \
	--document data/context/athlete_context.xml \
	--question "Summarize the athlete profile" \
	--embedding-model sentence-transformers/all-MiniLM-L6-v2 \
	--collection-name rag_chunks \
	--chunk-size 220 \
	--chunk-overlap 40 \
	--top-k 4 \
	--mode auto

# Ingest Excel and generate specification markdown
python -m src.rag \
	--document data/context/requirements.xlsx \
	--collection-name rag_chunks \
	--mode demo \
	--generate-spec-doc \
	--implementation-target backend \
	--spec-doc-path data/spec_docs/requirements_spec.md
```

RAG eval:

```bash
python -m src.rag \
	--document data/context/athlete_context.xml \
	--eval-dataset data/datasets/rag_eval_sample.jsonl \
	--collection-name rag_chunks \
	--grader lexical \
	--rows 10 \
	--mode demo
```

RAG eval reuses the existing grader implementation from `src.eval`:
- `--grader lexical|model|code|coding`
- `--grader-model`, `--grader-fallback-model`, `--grader-max-tokens`
- `--code-timeout-seconds`

Example model grader run:

```bash
python -m src.rag \
	--eval-dataset data/datasets/rag_eval_sample.jsonl \
	--mode api \
	--grader model \
	--model claude-3-5-sonnet-latest \
	--grader-model claude-3-5-sonnet-latest
```

RAG run artifacts are stored under `data/eval_runs/<run_name>/`:
- `predictions.jsonl`: per-row prediction + retrieved chunks + lexical metrics
- `report.json`: aggregate RAG metrics (`avg_overall_score`, `avg_token_f1`, `exact_match_rate`, `avg_retrieved_chunks`)

Spec generation options:
- `--generate-spec-doc`
- `--spec-doc-path <path>`
- `--spec-query "..."`
- `--implementation-target backend|frontend|fullstack`
- `--max-spec-chunks <int>`

Standards behavior:
- `backend`: spec is guided by .NET backend coding standards
- `frontend`: spec is guided by React frontend standards
- `fullstack`: includes both backend and frontend standards

## Run Evaluation Pipeline

Evaluate a JSONL dataset (`prompt` + `answer`) and generate per-row predictions + aggregated metrics.

You can pass one or many datasets by repeating `--dataset`.

Optional: include `eval_criteria` in each row to score criteria coverage (`avg_criteria_score`).

Example row:

```json
{"id":"1","prompt":"...","answer":"...","eval_criteria":["Includes breakfast/lunch/dinner","Aligns with goal"]}
```

Auto mode (uses API if `ANTHROPIC_API_KEY` exists, otherwise demo mode):

```bash
python -m src.eval --dataset data/datasets/synthetic_dataset.jsonl --rows 50

# Single feed with multiple datasets
python -m src.eval \
	--dataset data/datasets/synthetic_dataset.jsonl \
	--dataset data/datasets/workflow_dataset.jsonl \
	--dataset data/datasets/code_eval_sample.jsonl
```

Force Anthropic API mode:

```bash
python -m src.eval --mode api --model claude-3-5-sonnet-latest --dataset data/datasets/synthetic_dataset.jsonl
```

Force local demo mode (no API key required):

```bash
python -m src.eval --mode demo --dataset data/datasets/synthetic_dataset.jsonl --rows 25
```

Inject XML/context inline:

```bash
python -m src.eval \
	--mode demo \
		--dataset data/datasets/synthetic_dataset.jsonl \
	--extra-prompt "<athlete_information><goal>Build muscle</goal></athlete_information>"
```

Inject XML/context from file:

```bash
python -m src.eval \
	--mode demo \
		--dataset data/datasets/synthetic_dataset.jsonl \
		--extra-prompt-file data/context/athlete_context.xml
```

Evaluation artifacts are written under `data/eval_runs/<run_name>/`:

- `report.json`: aggregate metrics (`exact_match_rate`, `avg_token_f1`, token lengths, metadata)
- `predictions.jsonl`: one row per evaluated sample with prediction + metrics
- `report.html`: human-friendly score dashboard with overall metrics and per-row scores

Use model-based grading (LLM judge) instead of lexical-only scoring:

```bash
python -m src.eval \
	--mode api \
	--grader model \
	--model claude-3-5-sonnet-latest \
	--grader-model claude-3-5-sonnet-latest \
	--dataset data/datasets/synthetic_dataset.jsonl
```

Grader-related options:

- `--grader lexical|model|code|coding` (default: lexical; `coding` aliases `code`)
- `--grader-model <model_name>`
- `--grader-fallback-model <model_name>`
- `--grader-max-tokens <int>`
- `--code-timeout-seconds <float>`

### Code Grader (deterministic tests)

Use `--grader coding` (or `--grader code`) to score generated code by executing test cases declared in each dataset row.

Dataset row format (additional optional field):

```json
{
	"id": "code-1",
	"prompt": "Write a Python function add(a, b) that returns the sum.",
	"answer": "def add(a, b):\n    return a + b",
	"code_grader": {
		"mode": "function",
		"function_name": "add",
		"tests": [
			{"name": "ints", "args": [2, 3], "expected": 5},
			{"name": "negative", "args": [-1, 7], "expected": 6}
		]
	}
}
```

Run with code grading:

```bash
python -m src.eval \
	--mode demo \
	--grader coding \
	--dataset data/datasets/code_eval_sample.jsonl
```

Notes:
- Code grading expects Python code in the model output (fenced `python` block preferred).
- If `code_grader` is missing for a row, lexical metrics are kept and a rationale is added.
- Generated code execution is intended for trusted datasets.

## One-Command Workflow (Dataset + Eval)

Run end-to-end workflow (class-based and reusable) with grader integrated:

```bash
python -m src.workflow --mode demo --grader lexical --rows 25
```

Use a custom reusable prompt template (supports `{question}`, `{prompt}`, `{extra_context}`):

```bash
python -m src.workflow \
	--mode demo \
	--dataset data/datasets/dataset_class_smoke.jsonl \
	--prompt-template "Short answer for: {question}"
```

Pass multiple datasets into one workflow feed:

```bash
python -m src.workflow \
	--mode demo \
	--dataset data/datasets/synthetic_dataset.jsonl \
	--dataset data/datasets/code_eval_sample.jsonl \
	--eval-rows 20
```

Use model grader in API mode:

```bash
python -m src.workflow \
	--mode api \
	--grader model \
	--model claude-3-5-sonnet-latest \
	--grader-model claude-3-5-sonnet-latest \
	--rows 50
```

Run workflow with RAG document ingest + spec generation:

```bash
python -m src.workflow \
	--mode demo \
	--rag-document data/context/sample_requirements.xlsx \
	--rag-collection-name rag_chunks_workflow \
	--generate-spec-doc \
	--implementation-target backend \
	--spec-doc-path data/spec_docs/workflow_requirements_spec.md \
	--spec-query "Create a complete product and technical specification" \
	--max-spec-chunks 10
```

Workflow RAG options:
- `--rag-document` (repeatable)
- `--rag-chunk-size`, `--rag-chunk-overlap`
- `--rag-embedding-model`
- `--rag-vector-db-path`, `--rag-collection-name`
- `--generate-spec-doc`, `--spec-doc-path`, `--spec-query`, `--max-spec-chunks`

Workflow defaults:

- Creates `data/datasets/workflow_dataset.jsonl` if missing
- Reuses existing dataset unless `--regenerate-dataset` is set
- Uses reusable classes in `src/workflow_classes.py`:
	- `PromptDraft` for prompt templating
	- `EvalDataset` for dataset loading/generation
	- `ClaudeRunner` for prediction
	- `EvalGrader` for scoring
	- `WorkflowRunner` for orchestration and artifact writing
