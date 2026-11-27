from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.rag import ChromaVectorDB, RagConfig, RagPipeline, SentenceTransformerEmbedder, normalize_path
from src.workflow_classes import ClaudeRunner, EvalDataset, EvalGrader, PromptDraft, WorkflowRunner, resolve_prompt_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end dataset + evaluation workflow.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Path to a workflow dataset file. Repeat to pass multiple datasets.",
    )
    parser.add_argument("--rows", type=int, default=25, help="Rows to generate when a dataset file does not exist")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset generation")
    parser.add_argument(
        "--regenerate-dataset",
        action="store_true",
        help="Regenerate generated dataset file(s) even if already present",
    )
    parser.add_argument("--eval-rows", type=int, default=None, help="Optional max rows to evaluate after loading datasets")

    parser.add_argument(
        "--mode",
        choices=["auto", "api", "demo"],
        default="auto",
        help="Execution mode for Claude runner",
    )
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"), help="Prediction model")
    parser.add_argument("--fallback-model", default=os.getenv("ANTHROPIC_FALLBACK_MODEL"), help="Prediction fallback model")
    parser.add_argument("--max-tokens", type=int, default=256, help="Prediction max tokens")

    parser.add_argument(
        "--grader",
        choices=["lexical", "model", "code", "coding"],
        default=os.getenv("EVAL_GRADER", "lexical"),
        help="Grading strategy (lexical|model|code|coding)",
    )
    parser.add_argument(
        "--grader-model",
        default=os.getenv("ANTHROPIC_GRADER_MODEL"),
        help="Grader model when --grader model",
    )
    parser.add_argument(
        "--grader-fallback-model",
        default=os.getenv("ANTHROPIC_GRADER_FALLBACK_MODEL"),
        help="Grader fallback model",
    )
    parser.add_argument("--grader-max-tokens", type=int, default=300, help="Grader max tokens")
    parser.add_argument(
        "--code-timeout-seconds",
        type=float,
        default=3.0,
        help="Timeout in seconds for per-row code execution when --grader code",
    )

    parser.add_argument("--output-dir", default="data/eval_runs", help="Output directory for eval artifacts")
    parser.add_argument("--run-name", default=None, help="Optional run name")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Optional prompt drafting template. Supported placeholders: {question}, {prompt}, {extra_context}",
    )
    parser.add_argument("--extra-prompt", default=None, help="Optional inline extra prompt")
    parser.add_argument("--extra-prompt-file", default=None, help="Optional extra prompt file")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if one row fails")

    parser.add_argument(
        "--rag-document",
        dest="rag_documents",
        action="append",
        default=[],
        help="Optional document path to ingest into workflow RAG index. Repeat for multiple files.",
    )
    parser.add_argument("--rag-chunk-size", type=int, default=220, help="RAG chunk size in words")
    parser.add_argument("--rag-chunk-overlap", type=int, default=40, help="RAG chunk overlap in words")
    parser.add_argument(
        "--rag-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for workflow RAG embeddings",
    )
    parser.add_argument("--rag-vector-db-path", default="data/vector_db/chroma", help="Workflow RAG Chroma directory")
    parser.add_argument("--rag-collection-name", default="rag_chunks", help="Workflow RAG Chroma collection name")
    parser.add_argument(
        "--rag-retrieval-mode",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Workflow RAG retrieval strategy",
    )
    parser.add_argument("--rag-dense-k", type=int, default=20, help="Workflow RAG dense candidate count")
    parser.add_argument("--rag-sparse-k", type=int, default=20, help="Workflow RAG BM25 candidate count")
    parser.add_argument("--rag-rrf-k", type=int, default=60, help="Workflow RAG RRF constant")
    parser.add_argument(
        "--rag-rerank-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reranking in workflow RAG",
    )
    parser.add_argument("--rag-rerank-candidates", type=int, default=20, help="Workflow RAG rerank pool size")
    parser.add_argument("--rag-final-context-k", type=int, default=6, help="Workflow RAG final context chunk count")
    parser.add_argument(
        "--generate-spec-doc",
        action="store_true",
        help="Generate a spec markdown from workflow RAG retrieval context.",
    )
    parser.add_argument(
        "--spec-doc-path",
        default="data/spec_docs/workflow_generated_spec.md",
        help="Output path for workflow-generated spec markdown.",
    )
    parser.add_argument(
        "--spec-query",
        default="Generate a complete product and technical specification from uploaded source documents.",
        help="Objective query used for workflow spec generation retrieval.",
    )
    parser.add_argument(
        "--implementation-target",
        choices=["backend", "frontend", "fullstack"],
        default="fullstack",
        help="Standards target for workflow spec output (backend=.NET, frontend=React, fullstack=both).",
    )
    parser.add_argument("--max-spec-chunks", type=int, default=20, help="Max chunks used for workflow spec generation")
    return parser.parse_args()


def run_rag_steps(args: argparse.Namespace) -> None:
    should_run = bool(args.rag_documents) or bool(args.generate_spec_doc)
    if not should_run:
        return

    vector_db_path = normalize_path(args.rag_vector_db_path)
    vector_db = ChromaVectorDB(vector_db_path, collection_name=args.rag_collection_name)
    vector_db.load()
    print(
        f"[workflow-rag] loaded vector DB entries={vector_db.count()} "
        f"from {vector_db_path} collection={args.rag_collection_name}"
    )

    rag_config = RagConfig(
        mode=args.mode,
        model=args.model,
        fallback_model=args.fallback_model,
        max_tokens=args.max_tokens,
        top_k=4,
        embedding_model=args.rag_embedding_model,
        grader=args.grader,
        grader_model=args.grader_model,
        grader_fallback_model=args.grader_fallback_model,
        grader_max_tokens=args.grader_max_tokens,
        code_timeout_seconds=args.code_timeout_seconds,
        retrieval_mode=args.rag_retrieval_mode,
        dense_k=args.rag_dense_k,
        sparse_k=args.rag_sparse_k,
        rrf_k=args.rag_rrf_k,
        rerank_enabled=args.rag_rerank_enabled,
        rerank_candidates=args.rag_rerank_candidates,
        final_context_k=args.rag_final_context_k,
    )
    rag_pipeline = RagPipeline(
        vector_db=vector_db,
        embedder=SentenceTransformerEmbedder(model_name=args.rag_embedding_model),
        config=rag_config,
    )
    rag_pipeline.setup()

    ingested_total = 0
    for raw_path in args.rag_documents:
        document_path = normalize_path(raw_path)
        ingested_total += rag_pipeline.ingest_document(
            document_path,
            chunk_size=args.rag_chunk_size,
            chunk_overlap=args.rag_chunk_overlap,
            show_chunks=True,
        )

    if args.rag_documents:
        print(
            f"[workflow-rag] completed upload+index for {len(args.rag_documents)} document(s), "
            f"chunks={ingested_total}"
        )

    if args.generate_spec_doc:
        if args.max_spec_chunks <= 0:
            raise ValueError("--max-spec-chunks must be greater than 0")

        spec_result = rag_pipeline.generate_spec_doc(
            spec_query=args.spec_query,
            max_spec_chunks=args.max_spec_chunks,
            implementation_target=args.implementation_target,
        )
        spec_markdown = str(spec_result["spec_markdown"]).strip()
        if not spec_markdown:
            raise ValueError("Generated workflow spec markdown is empty; file write aborted.")
        spec_doc_path = normalize_path(args.spec_doc_path)
        spec_doc_path.parent.mkdir(parents=True, exist_ok=True)
        spec_doc_path.write_text(spec_markdown, encoding="utf-8")
        print(f"[workflow-rag] retrieved contexts: {len(spec_result['retrieved_contexts'])}")
        print(f"[workflow-rag] wrote markdown: {spec_doc_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    run_rag_steps(args)

    dataset_paths = [Path(path) for path in (args.datasets or ["data/datasets/workflow_dataset.jsonl"])]
    for dataset_path in dataset_paths:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

    extra_context = resolve_prompt_context(args.extra_prompt, args.extra_prompt_file)

    dataset = EvalDataset(
        dataset_paths=dataset_paths,
        rows_when_generating=args.rows,
        seed=args.seed,
        regenerate=args.regenerate_dataset,
    )
    prompt_draft = PromptDraft(template=args.prompt_template, extra_context=extra_context)
    claude_runner = ClaudeRunner(
        mode=args.mode,
        model=args.model,
        fallback_model=args.fallback_model,
        max_tokens=args.max_tokens,
    )
    grader = EvalGrader(
        method=args.grader,
        grader_model=args.grader_model,
        grader_fallback_model=args.grader_fallback_model,
        grader_max_tokens=args.grader_max_tokens,
        code_timeout_seconds=args.code_timeout_seconds,
    )

    workflow = WorkflowRunner(
        dataset=dataset,
        prompt_draft=prompt_draft,
        claude=claude_runner,
        grader=grader,
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        eval_rows=args.eval_rows,
        fail_fast=args.fail_fast,
    )

    exit_code = workflow.run()
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
