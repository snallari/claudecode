from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.client import chat, get_text_from_response, print_api_error, resolve_model
from src.eval import (
    code_grader_result,
    criteria_score,
    exact_match,
    load_jsonl_dataset,
    model_grader_result,
    normalize_grader_method,
    overall_score,
    score_grade,
    token_f1,
    write_json,
    write_jsonl,
)

DEFAULT_MODEL = "claude-3-5-sonnet-latest"


def log_step(message: str) -> None:
    print(f"[rag] {message}")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def read_document_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".json":
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return raw

    return raw


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        segment = words[start : start + chunk_size]
        if not segment:
            continue
        chunks.append(" ".join(segment))
        if start + chunk_size >= len(words):
            break
    return chunks


@dataclass
class RagChunk:
    chunk_id: str
    source: str
    text: str


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []


class ChromaVectorDB:
    def __init__(self, persist_directory: Path, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client: chromadb.PersistentClient | None = None
        self.collection = None

    def load(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def count(self) -> int:
        if self.collection is None:
            return 0
        return int(self.collection.count())

    def upsert_chunks(self, chunks: list[RagChunk], vectors: list[list[float]]) -> None:
        if self.collection is None:
            raise RuntimeError("Vector DB collection is not initialized.")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [{"source": chunk.source, "chunk_id": chunk.chunk_id} for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )

    def search(self, query_vector: list[float], top_k: int = 4) -> list[dict[str, Any]]:
        if self.collection is None or top_k <= 0:
            return []
        if not query_vector:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        output: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            cosine_similarity = 1.0 - float(distance)
            output.append(
                {
                    "score": cosine_similarity,
                    "distance": float(distance),
                    "chunk_id": str(metadata_dict.get("chunk_id", "")),
                    "source": str(metadata_dict.get("source", "")),
                    "text": str(document or ""),
                }
            )
        return output


@dataclass
class RagConfig:
    mode: str
    model: str
    fallback_model: str | None
    max_tokens: int
    top_k: int
    embedding_model: str
    grader: str
    grader_model: str | None
    grader_fallback_model: str | None
    grader_max_tokens: int
    code_timeout_seconds: float


class RagPipeline:
    def __init__(self, *, vector_db: ChromaVectorDB, embedder: SentenceTransformerEmbedder, config: RagConfig):
        self.vector_db = vector_db
        self.embedder = embedder
        self.config = config
        self.client: anthropic.Anthropic | None = None
        self.active_mode = "demo"
        self.active_model = "demo-local"
        self.active_grader_method = "lexical"
        self.active_grader_model: str | None = None

    def setup(self) -> None:
        requested_mode = self.config.mode
        api_key = os.getenv("ANTHROPIC_API_KEY")

        self.active_mode = requested_mode
        if requested_mode == "auto":
            self.active_mode = "api" if api_key else "demo"

        log_step(f"mode resolved requested={requested_mode} active={self.active_mode}")

        if self.active_mode == "api":
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for API mode")

            self.client = anthropic.Anthropic(api_key=api_key)
            self.active_model, note = resolve_model(self.client, self.config.model, self.config.fallback_model)
            if note:
                print(note)
        else:
            self.active_model = "demo-local"

        self.active_grader_method = normalize_grader_method(self.config.grader)
        if self.active_grader_method == "model":
            if self.active_mode != "api" or self.client is None:
                raise RuntimeError("--grader model requires API mode.")
            requested_grader_model = self.config.grader_model or self.active_model
            self.active_grader_model, note = resolve_model(
                self.client,
                requested_grader_model,
                self.config.grader_fallback_model,
            )
            if note:
                print(f"Grader: {note}")

        if self.active_grader_method == "code" and self.config.code_timeout_seconds <= 0:
            raise ValueError("--code-timeout-seconds must be greater than 0")

    def ingest_document(self, document_path: Path, *, chunk_size: int, chunk_overlap: int, show_chunks: bool = True) -> int:
        source = str(document_path)
        text = read_document_text(document_path)
        chunks_text = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks_text:
            log_step(f"no chunks generated for {source}")
            return 0

        rag_chunks: list[RagChunk] = []
        for index, chunk in enumerate(chunks_text, start=1):
            chunk_id = f"{document_path.stem}-{index}"
            rag_chunks.append(RagChunk(chunk_id=chunk_id, source=source, text=chunk))

        if show_chunks:
            print(f"\n[upload] {source}")
            print(f"[chunking] produced {len(rag_chunks)} chunks")
            for chunk in rag_chunks:
                preview = chunk.text[:140].replace("\n", " ").strip()
                print(f"  - {chunk.chunk_id}: {preview}{'...' if len(chunk.text) > 140 else ''}")

        vectors = self.embedder.embed_batch([chunk.text for chunk in rag_chunks])
        vector_dim = len(vectors[0]) if vectors else 0
        print(
            f"[embedding] embedded {len(vectors)} chunks with model={self.config.embedding_model} dim={vector_dim}"
        )

        self.vector_db.upsert_chunks(rag_chunks, vectors)
        print(
            f"[vector-db] upserted into {self.vector_db.persist_directory} "
            f"collection={self.vector_db.collection_name} total_entries={self.vector_db.count()}"
        )
        return len(rag_chunks)

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed_query(question)
        results = self.vector_db.search(query_vector, top_k=self.config.top_k)
        print(f"[retrieval] top_k={self.config.top_k} hits={len(results)}")
        for result in results:
            preview = str(result.get("text", ""))[:120].replace("\n", " ").strip()
            print(
                f"  - chunk={result.get('chunk_id','')} score={float(result.get('score', 0.0)):.4f} "
                f"source={result.get('source','')} text={preview}{'...' if len(str(result.get('text',''))) > 120 else ''}"
            )
        return results

    def build_rag_prompt(self, question: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            context_block = "No context chunks available in vector DB."
        else:
            lines = []
            for index, context in enumerate(contexts, start=1):
                lines.append(
                    f"[{index}] source={context.get('source', '')} chunk={context.get('chunk_id', '')}\n"
                    f"{context.get('text', '')}"
                )
            context_block = "\n\n".join(lines)

        return (
            "You are a retrieval-augmented assistant. "
            "Answer only from the retrieved context. "
            "If context is insufficient, clearly state what is missing.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            "Return a concise answer and cite chunk ids like [1], [2] when used."
        )

    def answer_question(self, question: str) -> dict[str, Any]:
        contexts = self.retrieve(question)
        prompt = self.build_rag_prompt(question, contexts)

        if self.active_mode == "demo":
            if contexts:
                top = contexts[0]
                answer = (
                    "Demo mode answer (based on top retrieved chunk): "
                    f"{str(top.get('text', ''))[:280]}"
                )
            else:
                answer = "Demo mode answer: no retrieved context available."
        else:
            if self.client is None:
                raise RuntimeError("Anthropic client not initialized.")
            response = chat(
                client=self.client,
                messages=[{"role": "user", "content": prompt}],
                model=self.active_model,
                max_tokens=self.config.max_tokens,
                stream=False,
            )
            answer = get_text_from_response(response).strip()

        return {
            "question": question,
            "prompt": prompt,
            "contexts": contexts,
            "answer": answer,
        }

    def run_eval(
        self,
        *,
        dataset_path: Path,
        output_dir: Path,
        run_name: str | None,
        max_rows: int | None,
    ) -> dict[str, Any]:
        rows = load_jsonl_dataset(dataset_path)
        if max_rows is not None:
            if max_rows <= 0:
                raise ValueError("--rows must be greater than 0")
            rows = rows[:max_rows]

        prediction_rows: list[dict[str, Any]] = []
        errors = 0

        for index, row in enumerate(rows, start=1):
            row_id = str(row.get("id", index))
            question = str(row.get("prompt", "")).strip()
            reference = str(row.get("answer", "")).strip()
            eval_criteria = [str(item) for item in row.get("eval_criteria", [])]
            code_grader_spec = row.get("code_grader")
            print(f"[eval {index}/{len(rows)}] id={row_id}")

            try:
                result = self.answer_question(question)
                prediction = str(result["answer"])
                metrics = {
                    "exact_match": exact_match(reference, prediction),
                    "token_f1": token_f1(reference, prediction),
                    "criteria_score": criteria_score(eval_criteria, prediction),
                }
                criteria_value: float | None = metrics["criteria_score"] if eval_criteria else None
                metrics["overall_score"] = overall_score(
                    metrics["exact_match"],
                    metrics["token_f1"],
                    criteria_value,
                )
                metrics["grade"] = score_grade(metrics["overall_score"])
                metrics["retrieved_chunks"] = len(result["contexts"])
                metrics["grading_details"] = {"method": self.active_grader_method}

                if self.active_grader_method == "model":
                    if self.client is None or not self.active_grader_model:
                        raise RuntimeError("Model grader is not configured.")
                    grader_output = model_grader_result(
                        client=self.client,
                        grader_model=self.active_grader_model,
                        max_tokens=self.config.grader_max_tokens,
                        prompt=question,
                        reference=reference,
                        prediction=prediction,
                        criteria=eval_criteria,
                    )
                    metrics["overall_score"] = grader_output["score"]
                    metrics["grade"] = grader_output["grade"]
                    metrics["grading_details"] = {
                        "method": "model",
                        "model": self.active_grader_model,
                        "model_rationale": grader_output["rationale"],
                        "raw_response": grader_output["raw_response"],
                    }
                elif self.active_grader_method == "code":
                    if code_grader_spec:
                        timeout_seconds = float(code_grader_spec.get("timeout_seconds", self.config.code_timeout_seconds))
                        grader_output = code_grader_result(
                            prediction=prediction,
                            code_spec=code_grader_spec,
                            timeout_seconds=timeout_seconds,
                        )
                        metrics["overall_score"] = grader_output["score"]
                        metrics["grade"] = grader_output["grade"]
                        metrics["grading_details"] = {
                            "method": "code",
                            "rationale": grader_output["rationale"],
                            "details": grader_output["details"],
                        }
                    else:
                        metrics["grading_details"] = {
                            "method": "code",
                            "rationale": "No code_grader spec provided for this row; using lexical metrics only.",
                        }

                prediction_rows.append(
                    {
                        "id": row_id,
                        "dataset": str(dataset_path),
                        "prompt": question,
                        "reference": reference,
                        "prediction": prediction,
                        "eval_criteria": eval_criteria,
                        "code_grader": code_grader_spec,
                        "retrieved_contexts": result["contexts"],
                        "metrics": metrics,
                    }
                )
            except anthropic.APIError as error:
                errors += 1
                print_api_error(error)
            except Exception as error:
                errors += 1
                print(f"[eval] failed id={row_id} error={error}")

        count = len(prediction_rows)
        avg_score = sum(float(item["metrics"]["overall_score"]) for item in prediction_rows) / count if count else 0.0
        avg_f1 = sum(float(item["metrics"]["token_f1"]) for item in prediction_rows) / count if count else 0.0
        avg_em = sum(float(item["metrics"]["exact_match"]) for item in prediction_rows) / count if count else 0.0
        avg_retrieved = (
            sum(float(item["metrics"].get("retrieved_chunks", 0.0)) for item in prediction_rows) / count if count else 0.0
        )

        resolved_run_name = run_name or f"rag-{utc_timestamp()}"
        run_dir = output_dir / resolved_run_name
        predictions_path = run_dir / "predictions.jsonl"
        report_path = run_dir / "report.json"

        report = {
            "run_name": resolved_run_name,
            "dataset": str(dataset_path),
            "mode": self.active_mode,
            "model": self.active_model,
            "grader": self.active_grader_method,
            "grader_model": self.active_grader_model,
            "rows_evaluated": count,
            "errors": errors,
            "metrics": {
                "avg_overall_score": avg_score,
                "avg_token_f1": avg_f1,
                "exact_match_rate": avg_em,
                "avg_retrieved_chunks": avg_retrieved,
            },
            "vector_db_path": str(self.vector_db.persist_directory),
            "vector_db_collection": self.vector_db.collection_name,
            "embedding_model": self.config.embedding_model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        write_jsonl(predictions_path, prediction_rows)
        write_json(report_path, report)
        print(f"[eval] wrote predictions: {predictions_path}")
        print(f"[eval] wrote report: {report_path}")
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG pipeline: upload docs -> chunk -> embed -> vector DB -> retrieve -> answer/eval")
    parser.add_argument(
        "--document",
        dest="documents",
        action="append",
        default=[],
        help="Path to a document to ingest into the vector DB. Repeat for multiple documents.",
    )
    parser.add_argument("--question", default=None, help="Question to ask with RAG retrieval")
    parser.add_argument("--chunk-size", type=int, default=220, help="Chunk size in words")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Chunk overlap in words")
    parser.add_argument("--top-k", type=int, default=4, help="Top K chunks to retrieve")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embeddings",
    )
    parser.add_argument("--vector-db-path", default="data/vector_db/chroma", help="Persistent Chroma directory")
    parser.add_argument("--collection-name", default="rag_chunks", help="Chroma collection name")
    parser.add_argument("--mode", choices=["auto", "api", "demo"], default="auto", help="Execution mode")
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL), help="Model name")
    parser.add_argument("--fallback-model", default=os.getenv("ANTHROPIC_FALLBACK_MODEL"), help="Fallback model")
    parser.add_argument("--max-tokens", type=int, default=350, help="Max tokens for answer generation")
    parser.add_argument(
        "--grader",
        choices=["lexical", "model", "code", "coding"],
        default=os.getenv("EVAL_GRADER", "lexical"),
        help="Grading strategy: lexical (default), model, code, or coding",
    )
    parser.add_argument(
        "--grader-model",
        default=os.getenv("ANTHROPIC_GRADER_MODEL"),
        help="Model used for grading when --grader model (defaults to --model)",
    )
    parser.add_argument(
        "--grader-fallback-model",
        default=os.getenv("ANTHROPIC_GRADER_FALLBACK_MODEL"),
        help="Fallback model for grader model resolution",
    )
    parser.add_argument(
        "--grader-max-tokens",
        type=int,
        default=300,
        help="Max tokens for grader model responses",
    )
    parser.add_argument(
        "--code-timeout-seconds",
        type=float,
        default=3.0,
        help="Timeout in seconds for executing generated code per row when --grader code",
    )
    parser.add_argument("--eval-dataset", default=None, help="Optional JSONL dataset for RAG evaluation")
    parser.add_argument("--output-dir", default="data/eval_runs", help="Output directory for eval artifacts")
    parser.add_argument("--run-name", default=None, help="Optional eval run name")
    parser.add_argument("--rows", type=int, default=None, help="Optional max rows for eval")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    vector_db_path = normalize_path(args.vector_db_path)
    vector_db = ChromaVectorDB(vector_db_path, collection_name=args.collection_name)
    vector_db.load()
    log_step(
        f"loaded vector DB entries={vector_db.count()} "
        f"from {vector_db_path} collection={args.collection_name}"
    )

    config = RagConfig(
        mode=args.mode,
        model=args.model,
        fallback_model=args.fallback_model,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        grader=args.grader,
        grader_model=args.grader_model,
        grader_fallback_model=args.grader_fallback_model,
        grader_max_tokens=args.grader_max_tokens,
        code_timeout_seconds=args.code_timeout_seconds,
    )

    pipeline = RagPipeline(
        vector_db=vector_db,
        embedder=SentenceTransformerEmbedder(model_name=args.embedding_model),
        config=config,
    )
    pipeline.setup()

    ingested_total = 0
    for raw_path in args.documents:
        document_path = normalize_path(raw_path)
        ingested_total += pipeline.ingest_document(
            document_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            show_chunks=True,
        )

    if args.documents:
        log_step(f"completed upload+index for {len(args.documents)} document(s), chunks={ingested_total}")

    if args.question:
        print("\n[qa] running retrieval-augmented answer")
        try:
            result = pipeline.answer_question(args.question)
            print("\n[qa] answer:")
            print(result["answer"])
        except anthropic.APIError as error:
            print_api_error(error)
            raise SystemExit(1) from error

    if args.eval_dataset:
        dataset_path = normalize_path(args.eval_dataset)
        output_dir = normalize_path(args.output_dir)
        try:
            report = pipeline.run_eval(
                dataset_path=dataset_path,
                output_dir=output_dir,
                run_name=args.run_name,
                max_rows=args.rows,
            )
            print("\n[eval] summary:")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        except anthropic.APIError as error:
            print_api_error(error)
            raise SystemExit(1) from error

    if not args.question and not args.eval_dataset:
        print("\nNothing to answer/evaluate yet. Use --question and/or --eval-dataset.")


if __name__ == "__main__":
    main()
