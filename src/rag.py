from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import chromadb
from dotenv import load_dotenv
import pandas as pd
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

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        workbook = pd.ExcelFile(path)
        sheet_blocks: list[str] = []
        for sheet_name in workbook.sheet_names:
            dataframe = pd.read_excel(workbook, sheet_name=sheet_name, dtype=str).fillna("")
            sheet_blocks.append(f"# Sheet: {sheet_name}")
            columns = [str(column).strip() for column in dataframe.columns]
            if columns:
                sheet_blocks.append("Columns: " + ", ".join(columns))

            for index, row in dataframe.iterrows():
                values: list[str] = []
                for column in dataframe.columns:
                    key = str(column).strip()
                    value = str(row[column]).strip()
                    if key and value:
                        values.append(f"{key}: {value}")

                if values:
                    sheet_blocks.append(f"- Row {index + 2}: " + " | ".join(values))

        return "\n".join(sheet_blocks).strip()

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

    def get_all_chunks(self) -> list[dict[str, Any]]:
        if self.collection is None:
            return []

        results = self.collection.get(include=["documents", "metadatas"])
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        output: list[dict[str, Any]] = []
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            output.append(
                {
                    "chunk_id": str(metadata_dict.get("chunk_id", chunk_id or "")),
                    "source": str(metadata_dict.get("source", "")),
                    "text": str(document or ""),
                }
            )
        return output


class BM25KeywordIndex:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.records: list[dict[str, Any]] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_token_counts: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.avg_doc_len = 0.0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", str(text).lower())

    def build(self, records: list[dict[str, Any]]) -> None:
        self.records = list(records)
        self.doc_tokens = [self.tokenize(str(record.get("text", ""))) for record in self.records]
        self.doc_token_counts = [dict(collections.Counter(tokens)) for tokens in self.doc_tokens]

        lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = (sum(lengths) / len(lengths)) if lengths else 0.0

        doc_freq: collections.Counter[str] = collections.Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))

        total_docs = len(self.records)
        self.idf = {}
        for token, freq in doc_freq.items():
            self.idf[token] = math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not self.records:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, int]] = []
        for index, token_counts in enumerate(self.doc_token_counts):
            doc_len = max(1, len(self.doc_tokens[index]))
            score = 0.0
            for token in query_tokens:
                tf = float(token_counts.get(token, 0))
                if tf <= 0:
                    continue
                idf = self.idf.get(token, 0.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-9, self.avg_doc_len or 1.0)))
                score += idf * ((tf * (self.k1 + 1.0)) / max(1e-9, denominator))
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        output: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(scored[:top_k], start=1):
            record = self.records[index]
            output.append(
                {
                    "chunk_id": str(record.get("chunk_id", "")),
                    "source": str(record.get("source", "")),
                    "text": str(record.get("text", "")),
                    "bm25_score": float(score),
                    "bm25_rank": rank,
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
    retrieval_mode: str
    dense_k: int
    sparse_k: int
    rrf_k: int
    rerank_enabled: bool
    rerank_candidates: int
    final_context_k: int


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
        self.keyword_index = BM25KeywordIndex()
        self.keyword_index_dirty = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return BM25KeywordIndex.tokenize(text)

    @staticmethod
    def _token_overlap_ratio(a_text: str, b_text: str) -> float:
        a_tokens = set(BM25KeywordIndex.tokenize(a_text))
        b_tokens = set(BM25KeywordIndex.tokenize(b_text))
        if not a_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens)

    @staticmethod
    def _extract_citation_ids(answer: str) -> list[str]:
        matches = re.findall(r"\[([^\]]+)\]", str(answer))
        ids: list[str] = []
        for raw in matches:
            candidate = str(raw).strip()
            if not candidate:
                continue
            parts = [part.strip() for part in candidate.split(",") if part.strip()]
            ids.extend(parts)
        return ids

    def _ensure_keyword_index(self) -> None:
        if not self.keyword_index_dirty:
            return
        records = self.vector_db.get_all_chunks()
        self.keyword_index.build(records)
        self.keyword_index_dirty = False

    @staticmethod
    def _normalize_scores(values: list[float]) -> list[float]:
        if not values:
            return []
        min_value = min(values)
        max_value = max(values)
        if max_value <= min_value:
            return [1.0 if max_value > 0 else 0.0 for _ in values]
        return [(value - min_value) / (max_value - min_value) for value in values]

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
        if self.config.retrieval_mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("--retrieval-mode must be one of dense, sparse, hybrid")
        if self.config.dense_k <= 0:
            raise ValueError("--dense-k must be greater than 0")
        if self.config.sparse_k <= 0:
            raise ValueError("--sparse-k must be greater than 0")
        if self.config.rrf_k <= 0:
            raise ValueError("--rrf-k must be greater than 0")
        if self.config.rerank_candidates <= 0:
            raise ValueError("--rerank-candidates must be greater than 0")
        if self.config.final_context_k <= 0:
            raise ValueError("--final-context-k must be greater than 0")

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
        self.keyword_index_dirty = True
        print(
            f"[vector-db] upserted into {self.vector_db.persist_directory} "
            f"collection={self.vector_db.collection_name} total_entries={self.vector_db.count()}"
        )
        return len(rag_chunks)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        include_debug: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        final_k = top_k or self.config.final_context_k or self.config.top_k
        retrieval_mode = self.config.retrieval_mode

        dense_hits: list[dict[str, Any]] = []
        sparse_hits: list[dict[str, Any]] = []
        query_vector: list[float] = []

        if retrieval_mode in {"dense", "hybrid"}:
            query_vector = self.embedder.embed_query(question)
            dense_hits = self.vector_db.search(query_vector, top_k=max(self.config.dense_k, self.config.rerank_candidates))

        if retrieval_mode in {"sparse", "hybrid"}:
            self._ensure_keyword_index()
            sparse_hits = self.keyword_index.search(question, top_k=max(self.config.sparse_k, self.config.rerank_candidates))

        merged_by_chunk: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_hits, start=1):
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id:
                continue
            entry = merged_by_chunk.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "source": str(item.get("source", "")),
                    "text": str(item.get("text", "")),
                },
            )
            entry["dense_rank"] = rank
            entry["dense_score"] = float(item.get("score", 0.0))

        for rank, item in enumerate(sparse_hits, start=1):
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id:
                continue
            entry = merged_by_chunk.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "source": str(item.get("source", "")),
                    "text": str(item.get("text", "")),
                },
            )
            entry["bm25_rank"] = rank
            entry["bm25_score"] = float(item.get("bm25_score", 0.0))

        merged_candidates = list(merged_by_chunk.values())
        for candidate in merged_candidates:
            rrf_score = 0.0
            dense_rank = candidate.get("dense_rank")
            bm25_rank = candidate.get("bm25_rank")
            if isinstance(dense_rank, int):
                rrf_score += 1.0 / (self.config.rrf_k + dense_rank)
            if isinstance(bm25_rank, int):
                rrf_score += 1.0 / (self.config.rrf_k + bm25_rank)
            candidate["rrf_score"] = rrf_score

        if retrieval_mode == "dense":
            merged_candidates.sort(key=lambda item: float(item.get("dense_score", 0.0)), reverse=True)
        elif retrieval_mode == "sparse":
            merged_candidates.sort(key=lambda item: float(item.get("bm25_score", 0.0)), reverse=True)
        else:
            merged_candidates.sort(key=lambda item: float(item.get("rrf_score", 0.0)), reverse=True)

        rerank_pool = merged_candidates[: self.config.rerank_candidates]
        bm25_norm = self._normalize_scores([float(item.get("bm25_score", 0.0)) for item in rerank_pool])
        dense_norm = self._normalize_scores([float(item.get("dense_score", 0.0)) for item in rerank_pool])
        rrf_norm = self._normalize_scores([float(item.get("rrf_score", 0.0)) for item in rerank_pool])

        for index, candidate in enumerate(rerank_pool):
            overlap = self._token_overlap_ratio(question, str(candidate.get("text", "")))
            base = rrf_norm[index] if retrieval_mode == "hybrid" else dense_norm[index] if retrieval_mode == "dense" else bm25_norm[index]
            candidate["rerank_score"] = (0.5 * base) + (0.25 * dense_norm[index]) + (0.15 * bm25_norm[index]) + (0.10 * overlap)

        if self.config.rerank_enabled:
            rerank_pool.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)

        selected = rerank_pool[:final_k]
        results: list[dict[str, Any]] = []
        for candidate in selected:
            results.append(
                {
                    "score": float(candidate.get("rerank_score", candidate.get("rrf_score", candidate.get("dense_score", candidate.get("bm25_score", 0.0))))),
                    "chunk_id": str(candidate.get("chunk_id", "")),
                    "source": str(candidate.get("source", "")),
                    "text": str(candidate.get("text", "")),
                    "dense_rank": candidate.get("dense_rank"),
                    "bm25_rank": candidate.get("bm25_rank"),
                    "rrf_score": float(candidate.get("rrf_score", 0.0)),
                    "rerank_score": float(candidate.get("rerank_score", 0.0)),
                }
            )

        print(
            f"[retrieval] mode={retrieval_mode} dense_hits={len(dense_hits)} "
            f"sparse_hits={len(sparse_hits)} rerank_pool={len(rerank_pool)} final={len(results)}"
        )
        for result in results:
            preview = str(result.get("text", ""))[:120].replace("\n", " ").strip()
            print(
                f"  - chunk={result.get('chunk_id','')} score={float(result.get('score', 0.0)):.4f} "
                f"source={result.get('source','')} text={preview}{'...' if len(str(result.get('text',''))) > 120 else ''}"
            )

        if not include_debug:
            return results

        return {
            "results": results,
            "debug": {
                "retrieval_mode": retrieval_mode,
                "dense_hits": len(dense_hits),
                "sparse_hits": len(sparse_hits),
                "rerank_candidates": len(rerank_pool),
                "final_context_k": len(results),
            },
        }

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
            "Return a concise answer and cite exact chunk ids like [doc-3], [doc-9] when used."
        )

    def answer_question(self, question: str) -> dict[str, Any]:
        retrieval_output = self.retrieve(question, include_debug=True)
        contexts = retrieval_output["results"]
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
            "retrieval_debug": retrieval_output.get("debug", {}),
            "answer": answer,
        }

    def standards_guidance(self, implementation_target: str) -> str:
        normalized_target = implementation_target.strip().lower()
        if normalized_target == "backend":
            return (
                "Implement backend specifications using .NET coding standards:\n"
                "- Use clear layered architecture (API, application, domain, infrastructure).\n"
                "- Follow SOLID principles and dependency injection patterns.\n"
                "- Use PascalCase for public types/methods and camelCase for local variables/parameters.\n"
                "- Use async/await for I/O-bound operations and cancellation tokens where relevant.\n"
                "- Include validation, structured logging, exception handling, and testability guidance."
            )
        if normalized_target == "frontend":
            return (
                "Implement frontend specifications using React standards:\n"
                "- Prefer functional components and hooks, with clear component boundaries.\n"
                "- Use predictable state management and unidirectional data flow.\n"
                "- Enforce accessibility (semantic HTML, aria where needed, keyboard navigation).\n"
                "- Organize reusable UI components, API layer separation, and error/loading states.\n"
                "- Include testing guidance for components and user flows."
            )
        return (
            "Implementation target is fullstack. Include both .NET backend standards and React frontend standards "
            "in relevant sections."
        )

    def build_spec_prompt(
        self,
        retrieved_contexts: list[dict[str, Any]],
        spec_query: str,
        implementation_target: str,
    ) -> str:
        context_lines: list[str] = []
        for index, context in enumerate(retrieved_contexts, start=1):
            context_lines.append(
                f"[{index}] source={context.get('source', '')} chunk={context.get('chunk_id', '')}\n"
                f"{context.get('text', '')}"
            )

        context_block = "\n\n".join(context_lines) if context_lines else "No retrieved context was found."
        standards_section = self.standards_guidance(implementation_target)
        return (
            "You are a senior product and solutions architect.\n"
            "Use only the retrieved context to draft a specification document in Markdown.\n"
            "If information is missing, explicitly call it out in Open Questions.\n\n"
            "Required sections:\n"
            "1. Title\n"
            "2. Objective\n"
            "3. Scope (in / out)\n"
            "4. Assumptions\n"
            "5. Functional Requirements\n"
            "6. Data Dictionary (infer fields from spreadsheet context when available)\n"
            "7. Business Rules\n"
            "8. Non-Functional Requirements\n"
            "9. API / Integration Notes\n"
            "10. Validation & Acceptance Criteria\n"
            "11. Open Questions / Risks\n\n"
            f"Implementation target: {implementation_target}\n"
            f"Standards requirements:\n{standards_section}\n\n"
            f"Spec objective:\n{spec_query}\n\n"
            f"Retrieved context:\n{context_block}\n"
        )

    def build_demo_spec_doc(
        self,
        retrieved_contexts: list[dict[str, Any]],
        spec_query: str,
        implementation_target: str,
    ) -> str:
        context_preview_lines: list[str] = []
        for context in retrieved_contexts[:8]:
            preview = str(context.get("text", "")).strip()
            if preview:
                context_preview_lines.append(f"- {preview[:200]}")

        context_preview = "\n".join(context_preview_lines) if context_preview_lines else "- No context available"
        standards_section = self.standards_guidance(implementation_target)
        return (
            "# Specification Document (Draft)\n\n"
            f"## Objective\n{spec_query}\n\n"
            f"## Implementation Target\n{implementation_target}\n\n"
            f"## Standards Requirements\n{standards_section}\n\n"
            "## Scope\n"
            "- In scope: ingest documents, chunk content, embed chunks, retrieve context, generate grounded outputs.\n"
            "- Out of scope: unsupported source parsing beyond configured file readers.\n\n"
            "## Functional Requirements\n"
            "- System shall parse uploaded documents into normalized text blocks.\n"
            "- System shall create overlapping text chunks for retrieval.\n"
            "- System shall store chunk embeddings in vector storage.\n"
            "- System shall generate responses and artifacts grounded in retrieved chunks.\n\n"
            "## Data Dictionary\n"
            "- Derive entities and fields from uploaded source columns/content.\n\n"
            "## Validation & Acceptance Criteria\n"
            "- Chunk previews are printed to console after upload.\n"
            "- Retrieval returns relevant chunks for the spec query.\n"
            "- Generated spec doc is written to the target output path.\n\n"
            "## Open Questions / Risks\n"
            "- Confirm required output format and level of detail for downstream consumers.\n"
            "- Validate whether additional metadata fields are needed for traceability.\n\n"
            "## Retrieved Context Preview\n"
            f"{context_preview}\n"
        )

    def generate_spec_doc(self, spec_query: str, max_spec_chunks: int, implementation_target: str) -> dict[str, Any]:
        retrieved_contexts = self.retrieve(spec_query, top_k=max_spec_chunks)
        response_stop_reason: str | None = None
        response_block_types: list[str] = []
        response_block_text_lengths: list[int] = []

        if self.active_mode == "demo":
            spec_markdown = self.build_demo_spec_doc(retrieved_contexts, spec_query, implementation_target)
        else:
            if self.client is None:
                raise RuntimeError("Anthropic client not initialized.")
            prompt = self.build_spec_prompt(retrieved_contexts, spec_query, implementation_target)
            response = chat(
                client=self.client,
                messages=[{"role": "user", "content": prompt}],
                model=self.active_model,
                max_tokens=self.config.max_tokens,
                stream=False,
            )
            response_stop_reason = str(getattr(response, "stop_reason", "")) or None
            content_blocks = getattr(response, "content", []) or []
            response_block_types = [str(getattr(block, "type", type(block).__name__)) for block in content_blocks]
            response_block_text_lengths = [len(str(getattr(block, "text", "") or "")) for block in content_blocks]
            spec_markdown = get_text_from_response(response).strip()

        if not str(spec_markdown).strip():
            raise ValueError(
                "Generated specification markdown is empty. "
                "Try increasing --max-tokens, refining --spec-query, or switching models. "
                f"stop_reason={response_stop_reason}, "
                f"block_types={response_block_types}, "
                f"block_text_lengths={response_block_text_lengths}"
            )

        return {
            "spec_query": spec_query,
            "implementation_target": implementation_target,
            "retrieved_contexts": retrieved_contexts,
            "spec_markdown": spec_markdown,
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
                retrieved_contexts = list(result.get("contexts", []))
                retrieved_chunk_ids = [str(item.get("chunk_id", "")) for item in retrieved_contexts if str(item.get("chunk_id", ""))]
                gold_chunk_ids = [str(item).strip() for item in row.get("gold_chunk_ids", []) if str(item).strip()]
                required_citations = [str(item).strip() for item in row.get("required_citations", []) if str(item).strip()]
                cited_chunk_ids = self._extract_citation_ids(prediction)
                valid_citations = [chunk_id for chunk_id in cited_chunk_ids if chunk_id in set(retrieved_chunk_ids)]

                context_precision: float | None = None
                context_recall: float | None = None
                if gold_chunk_ids:
                    overlap = len(set(retrieved_chunk_ids) & set(gold_chunk_ids))
                    context_precision = overlap / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0.0
                    context_recall = overlap / len(gold_chunk_ids)

                context_text = "\n".join(str(item.get("text", "")) for item in retrieved_contexts)
                faithfulness = self._token_overlap_ratio(prediction, context_text)
                answer_relevance = token_f1(question, prediction)

                if cited_chunk_ids:
                    citation_correctness = len(valid_citations) / len(cited_chunk_ids)
                elif required_citations:
                    citation_correctness = 0.0
                else:
                    citation_correctness = 1.0

                metrics = {
                    "exact_match": exact_match(reference, prediction),
                    "token_f1": token_f1(reference, prediction),
                    "criteria_score": criteria_score(eval_criteria, prediction),
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "faithfulness": faithfulness,
                    "answer_relevance": answer_relevance,
                    "citation_correctness": citation_correctness,
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
                metrics["retrieval_metrics"] = {
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                }
                metrics["generation_metrics"] = {
                    "faithfulness": faithfulness,
                    "answer_relevance": answer_relevance,
                    "citation_correctness": citation_correctness,
                }

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
                        "retrieval_debug": result.get("retrieval_debug", {}),
                        "retrieved_chunk_ids": retrieved_chunk_ids,
                        "gold_chunk_ids": gold_chunk_ids,
                        "required_citations": required_citations,
                        "cited_chunk_ids": cited_chunk_ids,
                        "valid_cited_chunk_ids": valid_citations,
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
        avg_faithfulness = (
            sum(float(item["metrics"].get("faithfulness", 0.0)) for item in prediction_rows) / count if count else 0.0
        )
        avg_answer_relevance = (
            sum(float(item["metrics"].get("answer_relevance", 0.0)) for item in prediction_rows) / count if count else 0.0
        )
        avg_citation_correctness = (
            sum(float(item["metrics"].get("citation_correctness", 0.0)) for item in prediction_rows) / count if count else 0.0
        )
        precision_values = [
            float(item["metrics"]["context_precision"])
            for item in prediction_rows
            if item["metrics"].get("context_precision") is not None
        ]
        recall_values = [
            float(item["metrics"]["context_recall"])
            for item in prediction_rows
            if item["metrics"].get("context_recall") is not None
        ]
        avg_context_precision = (sum(precision_values) / len(precision_values)) if precision_values else None
        avg_context_recall = (sum(recall_values) / len(recall_values)) if recall_values else None

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
                "avg_context_precision": avg_context_precision,
                "avg_context_recall": avg_context_recall,
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_relevance": avg_answer_relevance,
                "avg_citation_correctness": avg_citation_correctness,
            },
            "vector_db_path": str(self.vector_db.persist_directory),
            "vector_db_collection": self.vector_db.collection_name,
            "embedding_model": self.config.embedding_model,
            "retrieval_mode": self.config.retrieval_mode,
            "dense_k": self.config.dense_k,
            "sparse_k": self.config.sparse_k,
            "rrf_k": self.config.rrf_k,
            "rerank_enabled": self.config.rerank_enabled,
            "rerank_candidates": self.config.rerank_candidates,
            "final_context_k": self.config.final_context_k,
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
        "--retrieval-mode",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval strategy: dense vectors, sparse BM25, or hybrid fusion",
    )
    parser.add_argument("--dense-k", type=int, default=20, help="Candidate count for dense retrieval")
    parser.add_argument("--sparse-k", type=int, default=20, help="Candidate count for BM25 retrieval")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant for hybrid fusion")
    parser.add_argument(
        "--rerank-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reranking of retrieved candidates before final context selection",
    )
    parser.add_argument("--rerank-candidates", type=int, default=20, help="Number of fused candidates to rerank")
    parser.add_argument("--final-context-k", type=int, default=6, help="Final number of chunks sent to generation")
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
    parser.add_argument(
        "--generate-spec-doc",
        action="store_true",
        help="Generate a Markdown specification document from retrieved context.",
    )
    parser.add_argument(
        "--spec-doc-path",
        default="data/spec_docs/generated_spec.md",
        help="Output path for generated spec markdown.",
    )
    parser.add_argument(
        "--spec-query",
        default="Generate a complete product and technical specification from the uploaded source documents.",
        help="Query objective used to retrieve context for spec generation.",
    )
    parser.add_argument(
        "--implementation-target",
        choices=["backend", "frontend", "fullstack"],
        default="fullstack",
        help="Standards target for spec output (backend=.NET, frontend=React, fullstack=both).",
    )
    parser.add_argument(
        "--max-spec-chunks",
        type=int,
        default=20,
        help="Maximum retrieved chunks used as context for spec generation.",
    )
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
        retrieval_mode=args.retrieval_mode,
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
        rrf_k=args.rrf_k,
        rerank_enabled=args.rerank_enabled,
        rerank_candidates=args.rerank_candidates,
        final_context_k=args.final_context_k,
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

    if args.generate_spec_doc:
        if args.max_spec_chunks <= 0:
            raise ValueError("--max-spec-chunks must be greater than 0")
        print("\n[spec] generating specification document")
        try:
            spec_result = pipeline.generate_spec_doc(
                spec_query=args.spec_query,
                max_spec_chunks=args.max_spec_chunks,
                implementation_target=args.implementation_target,
            )
            spec_markdown = str(spec_result["spec_markdown"]).strip()
            if not spec_markdown:
                raise ValueError("Generated spec markdown is empty; file write aborted.")
            spec_doc_path = normalize_path(args.spec_doc_path)
            spec_doc_path.parent.mkdir(parents=True, exist_ok=True)
            spec_doc_path.write_text(spec_markdown, encoding="utf-8")
            print(f"[spec] retrieved contexts: {len(spec_result['retrieved_contexts'])}")
            print(f"[spec] wrote markdown: {spec_doc_path}")
        except anthropic.APIError as error:
            print_api_error(error)
            raise SystemExit(1) from error

    if not args.question and not args.eval_dataset and not args.generate_spec_doc:
        print("\nNothing to answer/evaluate yet. Use --question, --eval-dataset, and/or --generate-spec-doc.")


if __name__ == "__main__":
    main()
