from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from src.client import chat, get_text_from_response, print_api_error, resolve_model
from src.dataset import generate_dataset, write_jsonl as write_generated_jsonl
from src.demo import build_demo_answer
from src.eval import (
    aggregate_metrics,
    build_eval_prompt,
    code_grader_result,
    criteria_score,
    exact_match,
    load_jsonl_dataset,
    model_grader_result,
    normalize_grader_method,
    overall_score,
    render_html_report,
    resolve_extra_prompt,
    score_grade,
    token_f1,
    utc_timestamp,
    write_html,
    write_json,
    write_jsonl,
)


def log_step(message: str) -> None:
    print(f"[step] {message}")


@dataclass
class PromptDraft:
    template: str | None = None
    extra_context: str | None = None

    def draft(self, question: str) -> str:
        clean_question = str(question).strip()
        if not clean_question:
            raise ValueError("Question/prompt cannot be empty.")

        if not self.template:
            return build_eval_prompt(clean_question, extra_prompt=self.extra_context)

        template = self.template.strip()
        if not template:
            return build_eval_prompt(clean_question, extra_prompt=self.extra_context)

        if "{" in template and "}" in template:
            try:
                drafted = template.format(
                    question=clean_question,
                    prompt=clean_question,
                    extra_context=self.extra_context or "",
                )
            except KeyError as error:
                missing_key = str(error).strip("'")
                raise ValueError(
                    f"Prompt template contains unsupported placeholder '{missing_key}'. "
                    "Use {question}, {prompt}, or {extra_context}."
                ) from error
        else:
            drafted = f"{template}\n\nQuestion: {clean_question}"

        if self.extra_context and "{extra_context}" not in template:
            drafted = f"{drafted}\n\nAdditional context:\n{self.extra_context}"

        return drafted.strip()


@dataclass
class EvalDataset:
    dataset_paths: list[Path]
    rows_when_generating: int = 25
    seed: int = 42
    regenerate: bool = False

    def ensure_available(self) -> None:
        for dataset_path in self.dataset_paths:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            if dataset_path.exists() and not self.regenerate:
                print(f"Using existing dataset: {dataset_path}")
                continue

            log_step(f"Generating dataset at {dataset_path} rows={self.rows_when_generating} seed={self.seed}")
            generated = generate_dataset(rows=self.rows_when_generating, seed=self.seed)
            write_generated_jsonl(generated, dataset_path)
            print(f"Generated dataset: {dataset_path} (rows={len(generated)}, seed={self.seed})")

    def load_rows(self, eval_rows: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for dataset_path in self.dataset_paths:
            log_step(f"Loading dataset from {dataset_path}")
            dataset_rows = load_jsonl_dataset(dataset_path)
            for row in dataset_rows:
                row["dataset"] = str(dataset_path)
            rows.extend(dataset_rows)
            log_step(f"Loaded {len(dataset_rows)} rows from {dataset_path}")

        if eval_rows is not None:
            if eval_rows <= 0:
                raise ValueError("--eval-rows must be greater than 0 when provided")
            rows = rows[:eval_rows]
            log_step(f"Applied eval row limit: {len(rows)}")

        return rows


@dataclass
class ClaudeRunner:
    mode: str
    model: str
    fallback_model: str | None
    max_tokens: int

    client: anthropic.Anthropic | None = None
    active_mode: str = "demo"
    active_model: str = "demo-local"

    def setup(self) -> None:
        requested_mode = self.mode
        api_key = os.getenv("ANTHROPIC_API_KEY")

        self.active_mode = requested_mode
        if requested_mode == "auto":
            self.active_mode = "api" if api_key else "demo"

        log_step(f"Execution mode resolved: requested={requested_mode}, active={self.active_mode}")

        if self.active_mode == "api":
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for API mode")

            log_step("Initializing Anthropic client")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.active_model, note = resolve_model(self.client, self.model, self.fallback_model)
            if note:
                print(note)
        else:
            self.active_model = "demo-local"

    def generate(self, drafted_prompt: str) -> str:
        if self.active_mode == "demo":
            return build_demo_answer(drafted_prompt)

        if self.client is None:
            raise RuntimeError("Anthropic client is required in API mode.")

        response = chat(
            client=self.client,
            messages=[{"role": "user", "content": drafted_prompt}],
            model=self.active_model,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return get_text_from_response(response).strip()


@dataclass
class EvalGrader:
    method: str
    grader_model: str | None
    grader_fallback_model: str | None
    grader_max_tokens: int
    code_timeout_seconds: float

    client: anthropic.Anthropic | None = None
    active_method: str = "lexical"
    active_grader_model: str | None = None

    def setup(self, claude_runner: ClaudeRunner) -> None:
        self.active_method = normalize_grader_method(self.method)
        self.client = claude_runner.client

        if self.active_method == "model":
            if claude_runner.active_mode != "api" or self.client is None:
                raise RuntimeError("--grader model requires API mode.")

            requested_model = self.grader_model or claude_runner.active_model
            resolved_model, note = resolve_model(self.client, requested_model, self.grader_fallback_model)
            self.active_grader_model = resolved_model
            if note:
                print(f"Grader: {note}")

        if self.active_method == "code" and self.code_timeout_seconds <= 0:
            raise ValueError("--code-timeout-seconds must be greater than 0")

    def grade(self, row: dict[str, Any], prediction: str) -> dict[str, Any]:
        reference = str(row.get("answer", ""))
        eval_criteria = [str(item) for item in row.get("eval_criteria", [])]
        code_grader_spec = row.get("code_grader")

        metrics: dict[str, Any] = {
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

        if self.active_method == "model":
            if self.client is None or not self.active_grader_model:
                raise RuntimeError("Model grader is not configured.")

            grader_output = model_grader_result(
                client=self.client,
                grader_model=self.active_grader_model,
                max_tokens=self.grader_max_tokens,
                prompt=str(row.get("prompt", "")),
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
            return metrics

        if self.active_method == "code":
            if code_grader_spec:
                timeout_seconds = float(code_grader_spec.get("timeout_seconds", self.code_timeout_seconds))
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
            return metrics

        metrics["grading_details"] = {"method": "lexical"}
        return metrics


@dataclass
class WorkflowRunner:
    dataset: EvalDataset
    prompt_draft: PromptDraft
    claude: ClaudeRunner
    grader: EvalGrader
    output_dir: Path
    run_name: str | None = None
    eval_rows: int | None = None
    fail_fast: bool = False

    def run(self) -> int:
        self.dataset.ensure_available()
        rows = self.dataset.load_rows(eval_rows=self.eval_rows)
        log_step(f"Loaded {len(rows)} total rows across {len(self.dataset.dataset_paths)} dataset(s)")

        self.claude.setup()
        self.grader.setup(self.claude)

        run_name = self.run_name or utc_timestamp()
        run_dir = self.output_dir / run_name
        predictions_path = run_dir / "predictions.jsonl"
        report_path = run_dir / "report.json"
        html_report_path = run_dir / "report.html"
        log_step(f"Run artifacts directory: {run_dir}")

        started_at = datetime.now(timezone.utc).isoformat()
        prediction_rows: list[dict[str, Any]] = []
        error_count = 0

        print(
            f"Starting evaluation: rows={len(rows)} mode={self.claude.active_mode} "
            f"grader={self.grader.active_method} run={run_name}"
        )

        for index, row in enumerate(rows, start=1):
            row_id = str(row.get("id", index))
            print(f"[row {index}/{len(rows)}] start id={row_id}")

            try:
                drafted_prompt = self.prompt_draft.draft(str(row.get("prompt", "")))
                prediction = self.claude.generate(drafted_prompt)
                metrics = self.grader.grade(row, prediction)

                prediction_rows.append(
                    {
                        "id": row_id,
                        "dataset": row.get("dataset", ""),
                        "topic": row.get("topic", ""),
                        "prompt": row.get("prompt", ""),
                        "drafted_prompt": drafted_prompt,
                        "reference": row.get("answer", ""),
                        "prediction": prediction,
                        "eval_criteria": row.get("eval_criteria", []),
                        "code_grader": row.get("code_grader"),
                        "metrics": metrics,
                    }
                )
                print(f"[row {index}/{len(rows)}] completed")
            except anthropic.APIError as error:
                error_count += 1
                print_api_error(error)
                prediction_rows.append(
                    {
                        "id": row_id,
                        "dataset": row.get("dataset", ""),
                        "topic": row.get("topic", ""),
                        "prompt": row.get("prompt", ""),
                        "reference": row.get("answer", ""),
                        "prediction": "",
                        "eval_criteria": row.get("eval_criteria", []),
                        "code_grader": row.get("code_grader"),
                        "metrics": {
                            "exact_match": 0.0,
                            "token_f1": 0.0,
                            "criteria_score": 0.0,
                            "overall_score": 0.0,
                            "grade": "F",
                            "grading_details": {"method": self.grader.active_method},
                        },
                        "error": str(error),
                    }
                )
                if self.fail_fast:
                    raise
            except Exception as error:
                error_count += 1
                print(f"[row {index}/{len(rows)}] failed with error: {error}")
                prediction_rows.append(
                    {
                        "id": row_id,
                        "dataset": row.get("dataset", ""),
                        "topic": row.get("topic", ""),
                        "prompt": row.get("prompt", ""),
                        "reference": row.get("answer", ""),
                        "prediction": "",
                        "eval_criteria": row.get("eval_criteria", []),
                        "code_grader": row.get("code_grader"),
                        "metrics": {
                            "exact_match": 0.0,
                            "token_f1": 0.0,
                            "criteria_score": 0.0,
                            "overall_score": 0.0,
                            "grade": "F",
                            "grading_details": {"method": self.grader.active_method},
                        },
                        "error": str(error),
                    }
                )
                if self.fail_fast:
                    raise

            if index % 10 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)}")

        aggregate = aggregate_metrics(prediction_rows)
        finished_at = datetime.now(timezone.utc).isoformat()

        report = {
            "run_name": run_name,
            "datasets": [str(path) for path in self.dataset.dataset_paths],
            "rows_evaluated": len(prediction_rows),
            "errors": error_count,
            "mode": self.claude.active_mode,
            "requested_mode": self.claude.mode,
            "model": self.claude.active_model,
            "grader": {
                "method": self.grader.active_method,
                "model": self.grader.active_grader_model if self.grader.active_method == "model" else None,
            },
            "started_at": started_at,
            "finished_at": finished_at,
            "overall_score": aggregate["avg_overall_score"],
            "overall_grade": score_grade(aggregate["avg_overall_score"]),
            "metrics": aggregate,
            "prompting": {
                "template": self.prompt_draft.template,
                "extra_context_enabled": bool(self.prompt_draft.extra_context),
            },
            "artifacts": {
                "predictions": str(predictions_path),
                "report": str(report_path),
                "html_report": str(html_report_path),
            },
        }

        log_step(f"Writing predictions JSONL to {predictions_path}")
        write_jsonl(predictions_path, prediction_rows)
        log_step(f"Writing JSON report to {report_path}")
        write_json(report_path, report)
        log_step(f"Writing HTML report to {html_report_path}")
        write_html(html_report_path, render_html_report(report, prediction_rows))

        print("Evaluation complete")
        print(f"Report: {report_path}")
        print(f"Predictions: {predictions_path}")
        print(f"HTML score report: {html_report_path}")
        print(
            "Summary: "
            f"avg_overall_score={aggregate['avg_overall_score']:.1f}, "
            f"exact_match_rate={aggregate['exact_match_rate']:.3f}, "
            f"avg_token_f1={aggregate['avg_token_f1']:.3f}, "
            f"avg_criteria_score={aggregate['avg_criteria_score']:.3f}, "
            f"errors={error_count}"
        )

        return 0


def resolve_prompt_context(inline_text: str | None, file_path: str | None) -> str | None:
    return resolve_extra_prompt(inline_text, file_path)
