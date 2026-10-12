from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.client import chat, get_text_from_response, print_api_error, resolve_model
from src.demo import build_demo_answer

DEFAULT_MODEL = "claude-3-5-sonnet-latest"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def exact_match(reference: str, prediction: str) -> float:
    return float(normalize_text(reference) == normalize_text(prediction))


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(ref_counts[token], pred_counts[token]) for token in ref_counts)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_jsonl_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row = json.loads(stripped)
            prompt = str(row.get("prompt", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not prompt or not answer:
                raise ValueError(f"Row {index} must contain non-empty 'prompt' and 'answer'.")

            rows.append(
                {
                    "id": str(row.get("id", index)),
                    "topic": str(row.get("topic", "")),
                    "prompt": prompt,
                    "answer": answer,
                }
            )

    if not rows:
        raise ValueError("Dataset is empty after parsing.")
    return rows


def build_eval_prompt(prompt: str) -> str:
    return (
        "Answer the following user question directly and concisely in one paragraph. "
        "Do not include extra meta-commentary.\n\n"
        f"Question: {prompt}"
    )


def generate_prediction(
    row_prompt: str,
    mode: str,
    client: anthropic.Anthropic | None,
    model: str,
    max_tokens: int,
) -> str:
    if mode == "demo":
        return build_demo_answer(row_prompt)

    if client is None:
        raise RuntimeError("Anthropic client is required in API mode.")

    response = chat(
        client=client,
        messages=[{"role": "user", "content": build_eval_prompt(row_prompt)}],
        model=model,
        max_tokens=max_tokens,
        stream=False,
    )
    return get_text_from_response(response).strip()


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    count = len(records)
    if count == 0:
        return {
            "exact_match_rate": 0.0,
            "avg_token_f1": 0.0,
            "avg_reference_tokens": 0.0,
            "avg_prediction_tokens": 0.0,
        }

    exact_sum = sum(float(record["metrics"]["exact_match"]) for record in records)
    f1_sum = sum(float(record["metrics"]["token_f1"]) for record in records)
    ref_tokens_sum = sum(len(tokenize(str(record["reference"]))) for record in records)
    pred_tokens_sum = sum(len(tokenize(str(record["prediction"]))) for record in records)

    return {
        "exact_match_rate": exact_sum / count,
        "avg_token_f1": f1_sum / count,
        "avg_reference_tokens": ref_tokens_sum / count,
        "avg_prediction_tokens": pred_tokens_sum / count,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt/answer evaluation on a JSONL dataset.")
    parser.add_argument("--dataset", default="data/synthetic_dataset.jsonl", help="Path to JSONL dataset")
    parser.add_argument(
        "--output-dir",
        default="data/eval_runs",
        help="Directory where evaluation artifacts are stored",
    )
    parser.add_argument("--run-name", default=None, help="Optional run folder name")
    parser.add_argument("--rows", type=int, default=None, help="Optional max rows to evaluate")
    parser.add_argument(
        "--mode",
        choices=["auto", "api", "demo"],
        default="auto",
        help="Execution mode: api uses Anthropic, demo uses local stub, auto selects based on API key",
    )
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL), help="Model name")
    parser.add_argument(
        "--fallback-model",
        default=os.getenv("ANTHROPIC_FALLBACK_MODEL"),
        help="Fallback model name",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per model response")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if one row fails")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    rows = load_jsonl_dataset(Path(args.dataset))
    if args.rows is not None:
        if args.rows <= 0:
            raise ValueError("--rows must be greater than 0 when provided")
        rows = rows[: args.rows]

    requested_mode = args.mode
    api_key = os.getenv("ANTHROPIC_API_KEY")

    client: anthropic.Anthropic | None = None
    mode = requested_mode
    resolved_model = args.model

    if requested_mode == "auto":
        mode = "api" if api_key else "demo"

    if mode == "api":
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for --mode api")

        client = anthropic.Anthropic(api_key=api_key)
        resolved_model, resolution_note = resolve_model(client, args.model, args.fallback_model)
        if resolution_note:
            print(resolution_note)

    run_name = args.run_name or utc_timestamp()
    run_dir = Path(args.output_dir) / run_name
    predictions_path = run_dir / "predictions.jsonl"
    report_path = run_dir / "report.json"

    started_at = datetime.now(timezone.utc).isoformat()
    prediction_rows: list[dict[str, Any]] = []
    error_count = 0

    print(f"Starting evaluation: rows={len(rows)} mode={mode} run={run_name}")

    for index, row in enumerate(rows, start=1):
        row_id = row["id"]
        prompt = row["prompt"]
        reference = row["answer"]

        try:
            prediction = generate_prediction(
                row_prompt=prompt,
                mode=mode,
                client=client,
                model=resolved_model,
                max_tokens=args.max_tokens,
            )
            metrics = {
                "exact_match": exact_match(reference, prediction),
                "token_f1": token_f1(reference, prediction),
            }
            prediction_rows.append(
                {
                    "id": row_id,
                    "topic": row.get("topic", ""),
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": prediction,
                    "metrics": metrics,
                }
            )
        except anthropic.APIError as error:
            error_count += 1
            print_api_error(error)
            prediction_rows.append(
                {
                    "id": row_id,
                    "topic": row.get("topic", ""),
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": "",
                    "metrics": {"exact_match": 0.0, "token_f1": 0.0},
                    "error": str(error),
                }
            )
            if args.fail_fast:
                raise
        except Exception as error:  # pragma: no cover
            error_count += 1
            prediction_rows.append(
                {
                    "id": row_id,
                    "topic": row.get("topic", ""),
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": "",
                    "metrics": {"exact_match": 0.0, "token_f1": 0.0},
                    "error": str(error),
                }
            )
            if args.fail_fast:
                raise

        if index % 10 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)}")

    aggregate = aggregate_metrics(prediction_rows)
    finished_at = datetime.now(timezone.utc).isoformat()

    report = {
        "run_name": run_name,
        "dataset": str(Path(args.dataset)),
        "rows_evaluated": len(prediction_rows),
        "errors": error_count,
        "mode": mode,
        "requested_mode": requested_mode,
        "model": resolved_model if mode == "api" else "demo-local",
        "started_at": started_at,
        "finished_at": finished_at,
        "metrics": aggregate,
        "artifacts": {
            "predictions": str(predictions_path),
            "report": str(report_path),
        },
    }

    write_jsonl(predictions_path, prediction_rows)
    write_json(report_path, report)

    print("Evaluation complete")
    print(f"Report: {report_path}")
    print(f"Predictions: {predictions_path}")
    print(
        "Summary: "
        f"exact_match_rate={aggregate['exact_match_rate']:.3f}, "
        f"avg_token_f1={aggregate['avg_token_f1']:.3f}, "
        f"errors={error_count}"
    )


if __name__ == "__main__":
    main()