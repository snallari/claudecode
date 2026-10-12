from __future__ import annotations

import argparse
import html
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


def criteria_score(criteria: list[str], prediction: str) -> float:
    if not criteria:
        return 0.0

    prediction_tokens = set(tokenize(prediction))
    if not prediction_tokens:
        return 0.0

    criterion_scores: list[float] = []
    for criterion in criteria:
        criterion_tokens = set(tokenize(criterion))
        if not criterion_tokens:
            continue
        overlap = len(criterion_tokens & prediction_tokens)
        criterion_scores.append(overlap / len(criterion_tokens))

    if not criterion_scores:
        return 0.0

    return sum(criterion_scores) / len(criterion_scores)


def overall_score(exact_match_value: float, token_f1_value: float, criteria_score_value: float | None) -> float:
    if criteria_score_value is None:
        weighted = (0.3 * exact_match_value) + (0.7 * token_f1_value)
    else:
        weighted = (0.2 * exact_match_value) + (0.5 * token_f1_value) + (0.3 * criteria_score_value)
    return max(0.0, min(100.0, weighted * 100.0))


def score_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def parse_criteria(raw_criteria: Any) -> list[str]:
    if raw_criteria is None:
        return []
    if isinstance(raw_criteria, str):
        criteria = [raw_criteria]
    elif isinstance(raw_criteria, list):
        criteria = [str(item) for item in raw_criteria]
    else:
        return []

    return [criterion.strip() for criterion in criteria if str(criterion).strip()]


def load_jsonl_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
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
                    "eval_criteria": parse_criteria(row.get("eval_criteria", row.get("criteria", []))),
                }
            )

    if not rows:
        raise ValueError("Dataset is empty after parsing.")
    return rows


def build_eval_prompt(prompt: str, extra_prompt: str | None = None) -> str:
    base = (
        "Answer the following user question directly and concisely in one paragraph. "
        "Do not include extra meta-commentary.\n\n"
        f"Question: {prompt}"
    )
    if extra_prompt and extra_prompt.strip():
        return f"{base}\n\nAdditional context:\n{extra_prompt.strip()}"
    return base


def generate_prediction(
    row_prompt: str,
    mode: str,
    client: anthropic.Anthropic | None,
    model: str,
    max_tokens: int,
    extra_prompt: str | None = None,
) -> str:
    if mode == "demo":
        return build_demo_answer(build_eval_prompt(row_prompt, extra_prompt=extra_prompt))

    if client is None:
        raise RuntimeError("Anthropic client is required in API mode.")

    response = chat(
        client=client,
        messages=[{"role": "user", "content": build_eval_prompt(row_prompt, extra_prompt=extra_prompt)}],
        model=model,
        max_tokens=max_tokens,
        stream=False,
    )
    return get_text_from_response(response).strip()


def resolve_extra_prompt(inline_text: str | None, file_path: str | None) -> str | None:
    parts: list[str] = []

    if inline_text and inline_text.strip():
        parts.append(inline_text.strip())

    if file_path and file_path.strip():
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Extra prompt file not found: {path}")
        file_text = path.read_text(encoding="utf-8").strip()
        if file_text:
            parts.append(file_text)

    if not parts:
        return None

    return "\n\n".join(parts)


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    count = len(records)
    if count == 0:
        return {
            "exact_match_rate": 0.0,
            "avg_token_f1": 0.0,
            "avg_criteria_score": 0.0,
            "avg_overall_score": 0.0,
            "avg_reference_tokens": 0.0,
            "avg_prediction_tokens": 0.0,
        }

    exact_sum = sum(float(record["metrics"]["exact_match"]) for record in records)
    f1_sum = sum(float(record["metrics"]["token_f1"]) for record in records)
    criteria_scores = [
        float(record["metrics"]["criteria_score"])
        for record in records
        if record.get("eval_criteria")
    ]
    overall_scores = [float(record["metrics"].get("overall_score", 0.0)) for record in records]
    ref_tokens_sum = sum(len(tokenize(str(record["reference"]))) for record in records)
    pred_tokens_sum = sum(len(tokenize(str(record["prediction"]))) for record in records)

    return {
        "exact_match_rate": exact_sum / count,
        "avg_token_f1": f1_sum / count,
        "avg_criteria_score": (sum(criteria_scores) / len(criteria_scores)) if criteria_scores else 0.0,
        "avg_overall_score": (sum(overall_scores) / len(overall_scores)) if overall_scores else 0.0,
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


def render_html_report(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    metrics = report.get("metrics", {})
    run_name = html.escape(str(report.get("run_name", "")))
    dataset = html.escape(str(report.get("dataset", "")))
    mode = html.escape(str(report.get("mode", "")))
    model = html.escape(str(report.get("model", "")))
    rows_evaluated_value = int(report.get("rows_evaluated", 0) or 0)
    rows_evaluated = html.escape(str(rows_evaluated_value))
    errors = html.escape(str(report.get("errors", 0)))

    avg_overall = float(metrics.get("avg_overall_score", 0.0))
    pass_count = sum(1 for row in rows if float(row.get("metrics", {}).get("overall_score", 0.0)) >= 7.0)
    pass_rate = (pass_count / rows_evaluated_value * 100.0) if rows_evaluated_value else 0.0

    table_rows: list[str] = []
    for row in rows:
        row_id = html.escape(str(row.get("id", "")))
        topic = html.escape(str(row.get("topic", "")))
        prompt = html.escape(str(row.get("prompt", "")))
        prediction = html.escape(str(row.get("prediction", "")))

        row_metrics = row.get("metrics", {})
        exact_match_value = float(row_metrics.get("exact_match", 0.0))
        token_f1_value = float(row_metrics.get("token_f1", 0.0))
        criteria_value = float(row_metrics.get("criteria_score", 0.0))
        overall_value = float(row_metrics.get("overall_score", 0.0))

        criteria_items = row.get("eval_criteria", [])
        criteria_html = "".join(f"<li>{html.escape(str(c))}</li>" for c in criteria_items)
        criteria_block = f"<ul>{criteria_html}</ul>" if criteria_html else "<span class='muted'>No criteria</span>"

        reasoning = (
            f"Exact match={exact_match_value:.2f}; "
            f"Token F1={token_f1_value:.2f}; "
            f"Criteria={criteria_value:.2f}."
        )
        score_cell_class = "score-high" if overall_value >= 7 else "score-low"

        table_rows.append(
            "<tr>"
            f"<td><strong>id:</strong> {row_id}<br/><strong>topic:</strong> {topic}</td>"
            f"<td>{prompt}</td>"
            f"<td>{criteria_block}</td>"
            f"<td>{prediction}</td>"
            f"<td><span class='score-pill {score_cell_class}'>{overall_value:.1f}</span></td>"
            f"<td>{html.escape(reasoning)}</td>"
            "</tr>"
        )

    table_html = "\n".join(table_rows)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Prompt Evaluation Report - {run_name}</title>
    <style>
        body {{ font-family: Arial, Helvetica, sans-serif; margin: 18px; color: #111827; background: #f8f8f8; }}
        h1 {{ margin: 0 0 12px 0; font-size: 34px; font-weight: 700; }}
        .meta {{ font-size: 12px; color: #4b5563; margin-bottom: 12px; }}
        .cards {{ display: grid; grid-template-columns: repeat(3, minmax(190px, 1fr)); gap: 12px; margin-bottom: 14px; }}
        .card {{ border: 1px solid #d1d5db; border-radius: 2px; background: #fff; padding: 10px; }}
        .label {{ font-size: 11px; color: #374151; text-transform: uppercase; }}
        .value {{ margin-top: 4px; font-size: 28px; font-weight: 700; }}
        table {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 12px; }}
        th, td {{ border: 1px solid #d1d5db; text-align: left; vertical-align: top; padding: 8px; }}
        th {{ background: #111827; color: #fff; font-size: 11px; text-transform: uppercase; }}
        ul {{ margin: 0; padding-left: 16px; }}
        li {{ margin: 0 0 3px 0; }}
        .muted {{ color: #6b7280; font-size: 11px; }}
        .score-pill {{ display: inline-block; border-radius: 3px; min-width: 40px; text-align: center; padding: 2px 6px; font-weight: 700; }}
        .score-high {{ background: #dcfce7; color: #14532d; }}
        .score-low {{ background: #fee2e2; color: #7f1d1d; }}
    </style>
</head>
<body>
    <h1>Prompt Evaluation Report</h1>
    <div class="meta">Run: <strong>{run_name}</strong> · Dataset: {dataset} · Mode: {mode} · Model: {model} · Errors: {errors}</div>

    <section class=\"cards\">
        <div class="card"><div class="label">Total Test Cases</div><div class="value">{rows_evaluated}</div></div>
        <div class="card"><div class="label">Average Score</div><div class="value">{avg_overall:.1f} / 10</div></div>
        <div class="card"><div class="label">Pass Rate (≥7)</div><div class="value">{pass_rate:.1f}%</div></div>
    </section>

    <table>
        <thead>
            <tr>
                <th>Prompt Inputs</th>
                <th>Scenario</th>
                <th>Solution Criteria</th>
                <th>Prediction</th>
                <th>Score</th>
                <th>Reasoning</th>
            </tr>
        </thead>
        <tbody>
            {table_html}
        </tbody>
    </table>
</body>
</html>
"""


def write_html(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
    parser.add_argument(
        "--extra-prompt",
        default=None,
        help="Optional inline extra prompt/context (supports XML tags)",
    )
    parser.add_argument(
        "--extra-prompt-file",
        default=None,
        help="Path to a file containing extra prompt/context to append to each evaluation prompt",
    )
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

    resolved_extra_prompt = resolve_extra_prompt(args.extra_prompt, args.extra_prompt_file)

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
    html_report_path = run_dir / "report.html"

    started_at = datetime.now(timezone.utc).isoformat()
    prediction_rows: list[dict[str, Any]] = []
    error_count = 0

    print(f"Starting evaluation: rows={len(rows)} mode={mode} run={run_name}")

    for index, row in enumerate(rows, start=1):
        row_id = row["id"]
        prompt = row["prompt"]
        reference = row["answer"]
        eval_criteria = [str(criterion) for criterion in row.get("eval_criteria", [])]

        try:
            prediction = generate_prediction(
                row_prompt=prompt,
                mode=mode,
                client=client,
                model=resolved_model,
                max_tokens=args.max_tokens,
                extra_prompt=resolved_extra_prompt,
            )
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
            prediction_rows.append(
                {
                    "id": row_id,
                    "topic": row.get("topic", ""),
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": prediction,
                    "eval_criteria": eval_criteria,
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
                    "eval_criteria": eval_criteria,
                    "metrics": {
                        "exact_match": 0.0,
                        "token_f1": 0.0,
                        "criteria_score": 0.0,
                        "overall_score": 0.0,
                        "grade": "F",
                    },
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
                    "eval_criteria": eval_criteria,
                    "metrics": {
                        "exact_match": 0.0,
                        "token_f1": 0.0,
                        "criteria_score": 0.0,
                        "overall_score": 0.0,
                        "grade": "F",
                    },
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
        "overall_score": aggregate["avg_overall_score"],
        "overall_grade": score_grade(aggregate["avg_overall_score"]),
        "metrics": aggregate,
        "extra_prompt": {
            "enabled": bool(resolved_extra_prompt),
            "file": args.extra_prompt_file,
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "report": str(report_path),
            "html_report": str(html_report_path),
        },
    }

    write_jsonl(predictions_path, prediction_rows)
    write_json(report_path, report)
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


if __name__ == "__main__":
    main()