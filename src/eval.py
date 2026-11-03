from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.client import chat, get_text_from_response, print_api_error, resolve_model
from src.demo import build_demo_answer

DEFAULT_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_GRADER = "lexical"
DEFAULT_CODE_TIMEOUT_SECONDS = 3.0


def log_step(message: str) -> None:
    print(f"[step] {message}")


def normalize_grader_method(grader: str) -> str:
    normalized = str(grader).strip().lower()
    if normalized == "coding":
        return "code"
    return normalized


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


def clamp_score_0_100(score: float) -> float:
    return max(0.0, min(100.0, score))


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None

    candidate = stripped[start_index : end_index + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


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


def parse_code_grader_spec(raw_spec: Any) -> dict[str, Any] | None:
    if raw_spec is None or not isinstance(raw_spec, dict):
        return None

    mode = str(raw_spec.get("mode", "function")).strip().lower() or "function"
    function_name_raw = raw_spec.get("function_name")
    function_name = str(function_name_raw).strip() if function_name_raw is not None else ""

    tests_raw = raw_spec.get("tests", [])
    normalized_tests: list[dict[str, Any]] = []
    if isinstance(tests_raw, list):
        for index, test_case in enumerate(tests_raw, start=1):
            if not isinstance(test_case, dict):
                continue

            raw_args = test_case.get("args", [])
            if isinstance(raw_args, list):
                args = raw_args
            else:
                args = [raw_args]

            raw_kwargs = test_case.get("kwargs", {})
            kwargs = raw_kwargs if isinstance(raw_kwargs, dict) else {}

            normalized_tests.append(
                {
                    "name": str(test_case.get("name", f"case_{index}")),
                    "args": args,
                    "kwargs": kwargs,
                    "expected": test_case.get("expected"),
                }
            )

    result: dict[str, Any] = {
        "mode": mode,
        "tests": normalized_tests,
    }
    if function_name:
        result["function_name"] = function_name

    timeout_raw = raw_spec.get("timeout_seconds")
    if isinstance(timeout_raw, (int, float)) and timeout_raw > 0:
        result["timeout_seconds"] = float(timeout_raw)

    return result


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
                    "code_grader": parse_code_grader_spec(row.get("code_grader")),
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


def build_grader_prompt(
    prompt: str,
    reference: str,
    prediction: str,
    criteria: list[str],
) -> str:
    criteria_text = "\n".join(f"- {criterion}" for criterion in criteria) if criteria else "- No explicit criteria"
    return (
        "You are an evaluation grader.\n"
        "Score the model prediction using the user question, reference answer, and optional criteria.\n"
        "Use a strict 0-100 rubric where:\n"
        "- 90-100: excellent, highly correct, complete, and aligned\n"
        "- 80-89: strong, minor omissions\n"
        "- 70-79: acceptable, notable gaps\n"
        "- 60-69: weak, several issues\n"
        "- 0-59: poor or incorrect\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\"score\": <number 0-100>, \"grade\": \"A|B|C|D|F\", \"rationale\": \"brief explanation\"}\n\n"
        f"Question:\n{prompt}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Prediction:\n{prediction}\n\n"
        f"Criteria:\n{criteria_text}\n"
    )


def model_grader_result(
    *,
    client: anthropic.Anthropic,
    grader_model: str,
    max_tokens: int,
    prompt: str,
    reference: str,
    prediction: str,
    criteria: list[str],
) -> dict[str, Any]:
    grader_response = chat(
        client=client,
        messages=[
            {
                "role": "user",
                "content": build_grader_prompt(
                    prompt=prompt,
                    reference=reference,
                    prediction=prediction,
                    criteria=criteria,
                ),
            }
        ],
        model=grader_model,
        max_tokens=max_tokens,
        stream=False,
    )
    grader_text = get_text_from_response(grader_response).strip()
    parsed = extract_first_json_object(grader_text)

    if not parsed:
        raise ValueError("Model grader did not return valid JSON.")

    raw_score = float(parsed.get("score", 0.0))
    score = clamp_score_0_100(raw_score)
    grade = str(parsed.get("grade") or score_grade(score)).strip().upper()
    if grade not in {"A", "B", "C", "D", "F"}:
        grade = score_grade(score)

    rationale = str(parsed.get("rationale", "")).strip() or "No rationale provided by grader model."
    return {
        "score": score,
        "grade": grade,
        "rationale": rationale,
        "raw_response": grader_text,
    }


def extract_python_code(text: str) -> str:
    if not text.strip():
        return ""

    fenced_python = re.search(r"```(?:python|py)\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_python:
        return fenced_python.group(1).strip()

    fenced_generic = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if fenced_generic:
        return fenced_generic.group(1).strip()

    return text.strip()


def build_code_grader_runner_script() -> str:
    return """
import json
import sys

def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))

def error_payload(message):
    return {
        "passed": 0,
        "total": 0,
        "failures": [],
        "error": message,
    }

raw = sys.stdin.read()
try:
    payload = json.loads(raw or "{}")
except Exception:
    emit(error_payload("Invalid grader payload JSON."))
    raise SystemExit(0)

code = str(payload.get("code", ""))
spec = payload.get("spec", {})
if not isinstance(spec, dict):
    emit(error_payload("Invalid code grader spec."))
    raise SystemExit(0)

mode = str(spec.get("mode", "function")).strip().lower() or "function"
if mode != "function":
    emit(error_payload(f"Unsupported code grader mode: {mode}."))
    raise SystemExit(0)

tests = spec.get("tests", [])
if not isinstance(tests, list) or not tests:
    emit(error_payload("Code grader tests are missing or invalid."))
    raise SystemExit(0)

namespace = {}
try:
    exec(code, namespace)
except Exception as exc:
    emit(error_payload(f"Code execution failed: {exc}"))
    raise SystemExit(0)

function_name = spec.get("function_name")
target = None
if isinstance(function_name, str) and function_name.strip():
    target = namespace.get(function_name.strip())
    if not callable(target):
        emit(error_payload(f"Function '{function_name.strip()}' not found or not callable."))
        raise SystemExit(0)
else:
    for key, value in namespace.items():
        if key.startswith("__"):
            continue
        if callable(value):
            target = value
            function_name = key
            break
    if target is None:
        emit(error_payload("No callable function found in generated code."))
        raise SystemExit(0)

passed = 0
failures = []
for index, case in enumerate(tests, start=1):
    if not isinstance(case, dict):
        failures.append({"index": index, "message": "Test case must be an object."})
        continue

    args = case.get("args", [])
    kwargs = case.get("kwargs", {})
    expected = case.get("expected")
    if not isinstance(args, list):
        args = [args]
    if not isinstance(kwargs, dict):
        kwargs = {}

    try:
        actual = target(*args, **kwargs)
    except Exception as exc:
        failures.append({
            "index": index,
            "name": str(case.get("name", f"case_{index}")),
            "message": f"Runtime error: {exc}",
        })
        continue

    if actual == expected:
        passed += 1
    else:
        failures.append(
            {
                "index": index,
                "name": str(case.get("name", f"case_{index}")),
                "message": f"Expected {expected!r}, got {actual!r}",
            }
        )

emit(
    {
        "passed": passed,
        "total": len(tests),
        "failures": failures,
        "error": None,
        "function_name": function_name,
    }
)
"""


def code_grader_result(
    *,
    prediction: str,
    code_spec: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    code = extract_python_code(prediction)
    if not code:
        return {
            "score": 0.0,
            "grade": "F",
            "rationale": "No Python code found in prediction.",
            "details": {
                "passed": 0,
                "total": 0,
                "failures": [],
            },
        }

    payload = {
        "code": code,
        "spec": code_spec,
    }

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", build_code_grader_runner_script()],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "score": 0.0,
            "grade": "F",
            "rationale": f"Code execution timed out after {timeout_seconds:.1f}s.",
            "details": {
                "passed": 0,
                "total": 0,
                "failures": [
                    {
                        "index": 0,
                        "name": "timeout",
                        "message": f"Execution exceeded {timeout_seconds:.1f}s",
                    }
                ],
            },
        }

    stdout = completed.stdout.strip()
    if not stdout:
        stderr_text = completed.stderr.strip()
        stderr_suffix = f" stderr={stderr_text}" if stderr_text else ""
        return {
            "score": 0.0,
            "grade": "F",
            "rationale": "Code grader returned an empty result." + stderr_suffix,
            "details": {
                "passed": 0,
                "total": 0,
                "failures": [],
            },
        }

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "grade": "F",
            "rationale": "Code grader returned invalid JSON output.",
            "details": {
                "passed": 0,
                "total": 0,
                "failures": [],
            },
        }

    passed = int(parsed.get("passed", 0) or 0)
    total = int(parsed.get("total", 0) or 0)
    failures = parsed.get("failures", [])
    if not isinstance(failures, list):
        failures = []
    execution_error = str(parsed.get("error") or "").strip()

    if total > 0 and not execution_error:
        score = (passed / total) * 100.0
    else:
        score = 0.0

    score = clamp_score_0_100(score)
    grade = score_grade(score)

    if execution_error:
        rationale = execution_error
    else:
        rationale = f"Passed {passed}/{total} tests."
        if failures:
            first_failure = failures[0]
            if isinstance(first_failure, dict):
                failure_message = str(first_failure.get("message", "")).strip()
                if failure_message:
                    rationale = f"{rationale} First failure: {failure_message}"

    return {
        "score": score,
        "grade": grade,
        "rationale": rationale,
        "details": {
            "passed": passed,
            "total": total,
            "failures": failures,
            "function_name": parsed.get("function_name"),
        },
    }


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
    pass_count = sum(1 for row in rows if float(row.get("metrics", {}).get("overall_score", 0.0)) >= 70.0)
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

        grading_details = row_metrics.get("grading_details", {})
        grader_rationale = str(grading_details.get("rationale", "")).strip()
        model_rationale = str(grading_details.get("model_rationale", "")).strip()
        if grader_rationale:
            reasoning = grader_rationale
        elif model_rationale:
            reasoning = model_rationale
        else:
            reasoning = (
                f"Exact match={exact_match_value:.2f}; "
                f"Token F1={token_f1_value:.2f}; "
                f"Criteria={criteria_value:.2f}."
            )
        score_cell_class = "score-high" if overall_value >= 70 else "score-low"

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
        <div class="card"><div class="label">Average Score</div><div class="value">{avg_overall:.1f} / 100</div></div>
        <div class="card"><div class="label">Pass Rate (≥70)</div><div class="value">{pass_rate:.1f}%</div></div>
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
    parser.add_argument("--dataset", default="data/datasets/synthetic_dataset.jsonl", help="Path to JSONL dataset")
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
        "--grader",
        choices=["lexical", "model", "code", "coding"],
        default=os.getenv("EVAL_GRADER", DEFAULT_GRADER),
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
        default=DEFAULT_CODE_TIMEOUT_SECONDS,
        help="Timeout in seconds for executing generated code per row when --grader code",
    )
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
    grader_method = normalize_grader_method(args.grader)
    log_step(f"Arguments parsed. Grader method resolved to '{grader_method}'.")

    log_step(f"Loading dataset from {args.dataset}")
    rows = load_jsonl_dataset(Path(args.dataset))
    log_step(f"Loaded {len(rows)} dataset rows")
    if args.rows is not None:
        if args.rows <= 0:
            raise ValueError("--rows must be greater than 0 when provided")
        rows = rows[: args.rows]
        log_step(f"Applied --rows limit: evaluating first {len(rows)} rows")

    log_step("Resolving extra prompt context")
    resolved_extra_prompt = resolve_extra_prompt(args.extra_prompt, args.extra_prompt_file)
    log_step(f"Extra prompt enabled: {'yes' if resolved_extra_prompt else 'no'}")

    requested_mode = args.mode
    api_key = os.getenv("ANTHROPIC_API_KEY")

    client: anthropic.Anthropic | None = None
    mode = requested_mode
    resolved_model = args.model
    resolved_grader_model = args.grader_model or args.model

    if requested_mode == "auto":
        mode = "api" if api_key else "demo"
    log_step(f"Execution mode resolved: requested={requested_mode}, active={mode}")

    if mode == "api":
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for --mode api")

        log_step("Initializing Anthropic client")
        client = anthropic.Anthropic(api_key=api_key)
        log_step(f"Resolving prediction model: requested={args.model}")
        resolved_model, resolution_note = resolve_model(client, args.model, args.fallback_model)
        if resolution_note:
            print(resolution_note)

        if grader_method == "model":
            log_step(f"Resolving grader model: requested={resolved_grader_model}")
            resolved_grader_model, grader_resolution_note = resolve_model(
                client,
                resolved_grader_model,
                args.grader_fallback_model,
            )
            if grader_resolution_note:
                print(f"Grader: {grader_resolution_note}")
    elif grader_method == "model":
        raise RuntimeError("--grader model requires API mode. Use --mode api (or auto with API key).")

    if grader_method == "code" and args.code_timeout_seconds <= 0:
        raise ValueError("--code-timeout-seconds must be greater than 0")

    run_name = args.run_name or utc_timestamp()
    run_dir = Path(args.output_dir) / run_name
    predictions_path = run_dir / "predictions.jsonl"
    report_path = run_dir / "report.json"
    html_report_path = run_dir / "report.html"
    log_step(f"Run artifacts directory: {run_dir}")

    started_at = datetime.now(timezone.utc).isoformat()
    prediction_rows: list[dict[str, Any]] = []
    error_count = 0

    print(
        f"Starting evaluation: rows={len(rows)} mode={mode} grader={grader_method} run={run_name}"
    )

    for index, row in enumerate(rows, start=1):
        row_id = row["id"]
        prompt = row["prompt"]
        reference = row["answer"]
        eval_criteria = [str(criterion) for criterion in row.get("eval_criteria", [])]
        code_grader_spec = row.get("code_grader")
        print(f"[row {index}/{len(rows)}] start id={row_id}")

        try:
            prediction = generate_prediction(
                row_prompt=prompt,
                mode=mode,
                client=client,
                model=resolved_model,
                max_tokens=args.max_tokens,
                extra_prompt=resolved_extra_prompt,
            )
            print(f"[row {index}/{len(rows)}] prediction generated")
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
            print(f"[row {index}/{len(rows)}] lexical metrics computed")

            if grader_method == "model":
                if client is None:
                    raise RuntimeError("Anthropic client is required for model grading.")
                print(f"[row {index}/{len(rows)}] applying model grader")
                grader_output = model_grader_result(
                    client=client,
                    grader_model=resolved_grader_model,
                    max_tokens=args.grader_max_tokens,
                    prompt=prompt,
                    reference=reference,
                    prediction=prediction,
                    criteria=eval_criteria,
                )
                metrics["overall_score"] = grader_output["score"]
                metrics["grade"] = grader_output["grade"]
                metrics["grading_details"] = {
                    "method": "model",
                    "model": resolved_grader_model,
                    "model_rationale": grader_output["rationale"],
                    "raw_response": grader_output["raw_response"],
                }
                print(f"[row {index}/{len(rows)}] model grading complete")
            elif grader_method == "code":
                if code_grader_spec:
                    print(f"[row {index}/{len(rows)}] applying code grader")
                    timeout_seconds = float(code_grader_spec.get("timeout_seconds", args.code_timeout_seconds))
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
                    print(f"[row {index}/{len(rows)}] code grading complete")
                else:
                    metrics["grading_details"] = {
                        "method": "code",
                        "rationale": "No code_grader spec provided for this row; using lexical metrics only.",
                    }
                    print(f"[row {index}/{len(rows)}] code_grader spec missing; kept lexical metrics")
            else:
                metrics["grading_details"] = {
                    "method": "lexical",
                }
                print(f"[row {index}/{len(rows)}] using lexical grading")

            prediction_rows.append(
                {
                    "id": row_id,
                    "topic": row.get("topic", ""),
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": prediction,
                    "eval_criteria": eval_criteria,
                    "code_grader": code_grader_spec,
                    "metrics": metrics,
                }
            )
            print(f"[row {index}/{len(rows)}] completed")
        except anthropic.APIError as error:
            error_count += 1
            print_api_error(error)
            print(f"[row {index}/{len(rows)}] failed with API error")
            prediction_rows.append(
                {
                    "id": row_id,
                    "topic": row.get("topic", ""),
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": "",
                    "eval_criteria": eval_criteria,
                    "code_grader": code_grader_spec,
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
            print(f"[row {index}/{len(rows)}] failed with error: {error}")
            prediction_rows.append(
                {
                    "id": row_id,
                    "topic": row.get("topic", ""),
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": "",
                    "eval_criteria": eval_criteria,
                    "code_grader": code_grader_spec,
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

    log_step("Computing aggregate metrics")
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
        "grader": {
            "method": grader_method,
            "model": resolved_grader_model if grader_method == "model" else None,
        },
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


if __name__ == "__main__":
    main()