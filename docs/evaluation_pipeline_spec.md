# Evaluation Pipeline Specification

## 1) Purpose
This document specifies the current evaluation and dataset-generation behavior implemented in:
- `src/evaluator.py`
- `src/eval.py`

It defines interfaces, data contracts, scoring, artifacts, and extension points for the meal-plan prompt evaluation workflow.

## 2) Scope
In scope:
- Synthetic dataset generation for athlete meal-plan prompts.
- Batch evaluation of `prompt`/`answer` JSONL datasets.
- Optional criteria-based scoring.
- Optional XML/context prompt injection.
- JSON, JSONL, and HTML reporting.

Out of scope:
- Ground-truth nutritional correctness validation.
- Human-in-the-loop review workflow.
- Long-term run comparison storage beyond file artifacts.

## 3) Components

### 3.1 Dataset Generator (`src/evaluator.py`)
Provides singleton `evaluator` with:

```python
evaluator.generate_dataset(
    task_description: str = DEFAULT_TASK_DESCRIPTION,
    *,
    prompt_inputs_spec: dict[str, str],
    output_file: str,
    num_cases: int,
    eval_criteria: list[str] | None = None,
    extra_prompt: str | None = None,
) -> list[dict[str, Any]]
```

Behavior:
- Generates deterministic synthetic inputs per row (`seed + index`).
- Builds prompt from `task_description`, `prompt_inputs_spec`, sampled input values.
- Appends `extra_prompt` under `Additional context:` when provided.
- Uses `DEFAULT_EVAL_CRITERIA` when `eval_criteria` is omitted.
- Writes JSON array to `output_file`.

Defaults:
- `DEFAULT_TASK_DESCRIPTION`: compact 1-day athlete meal-plan instruction.
- `DEFAULT_EVAL_CRITERIA`:
  - Includes breakfast/lunch/dinner
  - Aligns with athlete goal
  - Respects dietary restrictions
  - Concise and practical

### 3.2 Evaluation Runner (`src/eval.py`)
CLI module:

```bash
python -m src.eval [options]
```

High-level flow:
1. Load JSONL dataset (`prompt`, `answer`, optional criteria).
2. Resolve execution mode (`auto|api|demo`).
3. Resolve extra prompt context (inline and/or file).
4. Generate prediction per row.
5. Compute row metrics + aggregate metrics.
6. Write `predictions.jsonl`, `report.json`, `report.html`.

## 4) Data Contracts

### 4.1 Evaluation Input Row (JSONL)
Required fields:
- `prompt: str`
- `answer: str`

Optional fields:
- `id: str|int`
- `topic: str`
- `eval_criteria: list[str] | str`
- `criteria: list[str] | str` (alias)

Parsing rules:
- Empty lines ignored.
- Missing/empty `prompt` or `answer` -> validation error.
- `eval_criteria` normalized to `list[str]`.

### 4.2 Prediction Output Row (`predictions.jsonl`)
Per-row fields:
- `id`, `topic`, `prompt`, `reference`, `prediction`
- `eval_criteria`
- `metrics`:
  - `exact_match` (`0..1`)
  - `token_f1` (`0..1`)
  - `criteria_score` (`0..1`)
  - `overall_score` (`0..100`)
  - `grade` (`A|B|C|D|F`)
- Optional `error` string on failure

### 4.3 Run Report (`report.json`)
Includes:
- Run metadata: `run_name`, `dataset`, `rows_evaluated`, `errors`, `mode`, `model`, timestamps
- `metrics`:
  - `exact_match_rate`
  - `avg_token_f1`
  - `avg_criteria_score`
  - `avg_overall_score`
  - `avg_reference_tokens`
  - `avg_prediction_tokens`
- `overall_score` and `overall_grade`
- `extra_prompt`:
  - `enabled: bool`
  - `file: str | null`
- `artifacts`: paths to JSONL/JSON/HTML outputs

## 5) Scoring Specification

### 5.1 Row Metrics
- `exact_match`: normalized string equality.
- `token_f1`: token-overlap F1 between reference and prediction.
- `criteria_score`: average token-coverage across each criterion text.

### 5.2 Overall Score
`overall_score` is currently scaled to `0..100`:

- Without criteria:
  - `0.3 * exact_match + 0.7 * token_f1`
- With criteria:
  - `0.2 * exact_match + 0.5 * token_f1 + 0.3 * criteria_score`

Final scale:
- `overall_score = clamp(weighted * 100, 0, 100)`

Grades:
- `A >= 90`, `B >= 80`, `C >= 70`, `D >= 60`, else `F`

## 6) Prompt Construction

### 6.1 Base Evaluation Prompt
Model receives:
- Instruction to answer directly/concisely.
- `Question: <row prompt>`

### 6.2 Extra Prompt/XML Context
The runner supports:
- `--extra-prompt "<xml>..."`
- `--extra-prompt-file path/to/context.xml`

If provided, context is appended as:

```text
Additional context:
<...>
```

If both inline and file are provided, both are concatenated with a blank line separator.

## 7) CLI Interface (`src.eval`)

Supported options (current):
- `--dataset`
- `--output-dir`
- `--run-name`
- `--rows`
- `--mode` (`auto|api|demo`)
- `--model`
- `--fallback-model`
- `--max-tokens`
- `--extra-prompt`
- `--extra-prompt-file`
- `--fail-fast`

## 8) Artifacts
Per run directory (`data/eval_runs/<run_name>/`):
- `predictions.jsonl`
- `report.json`
- `report.html`

HTML report presents:
- Summary cards (`Total Test Cases`, `Average Score`, `Pass Rate`)
- Per-row table with scenario, criteria, prediction, score, reasoning

## 9) Failure Handling
- API errors produce row-level fallback with zeroed metrics and `error` field.
- With `--fail-fast`, evaluation raises immediately on first failure.
- Missing dataset or invalid schema raises early validation errors.

## 10) Non-Functional Notes
- Deterministic generator inputs rely on seeded randomization.
- Demo mode returns deterministic local placeholder responses.
- API mode depends on `ANTHROPIC_API_KEY` and model access.

## 11) Known Constraints
- `overall_score` is computed on a `0..100` scale, while some HTML labels display `/10` style text.
- Criteria scoring is lexical token overlap, not semantic compliance.

## 12) Suggested Future Enhancements
- Align score display scale consistently across CLI/JSON/HTML.
- Add semantic judge model or rubric-based scoring.
- Add per-criterion pass/fail breakdown with evidence snippets.
- Add trend report across multiple runs.
