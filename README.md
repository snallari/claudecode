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

Create CSV instead:

```bash
python -m src.dataset --rows 200 --format csv --output data/datasets/synthetic_dataset.csv
```

For more specific generated outputs, provide an ideal output example when using `src.evaluator`:

```python
from src.evaluator import evaluator

dataset = evaluator.generate_dataset(
	prompt_inputs_spec={
		"height": "Athlete's height in cm",
		"weight": "Athlete's weight in kg",
		"goal": "Goal of the athlete",
		"restrictions": "Dietary restrictions of the athlete",
	},
	output_file="data/dataset_with_ideal.json",
	num_cases=3,
	ideal_output="""
Breakfast: 3 eggs, 2 slices whole-grain toast, 1 cup berries.
Lunch: 170g grilled chicken, 1 cup rice, 2 cups mixed vegetables.
Snack: Greek yogurt (200g) + banana.
Dinner: 180g lean beef, 250g potatoes, salad with olive oil.
""",
)
```

## Run Evaluation Pipeline

Evaluate a JSONL dataset (`prompt` + `answer`) and generate per-row predictions + aggregated metrics.

Optional: include `eval_criteria` in each row to score criteria coverage (`avg_criteria_score`).

Example row:

```json
{"id":"1","prompt":"...","answer":"...","eval_criteria":["Includes breakfast/lunch/dinner","Aligns with goal"]}
```

Auto mode (uses API if `ANTHROPIC_API_KEY` exists, otherwise demo mode):

```bash
python -m src.eval --dataset data/datasets/synthetic_dataset.jsonl --rows 50
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

Run end-to-end workflow with grader integrated:

```bash
python -m src.workflow --mode demo --grader lexical --rows 25
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

Workflow defaults:

- Creates `data/datasets/workflow_dataset.jsonl` if missing
- Reuses existing dataset unless `--regenerate-dataset` is set
- Forwards all grader options directly into `src.eval`
