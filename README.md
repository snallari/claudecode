# Claude Code Container Starter

Minimal Python container starter for experimenting with Anthropic's SDK.

## Files
- `app.py`: tiny runnable entrypoint
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
python app.py
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
python -m src.dataset --rows 200 --output data/synthetic_dataset.jsonl
```

Create CSV instead:

```bash
python -m src.dataset --rows 200 --format csv --output data/synthetic_dataset.csv
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
python -m src.eval --dataset data/synthetic_dataset.jsonl --rows 50
```

Force Anthropic API mode:

```bash
python -m src.eval --mode api --model claude-3-5-sonnet-latest --dataset data/synthetic_dataset.jsonl
```

Force local demo mode (no API key required):

```bash
python -m src.eval --mode demo --dataset data/synthetic_dataset.jsonl --rows 25
```

Evaluation artifacts are written under `data/eval_runs/<run_name>/`:

- `report.json`: aggregate metrics (`exact_match_rate`, `avg_token_f1`, token lengths, metadata)
- `predictions.jsonl`: one row per evaluated sample with prediction + metrics
- `report.html`: human-friendly score dashboard with overall metrics and per-row scores
