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
