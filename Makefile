.PHONY: dev test eval

dev:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest -v

eval:
	uv run python scripts/eval_extract.py
