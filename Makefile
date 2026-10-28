.PHONY: dev test eval eval-tools db-up db-down db-init kb-build kb-vectorize kb-reset kb-eval kb-mine kb-mining-eval smoke-embed

dev:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest -v

eval:
	uv run python scripts/eval_extract.py

eval-tools:
	uv run python scripts/eval_tools.py

db-up:
	docker compose up -d

db-down:
	docker compose down

db-init:
	uv run python scripts/init_db.py

kb-build:
	uv run python scripts/build_kb.py

kb-vectorize:
	uv run python scripts/vectorize_kb.py

kb-reset:
	uv run python scripts/reset_kb.py

kb-eval:
	uv run python scripts/eval_retrieval.py

kb-mine:
	uv run python scripts/mine_knowledge.py

kb-mining-eval:
	uv run python scripts/eval_mining.py

smoke-embed:
	uv run python scripts/smoke_embed.py
