.PHONY: setup lint test serve obs-up obs-down
setup:
	uv sync --extra dev
lint:
	uv run ruff check . && uv run ruff format --check . && uv run pyright
test:
	uv run pytest -m "not slow" -q
serve:
	uv run belge-gozu serve
obs-up:
	docker compose -f observability/docker-compose.yml up -d
obs-down:
	docker compose -f observability/docker-compose.yml down
