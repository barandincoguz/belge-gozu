.PHONY: setup lint test serve
setup:
	uv sync --extra dev
lint:
	uv run ruff check . && uv run ruff format --check . && uv run pyright
test:
	uv run pytest -m "not slow" -q
serve:
	uv run belge-gozu serve
