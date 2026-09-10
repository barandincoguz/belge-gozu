.PHONY: setup lab-setup lint test serve obs-up obs-down

# Yerel gate'ler ortak lab venv'i üstündeki overlay'den koşar (bkz.
# scripts/setup_lab_env.sh): araçlar uv.lock sürümlerine sabitli, torch ve
# model önbelleği ORTAK. Overlay yoksa davranış eskisi gibi `uv run`dır —
# CI ve Docker o yolda kalır, bu dosya iki ortamda da AYNI kapıları koşar.
LAB_BIN := .venv-lab/bin
ifneq ($(wildcard $(LAB_BIN)/python),)
RUFF := $(LAB_BIN)/ruff
PYRIGHT := $(LAB_BIN)/pyright --pythonpath $(LAB_BIN)/python
PYTEST := $(LAB_BIN)/pytest
BG := $(LAB_BIN)/belge-gozu
else
RUFF := uv run ruff
PYRIGHT := uv run pyright
PYTEST := uv run pytest
BG := uv run belge-gozu
endif

setup:
	uv sync --extra dev
lab-setup:
	bash scripts/setup_lab_env.sh
lint:
	$(RUFF) check . && $(RUFF) format --check . && $(PYRIGHT)
test:
	$(PYTEST) -m "not slow" -q
serve:
	$(BG) serve
obs-up:
	docker compose -f observability/docker-compose.yml up -d
obs-down:
	docker compose -f observability/docker-compose.yml down
