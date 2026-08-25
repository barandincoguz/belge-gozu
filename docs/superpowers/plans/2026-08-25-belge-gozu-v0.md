# Belge-Gözü v0 (Hafta 1 — Canlı Demo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Türkçe mevzuat PDF'lerini sayfa görüntüsü olarak indeksleyip (OCR'sız), iki aşamalı binary late-interaction retrieval + atıflı VLM yanıtıyla HF Space'te canlı çalışan v0'ı yayınlamak.

**Architecture:** Çevrimdışı boru hattı (manifest → indir → WebP render → ColSmol-sınıfı embedding → binary packed indeks → HF Datasets) + çevrimiçi FastAPI servisi (mmap indeks, kendi yazdığımız Hamming/MaxSim çekirdeği, tak-çıkar Gemini answerer, tek sayfa UI). Model'e dokunan her şey `Encoder`/`Answerer` arayüzlerinin arkasında; CI hiçbir zaman ağ/GPU/model kullanmaz (Fake uygulamalar).

**Tech Stack:** Python 3.12, uv, numpy≥2.0 (bitwise_count), PyMuPDF, Pillow, pandas+pyarrow, httpx, typer, pydantic-settings, FastAPI+uvicorn, huggingface_hub, google-genai (FreeAPI answerer), colpali-engine+torch (yalnızca `ml` extra'sı), pytest, ruff, pyright, GitHub Actions, HF Space (Docker).

**Spec:** `docs/superpowers/specs/2026-08-25-belge-gozu-design.md`

## Global Constraints

- Bütçe: her şey ücretsiz katman (HF free Space/Datasets, Gemini free tier). Ücretli servis yok.
- Ana boru hattında OCR/PDF metin çıkarımı YOK — sayfalar yalnızca görüntü.
- numpy ≥ 2.0 zorunlu (`np.bitwise_count`); Python ≥ 3.12; embedding boyutu 128.
- CI'da test hiçbir zaman ağa çıkmaz, model indirmez, GPU istemez (`-m "not slow"`).
- gov.tr sitelerine istekler arası ≥ 1.0 sn bekleme, kimlikli User-Agent, idempotent indirme.
- README İngilizce, UI metinleri Türkçe (v0'da yalnız TR; dil anahtarı Plan 3).
- Varsayılan retriever `vidore/colSmol-500M`, answerer `gemini-2.0-flash` — Task 13'te
  güncel sürümleri doğrulanır ve YALNIZ config'te güncellenir (spec §11).
- Tüm kod `ruff` (line-length 100) + `pyright` basic'ten geçer; her task kendi commit'iyle biter.
- Hukuki not: resmî metinler FSEK m.31 kapsamında telif dışı; README'de belirtilir.

## File Structure

```
pyproject.toml, Makefile, Dockerfile, .pre-commit-config.yaml
.github/workflows/ci.yml
src/belge_gozu/
  config.py            # Settings (pydantic-settings, BG_ env prefix)
  cli.py               # typer app: corpus/index/serve komutları
  corpus/manifest.py   # Manifest okuma/doğrulama/probe
  corpus/download.py   # rate-limit'li idempotent indirici
  corpus/render.py     # PDF → WebP + meta.parquet
  index/encode.py      # Encoder protokolü, FakeEncoder, ColSmolEncoder (lazy)
  index/store.py       # PackedIndex: binarize/pack/save/load(mmap)
  index/hub.py         # HF Datasets push/pull
  retrieval/types.py   # PageHit
  retrieval/core.py    # TwoStageRetriever (Hamming eleme + binary MaxSim)
  answer/base.py       # Answer, Answerer protokolü, AskService (abstain eşiği)
  answer/gemini.py     # GeminiAnswerer (google-genai sarmalayıcı)
  app/main.py          # FastAPI + istek logu (sqlite) + /stats + statik UI
  app/static/index.html
data/manifest/v0_manifest.csv
tests/ (her modülün aynası) + tests/conftest.py
```

---

### Task 1: Repo iskeleti, config, CI

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `src/belge_gozu/__init__.py`, `src/belge_gozu/config.py`, `tests/test_config.py`, `README.md`

**Interfaces:**
- Produces: `belge_gozu.config.Settings` — alanlar: `data_dir: Path = Path("data")`, `index_dir: Path = Path("data/index")`, `retriever_model: str = "vidore/colSmol-500M"`, `device: str = "auto"`, `hf_dataset_repo: str = ""`, `gemini_model: str = "gemini-2.0-flash"`, `gemini_api_key: str = ""`, `stage1_candidates: int = 200`, `top_k: int = 5`, `min_score_threshold: float = 20.0`, `request_delay_s: float = 1.0`. Env prefix `BG_`. `get_settings() -> Settings` (lru_cache).

- [ ] **Step 1: pyproject.toml yaz**

```toml
[project]
name = "belge-gozu"
version = "0.1.0"
description = "Visual document RAG for Turkish legal documents"
requires-python = ">=3.12"
dependencies = [
  "numpy>=2.0",
  "httpx>=0.27",
  "pymupdf>=1.24",
  "pillow>=10.0",
  "pandas>=2.2",
  "pyarrow>=16.0",
  "typer>=0.12",
  "pydantic-settings>=2.4",
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "huggingface-hub>=0.24",
  "google-genai>=1.0",
]

[project.optional-dependencies]
ml = ["colpali-engine>=0.3", "torch>=2.4"]
dev = ["pytest>=8.0", "ruff>=0.6", "pyright>=1.1.380", "pre-commit>=3.8"]

[project.scripts]
belge-gozu = "belge_gozu.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/belge_gozu"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pyright]
include = ["src"]
typeCheckingMode = "basic"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: gerçek model/ağ gerektirir; CI'da koşmaz"]
```

- [ ] **Step 2: Makefile, .gitignore, pre-commit, CI yaz**

`Makefile`:

```makefile
.PHONY: setup lint test serve
setup:
	uv sync --extra dev
lint:
	uv run ruff check . && uv run ruff format --check . && uv run pyright
test:
	uv run pytest -m "not slow" -q
serve:
	uv run belge-gozu serve
```

`.gitignore`:

```
.venv/
__pycache__/
data/*
!data/manifest/
!data/manifest/**
*.egg-info/
.pytest_cache/
.ruff_cache/
requests.sqlite
.env
.env.*
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

`.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push: {branches: [main]}
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - run: make lint
      - run: make test
```

`README.md` (v0 taslağı — canlı link Task 13'te eklenecek):

```markdown
# Belge-Gözü

Visual document RAG for Turkish legal documents. Pages are indexed as images
(no OCR, no parsing) with ColPali-class late-interaction retrieval and a
pluggable VLM answerer. Work in progress — v0.

Legal note: Turkish official texts are exempt from copyright (FSEK art. 31).
```

- [ ] **Step 3: Başarısız testi yaz** — `tests/test_config.py`

```python
from belge_gozu.config import Settings


def test_defaults():
    s = Settings()
    assert s.retriever_model == "vidore/colSmol-500M"
    assert s.stage1_candidates == 200
    assert s.top_k == 5
    assert s.request_delay_s == 1.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("BG_TOP_K", "3")
    assert Settings().top_k == 3
```

- [ ] **Step 4: Testin başarısız olduğunu gör**

Run: `uv sync --extra dev && uv run pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: belge_gozu.config`)

- [ ] **Step 5: config.py yaz**

```python
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BG_")

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    retriever_model: str = "vidore/colSmol-500M"
    device: str = "auto"
    hf_dataset_repo: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = ""
    stage1_candidates: int = 200
    top_k: int = 5
    # Uncalibrated in v0; calibrated against the benchmark in Plan 2 (spec §6).
    min_score_threshold: float = 20.0
    request_delay_s: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`src/belge_gozu/__init__.py` boş dosya.

- [ ] **Step 6: Testlerin geçtiğini gör**

Run: `uv run pytest tests/test_config.py -v && make lint`
Expected: 2 PASS; lint temiz

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: project skeleton, settings, CI"
```

---

### Task 2: Manifest modülü + v0 manifest verisi

**Files:**
- Create: `src/belge_gozu/corpus/__init__.py`, `src/belge_gozu/corpus/manifest.py`, `data/manifest/v0_manifest.csv`, `tests/corpus/test_manifest.py`

**Interfaces:**
- Produces: `ManifestRow` (pydantic model: `doc_id: str`, `doc_name: str`, `doc_type: Literal["kanun", "rg_tarihi"]`, `url: str`), `load_manifest(path: Path) -> list[ManifestRow]` (hatalı satırda `ValueError`), `probe(rows, client) -> list[tuple[str, int]]` (doc_id, HTTP status — HEAD isteği).

- [ ] **Step 1: Başarısız testi yaz** — `tests/corpus/test_manifest.py`

```python
from pathlib import Path

import httpx
import pytest

from belge_gozu.corpus.manifest import load_manifest, probe

CSV = """doc_id,doc_name,doc_type,url
k6098,Türk Borçlar Kanunu,kanun,https://example.org/1.5.6098.pdf
rg1930,RG 1930 örneği,rg_tarihi,https://example.org/arsiv/1519.pdf
"""


def test_load(tmp_path: Path):
    p = tmp_path / "m.csv"
    p.write_text(CSV, encoding="utf-8")
    rows = load_manifest(p)
    assert [r.doc_id for r in rows] == ["k6098", "rg1930"]
    assert rows[0].doc_type == "kanun"


def test_bad_type_rejected(tmp_path: Path):
    p = tmp_path / "m.csv"
    p.write_text(CSV.replace("rg_tarihi", "bilinmeyen"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(p)


def test_probe():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200 if "6098" in str(request.url) else 404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = load_manifest_from_text(CSV)
    assert probe(rows, client) == [("k6098", 200), ("rg1930", 404)]
```

Not: `load_manifest_from_text(text: str) -> list[ManifestRow]` de üret (test ve CLI kolaylığı).
Teste import'unu ekle: `from belge_gozu.corpus.manifest import load_manifest_from_text`.

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/corpus -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: manifest.py yaz**

```python
import csv
import io
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError


class ManifestRow(BaseModel):
    doc_id: str
    doc_name: str
    doc_type: Literal["kanun", "rg_tarihi"]
    url: str


def load_manifest_from_text(text: str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for i, rec in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        try:
            rows.append(ManifestRow(**rec))
        except ValidationError as e:
            raise ValueError(f"manifest satır {i}: {e}") from e
    if not rows:
        raise ValueError("manifest boş")
    return rows


def load_manifest(path: Path) -> list[ManifestRow]:
    return load_manifest_from_text(path.read_text(encoding="utf-8"))


def probe(rows: list[ManifestRow], client: httpx.Client) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for r in rows:
        try:
            resp = client.head(r.url, follow_redirects=True, timeout=20)
            out.append((r.doc_id, resp.status_code))
        except (httpx.HTTPError, httpx.InvalidURL):
            out.append((r.doc_id, 0))
    return out
```

`tests/corpus/__init__.py` gerekmiyor (pytest rootdir'den bulur); `src/belge_gozu/corpus/__init__.py` boş.

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/corpus -v`
Expected: 3 PASS

- [ ] **Step 5: v0 manifest verisini yaz** — `data/manifest/v0_manifest.csv`

Başlangıç seti: 24 temel kanun (mevzuat.gov.tr `MevzuatMetin/1.5.<no>.pdf` deseni) +
6 tarihî RG adayı (`resmigazete.gov.tr/arsiv/<sayı>.pdf` deseni). URL desenleri Task 13'te
`belge-gozu corpus probe` ile canlı doğrulanır; ölü çıkanlar orada düzeltilir ve liste
50 kanuna genişletilir (spec §4).

```csv
doc_id,doc_name,doc_type,url
k6098,Türk Borçlar Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.6098.pdf
k4721,Türk Medeni Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.4721.pdf
k5237,Türk Ceza Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.5237.pdf
k6100,Hukuk Muhakemeleri Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.6100.pdf
k5271,Ceza Muhakemesi Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.5271.pdf
k4857,İş Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.4857.pdf
k6102,Türk Ticaret Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.6102.pdf
k2004,İcra ve İflas Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.2004.pdf
k6502,Tüketicinin Korunması Hakkında Kanun,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.6502.pdf
k5510,Sosyal Sigortalar ve GSS Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.5510.pdf
k213,Vergi Usul Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.213.pdf
k193,Gelir Vergisi Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.193.pdf
k3065,Katma Değer Vergisi Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.3065.pdf
k6331,İş Sağlığı ve Güvenliği Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.6331.pdf
k6698,Kişisel Verilerin Korunması Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.6698.pdf
k4054,Rekabetin Korunması Hakkında Kanun,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.4054.pdf
k2577,İdari Yargılama Usulü Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.2577.pdf
k5651,İnternet Yayınları Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.5651.pdf
k1136,Avukatlık Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.1136.pdf
k2942,Kamulaştırma Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.2942.pdf
k634,Kat Mülkiyeti Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.634.pdf
k5188,Özel Güvenlik Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.5188.pdf
k3194,İmar Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.3194.pdf
k2872,Çevre Kanunu,kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.5.2872.pdf
rg1928a,RG arşiv örneği 1,rg_tarihi,https://www.resmigazete.gov.tr/arsiv/1054.pdf
rg1935a,RG arşiv örneği 2,rg_tarihi,https://www.resmigazete.gov.tr/arsiv/3035.pdf
rg1945a,RG arşiv örneği 3,rg_tarihi,https://www.resmigazete.gov.tr/arsiv/6034.pdf
rg1955a,RG arşiv örneği 4,rg_tarihi,https://www.resmigazete.gov.tr/arsiv/9022.pdf
rg1965a,RG arşiv örneği 5,rg_tarihi,https://www.resmigazete.gov.tr/arsiv/12011.pdf
rg1975a,RG arşiv örneği 6,rg_tarihi,https://www.resmigazete.gov.tr/arsiv/15408.pdf
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: corpus manifest loader and v0 manifest data"
```

---

### Task 3: İndirici (rate-limit'li, idempotent)

**Files:**
- Create: `src/belge_gozu/corpus/download.py`, `tests/corpus/test_download.py`

**Interfaces:**
- Consumes: `ManifestRow`, `Settings.request_delay_s`
- Produces: `download_all(rows: list[ManifestRow], out_dir: Path, client: httpx.Client, delay_s: float = 1.0, sleep=time.sleep) -> DownloadReport`; PDF'ler `out_dir/pdf/<doc_id>.pdf`; durum `out_dir/state.json` (`{doc_id: {"sha256": str, "status": "ok"|"failed"}}`). `DownloadReport` (pydantic): `ok: list[str]`, `skipped: list[str]`, `failed: list[str]`.

- [ ] **Step 1: Başarısız testleri yaz** — `tests/corpus/test_download.py`

```python
import json
from pathlib import Path

import httpx

from belge_gozu.corpus.download import download_all
from belge_gozu.corpus.manifest import load_manifest_from_text

CSV = """doc_id,doc_name,doc_type,url
a,Belge A,kanun,https://example.org/a.pdf
b,Belge B,kanun,https://example.org/b.pdf
"""


def make_client(fail_ids: set[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        name = str(request.url).rsplit("/", 1)[-1].removesuffix(".pdf")
        if name in fail_ids:
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF-1.4 " + name.encode())

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_and_skip_on_rerun(tmp_path: Path):
    rows = load_manifest_from_text(CSV)
    sleeps: list[float] = []
    r1 = download_all(rows, tmp_path, make_client(set()), delay_s=1.0, sleep=sleeps.append)
    assert r1.ok == ["a", "b"] and r1.failed == []
    assert (tmp_path / "pdf" / "a.pdf").read_bytes().startswith(b"%PDF")
    assert sleeps == [1.0, 1.0]  # her istekten önce nazik bekleme
    r2 = download_all(rows, tmp_path, make_client(set()), delay_s=1.0, sleep=sleeps.append)
    assert r2.skipped == ["a", "b"] and len(sleeps) == 2  # idempotent: yeniden inmez


def test_failure_recorded(tmp_path: Path):
    rows = load_manifest_from_text(CSV)
    r = download_all(rows, tmp_path, make_client({"b"}), delay_s=0, sleep=lambda _: None)
    assert r.ok == ["a"] and r.failed == ["b"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["b"]["status"] == "failed"
```

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/corpus/test_download.py -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: download.py yaz**

```python
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from pydantic import BaseModel

from belge_gozu.corpus.manifest import ManifestRow

USER_AGENT = "belge-gozu/0.1 (acik kaynak arastirma projesi)"  # ASCII: header'lar latin-1/ascii alanı


class DownloadReport(BaseModel):
    ok: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}  # bozuk state: kendini onarır, idempotent yeniden indirme güvenli


def download_all(
    rows: list[ManifestRow],
    out_dir: Path,
    client: httpx.Client,
    delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadReport:
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = _load_state(state_path)
    report = DownloadReport()

    for row in rows:
        target = pdf_dir / f"{row.doc_id}.pdf"
        if state.get(row.doc_id, {}).get("status") == "ok" and target.exists():
            report.skipped.append(row.doc_id)
            continue
        if delay_s:
            sleep(delay_s)
        try:
            resp = client.get(
                row.url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60
            )
            resp.raise_for_status()
            target.write_bytes(resp.content)
            sha = hashlib.sha256(resp.content).hexdigest()
            state[row.doc_id] = {"sha256": sha, "status": "ok"}
            report.ok.append(row.doc_id)
        except (httpx.HTTPError, httpx.InvalidURL, OSError):
            state[row.doc_id] = {"sha256": "", "status": "failed"}
            report.failed.append(row.doc_id)
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        os.replace(tmp, state_path)  # atomik: yarıda kesilme state'i bozamaz
    return report
```

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/corpus -v && make lint`
Expected: hepsi PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: rate-limited idempotent PDF downloader"
```

---

### Task 4: Render (PDF → WebP + meta.parquet)

**Files:**
- Create: `src/belge_gozu/corpus/render.py`, `tests/corpus/test_render.py`

**Interfaces:**
- Consumes: `ManifestRow` listesi + `out_dir` (Task 3'ün indirdiği `pdf/` klasörü)
- Produces: `render_all(rows, corpus_dir: Path, dpi: int = 150) -> pd.DataFrame`; görüntüler `corpus_dir/images/<doc_id>/<page_no:04d>.webp`; meta `corpus_dir/meta.parquet`. Meta kolonları (Plan 2/3 dahil her tüketici için sabit sözleşme): `page_id` (=`f"{doc_id}:{page_no}"`), `doc_id`, `doc_name`, `doc_type`, `source_url`, `page_no` (1-tabanlı int), `image_path` (corpus_dir'e göreli str).

- [ ] **Step 1: Başarısız testi yaz** — `tests/corpus/test_render.py`

```python
from pathlib import Path

import pandas as pd
import pymupdf as fitz

from belge_gozu.corpus.manifest import load_manifest_from_text
from belge_gozu.corpus.render import render_all

CSV = """doc_id,doc_name,doc_type,url
d1,Deneme Belgesi,kanun,https://example.org/d1.pdf
"""


def make_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((50, 50), f"Sayfa {i + 1}")
    doc.save(path)


def test_render(tmp_path: Path):
    (tmp_path / "pdf").mkdir()
    make_pdf(tmp_path / "pdf" / "d1.pdf", pages=3)
    rows = load_manifest_from_text(CSV)
    df = render_all(rows, tmp_path, dpi=72)
    assert len(df) == 3
    assert df.page_id.tolist() == ["d1:1", "d1:2", "d1:3"]
    assert (tmp_path / "images" / "d1" / "0002.webp").exists()
    saved = pd.read_parquet(tmp_path / "meta.parquet")
    assert saved.page_id.tolist() == df.page_id.tolist()


def test_render_skips_missing_pdf(tmp_path: Path):
    (tmp_path / "pdf").mkdir()
    df = render_all(load_manifest_from_text(CSV), tmp_path, dpi=72)
    assert df.empty
```

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/corpus/test_render.py -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: render.py yaz**

```python
from pathlib import Path

import pandas as pd
import pymupdf as fitz

from belge_gozu.corpus.manifest import ManifestRow

META_COLUMNS = ["page_id", "doc_id", "doc_name", "doc_type", "source_url", "page_no", "image_path"]


def render_all(rows: list[ManifestRow], corpus_dir: Path, dpi: int = 150) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows:
        pdf_path = corpus_dir / "pdf" / f"{row.doc_id}.pdf"
        if not pdf_path.exists():
            continue
        img_dir = corpus_dir / "images" / row.doc_id
        img_dir.mkdir(parents=True, exist_ok=True)
        try:
            _render_doc(row, pdf_path, corpus_dir, dpi, records)
        except (fitz.FileDataError, RuntimeError):
            continue  # bozuk PDF: belgeyi atla, parti devam eder (meta yine yazılır)
    df = pd.DataFrame.from_records(records, columns=META_COLUMNS)
    df.to_parquet(corpus_dir / "meta.parquet", index=False)
    return df


def _render_doc(row: ManifestRow, pdf_path: Path, corpus_dir: Path, dpi: int, records: list[dict]) -> None:
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            rel = f"images/{row.doc_id}/{i:04d}.webp"
            out = corpus_dir / rel
            if not out.exists():
                pix = page.get_pixmap(dpi=dpi)
                pix.pil_save(out, format="WEBP", quality=80)
            records.append(
                {
                    "page_id": f"{row.doc_id}:{i}",
                    "doc_id": row.doc_id,
                    "doc_name": row.doc_name,
                    "doc_type": row.doc_type,
                    "source_url": row.url,
                    "page_no": i,
                    "image_path": rel,
                }
            )
```

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/corpus -v && make lint`
Expected: hepsi PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: PDF page renderer with parquet metadata"
```

---

### Task 5: Index store (binarize/pack/save/mmap load)

**Files:**
- Create: `src/belge_gozu/index/__init__.py`, `src/belge_gozu/index/store.py`, `tests/index/test_store.py`

**Interfaces:**
- Produces: `binarize_pack(emb: np.ndarray) -> np.ndarray` (float `(n,128)` → uint8 `(n,16)`; kural: `v > 0` → bit 1, `np.packbits` big-endian varsayılanı), `PackedIndex` dataclass — alanlar `tokens: np.ndarray uint8 (total_tokens,16)`, `offsets: np.ndarray int64 (n_pages+1,)`, `page_vecs: np.ndarray uint8 (n_pages,16)`, `page_ids: list[str]`; metodlar `build(page_ids: list[str], embs: list[np.ndarray]) -> PackedIndex` (classmethod; `page_vecs` = token ortalamasının binarize'ı), `page_tokens(i: int) -> np.ndarray`, `save(dir: Path)`, `load(dir: Path, mmap: bool = True) -> PackedIndex` (classmethod). Dosyalar: `tokens.npy`, `offsets.npy`, `page_vecs.npy`, `page_ids.json`.

- [ ] **Step 1: Başarısız testleri yaz** — `tests/index/test_store.py`

```python
from pathlib import Path

import numpy as np

from belge_gozu.index.store import PackedIndex, binarize_pack


def test_binarize_pack_bits():
    v = np.zeros((1, 128), dtype=np.float32)
    v[0, 0] = 1.0   # ilk bit set → ilk byte 0b10000000
    v[0, 127] = 1.0  # son bit set → son byte 0b00000001
    packed = binarize_pack(v)
    assert packed.shape == (1, 16) and packed.dtype == np.uint8
    assert packed[0, 0] == 0x80 and packed[0, 15] == 0x01


def rand_embs(rng, n_pages: int) -> list[np.ndarray]:
    return [rng.standard_normal((rng.integers(5, 12), 128)).astype(np.float32) for _ in range(n_pages)]


def test_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(7)
    embs = rand_embs(rng, 4)
    idx = PackedIndex.build([f"p{i}" for i in range(4)], embs)
    assert idx.offsets[-1] == sum(e.shape[0] for e in embs)
    idx.save(tmp_path)
    loaded = PackedIndex.load(tmp_path)
    assert loaded.page_ids == idx.page_ids
    np.testing.assert_array_equal(loaded.page_tokens(2), idx.page_tokens(2))
    np.testing.assert_array_equal(loaded.page_vecs, idx.page_vecs)
```

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/index/test_store.py -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: store.py yaz**

```python
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def binarize_pack(emb: np.ndarray) -> np.ndarray:
    if emb.ndim != 2 or emb.shape[1] != 128:
        raise ValueError(f"beklenen (n,128), gelen {emb.shape}")
    return np.packbits((emb > 0).astype(np.uint8), axis=1)


@dataclass
class PackedIndex:
    tokens: np.ndarray
    offsets: np.ndarray
    page_vecs: np.ndarray
    page_ids: list[str]

    @classmethod
    def build(cls, page_ids: list[str], embs: list[np.ndarray]) -> "PackedIndex":
        if len(page_ids) != len(embs):
            raise ValueError(f"page_ids ({len(page_ids)}) ve embs ({len(embs)}) uzunlukları eşleşmiyor")
        if not embs:
            raise ValueError("boş korpus: en az bir sayfa embedding'i gerekli")
        for pid, e in zip(page_ids, embs, strict=True):
            if e.shape[0] == 0:
                raise ValueError(f"sıfır token'lı sayfa: {pid}")
        packed = [binarize_pack(e) for e in embs]
        offsets = np.zeros(len(embs) + 1, dtype=np.int64)
        np.cumsum([p.shape[0] for p in packed], out=offsets[1:])
        page_vecs = np.vstack([binarize_pack(e.mean(axis=0, keepdims=True)) for e in embs])
        return cls(np.vstack(packed), offsets, page_vecs, list(page_ids))

    def page_tokens(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i] : self.offsets[i + 1]]

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / "tokens.npy", self.tokens)
        np.save(dir / "offsets.npy", self.offsets)
        np.save(dir / "page_vecs.npy", self.page_vecs)
        (dir / "page_ids.json").write_text(json.dumps(self.page_ids, ensure_ascii=False))

    @classmethod
    def load(cls, dir: Path, mmap: bool = True) -> "PackedIndex":
        mode = "r" if mmap else None
        return cls(
            tokens=np.load(dir / "tokens.npy", mmap_mode=mode),
            offsets=np.load(dir / "offsets.npy"),
            page_vecs=np.load(dir / "page_vecs.npy"),
            page_ids=json.loads((dir / "page_ids.json").read_text()),
        )
```

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/index -v && make lint`
Expected: hepsi PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: binary packed index store with mmap load"
```

---

### Task 6: Encoder arayüzü (Fake + ColSmol)

**Files:**
- Create: `src/belge_gozu/index/encode.py`, `tests/index/test_encode.py`

**Interfaces:**
- Produces: `Encoder` (Protocol): `encode_pages(images: list[Image.Image]) -> list[np.ndarray]` (her biri float32 `(n_tokens,128)`), `encode_query(text: str) -> np.ndarray` (float32 `(n_q,128)`); `FakeEncoder(tokens_per_item: int = 8)` — girdi baytlarının/metnin sha256'sından tohumlanmış deterministik rastgele embedding; `ColSmolEncoder(model_name: str, device: str = "auto")` — colpali-engine sarmalayıcı, **lazy import** (yalnız `ml` extra kuruluysa), `device="auto"` → mps/cuda/cpu seçimi. `resolve_device(pref: str) -> str` yardımcı fonksiyonu.

- [ ] **Step 1: Başarısız testleri yaz** — `tests/index/test_encode.py`

```python
import numpy as np
from PIL import Image

from belge_gozu.index.encode import FakeEncoder


def make_img(color: int) -> Image.Image:
    return Image.new("RGB", (32, 32), (color, 0, 0))


def test_fake_encoder_shapes_and_determinism():
    enc = FakeEncoder(tokens_per_item=8)
    a1, b1 = enc.encode_pages([make_img(10), make_img(20)])
    a2, _ = enc.encode_pages([make_img(10), make_img(20)])
    assert a1.shape == (8, 128) and a1.dtype == np.float32
    np.testing.assert_array_equal(a1, a2)      # aynı girdi → aynı embedding
    assert not np.array_equal(a1, b1)           # farklı girdi → farklı embedding
    q1 = enc.encode_query("kira artışı")
    q2 = enc.encode_query("kira artışı")
    np.testing.assert_array_equal(q1, q2)
    assert q1.shape == (8, 128)
```

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/index/test_encode.py -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: encode.py yaz**

```python
import hashlib
from typing import Protocol

import numpy as np
from PIL import Image


class Encoder(Protocol):
    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]: ...
    def encode_query(self, text: str) -> np.ndarray: ...


def _seeded(data: bytes, n_tokens: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_tokens, 128)).astype(np.float32)


class FakeEncoder:
    def __init__(self, tokens_per_item: int = 8):
        self.tokens_per_item = tokens_per_item

    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        return [_seeded(img.tobytes(), self.tokens_per_item) for img in images]

    def encode_query(self, text: str) -> np.ndarray:
        return _seeded(text.encode("utf-8"), self.tokens_per_item)


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ColSmolEncoder:
    """colpali-engine sarmalayıcı. Yalnız `ml` extra'sıyla çalışır; API yüzeyi
    Task 13'te gerçek modelle doğrulanır (spec §11 model-eskimesi riski)."""

    def __init__(self, model_name: str, device: str = "auto"):
        import torch
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor

        self.device = resolve_device(device)
        self.model = ColIdefics3.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=self.device
        ).eval()
        self.processor = ColIdefics3Processor.from_pretrained(model_name)

    def _run(self, batch) -> list[np.ndarray]:
        import torch

        with torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        return [e.cpu().float().numpy() for e in out]

    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        for i in range(0, len(images), 4):
            batch = self.processor.process_images(images[i : i + 4])
            results.extend(self._run(batch))
        return results

    def encode_query(self, text: str) -> np.ndarray:
        return self._run(self.processor.process_queries([text]))[0]
```

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/index -v && make lint`
Expected: hepsi PASS (ColSmolEncoder import'u lazy olduğundan torch'suz ortamda da geçer)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: encoder protocol with deterministic fake and ColSmol wrapper"
```

---

### Task 7: Retrieval çekirdeği (Hamming eleme + binary MaxSim)

**Files:**
- Create: `src/belge_gozu/retrieval/__init__.py`, `src/belge_gozu/retrieval/types.py`, `src/belge_gozu/retrieval/core.py`, `tests/retrieval/test_core.py`

**Interfaces:**
- Consumes: `PackedIndex`, `Encoder`, meta `pd.DataFrame` (Task 4 kolon sözleşmesi)
- Produces: `PageHit` (pydantic: `page_id: str`, `score: float`, `doc_name: str`, `page_no: int`, `image_path: str`, `source_url: str`); `TwoStageRetriever(index, meta, encoder)` — `search(query: str, k: int = 5, candidates: int = 200) -> list[PageHit]` ve `search_embedding(q_emb: np.ndarray, k: int, candidates: int) -> list[tuple[int, float]]` (sayfa indeksi + skor; testler ve Plan 2 ablasyonları bunu kullanır). Skor: `Σ_q max_d (128 - 2·hamming)`.

- [ ] **Step 1: Başarısız testleri yaz** — `tests/retrieval/test_core.py`

```python
import numpy as np
import pandas as pd

from belge_gozu.index.store import PackedIndex, binarize_pack
from belge_gozu.retrieval.core import TwoStageRetriever, binary_maxsim


def naive_maxsim(q_bits: np.ndarray, d_bits: np.ndarray) -> float:
    # saf referans: bit dizileri (n,128) 0/1 uint8
    total = 0
    for qrow in q_bits:
        best = -129
        for drow in d_bits:
            ham = int(np.sum(qrow != drow))
            best = max(best, 128 - 2 * ham)
        total += best
    return float(total)


def unpack(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=1)


def test_binary_maxsim_matches_naive():
    rng = np.random.default_rng(3)
    q = binarize_pack(rng.standard_normal((5, 128)).astype(np.float32))
    d = binarize_pack(rng.standard_normal((9, 128)).astype(np.float32))
    assert binary_maxsim(q, d) == naive_maxsim(unpack(q), unpack(d))


def build_fixture(n_pages: int = 30, seed: int = 11):
    rng = np.random.default_rng(seed)
    embs = [rng.standard_normal((8, 128)).astype(np.float32) for _ in range(n_pages)]
    ids = [f"d{i}:1" for i in range(n_pages)]
    idx = PackedIndex.build(ids, embs)
    meta = pd.DataFrame(
        {
            "page_id": ids,
            "doc_id": [f"d{i}" for i in range(n_pages)],
            "doc_name": [f"Belge {i}" for i in range(n_pages)],
            "doc_type": ["kanun"] * n_pages,
            "source_url": ["https://example.org"] * n_pages,
            "page_no": [1] * n_pages,
            "image_path": [f"images/d{i}/0001.webp" for i in range(n_pages)],
        }
    )
    return idx, meta, embs


def test_planted_needle_found():
    idx, meta, embs = build_fixture()
    retriever = TwoStageRetriever(idx, meta, encoder=None)
    hits = retriever.search_embedding(embs[17], k=5, candidates=10)
    assert hits[0][0] == 17  # sorgu = sayfa 17'nin embedding'i → ilk sırada 17
    assert hits[0][1] >= hits[1][1]


def test_stage1_prefilter_respected():
    idx, meta, embs = build_fixture()
    retriever = TwoStageRetriever(idx, meta, encoder=None)
    all_hits = retriever.search_embedding(embs[17], k=30, candidates=30)
    assert len(all_hits) == 30
    few = retriever.search_embedding(embs[17], k=5, candidates=3)
    assert len(few) == 3  # aday sayısı k'den küçükse sonuç aday sayısıyla sınırlı
```

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/retrieval -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: types.py ve core.py yaz**

`types.py`:

```python
from pydantic import BaseModel


class PageHit(BaseModel):
    page_id: str
    score: float
    doc_name: str
    page_no: int
    image_path: str
    source_url: str
```

`core.py`:

```python
import numpy as np
import pandas as pd

from belge_gozu.index.encode import Encoder
from belge_gozu.index.store import PackedIndex, binarize_pack
from belge_gozu.retrieval.types import PageHit


def _as_u64(packed: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(packed).view(np.uint64)  # (n,16) uint8 -> (n,2) uint64


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,16) uint8, b: (m,16) uint8 -> (n,m) int32 Hamming mesafeleri."""
    xa, xb = _as_u64(a), _as_u64(b)
    return np.bitwise_count(xa[:, None, :] ^ xb[None, :, :]).sum(axis=2).astype(np.int32)


def binary_maxsim(q_packed: np.ndarray, d_packed: np.ndarray) -> float:
    sim = 128 - 2 * hamming_matrix(q_packed, d_packed)  # (n_q, n_d)
    return float(sim.max(axis=1).sum())


class TwoStageRetriever:
    def __init__(self, index: PackedIndex, meta: pd.DataFrame, encoder: Encoder | None):
        self.index = index
        self.encoder = encoder
        self.meta = meta.set_index("page_id", drop=False)

    def search_embedding(
        self, q_emb: np.ndarray, k: int, candidates: int
    ) -> list[tuple[int, float]]:
        q_packed = binarize_pack(q_emb)
        q_vec = binarize_pack(q_emb.mean(axis=0, keepdims=True))
        # Aşama 1: sayfa vektörüyle Hamming eleme
        dists = hamming_matrix(q_vec, self.index.page_vecs)[0]
        n_cand = min(candidates, len(dists))
        cand_ids = np.argpartition(dists, n_cand - 1)[:n_cand]
        # Aşama 2: adaylarda kesin binary MaxSim
        scored = [
            (int(i), binary_maxsim(q_packed, self.index.page_tokens(int(i)))) for i in cand_ids
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def search(self, query: str, k: int = 5, candidates: int = 200) -> list[PageHit]:
        if self.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        hits = self.search_embedding(self.encoder.encode_query(query), k, candidates)
        out: list[PageHit] = []
        for i, score in hits:
            row = self.meta.loc[self.index.page_ids[i]]
            out.append(
                PageHit(
                    page_id=row["page_id"],
                    score=score,
                    doc_name=row["doc_name"],
                    page_no=int(row["page_no"]),
                    image_path=row["image_path"],
                    source_url=row["source_url"],
                )
            )
        return out
```

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/retrieval -v && make lint`
Expected: hepsi PASS (naive referansla birebir eşitlik dahil)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: two-stage binary retrieval core (hamming prefilter + maxsim)"
```

---

### Task 8: CLI — corpus/index komutları

**Files:**
- Create: `src/belge_gozu/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 2-7'nin tüm public fonksiyonları + `get_settings()`
- Produces: typer app `belge-gozu` komutları: `corpus download [--manifest PATH]`, `corpus render [--dpi N]`, `corpus probe [--manifest PATH]`, `index build [--fake]` (`--fake` FakeEncoder kullanır — CI/smoke için), `index push`, `index pull`, `serve [--pull]`. `index build`: `meta.parquet`'i okur, görüntüleri encoder'dan geçirir, `PackedIndex.save(settings.index_dir)` + `meta.parquet` kopyası indeks dizinine.

- [ ] **Step 1: Başarısız testi yaz** — `tests/test_cli.py`

```python
from pathlib import Path

import fitz
from typer.testing import CliRunner

from belge_gozu.cli import app

runner = CliRunner()

CSV = """doc_id,doc_name,doc_type,url
d1,Deneme Belgesi,kanun,https://example.org/d1.pdf
"""


def make_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((50, 50), f"Sayfa {i + 1}")
    doc.save(path)


def test_render_and_fake_build(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BG_INDEX_DIR", str(tmp_path / "index"))
    (tmp_path / "manifest").mkdir(parents=True)
    (tmp_path / "manifest" / "v0_manifest.csv").write_text(CSV, encoding="utf-8")
    (tmp_path / "pdf").mkdir()
    make_pdf(tmp_path / "pdf" / "d1.pdf", pages=2)

    r1 = runner.invoke(app, ["corpus", "render", "--dpi", "72"])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["index", "build", "--fake"])
    assert r2.exit_code == 0, r2.output
    assert (tmp_path / "index" / "tokens.npy").exists()
    assert (tmp_path / "index" / "meta.parquet").exists()
```

Not: `get_settings` lru_cache'lidir; CLI komutları env'i taze okusun diye `cli.py`
içinde `Settings()` doğrudan kurulur (aşağıdaki implementasyona bak).

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: cli.py yaz**

```python
import shutil
from pathlib import Path

import httpx
import pandas as pd
import typer
from PIL import Image

from belge_gozu.config import Settings
from belge_gozu.corpus.download import download_all
from belge_gozu.corpus.manifest import load_manifest, probe
from belge_gozu.corpus.render import render_all
from belge_gozu.index.encode import FakeEncoder
from belge_gozu.index.store import PackedIndex

app = typer.Typer(help="Belge-Gözü: Türkçe mevzuat için görsel belge RAG")
corpus_app = typer.Typer()
index_app = typer.Typer()
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")

DEFAULT_MANIFEST = Path("data/manifest/v0_manifest.csv")


def _settings() -> Settings:
    return Settings()


def _manifest_path(s: Settings, manifest: Path | None) -> Path:
    return manifest or (s.data_dir / "manifest" / "v0_manifest.csv")


@corpus_app.command("download")
def corpus_download(manifest: Path | None = typer.Option(None)) -> None:
    s = _settings()
    rows = load_manifest(_manifest_path(s, manifest))
    with httpx.Client() as client:
        report = download_all(rows, s.data_dir, client, delay_s=s.request_delay_s)
    typer.echo(f"ok={len(report.ok)} skipped={len(report.skipped)} failed={report.failed}")


@corpus_app.command("probe")
def corpus_probe(manifest: Path | None = typer.Option(None)) -> None:
    s = _settings()
    rows = load_manifest(_manifest_path(s, manifest))
    with httpx.Client() as client:
        for doc_id, status in probe(rows, client):
            typer.echo(f"{doc_id}\t{status}")


@corpus_app.command("render")
def corpus_render(dpi: int = typer.Option(150)) -> None:
    s = _settings()
    rows = load_manifest(_manifest_path(s, None))
    df = render_all(rows, s.data_dir, dpi=dpi)
    typer.echo(f"{len(df)} sayfa render edildi")


@index_app.command("build")
def index_build(fake: bool = typer.Option(False, "--fake")) -> None:
    s = _settings()
    meta = pd.read_parquet(s.data_dir / "meta.parquet")
    if fake:
        encoder = FakeEncoder()
    else:
        from belge_gozu.index.encode import ColSmolEncoder

        encoder = ColSmolEncoder(s.retriever_model, s.device)
    embs, ids = [], []
    for _, row in meta.iterrows():
        img = Image.open(s.data_dir / row["image_path"]).convert("RGB")
        embs.extend(encoder.encode_pages([img]))
        ids.append(row["page_id"])
    PackedIndex.build(ids, embs).save(s.index_dir)
    shutil.copy(s.data_dir / "meta.parquet", s.index_dir / "meta.parquet")
    typer.echo(f"{len(ids)} sayfa indekslendi -> {s.index_dir}")


@index_app.command("push")
def index_push() -> None:
    from belge_gozu.index.hub import push_index

    s = _settings()
    push_index(s.index_dir, s.hf_dataset_repo)
    typer.echo(f"indeks {s.hf_dataset_repo} reposuna gönderildi")


@index_app.command("pull")
def index_pull() -> None:
    from belge_gozu.index.hub import pull_index

    s = _settings()
    pull_index(s.hf_dataset_repo, s.index_dir)
    typer.echo(f"indeks {s.hf_dataset_repo} reposundan indirildi")


@app.command("serve")
def serve(pull: bool = typer.Option(False, "--pull"), port: int = typer.Option(7860)) -> None:
    import uvicorn

    s = _settings()
    if pull and s.hf_dataset_repo:
        from belge_gozu.index.hub import pull_index

        pull_index(s.hf_dataset_repo, s.index_dir)
    uvicorn.run("belge_gozu.app.main:create_app", factory=True, host="0.0.0.0", port=port)
```

Not: `index build` sayfa sayfa encode eder (v0 korpusu ~2 bin sayfa; Colab/parti
optimizasyonu Plan 2'de). `hub` ve `app.main` importları lazy — Task 9 ve 11'de gelecekler;
bu task'ta `index push/pull` ve `serve` komutları test EDİLMEZ (testte çağrılmıyorlar).

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/test_cli.py -v && make lint`
Expected: PASS (pyright, lazy importlar henüz var olmayan `hub`/`app.main` modüllerine
takılırsa bu iki komutun gövdesini Task 9/11'e kadar `raise typer.Exit(1)` + TODO YERİNE
şu geçici gerçek davranışla bırak: `typer.echo("hub modülü Task 9'da geliyor"); raise typer.Exit(1)`.
Task 9 ve 11 bu gövdeleri yukarıdaki gerçek halleriyle DEĞİŞTİRİR.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: typer CLI for corpus and index pipelines"
```

---

### Task 9: HF Hub senkronizasyonu (push/pull)

**Files:**
- Create: `src/belge_gozu/index/hub.py`, `tests/index/test_hub.py`
- Modify: `src/belge_gozu/cli.py` (Task 8'deki geçici `index push/pull` gövdelerini gerçek çağrılarla değiştir — kod Task 8 Step 3'te verildi)

**Interfaces:**
- Produces: `push_index(index_dir: Path, repo_id: str, api: HfApi | None = None) -> None` (repo yoksa `create_repo(repo_type="dataset", exist_ok=True)` sonra `upload_folder(folder_path=index_dir, repo_id=repo_id, repo_type="dataset", path_in_repo="index")`), `pull_index(repo_id: str, index_dir: Path, api: HfApi | None = None) -> None` (`snapshot_download(repo_id, repo_type="dataset", allow_patterns=["index/*"], local_dir=...)` sonra `index/` içeriğini `index_dir`'e taşır). `repo_id` boşsa `ValueError`.

- [ ] **Step 1: Başarısız testi yaz** — `tests/index/test_hub.py`

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from belge_gozu.index.hub import pull_index, push_index


def test_push_calls_hub(tmp_path: Path):
    (tmp_path / "tokens.npy").write_bytes(b"x")
    api = MagicMock()
    push_index(tmp_path, "user/belge-gozu-index", api=api)
    api.create_repo.assert_called_once_with(
        "user/belge-gozu-index", repo_type="dataset", exist_ok=True
    )
    api.upload_folder.assert_called_once()
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_type"] == "dataset" and kwargs["path_in_repo"] == "index"


def test_push_empty_repo_id_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        push_index(tmp_path, "")


def test_pull_moves_files(tmp_path: Path):
    api = MagicMock()

    def fake_snapshot(**kwargs):
        d = Path(kwargs["local_dir"]) / "index"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tokens.npy").write_bytes(b"x")
        return str(kwargs["local_dir"])

    api.snapshot_download.side_effect = fake_snapshot
    out = tmp_path / "idx"
    pull_index("user/belge-gozu-index", out, api=api)
    assert (out / "tokens.npy").exists()
```

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/index/test_hub.py -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: hub.py yaz**

```python
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def _api(api: HfApi | None) -> HfApi:
    return api or HfApi()


def push_index(index_dir: Path, repo_id: str, api: HfApi | None = None) -> None:
    if not repo_id:
        raise ValueError("BG_HF_DATASET_REPO ayarlı değil")
    a = _api(api)
    a.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    a.upload_folder(
        folder_path=str(index_dir), repo_id=repo_id, repo_type="dataset", path_in_repo="index"
    )


def pull_index(repo_id: str, index_dir: Path, api: HfApi | None = None) -> None:
    if not repo_id:
        raise ValueError("BG_HF_DATASET_REPO ayarlı değil")
    a = _api(api)
    with tempfile.TemporaryDirectory() as tmp:
        a.snapshot_download(
            repo_id=repo_id, repo_type="dataset", allow_patterns=["index/*"], local_dir=tmp
        )
        src = Path(tmp) / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy(f, index_dir / f.name)
```

- [ ] **Step 4: Testlerin geçtiğini gör + CLI gövdelerini gerçekle**

Run: `uv run pytest tests/index -v && uv run pytest tests/test_cli.py -v && make lint`
Expected: hepsi PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: HF dataset hub sync for index artifacts"
```

---

### Task 10: Answerer katmanı (Gemini + abstain servisi)

**Files:**
- Create: `src/belge_gozu/answer/__init__.py`, `src/belge_gozu/answer/base.py`, `src/belge_gozu/answer/gemini.py`, `tests/answer/test_base.py`, `tests/answer/test_gemini.py`

**Interfaces:**
- Consumes: `PageHit`, `Settings`
- Produces: `Answer` (pydantic: `text: str`, `citations: list[str]`, `abstained: bool = False`); `Answerer` Protocol: `answer(question: str, pages: list[PageHit], image_loader: Callable[[str], bytes]) -> Answer`; `AskService(retriever, answerer, min_score: float, image_loader)` — `ask(question: str, k: int, candidates: int) -> tuple[Answer, list[PageHit]]`: en iyi skor `min_score * n_query_token_yaklaşığı` DEĞİL — skor zaten toplamdır; normalizasyon: `top_score / q_token_sayısı`; v0'da `PageHit.score`'u sorgu token sayısına bölmek için retriever normalizasyonu YOK, bunun yerine eşik ham skora `min_score * 8` (FakeEncoder token sayısı) gibi kırılgan olurdu → NET KURAL: `TwoStageRetriever.search` skorları sorgu token sayısına bölerek NORMALİZE döndürür (bkz. Step 3'teki küçük değişiklik). Eşik: `top_hit.score < min_score` → `Answer(text="Bu soruya korpustaki belgelerde dayanak bulamadım.", citations=[], abstained=True)`; `GeminiAnswerer(model: str, api_key: str, client=None)` — [S1],[S2] atıf formatlı Türkçe prompt, yanıttan `\[S(\d+)\]` regex'iyle citation parse; hiç atıf yoksa `citations=[pages[0].page_id]`.

- [ ] **Step 1: Başarısız testleri yaz**

`tests/answer/test_base.py`:

```python
from belge_gozu.answer.base import Answer, AskService
from belge_gozu.retrieval.types import PageHit


def hit(pid: str, score: float) -> PageHit:
    return PageHit(
        page_id=pid, score=score, doc_name="Belge", page_no=1,
        image_path=f"images/{pid}.webp", source_url="https://example.org",
    )


class FakeRetriever:
    def __init__(self, hits): self._hits = hits
    def search(self, query, k=5, candidates=200): return self._hits[:k]


class EchoAnswerer:
    def answer(self, question, pages, image_loader):
        return Answer(text="cevap", citations=[pages[0].page_id])


def test_ask_answers_above_threshold():
    svc = AskService(FakeRetriever([hit("a:1", 90.0)]), EchoAnswerer(),
                     min_score=20.0, image_loader=lambda p: b"img")
    answer, hits = svc.ask("soru", k=5, candidates=200)
    assert not answer.abstained and answer.citations == ["a:1"] and hits[0].page_id == "a:1"


def test_ask_abstains_below_threshold():
    svc = AskService(FakeRetriever([hit("a:1", 5.0)]), EchoAnswerer(),
                     min_score=20.0, image_loader=lambda p: b"img")
    answer, _ = svc.ask("soru", k=5, candidates=200)
    assert answer.abstained and answer.citations == []


def test_ask_abstains_on_empty_index():
    svc = AskService(FakeRetriever([]), EchoAnswerer(), min_score=20.0,
                     image_loader=lambda p: b"img")
    answer, hits = svc.ask("soru", k=5, candidates=200)
    assert answer.abstained and hits == []
```

`tests/answer/test_gemini.py`:

```python
from unittest.mock import MagicMock

from belge_gozu.answer.gemini import GeminiAnswerer, build_prompt
from belge_gozu.retrieval.types import PageHit


def hit(pid: str) -> PageHit:
    return PageHit(page_id=pid, score=50.0, doc_name="Türk Borçlar Kanunu", page_no=12,
                   image_path=f"images/{pid}.webp", source_url="https://example.org")


def test_prompt_mentions_sources():
    p = build_prompt("kira artışı sınırı nedir?", [hit("k6098:12"), hit("k6098:13")])
    assert "[S1]" in p and "[S2]" in p and "kira artışı" in p
    assert "Türk Borçlar Kanunu" in p and "sayfa 12" in p


def test_citations_parsed_from_response():
    client = MagicMock()
    client.generate.return_value = "Kira artışı TÜFE ile sınırlıdır [S2]."
    ans = GeminiAnswerer("gemini-2.0-flash", "key", client=client)
    a = ans.answer("soru", [hit("k6098:12"), hit("k6098:13")], image_loader=lambda p: b"img")
    assert a.citations == ["k6098:13"] and not a.abstained


def test_citation_fallback_top1():
    client = MagicMock()
    client.generate.return_value = "Atıfsız bir yanıt."
    ans = GeminiAnswerer("gemini-2.0-flash", "key", client=client)
    a = ans.answer("soru", [hit("k6098:12")], image_loader=lambda p: b"img")
    assert a.citations == ["k6098:12"]
```

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/answer -v`
Expected: FAIL (modüller yok)

- [ ] **Step 3: base.py, gemini.py yaz + retriever normalizasyonu**

`base.py`:

```python
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from belge_gozu.retrieval.types import PageHit

ABSTAIN_TEXT = "Bu soruya korpustaki belgelerde dayanak bulamadım."


class Answer(BaseModel):
    text: str
    citations: list[str]
    abstained: bool = False


class Answerer(Protocol):
    def answer(
        self, question: str, pages: list[PageHit], image_loader: Callable[[str], bytes]
    ) -> Answer: ...


class AskService:
    def __init__(self, retriever, answerer: Answerer, min_score: float,
                 image_loader: Callable[[str], bytes]):
        self.retriever = retriever
        self.answerer = answerer
        self.min_score = min_score
        self.image_loader = image_loader

    def ask(self, question: str, k: int, candidates: int) -> tuple[Answer, list[PageHit]]:
        hits = self.retriever.search(question, k=k, candidates=candidates)
        if not hits or hits[0].score < self.min_score:
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True), hits
        return self.answerer.answer(question, hits, self.image_loader), hits
```

`gemini.py`:

```python
import re
from collections.abc import Callable

from belge_gozu.answer.base import Answer
from belge_gozu.retrieval.types import PageHit

SYSTEM = (
    "Sen Türk mevzuatı üzerine bir asistansın. YALNIZCA sana verilen sayfa "
    "görüntülerindeki bilgiye dayanarak Türkçe yanıt ver. Her iddianın sonuna "
    "dayandığı kaynağı [S1] gibi işaretle. Sayfalarda yanıt yoksa açıkça "
    "'verilen sayfalarda bulamadım' de. Sayfa dışı bilgi ekleme."
)


def build_prompt(question: str, pages: list[PageHit]) -> str:
    src_lines = [
        f"[S{i + 1}] {p.doc_name}, sayfa {p.page_no}" for i, p in enumerate(pages)
    ]
    return f"{SYSTEM}\n\nKaynaklar:\n" + "\n".join(src_lines) + f"\n\nSoru: {question}"


class GeminiClient:
    """google-genai ince sarmalayıcısı. SDK yüzeyi Task 13'te canlı doğrulanır."""

    def __init__(self, model: str, api_key: str):
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, images: list[bytes]) -> str:
        from google.genai import types

        parts = [types.Part.from_bytes(data=b, mime_type="image/webp") for b in images]
        resp = self.client.models.generate_content(
            model=self.model, contents=[*parts, prompt]
        )
        return resp.text or ""


class GeminiAnswerer:
    def __init__(self, model: str, api_key: str, client=None):
        self._client = client or GeminiClient(model, api_key)

    def answer(
        self, question: str, pages: list[PageHit], image_loader: Callable[[str], bytes]
    ) -> Answer:
        prompt = build_prompt(question, pages)
        images = [image_loader(p.image_path) for p in pages]
        text = self._client.generate(prompt, images)
        idxs = {int(m) for m in re.findall(r"\[S(\d+)\]", text)}
        citations = [pages[i - 1].page_id for i in sorted(idxs) if 0 < i <= len(pages)]
        if not citations and pages:
            citations = [pages[0].page_id]
        return Answer(text=text, citations=citations)
```

`retrieval/core.py`'de normalizasyon değişikliği — `search` içinde `PageHit(score=...)`
satırını şu şekilde güncelle (skor sorgu token sayısına bölünür; `search_embedding` HAM kalır):

```python
        q_emb = self.encoder.encode_query(query)
        hits = self.search_embedding(q_emb, k, candidates)
        n_q = max(1, q_emb.shape[0])
        ...
                    score=score / n_q,
```

Ve `tests/retrieval/test_core.py`'ye normalizasyon testi ekle:

```python
def test_search_scores_normalized_per_query_token():
    idx, meta, embs = build_fixture()

    class OneTokenEncoder:
        def encode_pages(self, images): raise NotImplementedError
        def encode_query(self, text):
            return embs[17][:1]  # tek token → normalize skor = ham skor

    r = TwoStageRetriever(idx, meta, OneTokenEncoder())
    hits = r.search("x", k=1, candidates=30)
    raw = r.search_embedding(embs[17][:1], k=1, candidates=30)
    assert hits[0].score == raw[0][1] / 1
```

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/answer tests/retrieval -v && make lint`
Expected: hepsi PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: pluggable answerer with Gemini impl and abstain service"
```

---

### Task 11: FastAPI uygulaması + istek logu + /stats

**Files:**
- Create: `src/belge_gozu/app/__init__.py`, `src/belge_gozu/app/main.py`, `tests/app/test_api.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `TwoStageRetriever`, `AskService`, `GeminiAnswerer`, `FakeEncoder`, `PackedIndex.load`, `Settings`
- Produces: `create_app(settings: Settings | None = None, retriever=None, answerer=None) -> FastAPI` (test enjeksiyonu için parametreli). Uçlar: `GET /healthz` → `{"status":"ok","pages":N}`; `POST /search` gövde `{"query": str, "k": int|None}` → `{"hits":[PageHit...]}`; `POST /ask` gövde `{"question": str}` → `{"answer": Answer, "hits":[PageHit...]}`; `GET /stats` → `{"requests": N, "avg_ms": float}`; `GET /pages/<image_path>` statik sayfa görüntüleri; `GET /` → `index.html`. İstek logu: `settings.data_dir/requests.sqlite`, tablo `log(ts TEXT, path TEXT, ms REAL, top_score REAL)`.

- [ ] **Step 1: conftest fixture'ı ve başarısız testleri yaz**

`tests/conftest.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from belge_gozu.index.encode import FakeEncoder
from belge_gozu.index.store import PackedIndex


@pytest.fixture
def tiny_corpus(tmp_path: Path):
    """3 sayfalık sahte korpus: görüntüler + meta.parquet + FakeEncoder indeksi."""
    enc = FakeEncoder()
    images, ids, records = [], [], []
    for i in range(3):
        rel = f"images/d{i}/0001.webp"
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (32, 32), (i * 40, 10, 10))
        img.save(p, format="WEBP")
        images.append(img)
        ids.append(f"d{i}:1")
        records.append(
            {"page_id": f"d{i}:1", "doc_id": f"d{i}", "doc_name": f"Belge {i}",
             "doc_type": "kanun", "source_url": "https://example.org",
             "page_no": 1, "image_path": rel}
        )
    meta = pd.DataFrame.from_records(records)
    meta.to_parquet(tmp_path / "meta.parquet", index=False)
    idx = PackedIndex.build(ids, enc.encode_pages(images))
    idx_dir = tmp_path / "index"
    idx.save(idx_dir)
    meta.to_parquet(idx_dir / "meta.parquet", index=False)
    return tmp_path, enc, np.array([])  # (data_dir, encoder, _)
```

`tests/app/test_api.py`:

```python
from fastapi.testclient import TestClient

from belge_gozu.answer.base import Answer
from belge_gozu.app.main import create_app
from belge_gozu.config import Settings


class StubAnswerer:
    def answer(self, question, pages, image_loader):
        return Answer(text=f"yanıt: {question}", citations=[pages[0].page_id])


def make_client(tiny_corpus) -> TestClient:
    data_dir, enc, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=-1e9)
    app = create_app(settings=settings, encoder=enc, answerer=StubAnswerer())
    return TestClient(app)


def test_healthz(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "pages": 3}


def test_search_returns_hits(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.post("/search", json={"query": "deneme sorgusu"})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 3 and {"page_id", "score", "image_path"} <= hits[0].keys()


def test_ask_returns_answer_and_logs(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.post("/ask", json={"question": "kira artışı nedir?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"]["text"].startswith("yanıt:") and body["answer"]["citations"]
    stats = c.get("/stats").json()
    assert stats["requests"] >= 1 and stats["avg_ms"] >= 0


def test_page_image_served(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.get("/pages/images/d0/0001.webp")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"


def test_root_serves_ui(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.get("/")
    assert r.status_code == 200 and "Belge-Gözü" in r.text
```

- [ ] **Step 2: Testin başarısız olduğunu gör**

Run: `uv run pytest tests/app -v`
Expected: FAIL (modül yok)

- [ ] **Step 3: app/main.py yaz** (statik UI Task 12'de gelecek; şimdilik `index.html`
yoksa `/` inline yer tutucu HTML döndürür — içinde "Belge-Gözü" başlığı geçer)

```python
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from belge_gozu.answer.base import Answer, AskService
from belge_gozu.config import Settings, get_settings
from belge_gozu.index.store import PackedIndex
from belge_gozu.retrieval.core import TwoStageRetriever
from belge_gozu.retrieval.types import PageHit

STATIC_DIR = Path(__file__).parent / "static"


class SearchBody(BaseModel):
    query: str
    k: int | None = None


class AskBody(BaseModel):
    question: str


def _log_db(settings: Settings) -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(settings.data_dir / "requests.sqlite", check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS log (ts TEXT, path TEXT, ms REAL, top_score REAL)")
    return db


def create_app(settings: Settings | None = None, encoder=None, answerer=None) -> FastAPI:
    s = settings or get_settings()
    index = PackedIndex.load(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    if encoder is None:
        from belge_gozu.index.encode import ColSmolEncoder

        encoder = ColSmolEncoder(s.retriever_model, s.device)
    if answerer is None:
        from belge_gozu.answer.gemini import GeminiAnswerer

        answerer = GeminiAnswerer(s.gemini_model, s.gemini_api_key)

    def load_image(image_path: str) -> bytes:
        return (s.data_dir / image_path).read_bytes()

    retriever = TwoStageRetriever(index, meta, encoder)
    service = AskService(retriever, answerer, s.min_score_threshold, load_image)
    db = _log_db(s)
    app = FastAPI(title="Belge-Gözü")

    def log(path: str, ms: float, top_score: float) -> None:
        db.execute(
            "INSERT INTO log VALUES (?,?,?,?)",
            (datetime.now(UTC).isoformat(), path, ms, top_score),
        )
        db.commit()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "pages": len(index.page_ids)}

    @app.post("/search")
    def search(body: SearchBody) -> dict[str, list[PageHit]]:
        t0 = time.perf_counter()
        hits = retriever.search(body.query, k=body.k or s.top_k, candidates=s.stage1_candidates)
        log("/search", (time.perf_counter() - t0) * 1000, hits[0].score if hits else 0.0)
        return {"hits": hits}

    @app.post("/ask")
    def ask(body: AskBody) -> dict:
        t0 = time.perf_counter()
        answer, hits = service.ask(body.question, k=s.top_k, candidates=s.stage1_candidates)
        log("/ask", (time.perf_counter() - t0) * 1000, hits[0].score if hits else 0.0)
        return {"answer": answer.model_dump(), "hits": [h.model_dump() for h in hits]}

    @app.get("/stats")
    def stats() -> dict:
        row = db.execute("SELECT COUNT(*), COALESCE(AVG(ms),0) FROM log").fetchone()
        return {"requests": row[0], "avg_ms": round(row[1], 1)}

    @app.get("/pages/{image_path:path}")
    def page_image(image_path: str) -> FileResponse:
        full = (s.data_dir / image_path).resolve()
        if not full.is_relative_to(s.data_dir.resolve()) or not full.exists():
            raise HTTPException(404)
        return FileResponse(full, media_type="image/webp")

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        page = STATIC_DIR / "index.html"
        if page.exists():
            return page.read_text(encoding="utf-8")
        return "<html><body><h1>Belge-Gözü</h1><p>UI yakında.</p></body></html>"

    return app
```

Not: `Answer` importu `/ask` dönüş tipi için değil, tip denetimi netliği için değilse
kullanılmıyor — pyright uyarısı verirse importu kaldır (`answer.model_dump()` yeter).

- [ ] **Step 4: Testlerin geçtiğini gör**

Run: `uv run pytest tests/app -v && uv run pytest -m "not slow" -q && make lint`
Expected: hepsi PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: FastAPI service with request log, stats and image serving"
```

---

### Task 12: Tek sayfa UI

**Files:**
- Create: `src/belge_gozu/app/static/index.html`
- Modify: `tests/app/test_api.py` (root testi statik dosyadan gelen gerçek başlığı da doğrular)

**Interfaces:**
- Consumes: `POST /search`, `POST /ask`, `GET /pages/...` sözleşmeleri (Task 11)

- [ ] **Step 1: Testi güncelle** — `test_root_serves_ui` fonksiyonunu şu hale getir:

```python
def test_root_serves_ui(tiny_corpus):
    c = make_client(tiny_corpus)
    r = c.get("/")
    assert r.status_code == 200
    assert "Belge-Gözü" in r.text
    assert 'id="q"' in r.text and 'id="ask-btn"' in r.text  # gerçek UI yüklendi
```

Run: `uv run pytest tests/app/test_api.py::test_root_serves_ui -v`
Expected: FAIL (statik dosya henüz yok, yer tutucuda `id="q"` yok)

- [ ] **Step 2: index.html yaz**

```html
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Belge-Gözü</title>
<style>
  :root { --ink:#17202B; --muted:#56637A; --line:#D8DFE9; --accent:#1E44B0; --bg:#F5F7FA; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 880px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 24px; font-size: .95rem; }
  .bar { display:flex; gap:8px; }
  input { flex:1; padding:12px 14px; font-size:1rem; border:1px solid var(--line);
          border-radius:8px; background:#fff; color:var(--ink); }
  button { padding:12px 18px; font-size:1rem; border:0; border-radius:8px;
           background:var(--accent); color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
  #answer { margin-top:24px; padding:18px 20px; background:#fff; border:1px solid var(--line);
            border-radius:10px; white-space:pre-wrap; display:none; }
  #answer.abstained { border-left:4px solid #B5402A; }
  .cites { margin-top:10px; font-size:.85rem; color:var(--muted); }
  #hits { margin-top:20px; display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
          gap:12px; }
  .hit { background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .hit img { width:100%; display:block; }
  .hit .cap { padding:6px 8px; font-size:.78rem; color:var(--muted); }
  .hit .cap b { color:var(--ink); display:block; }
  #status { margin-top:16px; color:var(--muted); font-size:.9rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Belge-Gözü</h1>
  <p class="sub">Türk mevzuatında görsel arama — sayfalar OCR'sız, görüntü olarak indekslendi.
     Soru sor, dayanak sayfalarıyla yanıt al.</p>
  <div class="bar">
    <input id="q" placeholder="ör. Kira artışı en fazla ne kadar olabilir?" autofocus>
    <button id="ask-btn">Sor</button>
  </div>
  <div id="status"></div>
  <div id="answer"></div>
  <div id="hits"></div>
</div>
<script>
const $ = (id) => document.getElementById(id);
async function ask() {
  const q = $("q").value.trim();
  if (!q) return;
  $("ask-btn").disabled = true;
  $("status").textContent = "Aranıyor ve yanıtlanıyor…";
  $("answer").style.display = "none";
  $("hits").innerHTML = "";
  try {
    const r = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    const a = $("answer");
    a.textContent = data.answer.text;
    a.className = data.answer.abstained ? "abstained" : "";
    if (data.answer.citations.length) {
      const c = document.createElement("div");
      c.className = "cites";
      c.textContent = "Dayanak: " + data.answer.citations.join(", ");
      a.appendChild(c);
    }
    a.style.display = "block";
    for (const h of data.hits) {
      const d = document.createElement("div");
      d.className = "hit";
      d.innerHTML = `<img src="/pages/${h.image_path}" alt="${h.page_id}">
        <div class="cap"><b>${h.doc_name}</b>sayfa ${h.page_no} · skor ${h.score.toFixed(1)}</div>`;
      $("hits").appendChild(d);
    }
    $("status").textContent = "";
  } catch (e) {
    $("status").textContent = "Hata: " + e.message + " — tekrar dene.";
  } finally {
    $("ask-btn").disabled = false;
  }
}
$("ask-btn").addEventListener("click", ask);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
</script>
</body>
</html>
```

- [ ] **Step 3: Testlerin geçtiğini gör**

Run: `uv run pytest tests/app -v && make lint`
Expected: hepsi PASS

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: single-page search and ask UI"
```

---

### Task 13: Uçtan uca v0 koşusu + Dockerfile + deploy (runbook)

Bu task kod + operasyon karışığıdır; adımlar sırayla uygulanır, her doğrulama çıktısı
görülmeden ilerlenmez. Kullanıcının HF hesabı ve Gemini API anahtarı gerekir
(anahtarlar YALNIZ env/Space secret olarak — asla commit'e girmez).

**Files:**
- Create: `Dockerfile`
- Modify: `README.md` (canlı link + kullanım), `data/manifest/v0_manifest.csv` (probe sonuçlarına göre düzeltme + 50 kanuna genişletme)

- [ ] **Step 1: Dockerfile yaz**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache ".[ml]"
ENV BG_DEVICE=cpu
EXPOSE 7860
CMD ["belge-gozu", "serve", "--pull", "--port", "7860"]
```

Commit: `git add Dockerfile && git commit -m "feat: HF Space dockerfile"`

- [ ] **Step 2: Manifest'i canlı doğrula ve genişlet**

Run: `uv run belge-gozu corpus probe`
- 200 dönmeyen satırları düzelt (URL desenini tarayıcıda elle kontrol ederek).
- Kanun listesini mevzuat.gov.tr'den 50'ye genişlet (aynı `1.5.<no>.pdf` deseni),
  tarihî RG satırlarını çalışan arşiv URL'leriyle 100+ sayfayı kapsayacak şekilde güncelle.
- `uv run belge-gozu corpus probe` tamamı 200 olana dek tekrarla.
Commit: `git add data/ && git commit -m "data: verified v0 manifest (50 kanun + historic RG)"`

- [ ] **Step 3: Modeli ve SDK'yı doğrula (spec §11)**

- `uv sync --all-extras` ile ml bağımlılıklarını kur.
- HF'te `vidore/colSmol-500M` hâlâ güncel/önerilen küçük checkpoint mı bak
  (colpali-engine README'sindeki model tablosu). Değiştiyse YALNIZ `config.py`
  varsayılanını ve bu adımın notunu güncelle.
- Python REPL'de 1 görüntü + 1 sorgu encode et; `(n,128)` float çıktıyı doğrula.
  `ColIdefics3` sınıf adı değiştiyse `encode.py` sarmalayıcısını gerçek API'ye uyarla
  (arayüz İMZALARI sabit kalır).
- `google-genai` ile tek istek at (`GeminiClient.generate`), yanıt metni geldiğini gör.
Commit (değişiklik olduysa): `git commit -am "chore: pin verified model/SDK surfaces"`

- [ ] **Step 4: v0 boru hattını koş**

```bash
uv run belge-gozu corpus download        # ~50 kanun + tarihî RG; rate-limit'li, uzun sürebilir
uv run belge-gozu corpus render
uv run belge-gozu index build            # M4 Pro'da MPS; ~2 bin sayfa için saatler sürebilir
```

Doğrulama: `python -c "import pandas as pd; print(len(pd.read_parquet('data/index/meta.parquet')))"`
≈ 2000 ± ve `data/index/tokens.npy` mevcut.

- [ ] **Step 5: Lokal duman testi**

`BG_GEMINI_API_KEY=<anahtar> uv run belge-gozu serve` → tarayıcıda `http://localhost:7860`:
- "Kira artışı en fazla ne kadar olabilir?" sor → yanıt + TBK sayfaları geliyor mu?
- Anlamsız bir soru sor ("mor fil tarifi") → abstain mesajı geliyor mu? Gelmiyorsa
  `BG_MIN_SCORE_THRESHOLD`'u logdaki skorlara bakarak kaba ayarla (gerçek kalibrasyon Plan 2).
- `/stats` istek sayacı artıyor mu?

- [ ] **Step 6: İndeksi HF'e it, Space'i kur, canlıyı doğrula**

```bash
export BG_HF_DATASET_REPO=<hf-kullanici>/belge-gozu-index
huggingface-cli login
uv run belge-gozu index push
```

- huggingface.co'da yeni Space: Docker SDK, ücretsiz CPU, repo'yu bağla
  (`git remote add space ...` + push, veya HF web arayüzünden).
- Space Variables/Secrets: `BG_HF_DATASET_REPO` (variable), `BG_GEMINI_API_KEY` (secret),
  `BG_DEVICE=cpu`.
- Build loglarını izle; `https://<space-url>/healthz` `{"status":"ok",...}` dönene kadar bekle.
- Canlıda Step 5'teki iki soruyu tekrarla; gecikmeyi not et (README'ye yazılacak).

- [ ] **Step 7: README'yi tamamla ve v0'ı etiketle**

README'ye ekle: canlı demo linki + "ilk açılış ~1 dk sürebilir (free tier uyur)" notu,
mimari özeti (offline/online akış), hızlı başlangıç (`make setup`, `belge-gozu --help`),
FSEK m.31 veri notu, "v0 sınırları: 2 bin sayfa, kalibre edilmemiş eşik, tek mod —
benchmark ve rerank Plan 2'de" dürüstlük paragrafı.

```bash
git add -A && git commit -m "docs: v0 README with live demo link" && git tag v0
```

---

## Self-Review (yazar kontrolü — tamamlandı)

1. **Spec kapsaması:** Hafta-1 spec kalemleri ↔ tasklar: iskelet+CI (T1), scraper (T2-3),
   render (T4), embedding+indeks (T5-6-8), HF Datasets (T9), retrieval çekirdeği+testler (T5-7),
   FreeAPI answerer (T10), UI (T12), Space deploy + canlı link (T13). Abstain davranışı (spec §6)
   T10'da; izleme-lite + /stats (spec §8) T11'de. Hafta-2/3 kalemleri (benchmark, rerank,
   agentic, LocalVLM, tam korpus) bilinçli olarak Plan 2-3'te — bu plan kendi başına çalışan
   yazılım üretir.
2. **Yer tutucu taraması:** "TBD/TODO/sonra doldur" yok. Task 8'deki geçici gövde bile somut
   davranış tanımlıyor ve Task 9'da gerçekleniyor. Task 13 runbook adımları doğrulama
   çıktılarıyla tanımlı.
3. **Tip tutarlılığı:** `PageHit` alanları T7↔T10↔T11 testlerinde birebir; `Answer(text,
   citations, abstained)` T10↔T11; `PackedIndex.build/save/load/page_tokens` T5↔T7↔T8↔T11;
   `Encoder.encode_pages/encode_query` T6↔T8↔T11; meta kolon sözleşmesi T4'te tanımlı,
   T7/T11 fixture'ları aynı kolonları kullanıyor. `search_embedding` ham skor, `search`
   normalize skor — T10 Step 3'te test edildi.
