import json
import os
from pathlib import Path

import pymupdf as fitz
import pytest
from typer.testing import CliRunner

from belge_gozu.cli import _load_bench_mode, app
from tests.bench_question_factory import q_dict as _bench_q

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


def test_index_push_prints_sha_and_passes_credentials(tmp_path: Path, monkeypatch):
    calls: dict[str, object] = {}

    def fake_push(index_dir, repo_id, **kwargs):
        calls.update(index_dir=index_dir, repo_id=repo_id, **kwargs)
        return "a" * 40

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BG_HF_DATASET_REPO", "user/repo")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr("belge_gozu.index.hub.push_index", fake_push)

    result = runner.invoke(app, ["index", "push", "--revision", "main", "--no-images"])

    assert result.exit_code == 0, result.output
    assert "a" * 40 in result.output
    assert calls["repo_id"] == "user/repo"
    assert calls["revision"] == "main"
    assert calls["token"] == "secret-token"
    assert calls["images_dir"] is None


def test_index_pull_requires_repo_and_revision(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BG_HF_DATASET_REPO", raising=False)
    monkeypatch.delenv("BG_HF_REVISION", raising=False)

    missing_repo = runner.invoke(app, ["index", "pull"])
    assert missing_repo.exit_code != 0
    assert "BG_HF_DATASET_REPO" in missing_repo.output

    monkeypatch.setenv("BG_HF_DATASET_REPO", "user/repo")
    missing_revision = runner.invoke(app, ["index", "pull"])
    assert missing_revision.exit_code != 0
    assert "BG_HF_REVISION" in missing_revision.output


def test_serve_pull_requires_repo_and_revision_before_uvicorn(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BG_HF_DATASET_REPO", raising=False)
    monkeypatch.delenv("BG_HF_REVISION", raising=False)

    missing_repo = runner.invoke(app, ["serve", "--pull"])
    assert missing_repo.exit_code != 0
    assert "BG_HF_DATASET_REPO" in missing_repo.output

    monkeypatch.setenv("BG_HF_DATASET_REPO", "user/repo")
    missing_revision = runner.invoke(app, ["serve", "--pull"])
    assert missing_revision.exit_code != 0
    assert "BG_HF_REVISION" in missing_revision.output


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


def test_fake_build_multichunk_alignment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BG_INDEX_DIR", str(tmp_path / "index"))
    (tmp_path / "manifest").mkdir(parents=True)
    csv = "doc_id,doc_name,doc_type,url\n" + "\n".join(
        f"d{i},Belge {i},kanun,https://example.org/d{i}.pdf" for i in range(3)
    )
    (tmp_path / "manifest" / "v0_manifest.csv").write_text(csv, encoding="utf-8")
    (tmp_path / "pdf").mkdir()
    for i in range(3):
        make_pdf(tmp_path / "pdf" / f"d{i}.pdf", pages=7)
    assert runner.invoke(app, ["corpus", "render", "--dpi", "72"]).exit_code == 0
    assert runner.invoke(app, ["index", "build", "--fake"]).exit_code == 0

    import numpy as np
    import pandas as pd
    from PIL import Image

    from belge_gozu.index.encode import FakeEncoder
    from belge_gozu.index.store import PackedIndex, binarize_pack

    idx = PackedIndex.load(tmp_path / "index", mmap=False)
    meta = pd.read_parquet(tmp_path / "index" / "meta.parquet")
    assert idx.page_ids == meta.page_id.tolist()  # sıra birebir
    enc = FakeEncoder()
    # rastgele 3 sayfanın embedding'i, bağımsız yeniden-encode ile birebir aynı mı?
    for pos in (0, 10, 20):
        img = Image.open(tmp_path / meta.iloc[pos]["image_path"]).convert("RGB")
        expected = binarize_pack(enc.encode_pages([img])[0])
        np.testing.assert_array_equal(idx.page_tokens(pos), expected)


def test_index_build_manifest_passes_compat_check(tmp_path: Path, monkeypatch):
    from belge_gozu.config import Settings
    from belge_gozu.index.compat import check_compatibility
    from belge_gozu.index.manifest import read_manifest

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

    index_dir = tmp_path / "index"
    manifest = read_manifest(index_dir)
    assert manifest is not None

    s = Settings()
    # Final review CRITICAL-1: beklenen format LITERAL değil config'ten okunur.
    # Sabit "cpe-0.3.18" yazılıydı ve CLI varsayılanı da sabitti; ikisi birlikte
    # Settings'ten sürüklendiği için bu test sürüklenmeyi göremiyordu.
    problems = check_compatibility(
        manifest,
        model_name=s.retriever_model,
        model_revision=None,
        query_format_id=s.query_format_id,
        index_dir=index_dir,
    )
    assert problems == []
    assert manifest.query_format.format_id == s.query_format_id


def test_index_build_option_defaults_come_from_settings():
    """Final review CRITICAL-1: `--query-format`/`--doc-prompt` varsayılanları
    Settings'ten gelmeli. Sabit literal'ken serve config'i train-compat'e
    geçtiğinde sürüklendiler ve belgelenmiş `index build` çağrısı üretim
    indeksini KAYBEDEN formatla ezecek hale geldi."""
    from belge_gozu.cli import DEFAULT_DOC_PROMPT, DEFAULT_QUERY_FORMAT
    from belge_gozu.config import Settings

    s = Settings()
    assert DEFAULT_QUERY_FORMAT.value == s.query_format_id
    assert DEFAULT_DOC_PROMPT.value == s.doc_prompt_id

    result = runner.invoke(app, ["index", "build", "--help"])
    assert result.exit_code == 0, result.output
    unwrapped = "".join(result.output.split())  # help metni sarmalanabilir
    assert s.query_format_id in unwrapped and s.doc_prompt_id in unwrapped


def test_index_build_refuses_to_overwrite_prod_index_with_other_format(tmp_path, monkeypatch):
    """--out verilmediğinde hedef üretim indeksidir; serve config'inden sapan
    bir format/prompt ile o dizin ezilemez (CRITICAL-1 veri kaybı yolu)."""
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BG_INDEX_DIR", str(tmp_path / "index"))

    result = runner.invoke(app, ["index", "build", "--fake", "--query-format", "cpe-0.3.18"])

    assert result.exit_code != 0
    assert "--out" in result.output
    assert not (tmp_path / "index").exists()


def test_index_build_refuses_to_overwrite_prod_index_with_other_quantization(tmp_path, monkeypatch):
    """Aynı korkuluğun KUANTİZASYON ekseni (T14).

    Üretim indeksi artık int8 ama `index build` yalnız packed/f16 üretir.
    --out'suz bir build sessizce int8'in üstüne 1-bit yazar ve manifest'i de
    "sign-1bit"e çevirdiği için yükleyici hiçbir şey fark etmeden onu servis
    ederdi: ölçümde KAYBEDEN temsile sessiz geri dönüş."""
    from belge_gozu.index.manifest import write_manifest
    from tests.index.test_manifest import make_manifest

    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    write_manifest(index_dir, make_manifest(quantization="int8"))
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BG_INDEX_DIR", str(index_dir))

    result = runner.invoke(app, ["index", "build", "--fake"])

    assert result.exit_code != 0
    assert "int8" in result.output and "sign-1bit" in result.output
    assert "--out" in result.output and "derive" in result.output
    assert not (index_dir / "tokens.npy").exists()  # hiçbir şey yazılmadı


# --- P1: index build-text (hibrit metin kanalı artefaktı) --------------------


def _built_index(tmp_path: Path, monkeypatch, pages: int = 2) -> Path:
    """`corpus render` + `index build --fake` ile küçük bir üretim indeksi."""
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BG_INDEX_DIR", str(tmp_path / "index"))
    (tmp_path / "manifest").mkdir(parents=True)
    (tmp_path / "manifest" / "v0_manifest.csv").write_text(CSV, encoding="utf-8")
    (tmp_path / "pdf").mkdir()
    make_pdf(tmp_path / "pdf" / "d1.pdf", pages=pages)
    assert runner.invoke(app, ["corpus", "render", "--dpi", "72"]).exit_code == 0
    assert runner.invoke(app, ["index", "build", "--fake"]).exit_code == 0
    return tmp_path / "index"


def test_index_build_text_writes_aligned_parquet(tmp_path: Path, monkeypatch):
    import pandas as pd

    index_dir = _built_index(tmp_path, monkeypatch)
    result = runner.invoke(app, ["index", "build-text"])
    assert result.exit_code == 0, result.output
    assert "2 sayfa" in result.output and "boş" in result.output

    df = pd.read_parquet(index_dir / "page_texts.parquet")
    page_ids = json.loads((index_dir / "page_ids.json").read_text(encoding="utf-8"))
    assert df["page_id"].tolist() == page_ids  # serve tarafı bunu birebir arar
    assert "Sayfa 1" in df["text"][0]


def test_index_build_text_does_not_invalidate_manifest(tmp_path: Path, monkeypatch):
    """Artefakt indeks dizinine yazılır ama `corpus_checksum`u DEĞİŞTİRMEMELİ —
    aksi halde her build-text serve'ü uyumsuzluk hatasına düşürürdü."""
    from belge_gozu.config import Settings
    from belge_gozu.index.compat import check_compatibility
    from belge_gozu.index.manifest import read_manifest

    index_dir = _built_index(tmp_path, monkeypatch)
    assert runner.invoke(app, ["index", "build-text"]).exit_code == 0
    s = Settings()
    problems = check_compatibility(
        read_manifest(index_dir),
        model_name=s.retriever_model,
        model_revision=None,
        query_format_id=s.query_format_id,
        index_dir=index_dir,
    )
    assert problems == []


def test_index_build_text_refuses_partial_corpus(tmp_path: Path, monkeypatch):
    """Yarım kalmış `corpus download` SESSİZ bozulma üretiyordu (review M3).

    PDF'i olmayan bir dokümanın tüm sayfaları boş metinle yazılır; artefakt
    satır-hizalı olduğu için serve'ün kontrolünden GEÇER ve korpusun o kısmı
    BM25 tarafından hiç görülmez."""
    index_dir = _built_index(tmp_path, monkeypatch, pages=2)
    # indeks d1'i tanıyor; sanki indirme yarıda kesilmiş gibi PDF'i kaldır
    (tmp_path / "pdf" / "d1.pdf").unlink()

    result = runner.invoke(app, ["index", "build-text"])

    assert result.exit_code != 0
    assert "d1" in result.output and "corpus download" in result.output
    assert not (index_dir / "page_texts.parquet").exists()  # bozuk artefakt YAZILMADI


def test_index_build_text_allow_missing_escape_hatch(tmp_path: Path, monkeypatch):
    """Bilinçli kısmi koşum mümkün, ama sessiz değil: uyarı + doküman kırılımı."""
    import pandas as pd

    index_dir = _built_index(tmp_path, monkeypatch, pages=2)
    (tmp_path / "pdf" / "d1.pdf").unlink()

    result = runner.invoke(app, ["index", "build-text", "--allow-missing"])

    assert result.exit_code == 0, result.output
    assert "UYARI" in result.output
    assert "boş: d1 2/2 sayfa" in result.output  # doküman başına kırılım
    df = pd.read_parquet(index_dir / "page_texts.parquet")
    assert len(df) == 2 and (df["text"] == "").all()


def test_index_build_text_reports_no_empty_docs_on_healthy_corpus(tmp_path: Path, monkeypatch):
    _built_index(tmp_path, monkeypatch, pages=2)
    result = runner.invoke(app, ["index", "build-text"])
    assert result.exit_code == 0, result.output
    assert "2 sayfa, 0 metin katmanı boş" in result.output
    assert "boş: " not in result.output and "UYARI" not in result.output


def test_index_build_text_refuses_without_index(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BG_INDEX_DIR", str(tmp_path / "yok"))
    result = runner.invoke(app, ["index", "build-text"])
    assert result.exit_code != 0 and "page_ids.json" in result.output


def test_index_build_text_refuses_without_manifest(tmp_path: Path, monkeypatch):
    index_dir = _built_index(tmp_path, monkeypatch)
    (index_dir / "manifest.json").unlink()
    result = runner.invoke(app, ["index", "build-text"])
    assert result.exit_code != 0 and "manifest.json" in result.output


def test_bench_pipeline_default_follows_settings():
    """`bench run --pipeline` varsayılanı config'ten gelmeli: sabit bir literal
    olsaydı üretim hibrite geçtiğinde bench sessizce ESKİ yolu ölçerdi."""
    from belge_gozu.cli import DEFAULT_PIPELINE
    from belge_gozu.config import Settings

    assert DEFAULT_PIPELINE.value == Settings().retrieval_pipeline
    result = runner.invoke(app, ["bench", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "hybrid" in "".join(result.output.split())


def test_index_derive_rejects_float16_quant(tmp_path):
    """`Quantization` T14'te float16 üyesini kazandı; `derive` onu türetemez.

    Açıkça reddedilmezse dallanma sessizce int8 üretir ve manifest'e
    "float16" yazardı: diskteki veriyle etiketi çelişen bir indeks."""
    result = runner.invoke(
        app,
        [
            "index",
            "derive",
            "--from",
            str(tmp_path / "f16"),
            "--quant",
            "float16",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "float16" in result.output
    assert not (tmp_path / "out").exists()


def test_write_manifest_legacy_refuses_to_overwrite_existing_manifest(tmp_path, monkeypatch):
    index_dir = _built_index(tmp_path, monkeypatch)
    manifest_path = index_dir / "manifest.json"
    before = manifest_path.read_bytes()

    result = runner.invoke(app, ["index", "write-manifest", "--legacy"])

    assert result.exit_code != 0
    assert "manifest.json zaten var" in result.output
    assert manifest_path.read_bytes() == before


def test_index_derive_refuses_to_write_into_an_existing_index_target(tmp_path):
    import numpy as np
    import pandas as pd

    from belge_gozu.index.float_store import FloatIndex
    from tests.index.test_manifest import make_manifest

    from_dir = tmp_path / "float-source"
    source = FloatIndex.build(
        ["d1:1"],
        [np.ones((2, 128), dtype=np.float32)],
        manifest=make_manifest(quantization="float16", n_pages=1, n_tokens=2),
    )
    source.save(from_dir)
    pd.DataFrame({"page_id": ["d1:1"]}).to_parquet(from_dir / "meta.parquet", index=False)

    out = tmp_path / "existing-target"
    out.mkdir()
    sentinel = out / "manifest.json"
    sentinel.write_text("user-owned", encoding="utf-8")

    result = runner.invoke(
        app,
        ["index", "derive", "--from", str(from_dir), "--quant", "int8", "--out", str(out)],
    )

    assert result.exit_code != 0
    assert "--out hedefi boş olmalı" in result.output
    assert sentinel.read_text(encoding="utf-8") == "user-owned"
    assert not (out / "codes.npy").exists()


def test_metrics_export_cli(tmp_path, monkeypatch):
    from belge_gozu.telemetry.recorder import EventRecorder
    from belge_gozu.telemetry.schema import RequestEvent

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    rec = EventRecorder(tmp_path / "requests.sqlite")
    rec.record(
        RequestEvent(
            ts="t",
            endpoint="/search",
            status="ok",
            http_status=200,
            total_ms=1.0,
            query_sha256="f" * 64,
        )
    )
    rec.close()
    result = runner.invoke(app, ["metrics", "export", "--out", str(tmp_path / "e.parquet")])
    assert result.exit_code == 0 and (tmp_path / "e.parquet").exists()


def test_metrics_summary_cli(tmp_path, monkeypatch):
    from belge_gozu.telemetry.recorder import EventRecorder
    from belge_gozu.telemetry.schema import RequestEvent

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    rec = EventRecorder(tmp_path / "requests.sqlite")
    rec.record(
        RequestEvent(
            ts="t",
            endpoint="/ask",
            status="answered",
            http_status=200,
            total_ms=10.0,
            abstained=False,
            tokens_in=5,
            tokens_out=7,
            est_cost_usd=0.001,
            query_sha256="a" * 64,
        )
    )
    rec.close()
    result = runner.invoke(app, ["metrics", "summary"])
    assert result.exit_code == 0, result.output
    assert "istek=1 ort=10ms p95=10ms abstain=0.0%" in result.output
    assert "token in/out=5/7 maliyet≈$0.0010" in result.output


def test_metrics_summary_excludes_degraded_from_abstain(tmp_path, monkeypatch):
    from belge_gozu.telemetry.recorder import EventRecorder
    from belge_gozu.telemetry.schema import RequestEvent

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    rec = EventRecorder(tmp_path / "requests.sqlite")
    rec.record(
        RequestEvent(
            ts="t",
            endpoint="/ask",
            status="answered",
            http_status=200,
            total_ms=10.0,
            abstained=False,
            query_sha256="a" * 64,
        )
    )
    # degraded satır: abstained=1 olsa bile abstain oranından hariç tutulmalı.
    rec.record(
        RequestEvent(
            ts="t",
            endpoint="/ask",
            status="degraded",
            http_status=200,
            total_ms=10.0,
            abstained=True,
            query_sha256="b" * 64,
        )
    )
    rec.close()
    result = runner.invoke(app, ["metrics", "summary"])
    assert result.exit_code == 0, result.output
    # degraded satır hariç tutulmasaydı abstain %50.0 olurdu (1/2); dışlanınca %0.0 (0/1).
    assert "abstain=0.0%" in result.output


def test_metrics_summary_p95_nearest_rank(tmp_path, monkeypatch):
    from belge_gozu.telemetry.recorder import EventRecorder
    from belge_gozu.telemetry.schema import RequestEvent

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    rec = EventRecorder(tmp_path / "requests.sqlite")
    for i in range(1, 11):
        rec.record(
            RequestEvent(
                ts="t",
                endpoint="/search",
                status="ok",
                http_status=200,
                total_ms=float(i),
                query_sha256=str(i) * 64,
            )
        )
    rec.close()
    result = runner.invoke(app, ["metrics", "summary"])
    assert result.exit_code == 0, result.output
    assert "p95=10ms" in result.output


def test_metrics_summary_no_events_table(tmp_path, monkeypatch):
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["metrics", "summary"])
    assert result.exit_code == 0, result.output
    assert "henüz olay kaydı yok" in result.output


def test_metrics_export_no_events_table(tmp_path, monkeypatch):
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    out = tmp_path / "e.parquet"
    result = runner.invoke(app, ["metrics", "export", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "0 olay — tablo yok" in result.output
    assert not out.exists()


# --- R15: bench run/oracle --only-verified/--all -----------------------------


def _write_bench_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_load_bench_mode_only_verified(tmp_path: Path, capsys):
    p = tmp_path / "bench.jsonl"
    _write_bench_jsonl(
        p,
        [
            _bench_q(question_id="verified1", verification_status="verified"),
            _bench_q(question_id="draft1", verification_status="draft"),
        ],
    )

    selection = _load_bench_mode(p, only_verified=True, min_verification=None)

    assert selection.only_verified is True
    assert [q.question_id for q in selection.questions] == ["verified1"]
    out = capsys.readouterr().out
    assert "toplam=2 seçilen=1 elenen=1" in out


def test_load_bench_mode_all(tmp_path: Path, capsys):
    p = tmp_path / "bench.jsonl"
    _write_bench_jsonl(
        p,
        [
            _bench_q(question_id="verified1", verification_status="verified"),
            _bench_q(question_id="draft1", verification_status="draft"),
        ],
    )

    selection = _load_bench_mode(p, only_verified=False, min_verification=None)

    assert selection.only_verified is False
    assert {q.question_id for q in selection.questions} == {"verified1", "draft1"}
    out = capsys.readouterr().out
    assert "toplam=2 seçilen=2 elenen=0" in out


def test_load_bench_mode_human_selects_three_of_48(tmp_path: Path, capsys):
    p = tmp_path / "bench.jsonl"
    rows = [_bench_q(question_id=f"human-{i}", verification_kind="human") for i in range(3)]
    rows.extend(
        _bench_q(
            question_id=f"mechanical-{i}",
            verification_kind="mechanical:manifest-absence",
        )
        for i in range(45)
    )
    _write_bench_jsonl(p, rows)

    selection = _load_bench_mode(
        p,
        only_verified=False,
        min_verification="human",
    )

    assert selection.total == 48
    assert selection.selected == 3
    assert selection.filtered_out == 45
    assert {q.verification_kind for q in selection.questions} == {"human"}
    assert "min=human" in capsys.readouterr().out


def test_bench_run_help_lists_only_verified_and_all():
    result = runner.invoke(app, ["bench", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--only-verified" in result.output
    assert "--all" in result.output
    assert "--min-verification" in result.output
    assert "--split" in result.output
    assert "--splits" in result.output
    assert "--yes-final-gate" in result.output


def test_bench_oracle_help_lists_only_verified_and_all():
    result = runner.invoke(app, ["bench", "oracle", "--help"])
    assert result.exit_code == 0, result.output
    assert "--only-verified" in result.output
    assert "--all" in result.output
    assert "--min-verification" in result.output
    assert "yalnız görsel" in result.output


def _oracle_index_pair(tmp_path: Path, *, float_checksum: str | None) -> tuple[Path, Path]:
    import numpy as np
    import pandas as pd

    from belge_gozu.index.float_store import FloatIndex
    from belge_gozu.index.manifest import corpus_checksum, write_manifest
    from belge_gozu.index.store import PackedIndex
    from tests.index.test_manifest import make_manifest

    ids = ["d1:1"]
    embs = [np.ones((2, 128), dtype=np.float32)]
    packed_dir = tmp_path / "packed"
    float_dir = tmp_path / "float"
    packed_manifest = make_manifest(n_pages=1, n_tokens=2, corpus_checksum="a" * 64)
    float_manifest = make_manifest(quantization="float16", n_pages=1, n_tokens=2)
    PackedIndex.build(
        ids,
        embs,
        manifest=packed_manifest,
    ).save(packed_dir)
    FloatIndex.build(
        ids,
        embs,
        manifest=float_manifest,
    ).save(float_dir)
    pd.DataFrame({"page_id": ids}).to_parquet(packed_dir / "meta.parquet", index=False)
    (float_dir / "meta.parquet").write_bytes((packed_dir / "meta.parquet").read_bytes())
    live_checksum = corpus_checksum(packed_dir)
    write_manifest(
        packed_dir, packed_manifest.model_copy(update={"corpus_checksum": live_checksum})
    )
    write_manifest(
        float_dir,
        float_manifest.model_copy(update={"corpus_checksum": float_checksum or live_checksum}),
    )
    return packed_dir, float_dir


def test_bench_oracle_rejects_matching_page_ids_from_different_corpora(tmp_path: Path, monkeypatch):
    """Aynı page_id dizisi, farklı kaynak içeriğini eşdeğer yapmaz."""
    from belge_gozu.index import encode

    # Guard, gerçek model yüklenmeden önce çalışmalı; CI'da torch da yok.
    class FailEncoder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("encoder was constructed before identity validation")

    monkeypatch.setattr(encode, "ColSmolEncoder", FailEncoder)
    packed_dir, float_dir = _oracle_index_pair(tmp_path, float_checksum="b" * 64)
    out = tmp_path / "oracle.json"

    result = runner.invoke(
        app,
        [
            "bench",
            "oracle",
            "--bench",
            str(tmp_path / "unused.jsonl"),
            "--packed-index",
            str(packed_dir),
            "--float-index",
            str(float_dir),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code != 0
    assert "corpus_checksum uyuşmuyor" in result.output
    assert not out.exists()


def test_bench_oracle_report_identifies_visual_retrieval_scope(tmp_path: Path, monkeypatch):
    import numpy as np

    from belge_gozu.index import encode
    from tests.bench_question_factory import q_dict

    class FixedEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_query(self, question):
            return np.ones((2, 128), dtype=np.float32)

    monkeypatch.setattr(encode, "ColSmolEncoder", FixedEncoder)
    packed_dir, float_dir = _oracle_index_pair(tmp_path, float_checksum=None)
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps(q_dict(gold_doc_ids=["d1"], gold_page_ids=["d1:1"], gold_article_ids=["d1:m1"]))
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "oracle.json"

    result = runner.invoke(
        app,
        [
            "bench",
            "oracle",
            "--bench",
            str(bench),
            "--packed-index",
            str(packed_dir),
            "--float-index",
            str(float_dir),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.exception or result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["retrieval_pipeline"] == "exhaustive-visual"
    assert report["summary"]["float"]["5"] == 1.0


def test_retrieval_cli_refuses_zero_answerable_questions_before_loading_model(
    tmp_path: Path, monkeypatch
):
    from belge_gozu.index import encode
    from tests.bench_question_factory import q_dict

    class FailEncoder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("model loaded before benchmark preflight")

    monkeypatch.setattr(encode, "ColSmolEncoder", FailEncoder)
    packed_dir, float_dir = _oracle_index_pair(tmp_path, float_checksum=None)
    monkeypatch.setenv("BG_INDEX_DIR", str(packed_dir))
    bench = tmp_path / "unanswerable.jsonl"
    bench.write_text(
        json.dumps(
            q_dict(
                answerable=False,
                gold_doc_ids=[],
                gold_page_ids=[],
                gold_article_ids=[],
                minimal_evidence_spans=[],
                reference_answer="",
                slice="korpus-disi",
                unanswerable_reason="korpus-disi",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    commands = [
        ["bench", "run", "--pipeline", "exhaustive", "--bench", str(bench)],
        [
            "bench",
            "oracle",
            "--bench",
            str(bench),
            "--packed-index",
            str(packed_dir),
            "--float-index",
            str(float_dir),
        ],
    ]
    for index, command in enumerate(commands):
        out = tmp_path / f"report-{index}.json"
        result = runner.invoke(app, [*command, "--out", str(out)])
        assert result.exit_code != 0
        assert "cevaplanabilir soru yok" in result.output
        assert not out.exists()


def test_retrieval_cli_rejects_index_model_mismatch_before_loading_model(
    tmp_path: Path, monkeypatch
):
    from belge_gozu.index import encode
    from tests.bench_question_factory import q_dict

    class FailEncoder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("model loaded before index identity validation")

    monkeypatch.setattr(encode, "ColSmolEncoder", FailEncoder)
    packed_dir, float_dir = _oracle_index_pair(tmp_path, float_checksum=None)
    monkeypatch.setenv("BG_INDEX_DIR", str(packed_dir))
    monkeypatch.setenv("BG_RETRIEVER_MODEL", "different/model")
    bench = tmp_path / "bench.jsonl"
    bench.write_text(json.dumps(q_dict()) + "\n", encoding="utf-8")

    commands = [
        ["bench", "run", "--pipeline", "exhaustive", "--bench", str(bench)],
        [
            "bench",
            "oracle",
            "--bench",
            str(bench),
            "--packed-index",
            str(packed_dir),
            "--float-index",
            str(float_dir),
        ],
    ]
    for index, command in enumerate(commands):
        out = tmp_path / f"mismatch-{index}.json"
        result = runner.invoke(app, [*command, "--out", str(out)])
        assert result.exit_code != 0
        assert "model_name" in result.output
        assert not out.exists()


def test_bench_oracle_rejects_a_changed_corpus_with_unchanged_manifest(tmp_path: Path, monkeypatch):
    from belge_gozu.index import encode
    from tests.bench_question_factory import q_dict

    class FailEncoder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("model loaded before live corpus validation")

    monkeypatch.setattr(encode, "ColSmolEncoder", FailEncoder)
    packed_dir, float_dir = _oracle_index_pair(tmp_path, float_checksum=None)
    meta = float_dir / "meta.parquet"
    meta.write_bytes(meta.read_bytes() + b"changed")
    bench = tmp_path / "bench.jsonl"
    bench.write_text(json.dumps(q_dict()) + "\n", encoding="utf-8")
    out = tmp_path / "oracle.json"

    result = runner.invoke(
        app,
        [
            "bench",
            "oracle",
            "--bench",
            str(bench),
            "--packed-index",
            str(packed_dir),
            "--float-index",
            str(float_dir),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code != 0
    assert "corpus_checksum" in result.output
    assert not out.exists()


def test_retrieval_cli_rejects_loaded_model_revision_mismatch(tmp_path: Path, monkeypatch):
    from belge_gozu.index import encode
    from tests.bench_question_factory import q_dict

    class OtherRevisionEncoder:
        model_revision = "different-revision"

        def __init__(self, *args, **kwargs):
            pass

        def encode_query(self, question):
            raise AssertionError("scoring started before revision validation")

    monkeypatch.setattr(encode, "ColSmolEncoder", OtherRevisionEncoder)
    packed_dir, float_dir = _oracle_index_pair(tmp_path, float_checksum=None)
    monkeypatch.setenv("BG_INDEX_DIR", str(packed_dir))
    bench = tmp_path / "bench.jsonl"
    bench.write_text(json.dumps(q_dict()) + "\n", encoding="utf-8")

    commands = [
        ["bench", "run", "--pipeline", "exhaustive", "--bench", str(bench)],
        [
            "bench",
            "oracle",
            "--bench",
            str(bench),
            "--packed-index",
            str(packed_dir),
            "--float-index",
            str(float_dir),
        ],
    ]
    for index, command in enumerate(commands):
        out = tmp_path / f"revision-{index}.json"
        result = runner.invoke(app, [*command, "--out", str(out)])
        assert result.exit_code != 0
        assert "model_revision" in result.output
        assert not out.exists()


def test_bench_run_rejects_loaded_document_prompt_mismatch(tmp_path: Path, monkeypatch):
    from belge_gozu.index import encode
    from tests.bench_question_factory import q_dict

    class OtherPromptEncoder:
        model_revision = "abc123"
        doc_prompt_sha256 = "0" * 64

        def __init__(self, *args, **kwargs):
            pass

        def encode_query(self, question):
            raise AssertionError("scoring started before prompt validation")

    monkeypatch.setattr(encode, "ColSmolEncoder", OtherPromptEncoder)
    packed_dir, _ = _oracle_index_pair(tmp_path, float_checksum=None)
    monkeypatch.setenv("BG_INDEX_DIR", str(packed_dir))
    bench = tmp_path / "bench.jsonl"
    bench.write_text(json.dumps(q_dict()) + "\n", encoding="utf-8")
    out = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["bench", "run", "--pipeline", "exhaustive", "--bench", str(bench), "--out", str(out)],
    )

    assert result.exit_code != 0
    assert "doc_prompt_sha256" in result.output
    assert not out.exists()


def test_bench_run_accepts_matching_index_and_encoder_identity(tmp_path: Path, monkeypatch):
    import numpy as np

    from belge_gozu.index import encode
    from tests.bench_question_factory import q_dict
    from tests.index.test_manifest import TRAIN_COMPAT_DOC_PROMPT_SHA256

    class MatchingEncoder:
        model_revision = "abc123"
        doc_prompt_sha256 = TRAIN_COMPAT_DOC_PROMPT_SHA256

        def __init__(self, *args, **kwargs):
            pass

        def encode_query(self, question):
            return np.ones((2, 128), dtype=np.float32)

    monkeypatch.setattr(encode, "ColSmolEncoder", MatchingEncoder)
    packed_dir, _ = _oracle_index_pair(tmp_path, float_checksum=None)
    monkeypatch.setenv("BG_INDEX_DIR", str(packed_dir))
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps(q_dict(gold_doc_ids=["d1"], gold_page_ids=["d1:1"], gold_article_ids=[])) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["bench", "run", "--pipeline", "exhaustive", "--bench", str(bench), "--out", str(out)],
    )

    assert result.exit_code == 0, result.exception or result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["overall"]["n"] == 1
    assert report["overall"]["recall_at"]["5"] == 1.0
    assert report["index_manifest"]["corpus_checksum"]


def test_retrieval_cli_rejects_effective_query_format_mismatch(tmp_path: Path, monkeypatch):
    from belge_gozu.index import encode
    from belge_gozu.index.manifest import CPE_0_3_18
    from tests.bench_question_factory import q_dict

    class OtherFormatEncoder:
        model_revision = "abc123"
        query_format = CPE_0_3_18

        def __init__(self, *args, **kwargs):
            pass

        def encode_query(self, question):
            raise AssertionError("scoring started before query format validation")

    monkeypatch.setattr(encode, "ColSmolEncoder", OtherFormatEncoder)
    packed_dir, float_dir = _oracle_index_pair(tmp_path, float_checksum=None)
    monkeypatch.setenv("BG_INDEX_DIR", str(packed_dir))
    bench = tmp_path / "bench.jsonl"
    bench.write_text(json.dumps(q_dict()) + "\n", encoding="utf-8")

    commands = [
        ["bench", "run", "--pipeline", "exhaustive", "--bench", str(bench)],
        [
            "bench",
            "oracle",
            "--bench",
            str(bench),
            "--packed-index",
            str(packed_dir),
            "--float-index",
            str(float_dir),
        ],
    ]
    for index, command in enumerate(commands):
        out = tmp_path / f"format-{index}.json"
        result = runner.invoke(app, [*command, "--out", str(out)])
        assert result.exit_code != 0
        assert "query_format" in result.output
        assert not out.exists()


def test_bench_run_hybrid_uses_configured_late_channels_and_records_recipe(
    tiny_corpus, monkeypatch
):
    import numpy as np
    import pandas as pd

    from belge_gozu.bench.report_validation import validate_provenance_hashes
    from belge_gozu.index import encode
    from belge_gozu.index.store import PackedIndex
    from belge_gozu.retrieval.hybrid import HybridRetriever, load_text_channel
    from belge_gozu.retrieval.late import LateSearchResult
    from belge_gozu.retrieval.text import recipe_fingerprint
    from tests.bench_question_factory import q_dict
    from tests.index.test_manifest import TRAIN_COMPAT_DOC_PROMPT_SHA256

    data_dir, inner_encoder, _ = tiny_corpus
    index_dir = data_dir / "index"

    class MatchingEncoder:
        model_revision = "abc123"
        doc_prompt_sha256 = TRAIN_COMPAT_DOC_PROMPT_SHA256

        def __init__(self, *args, **kwargs):
            pass

        def encode_query(self, question):
            return inner_encoder.encode_query(question)

    monkeypatch.setattr(encode, "ColSmolEncoder", MatchingEncoder)
    monkeypatch.setenv("BG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BG_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("BG_LATE_CHANNEL_ENABLED", "true")
    monkeypatch.setenv("BG_LATE_CANDIDATE_LIMIT", "1")
    for position, (env_name, label) in enumerate(
        (
            ("BG_LATE_MOGAN_INDEX_DIR", "mogan"),
            ("BG_LATE_COLMM_INDEX_DIR", "colmm"),
        ),
        start=1,
    ):
        late_dir = data_dir / f"late-{label}"
        late_dir.mkdir()
        (late_dir / "colbert.json").write_text(
            json.dumps({"model_repo": f"test/{label}", "revision": f"revision-{position}"}),
            encoding="utf-8",
        )
        (late_dir / "chunk_ids.json").write_text("[]", encoding="utf-8")
        np.save(late_dir / "offsets.npy", np.array([0], dtype=np.int64))
        np.save(late_dir / "embs.npy", np.full((1, 2), position, dtype=np.float16))
        monkeypatch.setenv(env_name, str(late_dir))
    index = PackedIndex.load(index_dir)
    meta = pd.read_parquet(index_dir / "meta.parquet")
    bm25, doc_names = load_text_channel(index_dir, index.page_ids)
    question = "yerleşim yeri nedir"
    base = HybridRetriever(index, meta, inner_encoder, bm25, doc_names)
    late_page = base.rank_all(question)[-1]

    class FixedLateChannel:
        def search_with_scores(self, query: str, limit: int) -> LateSearchResult:
            return LateSearchResult(
                pages=(late_page,),
                query_tokens=2,
                raw_top1=2.0,
                raw_margin=1.0,
                mean_top1=1.0,
                mean_margin=0.5,
            )

    class EmptyLateChannel:
        def search_with_scores(self, query: str, limit: int) -> LateSearchResult:
            return LateSearchResult(
                pages=(),
                query_tokens=2,
                raw_top1=0.0,
                raw_margin=0.0,
                mean_top1=0.0,
                mean_margin=0.0,
            )

    loads = []

    def fake_load_channels(settings, page_ids):
        loads.append((settings.late_channel_enabled, page_ids))
        return (FixedLateChannel(), EmptyLateChannel())

    monkeypatch.setattr("belge_gozu.app.main.load_configured_late_channels", fake_load_channels)
    bench = data_dir / "bench.jsonl"
    bench.write_text(
        json.dumps(
            q_dict(
                question=question,
                gold_doc_ids=[late_page.split(":")[0]],
                gold_page_ids=[late_page],
                gold_article_ids=[],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    out = data_dir / "report.json"

    result = runner.invoke(
        app,
        ["bench", "run", "--pipeline", "hybrid", "--bench", str(bench), "--out", str(out)],
    )

    assert result.exit_code == 0, result.exception or result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert loads == [(True, index.page_ids)]
    assert report["config"]["recipe_fingerprint"] == recipe_fingerprint()
    assert report["config"]["late_channel_enabled"] is True
    late_evidence = report["config"]["late_index_evidence"]
    assert [entry["sidecar"]["revision"] for entry in late_evidence] == [
        "revision-1",
        "revision-2",
    ]
    assert all(len(entry["files"]) == 4 for entry in late_evidence)
    validate_provenance_hashes(report)
    embeddings = next(
        file for file in late_evidence[0]["files"] if file["path"].endswith("embs.npy")
    )
    Path(embeddings["path"]).write_bytes(Path(embeddings["path"]).read_bytes() + b"changed")
    with pytest.raises(ValueError, match="sha256"):
        validate_provenance_hashes(report)
    assert report["diagnostics"][0]["stages"][-1]["stage"] == "late_candidate_union"
    assert report["diagnostics"][0]["final_ranked"][1] == late_page


def test_bench_run_selects_dev_from_canonical_benchmark_and_guards_test_split(
    tiny_corpus, monkeypatch
):
    from belge_gozu.bench.report_validation import validate_retrieval_report_payload
    from belge_gozu.index import encode
    from tests.bench_question_factory import q_dict
    from tests.index.test_manifest import TRAIN_COMPAT_DOC_PROMPT_SHA256

    data_dir, inner_encoder, _ = tiny_corpus
    index_dir = data_dir / "index"

    class MatchingEncoder:
        model_revision = "abc123"
        doc_prompt_sha256 = TRAIN_COMPAT_DOC_PROMPT_SHA256

        def __init__(self, *args, **kwargs):
            pass

        def encode_query(self, question):
            return inner_encoder.encode_query(question)

    monkeypatch.setattr(encode, "ColSmolEncoder", MatchingEncoder)
    monkeypatch.setenv("BG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BG_INDEX_DIR", str(index_dir))
    bench = data_dir / "bench.jsonl"
    bench.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                q_dict(
                    question_id="dev",
                    gold_doc_ids=["d0"],
                    gold_page_ids=["d0:1"],
                    gold_article_ids=[],
                ),
                q_dict(
                    question_id="test",
                    gold_doc_ids=["d1"],
                    gold_page_ids=["d1:1"],
                    gold_article_ids=[],
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    splits = data_dir / "splits.json"
    splits.write_text(json.dumps({"dev_docs": ["d0"], "test_docs": ["d1"]}), encoding="utf-8")
    out = data_dir / "dev-report.json"
    command = [
        "bench",
        "run",
        "--pipeline",
        "exhaustive",
        "--bench",
        str(bench),
        "--min-verification",
        "human",
        "--splits",
        str(splits),
        "--split",
        "dev",
    ]

    result = runner.invoke(app, [*command, "--out", str(out)])

    assert result.exit_code == 0, result.exception or result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"]["split"] == "dev"
    assert payload["config"]["splits"]["sha256"]
    assert payload["config"]["benchmark"]["sha256"]
    assert payload["config"]["selected_after_split"] == 1
    assert [row["question_id"] for row in payload["diagnostics"]] == ["dev"]
    validate_retrieval_report_payload(payload, require_bench=True)
    split_sha = payload["config"]["splits"]["sha256"]
    payload["config"]["splits"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="splits.sha256"):
        validate_retrieval_report_payload(payload, require_bench=True)
    payload["config"]["splits"]["sha256"] = split_sha

    test_command = [*command[:-1], "test", "--out", str(data_dir / "test-report.json")]
    rejected = runner.invoke(app, test_command)
    assert rejected.exit_code != 0
    assert "--yes-final-gate" in rejected.output
    assert not (data_dir / "test-report.json").exists()


def test_broken_env_gives_readable_message_not_a_traceback(tmp_path: Path):
    """`belge-gozu --help` bozuk bir BG_* değerinde ham traceback BASMAZ.

    `_CLI_DEFAULTS = Settings()` import anında koşar, yani yardım metni bile
    ortamı doğrular (audit C9). Doğrulama alt süreçte yapılır çünkü hata tam
    olarak IMPORT sırasında oluşur — aynı süreçte modül zaten yüklü olurdu.
    Ayrıca `cwd=tmp_path`: repo kökündeki bir `.env` sonucu etkilemesin.
    """
    import subprocess
    import sys

    env = {**os.environ, "BG_QUERY_FORMAT_ID": "bogus-format"}
    r = subprocess.run(
        [sys.executable, "-c", "import belge_gozu.cli"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert r.returncode == 2, r.stderr
    assert "Yapılandırma hatası" in r.stderr
    assert "query_format_id" in r.stderr
    assert "Traceback" not in r.stderr


# --- P2: `verify run` harness'ı (stub istemci, AĞ YOK) ------------------------


def _bench_row(**over) -> dict:
    base = dict(
        question_id="q1",
        question="Yerleşim yeri nedir?",
        query_style="dogal",
        answerable=True,
        gold_doc_ids=["k4721"],
        gold_page_ids=["k4721:4"],
        gold_article_ids=[],
        minimal_evidence_spans=["Yerleşim yeri ..."],
        reference_answer="Sürekli kalma niyetiyle oturulan yerdir.",
        slice="paraphrase",
        difficulty="orta",
        source_type="insan",
        requires_visual=False,
        requires_multi_hop=False,
        unanswerable_reason=None,
        verified_by="baran",
        verification_status="verified",
    )
    base.update(over)
    return base


def _verify_fixture(tmp_path: Path, verdict: str, max_claims: int = 8):
    """`verify run`un ihtiyaç duyduğu her şeyi stub'lar: bench, split, servis."""
    from belge_gozu.answer.base import Answer, AskService
    from belge_gozu.answer.verify import ClaimVerifier, EvidenceGate, Gates, VerifierCache
    from belge_gozu.retrieval.types import PageHit

    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [_bench_row(), _bench_row(question_id="q2", question="Yıllık izin kaç gün?")]
        ),
        encoding="utf-8",
    )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"dev_docs": ["k4721"], "test_docs": []}), encoding="utf-8")

    class StubClient:
        def __init__(self):
            self.prompts = []

        def generate_json(self, prompt, schema=None):
            self.prompts.append(prompt)
            return json.dumps({"verdict": verdict, "gerekce": "stub"})

    class StubRetriever:
        last_bm25_scores = None

        def search(self, query, k=5, candidates=200):
            return [
                PageHit(
                    page_id="k4721:4",
                    score=42.0,
                    doc_name="TMK",
                    page_no=4,
                    image_path="images/x.webp",
                    source_url="https://example.org",
                )
            ]

    class StubAnswerer:
        def answer(self, question, pages, image_loader):
            if "Fransa" in question:
                return Answer(text="Verilen sayfalarda bulamadım.", citations=[])
            # Soruyu metne KATAR: iki soru iki AYRI iddia üretsin, yoksa
            # ikincisi önbellekten gelir ve bütçe testi ölçtüğünü ölçmez.
            return Answer(
                text=f"Sorunun ({question}) yanıtı sürekli kalma niyetiyle belirlenir [S1].",
                citations=[pages[0].page_id],
            )

    client = StubClient()

    def factory(s, budget):
        gate2 = EvidenceGate(
            ClaimVerifier(
                client=client,
                model=s.gemini_model,
                cache=VerifierCache(s.data_dir / "cache" / "verifier"),
            ),
            {"k4721:4": "TÜRK MEDENİ KANUNU\nYerleşim yeri sürekli kalma niyetiyle..."},
            max_claims=max_claims,
            budget=budget,
        )
        svc = AskService(
            StubRetriever(), StubAnswerer(), -1e9, lambda p: b"img", gate1=None, gate2=gate2
        )
        return (
            svc,
            Gates(evidence=gate2, detail={"gate2": {"stub": True}}),
            {"quantization": "int8", "corpus_checksum": "stub"},
            "rev/x/int8",
        )

    return bench, splits, factory, client


def test_bench_answers_writes_answer_metrics_and_provenance(tmp_path: Path, monkeypatch):
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, client = _verify_fixture(tmp_path, "supported")
    with bench.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            + json.dumps(
                _bench_row(
                    question_id="q-no-answer",
                    question="Fransa'da asgari ücret nedir?",
                    answerable=False,
                    gold_doc_ids=[],
                    gold_page_ids=[],
                    minimal_evidence_spans=[],
                    reference_answer="",
                    slice="korpus-disi",
                    unanswerable_reason="korpus-disi",
                ),
                ensure_ascii=False,
            )
        )
    monkeypatch.setattr(cli_mod, "_verify_service", factory)
    out = tmp_path / "answer-report.json"

    result = runner.invoke(
        cli_mod.app,
        [
            "bench",
            "answers",
            "--bench",
            str(bench),
            "--splits",
            str(splits),
            "--max-llm-attempts",
            "4",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["metrics"]["citation_precision"]["rate"] == 1.0
    assert report["metrics"]["false_supported_answer_rate"]["rate"] == 0.0
    assert report["metrics"]["false_supported_answer_rate"]["upper_bound_95"] is not None
    assert report["index_manifest"]["quantization"] == "int8"
    assert report["index_revision"] == "rev/x/int8"
    assert report["calibrator_key"] is None or isinstance(report["calibrator_key"], str)
    assert report["dataset"]["bench"]["sha256"]
    assert report["dataset"]["splits"]["sha256"]
    assert [row["question_id"] for row in report["records"]] == ["q1", "q2", "q-no-answer"]
    assert len(client.prompts) == 2


def test_bench_answers_requires_an_explicit_llm_attempt_budget(tmp_path: Path, monkeypatch):
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, _ = _verify_fixture(tmp_path, "supported")
    monkeypatch.setattr(cli_mod, "_verify_service", factory)

    result = runner.invoke(
        cli_mod.app,
        ["bench", "answers", "--bench", str(bench), "--splits", str(splits)],
    )

    assert result.exit_code != 0
    assert "max-llm-attempts" in result.output


def test_bench_answers_test_split_needs_the_final_gate_flag(tmp_path: Path, monkeypatch):
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, _ = _verify_fixture(tmp_path, "supported")
    monkeypatch.setattr(cli_mod, "_verify_service", factory)

    result = runner.invoke(
        cli_mod.app,
        [
            "bench",
            "answers",
            "--bench",
            str(bench),
            "--splits",
            str(splits),
            "--split",
            "test",
            "--max-llm-attempts",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "--yes-final-gate" in result.output


def test_verify_run_writes_a_kunyeli_report(tmp_path: Path, monkeypatch):
    """Harness iki soruyu koşar, kararları sayar ve künyeli JSON yazar."""
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, client = _verify_fixture(tmp_path, "supported")
    monkeypatch.setattr(cli_mod, "_verify_service", factory)
    out = tmp_path / "report.json"

    r = runner.invoke(
        cli_mod.app,
        [
            "verify",
            "run",
            "--bench",
            str(bench),
            "--splits",
            str(splits),
            "--max-llm-calls",
            "4",
            "--out",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["n"] == 2
    assert report["summary"]["by_status"] == {"answered": 2}
    assert report["summary"]["verdicts"] == {"supported": 2}
    assert report["summary"]["gate2_demoted"] == 0
    assert report["budget"] == {
        "unit": "api_attempts",
        "max_attempts": 4,
        "used": 2,
        "stopped": None,
    }
    assert report["summary"]["verifier_api_attempts"] == 2
    assert report["config"]["gate_calibrated"] and report["config"]["gate_verifier"]
    assert report["bench"]["sha256"] and report["git_commit"]
    assert [q["qid"] for q in report["per_question"]] == ["q1", "q2"]
    assert len(client.prompts) == 2


def test_verify_run_second_pass_is_free_thanks_to_the_sha256_cache(tmp_path: Path, monkeypatch):
    """AYNI koşum ikinci kez: sıfır LLM çağrısı (önbellek `BG_DATA_DIR` altında)."""
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, client = _verify_fixture(tmp_path, "supported")
    monkeypatch.setattr(cli_mod, "_verify_service", factory)
    argv = ["verify", "run", "--bench", str(bench), "--splits", str(splits), "--max-llm-calls", "4"]

    first = tmp_path / "a.json"
    assert runner.invoke(cli_mod.app, [*argv, "--out", str(first)]).exit_code == 0
    assert json.loads(first.read_text(encoding="utf-8"))["summary"]["verifier_llm_calls"] == 2
    assert len(client.prompts) == 2

    second = tmp_path / "b.json"
    assert runner.invoke(cli_mod.app, [*argv, "--out", str(second)]).exit_code == 0
    summary = json.loads(second.read_text(encoding="utf-8"))["summary"]
    assert summary["verifier_llm_calls"] == 0 and summary["verifier_cache_hits"] == 2
    assert len(client.prompts) == 2, "ikinci koşumda istemciye HİÇ gidilmedi"


def test_verify_run_demotes_and_reports_unsupported(tmp_path: Path, monkeypatch):
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, _ = _verify_fixture(tmp_path, "unsupported")
    monkeypatch.setattr(cli_mod, "_verify_service", factory)
    out = tmp_path / "report.json"

    r = runner.invoke(
        cli_mod.app,
        [
            "verify",
            "run",
            "--bench",
            str(bench),
            "--splits",
            str(splits),
            "--max-llm-calls",
            "4",
            "--out",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["by_status"] == {"abstained": 2}
    assert report["summary"]["gate2_demoted"] == 2


def test_verify_run_requires_an_explicit_llm_budget(tmp_path: Path, monkeypatch):
    """`--max-llm-calls` ZORUNLU: sınırsız varsayılan bir bütçe değildir."""
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, _ = _verify_fixture(tmp_path, "supported")
    monkeypatch.setattr(cli_mod, "_verify_service", factory)
    r = runner.invoke(
        cli_mod.app, ["verify", "run", "--bench", str(bench), "--splits", str(splits)]
    )
    assert r.exit_code != 0
    assert "max-llm-calls" in r.output


def test_verify_run_stops_when_the_budget_is_exhausted(tmp_path: Path, monkeypatch):
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, client = _verify_fixture(tmp_path, "supported")
    monkeypatch.setattr(cli_mod, "_verify_service", factory)
    out = tmp_path / "report.json"

    r = runner.invoke(
        cli_mod.app,
        [
            "verify",
            "run",
            "--bench",
            str(bench),
            "--splits",
            str(splits),
            "--max-llm-calls",
            "1",
            "--out",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["n"] == 1 and report["budget"]["used"] == 1
    assert "bütçe doldu" in report["budget"]["stopped"]
    assert len(client.prompts) == 1, "bütçe tavanı GERÇEKTEN çağrıyı kesiyor"


def test_verify_run_test_split_needs_the_final_gate_flag(tmp_path: Path, monkeypatch):
    """G2.4: test bölmesi tek koşumdur, kazayla koşulamaz (calibrate ile aynı bariyer)."""
    import belge_gozu.cli as cli_mod

    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    bench, splits, factory, _ = _verify_fixture(tmp_path, "supported")
    monkeypatch.setattr(cli_mod, "_verify_service", factory)
    r = runner.invoke(
        cli_mod.app,
        [
            "verify",
            "run",
            "--bench",
            str(bench),
            "--splits",
            str(splits),
            "--split",
            "test",
            "--max-llm-calls",
            "1",
        ],
    )
    assert r.exit_code != 0 and "--yes-final-gate" in r.output
