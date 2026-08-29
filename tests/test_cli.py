import json
from pathlib import Path

import pymupdf as fitz
from typer.testing import CliRunner

from belge_gozu.cli import _load_bench_mode, app

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


def _bench_q(**over) -> dict:
    base = dict(
        question_id="q1",
        question="Yerleşim yeri nedir?",
        query_style="dogal",
        answerable=True,
        gold_doc_ids=["k4721"],
        gold_page_ids=["k4721:4"],
        gold_article_ids=["k4721:m19"],
        minimal_evidence_spans=[
            "Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."
        ],
        reference_answer="Sürekli kalma niyetiyle oturulan yerdir (TMK m.19).",
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

    questions, only_verified = _load_bench_mode(p, only_verified=True)

    assert only_verified is True
    assert [q.question_id for q in questions] == ["verified1"]
    out = capsys.readouterr().out
    assert "bench modu: yalnız doğrulanmış (n=1)" in out


def test_load_bench_mode_all(tmp_path: Path, capsys):
    p = tmp_path / "bench.jsonl"
    _write_bench_jsonl(
        p,
        [
            _bench_q(question_id="verified1", verification_status="verified"),
            _bench_q(question_id="draft1", verification_status="draft"),
        ],
    )

    questions, only_verified = _load_bench_mode(p, only_verified=False)

    assert only_verified is False
    assert {q.question_id for q in questions} == {"verified1", "draft1"}
    out = capsys.readouterr().out
    assert "bench modu: TÜMÜ (taslak dahil, n=2)" in out


def test_bench_run_help_lists_only_verified_and_all():
    result = runner.invoke(app, ["bench", "run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--only-verified" in result.output
    assert "--all" in result.output


def test_bench_oracle_help_lists_only_verified_and_all():
    result = runner.invoke(app, ["bench", "oracle", "--help"])
    assert result.exit_code == 0, result.output
    assert "--only-verified" in result.output
    assert "--all" in result.output
