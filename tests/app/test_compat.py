from pathlib import Path

import pytest
from typer.testing import CliRunner

from belge_gozu.cli import app
from belge_gozu.index.compat import IndexCompatibilityError, check_compatibility
from tests.index.test_manifest import TRAIN_COMPAT_DOC_PROMPT_SHA256, make_manifest

runner = CliRunner()

# make_manifest() varsayılanı üretim formatıdır (bkz. tests/index/test_manifest.py);
# "eşleşen" çağrılar bunu, "uyuşmayan" çağrılar diğerini kullanır.
PROD_FORMAT_ID = "train-compat-v1"
OTHER_FORMAT_ID = "cpe-0.3.18"


def _index_stub(tmp_path: Path) -> str:
    """corpus_checksum'ın okuduğu iki dosyayı yazar, canlı checksum'ı döner."""
    from belge_gozu.index.manifest import corpus_checksum

    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    return corpus_checksum(tmp_path)


def test_missing_manifest_is_mismatch(tmp_path: Path):
    problems = check_compatibility(
        None,
        model_name="m",
        model_revision="r",
        query_format_id=PROD_FORMAT_ID,
        index_dir=tmp_path,
    )
    assert problems and "manifest" in problems[0]


def test_missing_manifest_hint_points_at_rebuild_not_legacy(tmp_path: Path):
    """Final review IMPORTANT-3: eski ipucu `write-manifest --legacy` diyordu,
    ama o komut mask_policy="none" + cpe-0.3.18 yazar ve aynı kontrolden yine
    geçemez (çıkmaz sokak). Metin yeniden inşayı önermeli, --legacy'yi yalnız
    teşhis olarak anmalı."""
    problems = check_compatibility(
        None,
        model_name="m",
        model_revision="r",
        query_format_id=PROD_FORMAT_ID,
        index_dir=tmp_path,
    )
    hint = problems[0]
    assert "index build" in hint
    assert "teşhis" in hint and "mask_policy" in hint


def test_matching_manifest_ok(tmp_path: Path):
    m = make_manifest(corpus_checksum=_index_stub(tmp_path))
    assert (
        check_compatibility(
            m,
            model_name="vidore/colSmol-500M",
            model_revision="abc123",
            query_format_id=PROD_FORMAT_ID,
            index_dir=tmp_path,
        )
        == []
    )


def test_format_mismatch_reported(tmp_path: Path):
    m = make_manifest(corpus_checksum=_index_stub(tmp_path))
    problems = check_compatibility(
        m,
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        query_format_id=OTHER_FORMAT_ID,
        index_dir=tmp_path,
    )
    assert any("query_format" in p for p in problems)


def test_matching_doc_prompt_ok(tmp_path: Path):
    m = make_manifest(corpus_checksum=_index_stub(tmp_path))
    problems = check_compatibility(
        m,
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        query_format_id=PROD_FORMAT_ID,
        doc_prompt_sha256=TRAIN_COMPAT_DOC_PROMPT_SHA256,
        index_dir=tmp_path,
    )
    assert problems == []


def test_doc_prompt_mismatch_reported(tmp_path: Path):
    m = make_manifest(corpus_checksum=_index_stub(tmp_path))
    problems = check_compatibility(
        m,
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        query_format_id=PROD_FORMAT_ID,
        doc_prompt_sha256="e" * 64,
        index_dir=tmp_path,
    )
    assert any("doc_prompt" in p for p in problems)


def test_doc_prompt_none_or_unknown_skips_check(tmp_path: Path):
    m = make_manifest(corpus_checksum=_index_stub(tmp_path))
    for value in (None, "unknown"):
        problems = check_compatibility(
            m,
            model_name="vidore/colSmol-500M",
            model_revision="abc123",
            query_format_id=PROD_FORMAT_ID,
            doc_prompt_sha256=value,
            index_dir=tmp_path,
        )
        assert not any("doc_prompt" in p for p in problems)


def test_mask_policy_mismatch_reported(tmp_path: Path):
    """`write-manifest --legacy`'nin yazdığı mask_policy="none" tam olarak bu
    dalı tetikler (bkz. IMPORTANT-3): padding token'ları düşürülmemiş bir
    indeks serve edilmemeli."""
    m = make_manifest(corpus_checksum=_index_stub(tmp_path), mask_policy="none")
    problems = check_compatibility(
        m,
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        query_format_id=PROD_FORMAT_ID,
        index_dir=tmp_path,
    )
    assert any("mask_policy" in p and "drop-padding" in p for p in problems)


def test_corpus_checksum_mismatch_reported(tmp_path: Path):
    """Manifest yazıldıktan SONRA meta.parquet/page_ids değişirse (kısmi
    kopya, elle düzenleme, yarım pull) sayfa kimlikleri ile embedding'ler
    kayar — bu dal onu yakalar."""
    m = make_manifest(corpus_checksum=_index_stub(tmp_path))
    (tmp_path / "meta.parquet").write_bytes(b"CHANGED")  # manifest yazıldıktan sonra değişti
    problems = check_compatibility(
        m,
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        query_format_id=PROD_FORMAT_ID,
        index_dir=tmp_path,
    )
    assert any("corpus_checksum" in p for p in problems)


def test_create_app_fails_fast_on_mismatch(tiny_corpus):
    """tiny_corpus indeksinin manifest'i kasten silinir -> create_app IndexCompatibilityError."""
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    (data_dir / "index" / "manifest.json").unlink()
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index")
    with pytest.raises(IndexCompatibilityError):
        create_app(settings=settings, encoder=enc, answerer=object())


def test_create_app_checks_injected_encoder_against_configured_format(tiny_corpus):
    """Final review IMPORTANT-2: `query_format` niteliği OLMAYAN bir encoder
    enjekte edildiğinde kontrol artık ESKİ literal'e (cpe-0.3.18) değil,
    config'ten çözülen formata karşı yapılır.

    Fikstür indeksi train-compat-v1 taşır; config cpe-0.3.18'e çevrilirse
    uyumsuzluk RAPORLANMALI (eskiden bu sessizce geçiyordu)."""
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    assert not hasattr(enc, "query_format")  # fallback yolunu gerçekten kullanıyoruz
    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        query_format_id=OTHER_FORMAT_ID,
        doc_prompt_id="processor-default",
    )
    with pytest.raises(IndexCompatibilityError, match="query_format"):
        create_app(settings=settings, encoder=enc, answerer=object())


def test_mismatch_override(tiny_corpus):
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    (data_dir / "index" / "manifest.json").unlink()
    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        allow_index_mismatch=True,
        min_score_threshold=-1e9,
    )
    app_ = create_app(settings=settings, encoder=enc, answerer=object())
    assert app_ is not None


# --- T14 korkulukları: ölçek + temsil ---------------------------------


def _int8_index_dir(tiny_corpus) -> Path:
    """tiny_corpus'un AYNI sayfalarından türetilmiş int8 indeks dizini.

    FakeEncoder deterministiktir (görüntü baytlarından seed'ler), bu yüzden
    aynı görüntülerden yeniden encode etmek aynı embedding'leri verir."""
    import shutil

    from PIL import Image

    from belge_gozu.index.float_store import FloatIndex
    from belge_gozu.index.manifest import corpus_checksum, write_manifest
    from belge_gozu.index.quantize import Int8Index

    data_dir, enc, _ = tiny_corpus
    ids = [f"d{i}:1" for i in range(3)]
    images = []
    for i in range(3):
        with Image.open(data_dir / f"images/d{i}/0001.webp") as raw:
            images.append(raw.convert("RGB"))
    i8 = Int8Index.derive(FloatIndex.build(ids, enc.encode_pages(images)))
    out = data_dir / "index-int8"
    i8.save(out)
    shutil.copy(data_dir / "meta.parquet", out / "meta.parquet")
    write_manifest(
        out,
        make_manifest(
            quantization="int8",
            corpus_checksum=corpus_checksum(out),
            n_pages=3,
            n_tokens=int(i8.offsets[-1]),
        ),
    )
    return out


def test_create_app_rejects_binary_scale_threshold(tiny_corpus):
    """Eski binary ölçeğinde (0-128) kalmış bir eşik fail-fast olmalı.

    Skorlar T14'ten sonra normalize [-1,1]; 60.0 gibi bir eşik asla
    aşılamaz, yani servis HER soruya sessizce "dayanak bulamadım" derdi.
    Ölçek geçişinin üretebileceği en sessiz hata tam olarak budur."""
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=60.0)
    with pytest.raises(IndexCompatibilityError, match="binary ölçeği"):
        create_app(settings=settings, encoder=enc, answerer=object())


def test_two_stage_on_int8_index_raises_cleanly(tiny_corpus):
    """two-stage ablasyonu YALNIZ PackedIndex ile çalışır (page_vecs gerekir).

    int8 indekste sessiz bir AttributeError yerine açık uyumsuzluk hatası."""
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir,
        index_dir=_int8_index_dir(tiny_corpus),
        retrieval_pipeline="two-stage",
        min_score_threshold=-1e9,
    )
    with pytest.raises(IndexCompatibilityError, match="two-stage"):
        create_app(settings=settings, encoder=enc, answerer=object())


def test_non_int8_index_warns_about_threshold_portability(tiny_corpus, caplog):
    """Eşik int8 DAĞILIMI üzerinde taşındı; başka temsilde doğrulanmamıştır.

    Ortak [-1,1] ölçeği temsilleri karşılaştırılabilir yapar ama dağılımlarını
    eşitlemez: aynı 0.58 canary'de int8'te 42/43, 1-bit'te 1/43 cevaplanabilir
    soruyu geçirir (review I1). Başlatma ENGELLENMEZ — temsil seçimi meşru bir
    ablasyon, per-temsil eşik config'i P2'nin işi (ruling R19) — ama sessiz de
    kalınmaz."""
    import logging

    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus  # fikstür indeksi sign-1bit
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index", min_score_threshold=-1e9)
    with caplog.at_level(logging.WARNING, logger="belge_gozu.app.main"):
        create_app(settings=settings, encoder=enc, answerer=object())
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("eşik taşınabilirlik" in w and "sign-1bit" in w for w in warnings), warnings


def test_int8_index_does_not_warn_about_threshold_portability(tiny_corpus, caplog):
    """Karşı taraf: üretim temsilinde (int8) uyarı ÇIKMAMALI — aksi halde
    uyarı gürültüye dönüşür ve anlamını yitirir."""
    import logging

    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir, index_dir=_int8_index_dir(tiny_corpus), min_score_threshold=-1e9
    )
    with caplog.at_level(logging.WARNING, logger="belge_gozu.app.main"):
        create_app(settings=settings, encoder=enc, answerer=object())
    assert not [r for r in caplog.records if "eşik taşınabilirlik" in r.getMessage()]


def test_threshold_guard_runs_before_index_load(tiny_corpus):
    """Ölçek korkuluğu SAF bir config kontrolüdür ve en başta çalışmalı
    (review M6): var olmayan bir index_dir ile bile eşik hatası vermeli —
    yani VLM/indeks yüklemesinden ÖNCE. Aksi halde yanlış yapılandırılmış bir
    eşik ancak yüzlerce MB yüklendikten sonra patlar."""
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "boyle-bir-dizin-yok",
        min_score_threshold=60.0,
    )
    with pytest.raises(IndexCompatibilityError, match="binary ölçeği"):
        create_app(settings=settings, encoder=enc, answerer=object())


def test_exhaustive_on_int8_index_serves(tiny_corpus):
    """Aynı int8 dizini VARSAYILAN (exhaustive) pipeline'da sorunsuz açılır —
    yukarıdaki hatanın temsile özgü olduğunu, int8'in genel olarak
    reddedilmediğini kanıtlar."""
    from fastapi.testclient import TestClient

    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir, index_dir=_int8_index_dir(tiny_corpus), min_score_threshold=-1e9
    )
    c = TestClient(create_app(settings=settings, encoder=enc, answerer=object()))
    assert c.get("/healthz").json()["index"]["quantization"] == "int8"
    hits = c.post("/search", json={"query": "deneme"}).json()["hits"]
    assert len(hits) == 3
    # (Bant iddiası [-1,1] burada SINANMAZ: FakeEncoder birim normlu
    # embedding üretmez — normalize ölçek testi gerçek ColPali gibi normlu
    # fikstürle tests/index/test_loader.py'de yapılır.)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_write_manifest_legacy_cli(tiny_corpus, monkeypatch):
    data_dir, _, _ = tiny_corpus
    (data_dir / "index" / "manifest.json").unlink()
    monkeypatch.setenv("BG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BG_INDEX_DIR", str(data_dir / "index"))

    result = runner.invoke(app, ["index", "write-manifest", "--legacy"])

    assert result.exit_code == 0, result.output
    from belge_gozu.index.manifest import read_manifest

    m = read_manifest(data_dir / "index")
    assert m is not None
    assert m.mask_policy == "none"


def test_write_manifest_legacy_output_still_fails_compat(tiny_corpus, monkeypatch):
    """IMPORTANT-3'ün çıkmaz sokağının kanıtı: --legacy'nin ürettiği manifest
    aynı kontrolden geçemez, bu yüzden ipucu olarak önerilemez."""
    data_dir, _, _ = tiny_corpus
    (data_dir / "index" / "manifest.json").unlink()
    monkeypatch.setenv("BG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BG_INDEX_DIR", str(data_dir / "index"))
    assert runner.invoke(app, ["index", "write-manifest", "--legacy"]).exit_code == 0

    from belge_gozu.config import Settings
    from belge_gozu.index.manifest import read_manifest

    s = Settings()
    problems = check_compatibility(
        read_manifest(data_dir / "index"),
        model_name=s.retriever_model,
        model_revision=None,
        query_format_id=s.query_format_id,
        index_dir=data_dir / "index",
    )
    assert any("mask_policy" in p for p in problems)
    assert any("query_format" in p for p in problems)
