from pathlib import Path

from belge_gozu.config import Settings


def test_defaults():
    s = Settings()
    assert s.retriever_model == "vidore/colSmol-500M"
    assert s.stage1_candidates == 200
    assert s.top_k == 5
    assert s.request_delay_s == 1.0


def test_production_index_pipeline_and_threshold_defaults():
    """Üç varsayılan BİRLİKTE kilitlenir: temsil + pipeline + o ölçeğe ait eşik.

    Bu alanlar eskiden hiç assert edilmiyordu — tam da bu yüzden bir ölçek
    kayması (indeks int8'e geçerken eşiğin binary 60.0'da kalması gibi)
    testlerde görünmezdi. P1'de eksen bir daha kaydı: eşiğin ölçeği artık
    PIPELINE'a bağlı (hibrit -> BM25 birimi), bu yüzden üçü birlikte okunur.

    Gerekçe ve ölçümler: config.py'deki yorumlar + findings
    2026-08-29-autoresearch-text-channel.md."""
    s = Settings()
    assert s.index_dir == Path("data/index-traincompat-int8")
    assert s.retrieval_pipeline == "hybrid"
    assert s.min_score_threshold == 10.6
    # ölçek korkuluğunun (app/main.py) hibrit kolda reddettiği banda düşmemeli
    assert not (0 < s.min_score_threshold <= 1.5) and s.min_score_threshold <= 200


def test_threshold_calibration_scale_matches_default_pipeline():
    """`THRESHOLD_CALIBRATED_ON` varsayılan pipeline'ın ÖLÇEĞİ olmalı.

    İkisi ayrışırsa `create_app` üretim yapılandırmasında kalıcı bir
    "taşınabilirlik uyarısı" basar ve uyarı gürültüye dönüp anlamını yitirir."""
    from belge_gozu.config import PIPELINE_SCORE_SCALE, THRESHOLD_CALIBRATED_ON

    s = Settings()
    assert PIPELINE_SCORE_SCALE[s.retrieval_pipeline] == THRESHOLD_CALIBRATED_ON
    # her pipeline'ın bir ölçek künyesi olmalı (yeni bir kol eklenip
    # burada unutulursa korkuluk KeyError'a düşerdi)
    assert set(PIPELINE_SCORE_SCALE) == {"hybrid", "exhaustive", "two-stage"}


def test_env_override(monkeypatch):
    monkeypatch.setenv("BG_TOP_K", "3")
    assert Settings().top_k == 3


def test_env_file_and_google_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("BG_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=test-key-123\n")
    assert Settings().gemini_api_key == "test-key-123"
