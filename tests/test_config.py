from pathlib import Path

from belge_gozu.config import Settings


def test_defaults():
    s = Settings()
    assert s.retriever_model == "vidore/colSmol-500M"
    assert s.stage1_candidates == 200
    assert s.top_k == 5
    assert s.request_delay_s == 1.0


def test_production_index_and_threshold_defaults():
    """İki varsayılan BİRLİKTE kilitlenir: temsil + o temsile ait eşik.

    Bu iki alan eskiden hiç assert edilmiyordu — tam da bu yüzden bir ölçek
    kayması (indeks int8'e geçerken eşiğin binary 60.0'da kalması gibi)
    testlerde görünmezdi. İkisi birbirine bağlıdır: eşik yalnız normalize
    [-1,1] skor ölçeğinde anlamlıdır ve o ölçek indeksin temsiliyle gelir.

    Gerekçe ve ölçümler: config.py'deki yorumlar +
    data/bench/results/int8-threshold-transfer.json."""
    s = Settings()
    assert s.index_dir == Path("data/index-traincompat-int8")
    assert s.min_score_threshold == 0.58
    # ölçek korkuluğunun (app/main.py) reddettiği banda düşmemeli
    assert s.min_score_threshold <= 1.5


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
