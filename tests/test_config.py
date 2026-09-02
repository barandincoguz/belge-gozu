from pathlib import Path

import pytest
from pydantic import ValidationError

from belge_gozu.config import Settings


def test_defaults():
    s = Settings()
    assert s.retriever_model == "vidore/colSmol-500M"
    assert s.stage1_candidates == 200
    assert s.top_k == 5
    assert s.request_delay_s == 1.0


def test_rate_limits_are_off_by_default():
    """0 = kapalı. Açık varsayılanlar DAĞITIM katmanında (Dockerfile), burada değil:
    yerel tek kullanıcı ve bench koşumları kendi kendini 429'a düşürmemeli."""
    s = Settings()
    assert s.rate_limit_ask_per_min == 0
    assert s.rate_limit_search_per_min == 0


def test_invalid_query_format_id_fails_at_config_time(monkeypatch):
    """Geçersiz BG_QUERY_FORMAT_ID config katmanında temiz bir hata verir.

    Alan düz `str` olduğunda (audit C8) bu değer Settings'ten SESSİZCE geçip
    uygulama kurulumunun ortasında ham bir ValueError'a dönüşüyordu; enum tipi
    hatayı doğru katmana, okunur bir mesajla taşır."""
    monkeypatch.setenv("BG_QUERY_FORMAT_ID", "bogus-format")
    with pytest.raises(ValidationError, match="query_format_id"):
        Settings()


def test_query_format_id_is_the_enum_value(monkeypatch):
    """Geçerli değer enum'a çözülür ama dize gibi de kullanılabilir (StrEnum)."""
    from belge_gozu.index.manifest import QUERY_FORMATS, QueryFormatChoice

    s = Settings()
    assert s.query_format_id is QueryFormatChoice.train_compat_v1
    assert s.query_format_id == "train-compat-v1"
    assert QUERY_FORMATS[s.query_format_id]  # doğrudan sözlük anahtarı olarak çalışır


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


def test_hf_token_accepts_standard_hugging_face_env_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    assert Settings().hf_token == "secret-token"


def test_hf_revision_uses_bg_prefix(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BG_HF_REVISION", "a" * 40)
    assert Settings().hf_revision == "a" * 40


def test_env_file_and_google_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("BG_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=test-key-123\n")
    assert Settings().gemini_api_key == "test-key-123"


# --- ikinci (yedek) anahtar: anahtar rotasyonunun kaynağı ---------------------
#
# Anahtar DEĞERLERİ testlerde apaçık sahtedir; gerçek bir anahtar dizesi
# depoya giremez.

_KEY2_ENVS = ("GOOGLE_API_KEY_2", "BG_GOOGLE_API_KEY_2", "GEMINI_API_KEY_2")


def _isolated_env(monkeypatch, tmp_path, dotenv: str = "") -> None:
    """Süreç ortamından ve depo kökündeki gerçek `.env`ten YALITILMIŞ ayar."""
    for name in _KEY2_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(dotenv)


def test_second_key_is_read_from_the_env_file(monkeypatch, tmp_path):
    """`extra="ignore"` yüzünden BEYAN EDİLMEYEN bir ortam değişkeni sessizce
    yok sayılır — ".env'e yazdım ama hiçbir şey olmadı" sınıfı arıza. Bu test
    `GOOGLE_API_KEY_2` -> `Settings.google_api_key_2` eşlemesini kilitler."""
    _isolated_env(monkeypatch, tmp_path, "GOOGLE_API_KEY_2=sahte-yedek-anahtar\n")
    assert Settings().google_api_key_2 == "sahte-yedek-anahtar"


def test_second_key_env_override(monkeypatch, tmp_path):
    _isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("GOOGLE_API_KEY_2", "sahte-ortam-anahtari")
    assert Settings().google_api_key_2 == "sahte-ortam-anahtari"


def test_second_key_defaults_to_empty(monkeypatch, tmp_path):
    """Boş = tek anahtarlı havuz = bugünkü davranış (rotasyon KAPALI)."""
    _isolated_env(monkeypatch, tmp_path)
    assert Settings().google_api_key_2 == ""
