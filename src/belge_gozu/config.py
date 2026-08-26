from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    retriever_model: str = "vidore/colSmol-500M"
    device: str = "auto"
    hf_dataset_repo: str = ""
    # gemini-2.0-flash Task 13'te canlı çağrıda 404 döndü (API: "no longer
    # available... use models/gemini-3.6-flash"); Task 13 canlı doğrulamasında güncellendi.
    gemini_model: str = "gemini-3.6-flash"
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("BG_GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    stage1_candidates: int = 200
    top_k: int = 5
    # exhaustive: her arama tüm korpusu tarar (kesin, varsayılan). two-stage:
    # mean-sign Hamming ile aday eleme + kesin MaxSim (ablasyon-only; spec §1.1
    # karşı-örneği nedeniyle üretimde kullanılmaz).
    retrieval_pipeline: Literal["exhaustive", "two-stage"] = "exhaustive"
    # kaba v0 ayarı; gerçek kalibrasyon Plan 2 (Task 13 smoke test: gerçek soru
    # top_score~70.6, saçma soru top_score~52.4 -- 20.0 hiçbir zaman tetiklemiyordu)
    min_score_threshold: float = 60.0
    request_delay_s: float = 1.0
    # Tahmini birim fiyatlar (USD / 1M token). Kesin değildir; runbook'taki
    # doğrulama adımıyla güncellenir, env ile geçersiz kılınır.
    gemini_price_in_usd_per_1m: float = 0.10
    gemini_price_out_usd_per_1m: float = 0.40
    # Ham sorgu metnini events tablosuna yaz (gizlilik hassasiyeti olan
    # dağıtımlarda kapatılabilir; sha256 her koşulda yazılır).
    log_query_text: bool = True
    # indeks/serve uyumsuzluğunda fail-fast yerine uyarı ile devam et (bilinçli
    # bir riske girildiğinde kullanılır; varsayılan olarak kapalıdır).
    allow_index_mismatch: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
