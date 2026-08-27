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
    # T11/Step 6 A/B ölçümü: float Recall@5 0.093->0.233, Recall@20 0.186->0.302
    # (train-compat-v1 + eğitim zamanı doc prompt, cpe-0.3.18'e karşı kazandı;
    # ölçüm tarihi 2026-08-27; ayrıntı p0-gate raporunda). Üretim indeksi artık bu.
    index_dir: Path = Path("data/index-traincompat-1bit")
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
    # T11/Step 6 A/B ölçümü: train-compat-v1 sorgu formatı cpe-0.3.18'i her
    # metrikte geçti (float R@5 0.093->0.233; ölçüm tarihi 2026-08-27; ayrıntı
    # p0-gate raporunda). CLI'nin QueryFormatChoice değerleriyle birebir aynı.
    query_format_id: str = "train-compat-v1"
    # T11/Step 6 A/B ölçümü: kazanan indeks eğitim-zamanı doküman prompt'uyla
    # (TRAIN_COMPAT_DOC_PROMPT) inşa edildi; ölçüm tarihi 2026-08-27, ayrıntı
    # p0-gate raporunda. CLI'nin DocPromptChoice değerleriyle birebir aynı.
    doc_prompt_id: Literal["processor-default", "train-compat"] = "train-compat"
    stage1_candidates: int = 200
    top_k: int = 5
    # exhaustive: her arama tüm korpusu tarar (kesin, varsayılan). two-stage:
    # mean-sign Hamming ile aday eleme + kesin MaxSim (ablasyon-only; spec §1.1
    # karşı-örneği nedeniyle üretimde kullanılmaz).
    retrieval_pipeline: Literal["exhaustive", "two-stage"] = "exhaustive"
    # kaba v0 kalıntısı (Task 13 smoke test: gerçek soru top_score~70.6, saçma soru
    # top_score~52.4 -- 20.0 hiçbir zaman tetiklemiyordu). Bu skor bir güven/olasılık
    # ölçüsü DEĞİLDİR (bkz. README "v0 limitations"); gerçek kalibrasyon P2'nin işi.
    # DİKKAT: yukarıdaki 70.6/52.4 rakamları T11 format değişikliğinden (train-compat-v1
    # + train-compat doküman prompt'u, yeni indeks data/index-traincompat-1bit) ÖNCE
    # ölçüldü; bugünkü skor dağılımını TEMSİL ETMİYORLAR. 2026-08-27 canary ölçümü:
    # cevaplanabilir n=43 min 59.85 / medyan 63.40 / maks 78.50, cevaplanamaz n=5
    # min 59.65 / medyan 67.88 / maks 71.95 -> dağılımlar iç içe, 60.0 (ya da başka
    # herhangi bir tek eşik) bu ikisini AYIRMIYOR; korpus-dışı üç soru da eşiği geçiyor.
    # Bu durum tests/retrieval/test_semantic_canary.py::
    # test_out_of_corpus_canary_scores_below_threshold ile xfail(strict) olarak
    # kilitlidir. Kalibrasyon (ve muhtemelen skor normalizasyonu) P2'nin işi.
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
