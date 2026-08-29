from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# `min_score_threshold`ın ÜZERİNDE ÖLÇÜLDÜĞÜ temsil. Eşik dağılıma bağlıdır,
# ölçeğe değil: aynı normalize [-1,1] bandında bile 0.58 int8'te 42/43,
# 1-bit'te 1/43 cevaplanabilir soruyu geçirir. `app/main.py` yüklü indeks
# bundan farklıysa uyarır (bkz. aşağıdaki eşik yorumu).
THRESHOLD_CALIBRATED_ON = "int8"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("data")
    # ÜRETİM İNDEKSİ. İki eksende de ölçümle seçildi:
    #
    # 1) Sorgu/doküman formatı (T11/Step 6 A/B, 2026-08-27): train-compat-v1 +
    #    eğitim zamanı doc prompt, cpe-0.3.18'e karşı kazandı — float Recall@5
    #    0.093->0.233, Recall@20 0.186->0.302 (ayrıntı p0-gate raporunda).
    # 2) Kuantizasyon (C1/C2 ablasyonu, 2026-08-27/29; T14'te üretime alındı):
    #    int8, float16 ile HER k'da birebir aynı kalite (R@1/5/20/50/200);
    #    1-bit float16'ya göre R@20'de 7.0 puan KAYBEDİYOR (0.233 vs 0.302) ve
    #    üstelik 4.3x DAHA YAVAŞ (CPU, 4222 sayfa: int8 0.24 sn/sorgu vs 1-bit
    #    1.08 sn/sorgu — int8/f16 BLAS matmul yoluna girerken 1-bit popcount
    #    indirgemesi için büyük geçici diziler kuruyor). 1-bit'in kazandığı tek
    #    eksen disk: 58 MB vs int8 474 MB vs f16 919 MB. int8 ölçümün kazananıydı
    #    ama servis tarafı yalnız PackedIndex yükleyebildiği için üretimde
    #    kullanılamıyordu; T14 o eksik bağlantıyı kurdu (ruling R16/D1).
    #    1-bit ablasyon/disk-bütçesi seçeneği olarak duruyor
    #    (data/index-traincompat-1bit).
    index_dir: Path = Path("data/index-traincompat-int8")
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
    # MEKANİK ÖLÇEK TAŞIMASI — KALİBRASYON DEĞİL.
    #
    # T14'te skorlar tek bir normalize ölçeğe alındı: sorgu jetonu başına
    # ortalama MaxSim, ~[-1,1] (binary kol EMBED_DIM'e bölünerek int8/float16
    # dot-product bandına taşındı). Eski eşik 60.0 ESKİ binary ölçeğindeydi
    # (0-128) ve yeni ölçekte hiçbir zaman aşılamazdı.
    #
    # 0.58, o eski 60.0'ın ÇALIŞMA NOKTASINI SAYICA yeniden üretir: canary'de
    # binary@60.0 cevaplanabilirlerin 42/43'ünü ve cevaplanamazların 4/5'ini
    # geçiriyordu; int8'te 0.58 de aynı sayıları verir (0.5767 kalır, 0.5860+
    # geçer; 0.5679 kalır, kalan 4 geçer). Yani bu bir dönüştürme, yeni bir
    # karar DEĞİL.
    #
    # AMA soru-soruya AYNI KÜME değil (review I3): iki satır taraf değiştirir —
    # c306 (1-bit ham 59.85 -> eşiğin altındaydı; int8 0.5965 -> artık geçiyor)
    # ve c211 (1-bit ham 61.78 -> geçiyordu; int8 0.5767 -> artık altında).
    # int8 ile binary aynı soruları aynı sırayla skorlamadığı için beklenen bir
    # sonuç; sayı korunur, kimlikler birebir korunmaz.
    #
    # TAŞINABİLİRLİK: eşik int8 DAĞILIMI üzerinde taşınmıştır; başka bir
    # temsile geçerken (BG_INDEX_DIR ya da two-stage ablasyonu) yeniden taşıma
    # ölçümü gerekir — ortak [-1,1] ölçeği temsilleri karşılaştırılabilir
    # yapar, dağılımlarını eşitlemez. Ölçüm: aynı canary'de 1-bit top-1'leri
    # min 0.4676 / medyan 0.4953 / maks 0.6133, yani 0.58 orada 43 sorunun
    # yalnız 1'ini geçirir (aynı çalışma noktası 1-bit'te ~0.47'ye denk gelir).
    # `create_app` int8 dışı bir temsil yüklendiğinde UYARI loglar.
    #
    # Skor hâlâ bir güven/olasılık ölçüsü DEĞİLDİR ve eşik hâlâ AYIRMIYOR:
    # 2026-08-29 canary ölçümü (data/index-traincompat-int8, MPS) cevaplanabilir
    # n=43 min 0.5767 / medyan 0.6250 / maks 0.7450; cevaplanamaz n=5 min 0.5679
    # / medyan 0.6550 / maks 0.6866 — dağılımlar hâlâ iç içe, korpus-dışı
    # sorular hâlâ eşiği geçiyor. Artefakt:
    # data/bench/results/int8-threshold-transfer.json. Bu durum
    # tests/retrieval/test_semantic_canary.py::
    # test_out_of_corpus_canary_scores_below_threshold ile xfail(strict) olarak
    # kilitlidir. Gerçek kalibrasyon P2'nin işi.
    min_score_threshold: float = 0.58
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
