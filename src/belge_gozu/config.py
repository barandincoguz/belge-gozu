from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from belge_gozu.index.manifest import QueryFormatChoice

# Skor ÖLÇEĞİ adları. Bu iki dize projedeki tek ölçek kimliğidir: korkuluk,
# taşınabilirlik uyarısı ve telemetri yönlendirmesi hepsi bunlara bakar.
BM25_SCALE = "hybrid-bm25"  # BM25 birimi, üst sınırsız (ölçülen bant ~4-70)
VISUAL_SCALE = "visual-normalized"  # sorgu jetonu başına ortalama MaxSim, ~[-1,1]

# `min_score_threshold`ın ÜZERİNDE TAŞINDIĞI SKOR ÖLÇEĞİ. P1'de eşiğin
# bağlandığı eksen KUANTİZASYONDAN PIPELINE'A geçti: hibrit yolda sıralamayı
# ve dolayısıyla eşikle karşılaştırılan skoru BM25 metin kanalı üretir
# — int8/1-bit ayrımı bu skoru etkilemez. `app/main.py` etkin pipeline'ın
# ölçeği bundan farklıysa UYARIR ve ölçek dışı bir eşikte fail-fast yapar.
THRESHOLD_CALIBRATED_ON = BM25_SCALE

# Hangi pipeline hangi skor ölçeğinde skorlar. Ölçek karışımı bu projenin en
# sessiz hata sınıfıdır (T14: binary 0-128 -> normalize [-1,1]); TEK yerde
# tutulur ki korkuluk, uyarı ve telemetri yönlendirmesi aynı kaynağa baksın —
# `telemetry/prom.py` bu sözlükten TÜRETİR, kopya tutmaz (review M1).
PIPELINE_SCORE_SCALE: dict[str, str] = {
    "hybrid": BM25_SCALE,
    "exhaustive": VISUAL_SCALE,
    "two-stage": VISUAL_SCALE,
}


def pipelines_on_scale(scale: str) -> frozenset[str]:
    """Verilen skor ölçeğinde skorlayan pipeline adları (tek kaynaktan türetilir)."""
    return frozenset(p for p, s in PIPELINE_SCORE_SCALE.items() if s == scale)


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
    # p0-gate raporunda).
    #
    # TİP = ENUM, düz `str` DEĞİL (audit C8): `resolve_formats` bu değeri
    # `QueryFormatChoice(...)` ile aramaya sokuyor, yani geçersiz bir
    # BG_QUERY_FORMAT_ID düz `str`'de config katmanından SESSİZCE geçip
    # uygulama kurulumunun ortasında ham bir ValueError'a dönüşüyordu. Enum
    # olarak pydantic'in kendisi başlangıçta temiz bir ValidationError verir
    # ("Input should be 'cpe-0.3.18' or 'train-compat-v1'").
    query_format_id: QueryFormatChoice = QueryFormatChoice.train_compat_v1
    # T11/Step 6 A/B ölçümü: kazanan indeks eğitim-zamanı doküman prompt'uyla
    # (TRAIN_COMPAT_DOC_PROMPT) inşa edildi; ölçüm tarihi 2026-08-27, ayrıntı
    # p0-gate raporunda. CLI'nin DocPromptChoice değerleriyle birebir aynı.
    doc_prompt_id: Literal["processor-default", "train-compat"] = "train-compat"
    stage1_candidates: int = 200
    top_k: int = 5
    # hybrid (VARSAYILAN, P1): sıralamayı PDF metin katmanı üzerindeki BM25 +
    # doküman-adı pencere-içi yönlendirmesi belirler; görsel MaxSim kanalı
    # koşmaya devam eder ama sıralamaya girmez (telemetri + P2 kalibrasyon
    # verisi). Ölçüm (canary answerable n=43, autoresearch exp7 -> exp8 ->
    # exp12): R@5 0.2326 -> 0.8140 -> 0.8372 -> 0.8605 (ikili tanım; 37/43),
    # R@20 0.302 -> 0.9302, vitrin sorgularının gold sıraları 664->2 ve 137->2
    # — findings 2026-08-29-autoresearch-text-channel.md + research/journal.md
    # #8 ve #12. exp12 (ASCII aksan katlama) sistemi YAZIM-DEĞİŞMEZ yapar:
    # aksanlı ve aksansız yazılan aynı sorgu aynı R@5'i (0.8605) verir.
    # exhaustive: yalnız görsel kanal,
    # her arama tüm korpusu tarar (P0 üretim yolu; artık ablasyon/karşılaştırma
    # kolu). two-stage: mean-sign Hamming ile aday eleme + kesin MaxSim
    # (ablasyon-only; spec §1.1 karşı-örneği nedeniyle üretimde kullanılmaz).
    #
    # DİKKAT: pipeline değiştirmek SKOR ÖLÇEĞİNİ değiştirir (bkz.
    # PIPELINE_SCORE_SCALE) — `min_score_threshold` da birlikte taşınmalıdır.
    retrieval_pipeline: Literal["hybrid", "exhaustive", "two-stage"] = "hybrid"
    # MEKANİK ÖLÇEK TAŞIMASI — KALİBRASYON DEĞİL.
    #
    # P1'de sıralamayı BM25 metin kanalı üretiyor, yani eşiğin karşılaştırdığı
    # skor artık normalize [-1,1] MaxSim değil BM25 birimi (üst sınırsız).
    # Eski 0.58 bu ölçekte her soruyu geçirirdi — tıpkı T14 öncesi 60.0'ın yeni
    # ölçekte hiçbir soruyu geçirmemesi gibi, aynı hatanın simetriği.
    #
    # 10.6, T14'ün 0.58'i gibi, bir öncekinin ÇALIŞMA NOKTASINI SAYICA yeniden
    # üretir. Ölçüm — hepsi SERVİS EDİLEN top-1 üzerinde, yani `AskService`in
    # eşikle karşılaştırdığı gerçek skor üzerinde (canary, BM25 ölçeği).
    #
    # ASCİİ AKSAN KATLAMASINDAN SONRA YENİDEN ÖLÇÜLDÜ (exp12; katlama IDF/df
    # dağılımını değiştirdiği için eşik bandı da yeniden ölçülmek ZORUNDAYDI):
    # cevaplanabilir n=43 min 10.5265 / 2. en küçük 10.7115 / medyan 23.7780 /
    # maks 66.6822; cevaplanamaz top-1'ler 4.23 (c006 anlamsız), 12.96 (c004),
    # 15.54 (c007), 17.86 (c005), 23.52 (c003). binary@60 / int8@0.58'in çalışma
    # noktası "42/43 cevaplanabilir + 4/5 cevaplanamaz geçer"di; katlama sonrası
    # o noktayı veren bant (10.5265, 10.7115] ve 10.6 HÂLÂ bu bandın İÇİNDE —
    # yani eşik DEĞİŞMEDİ, yeniden doğrulandı (42/43 + 4/5, ölçüm bu sprintte
    # tekrarlandı). Kaynak: research/journal.md #12 + vitrin sprintinin
    # doğrulama ledger'ı. Katlama öncesi bant (10.528, 10.712] idi — sayılar
    # kıl payı kaydı, çalışma noktası aynı kaldı. Yani mekanik ölçek taşıması,
    # kalibrasyon değil.
    #
    # SERVİS EDİLEN vs KANAL top-1 (review L1): pencere-içi yönlendirme
    # sıralamanın BİRİNCİSİNİ skora göre değil sorguda adı geçen kanuna göre
    # seçebilir, yani eşiğe giren sayfa kanalın en yüksek skorlu sayfası
    # OLMAYABİLİR. Kanalın kendi top-1 medyanı 26.05 (telemetride
    # `detail.retrieval.bm25_top1`); yukarıdaki bant ve çalışma noktası
    # kanaldan değil SERVİS EDİLEN skordan ölçüldü. (Pencere 20 -> 50 exp8
    # değişikliği bu dağılımı değiştirmedi: aynı min/2./medyan/maks.)
    #
    # TAŞINABİLİRLİK: eşik artık KUANTİZASYONA değil PIPELINE'a bağlı (bkz.
    # PIPELINE_SCORE_SCALE). BG_RETRIEVAL_PIPELINE=exhaustive/two-stage'e
    # geçmek skoru görsel normalize bandına geri döndürür ve 10.6 orada asla
    # aşılamaz; o kollarda eşik yeniden taşınmalıdır (P0 değeri 0.58'di).
    # `create_app` ölçek dışı bir eşikte FAIL-FAST yapar, ölçek uyuşmazlığında
    # UYARI loglar.
    #
    # Skor hâlâ bir güven/olasılık ölçüsü DEĞİLDİR ve eşik hâlâ AYIRMIYOR:
    # yukarıdaki iki dağılım hâlâ iç içe (cevaplanabilir alt sınır 10.53,
    # cevaplanamaz üst sınır 23.53 — üç korpus-dışı soru eşiğin ÜSTÜNDE).
    # Eşiği yükseltmek de çözüm değil: cevaplanabilir dağılımın alt yarısı
    # birlikte abstain'e düşer. Bu durum tests/retrieval/test_semantic_canary.py::
    # test_out_of_corpus_canary_scores_below_threshold ile xfail(strict) olarak
    # kilitlidir. Gerçek kalibrasyon P2'nin işi.
    min_score_threshold: float = 10.6
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
    # İSTEMCİ-IP BAŞINA dakikalık istek tavanı; 0 = KAPALI (varsayılan).
    #
    # Varsayılanın kapalı olması bilinçli: yerelde tek kullanıcı çalışırken bir
    # hız sınırı yalnız gürültü üretir, ayrıca bench/canlı doğrulama koşumları
    # kendi kendini 429'a düşürürdü. AÇIK varsayılanlar DAĞITIM katmanında
    # (Dockerfile: ask 10/dk, search 60/dk) — yani "herkese açık dağıtım"
    # kararı, kütüphane davranışı değil.
    #
    # /ask sınırı /search'ten çok daha düşük çünkü maliyeti farklı: /search saf
    # yerel hesap (BM25 ms + encode ~1 sn), /ask ise ÜCRETLİ bir LLM çağrısı.
    # Sınır kotayı korumak içindir, kötü niyeti durdurmak için değil — süreç
    # içi ve IP başına olduğu için ne kalıcıdır ne de dağıtık.
    rate_limit_ask_per_min: int = 0
    rate_limit_search_per_min: int = 0

    # --- P2 İKİ KAPI (T1+T2) — İKİSİ DE VARSAYILAN KAPALI --------------------
    #
    # Master §1'in kapı kuralı (`:38-40`): "bir sonraki fazın hiçbir
    # default-açık entegrasyonu önceki kapı raporu commit'lenmeden yapılamaz."
    # P1'in resmî G1 koşumu YOKTUR (R23 ile kapsam daraltıldı,
    # `p2-reality-audit.md §6`), bu yüzden bu iki bayrağın varsayılan-kapalı
    # olması bir tercih değil, kuralın kendisidir. Açık bir bayrak +
    # `BG_GATE_*=false` ile anında geri alma, entegrasyonun tek meşru biçimi.
    #
    # gate_calibrated: kalibre GETİRİM kapısı. `data/calibration/<anahtar>/`
    #   altındaki artefaktı yükler ve `p < tau` ise çekimser kalır. Artefakt
    #   yok/anahtar uyuşmuyorsa başlangıçta FAIL-FAST (sessizce kalibrasyonsuz
    #   açılmaz). ÖLÇÜLMÜŞ BEDEL: bugünkü artefaktta tau=0.504 ve dev kapsaması
    #   %2.2 — yani bu bayrak açıkken sistem soruların çok büyük çoğunluğunda
    #   çekimser kalır. Bu bir arıza değil, retrieval-yanı güvenin tek başına
    #   yetmediğinin ölçüsüdür (`2026-08-30-p2-baslangic.md §4`).
    # gate_verifier: KANIT kapısı. Üretilen yanıtın her iddiası atıf yaptığı
    #   sayfanın METNİNE karşı LLM ile doğrulanır; desteklenmeyen tek iddia
    #   yanıtı düşürür. İddia başına bir Gemini çağrısıdır (sha256 önbellekli)
    #   — kotayı doğrudan etkiler, `verifier_max_claims` üst sınırdır.
    gate_calibrated: bool = False
    gate_verifier: bool = False
    # İstek başına doğrulanacak EN FAZLA iddia (kota tavanı). Aşan iddialar
    # doğrulanmaz VE yanıt düşürülür — kırpmayı sessizce "geçti" saymak ilke
    # 20'nin yasakladığı şeydir.
    verifier_max_claims: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
