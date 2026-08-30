import hashlib
import logging
import math
import sqlite3
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from belge_gozu.answer.base import Answer, AskService, is_honest_miss
from belge_gozu.config import (
    PIPELINE_SCORE_SCALE,
    THRESHOLD_CALIBRATED_ON,
    Settings,
    get_settings,
)
from belge_gozu.index.compat import IndexCompatibilityError, check_compatibility
from belge_gozu.index.loader import load_scorable_index
from belge_gozu.index.manifest import (
    DOC_PROMPTS,
    QUERY_FORMATS,
    DocPromptChoice,
    IndexManifest,
    QueryFormat,
    QueryFormatChoice,
)
from belge_gozu.index.store import PackedIndex
from belge_gozu.retrieval.core import ExhaustiveRetriever, TwoStageRetriever
from belge_gozu.retrieval.hybrid import HybridRetriever, load_text_channel, require_text_artifact
from belge_gozu.retrieval.text import tokenize
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import StageCollector, collecting
from belge_gozu.telemetry.prom import PromMetrics
from belge_gozu.telemetry.recorder import EventRecorder
from belge_gozu.telemetry.schema import RequestEvent

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Sorgu uzunluk tavanı (karakter). Sondaj bulgusu (2026-08-30): 3000 karakterlik
# bir sorgu BM25 skorunu ~1053'e çıkarıyordu — yani eşik (10.6) böyle bir
# girdide ANLAMSIZ hale geliyor, üstelik uzun metin doğrudan LLM istemine de
# giriyordu. 500 karakter gerçek mevzuat sorularının çok üstünde (canary'nin en
# uzunu ~180) ama skor şişirmeyi ve istem enjeksiyonu yüzeyini kapatıyor.
MAX_QUERY_CHARS = 500

# `k` tavanı: sondajda `k=100000` tüm korpusu (4222 sayfa) tek yanıtta
# döküyordu, `k=-1` 4221 sonuç veriyordu, `k=0` sessizce varsayılana düşüyordu.
# ge=1 ve le=50 üçünü birden kapatır (50, top_k=5'in on katı — teşhis için
# fazlasıyla yeterli).
MAX_K = 50

# Boş-içerik reddi. `tokenize` işlev kelimelerini ve tek harfli parçaları
# eledikten sonra hiçbir token kalmıyorsa ortada ARANACAK bir şey yoktur:
# BM25 tüm korpusa 0 verir ve servis rastgele 5 sayfayı skor 0 ile döndürürdü
# (sondaj bulgusu: "", "   ", "bu ne için" hepsi böyle davranıyordu). 422 bunu
# sessiz saçmalık yerine açık bir girdi hatasına çevirir.
EMPTY_QUERY_DETAIL = "sorgu boş ya da yalnız işlev kelimeleri içeriyor"

# Reddedilen isteklerin (`status='rejected'`) `error_type` değerleri. HTTP
# kodundan TÜRETİLİR (aşağıda `_REJECT_REASONS`), elle yazılmaz.
REJECT_VALIDATION = "validation"
REJECT_RATE_LIMITED = "rate_limited"
_REJECT_REASONS = {422: REJECT_VALIDATION, 429: REJECT_RATE_LIMITED}


class SearchBody(BaseModel):
    query: str = Field(..., max_length=MAX_QUERY_CHARS)
    k: int | None = Field(None, ge=1, le=MAX_K)


class AskBody(BaseModel):
    question: str = Field(..., max_length=MAX_QUERY_CHARS)


def require_searchable(text: str) -> None:
    """Aranabilir içerik yoksa 422 (Türkçe detay) — /search ve /ask'te AYNI kural.

    Kontrol üretim tokenleştiricisinin KENDİSİYLE yapılır (`retrieval/text.py`),
    ayrı bir "boşluk mu?" sezgisiyle değil: eleme kuralları (>=2 harf, katlanmış
    işlev-kelime listesi) değişirse bu kapı da onunla birlikte değişsin.

    L1 (review 2026-08-30, DİSPUTED — bilerek DEĞİŞTİRİLMEDİ): görsel-yalnız
    pipeline'larda (`exhaustive`/`two-stage`) sıralamayı BM25 hiç kurmaz, yani
    bu kapı orada da metin kanalının kurallarına göre reddeder. İçeriksiz
    sorgu reddi ÜRÜN-DÜZEYİ bir kuraldır, pipeline'dan BAĞIMSIZDIR (kontrolcü
    kararı): "bu ne için" hiçbir kolda anlamlı bir yanıt üretmez.
    """
    if not tokenize(text):
        raise HTTPException(422, detail=EMPTY_QUERY_DETAIL)


# `RateLimiter._hits` sözlüğünün tavanı (review M1, 2026-08-30). Süpürme
# (aşağıda `_evict_expired`) yalnız süresi DOLMUŞ pencereleri temizler; kasıtlı
# olarak HER İSTEKTE FARKLI bir kaynak IP kullanan bir sel (IPv6'da ayrı kaynak
# adres üretmek bedavaya yakındır) pencere içinde kalarak sözlüğü sınırsız
# büyütebilirdi. Tavan bu senaryoyu kapatır.
RATE_LIMITER_MAX_CLIENTS = 10_000


class RateLimiter:
    """İSTEMCİ-IP başına kayan pencere (60 sn), süreç içi. 0 = kapalı.

    Kasıtlı olarak basit ve KALICI DEĞİL: tek süreçte, bellekte, yeniden
    başlatmada sıfırlanır. Amacı kötü niyeti durdurmak değil, herkese açık bir
    demoda LLM kotasını ve GPU'yu kazara tüketilmekten korumak.

    İstemci kimliği `request.client.host`tur; `X-Forwarded-For` BİLEREK
    okunmuyor — doğrulanmamış bir başlığa güvenmek sınırı tek satırlık bir
    sahtecilikle atlatılabilir kılar. Bedeli dürüstçe şudur: ters vekil
    ARKASINDA çalışırken bütün istekler aynı IP'den görünür ve sınır küresel bir
    tavana dönüşür (dağıtım tarafında düşük ama sıfır olmayan bir değerle
    yaşanabilir bir davranış).

    Bellek (review M1): `_hits` iki mekanizmayla sınırlı tutulur — (1) her
    `check()` çağrısı ÖNCE tüm sözlüğü tarar ve penceresi tamamen dolmuş (en
    son isteği bile `window_s`'ten eski) her istemciyi siler, (2)
    `max_clients` tavanı dolduğunda yeni bir istemci eklenmeden önce en son
    etkinliği en eski olan istemci düşürülür. Birlikte hem sessiz büyümeyi
    (binlerce tek-seferlik IP kalıcı girdiye dönüşmez) hem sahte-IP selini
    (her istekte yeni bir IP tavanı asla aşamaz) kapatırlar.
    Varsayılan-kapalı yol (`per_min<=0`) hiç durum tutmadığı için ikisinden de
    etkilenmez.
    """

    def __init__(
        self,
        per_min: int,
        window_s: float = 60.0,
        max_clients: int = RATE_LIMITER_MAX_CLIENTS,
    ) -> None:
        self.per_min = per_min
        self.window_s = window_s
        self.max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> float | None:
        """None = geç; float = kaç saniye sonra tekrar denenmeli (Retry-After)."""
        if self.per_min <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            q = self._hits.get(client)
            if q is None:
                if len(self._hits) >= self.max_clients:
                    self._evict_oldest()
                q = deque()
                self._hits[client] = q
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.per_min:
                return max(1.0, self.window_s - (now - q[0]))
            q.append(now)
            return None

    def _evict_expired(self, now: float) -> None:
        """Penceresi tamamen dolmuş istemcileri siler (M1: kalıcı-girdi sızıntısı).

        Bir istemcinin EN SON isteği bile `window_s`'ten eskiyse deque'deki
        tüm zaman damgaları zaten süresi dolmuş demektir — anahtar burada
        düşürülür (aksi halde `check` yalnız KENDİ istemcisinin deque'ini
        budadığı için bir daha hiç dönmeyen istemciler sonsuza dek kalırdı)."""
        stale = [c for c, q in self._hits.items() if not q or now - q[-1] >= self.window_s]
        for c in stale:
            del self._hits[c]

    def _evict_oldest(self) -> None:
        """Tavan (`max_clients`) dolduğunda EN SON etkinliği en eski olan
        istemciyi düşürür — sahte-IP seli (her istekte yeni bir IP) sözlüğü
        bu noktadan sonra büyütemez."""
        oldest = min(self._hits, key=lambda c: self._hits[c][-1])
        del self._hits[oldest]


def enforce_rate_limit(
    limiter: RateLimiter, request: Request, endpoint: str, prom: PromMetrics
) -> None:
    client = request.client.host if request.client else "bilinmeyen"
    retry_after = limiter.check(client)
    if retry_after is None:
        return
    # L3: 429'lar telemetri olayına (collecting/record_event) hiç girmez —
    # getirici çağrılmadı, tam bir RequestEvent üretmenin karşılığı yok. Ama
    # sınırlayıcının ÇALIŞTIĞINI gösteren tek bir sayaç olmalı, aksi halde bir
    # kötüye kullanım dalgası `/metrics`'te de `/stats`'ta da tamamen sessiz
    # geçer (422'ler kasıtlı olarak bu sayacın dışında kalır — framework
    # düzeyinde, ayrı bir karar).
    prom.rate_limited.labels(endpoint=endpoint).inc()
    raise HTTPException(
        429,
        detail=(
            f"çok fazla istek: {endpoint} için dakikada {limiter.per_min} sorgu sınırı "
            f"aşıldı, {int(retry_after)} saniye sonra tekrar deneyin"
        ),
        headers={"Retry-After": str(int(retry_after))},
    )


def resolve_formats(s: Settings) -> tuple[QueryFormat, str | None]:
    """Config'ten (Settings) çözülen (sorgu formatı, doküman prompt'u).

    CLI'nin `index build` sırasında kullandığı QUERY_FORMATS/DOC_PROMPTS
    sözlükleriyle aynı kaynak (belge_gozu.index.manifest) — T11/Step 6."""
    return (
        QUERY_FORMATS[QueryFormatChoice(s.query_format_id)],
        DOC_PROMPTS[DocPromptChoice(s.doc_prompt_id)],
    )


Retriever = ExhaustiveRetriever | TwoStageRetriever | HybridRetriever


def build_retriever(s: Settings, encoder) -> tuple[Retriever, IndexManifest | None]:
    """İndeksi yükler, uyumluluğu doğrular ve yapılandırılmış getiriciyi kurar.

    `create_app` ile slow canary fixture'ı (tests/retrieval/
    test_semantic_canary.py) BU fonksiyonu paylaşır: fixture eskiden aynı
    mantığı kopyalıyordu ve bu yüzden üretim yapılandırmasından sessizce
    sapabiliyordu (nitekim doğrudan `PackedIndex.load` çağırdığı için
    varsayılan indeks int8'e döndüğünde kopya sürüm bozulurdu).

    model_name/model_revision PARAMETRE DEĞİL, burada türetilir (review M5):
    ikisi de `s` ve `encoder`'dan tek biçimde çıkar, dışarıdan geçirilmeleri
    hem her çağrı yerinde aynı iki satırı tekrarlatıyordu hem de `s` ile
    tutarsız bir `model_name` geçirip fonksiyonun VAR OLUŞ SEBEBİ olan
    uyumluluk kontrolünü sessizce zayıflatmayı mümkün kılıyordu.

    T14: hangi kuantizasyonun diskte olduğu manifest'ten okunur (packed/
    int8/float16). Eskiden burada sabit `PackedIndex.load` vardı; ölçümde
    kazanan int8 indeks (float16 ile birebir kalite, 1-bit'e karşı +7 puan
    R@20 ve 4.3x hız) bu yüzden üretimde HİÇ yüklenemiyordu (ruling R16/D1).
    """
    index = load_scorable_index(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    resolved_query_format, resolved_doc_prompt = resolve_formats(s)
    # Final review IMPORTANT-2: fallback'ler ESKİ varsayılan literal'i (CPE_0_3_18)
    # ve None değil, config'ten çözülen ÜRETİM değerleridir. Aksi halde
    # `query_format`/`doc_prompt_sha256` taşımayan bir encoder enjekte edildiğinde
    # (testler, gömme/uzak encoder'lar) kontrol sessizce ölü hale geliyordu:
    # indeks train-compat, karşılaştırma cpe-0.3.18'e karşı yapılıyordu.
    resolved_doc_prompt_sha256 = (
        hashlib.sha256(resolved_doc_prompt.encode()).hexdigest()
        if resolved_doc_prompt is not None
        else None  # processor-default: etkin prompt yalnız processor'dan bilinir
    )
    # Manifest'i loader zaten okudu (üç indeks sınıfı da `.load` içinde okur);
    # ikinci bir `read_manifest` çağrısı aynı dosyayı tekrar ayrıştırırdı.
    problems = check_compatibility(
        index.manifest,
        model_name=s.retriever_model,
        model_revision=getattr(encoder, "model_revision", None),
        query_format_id=getattr(encoder, "query_format", resolved_query_format).format_id,
        doc_prompt_sha256=getattr(encoder, "doc_prompt_sha256", resolved_doc_prompt_sha256),
        index_dir=s.index_dir,
    )
    if problems:
        msg = "indeks/serve uyumsuzluğu: " + "; ".join(problems)
        if not s.allow_index_mismatch:
            raise IndexCompatibilityError(msg)
        logger.warning("BG_ALLOW_INDEX_MISMATCH=true ile devam ediliyor — %s", msg)

    retriever: Retriever
    if s.retrieval_pipeline == "hybrid":
        bm25, doc_names = load_text_channel(s.index_dir, list(index.page_ids))
        retriever = HybridRetriever(index, meta, encoder, bm25, doc_names)
    elif s.retrieval_pipeline == "exhaustive":
        retriever = ExhaustiveRetriever(index, meta, encoder)
    else:
        # two-stage mean-sign eleme YALNIZ paketli bit vektörleri üstünde
        # tanımlı (page_vecs + Hamming); int8/float16 indekste `page_vecs`
        # yoktur. Sessiz AttributeError yerine açık uyumsuzluk hatası.
        if not isinstance(index, PackedIndex):
            quant = index.manifest.quantization if index.manifest else "bilinmiyor"
            raise IndexCompatibilityError(
                "two-stage ablasyonu yalnız sign-1bit (PackedIndex) indeksle "
                f"çalışır; yüklü: {quant}"
            )
        retriever = TwoStageRetriever(index, meta, encoder)
    return retriever, index.manifest


def create_app(
    settings: Settings | None = None,
    encoder=None,
    answerer=None,
    recorder: EventRecorder | None = None,
) -> FastAPI:
    s = settings or get_settings()

    # Ölçek korkuluğu — PIPELINE'A DUYARLI (P1). Eşiğin anlamı skor ölçeğine
    # bağlıdır ve ölçeği artık pipeline belirler (config: PIPELINE_SCORE_SCALE):
    #
    #   * hybrid  -> BM25 birimi, üst sınırsız (ölçülen bant ~4-70). Buraya
    #     düşmüş bir GÖRSEL-ÖLÇEK eşiği (ör. 0.58) her soruyu geçirir: fren
    #     sessizce tamamen devre dışı kalır.
    #   * exhaustive/two-stage -> normalize [-1,1]. Buraya düşmüş bir BM25 ya
    #     da eski binary (0-128) eşiği HİÇBİR ZAMAN aşılamaz: servis her
    #     soruya sessizce "dayanak bulamadım" der.
    #
    # İki yön de sessizdir, ikisi de fail-fast'e çevriliyor. NEGATİF eşikler
    # ("her zaman cevapla" — testlerin -1e9'u) her pipeline'da SERBEST: kasıtlı
    # olarak kapatılmış bir frendir, ölçek kalıntısı değil.
    #
    # EN BAŞTA çalışır (review M6): saf config kontrolüdür, hiçbir şeye
    # bağımlı değildir — VLM ağırlıklarını ve 474 MB'lık indeksi yükledikten
    # sonra patlamasının hiçbir faydası yok.
    threshold_scale = PIPELINE_SCORE_SCALE[s.retrieval_pipeline]
    if threshold_scale == "hybrid-bm25":
        if 0 < s.min_score_threshold <= 1.5:
            raise IndexCompatibilityError(
                f"min_score_threshold={s.min_score_threshold} görsel-ölçek kalıntısı "
                f"görünüyor (normalize [-1,1]); {s.retrieval_pipeline} pipeline bm25 "
                "ölçeği ~5-70 üzerinde skorlar ve bu eşik orada HER soruyu geçirir "
                "(fren devre dışı). Taşınmış değer: 10.6."
            )
        if s.min_score_threshold > 200:
            raise IndexCompatibilityError(
                f"min_score_threshold={s.min_score_threshold} bm25 ölçeğinin çok "
                "üstünde (ölçülen top-1 bandı ~4-70); bu eşik hiçbir soruyu geçirmez"
            )
    elif s.min_score_threshold > 1.5:
        raise IndexCompatibilityError(
            f"min_score_threshold={s.min_score_threshold} eski binary ölçeği (0-128) "
            f"ya da bm25 ölçeği kalıntısı görünüyor; {s.retrieval_pipeline} pipeline'da "
            "skorlar normalize [-1,1] — bkz. "
            "data/bench/results/int8-threshold-transfer.json"
        )

    # Eşik-taşınabilirlik uyarısı (review I1, P1'de pipeline eksenine taşındı):
    # `min_score_threshold` hibrit BM25 dağılımı üzerinde taşındı. Başka bir
    # pipeline başka bir ÖLÇEKTE skorlar; korkuluk bariz kalıntıları keser ama
    # "ölçek doğru, dağılım farklı" durumunu KESEMEZ (bir operatör exhaustive
    # kolunda 0.58'i geri koyabilir — geçerli ama BU eşik değildir).
    # Başlatmayı ENGELLEMEZ (pipeline seçimi meşru bir ablasyon; per-pipeline
    # eşik config'i P2 kalibrasyonunun işi, ruling R19), yalnız sessiz kalmaz.
    if threshold_scale != THRESHOLD_CALIBRATED_ON:
        logger.warning(
            "eşik taşınabilirlik uyarısı: min_score_threshold=%s %s ölçeği üzerinde "
            "taşındı, etkin pipeline=%s ise %s ölçeğinde skorluyor — bu kolda çalışma "
            "noktası DOĞRULANMAMIŞTIR (P0 ölçümü o ölçekte 0.58'di). Eşiği bu "
            "pipeline'da yeniden ölçmeden üretimde kullanmayın.",
            s.min_score_threshold,
            THRESHOLD_CALIBRATED_ON,
            s.retrieval_pipeline,
            threshold_scale,
        )

    # Metin kanalı artefaktının VARLIĞI da saf bir dosya sistemi kontrolüdür ve
    # aynı gerekçeyle buraya alındı (review L6): "`index build-text` çalıştır"
    # mesajını almak için VLM ağırlıklarını ve 474 MB'lık indeksi yüklemek
    # gerekmez. HİZALAMA kontrolü indeksin page_ids'ini gerektirdiği için
    # `build_retriever` içinde (load_text_channel) kalır ve orada da koşar.
    if s.retrieval_pipeline == "hybrid":
        require_text_artifact(s.index_dir)

    resolved_query_format, resolved_doc_prompt = resolve_formats(s)
    if encoder is None:
        from belge_gozu.index.encode import ColSmolEncoder

        encoder = ColSmolEncoder(
            s.retriever_model,
            s.device,
            query_format=resolved_query_format,
            visual_prompt_override=resolved_doc_prompt,
        )
    if answerer is None:
        from belge_gozu.answer.gemini import GeminiAnswerer

        answerer = GeminiAnswerer(s.gemini_model, s.gemini_api_key)

    retriever, manifest = build_retriever(s, encoder)
    index = retriever.index

    def load_image(image_path: str) -> bytes:
        return (s.data_dir / image_path).read_bytes()

    if manifest is not None:
        index_revision = (
            f"{manifest.corpus_checksum[:12]}/{manifest.query_format.format_id}/"
            f"{manifest.quantization}"
        )
        query_format_id = manifest.query_format.format_id
        quantization = manifest.quantization
    else:
        index_revision = None
        query_format_id = None
        quantization = None

    service = AskService(retriever, answerer, s.min_score_threshold, load_image)

    rec = recorder or EventRecorder(s.data_dir / "requests.sqlite")
    prom = PromMetrics()
    try:
        from prometheus_client import GCCollector, PlatformCollector, ProcessCollector

        ProcessCollector(registry=prom.registry)
        PlatformCollector(registry=prom.registry)
        GCCollector(registry=prom.registry)
    except Exception:  # bazı platformlarda ProcessCollector yoktur; telemetri isteği düşürmez
        pass
    try:
        from importlib.metadata import version as pkg_version

        app_version = pkg_version("belge-gozu")
    except Exception:
        app_version = "0.0.0"
    prom.set_app_info(
        pages=len(index.page_ids),
        retriever_model=s.retriever_model,
        gemini_model=s.gemini_model,
        device=s.device,
        version=app_version,
        threshold=s.min_score_threshold,
        index_revision=index_revision or "unknown",
        query_format=query_format_id or "unknown",
    )

    app = FastAPI(title="Belge-Gözü")

    # Hız sınırları UYGULAMA BAŞINA (global değil): testler ve aynı süreçteki
    # ikinci bir app birbirinin sayacını görmez.
    search_limiter = RateLimiter(s.rate_limit_search_per_min)
    ask_limiter = RateLimiter(s.rate_limit_ask_per_min)
    if s.rate_limit_ask_per_min or s.rate_limit_search_per_min:
        logger.info(
            "hız sınırı etkin (istemci IP başına / dakika): /ask=%s /search=%s",
            s.rate_limit_ask_per_min or "kapalı",
            s.rate_limit_search_per_min or "kapalı",
        )

    def build_event(
        *,
        endpoint: str,
        status: str,
        http_status: int,
        total_ms: float,
        col: StageCollector,
        query: str,
        hits: list[PageHit],
        answer: Answer | None = None,
        error_type: str | None = None,
        k: int | None = None,
        candidates: int | None = None,
    ) -> RequestEvent:
        top = hits[0].score if hits else None
        margin = (hits[0].score - hits[1].score) if len(hits) >= 2 else None
        tokens_in = col.notes.get("tokens_in")
        tokens_out = col.notes.get("tokens_out")
        noted_error = col.notes.get("error_type")
        answer_ms = col.stages.get("answerer")
        tps = None
        if isinstance(tokens_out, int) and answer_ms and answer_ms > 0:
            tps = tokens_out / (answer_ms / 1000.0)
        cost = None
        if isinstance(tokens_in, int) and isinstance(tokens_out, int):
            cost = (tokens_in / 1e6) * s.gemini_price_in_usd_per_1m + (
                tokens_out / 1e6
            ) * s.gemini_price_out_usd_per_1m
        # Dürüst-ıska: TEK hesap yolu (`answer.base.is_honest_miss`). Aynı
        # değer `/ask` gövdesine, `events.honest_miss` kolonuna ve
        # `prom.observe` üzerinden `bg_honest_miss_total`a gider — üç yerde üç
        # ayrı sezgi tutulmaz (Y17/Y32/K27).
        honest_miss = is_honest_miss(answer) if answer is not None else None
        # Getirici kendi künyesini sunuyorsa (hibrit: iki kanalın top-1'i +
        # yönlendirilen dokümanlar) olaya karışır. `getattr` korumalı:
        # exhaustive/two-stage kolları etkilenmez, künye üretmeyen bir
        # getirici olayı düşürmez.
        retrieval_detail: dict = {"query_format": query_format_id, "quantization": quantization}
        extra = getattr(retriever, "last_retrieval_meta", None)
        if isinstance(extra, dict):
            retrieval_detail.update(extra)
        return RequestEvent(
            ts=datetime.now(UTC).isoformat(),
            endpoint=endpoint,
            status=status,
            http_status=http_status,
            total_ms=total_ms,
            encode_ms=col.stages.get("query_encode"),
            stage1_ms=col.stages.get("stage1_hamming"),
            stage2_ms=col.stages.get("stage2_maxsim"),
            answer_ms=answer_ms,
            top_score=top,
            margin_1_2=margin,
            abstained=answer.abstained if answer else None,
            honest_miss=honest_miss,
            k=k,
            candidates=candidates,
            query_len=len(query),
            query_text=query if s.log_query_text else None,
            query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            answer_len=len(answer.text) if answer else None,
            citations_n=len(answer.citations) if answer else None,
            tokens_in=tokens_in if isinstance(tokens_in, int) else None,
            tokens_out=tokens_out if isinstance(tokens_out, int) else None,
            tokens_per_s=tps,
            est_cost_usd=cost,
            # Y20: yanıtlayıcı hataları `service.ask`ten KAÇMAZ (AskService
            # yutup degraded'a çevirir), o yüzden buradaki açık `error_type`
            # kwarg'ı yalnız 500 yolunda dolar. Degraded satırların hata sınıfı
            # `AskService`in annotate ettiği notlardan gelir — bu bağlantı
            # kurulmadığı için hibrit satırların 114/114'ünde kolon NULL'du.
            error_type=error_type or (noted_error if isinstance(noted_error, str) else None),
            pipeline=s.retrieval_pipeline,
            score_scale=PIPELINE_SCORE_SCALE[s.retrieval_pipeline],
            index_revision=index_revision,
            detail={
                "hits": [{"page_id": h.page_id, "score": h.score} for h in hits],
                "threshold": s.min_score_threshold,
                "retriever_model": s.retriever_model,
                "gemini_model": s.gemini_model,
                "device": s.device,
                "app_version": app_version,
                "stages": dict(col.stages),
                "retrieval": retrieval_detail,
            },
        )

    def record_event(**kwargs) -> None:
        # Telemetri best-effort'tur: olay kurma/kaydetme hiçbir koşulda başarılı
        # bir isteği düşürmez.
        try:
            ev = build_event(**kwargs)
            rec.record(ev)
            prom.observe(ev)
        except Exception:
            logger.exception("telemetri olayı işlenemedi (istek etkilenmedi)")

    def record_rejection(*, endpoint: str, http_status: int, total_ms: float, query: str) -> None:
        """422/429 için MİNİMAL olay satırı + `bg_rejected_total{reason}` (Y23).

        Neden minimal: getirici hiç çağrılmadı, yani skor/aşama/atıf alanlarının
        hiçbirinin karşılığı yok — dolduruldukları takdirde P2'nin okuyacağı
        tabloya sahte sıfırlar girerdi. Yazılan şey isteğin VAR OLDUĞU: hangi
        uç nokta, hangi sebep, ne kadar sürdü, sorgunun kimliği.

        Neden yazılıyor: "sistem kaç kere içeriksiz sorgu reddetti?" sorusunun
        cevabı hiçbir yerde yoktu; oysa 422'ler P2 için **"cevaplanmaması
        gereken" sınıfının en temiz örnekleridir** ve 429'lar trafik profilinde
        görünmez bir delik açıyordu.

        `prom.observe` KULLANILMAZ: bu satırların `total_ms`'i sub-ms'dir ve
        gecikme histogramına karışırsa uç nokta p95'ini aşağı çeker. Görünürlük
        `bg_rejected_total{reason}` üzerinden sağlanır.

        Pydantic düzeyinde reddedilen istekler (gövde `max_length`, `k` aralığı)
        buraya HİÇ ULAŞMAZ — uç nokta gövdesi çalışmadan 422 dönerler. Bu
        dürüstçe bir kapsam sınırıdır, metrik kataloğunda yazılıdır.
        """
        reason = _REJECT_REASONS.get(http_status, "other")
        try:
            prom.rejected.labels(reason=reason).inc()
            rec.record(
                RequestEvent(
                    ts=datetime.now(UTC).isoformat(),
                    endpoint=endpoint,
                    status="rejected",
                    http_status=http_status,
                    total_ms=total_ms,
                    query_len=len(query),
                    query_text=query if s.log_query_text else None,
                    query_sha256=hashlib.sha256(query.encode()).hexdigest(),
                    error_type=reason,
                    pipeline=s.retrieval_pipeline,
                    score_scale=PIPELINE_SCORE_SCALE[s.retrieval_pipeline],
                    index_revision=index_revision,
                )
            )
        except Exception:
            logger.exception("ret olayı işlenemedi (istek etkilenmedi)")

    def guard(endpoint: str, text: str, request: Request, limiter: RateLimiter) -> None:
        """Doğrulama + hız sınırı; reddedilirse olay yazar ve HTTPException'ı geçirir.

        L2 (korunuyor): doğrulama ÖNCE, sınırlayıcı SONRA — geçersiz bir istek
        kota jetonu HARCAMAMALI. Sıra tersken 3000 karakterlik bir sorgu
        bedavaydı ama "bu ne için" jeton yakıyordu.
        """
        t0 = time.perf_counter()
        try:
            require_searchable(text)
            enforce_rate_limit(limiter, request, endpoint, prom)
        except HTTPException as exc:
            record_rejection(
                endpoint=endpoint,
                http_status=exc.status_code,
                total_ms=(time.perf_counter() - t0) * 1000,
                query=text,
            )
            raise

    @app.get("/healthz")
    def healthz() -> dict:
        # `index` bloğu (T14): eşik ile hangi TEMSİLİN servis edildiği tek
        # yerde görünür. `pipeline` (P1) aynı gerekçenin ikinci ekseni: eşiğin
        # ÖLÇEĞİ artık pipeline'a bağlı ("10.6 bm25" vs "0.58 normalize"), bu
        # yüzden eşikle birlikte okunmalı. `revision` telemetrideki
        # `index_revision` ile aynı dizedir (olay kayıtlarıyla eşleştirilebilir).
        return {
            "status": "ok",
            "pages": len(index.page_ids),
            "threshold": s.min_score_threshold,
            "top_k": s.top_k,
            "pipeline": s.retrieval_pipeline,
            "index": {"quantization": quantization, "revision": index_revision},
        }

    @app.post("/search")
    def search(body: SearchBody, request: Request) -> dict[str, list[PageHit]]:
        # Doğrulama + hız sınırı + RET OLAYI tek yerde (`guard`); sıra
        # gerekçesi orada. Pydantic'in kendi doğrulaması (gövde `max_length`,
        # `k` aralığı) buraya hiç ulaşmadan 422 verir ve olay yazılmaz.
        guard("/search", body.query, request, search_limiter)
        t0 = time.perf_counter()
        cand = s.stage1_candidates if s.retrieval_pipeline == "two-stage" else None
        with collecting() as col, prom.inflight("/search"):
            try:
                k = body.k or s.top_k
                if cand is None:
                    hits = retriever.search(body.query, k=k)
                else:
                    # cand is not None <=> retrieval_pipeline == "two-stage" (TwoStageRetriever);
                    # pyright can't narrow retriever's union type from cand's nullability.
                    hits = retriever.search(body.query, k=k, candidates=cand)  # pyright: ignore[reportCallIssue]
            except Exception as e:
                record_event(
                    endpoint="/search",
                    status="error",
                    http_status=500,
                    total_ms=(time.perf_counter() - t0) * 1000,
                    col=col,
                    query=body.query,
                    hits=[],
                    error_type=type(e).__name__,
                )
                raise
            record_event(
                endpoint="/search",
                status="ok",
                http_status=200,
                total_ms=(time.perf_counter() - t0) * 1000,
                col=col,
                query=body.query,
                hits=hits,
                k=k,
                candidates=cand,
            )
        return {"hits": hits}

    @app.post("/ask")
    def ask(body: AskBody, request: Request) -> dict:
        # bkz. /search'teki aynı `guard` yorumu.
        guard("/ask", body.question, request, ask_limiter)
        t0 = time.perf_counter()
        cand = s.stage1_candidates if s.retrieval_pipeline == "two-stage" else None
        with collecting() as col, prom.inflight("/ask"):
            try:
                answer, hits = service.ask(body.question, k=s.top_k, candidates=cand)
            except Exception as e:
                record_event(
                    endpoint="/ask",
                    status="error",
                    http_status=500,
                    total_ms=(time.perf_counter() - t0) * 1000,
                    col=col,
                    query=body.question,
                    hits=[],
                    error_type=type(e).__name__,
                )
                raise
            if col.notes.get("degraded"):
                status = "degraded"
            elif answer.abstained:
                status = "abstained"
            else:
                status = "answered"
            honest_miss = is_honest_miss(answer)
            record_event(
                endpoint="/ask",
                status=status,
                http_status=200,
                total_ms=(time.perf_counter() - t0) * 1000,
                col=col,
                query=body.question,
                hits=hits,
                answer=answer,
                k=s.top_k,
                candidates=cand,
            )
        # `status` gövdenin ÜST DÜZEYİNDE, telemetriye yazılan değerin AYNISI.
        # Arayüz durumları (mühür / "servis kapalı" bandı / normal yanıt)
        # buradan dallanır; daha önce ABSTAIN_TEXT ile DİZE KARŞILAŞTIRMASI
        # yapılıyordu ve o metnin tek bir noktalama değişikliği arayüzü sessizce
        # bozardı. Üç değer: "answered" (dürüst "bulamadım" DAHİL — o da bir
        # yanıttır), "abstained" (eşik altı, LLM hiç çağrılmadı),
        # "degraded" (yanıtlayıcı patladı, sayfalar hâlâ geçerli).
        #
        # `honest_miss` (Y17) `status`'a DÖRDÜNCÜ BİR DEĞER OLARAK EKLENMEDİ,
        # ayrı bir bayrak: dürüst ıska bir `answered` ALT durumudur (LLM
        # çağrıldı, yanıt üretti, sayfalarda dayanak bulamadığını söyledi) ve
        # `status`'a yeni bir değer eklemek her mevcut istemciyi sessizce
        # yanlış dala düşürürdü. Sunucunun HESAPLADIĞI değerin ta kendisidir —
        # `events.honest_miss` ve `bg_honest_miss_total` ile aynı satır.
        return {
            "status": status,
            "honest_miss": honest_miss,
            "answer": answer.model_dump(),
            "hits": [h.model_dump() for h in hits],
        }

    @app.get("/metrics")
    def metrics() -> Response:
        body, ctype = prom.render()
        return Response(content=body, media_type=ctype)

    @app.get("/stats")
    def stats() -> dict:
        # /stats bir telemetri okuma uç noktasıdır: herhangi bir sqlite hatası
        # isteği asla 500'e düşürmez, sıfırlanmış bir gövdeye geriler.
        degraded = {
            "requests": 0,
            "avg_ms": 0.0,
            "p95_ms": 0.0,
            "abstain_rate": 0.0,
            "by_endpoint": {},
        }
        db = None
        try:
            db = sqlite3.connect(rec.db_path)
            n, avg = db.execute("SELECT COUNT(*), COALESCE(AVG(total_ms),0) FROM events").fetchone()
            vals = [
                r[0] for r in db.execute("SELECT total_ms FROM events ORDER BY id DESC LIMIT 10000")
            ]
            vals.sort()
            p95 = vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)] if vals else 0.0
            ab = db.execute(
                "SELECT COALESCE(AVG(abstained),0) FROM events "
                "WHERE endpoint='/ask' AND status <> 'degraded'"
            ).fetchone()[0]
            by = dict(db.execute("SELECT endpoint, COUNT(*) FROM events GROUP BY endpoint"))
            return {
                "requests": n,
                "avg_ms": round(avg, 1),
                "p95_ms": round(p95, 1),
                "abstain_rate": round(ab, 3),
                "by_endpoint": by,
            }
        except Exception:
            logger.exception("/stats okunamadı, sıfırlanmış gövdeye gerilendi")
            return degraded
        finally:
            if db is not None:
                db.close()

    @app.get("/pages/{image_path:path}")
    def page_image(image_path: str) -> FileResponse:
        # /pages YALNIZCA sayfa görüntülerini sunar: data_dir/images altındaki
        # .webp dosyaları. Daha önce tüm data_dir ağacı servis ediliyordu, yani
        # telemetri veritabanı (requests.sqlite — ham sorgu metinleri), korpus
        # PDF'leri ve meta.parquet indirilebiliyordu.
        #
        # images_root kontrolü aynı zamanda yol aşımı (traversal) korumasıdır:
        # data_dir'in kendisinden daha dar bir köktür, ".." ile dışarı çıkan ya
        # da images/ dışına işaret eden her yol elenir. resolve() sembolik
        # bağları da çözdüğü için images/ içinden dışarı gösteren bir link de
        # reddedilir.
        #
        # Reddedilen her yol 404 döner (403/500 değil): uç nokta neyin var olup
        # olmadığını sızdırmaz. is_file() kontrolü exists() yerine kullanılır —
        # bir dizin yolu aksi hâlde FileResponse'ta IsADirectoryError'a, yani
        # 500'e düşerdi.
        images_root = (s.data_dir / "images").resolve()
        full = (s.data_dir / image_path).resolve()
        if (
            not full.is_relative_to(images_root)
            or full.suffix.lower() != ".webp"
            or not full.is_file()
        ):
            raise HTTPException(404)
        return FileResponse(full, media_type="image/webp")

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        page = STATIC_DIR / "index.html"
        if page.exists():
            return page.read_text(encoding="utf-8")
        return "<html><body><h1>Belge-Gözü</h1><p>UI yakında.</p></body></html>"

    return app
