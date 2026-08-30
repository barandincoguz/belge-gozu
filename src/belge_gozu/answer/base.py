import logging
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from belge_gozu.retrieval.text import tr_lower
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import annotate, stage

logger = logging.getLogger(__name__)

ABSTAIN_TEXT = "Bu soruya korpustaki belgelerde dayanak bulamadım."
SERVICE_ERROR_TEXT = (
    "Yanıt servisi şu anda kullanılamıyor (kota veya servis hatası). "
    "Bulunan sayfalar aşağıda listeleniyor."
)

# DÜRÜST-ISKA MÜHÜRÜ — sistemin en bilgilendirici olayının TEK kaynağı (Y17/K27).
#
# "Dürüst ıska" = getirim sayfaları getirdi, eşik geçti, LLM çağrıldı, ve model
# o sayfalarda dayanak BULAMADIĞINI söyledi. Bu bir arıza değil, sistemin doğru
# davranışıdır — ve P2 kalibratörünün hedef değişkenidir.
#
# Bu sabit iki yerde birden kullanılır ve o yüzden TEK yerde durur:
#   1. Gemini SİSTEM istemi bunu f-string ile GÖMER (`answer/gemini.py`) —
#      yani modele dayatılan ifade ile sunucunun aradığı ifade birbirinden
#      SAPAMAZ. Önceden istemdeki cümle ile `main.py`'deki alt-dize elle
#      eşleşiyordu (S35/D3 borcu): birini değiştirmek diğerini sessizce
#      bozardı.
#   2. `is_honest_miss()` yanıt metninde bunu arar.
#
# TAM İFADE aranır, çıplak "bulamadım" DEĞİL: "...bir istisna bulamadım ama
# m.45'te düzenlenmiştir" gibi bir cümle eski sezgide YANLIŞ POZİTİF veriyordu.
HONEST_MISS_MARKER = "verilen sayfalarda bulamadım"


class Answer(BaseModel):
    text: str
    citations: list[str]
    abstained: bool = False


def is_honest_miss(answer: Answer | None) -> bool:
    """Yanıt, modelin kendi dürüst ıskası mı? (TEK hesap yolu.)

    Sunucunun `/ask` gövdesi, `events.honest_miss` kolonu ve Prometheus'un
    `bg_honest_miss_total` sayacı ÜÇÜ DE buradan okur — üç ayrı yerde üç ayrı
    alt-dize sezgisi tutulmaz.

    `tr_lower` (Türkçe küçültme, `retrieval/text.py`) KULLANILIR, `str.lower()`
    DEĞİL: Python'un `lower()`ı "BULAMADIM"ı "bulamadim" yapar (I -> i) ve
    işaret eşleşmezdi. Aynı fonksiyon getirim tarafında da tokenleştirmenin ilk
    adımıdır; kopyası çıkarılmaz.

    Abstain (eşik altı, LLM hiç çağrılmadı) ve degraded (LLM patladı) dürüst
    ıska DEĞİLDİR: ikisinde de modelin sayfalar hakkında bir yargısı yoktur.
    """
    if answer is None or answer.abstained:
        return False
    return HONEST_MISS_MARKER in tr_lower(answer.text)


class AnswererError(Exception):
    """Yanıtlayıcı hatası + KÜÇÜK bir hata taksonomisi (Y15/Y20).

    `error_type` `ERROR_TYPES` içinden bir dizedir ve `events.error_type`
    kolonuna, oradan da P2'nin eğitim kümesine gider. Amacı "cevaplayamadı"
    (kota, zaman aşımı, kesinti) ile "cevaplamamalı" (güvenlik bloğu) ayrımını
    telemetride görünür kılmak: önceden degraded satırların 114/114'ünde bu
    kolon NULL'du ve bir günlük kesintinin sebebi geriye dönük OKUNAMIYORDU.
    """

    def __init__(self, error_type: str, message: str = "") -> None:
        super().__init__(message or error_type)
        self.error_type = error_type


# Taksonomi KÜÇÜK tutulur: her değer farklı bir OPERATÖR EYLEMİ ima eder.
#   timeout      -> sağlayıcı/ağ yavaş; retry edildi, yine yetişmedi
#   http_5xx     -> sağlayıcı kesintisi; retry edildi
#   http_429     -> kota/hız; retry EDİLMEZ (daha çok istek durumu kötüleştirir)
#   auth         -> anahtar yok/geçersiz — dağıtım hatası, kendiliğinden geçmez
#   safety_block -> model içerik nedeniyle üretmedi; "cevaplamamalı" sınıfı
#   parse        -> yanıt geldi ama okunamadı (SDK/şema uyuşmazlığı)
#   other        -> sınıflandırılamayan; sayısı ARTIYORSA taksonomi eksiktir
ERROR_TYPES = frozenset(
    {"timeout", "http_5xx", "http_429", "auth", "safety_block", "parse", "other"}
)


class Answerer(Protocol):
    def answer(
        self, question: str, pages: list[PageHit], image_loader: Callable[[str], bytes]
    ) -> Answer: ...


class AskService:
    def __init__(
        self, retriever, answerer: Answerer, min_score: float, image_loader: Callable[[str], bytes]
    ):
        self.retriever = retriever
        self.answerer = answerer
        self.min_score = min_score
        self.image_loader = image_loader

    def ask(
        self, question: str, k: int, candidates: int | None = None
    ) -> tuple[Answer, list[PageHit]]:
        if candidates is None:
            hits = self.retriever.search(question, k=k)
        else:
            hits = self.retriever.search(question, k=k, candidates=candidates)
        if not hits or hits[0].score < self.min_score:
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True), hits
        try:
            with stage("answerer"):
                return self.answerer.answer(question, hits, self.image_loader), hits
        except Exception as exc:
            # Y20: degraded olayı artık BİR HATA SINIFI taşır. Yanıtlayıcı
            # taksonomiyi kendisi bildiriyorsa (GeminiAnswerer -> AnswererError)
            # o kullanılır; bildirmiyorsa "other" — çünkü buradan istisna
            # SINIFI yazmak (`type(exc).__name__`) telemetriyi SDK'nın iç
            # sınıf adlarına bağlar ve operatöre eylem söylemez.
            error_type = exc.error_type if isinstance(exc, AnswererError) else "other"
            logger.exception("answerer failed (error_type=%s)", error_type)
            annotate("degraded", True)
            annotate("error_type", error_type)
            return Answer(text=SERVICE_ERROR_TEXT, citations=[], abstained=True), hits
