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

# KANIT KAPISI DÜŞÜRMESİ (P2 T2 / ilke 20). Yanıt ÜRETİLDİ ama iddialarının
# tamamı atıf yaptıkları sayfa metninde doğrulanamadı.
#
# Metin ABSTAIN_TEXT'ten AYRI olmak zorunda: ikisi farklı şeyler söyler ve
# ikisi de doğru olmak zorunda. "Dayanak bulamadım" (eşik altı, LLM hiç
# çağrılmadı) ile "yanıt üretildi ama kanıtlanamadı" aynı cümleye
# sıkıştırılırsa sistem kendi durumu hakkında yanlış konuşur. `status` yine
# `abstained`tır (istemci sözlüğü GENİŞLEMEZ, K12 kilidi korunur); ayrım
# `detail.gate2.demoted` alanındadır.
VERIFIER_DEMOTE_TEXT = (
    "Yanıt üretildi ancak kanıt doğrulamasından geçemedi; dayanak gösterilemiyor."
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


def gate2_skip_reason(answer: Answer) -> str | None:
    """Kanıt kapısının ATLANMA sebebi (yoksa None) — tek karar yolu.

    Dürüst ıska ve atıfsız yanıtlar atlanır: ikisi de zaten "kesin yanıt" diye
    sunulmuyor, doğrulamak yalnız kota yakardı.

    `answer/verify.py` yerine BURADA durur (review L5): yalnız `Answer`a
    dokunuyor ve `verify.py`de dururken `AskService` onu her istekte FONKSİYON
    İÇİNDE ithal etmek zorunda kalıyordu — bu, tek amacı `base <-> verify`
    döngüsünü kırmak olan bir kaçamaktı.
    """
    if answer.abstained:
        return "abstained"
    if is_honest_miss(answer):
        return "honest_miss"
    if not answer.citations:
        return "no_citations"
    return None


class RetrievalGate(Protocol):
    """Kalibre getirim kapısı (P2 T2, kapı 1) — `answer/calibrate.py` uygular."""

    def evaluate(self, question: str, *, bm25=None) -> dict: ...


class EvidenceGateProtocol(Protocol):
    """Kanıt kapısı (P2 T2, kapı 2) — `answer/verify.py` uygular."""

    def evaluate(self, answer: Answer, hits: list[PageHit]) -> dict: ...


class AskService:
    """İki kapılı yanıt servisi. HER İKİ KAPI DA VARSAYILAN OLARAK YOKTUR.

    `gate1`/`gate2` None iken bu sınıfın davranışı P1'inkiyle BİREBİR AYNIDIR
    (tek eşik + degradasyon koruması) — bayrak-kapalı üretim davranışının
    değişmemesi bir tercih değil, kapı kuralının gereğidir (master §1: G1+G2
    raporlanana kadar default-açık entegrasyon YOK).

    DÜRÜST SINIR (p2-reality-audit T2/1): bayraklar kapalıyken geri düşülen
    eşik GÜVENLİ bir kapı DEĞİLDİR — ölçümde cevaplanamaz 5 sorunun 4'ü onu
    geçiyor (`tests/retrieval/test_semantic_retrieval_eval.py` xfail(strict) kilidi).
    Yani "flag kapalı = eski davranış" cümlesi bir güvence değil, bir
    değişmezlik beyanıdır.
    """

    def __init__(
        self,
        retriever,
        answerer: Answerer,
        min_score: float,
        image_loader: Callable[[str], bytes],
        gate1: RetrievalGate | None = None,
        gate2: EvidenceGateProtocol | None = None,
    ):
        self.retriever = retriever
        self.answerer = answerer
        self.min_score = min_score
        self.image_loader = image_loader
        self.gate1 = gate1
        self.gate2 = gate2

    def ask(
        self, question: str, k: int, candidates: int | None = None
    ) -> tuple[Answer, list[PageHit]]:
        if candidates is None:
            hits = self.retriever.search(question, k=k)
        else:
            hits = self.retriever.search(question, k=k, candidates=candidates)
        # KAPI 1 — kalibre getirim güveni. Eşik kontrolünden ÖNCE ÖLÇÜLÜR
        # (p her koşulda telemetriye girsin: eşik altı satırlar kalibratörün
        # kendi değerlendirmesinin en bilgilendirici yarısıdır), ama eşikten
        # SONRA UYGULANIR ki bayrak açıkken bile eski frenin önceliği bozulmasın.
        gate1_detail = self._eval_gate1(question)
        if not hits or hits[0].score < self.min_score:
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True), hits
        if gate1_detail is not None and not gate1_detail.get("passed", True):
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True), hits
        try:
            with stage("answerer"):
                answer = self.answerer.answer(question, hits, self.image_loader)
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
        if self.gate2 is None:
            return answer, hits
        return self._apply_gate2(answer, hits), hits

    def _eval_gate1(self, question: str) -> dict | None:
        """Kalibre olasılık + eşik (kapı kapalıysa None). Hiçbir koşulda patlamaz.

        BM25 skorları getiriciden GERİ ALINIR (`last_bm25_scores`), yeniden
        hesaplanmaz: kalibrasyon istek başına İKİNCİ bir korpus taraması
        eklememelidir (`answer/calibrate.py::retrieval_context` sözleşmesi).
        """
        if self.gate1 is None:
            return None
        try:
            detail = self.gate1.evaluate(
                question, bm25=getattr(self.retriever, "last_bm25_scores", None)
            )
        except Exception:
            # Kapı 1 bir FREN'dir; freni hesaplayamamak yanıtı düşürmez ama
            # görünmez de kalmaz. Açık kalır (eski eşik hâlâ yerinde).
            logger.exception("kalibre getirim kapısı hesaplanamadı — kapı atlanıyor")
            detail = {"error": "gate1_failed", "passed": True}
        annotate("gate1", detail)
        return detail

    def _apply_gate2(self, answer: Answer, hits: list[PageHit]) -> Answer:
        """Kanıt kapısı: desteklenmeyen tek iddia bile yanıtı DÜŞÜRÜR (ilke 20)."""
        skip = gate2_skip_reason(answer)
        if skip is not None:
            annotate("gate2", {"demoted": False, "skipped": skip})
            return answer
        try:
            detail = self.gate2.evaluate(answer, hits)  # pyright: ignore[reportOptionalMemberAccess]
        except Exception as exc:
            # ŞÜPHEDE REDDET: doğrulayıcı arızası (kota, bütçe, disk) bir
            # yanıtı KESİN diye sunmak için gerekçe değildir.
            logger.exception("kanıt kapısı çalıştırılamadı — yanıt düşürülüyor")
            detail = {"demoted": True, "error": type(exc).__name__}
        annotate("gate2", detail)
        if not detail.get("demoted"):
            return answer
        return Answer(text=VERIFIER_DEMOTE_TEXT, citations=[], abstained=True)
