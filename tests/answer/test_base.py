from belge_gozu.answer.base import (
    ABSTAIN_TEXT,
    ERROR_TYPES,
    HONEST_MISS_MARKER,
    SERVICE_ERROR_TEXT,
    Answer,
    AnswererError,
    AskService,
    is_honest_miss,
)
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import collecting


def hit(pid: str, score: float) -> PageHit:
    return PageHit(
        page_id=pid,
        score=score,
        doc_name="Belge",
        page_no=1,
        image_path=f"images/{pid}.webp",
        source_url="https://example.org",
    )


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, k=5, candidates=200):
        return self._hits[:k]


class EchoAnswerer:
    def answer(self, question, pages, image_loader):
        return Answer(text="cevap", citations=[pages[0].page_id])


def test_ask_answers_above_threshold():
    svc = AskService(
        FakeRetriever([hit("a:1", 90.0)]),
        EchoAnswerer(),
        min_score=20.0,
        image_loader=lambda p: b"img",
    )
    answer, hits = svc.ask("soru", k=5, candidates=200)
    assert not answer.abstained and answer.citations == ["a:1"] and hits[0].page_id == "a:1"


def test_ask_abstains_below_threshold():
    svc = AskService(
        FakeRetriever([hit("a:1", 5.0)]),
        EchoAnswerer(),
        min_score=20.0,
        image_loader=lambda p: b"img",
    )
    answer, _ = svc.ask("soru", k=5, candidates=200)
    assert answer.abstained and answer.citations == []


def test_ask_abstains_on_empty_index():
    svc = AskService(
        FakeRetriever([]), EchoAnswerer(), min_score=20.0, image_loader=lambda p: b"img"
    )
    answer, hits = svc.ask("soru", k=5, candidates=200)
    assert answer.abstained and hits == []


def test_ask_degrades_gracefully_when_answerer_fails(caplog):
    class BoomAnswerer:
        def answer(self, question, pages, image_loader):
            raise RuntimeError("quota exceeded")

    svc = AskService(
        FakeRetriever([hit("a:1", 90.0)]),
        BoomAnswerer(),
        min_score=20.0,
        image_loader=lambda p: b"img",
    )
    answer, hits = svc.ask("soru", k=5, candidates=200)
    assert answer.abstained and answer.citations == []
    assert "kullanılamıyor" in answer.text
    assert hits and hits[0].page_id == "a:1"  # retrieval sonuçları kaybolmaz
    assert "answerer failed" in caplog.text


# --- Y20: degraded olayı bir HATA SINIFI taşır -------------------------------


def _boom_service(exc):
    class BoomAnswerer:
        def answer(self, question, pages, image_loader):
            raise exc

    return AskService(
        FakeRetriever([hit("a:1", 90.0)]),
        BoomAnswerer(),
        min_score=20.0,
        image_loader=lambda p: b"img",
    )


def test_degraded_annotates_the_answerers_own_error_type():
    """Yanıtlayıcı taksonomiyi bildiriyorsa o kullanılır (Y15 -> Y20 zinciri)."""
    with collecting() as col:
        _boom_service(AnswererError("http_429", "kota")).ask("soru", k=5, candidates=200)
    assert col.notes["degraded"] is True
    assert col.notes["error_type"] == "http_429"
    assert col.notes["error_type"] in ERROR_TYPES


def test_degraded_falls_back_to_other_for_unclassified_failures():
    """Sınıflandırılmamış hata `type(exc).__name__` DEĞİL "other" olur:
    telemetriyi SDK'nın iç sınıf adlarına bağlamak operatöre eylem söylemez."""
    with collecting() as col:
        _boom_service(RuntimeError("quota exceeded")).ask("soru", k=5, candidates=200)
    assert col.notes["error_type"] == "other"


# --- Y17/K27: dürüst-ıska TEK hesap yolu -------------------------------------


def test_honest_miss_requires_the_full_marker():
    """Çıplak "bulamadım" YETMEZ: eski sezgi "...bir istisna bulamadım ama
    m.45'te düzenlenmiştir" cümlesine yanlış pozitif veriyordu."""
    assert is_honest_miss(Answer(text=f"Soruya {HONEST_MISS_MARKER}.", citations=[]))
    assert not is_honest_miss(
        Answer(text="Bir istisna bulamadım ama m.45'te düzenlenmiştir [S1].", citations=["a:1"])
    )


def test_honest_miss_is_turkish_lowercase_aware():
    """`str.lower()` "BULAMADIM"ı "bulamadim" yapar (I -> i) ve işaret
    eşleşmezdi; `tr_lower` I -> ı yapar."""
    shouty = HONEST_MISS_MARKER.replace("i", "İ").replace("ı", "I").upper()
    assert "BULAMADIM" in shouty
    assert is_honest_miss(Answer(text=shouty, citations=[]))
    assert HONEST_MISS_MARKER not in shouty.lower()  # eski yol GERÇEKTEN kaçırırdı


def test_abstain_and_degraded_are_not_honest_misses():
    """İkisinde de modelin sayfalar hakkında bir yargısı YOKTUR."""
    assert not is_honest_miss(None)
    assert not is_honest_miss(Answer(text=ABSTAIN_TEXT, citations=[], abstained=True))
    assert not is_honest_miss(Answer(text=SERVICE_ERROR_TEXT, citations=[], abstained=True))
