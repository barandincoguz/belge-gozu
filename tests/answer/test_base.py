from belge_gozu.answer.base import Answer, AskService
from belge_gozu.retrieval.types import PageHit


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
