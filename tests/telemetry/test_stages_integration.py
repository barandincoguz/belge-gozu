import pandas as pd

from belge_gozu.answer.base import AskService
from belge_gozu.index.store import PackedIndex
from belge_gozu.retrieval.core import TwoStageRetriever
from belge_gozu.telemetry.collect import collecting


class StubAnswerer:
    def answer(self, question, pages, image_loader):
        from belge_gozu.answer.base import Answer

        return Answer(text="yanıt", citations=[pages[0].page_id])


def _retriever(tiny_corpus) -> TwoStageRetriever:
    data_dir, enc, _ = tiny_corpus
    index = PackedIndex.load(data_dir / "index")
    meta = pd.read_parquet(data_dir / "index" / "meta.parquet")
    return TwoStageRetriever(index, meta, enc)


def test_search_fills_retrieval_stages(tiny_corpus):
    r = _retriever(tiny_corpus)
    with collecting() as col:
        hits = r.search("deneme", k=3, candidates=10)
    assert hits
    assert {"query_encode", "stage1_hamming", "stage2_maxsim"} <= set(col.stages)


def test_ask_fills_answerer_stage(tiny_corpus):
    r = _retriever(tiny_corpus)
    svc = AskService(r, StubAnswerer(), min_score=-1e9, image_loader=lambda p: b"x")
    with collecting() as col:
        svc.ask("deneme", k=3, candidates=10)
    assert "answerer" in col.stages


def test_abstain_skips_answerer_stage(tiny_corpus):
    r = _retriever(tiny_corpus)
    svc = AskService(r, StubAnswerer(), min_score=1e9, image_loader=lambda p: b"x")
    with collecting() as col:
        answer, _ = svc.ask("deneme", k=3, candidates=10)
    assert answer.abstained and "answerer" not in col.stages
