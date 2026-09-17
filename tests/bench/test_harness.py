from pathlib import Path

import pytest

from belge_gozu.bench.dataset import BenchQuestion
from belge_gozu.bench.harness import StageRecord, run_retrieval_eval
from tests.bench_question_factory import q_dict


class MapPipeline:
    name = "map"

    def __init__(self, answers: dict[str, list[str]]):
        self.answers = answers

    def run(self, question: str):
        ranked = self.answers[question]
        rec = StageRecord(
            stage="final",
            gold_ranks={},
            top_ids=ranked,
            top_scores=[1.0] * len(ranked),
            latency_ms=1.0,
        )
        return ranked, [rec]


def test_retrieval_eval_rejects_a_selection_without_answerable_questions():
    question = BenchQuestion(
        **q_dict(
            answerable=False,
            gold_doc_ids=[],
            gold_page_ids=[],
            gold_article_ids=[],
            minimal_evidence_spans=[],
            reference_answer="",
            slice="korpus-disi",
            unanswerable_reason="korpus-disi",
        )
    )

    with pytest.raises(ValueError, match="cevaplanabilir soru yok"):
        run_retrieval_eval(MapPipeline({}), [question], known_page_ids=set())


def test_report_metrics_and_survival(tmp_path: Path):
    qs = [
        BenchQuestion(**q_dict()),  # gold k4721:4
        BenchQuestion(
            **q_dict(
                question_id="q2",
                question="ikinci",
                gold_doc_ids=["k6098"],
                gold_page_ids=["k6098:120"],
                gold_article_ids=[],
                slice="dogrudan-madde",
            )
        ),
    ]
    pipe = MapPipeline(
        {
            "Yerleşim yeri nedir?": ["k4721:4", "x:1"],
            "ikinci": ["x:1", "x:2"],
        }
    )
    rep = run_retrieval_eval(
        pipe, qs, known_page_ids={"k4721:4", "k6098:120", "x:1", "x:2"}, ks=(1, 5), run_id="t"
    )
    assert rep.overall.recall_at[1] == 0.5 and rep.overall.recall_at[5] == 0.5
    assert rep.overall.mrr == 0.5
    d = {x.question_id: x for x in rep.diagnostics}
    assert d["q1"].candidate_survival == {"k4721:4": True}
    assert d["q2"].candidate_survival == {"k6098:120": False}
    assert rep.per_slice["paraphrase"].n == 1
    out = tmp_path / "r.json"
    rep.to_json(out)
    assert out.exists()


def test_report_preserves_verification_selection_provenance(tmp_path: Path):
    qs = [BenchQuestion(**q_dict())]
    pipe = MapPipeline({"Yerleşim yeri nedir?": ["k4721:4"]})
    verification = {
        "only_verified": True,
        "min_verification": "human",
        "total": 48,
        "selected": 3,
        "filtered_out": 45,
    }

    report = run_retrieval_eval(
        pipe,
        qs,
        known_page_ids={"k4721:4"},
        config={"verification": verification},
    )
    out = tmp_path / "report.json"
    report.to_json(out)

    assert report.config["verification"] == verification
    assert '"filtered_out":45' in out.read_text(encoding="utf-8").replace(" ", "")


def test_missing_gold_page_reported():
    qs = [BenchQuestion(**q_dict())]
    pipe = MapPipeline({"Yerleşim yeri nedir?": ["x:1"]})
    rep = run_retrieval_eval(pipe, qs, known_page_ids={"x:1"}, ks=(1,))
    assert rep.missing_gold_pages == ["k4721:4"]


def test_diagnostics_distinguish_full_rank_from_the_record_window():
    question = BenchQuestion(**q_dict())

    class WindowedPipeline:
        name = "windowed"

        def run(self, question_text: str):
            assert question_text == question.question
            ranked = ["x:1", "x:2", "k4721:4"]
            return ranked, [
                StageRecord(
                    stage="final",
                    gold_ranks={},
                    top_ids=ranked[:1],
                    top_scores=[1.0],
                    latency_ms=1.0,
                    full_ranked=ranked,
                )
            ]

    report = run_retrieval_eval(
        WindowedPipeline(),
        [question],
        known_page_ids={"x:1", "x:2", "k4721:4"},
        ks=(1,),
    )

    diagnostic = report.diagnostics[0]
    assert diagnostic.stages[0].gold_ranks == {"k4721:4": 3}
    assert diagnostic.candidate_survival == {"k4721:4": True}
    assert "full_ranked" not in report.model_dump()["diagnostics"][0]["stages"][0]


def test_hybrid_adapter_uses_the_same_late_candidate_order_as_production(tiny_corpus, monkeypatch):
    import pandas as pd

    from belge_gozu.bench.harness import HybridDiagnosticAdapter
    from belge_gozu.index.store import PackedIndex
    from belge_gozu.retrieval.hybrid import HybridRetriever, load_text_channel
    from belge_gozu.retrieval.late import LateSearchResult

    data_dir, encoder, _ = tiny_corpus
    index_dir = data_dir / "index"
    index = PackedIndex.load(index_dir)
    meta = pd.read_parquet(index_dir / "meta.parquet")
    bm25, doc_names = load_text_channel(index_dir, index.page_ids)
    question = "yerleşim yeri nedir"
    base = HybridRetriever(index, meta, encoder, bm25, doc_names)
    late_page = base.rank_all(question)[-1]
    clock = {"now": 0.0, "late_delay": 0.0}

    class FixedLateChannel:
        def search_with_scores(self, query: str, limit: int) -> LateSearchResult:
            clock["now"] += clock["late_delay"]
            return LateSearchResult(
                pages=(late_page,),
                query_tokens=2,
                raw_top1=2.0,
                raw_margin=1.0,
                mean_top1=1.0,
                mean_margin=0.5,
            )

    retriever = HybridRetriever(
        index,
        meta,
        encoder,
        bm25,
        doc_names,
        late_channels=(FixedLateChannel(),),
        late_candidate_limit=1,
    )

    ranked, stages = HybridDiagnosticAdapter(retriever).run(question)
    production = [hit.page_id for hit in retriever.search(question, k=3)]

    assert production[1] == late_page
    assert ranked[:3] == production
    assert "late_candidate_union" in [stage.stage for stage in stages]

    # Rapor nesnesini kurmanın süresi geç kanal arama süresine katılmamalı.
    original_record = StageRecord

    def timed_record(**kwargs):
        if kwargs["stage"] == "route_fuse":
            clock["now"] += 0.05
        return original_record(**kwargs)

    monkeypatch.setattr("belge_gozu.bench.harness.StageRecord", timed_record)
    monkeypatch.setattr("belge_gozu.bench.harness.time.perf_counter", lambda: clock["now"])
    clock.update(now=0.0, late_delay=0.007)
    _, timed_stages = HybridDiagnosticAdapter(retriever).run(question)
    assert timed_stages[-1].latency_ms == pytest.approx(7.0)


def test_exhaustive_adapter_records_ranks():
    from belge_gozu.bench.harness import ExhaustiveDiagnosticAdapter
    from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever
    from tests.retrieval.test_core import build_fixture

    idx, meta, embs = build_fixture(n_pages=30)

    class SelfEnc:
        def encode_pages(self, images):
            raise NotImplementedError

        def encode_query(self, text):
            return embs[int(text)]

    ad = ExhaustiveDiagnosticAdapter(ExhaustiveBinaryRetriever(idx, meta, SelfEnc()), record_top=30)
    ranked, stages = ad.run("17")
    assert ranked[0] == "d17:1"
    assert stages[0].stage == "exhaustive-binary" and stages[0].latency_ms >= 0


def test_two_stage_adapter_matches_production_score():
    from belge_gozu.bench.harness import TwoStageDiagnosticAdapter
    from belge_gozu.retrieval.core import TwoStageRetriever
    from tests.retrieval.test_core import build_fixture

    idx, meta, embs = build_fixture(n_pages=30)

    class SelfEnc:
        def encode_pages(self, images):
            raise NotImplementedError

        def encode_query(self, text):
            return embs[int(text)]

    retriever = TwoStageRetriever(idx, meta, SelfEnc())
    ad = TwoStageDiagnosticAdapter(retriever, candidates=30, record_top=30)
    ranked, stages = ad.run("17")
    assert [s.stage for s in stages] == ["stage1", "stage2"]
    assert ranked[0] == "d17:1"
    prod_hits = retriever.search("17", k=1, candidates=30)
    assert stages[1].top_scores[0] == prod_hits[0].score
