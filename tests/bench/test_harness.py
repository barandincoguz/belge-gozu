from pathlib import Path

from belge_gozu.bench.dataset import BenchQuestion
from belge_gozu.bench.harness import StageRecord, run_retrieval_eval
from tests.bench.test_dataset import q_dict


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
