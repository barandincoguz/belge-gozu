import time
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from belge_gozu.bench.dataset import BenchQuestion
from belge_gozu.bench.metrics import bootstrap_ci, mrr, ndcg_at_k, recall_at_k
from belge_gozu.index.manifest import IndexManifest
from belge_gozu.index.store import binarize_pack
from belge_gozu.provenance import git_commit  # geriye dönük re-export (eski ev buradaydı)
from belge_gozu.retrieval.core import ExhaustiveRetriever, TwoStageRetriever, hamming_matrix


class StageRecord(BaseModel):
    stage: str  # "exhaustive-binary" | "stage1" | "stage2" | (P1: kanal adları) | "final"
    gold_ranks: dict[str, int]  # page_id -> 1-tabanlı sıra; listede yoksa -1
    top_ids: list[str]  # ilk record_top eleman
    top_scores: list[float]
    latency_ms: float


class QuestionDiagnostic(BaseModel):
    question_id: str
    stages: list[StageRecord]
    candidate_survival: dict[str, bool]  # gold page_id -> nihai aday havuzunda mı
    final_ranked: list[str]  # ilk record_top


class MetricBlock(BaseModel):
    recall_at: dict[int, float]
    mrr: float
    ndcg5: float
    n: int
    ci_recall5: tuple[float, float] | None = None


class EvalReport(BaseModel):
    run_id: str
    git_commit: str
    index_manifest: dict | None
    config: dict
    missing_gold_pages: list[str]  # korpus coverage ihlalleri
    overall: MetricBlock
    per_slice: dict[str, MetricBlock]
    per_doc: dict[str, MetricBlock]
    diagnostics: list[QuestionDiagnostic]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")


class DiagnosticPipeline(Protocol):
    name: str

    def run(self, question: str) -> tuple[list[str], list[StageRecord]]:
        """tam sıralı page_id listesi (en az record_top) + aşama kayıtları"""
        ...


class ExhaustiveDiagnosticAdapter:
    """ExhaustiveRetriever sarar — indeks temsilinden bağımsız.

    Getiriciden YALNIZ `encoder`, `index.page_ids` ve `score_all` kullanılır;
    `score_all` normalize [-1,1] skorları zaten kendisi döndürdüğü için
    burada hiçbir ölçek düzeltmesi YOKTUR (bkz. TwoStageDiagnosticAdapter —
    orada ham toplam normalize edilmek zorunda). Bu yüzden adapter
    packed/int8/float indekslerin üçüyle de çalışır."""

    name = "exhaustive"

    def __init__(self, retriever: ExhaustiveRetriever, record_top: int = 200):
        self.retriever = retriever
        self.record_top = record_top

    def run(self, question: str) -> tuple[list[str], list[StageRecord]]:
        if self.retriever.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        t0 = time.perf_counter()
        q_emb = self.retriever.encoder.encode_query(question)
        scores = self.retriever.score_all(q_emb)
        order = np.argsort(-scores, kind="stable")
        latency_ms = (time.perf_counter() - t0) * 1000
        page_ids = self.retriever.index.page_ids
        ranked = [page_ids[i] for i in order]
        top = order[: self.record_top]
        rec = StageRecord(
            stage="exhaustive-binary",
            gold_ranks={},
            top_ids=[page_ids[i] for i in top],
            top_scores=[float(scores[i]) for i in top],
            latency_ms=latency_ms,
        )
        return ranked, [rec]


class TwoStageDiagnosticAdapter:
    """TwoStageRetriever sarar (B1/B2 ablasyonu).

    Not: stage-1 kaydı `argsort` ile tam korpus üzerinde hesaplanır; üretim
    yolu (`TwoStageRetriever.search_embedding`) içeride `argpartition`
    kullanır — sınır (tie) durumlarında seçilen aday kümesi bu teşhis
    kaydından küçük farklarla ayrışabilir. `gold_ranks` yalnız `record_top`
    ile sınırlı `top_ids` listesine göre hesaplanır (-1 = gold sayfa ilk N'de
    yok, tam-korpus sırası değil); tam-korpus sıra teşhisi (ör. gold sayfanın
    gerçek global rütbesi) oracle koşumlarının işidir (controller ruling R12).
    """

    name = "two-stage"

    def __init__(self, retriever: TwoStageRetriever, candidates: int = 200, record_top: int = 200):
        self.retriever = retriever
        self.candidates = candidates
        # record_top < candidates olursa stage2'nin skorladığı adayların bir
        # kısmı kayda giremez (top_ids[:record_top] kırpması); en az
        # candidates kadar kaydedilmesi garanti edilir.
        self.record_top = max(record_top, candidates)

    def run(self, question: str) -> tuple[list[str], list[StageRecord]]:
        if self.retriever.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        page_ids = self.retriever.index.page_ids

        t0 = time.perf_counter()
        q_emb = self.retriever.encoder.encode_query(question)

        # Aşama 1: mean-sign vektörüyle tam Hamming sıralaması (küçük=iyi ->
        # skor olarak negatif mesafe kaydedilir, böylece "büyük skor=iyi"
        # sözleşmesi tüm aşamalarda tutarlı kalır).
        q_vec = binarize_pack(q_emb.mean(axis=0, keepdims=True))
        dists = hamming_matrix(q_vec, self.retriever.index.page_vecs)[0]
        order1 = np.argsort(dists, kind="stable")
        t1 = time.perf_counter()
        top1 = order1[: self.record_top]
        stage1 = StageRecord(
            stage="stage1",
            gold_ranks={},
            top_ids=[page_ids[i] for i in top1],
            top_scores=[float(-dists[i]) for i in top1],
            latency_ms=(t1 - t0) * 1000,
        )

        # Aşama 2: adaylarda kesin binary MaxSim (RAW toplam `n_q * 128`'e
        # bölünerek normalize edilir — ÜRETİMİN `TwoStageRetriever.search`
        # ile BİREBİR aynı ifadesi; teşhis kaydı üretim skorundan farklı bir
        # ölçekte olursa iki taraf sessizce ayrışır, bkz.
        # tests/bench/test_harness.py::test_two_stage_adapter_matches_production_score).
        # NOT: `search_embedding` üretim kodu, aday seçimini kendi içinde
        # tekrar hesaplar (stage-1 hamming'i `argpartition` ile yeniden
        # çalıştırır) — bu yüzden aşağıdaki latency_ms yalnız "aşama 2"
        # değil, o dahili tekrar-hesaplamayı da (sub-ms mertebesinde)
        # içerir; "stage2-only" değildir.
        hits = self.retriever.search_embedding(q_emb, k=self.candidates, candidates=self.candidates)
        t2 = time.perf_counter()
        n_q = max(1, q_emb.shape[0])
        stage2_ids = [page_ids[i] for i, _ in hits]
        stage2_scores = [score / (n_q * 128) for _, score in hits]
        stage2 = StageRecord(
            stage="stage2",
            gold_ranks={},
            top_ids=stage2_ids[: self.record_top],
            top_scores=stage2_scores[: self.record_top],
            latency_ms=(t2 - t1) * 1000,
        )

        ranked = stage2_ids
        return ranked, [stage1, stage2]


def run_retrieval_eval(
    pipeline: DiagnosticPipeline,
    questions: list[BenchQuestion],
    known_page_ids: set[str],
    ks: tuple[int, ...] = (1, 5, 10, 20, 50, 200),
    run_id: str = "",
    index_manifest: IndexManifest | None = None,
    config: dict | None = None,
) -> EvalReport:
    missing = sorted({p for q in questions for p in q.gold_page_ids if p not in known_page_ids})
    diags: list[QuestionDiagnostic] = []
    rows: list[tuple[BenchQuestion, list[str]]] = []
    for q in questions:
        if not q.answerable:
            continue
        ranked, stages = pipeline.run(q.question)
        rel = set(q.gold_page_ids)
        for st in stages:
            st.gold_ranks = {g: (st.top_ids.index(g) + 1 if g in st.top_ids else -1) for g in rel}
        final_ids = stages[-1].top_ids if stages else ranked
        diags.append(
            QuestionDiagnostic(
                question_id=q.question_id,
                stages=stages,
                candidate_survival={g: g in set(final_ids) for g in rel},
                final_ranked=ranked[: max(ks)],
            )
        )
        rows.append((q, ranked))

    def block(pairs: list[tuple[BenchQuestion, list[str]]]) -> MetricBlock:
        if not pairs:
            return MetricBlock(recall_at={k: 0.0 for k in ks}, mrr=0.0, ndcg5=0.0, n=0)
        r5 = [recall_at_k(set(q.gold_page_ids), r, 5) for q, r in pairs]
        return MetricBlock(
            recall_at={
                k: sum(recall_at_k(set(q.gold_page_ids), r, k) for q, r in pairs) / len(pairs)
                for k in ks
            },
            mrr=sum(mrr(set(q.gold_page_ids), r) for q, r in pairs) / len(pairs),
            ndcg5=sum(ndcg_at_k(set(q.gold_page_ids), r, 5) for q, r in pairs) / len(pairs),
            n=len(pairs),
            ci_recall5=bootstrap_ci(r5),
        )

    per_slice: dict[str, list[tuple[BenchQuestion, list[str]]]] = {}
    per_doc: dict[str, list[tuple[BenchQuestion, list[str]]]] = {}
    for q, r in rows:
        per_slice.setdefault(q.slice, []).append((q, r))
        for d in q.gold_doc_ids:
            per_doc.setdefault(d, []).append((q, r))
    return EvalReport(
        run_id=run_id,
        git_commit=git_commit(),
        index_manifest=index_manifest.model_dump() if index_manifest else None,
        config=config or {},
        missing_gold_pages=missing,
        overall=block(rows),
        per_slice={s: block(p) for s, p in per_slice.items()},
        per_doc={d: block(p) for d, p in per_doc.items()},
        diagnostics=diags,
    )
