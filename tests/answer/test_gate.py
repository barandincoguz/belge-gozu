"""P2 T2: iki kapı — kalibre getirim kapısı + kanıt kapısı. AĞ/MODEL YOK.

Bu dosyanın EN ÖNEMLİ testi `test_flags_off_is_byte_identical_to_p1`: kapıların
varsayılan-kapalı olması master §1'in kapı kuralının gereğidir, bir tercih
değil. Kilidin ikinci yarısı `tests/app/test_api.py`'deki MEVCUT /ask
testlerinin DEĞİŞMEDEN geçmesidir.
"""

import numpy as np
import pytest

from belge_gozu.answer.base import (
    ABSTAIN_TEXT,
    HONEST_MISS_MARKER,
    SERVICE_ERROR_TEXT,
    VERIFIER_DEMOTE_TEXT,
    Answer,
    AskService,
)
from belge_gozu.answer.calibrate import (
    FEATURE_ORDER,
    CalibratedRetrievalGate,
    CalibrationArtifact,
    Calibrator,
    calibration_key,
)
from belge_gozu.answer.verify import ClaimVerifier, EvidenceGate, VerifierCache, build_gates
from belge_gozu.config import Settings
from belge_gozu.index.compat import IndexCompatibilityError
from belge_gozu.retrieval.text import BM25Index
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import collecting

PAGE_TEXTS = {
    "a:1": "TÜRK MEDENİ KANUNU\nYerleşim yeri sürekli kalma niyetiyle oturulan yerdir.",
    "b:1": "İŞ KANUNU\nYıllık ücretli izin süresi hizmet süresine göre belirlenir.",
}


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
    """`AskService`in gördüğü yüzey + kalibre kapının okuduğu BM25 künyesi."""

    def __init__(self, hits, bm25=None):
        self._hits = hits
        self.last_bm25_scores = bm25

    def search(self, query, k=5, candidates=200):
        return self._hits[:k]


class CitingAnswerer:
    def __init__(self, text="Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir [S1]."):
        self.text = text
        self.calls = 0

    def answer(self, question, pages, image_loader):
        self.calls += 1
        return Answer(text=self.text, citations=[pages[0].page_id])


class StubClient:
    def __init__(self, reply='{"verdict": "supported", "gerekce": "kanıtta var"}'):
        self.reply = reply
        self.prompts = []

    def generate_json(self, prompt, schema=None):
        self.prompts.append(prompt)
        return self.reply


def evidence_gate(reply, page_texts=None, cache_dir=None):
    client = StubClient(reply)
    verifier = ClaimVerifier(
        client=client, model="m", cache=VerifierCache(cache_dir) if cache_dir else None
    )
    return EvidenceGate(verifier, page_texts or PAGE_TEXTS), client


class ConstantGate:
    """Sabit `p` üreten kalibre kapı ikamesi (kalibratör matematiği burada test edilmez)."""

    def __init__(self, p: float, tau: float = 0.5):
        self.p, self.tau = p, tau
        self.seen_bm25 = "yok"

    def evaluate(self, question, *, bm25=None):
        self.seen_bm25 = bm25
        return {"p": self.p, "tau": self.tau, "passed": self.p >= self.tau}


def service(**kw):
    kw.setdefault("hits", [hit("a:1", 90.0)])
    hits = kw.pop("hits")
    return AskService(
        FakeRetriever(hits, kw.pop("bm25", None)),
        kw.pop("answerer", None) or CitingAnswerer(),
        kw.pop("min_score", 20.0),
        lambda p: b"img",
        **kw,
    )


# --- 1. BAYRAK-KAPALI DEĞİŞMEZLİĞİ (asıl kilit) -------------------------------


def test_flags_off_is_byte_identical_to_p1():
    """Kapı yokken: eşik üstü cevaplanır, eşik altı ABSTAIN_TEXT ile çekimser,
    yanıtlayıcı patlarsa SERVICE_ERROR_TEXT — ve HİÇBİR kapı notu yazılmaz."""
    with collecting() as col:
        answer, hits = service().ask("soru", k=5, candidates=200)
    assert answer.text.endswith("[S1].") and answer.citations == ["a:1"]
    assert not answer.abstained
    assert "gate1" not in col.notes and "gate2" not in col.notes

    with collecting() as col:
        low, _ = service(hits=[hit("a:1", 1.0)]).ask("soru", k=5, candidates=200)
    assert low.abstained and low.text == ABSTAIN_TEXT
    assert "gate1" not in col.notes and "gate2" not in col.notes


def test_ask_service_gates_default_to_none():
    svc = service()
    assert svc.gate1 is None and svc.gate2 is None


# --- 2. Kapı 1 (kalibre getirim) ---------------------------------------------


def test_gate1_abstains_below_tau_and_never_calls_the_answerer():
    answerer = CitingAnswerer()
    with collecting() as col:
        answer, hits = service(answerer=answerer, gate1=ConstantGate(0.10)).ask("soru", k=5)
    assert answer.abstained and answer.text == ABSTAIN_TEXT
    assert answerer.calls == 0, "kapı 1 LLM'e gitmeden durdurmalı (kota)"
    assert col.notes["gate1"] == {"p": 0.10, "tau": 0.5, "passed": False}
    assert hits, "çekimserlikte de sayfalar kaybolmaz"


def test_gate1_passes_above_tau_and_records_p():
    with collecting() as col:
        answer, _ = service(gate1=ConstantGate(0.90)).ask("soru", k=5)
    assert not answer.abstained
    assert col.notes["gate1"]["passed"] is True and col.notes["gate1"]["p"] == 0.90


def test_gate1_is_measured_even_when_the_old_threshold_abstains():
    """Eşik altı satırlarda da `p` kaydedilir: kalibratörün kendi
    değerlendirmesinin en bilgilendirici yarısı tam olarak o satırlardır."""
    with collecting() as col:
        answer, _ = service(hits=[hit("a:1", 1.0)], gate1=ConstantGate(0.99)).ask("soru", k=5)
    assert answer.abstained and answer.text == ABSTAIN_TEXT
    assert col.notes["gate1"]["p"] == 0.99


def test_gate1_receives_the_retrievers_already_computed_bm25():
    scores = np.array([1.0, 2.0], dtype=np.float32)
    gate = ConstantGate(0.90)
    service(gate1=gate, bm25=scores).ask("soru", k=5)
    assert gate.seen_bm25 is scores, "ikinci bir korpus taraması yapılmamalı"


def test_gate1_failure_leaves_the_gate_open_and_is_logged(caplog):
    class BoomGate:
        def evaluate(self, question, *, bm25=None):
            raise RuntimeError("artefakt bozuk")

    with collecting() as col:
        answer, _ = service(gate1=BoomGate()).ask("soru", k=5)
    assert not answer.abstained  # fren hesaplanamadı; eski eşik hâlâ yerinde
    assert col.notes["gate1"]["error"] == "gate1_failed"
    assert "kalibre getirim kapısı" in caplog.text


# --- 3. Kapı 2 (kanıt) --------------------------------------------------------


def test_gate2_lets_a_fully_supported_answer_through():
    gate, client = evidence_gate('{"verdict": "supported", "gerekce": "aynen yazıyor"}')
    with collecting() as col:
        answer, _ = service(gate2=gate).ask("soru", k=5)
    assert not answer.abstained and answer.citations == ["a:1"]
    assert col.notes["gate2"]["demoted"] is False
    assert "Yerleşim yeri" in client.prompts[0]


def test_gate2_demotes_an_unsupported_answer_to_abstained():
    gate, _ = evidence_gate('{"verdict": "unsupported", "gerekce": "sayfada geçmiyor"}')
    with collecting() as col:
        answer, hits = service(gate2=gate).ask("soru", k=5)
    assert answer.abstained is True
    assert answer.text == VERIFIER_DEMOTE_TEXT
    assert answer.text != ABSTAIN_TEXT, "iki durum aynı cümleye sıkıştırılmaz"
    assert answer.citations == [], "doğrulanmamış atıf gösterilmez"
    assert col.notes["gate2"]["demoted"] is True
    assert col.notes["gate2"]["claims"][0]["verdict"] == "unsupported"
    assert hits, "sayfalar kullanıcıdan saklanmaz"


def test_gate2_is_skipped_for_honest_miss():
    gate, client = evidence_gate('{"verdict": "unsupported", "gerekce": "-"}')
    answerer = CitingAnswerer(text=f"Sorunun cevabını {HONEST_MISS_MARKER}.")
    with collecting() as col:
        answer, _ = service(answerer=answerer, gate2=gate).ask("soru", k=5)
    assert not answer.abstained and HONEST_MISS_MARKER in answer.text
    assert col.notes["gate2"] == {"demoted": False, "skipped": "honest_miss"}
    assert client.prompts == [], "dürüst ıska için kota harcanmaz"


def test_gate2_is_skipped_when_there_are_no_citations():
    gate, client = evidence_gate('{"verdict": "unsupported", "gerekce": "-"}')

    class NoCiteAnswerer:
        def answer(self, question, pages, image_loader):
            return Answer(text="Atıfsız ama uzun bir yanıt cümlesi.", citations=[])

    with collecting() as col:
        answer, _ = service(answerer=NoCiteAnswerer(), gate2=gate).ask("soru", k=5)
    assert not answer.abstained
    assert col.notes["gate2"]["skipped"] == "no_citations" and client.prompts == []


def test_gate2_is_skipped_on_the_abstain_and_degraded_paths():
    gate, client = evidence_gate('{"verdict": "unsupported", "gerekce": "-"}')

    class BoomAnswerer:
        def answer(self, question, pages, image_loader):
            raise RuntimeError("kota")

    low, _ = service(hits=[hit("a:1", 1.0)], gate2=gate).ask("soru", k=5)
    assert low.text == ABSTAIN_TEXT
    degraded, _ = service(answerer=BoomAnswerer(), gate2=gate).ask("soru", k=5)
    assert degraded.text == SERVICE_ERROR_TEXT
    assert client.prompts == []


def test_gate2_failure_demotes_rather_than_presenting_a_certain_answer(caplog):
    class BoomGate:
        def evaluate(self, answer, hits):
            raise RuntimeError("disk dolu")

    with collecting() as col:
        answer, _ = service(gate2=BoomGate()).ask("soru", k=5)
    assert answer.abstained and answer.text == VERIFIER_DEMOTE_TEXT
    assert col.notes["gate2"] == {"demoted": True, "error": "RuntimeError"}
    assert "kanıt kapısı" in caplog.text


def test_both_gates_together_gate1_first():
    """Kapı 1 kapalıysa kapı 2'ye (ve LLM'e) hiç gidilmez — sıralama kota kararıdır."""
    gate2, client = evidence_gate('{"verdict": "supported", "gerekce": "-"}')
    answerer = CitingAnswerer()
    answer, _ = service(answerer=answerer, gate1=ConstantGate(0.1), gate2=gate2).ask("soru", k=5)
    assert answer.text == ABSTAIN_TEXT
    assert answerer.calls == 0 and client.prompts == []


# --- 4. Kapı kurulumu (`build_gates`) + fail-fast ------------------------------


def _artifact(key: str) -> CalibrationArtifact:
    cal = Calibrator(
        feature_names=FEATURE_ORDER,
        mean=(0.0,) * len(FEATURE_ORDER),
        std=(1.0,) * len(FEATURE_ORDER),
        weights=(0.0,) * len(FEATURE_ORDER),
        bias=0.0,
        fit_info={},
    )
    return CalibrationArtifact(
        key=key,
        index_revision="rev/train-compat-v1/int8",
        pipeline="hybrid",
        recipe_fingerprint=key.split("__")[-1],
        calibrator=cal,
        thresholds={
            "chosen": {"name": "risk_budget", "value": 0.4, "statistical_guarantee": "none"}
        },
        kunye={},
    )


class TinyRetriever:
    text = BM25Index(["a:1", "b:1"], list(PAGE_TEXTS.values()))
    doc_names: dict[str, frozenset[str]] = {}
    last_bm25_scores = None

    def search(self, query, k=5, candidates=200):
        return [hit("a:1", 90.0)]


def test_build_gates_loads_nothing_when_both_flags_are_off(tmp_path):
    s = Settings(data_dir=tmp_path, index_dir=tmp_path / "yok")
    gates = build_gates(s, TinyRetriever(), index_revision="rev/train-compat-v1/int8")
    assert gates.retrieval is None and gates.evidence is None and gates.detail == {}


def test_missing_calibrator_artifact_fails_fast_with_the_fit_hint(tmp_path):
    s = Settings(data_dir=tmp_path, index_dir=tmp_path / "yok", gate_calibrated=True)
    with pytest.raises(IndexCompatibilityError) as exc:
        build_gates(s, TinyRetriever(), index_revision="rev/train-compat-v1/int8")
    assert "calibrate fit" in str(exc.value) and "BG_GATE_CALIBRATED" in str(exc.value)


def test_gate1_is_built_from_the_versioned_artifact(tmp_path):
    revision = "rev/train-compat-v1/int8"
    key = calibration_key(revision, "hybrid")
    _artifact(key).save(tmp_path / "calibration" / key)
    s = Settings(data_dir=tmp_path, index_dir=tmp_path / "yok", gate_calibrated=True)
    gates = build_gates(s, TinyRetriever(), index_revision=revision)
    assert isinstance(gates.retrieval, CalibratedRetrievalGate)
    assert gates.detail["gate1"] == {"key": key, "tau": 0.4, "guarantee": "none"}
    assert gates.evidence is None  # doğrulayıcı bayrağı kapalıydı


def test_calibrated_gate_evaluate_reports_p_tau_and_features(tmp_path):
    revision = "rev/train-compat-v1/int8"
    key = calibration_key(revision, "hybrid")
    gate = CalibratedRetrievalGate(_artifact(key), TinyRetriever.text, {})
    out = gate.evaluate("yerleşim yeri nedir")
    assert set(out) == {"p", "tau", "passed", "key", "guarantee", "features"}
    assert out["tau"] == 0.4 and 0.0 <= out["p"] <= 1.0
    assert set(out["features"]) == set(FEATURE_ORDER)
    assert out["passed"] is (out["p"] >= 0.4)


def test_calibrated_gate_is_not_defined_outside_the_hybrid_pipeline(tmp_path):
    s = Settings(
        data_dir=tmp_path,
        index_dir=tmp_path / "yok",
        gate_calibrated=True,
        retrieval_pipeline="exhaustive",
        min_score_threshold=0.58,
    )
    with pytest.raises(IndexCompatibilityError, match="hibrit"):
        build_gates(s, TinyRetriever(), index_revision="rev/x/int8")
