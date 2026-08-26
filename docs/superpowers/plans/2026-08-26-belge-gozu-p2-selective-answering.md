# Belge-Gözü P2 — Selective Answering, Citation ve Kalibrasyon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Yalnız kanıtlı yanıt sunmak: claim-level evidence verification + claim-level
citation (auto-citation fallback kaldırılır), retrieval gate'ten ayrı evidence
sufficiency gate, raw skor yerine çok-özellikli kalibre güven ve risk-coverage'la
seçilmiş abstention.

**Architecture:** `answer/verify.py` (claim segmentasyonu + verifier),
`answer/calibrate.py` (özellikler + kalibratör), `bench/answer_eval.py`
(citation/selective metrikleri), AskService'in iki-kapılı yeniden düzenlenmesi,
UI claim-citation, outcome telemetry. Fine-tuning yalnız koşullu alt plan kapısı.

**Tech Stack:** Mevcut stack; kalibratör fit'i için `eval` extra (`scikit-learn`),
çalışma anı saf numpy (katsayılar JSON'da). Verifier mevcut Gemini istemcisi üstünde
structured output + önbellek.

**Spec:** `docs/superpowers/specs/2026-08-26-belge-gozu-rag-quality-v2-design.md`
**Master:** `docs/superpowers/plans/2026-08-26-belge-gozu-rag-quality-master.md`

## Global Constraints

- **Önkoşul: P1 kapısı (G1) PASS ve raporu commit'li.**
- CI'da ağ/GPU/model/API yok; verifier/answerer testlerde stub. Gemini kotası
  (≈20 çağrı/gün): tüm verifier/answer koşumları sha256-önbeklekli
  (`data/cache/verifier/`), koşum bütçeleri her runbook adımında açık; koşumlar güne
  bölünür.
- Kalibrasyon verisi = **dev split**; final sayılar = **test split**, faz sonunda bir
  kez (G2.4). Threshold'lar `data/calibration/<index_revision>/` altında versiyonlu
  (G2.5); `index_revision` P0 T13'teki dizedir.
- Kapı: master §5 G2.1-G2.8. Verifier geçmeyen yanıt kullanıcıya kesin yanıt olarak
  gösterilmez (ilke 20); raw skor güven gibi gösterilmez (ilke 18).

## File Structure

```
src/belge_gozu/
  answer/verify.py         # Claim, ClaimVerdict, CitationRef, VerifiedAnswer,
                           # segment_claims, VerifierClient, GeminiVerifier            [T1]
  answer/base.py           # AskService: retrieval gate + evidence gate ayrımı         [T2,T8]
  answer/gemini.py         # auto-citation fallback KALDIRILIR                         [T3]
  answer/calibrate.py      # ConfidenceFeatures, extract_features, Calibrator          [T5,T6]
  bench/answer_eval.py     # citation precision/completeness, selective metrikler,
                           # answerable/unanswerable koşum harness'ı                   [T3,T4]
  bench/calibration_metrics.py  # brier, ece, auroc, risk_coverage                     [T7]
  app/main.py + app/static/index.html  # /feedback, claim-citation UI, outcome alanlar [T9]
  telemetry/schema.py      # outcome alanları                                          [T9]
scripts/collect_calibration.py, scripts/drift_report.py                                [T5,T9]
data/calibration/<index_revision>/calibrator.json                                      [T6]
tests/answer/test_verify.py, tests/answer/test_gate.py, tests/answer/test_calibrate.py,
tests/bench/test_answer_eval.py, tests/bench/test_calibration_metrics.py
docs/research/findings/2026-XX-XX-p2-gate.md                                           [T12]
```

---

### Task 1: Claim segmentasyonu + verifier (`answer/verify.py`)

**Files:**
- Create: `src/belge_gozu/answer/verify.py`, `tests/answer/test_verify.py`

**Interfaces:**
- Consumes: P1 `EvidencePack/EvidenceUnit` (`retrieval/evidence.py`) — birebir.
- Produces:

```python
class Claim(BaseModel):
    claim_id: str            # "c1", "c2", ...
    text: str

class CitationRef(BaseModel):
    claim_id: str
    article_id: str | None
    page_id: str

class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal["supported", "refuted", "insufficient"]
    evidence_refs: list[CitationRef]     # supported ise >=1
    quote: str                           # kanıttan birebir alıntı ("" olabilir)

class VerifiedAnswer(BaseModel):
    text: str
    claims: list[Claim]
    verdicts: list[ClaimVerdict]
    citations: list[CitationRef]         # yalnız supported claim'lerinkiler
    abstained: bool
    abstain_reason: Literal["retrieval", "evidence", "confidence", "service"] | None = None

def segment_claims(text: str) -> list[Claim]
    # Deterministik (kota yakmaz): cümle bölme — "." "!" "?" sonu + büyük harf/rakam
    # başlangıcı; "[S1]" tarzı işaretler cümleden temizlenir; kısa (<15 karakter) ve
    # yalnız-bağlaç parçalar önceki cümleye eklenir; boş metin -> [].

class VerifierClient(Protocol):
    def verify(self, question: str, claims: list[Claim],
               pack: EvidencePack) -> list[ClaimVerdict]: ...

class GeminiVerifier:
    """google-genai structured output (response_schema=list[ClaimVerdict şeması]).
    İstem: her claim için pack.units metin/görüntülerine dayanarak
    supported/refuted/insufficient + dayanak birimi + birebir alıntı ister.
    Önbellek: sha256(question + claim metinleri + pack unit kimlikleri) ->
    data/cache/verifier/<hash>.json; isabet API'ye gitmez.
    Parse edilemeyen/eksik claim yanıtı 'insufficient' sayılır (asla 'supported'
    varsayılmaz)."""
    def __init__(self, model: str, api_key: str, cache_dir: Path, client=None): ...
```

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/answer/test_verify.py
from belge_gozu.answer.verify import Claim, segment_claims


def test_segment_basic():
    text = ("Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir [S1]. "
            "Bir kimsenin birden çok yerleşim yeri olamaz [S1].")
    claims = segment_claims(text)
    assert [c.claim_id for c in claims] == ["c1", "c2"]
    assert "[S1]" not in claims[0].text and "sürekli kalma" in claims[0].text


def test_segment_merges_short_fragments():
    claims = segment_claims("Kural budur. Ancak. İstisna 320. maddededir.")
    assert len(claims) == 2  # "Ancak." tek başına claim olmaz


def test_segment_empty():
    assert segment_claims("") == []


def test_gemini_verifier_cache_and_unparseable(tmp_path):
    from belge_gozu.answer.verify import GeminiVerifier
    from belge_gozu.retrieval.evidence import EvidencePack
    from belge_gozu.retrieval.query import QueryFacets

    calls = []

    class StubClient:
        def generate_structured(self, prompt, images, schema):
            calls.append(prompt)
            return [{"claim_id": "c1", "verdict": "supported",
                     "evidence_refs": [{"claim_id": "c1", "article_id": "k4721:m19",
                                        "page_id": "k4721:4"}],
                     "quote": "sürekli kalma niyetiyle"}]

    pack = EvidencePack(question="q", facets=QueryFacets(law_numbers=[], doc_ids=[],
                        article_nos=[], quoted_phrases=[]), units=[])
    v = GeminiVerifier("m", "key", cache_dir=tmp_path, client=StubClient())
    claims = [Claim(claim_id="c1", text="a"), Claim(claim_id="c2", text="b")]
    out1 = v.verify("q", claims, pack)
    assert {x.claim_id: x.verdict for x in out1} == {"c1": "supported", "c2": "insufficient"}
    out2 = v.verify("q", claims, pack)   # önbellek isabeti
    assert len(calls) == 1 and out2 == out1
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/answer/test_verify.py -v`
- [ ] **Step 3: verify.py yaz** — sözleşme yukarıda; `GeminiVerifier.verify`:
  istem = soru + numaralı claim listesi + birim başına `article_id/page_id` etiketli
  metin (metin yoksa sayfa görüntüsü eklenir); tek API çağrısı (claim başına değil);
  şema-dışı/eksik yanıtlar `insufficient`. `GeminiClient`'a
  `generate_structured(prompt, images, schema) -> list[dict]` metodu eklenir
  (`google-genai` `response_mime_type="application/json"` + `response_schema`).
- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/answer -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(verify): deterministic claim segmentation + cached structured verifier`

---

### Task 2: İki kapı — retrieval gate ↔ evidence sufficiency gate (`answer/base.py`)

**Files:**
- Modify: `src/belge_gozu/answer/base.py`, `tests/answer/test_gate.py` (yeni),
  `src/belge_gozu/config.py` (`evidence_verifier_enabled: bool = False`)

**Interfaces:**
- Produces: `AskService.ask(...) -> tuple[VerifiedAnswer, list[PageHit]]`
  (dönüş tipi `Answer` yerine `VerifiedAnswer`; `Answer` alanları korunduğundan
  API şeması geriye uyumlu genişler). Akış:
  1. retrieval (P1 `retrieve_evidence` → pack) — **retrieval gate**: hit yoksa
     `abstained=True, abstain_reason="retrieval"` (eski skor eşiği artık TEK kapı
     değil; `min_score_threshold` yalnız flag kapalıyken eski davranış olarak kalır);
  2. answerer → aday yanıt metni;
  3. flag açıksa: `segment_claims` → `verifier.verify` — **evidence gate**:
     karar politikası `decide_verdicts` (aşağıda);
  4. sonuç `VerifiedAnswer` (citations yalnız supported claim'lerden).

```python
def decide_verdicts(verdicts: list[ClaimVerdict]) -> Literal["present", "retry", "abstain"]
    # herhangi bir 'refuted' -> "abstain"
    # tümü 'supported' -> "present"
    # 'insufficient' var, 'refuted' yok -> "retry" (bir kez: unsupported claim'ler
    #   çıkarılmış kısıtlı istemle yeniden üretim; ikinci turda hâlâ insufficient
    #   varsa "abstain")
```

- Consumes: T1 tipleri, P1 `EvidencePack`.

- [ ] **Step 1: Başarısız testleri yaz** — `tests/answer/test_gate.py`: stub retriever +
  stub answerer + stub verifier ile: (a) tümü supported → `abstained=False`, citations
  supported claim'lerden; (b) bir refuted → abstain, `abstain_reason="evidence"`;
  (c) insufficient → answerer İKİNCİ kez kısıtlı istemle çağrılır; ikinci tur temizse
  present, değilse abstain; (d) flag kapalıyken davranış P1 ile birebir (verifier hiç
  çağrılmaz, eski eşik yolu işler); (e) `decide_verdicts` üç dalı birim testli.
- [ ] **Step 2: RED → uygula → GREEN** — Run: `uv run pytest tests/answer -v`
- [ ] **Step 3: Full regression** (app testleri dahil — `/ask` şeması genişledi ama
  eski alanlar duruyor) — Run: `uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 4: Commit** — `feat(gate): separate retrieval and evidence-sufficiency gates`

---

### Task 3: Auto-citation kaldırma + citation metrikleri

**Files:**
- Modify: `src/belge_gozu/answer/gemini.py` (fallback satırları SİLİNİR),
  `tests/answer/test_gemini.py`
- Create: `src/belge_gozu/bench/answer_eval.py`, `tests/bench/test_answer_eval.py`

**Interfaces:**
- Produces (gemini.py): citation üretilmediyse `citations=[]` döner — nokta.
  (G2.7; sayfa-1'e bağlama davranışı `answer/gemini.py:81-82`'den kaldırılır.)
- Produces (answer_eval.py):

```python
class AnswerRecord(BaseModel):
    question_id: str
    decision: Literal["answered", "abstained"]
    claims_total: int
    claims_supported: int
    citations: list[CitationRef]
    calibrated_conf: float | None = None

def citation_precision(records: list[AnswerRecord],
                       gold: dict[str, set[str]]) -> float
    # verilen citation'lardan gold_page_ids (veya gold_article_ids) kümesine düşenlerin oranı
def citation_completeness(records, gold) -> float
    # answered kayıtlarda gold kümesinden en az bir öğeye atıf yapılan soru oranı
def selective_metrics(records: list[AnswerRecord],
                      answerable: dict[str, bool]) -> dict[str, float]
    # coverage, false_answer_rate (unanswerable'da answered), false_abstain_rate
    # (answerable'da abstained), unsupported_claim_rate
```

- [ ] **Step 1: Başarısız testleri yaz** — (a) `tests/answer/test_gemini.py`'deki
  mevcut "fallback top-1" beklentisi TERSine çevrilir: citation'sız yanıt →
  `citations == []`; (b) `test_answer_eval.py`: küçük elle kurulmuş kayıt setinde
  dört metriğin bilinen değerleri (ör. 2 answered/1 doğru atıf → precision hesabı,
  unanswerable'da 1 answered → false_answer_rate 0.5).
- [ ] **Step 2: RED → uygula → GREEN** — Run:
  `uv run pytest tests/answer tests/bench/test_answer_eval.py -v`
- [ ] **Step 3: Full regression + Commit** —
  `fix(citation): remove auto-citation fallback; add citation/selective metrics`

---

### Task 4: Answerable/unanswerable davranış koşum harness'ı

**Files:**
- Modify: `src/belge_gozu/bench/answer_eval.py`, `src/belge_gozu/cli.py`
  (`bench answers`)
- Create: `tests/bench/test_answer_eval.py` genişletmesi

**Interfaces:**
- Produces: `run_answer_eval(service: AskService, questions: list[BenchQuestion],
  judge: "AnswerJudge | None" = None) -> AnswerEvalReport` — her soru için `/ask`
  akışını koşar, `AnswerRecord` üretir; answerable sorularda `answer_accuracy` yalnız
  `judge` verilirse doldurulur (T10), yoksa `None` (insan spot-check runbook'u).
  `AnswerEvalReport`: kayıtlar + `citation_precision/completeness` +
  `selective_metrics` + dilim kırılımı (`korpus-disi`/`eksik-kanit`/`anlamsiz-ood`
  ayrı satırlar — UAEval4RAG uyarlaması) + koşum künyesi (P0 EvalReport deseni).
  CLI: `belge-gozu bench answers --bench PATH --split {dev,test} --out PATH`
  (kota uyarısı + `--yes-burn-quota`; önbellek sayesinde tekrar koşumlar bedava).
- Consumes: T2 AskService, T3 metrikleri, P0 `BenchQuestion`.

- [ ] **Step 1: Başarısız test yaz** — stub AskService (soru→sabit VerifiedAnswer
  haritası) ile: rapor metrikleri, unanswerable dilim kırılımı, `answerable=False`
  soruların da koşulduğu (retrieval metriklerinin aksine).
- [ ] **Step 2: RED → uygula → GREEN** — Run: `uv run pytest tests/bench -v`
- [ ] **Step 3: Full regression + Commit** —
  `feat(bench): end-to-end answer/abstention evaluation harness`

---

### Task 5: Güven özellikleri (`answer/calibrate.py` — özellik çıkarımı)

**Files:**
- Create: `src/belge_gozu/answer/calibrate.py`, `tests/answer/test_calibrate.py`,
  `scripts/collect_calibration.py`

**Interfaces:**
- Produces:

```python
class ConfidenceFeatures(BaseModel):
    rerank_top1: float          # CE skoru (rerank kapalıysa fused top1)
    margin_1_2: float           # top1 - top2 (aktif skor uzayında)
    channel_agreement: float    # gold-adayı ilk 10'unda bulunduran kanal oranı [0,1]
    exact_match: bool           # tırnaklı ifade/phrase_hits isabeti var mı
    law_match: bool             # facets.doc_ids ile top1 doc_id örtüşüyor mu
    article_match: bool         # facets.article_nos ile top1 article örtüşüyor mu
    verifier_support_ratio: float  # supported / toplam claim (verifier yoksa -1.0)

def extract_features(pack: EvidencePack, channel_rankings: dict[str, list[str]],
                     verdicts: list[ClaimVerdict] | None) -> ConfidenceFeatures
FEATURE_ORDER: tuple[str, ...]           # vektörleştirme sırası (sabit, versiyonlu)
def to_vector(f: ConfidenceFeatures) -> np.ndarray   # (7,) float32; bool->0/1
```

- Consumes: P1 `EvidencePack` + `HybridRetriever.channel_rankings`, T1 `ClaimVerdict`.
- `scripts/collect_calibration.py`: dev split answerable+unanswerable soruları koşup
  `(features, label)` çiftlerini `data/calibration/raw_dev.jsonl`'e yazar; `label=1`
  ⇔ soru answerable VE top-5'te gold sayfa var VE (verifier açıksa) karar "present"
  doğru atıfla — yani "bu yanıtı sunmak güvenliydi" etiketi.

- [ ] **Step 1: Başarısız testleri yaz** — elle kurulmuş pack/rankings/verdicts ile
  her özelliğin beklenen değeri; `to_vector` sırasının `FEATURE_ORDER` ile birebir
  olduğu; verifier'sız `verifier_support_ratio == -1.0`.
- [ ] **Step 2: RED → uygula → GREEN** — Run: `uv run pytest tests/answer/test_calibrate.py -v`
- [ ] **Step 3: Toplama koşumu (runbook)** — Run:
  `uv run python scripts/collect_calibration.py --split dev` (kota bütçesi: verifier
  önbelleklidir; ilk koşum güne bölünür). Çıktı satır sayısı basılır.
- [ ] **Step 4: Full regression + Commit** —
  `feat(calibrate): multi-signal confidence features (raw score is not confidence)`

---

### Task 6: Kalibratör + versiyonlu threshold (`answer/calibrate.py` devamı)

**Files:**
- Modify: `src/belge_gozu/answer/calibrate.py`, `tests/answer/test_calibrate.py`,
  `pyproject.toml` (`eval` extra: `scikit-learn>=1.5`)

**Interfaces:**
- Produces:

```python
class Calibrator:
    """Lojistik regresyon (birincil) — fit offline sklearn ile (eval extra),
    çalışma anı saf numpy: p = sigmoid(w·x + b). Platt zaten lojistik; isotonic
    ikincil deney olarak fit edilir ve (x_thresholds, y_values) dizileriyle
    saklanır, predict np.interp."""
    kind: Literal["logistic", "isotonic"]
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Calibrator"       # lazy sklearn
    def predict_proba(self, x: np.ndarray) -> float                    # saf numpy
    def save(self, path: Path) -> None                                 # JSON (w, b | eğri)
    @classmethod
    def load(cls, path: Path) -> "Calibrator"

class CostMatrix(BaseModel):
    false_answer: float = 10.0   # hukuk alanı: yanlış kesin yanıt en pahalı
    false_abstain: float = 1.0

def choose_threshold(probs: np.ndarray, labels: np.ndarray,
                     cost: CostMatrix) -> float
    # dev üzerinde beklenen maliyeti minimize eden tau (aday tau'lar: benzersiz probs)

def calibration_dir(index_revision: str) -> Path
    # data/calibration/<index_revision-guvenli-ad>/ ; calibrator.json + threshold.json
    # + fit raporu (n, metrikler, git sha) — G2.5 versiyonlama
```

- [ ] **Step 1: Başarısız testleri yaz** — sentetik ayrılabilir veri: fit sonrası
  AUROC>0.9 (T7 fonksiyonuyla); save/load roundtrip `predict_proba` birebir;
  `choose_threshold` — elle kurulmuş 4-örnekli durumda bilinen optimum tau;
  sklearn importu YALNIZ `fit`'te (load+predict sklearn'süz çalışır — monkeypatch ile
  import engellenip doğrulanır).
- [ ] **Step 2: RED → uygula → GREEN** — Run: `uv run pytest tests/answer/test_calibrate.py -v`
- [ ] **Step 3: Fit koşumu (runbook)** — Run:
  `uv run --extra eval python -c "..."` yerine CLI: `belge-gozu calibrate fit
  --raw data/calibration/raw_dev.jsonl` (bu adımda `cli.py`'ye eklenir) →
  `data/calibration/<index_revision>/` artefaktları; logistic vs isotonic dev
  karşılaştırması fit raporuna.
- [ ] **Step 4: Full regression + Commit** —
  `feat(calibrate): versioned logistic/isotonic calibrator with legal cost matrix`

---

### Task 7: Kalibrasyon metrikleri + risk-coverage (`bench/calibration_metrics.py`)

**Files:**
- Create: `src/belge_gozu/bench/calibration_metrics.py`,
  `tests/bench/test_calibration_metrics.py`

**Interfaces:**
- Produces:

```python
def brier(probs: np.ndarray, labels: np.ndarray) -> float          # mean (p-y)^2
def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float
def auroc(probs: np.ndarray, labels: np.ndarray) -> float          # rank-tabanlı (bağ düzeltmeli)
def risk_coverage(probs: np.ndarray, labels: np.ndarray) -> list[tuple[float, float, float]]
    # tau taraması -> (tau, coverage, risk=yanlışların answered içindeki oranı)
def conformal_threshold(probs: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> float
    # koşullu deney (split-conformal, hata üst sınırı alpha): dev kalibrasyon
    # alt kümesinde (1-alpha) kuantil; raporda karşılaştırma satırı — üretime ancak
    # dev'de cost-matrix seçiminden iyi kalırsa girer
```

- [ ] **Step 1: Başarısız testleri yaz** — bilinen küçük örneklerle: mükemmel
  ayrımda auroc=1.0, rastgelede ~0.5; brier([1,0],[1,0])=0; ece mükemmel kalibrede 0;
  risk_coverage monotonluğu (tau ↑ ⇒ coverage ↓); conformal_threshold kuantil doğruluğu.
- [ ] **Step 2: RED → uygula → GREEN** — Run:
  `uv run pytest tests/bench/test_calibration_metrics.py -v`
- [ ] **Step 3: Full regression + Commit** —
  `feat(bench): calibration metrics (Brier/ECE/AUROC) and risk-coverage`

---

### Task 8: Selective answering entegrasyonu + güvenli fallback

**Files:**
- Modify: `src/belge_gozu/answer/base.py`, `src/belge_gozu/app/main.py`,
  `src/belge_gozu/config.py` (`selective_answering_enabled: bool = False`),
  `tests/answer/test_gate.py`

**Interfaces:**
- Produces: AskService akışına 3. kapı: **confidence gate** — `extract_features` →
  `Calibrator.predict_proba` → `p < tau` ise `abstained=True,
  abstain_reason="confidence"` (yanıt üretilmiş olsa bile sunulmaz; UI "emin değilim"
  + bulunan sayfaları kaynak önerisi olarak gösterir — kesin yanıt DEĞİL).
  Kapı sırası: retrieval → answer üretimi → evidence gate (T2) → confidence gate.
  **Güvenli fallback (G2.8):** kalibratör dosyası yok/yüklenemedi → confidence gate
  devre dışı + log WARNING + `/healthz`'de `"calibrator": "missing"`; verifier hatası
  → `abstain_reason="service"` (mevcut degraded yolu). Flag'ler kapalıyken P1 davranışı.
- Consumes: T2 kapıları, T5/T6 kalibratör.

- [ ] **Step 1: Başarısız testleri yaz** — stub kalibratörle: p düşük → confidence
  abstain (yanıt metni sunulmaz); p yüksek → present; kalibratör yokken gate atlanır
  ve healthz raporlar; kapı sırasının doğruluğu (refuted claim varken p yüksek olsa
  da evidence abstain).
- [ ] **Step 2: RED → uygula → GREEN** — Run: `uv run pytest tests/answer tests/app -v`
- [ ] **Step 3: Full regression + Commit** —
  `feat(answer): calibrated selective answering with safe fallbacks`

---

### Task 9: UI claim-citation + outcome telemetry + feedback + drift

**Files:**
- Modify: `src/belge_gozu/app/static/index.html`, `src/belge_gozu/app/main.py`,
  `src/belge_gozu/telemetry/schema.py`, `src/belge_gozu/telemetry/prom.py`,
  `tests/app/test_api.py`
- Create: `scripts/drift_report.py`

**Interfaces:**
- Produces:
  - UI: yanıt claim-claim gösterilir; her claim'in yanında atıf çipleri
    (madde adı + sayfa küçük resmi linki); abstain ekranı `abstain_reason`'a göre üç
    farklı, dürüst mesaj (retrieval/evidence/confidence). Skor etiketi P0 T14 metnini korur.
  - Telemetri: `RequestEvent`'e `claims_total, claims_supported, verifier_ms,
    calibrated_conf, decision` nullable alanları (+ recorder ALTER migrasyonu, P0 T13
    deseni); prom: `bg_claims_supported_ratio` Histogram, `bg_abstain_total`'a
    `reason` etiket değerleri (`evidence`, `confidence` — mevcut `threshold/degraded`
    korunur).
  - `/feedback` endpoint'i: `{request_ts, verdict: "up"|"down", note?}` → events
    tablosuna `endpoint="/feedback"` olayı (error taxonomy alanı `detail.taxonomy`:
    kullanıcı `down` derse UI 4 seçenek sunar: `yanlis-cevap`, `yanlis-kaynak`,
    `bulamadi`, `diger`).
  - `scripts/drift_report.py`: events'ten haftalık pencerelerle top_score /
    calibrated_conf / abstain-rate dağılım karşılaştırması (KS istatistiği + tablo);
    çıktı `docs/research/findings/` altına tarihli not.
- [ ] **Step 1: Başarısız testler → uygula → GREEN** — feedback endpoint'i olay yazar;
  yeni alanlar NULL-geriye-uyumlu; `/metrics` yeni serileri içerir. Run:
  `uv run pytest tests/app tests/telemetry -v`
- [ ] **Step 2: Full regression + Commit** —
  `feat(ui,telemetry): claim-level citations, outcome events, feedback and drift tooling`

---

### Task 10: İnsan-kalibreli otomatik değerlendirici (LLM-judge yardımcı)

**Files:**
- Create: `src/belge_gozu/bench/judge.py`, `tests/bench/test_judge.py`

**Interfaces:**
- Produces:

```python
class AnswerJudge(Protocol):
    def judge(self, question: str, reference_answer: str, answer_text: str) -> bool
        # "yanıt referansla tutarlı mı" ikili kararı

class GeminiJudge:
    """Structured output + önbellek (GeminiVerifier deseni)."""

class PPICorrector:
    """ARES tarzı: judge kararları, insan-etiketli alt kümede ölçülen hata oranıyla
    düzeltilir. correct(judge_rate, human_pairs) -> (tahmin, guven_araligi):
    prediction-powered düzeltme: tahmin = judge_rate - (judge_bias), CI bootstrap."""
    def correct(self, judged: list[bool],
                human_pairs: list[tuple[bool, bool]]) -> tuple[float, tuple[float, float]]
        # human_pairs: (judge_karari, insan_karari) aynı örnekler üstünde
```

Kural (spec ilkesi): judge çıktısı TEK BAŞINA kapı kararı olamaz; yalnız
`human_pairs` ≥ 30 örnekle kalibre edilmiş haliyle rapora girer; G2 sayılarında
answer_accuracy "judge (PPI-düzeltmeli, CI'lı)" + "insan spot-check (n)" olarak iki
satır gösterilir.

- [ ] **Step 1: Başarısız testler** — `PPICorrector.correct`: judge her zaman doğruysa
  düzeltme ≈ judge_rate; sistematik iyimser judge'da (insan %20 reddediyor) tahmin
  düşer; CI deterministik (seed'li bootstrap).
- [ ] **Step 2: RED → uygula → GREEN → Commit** —
  `feat(bench): PPI-corrected LLM judge as human-calibrated helper`

---

### Task 11: Koşullu fine-tuning alt plan kapısı (doküman görevi)

**Files:**
- Modify: `docs/superpowers/plans/2026-08-26-belge-gozu-p2-selective-answering.md`
  (bu bölüm zaten kapıyı tanımlar) — koşul tetiklenirse Create:
  `docs/superpowers/plans/2026-XX-XX-belge-gozu-p2ft-finetune.md`

**Kapı (G2-FT):** fine-tuning alt planı YALNIZ şu üçü birden doğruysa yazılır ve
kullanıcı onayına sunulur:
1. G1 ve G2 kapıları PASS (baseline'lar tamamlanmış — ilke 23'ün P2 karşılığı);
2. hybrid+rerank hattında `paraphrase` veya `dogrudan-madde` diliminde Recall@5 < %80
   (yani mimari-dışı, model-içi bir açık kanıtlı);
3. bBSARD kanıtı yerelde doğrulanmış (küçük dile-özgü model fine-tune'unun kazanç
   potansiyeli — spec §9.1).

**Alt planın zorunlu içeriği (şimdiden bağlayıcı):** Türkçe hukuk query→madde/sayfa
çiftleriyle retriever (dense ve/veya ColSmol LoRA) fine-tune; hard negatives —
aynı kanunun komşu maddeleri, kapak/içindekiler sayfaları, aynı terimi kullanan farklı
kanun maddeleri (benchmark `ayni-kanun-hard-negative`/`capraz-kanun-terim`
dilimlerinin üretim mantığıyla); **entire-law held-out** değerlendirme (eğitimde hiç
görülmemiş kanunlar); üretilmiş-sorgu stiline overfit'i önlemek için sorgu stili
karışımı (`query_style` alanı katmanlı örnekleme) + insan paraphrase seti; abstention
ölçümü fine-tune sonrası ZORUNLU tekrar (SearchFireSafety bulgusu: domain-adapted
modeller eksik kanıtta daha çok halüsinasyon — spec §9.1); eğitim/test leakage'i
law-grouped split'le engelli.

- [ ] **Step 1:** G2 raporu yazılırken kapı koşulları değerlendirilir; sonuç
  (tetiklendi/tetiklenmedi + sayılar) p2-gate raporuna bir bölüm olarak yazılır.
- [ ] **Step 2 (koşullu):** tetiklendiyse alt plan dosyası yazılır ve KULLANICI
  ONAYINA sunulur; onaysız uygulanmaz. Commit —
  `plan: conditional fine-tuning sub-plan (gate triggered, run <run_id>)`

---

### Task 12: P2 kapı raporu + final koşum + README

**Files:**
- Create: `docs/research/findings/2026-XX-XX-p2-gate.md`
- Modify: `README.md`

- [ ] **Step 1: Final test koşumu (bir kez, test split):** Run:
  `uv run belge-gozu bench answers --bench data/bench/bench_v2.jsonl --split test --yes-burn-quota --out data/bench/results/<run_id>.json`
  (kota planı: önbellek + güne bölme; koşum künyesi rapora). H1 (verifier aç/kapa) ve
  H2 (raw eşik vs kalibre) karşılaştırmaları dev'de; test'te yalnız kilitli
  konfigürasyon.
- [ ] **Step 2: Kapı raporu:** master §5 G2.1-G2.8 satır satır sayılarla; risk-coverage
  eğrisi (tablo + `docs/research/figures/` PNG); false-answer ≤ %2 ve citation
  precision ≥ %98 kontrolleri; FAIL varsa güvenli fallback kararı (hangi flag kapalı
  kalıyor) açıklanır.
- [ ] **Step 3: README:** Sonuç bölümü — retrieval tabloları (P1) + answering
  tabloları (P2) + risk-coverage özeti + dürüst sınırlar; skor/etiket dili P0 T14 ile
  tutarlı.
- [ ] **Step 4: Full regression + slow süit** — Run:
  `uv run pytest -q -m "not slow" && make lint && uv run pytest -m slow -v`
- [ ] **Step 5: Commit** — `docs: P2 gate report and final results`

---

## P2 Tamamlanma Kapısı (go/no-go)

Master §5 G2.1-G2.8 + bu plandaki ek kurallar: unanswerable false supported-answer
≤ %2 (G2.1), claim citation support precision ≥ %98 (G2.2), coverage bedeli
risk-coverage ile raporlu (G2.3), kalibrasyon/test ayrıklığı (G2.4), threshold
versiyonlama (G2.5), verifier'sız kesin yanıt yok (G2.6), auto-citation kaldırıldı
(G2.7), güvenli fallback tanımlı+testli (G2.8). Hedefler sağlanamazsa: verifier ve
selective flag'leri kapalı kalır (sistem P1 davranışına döner), eksikler raporda,
yayın G2'siz yapılmaz.

## Self-Review (yazar kontrolü)

1. **Spec kapsaması:** retrieval/evidence gate ayrımı (T2), candidate answer (mevcut
   answerer + T2 retry), claim segmentation (T1), support/refute/insufficient (T1),
   minimum supporting set → `evidence_refs` + citations (T1/T3), citation precision +
   completeness (T3), auto-citation kaldırma (T3), unsupported → düzeltme/abstention
   (T2 `decide_verdicts`), answerable/unanswerable/OOD/corpus-missing/corrupted testleri
   (T4 + bench dilimleri), confidence özellikleri — reranker skoru/exact match/
   law-article match/retriever agreement/margin/verifier skoru (T5), logistic/Platt/
   isotonic (T6), held-out threshold (T6), cost matrix (T6), risk-coverage (T7),
   Brier/ECE/AUROC (T7), false-answer/false-abstain/selective accuracy (T3/T4),
   conformal koşullu deney (T7), RAGAS/ARES-tarzı yardımcı — insan-kalibreli (T10),
   claim-level citation UI (T9), outcome telemetry (T9), feedback + error taxonomy
   (T9), drift izleme (T9), fine-tuning karar kapısı + hard negatives + entire-law
   held-out + overfit önlemi (T11).
2. **Placeholder taraması:** yok; koşullu alt plan (T11) bir TBD değil, tetikleme
   koşulları ve zorunlu içeriği tanımlı bir kapıdır.
3. **Tip tutarlılığı:** `EvidencePack/EvidenceUnit` P1 T11 ile, `ClaimVerdict/
   CitationRef/VerifiedAnswer` master §3 ile, `ConfidenceFeatures/Calibrator` master
   §3 ile birebir; `AnswerRecord` yalnız bu planda ve T3/T4 arasında tutarlı.
4. **Bağımlılık:** T1→T2→T3→T4; T5→T6→T7→T8; T9 T8 sonrası; T10 bağımsız; T11/T12 en
   son. P1'de tanımlanmamış hiçbir interface tüketilmiyor.
