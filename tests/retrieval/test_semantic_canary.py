"""Gerçek-model semantik canary regression kilitleri (P0 Task 10, Step 4).

Tamamı `-m slow`: gerçek `ColSmolEncoder` (MPS/CUDA/CPU) ve gerçek üretim
indeksini yükler; `pytest -m "not slow"` bu dosyaya hiç değmez.

Dört kilit:
  * G0.1 — canary'deki her gold sayfa üretim indeksinde var (korpus kapsamı).
  * G0.8 — kısa sorgunun gold'u top-5'te (P0'ın ana davranış düzelmesi; bu
    kırılırsa P0 sessizce regresse olmuş demektir).
  * rank cırcırı — uzun sorgunun tam-korpus sırası yalnız SIKILAŞTIRILABİLİR
    (düşürülebilir), asla sessizce gevşetilemez (yükseltilemez).
  * abstain kilidi — korpus-dışı sorularda top-1 skoru yapılandırılmış eşiğin
    ALTINDA kalır (yoksa "halüsinasyon freni" iddiası ölçülmemiş bir yorum).

Dosya yolları CWD'ye değil repo köküne göre çözülür ve hiçbir veri dosyası
import/collection anında OKUNMAZ (`-m "not slow"` koşumları bu dosyaya
dokunduğunda veri gerektirmesin diye).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from belge_gozu.bench.dataset import load_bench
from belge_gozu.bench.oracle import rank_of
from belge_gozu.config import get_settings
from belge_gozu.index.manifest import (
    DOC_PROMPTS,
    QUERY_FORMATS,
    DocPromptChoice,
    QueryFormatChoice,
)
from belge_gozu.index.store import PackedIndex
from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever, TwoStageRetriever

pytestmark = pytest.mark.slow

Q_SHORT = "Yerleşim yeri nedir?"
Q_LONG = "Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?"
GOLD = "k4721:4"
REPO_ROOT = Path(__file__).resolve().parents[2]
CANARY_PATH = REPO_ROOT / "data" / "bench" / "canary_v1.jsonl"
EXPECT_PATH = Path(__file__).resolve().parent / "canary_expectations.json"


def _expectations() -> dict:
    """Cırcır eşikleri — import anında değil, kullanıldığı testte okunur."""
    return json.loads(EXPECT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prod_retriever():
    """Üretim retriever'ını `get_settings()`'ten kurar.

    `app/main.py::create_app`'ın encoder/retriever inşa mantığını birebir
    yansıtır (aynı `QUERY_FORMATS`/`DOC_PROMPTS` sözlükleri, aynı
    `retrieval_pipeline` dallanması) ki bu test üretim yapılandırmasından
    sessizce sapamasın.
    """
    s = get_settings()
    if not s.index_dir.exists():
        pytest.skip(f"index_dir yok: {s.index_dir} (fresh clone / veri indirilmemiş)")

    from belge_gozu.index.encode import ColSmolEncoder

    index = PackedIndex.load(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    resolved_query_format = QUERY_FORMATS[QueryFormatChoice(s.query_format_id)]
    resolved_doc_prompt = DOC_PROMPTS[DocPromptChoice(s.doc_prompt_id)]
    encoder = ColSmolEncoder(
        s.retriever_model,
        s.device,
        query_format=resolved_query_format,
        visual_prompt_override=resolved_doc_prompt,
    )
    if s.retrieval_pipeline == "exhaustive":
        return ExhaustiveBinaryRetriever(index, meta, encoder)
    return TwoStageRetriever(index, meta, encoder)


def test_canary_gold_pages_covered(prod_retriever):
    """G0.1: canary'deki her gold sayfa kimliği üretim indeksinde var olmalı.

    İnsan doğrulaması hâlâ bekleniyor (Step 2), bu yüzden `only_verified=False`
    ile TÜM satırlar (draft dahil) okunur — kapsam kilidi doğrulama
    tamamlanmadan da geçerli olsun diye.
    """
    if not CANARY_PATH.exists():
        pytest.skip(f"canary seti yok: {CANARY_PATH}")
    known = set(prod_retriever.index.page_ids)
    missing = [
        (q.question_id, g)
        for q in load_bench(CANARY_PATH, only_verified=False)
        for g in q.gold_page_ids
        if g not in known
    ]
    assert not missing, f"korpusta olmayan gold sayfa(lar) (G0.1 ihlali): {missing}"


def test_short_query_gold_in_top5(prod_retriever):
    """G0.8: P0'ın ana davranış düzelmesinin regresyon kilidi.

    Üretim yolunda (bugün: exhaustive + train-compat-v1 + train-compat doc
    prompt, `data/index-traincompat-1bit`) kısa sorgu gold'u top-5 içinde
    döndürmeli. Ölçüm (2026-08-26, canlı `create_app`+TestClient `/search`):
    rank 4, score 73.17. Bu test kırılırsa P0'ın retrieval düzeltmesi sessizce
    regresse olmuş demektir.
    """
    hits = prod_retriever.search(Q_SHORT, k=5)
    ranked = [(h.page_id, round(h.score, 2)) for h in hits]
    assert GOLD in [pid for pid, _ in ranked], (
        f"'{Q_SHORT}' için gold {GOLD} top-5'te değil (G0.8 / P0 regresyonu!). "
        f"alınan hit'ler (page_id, score): {ranked}"
    )


def test_long_query_rank_ratchet(prod_retriever):
    """Uzun sorgunun tam-korpus gold sırası: yalnız SIKILAŞTIRILABİLİR cırcır.

    P1'in hibrit retrieval'ı bu sırayı iyileştirene kadar top-5'te (hatta
    top-N'de) olması BEKLENMİYOR — bugünkü ölçüm (2026-08-26 oracle koşumu,
    `data/index-traincompat-1bit`, exhaustive): rank 1221/4222.
    `canary_expectations.json`'daki eşik yalnızca bilinçli, ölçülmüş bir
    iyileşmeyle DÜŞÜRÜLEBİLİR; asla sessizce YÜKSELTİLMEMELİDİR.
    """
    if not hasattr(prod_retriever, "score_all"):
        pytest.skip(
            "rank cırcırı tam-korpus score_all gerektirir; "
            f"{type(prod_retriever).__name__} bunu sağlamıyor "
            "(retrieval_pipeline != 'exhaustive')"
        )
    q_emb = prod_retriever.encoder.encode_query(Q_LONG)
    scores = prod_retriever.score_all(q_emb)
    rank = rank_of(scores, prod_retriever.index.page_ids, GOLD)
    max_allowed = _expectations()["long_query_gold_rank_max"]
    assert rank <= max_allowed, (
        f"uzun sorgu için gold {GOLD} tam-korpus sırası {rank} > cırcır {max_allowed}. "
        "Bu cırcır yalnızca BİLİNÇLİ bir commit'le (tests/retrieval/canary_expectations.json) "
        "DÜŞÜRÜLEBİLİR; asla sessizce YÜKSELTİLMEMELİDİR."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ÖLÇÜLDÜ (2026-08-27, data/index-traincompat-1bit, exhaustive): eşik 60.0 "
        "T11 formatı altında ARTIK AYIRMIYOR — korpus-dışı c003/c004/c005 top-1 "
        "skorları 66.28/71.95/67.88, yani hepsi eşiğin ÜSTÜNDE. Tüm canary'de "
        "cevaplanabilir n=43 (min 59.85 / medyan 63.40 / maks 78.50) ile "
        "cevaplanamaz n=5 (min 59.65 / medyan 67.88 / maks 71.95) dağılımları "
        "iç içe geçmiş durumda: hiçbir tek eşik bu ikisini ayırmıyor. Kalibrasyon "
        "P2'nin işi (spec). strict=True: eşik gerçekten kalibre edilip bu iddia "
        "tuttuğunda test KIRMIZI olur ve xfail'in kaldırılmasını zorlar — "
        "abstain sözü sessizce ne bozulabilir ne de düzelmiş sayılabilir."
    ),
)
def test_out_of_corpus_canary_scores_below_threshold(prod_retriever):
    """Abstain sözü BUGÜNKÜ pipeline'a karşı kilitlenir (final review IMPORTANT-6).

    `Settings.min_score_threshold=60.0`'ın gerekçe yorumundaki skorlar
    (70.6 / 52.4) T11 format değişikliğinden ÖNCE, eski indeks+formatla
    ölçüldü — yani eşiğin hâlâ ayırdığına dair hiçbir canlı kanıt yoktu.
    Burada canary'nin `korpus-disi` (cevaplanamaz, konusu korpusta olmayan)
    satırları üretim yolundan geçirilir ve top-1 skorunun eşiğin ALTINDA
    kaldığı doğrulanır: yani bu sorular LLM'e hiç gitmeden abstain'e düşmeli.

    İddia BİLEREK gevşetilmedi: ölçüm bugün tutmuyor (bkz. xfail reason), bu
    yüzden `xfail(strict=True)` ile MEVCUT GERÇEK kilitlenir. Eşiği yükseltmek
    çözüm değil — cevaplanabilir dağılım da aynı bantta (63.40 medyan), yani
    eşik yükseltmek gerçek soruları abstain'e düşürür. Gerçek düzeltme
    kalibrasyon + skor normalizasyonudur (P2).
    """
    if not CANARY_PATH.exists():
        pytest.skip(f"canary seti yok: {CANARY_PATH}")
    threshold = get_settings().min_score_threshold
    # İnsan doğrulaması sürüyor -> taslak satırlar da dahil (only_verified=False).
    ood = [
        q
        for q in load_bench(CANARY_PATH, only_verified=False)
        if q.slice == "korpus-disi" and not q.answerable
    ]
    assert ood, "canary'de 'korpus-disi' satırı yok — abstain kilidi anlamsızlaşır"

    over = []
    for q in ood:
        hits = prod_retriever.search(q.question, k=1)
        top = hits[0].score if hits else float("-inf")
        if top >= threshold:
            over.append((q.question_id, round(top, 2)))
    assert not over, (
        f"korpus-dışı soru(lar) eşiği ({threshold}) geçti -> abstain yerine LLM çağrılırdı: "
        f"{over}. Eşik T11 formatı altında yeniden kalibre edilmeli (P2); bu testi gevşetmeyin."
    )
