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

import pytest

from belge_gozu.bench.dataset import load_bench
from belge_gozu.bench.oracle import rank_of
from belge_gozu.config import get_settings

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

    `app/main.py::build_retriever`'ı ÇAĞIRIR — create_app ile bire bir aynı
    fonksiyon. Bu fixture eskiden aynı mantığın (indeks yükleme, format
    çözme, pipeline dallanması) bir KOPYASINI taşıyordu ve o kopya sessizce
    sapabiliyordu: varsayılan indeks int8'e döndüğünde kopyadaki sabit
    `PackedIndex.load` tüm slow testleri FileNotFoundError'a düşürürdü.
    """
    s = get_settings()
    if not s.index_dir.exists():
        pytest.skip(f"index_dir yok: {s.index_dir} (fresh clone / veri indirilmemiş)")

    from belge_gozu.app.main import build_retriever, resolve_formats
    from belge_gozu.index.encode import ColSmolEncoder

    resolved_query_format, resolved_doc_prompt = resolve_formats(s)
    encoder = ColSmolEncoder(
        s.retriever_model,
        s.device,
        query_format=resolved_query_format,
        visual_prompt_override=resolved_doc_prompt,
    )
    retriever, _ = build_retriever(
        s,
        encoder,
        model_name=s.retriever_model,
        model_revision=getattr(encoder, "model_revision", None),
    )
    return retriever


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
    prompt, `data/index-traincompat-int8`) kısa sorgu gold'u top-5 içinde
    döndürmeli. Ölçüm (2026-08-29, int8 indeks, MPS): gold rank 4, top-1
    skoru 0.7450 (normalize [-1,1] ölçek — 1-bit'teki 73.17 ile aynı şey
    DEĞİL, farklı temsil ve farklı ölçek). Bu test kırılırsa P0'ın
    retrieval düzeltmesi sessizce regresse olmuş demektir.
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
    top-N'de) olması BEKLENMİYOR — bugünkü ölçüm (2026-08-29,
    `data/index-traincompat-int8`, exhaustive): rank 664/4222 (1-bit'te
    1221'di). `canary_expectations.json`'daki eşik yalnızca bilinçli,
    ölçülmüş bir iyileşmeyle DÜŞÜRÜLEBİLİR; asla sessizce
    YÜKSELTİLMEMELİDİR.

    Cırcır TEMSİLE göre anahtarlanmıştır: sıra kuantizasyona bağlı
    (int8 664 vs 1-bit 1221), bu yüzden başka bir temsile karşı
    uygulanması sessizce yanlış bir iddia olurdu.
    """
    if not hasattr(prod_retriever, "score_all"):
        pytest.skip(
            "rank cırcırı tam-korpus score_all gerektirir; "
            f"{type(prod_retriever).__name__} bunu sağlamıyor "
            "(retrieval_pipeline != 'exhaustive')"
        )
    expectations = _expectations()
    manifest = prod_retriever.index.manifest
    measured_on = expectations["quantization"]
    assert manifest is not None and manifest.quantization == measured_on, (
        f"cırcır başka temsilde ölçülmüş: beklenen quantization={measured_on}, "
        f"yüklü={manifest.quantization if manifest else None}. Sıra kuantizasyona "
        "bağlıdır (int8 664 vs 1-bit 1221); eşiği başka bir temsile uygulamak "
        "ölçülmemiş bir iddiadır. Temsil bilinçli değiştiyse cırcır yeniden "
        "ölçülüp tests/retrieval/canary_expectations.json güncellenmelidir."
    )
    q_emb = prod_retriever.encoder.encode_query(Q_LONG)
    scores = prod_retriever.score_all(q_emb)
    rank = rank_of(scores, prod_retriever.index.page_ids, GOLD)
    max_allowed = expectations["long_query_gold_rank_max"]
    assert rank <= max_allowed, (
        f"uzun sorgu için gold {GOLD} tam-korpus sırası {rank} > cırcır {max_allowed}. "
        "Bu cırcır yalnızca BİLİNÇLİ bir commit'le (tests/retrieval/canary_expectations.json) "
        "DÜŞÜRÜLEBİLİR; asla sessizce YÜKSELTİLMEMELİDİR."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ÖLÇÜLDÜ (2026-08-29, data/index-traincompat-int8, exhaustive, MPS): "
        "eşik 0.58 AYIRMIYOR — korpus-dışı c003/c004/c005 top-1 skorları "
        "0.6550/0.6866/0.6516 ve anlamsız-ood c006 0.6678, yani dördü de eşiğin "
        "ÜSTÜNDE (yalnız c007 0.5679 altında kalıyor: 5'te 4'ü geçiyor). Tüm "
        "canary'de cevaplanabilir n=43 (min 0.5767 / medyan 0.6250 / maks 0.7450) "
        "ile cevaplanamaz n=5 (min 0.5679 / medyan 0.6550 / maks 0.6866) "
        "dağılımları iç içe geçmiş durumda: hiçbir tek eşik bu ikisini ayırmıyor. "
        "0.58 eski binary 60.0'ın MEKANİK ölçek taşımasıdır (aynı çalışma "
        "noktası: 42/43 + 4/5), kalibrasyon DEĞİL — artefakt: "
        "data/bench/results/int8-threshold-transfer.json. Kalibrasyon P2'nin işi "
        "(spec). strict=True: eşik gerçekten kalibre edilip bu iddia tuttuğunda "
        "test KIRMIZI olur ve xfail'in kaldırılmasını zorlar — abstain sözü "
        "sessizce ne bozulabilir ne de düzelmiş sayılabilir."
    ),
)
def test_out_of_corpus_canary_scores_below_threshold(prod_retriever):
    """Abstain sözü BUGÜNKÜ pipeline'a karşı kilitlenir (final review IMPORTANT-6).

    Burada canary'nin `korpus-disi` (cevaplanamaz, konusu korpusta olmayan)
    satırları üretim yolundan geçirilir ve top-1 skorunun eşiğin ALTINDA
    kaldığı doğrulanır: yani bu sorular LLM'e hiç gitmeden abstain'e düşmeli.

    İddia BİLEREK gevşetilmedi: ölçüm bugün de tutmuyor (bkz. xfail reason),
    bu yüzden `xfail(strict=True)` ile MEVCUT GERÇEK kilitlenir. T14'ün skor
    normalizasyonu bunu DEĞİŞTİRMEDİ ve değiştirmesi de beklenmiyordu: ölçek
    taşıması monotonik bir dönüşümdür, iki dağılımın ÖRTÜŞMESİNİ kaldırmaz —
    0.58 eşiği eski 60.0'ın aynı çalışma noktasıdır (42/43 cevaplanabilir +
    4/5 cevaplanamaz geçer). Eşiği yükseltmek de çözüm değil: cevaplanabilir
    dağılım aynı bantta (medyan 0.6250), yani yükseltmek gerçek soruları
    abstain'e düşürür. Gerçek düzeltme kalibrasyondur (P2).
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
