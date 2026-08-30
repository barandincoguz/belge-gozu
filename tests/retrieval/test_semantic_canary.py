"""Gerçek-model semantik canary regression kilitleri (P0 Task 10, Step 4).

Tamamı `-m slow`: gerçek `ColSmolEncoder` (MPS/CUDA/CPU) ve gerçek üretim
indeksini yükler; `pytest -m "not slow"` bu dosyaya hiç değmez.

Beş kilit:
  * G0.1 — canary'deki her gold sayfa üretim indeksinde var (korpus kapsamı).
  * G0.8 — kısa sorgunun gold'u top-5'te (P0'ın ana davranış düzelmesi; bu
    kırılırsa P0 sessizce regresse olmuş demektir).
  * rank cırcırı — uzun sorgunun tam-korpus sırası yalnız SIKILAŞTIRILABİLİR
    (düşürülebilir), asla sessizce gevşetilemez (yükseltilemez). Cırcır
    PIPELINE'a göre anahtarlı (`canary_expectations.json`): sırayı hangi
    kanalın kurduğu sonucu tamamen değiştirir.
  * yazım-değişmezlik — aynı sorunun AKSANSIZ yazımı üretim indeksinde AYNI
    sırayı vermeli (exp12'nin ürün vaadi; journal #11-#12).
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
# Aynı uzun sorunun AKSANSIZ yazımı — exp12 kilidi. İDDİANIN KAPSAMI dar ve
# kasıtlı: bu, Q_LONG'un yalnız AKSANLARI ASCII'ye katlanmış hâlidir (journal
# #11'in ölçüm yöntemi de budur). Noktalama aynen korunur — apostrofu düşürmek
# ("Kanunu'na" -> "Kanununa") ayrı bir yazım değişikliğidir, tokenizasyonu
# gerçekten değiştirir ("na" token'ı düşer) ve katlamanın verdiği garantinin
# kapsamında DEĞİLDİR.
Q_LONG_PLAIN = "Turk Medeni Kanunu'na gore yerlesim yeri nasil tanimlanir?"
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
    retriever, _ = build_retriever(s, encoder)
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

    Üretim yolunda (bugün: hibrit — BM25 metin kanalı + doküman-adı
    yönlendirmesi + ASCII aksan katlaması, `data/index-traincompat-int8` +
    page_texts.parquet) kısa sorgu gold'u top-5 içinde döndürmeli. Ölçüm
    (2026-08-30, katlama SONRASI yeniden ölçüldü): gold rank 1, top-1 skoru
    10.71 (BM25 ölçeği — P0'daki 0.7450 normalize MaxSim ile aynı şey DEĞİL);
    katlama bu sorguda sırayı da skoru da değiştirmedi. P0 ölçümü aynı sorguda
    rank 4'tü.

    Metin kanalı DETERMİNİSTİK (model yok): bu test kırılırsa reçete ya da
    metin artefaktı sessizce değişmiş demektir.
    """
    hits = prod_retriever.search(Q_SHORT, k=5)
    ranked = [(h.page_id, round(h.score, 2)) for h in hits]
    assert GOLD in [pid for pid, _ in ranked], (
        f"'{Q_SHORT}' için gold {GOLD} top-5'te değil (G0.8 / P0 regresyonu!). "
        f"alınan hit'ler (page_id, score): {ranked}"
    )


def test_long_query_rank_ratchet(prod_retriever):
    """Uzun sorgunun tam-korpus gold sırası: yalnız SIKILAŞTIRILABİLİR cırcır.

    Ölçüm (2026-08-29, `data/index-traincompat-int8`): hibrit yolda rank
    2/4222 — P0'ın exhaustive yolunda 664, ondan önce 1-bit'te 1221'di.
    ASCII katlaması (exp12) sonrası yeniden ölçüldü: HÂLÂ 2/4222, cırcır
    değişmedi.
    `canary_expectations.json`'daki eşik yalnızca bilinçli, ölçülmüş bir
    iyileşmeyle DÜŞÜRÜLEBİLİR; asla sessizce YÜKSELTİLMEMELİDİR.

    Cırcır PIPELINE'a göre anahtarlanmıştır: sırayı hangi kanalın kurduğu
    (BM25 vs görsel MaxSim) sonucu tamamen değiştirir, bu yüzden bir kolun
    eşiğini diğerine uygulamak ölçülmemiş bir iddia olurdu. Görsel kolda
    ayrıca TEMSİL de (int8 vs 1-bit) sırayı değiştirir; o kontrol korundu.
    """
    s = get_settings()
    block = _expectations().get(s.retrieval_pipeline)
    if block is None:
        pytest.skip(
            f"cırcır bu pipeline'da ölçülmemiş: {s.retrieval_pipeline} "
            "(tests/retrieval/canary_expectations.json)"
        )
    # Bloğun kendi `pipeline` künyesi anahtarıyla tutarlı olmalı (review L9:
    # aksi halde alan "kontrol ediliyor" izlenimi verip hiç okunmuyordu).
    assert block["pipeline"] == s.retrieval_pipeline, (
        f"canary_expectations.json anahtarı ({s.retrieval_pipeline}) ile bloğun "
        f"künyesi ({block['pipeline']}) çelişiyor — cırcır yanlış kola uygulanır"
    )
    if s.retrieval_pipeline == "hybrid":
        # Metin kanalı tam sıralamayı kendisi verir (model gerektirmez).
        ranking = prod_retriever.rank_all(Q_LONG)
        rank = ranking.index(GOLD) + 1 if GOLD in ranking else len(ranking) + 1
    else:
        manifest = prod_retriever.index.manifest
        measured_on = block["quantization"]
        assert manifest is not None and manifest.quantization == measured_on, (
            f"cırcır başka temsilde ölçülmüş: beklenen quantization={measured_on}, "
            f"yüklü={manifest.quantization if manifest else None}. Sıra kuantizasyona "
            "bağlıdır (int8 664 vs 1-bit 1221); eşiği başka bir temsile uygulamak "
            "ölçülmemiş bir iddiadır. Temsil bilinçli değiştiyse cırcır yeniden "
            "ölçülüp tests/retrieval/canary_expectations.json güncellenmelidir."
        )
        q_emb = prod_retriever.encoder.encode_query(Q_LONG)
        rank = rank_of(prod_retriever.score_all(q_emb), prod_retriever.index.page_ids, GOLD)
    max_allowed = block["long_query_gold_rank_max"]
    assert rank <= max_allowed, (
        f"uzun sorgu için gold {GOLD} tam-korpus sırası {rank} > cırcır {max_allowed} "
        f"(pipeline={s.retrieval_pipeline}). Bu cırcır yalnızca BİLİNÇLİ bir commit'le "
        "(tests/retrieval/canary_expectations.json) DÜŞÜRÜLEBİLİR; asla sessizce "
        "YÜKSELTİLMEMELİDİR."
    )


def test_accentless_query_ranks_identically(prod_retriever):
    """exp12'nin ÜRÜN VAADİ üretim indeksinde kilitli: yazım-değişmezlik.

    Aksansız yazmak yaygın Türkçe klavye davranışıdır. Katlama ÖNCESİ ölçüm
    (journal #11): sorgular ASCII'ye katlandığında canary R@5 0.8372 -> 0.5814
    çöküyordu. Katlama SONRASI (exp12) iki koşulda da 0.8605 (37/43).

    Burada tam korpus sırası iki yazım için birebir karşılaştırılır — metin
    kanalı deterministik olduğu için bu bir eşitlik iddiasıdır, istatistik
    değil. Ölçüm (2026-08-30): iki yazım da AYNI token listesini
    (`turk meden kanun na yerle yeri tanim`) ve aynı sıralamayı üretir, gold
    k4721:4 iki koşulda da 2. sırada.
    """
    if getattr(prod_retriever, "rank_all", None) is None:
        pytest.skip("yazım-değişmezlik kilidi metin kanalı gerektirir (hibrit pipeline)")
    accented = prod_retriever.rank_all(Q_LONG)
    plain = prod_retriever.rank_all(Q_LONG_PLAIN)
    assert accented[:20] == plain[:20], (
        "aksanlı ve aksansız yazım FARKLI sıralama verdi — ASCII katlaması "
        "(retrieval/text.py `tokenize`) bozulmuş olabilir. İlk 5: "
        f"aksanlı={accented[:5]} aksansız={plain[:5]}"
    )
    assert accented.index(GOLD) == plain.index(GOLD)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ÖLÇÜLDÜ (2026-08-30, ASCII katlaması SONRASI yeniden; "
        "data/index-traincompat-int8 + page_texts.parquet, hibrit/BM25 ölçeği): "
        "eşik 10.6 AYIRMIYOR — korpus-dışı c003/c004/c005 "
        "top-1 skorları 23.52/12.96/17.86, yani ÜÇÜ DE eşiğin ÜSTÜNDE (c007 15.54 "
        "de üstünde; yalnız anlamsız c006 4.23 altında kalıyor: 5'te 4'ü geçiyor — "
        "P0'daki 4/5 ile aynı çalışma noktası). Tüm canary'de cevaplanabilir n=43 "
        "servis edilen top-1 bandı (min 10.5265 / medyan 23.78 / maks 66.68) ile "
        "cevaplanamazların bandı iç içe "
        "geçmiş durumda: hiçbir tek eşik bu ikisini ayırmıyor. 10.6, int8@0.58'in "
        "(o da binary@60.0'ın) MEKANİK ölçek taşımasıdır — çalışma noktasını veren "
        "bant katlama sonrası (10.5265, 10.7115] — kalibrasyon DEĞİL. "
        "Kalibrasyon P2'nin işi (spec). "
        "strict=True: eşik gerçekten kalibre edilip bu iddia tuttuğunda test KIRMIZI "
        "olur ve xfail'in kaldırılmasını zorlar — abstain sözü sessizce ne "
        "bozulabilir ne de düzelmiş sayılabilir."
    ),
)
def test_out_of_corpus_canary_scores_below_threshold(prod_retriever):
    """Abstain sözü BUGÜNKÜ pipeline'a karşı kilitlenir (final review IMPORTANT-6).

    Burada canary'nin `korpus-disi` (cevaplanamaz, konusu korpusta olmayan)
    satırları üretim yolundan geçirilir ve top-1 skorunun eşiğin ALTINDA
    kaldığı doğrulanır: yani bu sorular LLM'e hiç gitmeden abstain'e düşmeli.

    İddia BİLEREK gevşetilmedi: ölçüm bugün de tutmuyor (bkz. xfail reason),
    bu yüzden `xfail(strict=True)` ile MEVCUT GERÇEK kilitlenir. P1'in hibrit
    geçişi bunu DEĞİŞTİRMEDİ ve değiştirmesi de beklenmiyordu (exp12'nin aksan
    katlaması da değiştirmedi — yeniden ölçüldü): metin kanalı
    SIRALAMAYI düzeltiyor (R@5 0.2326 -> 0.8605), skoru bir güven ölçüsüne
    ÇEVİRMİYOR — korpus-dışı bir soru da korpusta geçen kelimeler içerdiği
    sürece yüksek BM25 alır. Eşiği yükseltmek de çözüm değil: cevaplanabilir
    dağılımın alt ucu 10.53, yani yükseltmek gerçek soruları abstain'e
    düşürür. Gerçek düzeltme kalibrasyondur (P2).
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
