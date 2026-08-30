"""Metin kanalı: Türkçe-uyarlı BM25 + doküman-adı pencere-içi yönlendirmesi.

autoresearch exp12 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md
(round 2) + research/journal.md #11-#13 (round 3). Bu modül `research/retrieve.py`'nin
ÜRETİM PORTUDUR: sabitler (STOPWORDS, F5=5, k1=1.5, b=0.75, WINDOW=50, _FOLD,
_GENERIC, _TITLE_LINE) ve tokenleştirme mantığı birebir taşınmıştır. Reçete
ölçülmüş bir bütündür — tek tek parçalar "iyileştirilirse" ölçüm geçersiz olur:

| deney | R@5 | karar |
|---|---|---|
| taban: yalnız görsel (int8) | 0.2326 | — |
| + BM25 (PDF metin katmanı) | 0.6744 | KEPT |
| eşit-ağırlık RRF(görsel, BM25) | 0.3953 | DISCARDED (zayıf kanal zarar veriyor) |
| + F5 ön-ek kırpması | 0.7674 | KEPT |
| + bigram shingle | 0.6279 | DISCARDED |
| + sabit işlev-kelime listesi | 0.7674 | KEPT (R@20 0.884->0.907, MRR+) |
| + mutlak doküman-adı bölümleme | 0.7907 | DISCARDED (R@20 gerilemesi, veto) |
| + pencere-içi (top-20) yönlendirme | 0.8140 | KEPT |
| pencere 20 -> 50 (round 2, exp8) | 0.8372 | KEPT (R@20 0.907 -> 0.9302) |
| + ASCII aksan katlama (round 3, exp12) | **0.8605** | KEPT (YAZIM-DEĞİŞMEZ) |
| çift-biçim yayım (round 3, exp13) | 0.8372 | DISCARDED (iki guardrail düştü) |
| + qtf≤2 sorgu doygunluğu (exp14, Y1) | **0.8605** | KEPT (birebir aynı; saldırı 667.5→16.7) |

YAZIM-DEĞİŞMEZLİK (exp12, journal #11-#12) reçetenin en önemli GERÇEK-KULLANICI
özelliğidir: aksansız yazmak yaygın Türkçe klavye davranışıdır ve katlama
ÖNCESİNDE sorguları ASCII'ye katlamak R@5'i 0.8372 -> 0.5814'e düşürüyordu.
Katlama iki tarafa da (indeks + sorgu) uygulanınca sistem İKİ KOŞULDA DA
0.8605 (37/43) veriyor. Bedeli ölçüldü ve bilinçle kabul edildi (round-3 R26
istisnası): aksanlı MRR 0.655 -> 0.632, R@1 -2 — düşen sorular SERVİS EDİLEN
top-5'in İÇİNDE kalıyor.

Round 2 ayrıca üç füzyon biçimini daha ölçüp REDDETTİ (küresel eşit-RRF 0.395,
mutlak bölümleme guardrail vetosu, pencere-içi RRF 0.535) — bu korpusta hayatta
kalan tek birleşim sözcüksel-birincil + kural yönlendirmesi (journal #10).

Görsel kanal F5'ten sonra top-5'e BENZERSİZ hiçbir soru katmıyor (bulgu 3),
ama serve'de telemetri/gösterim ve P2 kalibrasyon verisi için KORUNUYOR —
bkz. `retrieval/hybrid.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)

# Türkçe ön-ek kırpma uzunluğu (Can vd., Turkish IR): eklemeli dilde gövde
# yaklaşık ilk 5 harftir. Ölçüm: R@5 0.6744 -> 0.7674 (+9.3 puan).
F5 = 5

# Tokenleştirmenin en kısa kabul edilen kelime uzunluğu (`len(t) > 1`).
# Modül sabiti olarak duruyor çünkü `recipe_fingerprint()` reçetenin DAVRANIŞ
# TAŞIYAN her sabitini anahtara katmak zorunda; gövdede gömülü bir literal
# sessizce değiştirilebilirdi.
MIN_TOKEN_CHARS = 2

# BM25 doygunluk/uzunluk-normalizasyon parametreleri (`research/retrieve.py`
# portunun ölçülmüş değerleri). `BM25Index.__init__` bunları VARSAYILAN olarak
# alır — modül düzeyinde durmalarının nedeni F5/QTF_CAP ile aynı:
# `recipe_fingerprint()` reçete kimliğini bunlardan türetir.
K1 = 1.5
B = 0.75

# Standart Türkçe işlev kelimeleri (tam-kelime, kırpmadan ÖNCE uygulanır).
# CANARY'YE AYARLI DEĞİL: "zaman"/"iş" gibi içerik taşıyabilecek kelimeler
# bilinçli olarak dışarıda bırakıldı. Ölçüm: R@5 eşit kalır, R@20 0.884->0.907
# ve vitrin sorgusu chip1 8->4 (soru-kalıbı kelimeleri en çok ORTA sıralarda
# karışıklık üretiyor).
STOPWORDS = frozenset(
    "ve veya ile için gibi göre kadar sonra önce bir bu şu o ne nasıl neden "
    "niçin hangi kaç kim mi mı mu mü midir mıdır mudur müdür da de ki en çok "
    "az her ise olan olarak üzere ancak ama fakat yoksa değil nedir sayılı".split()
)

# Yönlendirme penceresi. exp7'de 20'ydi ve R@20 guardrail'iyle YAPISAL olarak
# hizalıydı (pencere kümesi değişmediği için R@20 tanım gereği korunuyordu).
# exp8 (autoresearch round 2, journal #8) bunu 50'ye çıkardı: yapısal garanti
# artık YOK, yerine ÖLÇÜM var — R@20 korunmadı, **yükseldi** (0.907 -> 0.9302),
# R@5 0.8140 -> 0.8372 (+c214), MRR 0.655, vitrin sorguları 2/2. Pencere-İÇİ
# yeniden sıralama sözleşmesi (küme değişmez, pencere sonrası dokunulmaz)
# window değerinden BAĞIMSIZ olarak yapısaldır ve aynen geçerlidir.
WINDOW = 50

# SORGU-TERİM DOYGUNLUK TAVANI (Y1, ölçülmüş karar R30 — 2026-08-30).
#
# Klasik BM25'in üçüncü doygunluk çarpanı `(k3+1)·qtf/(k3+qtf)` bu porta hiç
# gelmemişti: araştırma döngüsünde sorgular canary'den geliyordu ve hiçbirinde
# terim TEKRAR ETMİYORDU, yani ölçülen uzayda qtf her zaman 1'di. Ölçülmeyen
# uzayda ise skor sorgudaki tekrar sayısıyla DOĞRUSAL şişiyordu: 480 karakterlik
# `"ihbar "×80` sorgusu top-1'i 667.50'ye çıkarıyor (eşik 10.6'nın 63 katı),
# freni geçiriyor ve ücretli bir LLM çağrısı tetikliyordu. `MAX_QUERY_CHARS=500`
# bunu KAPATMIYOR, yalnız ölçekliyordu.
#
# Tavan `min(qtf, 2)` biçiminde SERT seçildi (yumuşak k3 eğrisi değil): tek
# parametreli, açıklaması bir satır, ve saldırı yüzeyini sabit bir çarpana
# kilitliyor. 2 (1 değil) çünkü gerçek sorgularda bir terimin iki kez geçmesi
# meşru bir vurgudur ("kira artışı ... artış oranı"), 80 kez geçmesi değildir.
#
# ÖLÇÜM (exp14-qtf-cap2-parity, research/results.jsonl, 2026-08-30): canary
# birebir DEĞİŞMEDİ — R@5 0.8605 (37/43), MRR 0.6320, R@1/R@20/visual_R@5 aynı;
# saldırı sorgusu 667.5 -> 16.7 (tam olarak tek-geçişin 2 katı).
QTF_CAP = 2

# Doküman adından atılan jenerik parçalar (F5-kırpık): bunlar tek başına
# kalırsa neredeyse her sorgu her kanunu "yönlendirir".
#
# DİKKAT — ölçülen biçim BUDUR, "düzeltilmedi": `tokenize` artık katlanmış
# ("turk"/"turki") token üretir, yani buradaki AKSANLI "türk"/"türki" girdileri
# exp12'den sonra hiçbir token'la eşleşmez ve pratikte yalnız "kanun"/"cumhu"
# eleniyor. Sonuç: "Türk ..." ile başlayan kanunların ad kümesinde "turk"
# token'ı KALIR, dolayısıyla o kanunlar ancak sorguda "türk/turk" da geçerse
# yönlendirilir (daha DAR bir yönlendirme). 0.8605 tam olarak bu davranışla
# ölçüldü (journal #12); listeyi katlamak reçeteyi ölçülmemiş bir varyanta
# çevirirdi. Değiştirmek isteyen önce bench'i yeniden koşmalıdır.
_GENERIC = frozenset({"kanun", "türk", "türki", "cumhu"})
# 1. sayfadaki büyük-harfli başlık satırı (elle doküman-adı tablosu YOK ->
# canary'den sızıntı yok; ad korpusun kendisinden türetiliyor).
_TITLE_LINE = re.compile(r"^[A-ZÇĞİÖŞÜÂÎÛ0-9 ()'’.,;:-]{8,}$")

# Başlık adayı KAPISI: satırın doküman adı sayılabilmesi için içermesi gereken
# anahtar kelimeler. Modül sabiti (review M2): gövdeye gömülü bir literal olarak
# durduğunda `recipe_fingerprint()` onu göremiyordu, oysa bu küme `doc_names` ->
# `routed_docs` -> `routed` ÖZELLİĞİ ve `route_window` üzerinden hem `served_top1`i
# hem top-5 etiketini değiştirir. Yani değiştirmek reçeteyi değiştirir.
_TITLE_KEYWORDS = ("KANUN", "ANAYASA")

# Türkçe küçültmenin ÖZEL EŞLEMESİ (Python'un `lower()`ından ayrılan kısım).
# Yine modül sabiti (review M2): bu bir tablo, algoritmik biçim değil — değişirse
# HER sorgu ve HER sayfa yeniden tokenleşir, bu yüzden parmak izine girmek zorunda.
_TR_LOWER_MAP = (("İ", "i"), ("I", "ı"))


def tr_lower(s: str) -> str:
    """Türkçe küçültme: İ->i, I->ı (Python'un `lower()`ı ikisini de 'i' yapar).

    autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
    """
    for src, dst in _TR_LOWER_MAP:
        s = s.replace(src, dst)
    return s.lower()


# ASCII aksan katlama tablosu (exp12). tr_lower'dan SONRA uygulanır: "İ"/"I"
# ayrımı önce Türkçe kurallarıyla çözülür, katlama ondan sonra devreye girer.
# "î"/"â"/"û" de listede — eski mevzuat metninde düzeltme işaretli biçimler
# ("hâkim", "kanunî") sıkça geçer ve kullanıcı bunları hiç yazmaz.
_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def ascii_fold(s: str) -> str:
    """Türkçe aksanları ASCII karşılıklarına katlar (ç->c, ğ->g, ı->i, ...).

    İNDEKS VE SORGU İÇİN AYNI şekilde uygulanır — tek taraflı katlama iki
    tarafı birbirinden uzaklaştırır ve ölçümde ANA hasarı üretir (exp11:
    R@5 0.8372 -> 0.5814).
    """
    return s.translate(_FOLD)


# Stopword eleme KATLANMIŞ uzayda yapılır: aksansız yazan kullanıcının "gore",
# "nicin", "kac" gibi işlev kelimeleri de elenmeli, yoksa aynı sorgu iki
# yazımda iki farklı token kümesi verirdi (yazım-değişmezliğin ikinci yarısı).
_STOP_FOLDED = frozenset(ascii_fold(w) for w in STOPWORDS)


def tokenize(s: str) -> list[str]:
    """`\\w+` -> tr_lower -> >=2 harf -> ASCII katlama -> stopword eleme -> F5 kırpma.

    Sıra ölçülmüş reçetenin parçasıdır (exp12, journal #12):
      * stopword eleme TAM KELİME üzerinde ve F5 kırpmasından ÖNCE
        (kırpma sonrası "göre" ile "görev" aynı token'a düşerdi);
      * ama KATLAMADAN SONRA — eleme katlanmış uzayda (`_STOP_FOLDED`) yapılır,
        böylece "göre" ve "gore" aynı kararı alır;
      * F5 kırpması en sonda: katlama karakter sayısını değiştirmez, yani
        kırpma sınırı iki yazımda da aynı yere düşer.

    Sonuç: sistem YAZIM-DEĞİŞMEZ — aksanlı ve aksansız yazılan aynı sorgu aynı
    token listesini üretir, R@5 iki koşulda da 0.8605 (37/43).

    autoresearch exp12 reçetesi; ölçüm: research/journal.md #11-#13.
    """
    words = [ascii_fold(t) for t in _WORD.findall(tr_lower(s)) if len(t) >= MIN_TOKEN_CHARS]
    return [t[:F5] for t in words if t not in _STOP_FOLDED]


# Reçetenin ALGORİTMİK BİÇİMİ (sabit değerleri değil) değiştiğinde ELLE artırılır.
#
# `recipe_fingerprint()` aşağıdaki sabitlerin DEĞERLERİNİ otomatik yakalar
# (F5'i 5'ten 6'ya çekmek anahtarı değiştirir), ama sabitlerin nasıl
# KULLANILDIĞINI yakalayamaz: `tokenize` içindeki adım sırası (katlama ->
# eleme -> kırpma), `route_window`ın pencere-içi sözleşmesi, `scores`ın
# doygunluk ifadesi. Bunlardan biri değişirse ölçülen reçete başka bir
# reçetedir ve bu sayı ELLE artırılmalıdır — aksi halde eski bir kalibratör
# yeni bir boru hattına sessizce takılır (P2 denetimi, T6 versiyonlama bulgusu).
RECIPE_VERSION = 1


def recipe_fingerprint() -> str:
    """Metin reçetesinin sabitleri üzerinden sha256'nın ilk 12 hex hanesi.

    P2 kalibrasyon artefaktının versiyon anahtarının ÜÇÜNCÜ bileşeni
    (`<index_revision>__<pipeline>__<recipe_fp>`). Gerekçe (P2 gerçeklik
    denetimi, T6): `index_revision` yalnız korpus checksum'ını, sorgu formatını
    ve kuantizasyonu kodlar — getirim REÇETESİNİ (BM25 parametreleri, F5
    kırpması, stopword listesi, pencere, aksan katlaması) hiç görmez. Oysa
    eşiğin bağlı olduğu eksen tam olarak budur: `config.py`, ASCII aksan
    katlamasından SONRA eşik bandının yeniden ölçülmek zorunda kaldığını
    yazıyor. Bu bileşen olmadan reçete değişince kalibratör geçersizleşmez —
    sessizce YANLIŞ kalır.

    KAPSAM: modül düzeyindeki reçete sabitleri + `RECIPE_VERSION`. Kapsamadığı
    iki şey açıkça yazılır: (a) algoritmik biçim — `RECIPE_VERSION` elle
    artırılır; (b) `BM25Index(k1=..., b=...)` ile VARSAYILANDAN sapan bir
    örnek — üretim yolu (`retrieval/hybrid.load_text_channel`) varsayılanları
    kullanır, ablasyon koşumları kendi anahtarlarını taşımalıdır.
    """
    recipe = {
        "recipe_version": RECIPE_VERSION,
        "word_re": _WORD.pattern,
        "min_token_chars": MIN_TOKEN_CHARS,
        "f5": F5,
        # review M2: `tr_lower`ın özel eşlemesi ve başlık kapısı da BİRER DEĞERDİR
        # (algoritmik biçim değil) — ikisi de tokenleştirmeyi/yönlendirmeyi
        # değiştirir, dolayısıyla anahtarı değiştirmeleri gerekir.
        "tr_lower_map": ["".join(pair) for pair in _TR_LOWER_MAP],
        "title_keywords": sorted(_TITLE_KEYWORDS),
        # `str.maketrans` ORDİNAL -> ORDİNAL sözlüğü üretir; JSON'a yazılabilir
        # ve okunabilir olsun diye karakter çiftlerine geri çevriliyor.
        "fold": "".join(chr(k) + chr(v) for k, v in sorted(_FOLD.items())),
        "stopwords": sorted(STOPWORDS),
        "k1": K1,
        "b": B,
        "qtf_cap": QTF_CAP,
        "window": WINDOW,
        "generic": sorted(_GENERIC),
        "title_line_re": _TITLE_LINE.pattern,
    }
    blob = json.dumps(recipe, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class BM25Index:
    """Sayfa metinleri üzerinde BM25 (k1=1.5, b=0.75) — `page_ids` ile hizalı.

    `research/retrieve.py`'deki `BM25` sınıfının portu (adı üretimde
    BM25Index); skorlama ifadesi birebir aynıdır.

    autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
    """

    def __init__(self, page_ids: list[str], texts: list[str], k1: float = K1, b: float = B) -> None:
        if len(page_ids) != len(texts):
            raise ValueError(f"page_ids ({len(page_ids)}) ve texts ({len(texts)}) eşleşmiyor")
        if not page_ids:
            raise ValueError("boş korpus: en az bir sayfa metni gerekli")
        self.page_ids = list(page_ids)
        self.k1, self.b = k1, b
        docs_tokens = [tokenize(t) for t in texts]
        self.doc_freqs = [Counter(toks) for toks in docs_tokens]
        self.doc_lens = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) or 1.0
        df: Counter[str] = Counter()
        for toks in docs_tokens:
            df.update(set(toks))
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        """(n_pages,) float32 BM25 skorları — `page_ids` sırasıyla hizalı.

        ÖLÇEK: kalibre edilmemiş, ÜST SINIRSIZ BM25 birimi. Ölçülen bant
        (canary, 4222 sayfa): cevaplanabilir top-1'ler min 10.53 / medyan
        26.05 / maks 69.30. Görsel kanalın normalize [-1,1] skorlarıyla AYNI
        ŞEY DEĞİLDİR — eşik bu ölçekte taşınmıştır (bkz. config.py).

        SORGU-TERİM DOYGUNLUĞU (Y1/R30): sorgu BENZERSİZ token'lar üzerinde
        gezilir ve her token'ın katkısı `min(qtf, QTF_CAP)` ile ağırlıklanır.
        Yani aynı terimi 80 kez yazmak skoru 80 değil en fazla 2 katına
        çıkarır; ölçek üst sınırsız kalır (doküman tarafı doygunluğu zaten
        `k1` ile ayrı), ama SORGU tarafı artık sömürülemez.
        """
        out = np.zeros(len(self.doc_freqs), dtype=np.float32)
        for tok, qtf in Counter(tokenize(query)).items():
            idf = self.idf.get(tok)
            if idf is None:
                continue
            qw = min(qtf, QTF_CAP)
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(tok)
                if f:
                    dl = self.doc_lens[i]
                    out[i] += (
                        qw
                        * idf
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    )
        return out


def extract_doc_name_tokens(page_ids: list[str], texts: list[str]) -> dict[str, frozenset[str]]:
    """doc_id -> 1. sayfa başlık satırından türetilmiş JENERİK-DIŞI ad token'ları.

    Adayı 1. sayfanın "KANUN"/"ANAYASA" geçen büyük-harfli satırlarından en
    UZUNU olarak seçer, tokenleştirir ve `_GENERIC`'i çıkarır. Elle yazılmış
    doküman-adı tablosu YOKTUR: ad korpusun kendisinden gelir, bu yüzden
    canary'ye ayar (sızıntı) riski yoktur.

    Adı çıkarılamayan doküman sözlükte yer almaz — yönlendirme onu hiç
    tetiklemez (kapsam dışı; ölçülen ıskalar: c206 KVKK kısaltması, c209
    Anayasa kısmi ad).

    autoresearch exp7 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md.
    """
    names: dict[str, frozenset[str]] = {}
    for pid, text in zip(page_ids, texts, strict=True):
        doc, _, page = pid.partition(":")
        if page != "1":
            continue
        cands = [
            ln.strip()
            for ln in text.splitlines()
            if any(kw in ln for kw in _TITLE_KEYWORDS) and _TITLE_LINE.match(ln.strip())
        ]
        if not cands:
            continue
        toks = frozenset(tokenize(max(cands, key=len))) - _GENERIC
        if toks:
            names[doc] = toks
    return names


def rank_order(scores: np.ndarray) -> np.ndarray:
    """Skor dizisinin AZALAN sırasındaki indeksler — reçetenin sıralama sözleşmesi.

    `kind="stable"` reçetenin parçasıdır: beraberlikte `page_ids` sırası korunur,
    yani aynı skoru alan sayfalar korpus sırasıyla gelir ve sıralama
    deterministiktir. Tek satırlık bir ifade ama İKİ yerde birden kullanılıyordu
    (`HybridRetriever.rank` ve P2 özellik çıkarımı); kopya kalsaydı biri
    "iyileştirildiğinde" kalibratör ölçtüğünden başka bir sıraya takılırdı
    (review m11).

    NOT (T8 borcu): `HybridRetriever.rank` bugün `(ranking, routed)` döndürüyor,
    `order`ı DEĞİL — bu yüzden serve tarafı özellik çıkarımına geçtiğinde argsort
    hâlâ iki kez koşar. Ortadan kaldırmak `rank`ın imzasını değiştirmeyi gerektirir
    ve serve'e dokunmak T8'in işidir; burada yalnız İFADE tekilleştirildi.
    """
    return np.argsort(-scores, kind="stable")


def routed_docs(query: str, doc_names: Mapping[str, frozenset[str]]) -> set[str]:
    """Adının jenerik-dışı TÜM token'ları sorguda geçen doküman(lar).

    Yönlendirme YÜKLEMİ tek yerde durur: `HybridRetriever.routed_docs` buna
    delege eder ve P2 kalibrasyonunun `routed` özelliği de buradan hesaplanır.
    İki ayrı kopya olsaydı, biri değiştiğinde kalibratör ölçtüğünden BAŞKA bir
    boru hattına takılırdı ve `recipe_fingerprint()` bunu göremezdi (kopya
    `hybrid.py`'de, parmak izi `text.py`'nin sabitlerinden türetiliyor).
    """
    q_toks = set(tokenize(query))
    return {doc for doc, toks in doc_names.items() if toks <= q_toks}


def route_window(ranking: list[str], routed_docs: set[str], window: int = WINDOW) -> list[str]:
    """İlk `window` girdiyi YENİDEN SIRALAR: yönlendirilen dokümanların sayfaları öne.

    Sözleşme (yapısal, `window` değerinden bağımsız):
      * ilk `window` girdinin KÜMESİ değişmez (yalnız kendi içinde sıralanır);
      * `window`'dan sonrası hiç dokunulmaz;
      * her iki grup içinde göreli sıra (BM25 sırası) korunur.

    exp6'nın MUTLAK bölümlemesi (aday kümesini değiştiren sürüm) tam da bu
    sözleşmeyi deldiği için veto edildi: R@5 0.7907'ye çıkarken R@20
    0.907->0.837'ye geriliyordu. Pencere-içi sürüm exp7'de R@5 0.8140 + R@20
    0.907; exp8'de pencere 50'ye çıkınca R@5 0.8372 + R@20 0.9302.

    autoresearch exp7/exp8 reçetesi; ölçüm: findings
    2026-08-29-autoresearch-text-channel.md + research/journal.md #8.
    """
    if not routed_docs:
        return list(ranking)
    win = ranking[:window]
    front = [pid for pid in win if pid.partition(":")[0] in routed_docs]
    back = [pid for pid in win if pid.partition(":")[0] not in routed_docs]
    return front + back + list(ranking[window:])
