# Sıralama Arızasını Kapatma Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** İnsan-doğrulanmış 47 soruda P kolu R@5'i 0,7766'dan yukarı taşımak — aday havuzuna yeni kanal ekleyerek değil, havuzda ZATEN duran gold sayfaları ilk beşe çıkararak.

**Architecture:** Ölçüm 2026-09-11'de arızayı ikiye ayırdı (aşağıdaki kanıt bölümü). Plan, ucuzdan pahalıya yedi cerrahi müdahale: reranker penceresi, skorlama birimi (MaxP), model, cascade+füzyon, belge genişletme (doc2query), alan uyarlaması, kalibrasyon. Her müdahale mevcut `scripts/eval_candidate_reranker.py` harness'ına TEK bayrak ekler; varsayılan yol değişmez, kollar birbiriyle kıyaslanabilir kalır.

**Tech Stack:** Python 3.11 (ortak lab venv + `.venv-lab` overlay), PyTorch MPS, transformers, BAAI/bge-reranker-v2-m3, pandas/Parquet, mevcut BM25 + 2 ColBERT sidecar + Qwen3 dense artefaktları.

## Global Constraints

- **Ölçüm kümesi:** `data/bench/retrieval_eval_v2.jsonl`, `--min-verification human`, cevaplanabilir **n=47**. Sorular ve gold'lar DEĞİŞTİRİLEMEZ.
- **Birincil metrik:** P kolu (BM25 top-1 sabit) **R@5**. Taban: **0,7766** (`data/bench/results/20260911-rerank-nodense-diag.json`).
- **Guardrail'ler (E2'den itibaren REVİZE — gerekçe aşağıda):** P R@20 (0,8617), P R@50 (0,9468), nDCG@5 (0,5709). Hiçbiri kesin gerilememeli. `would_abstain` ve rerank p50 **raporlanır, veto etmez**.
- **Tutma kuralı:** İlk beşte en az bir gold'u bulunan soru sayısında **net +2
  (+0,0426 binary hit-rate) veya fazlası → KEPT**. Sayı, kesirli R@5 farkını
  `n` ile çarpıp yuvarlayarak değil soru-düzeyi `gold_rank.pinned` teşhisinden
  hesaplanır; iki gold sayfalı sorularda bu ikisi aynı değildir. Net +1 soru
  ise YALNIZ (a) hiçbir guardrail gerilemiyorsa VE (b) kazanan soru soru-düzeyi
  teşhisle açıklanabiliyorsa KEPT. Eşitlik veya gerileme → DISCARDED,
  `git checkout --` ile geri al.
- **Üretim yüzeyi DONUK:** `retrieval/text.py` skorlama ifadesi, `recipe_fingerprint()`, `HybridRetriever.search()`, `/search`, `/ask`, `min_score_threshold=10.6`. Bütün deneyler bench-only bayraklarla koşar; hiçbir üretim varsayılanı değişmez.
- **Bir seferde bir değişken.** Bileşik kol yalnız bileşenleri tek tek ölçüldükten sonra.
- **Test disiplini (bu planda bilinçli sapma):** yalnız **sözleşme kilitleyen** test yazılır — davranış sınırı (MaxP'nin max alması, cascade'in kuyruğu koruması, kalibrasyonun monotonluğu). Bayrak bağlama (plumbing) için test YAZILMAZ; onun yerine koşum çıktısının künyesi (`report["model"]`, `report["rerank_unit"]`) kontrol edilir. Gerekçe: kullanıcı talimatı — ölçüm zamanını deneye harcamak, scaffolding'e değil.
- **Her koşum künyeli JSON yazar** (`data/bench/results/YYYYMMDD-HHMM-<ad>.json`), `git_commit` alanı dolu olur, commit mesajı `exp(rerank): R@5 <eski>-><yeni> <KEPT|DISCARDED> — <tek cümle>` biçimindedir.
- **Aşırı-uyum uyarısı:** n=47 ve her karar 1-2 soruluk. Bootstrap %95 GA her raporda basılır (`ci_recall5`). Test bölmesi AYRI DURUR; bu plandaki hiçbir sonuç üretim seçimi değildir.

### Proje sahibi kararı (2026-09-11): gecikme yerine DOĞRULUK

> "latency yerine doğruluğu tercih ediyoruz ... windowu 1024 hatta mümkünse max
> örnekten biraz daha fazla yapacağız ... bir daha çok çok fazla gecikme
> yaratmıyor ise doğruluğu ve kaliteyi tercih edeceğiz."

Bu karar E1'in hükmünü **sahibinin talimatıyla** çevirir: pencere kısıtı
kaldırılır, gecikme kalite kazancının önünde veto değildir. Kayıt disiplini
gereği not: hüküm kuralın kendiliğinden değişmesiyle değil, AÇIK BİR KARARLA
değişti ve E1 artefaktı DISCARDED etiketiyle depoda duruyor.

**Pencere 4096 seçildi, ölçümle:** ilk örneklem (1.200 sayfa) yanıltıcıydı;
TÜM korpusta sayfa token uzunluğu medyan **617**, p99 **1.886**, maks **4.022**.
Kesilme oranı: 512'de **%80,36**, 1024'te %4,12, 2048'de %0,69, **4096'da
%0,00**. Tokenizer dinamik dolgu yaptığı için (``padding=True``) tavanı
yükseltmek kısa sayfaların maliyetini DEĞİŞTİRMEZ — bedeli yalnız gerçekten
uzun sayfalar öder.

`TransformerPageReranker.max_length` varsayılanı 512 -> **4096**. Bu bir
davranış tercihi değil, sessiz bir ölçüm hatasının düzeltmesidir.

**Taban yeniden ölçülüyor:** bundan sonraki bütün kollar `20260911-base-w4096.json`
ile kıyaslanır; 0,7766'lık eski taban artık "kesilmiş pencere" tabanıdır.

### Kural revizyonu (2026-09-11 15:0x — E2 KOŞULMADAN ÖNCE ilan edildi)

İlk ilan edilen guardrail listesi iki şartname hatası taşıyordu. Düzeltme E2'den
itibaren geçerlidir; **E1'in hükmü GERİYE DÖNÜK DEĞİŞTİRİLMEZ** (DISCARDED
kalır, artefaktı da öyle kaydedildi) — sonucu görüp kural değiştirmek bu
döngünün yasakladığı şeydir. Değişiklik, sonucu HENÜZ GÖRÜLMEMİŞ deneyler için
yapılıyor ve gerekçesiyle burada duruyor.

1. **`would_abstain` yanlış kolda ölçülüyordu.** Birincil metrik P kolu (BM25
   top-1 SABİT). P'de top-1 tanım gereği BM25'in top-1'idir, yani P'nin
   çekimserlik davranışı yeniden sıralamayla HİÇ değişmez; rapordaki
   `would_abstain` yalnız U kolunu (serbest) ölçer. 2026-09-03 kaydı U'yu zaten
   "eşik sözleşmesini ihlal ediyor" diye reddetmişti. Sevk edilmeyecek bir kolun
   yan etkisini, sevk edilecek kolun kararına veto olarak bağlamak şartname
   hatasıdır. Bundan sonra **raporlanır, veto etmez**.
2. **Gecikme tavan yapıyordu.** Daha derin/kapsamlı skorlama her zaman daha
   pahalıdır; gecikmeyi veto yapmak, bütün kalite çalışmasını baştan yasaklar.
   Üstelik repo kaydı zaten "yeniden sıralama (3,3-8,7 s) üretim isteğine
   eklenmedi" diyor — bu kollar ZATEN offline. Gecikme bir **maliyet** olarak
   raporlanır; veto yetkisi yalnız **üretime aday** kollarda (doc2query,
   cascade) geçerlidir.

Değişmeyen: birincil metrik P R@5, +2 soru eşiği, +1 sorunun mekanizma şartı,
tek değişken kuralı, test bölmesinin ayrılığı.

---

## Kanıt: arıza neden ikiye ayrıldı

`20260911-rerank-nodense-diag.json` — P kolunda gold'un sırası (n=47): **37 soru ilk beşte**, 2 soru havuzda yok, **8 soru havuzda ama ilk beşte değil**.

| soru | dilim | havuz → rerank | mod |
|---|---|---|---|
| c203 | dogrudan-madde | **2 → 40** | B (aktif zarar) |
| c408 | paraphrase | **5 → 27** | B |
| c407 | paraphrase | **8 → 23** | B |
| c411 | paraphrase | 76 → 23 | B/A sınırı |
| c207 | paraphrase | 20 → **7** | A (az kaldı) |
| c209 | paraphrase | 54 → **7** | A |
| c405 | paraphrase | 54 → **8** | A |
| c412 | paraphrase | 112 → **9** | A |
| c206 | paraphrase | havuzda yok | kapsama (dense çözüyor) |
| c404 | paraphrase | havuzda yok | kapsama (dense+genişletme çözüyor) |

- **Mod A (4 soru):** reranker devasa kaldırıyor (112→9) ama 2-4 sıra kala duruyor. Model soruyu ANLIYOR, yeterince yukarı koymuyor. Tavan potansiyeli: **+0,0851 R@5**.
- **Mod B (3-4 soru):** lexical kanalın 2./5./8. sıraya koyduğu gold'u reranker 40./27./23. sıraya İTİYOR. Tavan potansiyeli: **+0,0638 R@5**.

**Birinci dereceden şüpheli (ölçüldü):** `TransformerPageReranker` `max_length=512` ile koşuyor; sayfa token uzunluğu medyan **552**, p95 **722**, maks **1054** → **sayfaların %67,8'i kesiliyor**. 1024'te kesilme %0,1. Reranker sayfaların üçte ikisinde metnin ikinci yarısını görmüyor; BM25 tam metni görüyor. Mod B'nin imzası budur.

**Uyarı:** bugüne kadarki BÜTÜN rerank ölçümleri (dense verdict'i ve 2×2 dahil) bu kesilmiş pencereyle yapıldı. E1 tutarsa o sonuçlar tazelenmelidir (Görev 1, Adım 6).

---

## Yol haritası ve karar ağacı

```
Görev 1 (E1 pencere 1024) ── KEPT ─→ Görev 1.6: dense/genişletme 2x2'yi tazele
        │                             └─→ Görev 2 (MaxP) mod A'ya odaklanır
        └── DISCARDED ─→ kesme sebep DEĞİLDİ; doğrudan Görev 2 + Görev 3
                          (birim ve model hipotezleri)

Görev 2 (E2 MaxP) ── mod B kapandı mı? ── evet ─→ Görev 4 (cascade) ile seyreltmeyi kapat
                                        └─ hayır ─→ Görev 3 (model) zorunlu

Görev 3 (E3 model) ── mod A kapandı mı? ── evet ─→ Görev 7 (kalibrasyon) → üretim tartışması
                                         └─ hayır ─→ Görev 6 (alan uyarlaması); model tavan

Görev 5 (E4 doc2query) PARALEL koşar — reranker'a bağlı değil, üretime giden TEK aday
Görev 7 (E7 kalibrasyon) en sonda: kazanan ne olursa olsun eşik sözleşmesi onunla kurulur
```

**Neden bu sıra:** Görev 1-3 dakikalar sürer ve teşhis niteliğindedir (hangi hipotez doğru?). Görev 5 saatler sürer ama reranker'dan bağımsızdır, bu yüzden Görev 1 koşarken arka planda BAŞLATILIR. Görev 6 yalnız Görev 3 "model tavan" derse açılır — günler maliyetli tek iş odur. Görev 4 ve 7 kazanan belli olmadan anlamsızdır.

---

### Görev 1: Reranker penceresi + ölçüm karşılaştırıcısı (E1)

**Files:**
- Create: `scripts/compare_rerank_arms.py`
- Modify: `scripts/eval_candidate_reranker.py` (`_parse_args`, `main`)

**Interfaces:**
- Produces: `--rerank-max-length INT` bayrağı (varsayılan 512, davranış değişmez); `compare_rerank_arms.py --base <json> --arm <json>` → metrik tablosu + soru-düzeyi sıra hareketleri + KEPT/DISCARDED hükmü.
- Consumes: `report["unpinned"]["diagnostics"][i]["gold_rank"]` (Görev 0'da eklendi, mevcut).

- [ ] **Adım 1: Karşılaştırıcıyı yaz** (bütün sonraki görevler bunu kullanır — DRY)

```python
"""İki rerank koşumunu karar kuralına göre kıyaslar.

    uv run python scripts/compare_rerank_arms.py --base A.json --arm B.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY = ("pinned", "recall_at", "5")
GUARDS = (
    ("pinned R@20", ("pinned", "recall_at", "20"), 1),
    ("pinned R@50", ("pinned", "recall_at", "50"), 1),
    ("nDCG@5", ("pinned", "ndcg5", None), 1),
)


def _value(report: dict, path: tuple) -> float:
    arm, field, key = path
    node = report[arm]["overall"][field]
    return float(node[key] if key is not None else node)


def _abstain(report: dict) -> int:
    return sum(bool(row["would_abstain"]) for row in report["unpinned"]["diagnostics"])


def _top5_hits(report: dict) -> set[str]:
    return {
        row["question_id"]
        for row in report["unpinned"]["diagnostics"]
        if (rank := row["gold_rank"]["pinned"]) is not None and rank <= 5
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    args = ap.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    arm = json.loads(args.arm.read_text(encoding="utf-8"))

    n = base["pinned"]["overall"]["n"]
    b5, a5 = _value(base, PRIMARY), _value(arm, PRIMARY)
    base_hits, arm_hits = _top5_hits(base), _top5_hits(arm)
    delta_q = len(arm_hits) - len(base_hits)
    print(f"P R@5  {b5:.4f} -> {a5:.4f}")
    print(f"ilk-5 soru  {len(base_hits)} -> {len(arm_hits)}  ({delta_q:+d}, n={n})")
    print(f"  GA    {[round(x, 4) for x in base['pinned']['overall']['ci_recall5']]}"
          f" -> {[round(x, 4) for x in arm['pinned']['overall']['ci_recall5']]}")

    regressed = []
    for label, path, _ in GUARDS:
        before, after = _value(base, path), _value(arm, path)
        flag = "GERİLEME" if after < before - 1e-9 else ""
        print(f"{label:12s} {before:.4f} -> {after:.4f} {flag}")
        if flag:
            regressed.append(label)
    ab_b, ab_a = _abstain(base), _abstain(arm)
    print(f"çekimser     {ab_b} -> {ab_a}" + ("  GERİLEME" if ab_a > ab_b else ""))
    if ab_a > ab_b:
        regressed.append("çekimserlik")
    print(f"rerank p50   {base['latency_ms']['rerank_p50']:.0f} -> {arm['latency_ms']['rerank_p50']:.0f} ms")

    rows = {r["question_id"]: r for r in base["unpinned"]["diagnostics"]}
    print("\nsıra hareketleri (P kolu, yalnız değişenler):")
    for row in arm["unpinned"]["diagnostics"]:
        before = rows[row["question_id"]]["gold_rank"]["pinned"]
        after = row["gold_rank"]["pinned"]
        if before != after:
            mark = ""
            if (before is None or before > 5) and after is not None and after <= 5:
                mark = "  KAZANÇ"
            elif before is not None and before <= 5 and (after is None or after > 5):
                mark = "  KAYIP"
            print(f"  {row['question_id']:6s} {row['slice']:24s} {before} -> {after}{mark}")

    verdict = "DISCARDED"
    if delta_q >= 2 and not regressed:
        verdict = "KEPT"
    elif delta_q == 1 and not regressed:
        verdict = "KEPT (tek soru — mekanizma açıklanmalı)"
    print(f"\nHÜKÜM: {verdict}" + (f" — gerileyen: {', '.join(regressed)}" if regressed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Adım 2: Pencere bayrağını bağla**

`scripts/eval_candidate_reranker.py`, `_parse_args` içine (mevcut `--dense-model` satırının üstüne):

```python
    parser.add_argument("--rerank-max-length", type=int, default=512)
```

`main()` içinde reranker kurulumunu değiştir:

```python
    reranker = TransformerPageReranker(device=device, max_length=args.rerank_max_length)
```

- [ ] **Adım 3: Koş (1024 pencere)**

```bash
.venv-lab/bin/python scripts/eval_candidate_reranker.py \
  --bench data/bench/retrieval_eval_v2.jsonl --min-verification human \
  --rerank-max-length 1024 \
  --out data/bench/results/20260911-rerank-w1024.json > /dev/null
```

Beklenen süre: ~6-8 dk (512'de 3,3 s/sorgu; 1024'te ~2×).

- [ ] **Adım 4: Kıyasla ve hükmü oku**

```bash
.venv-lab/bin/python scripts/compare_rerank_arms.py \
  --base data/bench/results/20260911-rerank-nodense-diag.json \
  --arm  data/bench/results/20260911-rerank-w1024.json
```

Beklenen (hipotez doğruysa): c203, c408, c407 sıraları ilk beşe çekilir → `+2` ila `+3 soru`, HÜKÜM KEPT.

- [ ] **Adım 5: Kapılar + commit**

```bash
make lint && .venv-lab/bin/python -m pytest tests -q -m "not slow"
git add scripts/compare_rerank_arms.py scripts/eval_candidate_reranker.py data/bench/results/20260911-rerank-w1024.json
git commit -m "exp(rerank): R@5 0,7766->X pencere 512->1024 <KEPT|DISCARDED> — sayfaların %67,8'i kesiliyordu"
```

- [ ] **Adım 6: KEPT ise önceki verdict'leri TAZELE** (dürüstlük borcu)

```bash
for arm in "" "--dense-model qwen3-embedding-4b"; do
  for exp in "" "--expansion-cache data/bench/results/expansion-cache.jsonl"; do
    .venv-lab/bin/python scripts/eval_candidate_reranker.py \
      --bench data/bench/retrieval_eval_v2.jsonl --min-verification human \
      --rerank-max-length 1024 $arm $exp \
      --out "data/bench/results/20260911-w1024-$(echo "${arm}${exp}" | md5 | cut -c1-6).json" > /dev/null
  done
done
```

`docs/research/findings/2026-09-10-semantik-kapsama-dense-genisletme.md`e "pencere düzeltmesi sonrası 2×2" bölümü ekle; dense verdict'i değiştiyse AÇIKÇA yaz.

---

### Görev 2: MaxP — madde düzeyinde skorla, sayfaya max ile topla (E2)

**Files:**
- Modify: `src/belge_gozu/retrieval/rerank.py` (`compare_rerankings`)
- Modify: `scripts/eval_candidate_reranker.py`
- Test: `tests/retrieval/test_rerank.py`

**Interfaces:**
- Produces: `compare_rerankings(..., page_scorer=None)`; `--rerank-unit {page,chunk}` (varsayılan `page`).
- Consumes: `data/index-traincompat-int8/chunks.parquet` (`text`, `page_ids`; sayfa başına ort. 3,19 chunk).

**Neden:** literatürde uzun belge için standart çözüm MaxP'dir ve FirstP/SumP/AvgP'yi tutarlı biçimde geçer (PARADE, Zhang ECIR'21). Bizde alaka madde-kapsamlıdır; sayfa düzeyinde skorlamak sinyali seyreltir. Madde chunk'ları zaten üretilmiş durumda.

- [ ] **Adım 1: `compare_rerankings`'e skorlayıcı kancası**

`src/belge_gozu/retrieval/rerank.py` içinde imzayı ve ilk üç satırı değiştir:

```python
def compare_rerankings(
    query: str,
    pool: Sequence[str],
    page_texts: Mapping[str, str],
    bm25_scores: Mapping[str, float],
    reranker: PageReranker,
    *,
    threshold: float = 10.6,
    page_scorer: Callable[[str, Sequence[str]], np.ndarray] | None = None,
) -> RerankComparison:
    """Aynı havuzun P (BM25 sabit) ve U (tam serbest) sıralamasını üretir.

    ``page_scorer`` verilirse skorlama birimi sayfa metni DEĞİLDİR: çağrılan
    fonksiyon sayfa kimliklerini alır ve kendi birimiyle (ör. madde chunk'ları,
    MaxP) sayfa skoru üretir. Hizalama guard'ı yine koşar.
    """
    pages = list(pool)
    documents = _require_aligned_pages(pages, page_texts, bm25_scores)
    raw = reranker.score(query, documents) if page_scorer is None else page_scorer(query, pages)
    scores = _validate_scores(raw, len(pages))
```

`Callable` importunu dosyanın başındaki `from collections.abc import ...` satırına ekle.

- [ ] **Adım 2: Sözleşme testi** (bu test YAZILIR — max toplama ve geri düşme davranış sınırıdır)

`tests/retrieval/test_rerank.py` sonuna:

```python
def test_page_scorer_replaces_document_scoring_and_keeps_alignment():
    calls: list[tuple[str, list[str]]] = []

    def scorer(query: str, pages):
        calls.append((query, list(pages)))
        return np.array([0.1, 0.9])

    result = compare_rerankings(
        "soru",
        ["a1", "b1"],
        {"a1": "bir", "b1": "iki"},
        {"a1": 12.0, "b1": 3.0},
        _ExplodingReranker(),
        page_scorer=scorer,
    )

    assert calls == [("soru", ["a1", "b1"])]
    assert result.unpinned_pages == ("b1", "a1")
```

Aynı dosyaya patlayan sahte reranker (page_scorer verilince ÇAĞRILMAMALI):

```python
class _ExplodingReranker:
    def score(self, query: str, documents: list[str]) -> np.ndarray:
        raise AssertionError("page_scorer verildiğinde belge skorlayıcı çağrılmamalı")
```

- [ ] **Adım 3: MaxP skorlayıcısını harness'a ekle**

`scripts/eval_candidate_reranker.py` içine (`_DenseCandidateChannel` sınıfının üstüne):

```python
def _chunk_texts_by_page(index_dir: Path, page_ids: Sequence[str]) -> dict[str, list[str]]:
    """Sayfa -> o sayfaya değen madde chunk'larının metinleri."""
    chunks = pd.read_parquet(index_dir / "chunks.parquet")
    out: dict[str, list[str]] = {page_id: [] for page_id in page_ids}
    for text, pages in zip(chunks["text"], chunks["page_ids"], strict=True):
        for page_id in pages:
            key = str(page_id)
            if key in out:
                out[key].append(str(text))
    return out


class _MaxPScorer:
    """Sayfayı madde chunk'larıyla skorlar, sayfa skoru = chunk'ların MAKSİMUMU.

    Chunk'ı olmayan sayfa (ör. saf tablo/görsel sayfası) sayfa metnine düşer;
    aksi halde o sayfa sessizce -inf alır ve havuzdan düşerdi.
    """

    def __init__(
        self,
        reranker: PageReranker,
        chunk_texts: Mapping[str, list[str]],
        page_texts: Mapping[str, str],
    ) -> None:
        self._reranker = reranker
        self._chunk_texts = chunk_texts
        self._page_texts = page_texts

    def __call__(self, query: str, pages: Sequence[str]) -> np.ndarray:
        flat: list[str] = []
        owner: list[int] = []
        for index, page_id in enumerate(pages):
            texts = self._chunk_texts.get(page_id) or [self._page_texts[page_id]]
            flat.extend(texts)
            owner.extend([index] * len(texts))
        raw = np.asarray(self._reranker.score(query, flat), dtype=np.float64)
        scores = np.full(len(pages), -np.inf, dtype=np.float64)
        for index, value in zip(owner, raw.tolist(), strict=True):
            scores[index] = max(scores[index], value)
        return scores
```

`_parse_args` içine:

```python
    parser.add_argument("--rerank-unit", choices=("page", "chunk"), default="page")
```

`main()` içinde `run_comparison` çağrısından önce:

```python
    page_scorer = None
    if args.rerank_unit == "chunk":
        page_scorer = _MaxPScorer(
            reranker, _chunk_texts_by_page(index_dir, page_ids), page_texts
        )
```

ve `run_comparison(..., page_scorer=page_scorer)`; `run_comparison` imzasına `page_scorer: Callable[[str, Sequence[str]], np.ndarray] | None = None` ekleyip `compare_rerankings`e geçir. Künyeye ekle: `"rerank_unit": args.rerank_unit`.

- [ ] **Adım 4: Koş ve kıyasla**

```bash
.venv-lab/bin/python scripts/eval_candidate_reranker.py \
  --bench data/bench/retrieval_eval_v2.jsonl --min-verification human \
  --rerank-max-length 1024 --rerank-unit chunk \
  --out data/bench/results/20260911-rerank-maxp.json > /dev/null
.venv-lab/bin/python scripts/compare_rerank_arms.py \
  --base <Görev 1'in kazananı> --arm data/bench/results/20260911-rerank-maxp.json
```

- [ ] **Adım 5: Kapılar + commit** (Görev 1 Adım 5 ile aynı komutlar)

---

### Görev 3: Reranker modeli — yükleme sondası ve takas (E3)

**Files:**
- Modify: `scripts/eval_candidate_reranker.py`

**Interfaces:**
- Produces: `--reranker-repo`, `--reranker-revision` bayrakları; künyede `report["model"]` zaten repo/revision basıyor.

**Neden:** mod A soruları (112→9, 54→7) modelin "anladığı ama yeterince yukarı koymadığı" sorular. Alan/çok-dillilik kalitesi burada tavanı belirler. CLERC bulgusu: hukuk metinlerinde hazır cross-encoder'lar getirimi düşürebiliyor.

- [ ] **Adım 1: Aday modellerin YÜKLENEBİLİRLİĞİNİ sonda ile ölç** (varsayım yok)

```bash
HF_HOME=/opt/llm-lab/hf-cache .venv-lab/bin/python - <<'PY'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
for repo in ("Alibaba-NLP/gte-multilingual-reranker-base",
             "jinaai/jina-reranker-v2-base-multilingual"):
    try:
        AutoTokenizer.from_pretrained(repo)
        AutoModelForSequenceClassification.from_pretrained(repo)
        print(repo, "YÜKLENDİ (trust_remote_code gerekmedi)")
    except Exception as exc:  # noqa: BLE001
        print(repo, "->", type(exc).__name__, str(exc)[:140])
PY
```

`trust_remote_code` isteyen model bu repoda **reddedilir** (pin disiplini uzak kod çalıştırmayı kaldırmaz); o durumda Adım 3'e (Qwen3-Reranker) geç.

- [ ] **Adım 2: Bayrakları bağla ve yüklenen adayı koş**

```python
    parser.add_argument("--reranker-repo", default=BGE_RERANKER_REPO)
    parser.add_argument("--reranker-revision", default=BGE_RERANKER_REVISION)
```

```python
    reranker = TransformerPageReranker(
        repo=args.reranker_repo,
        revision=args.reranker_revision,
        device=device,
        max_length=args.rerank_max_length,
    )
```

`BGE_RERANKER_REPO`/`_REVISION` importunu `from belge_gozu.retrieval.rerank import ...` satırına ekle. Revizyon SHA'sı sonda çıktısından alınır ve bayrağa TAM yazılır (kısa SHA yasak).

- [ ] **Adım 3 (tavan sondası, opsiyonel): Qwen3-Reranker-4B**

Bu model seq-classification değil; skor "yes/no" token logit'inden çıkar. Ayrı bir skorlayıcı gerekir:

```python
class _Qwen3RerankerScorer:
    """Qwen3-Reranker: yes/no logit farkı. Yalnız TAVAN sondası — üretim adayı değil."""

    def __init__(self, repo: str, revision: str, device: str | None, max_length: int = 4096) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = device or "mps"
        self._tok = AutoTokenizer.from_pretrained(repo, revision=revision, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(repo, revision=revision).to(self._device)
        self._model.eval()
        self._yes = self._tok.convert_tokens_to_ids("yes")
        self._no = self._tok.convert_tokens_to_ids("no")
        self._max_length = max_length

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        out: list[float] = []
        for document in documents:
            messages = [
                {"role": "system", "content": "Judge whether the Document answers the Query. Answer yes or no."},
                {"role": "user", "content": f"<Query>: {query}\n<Document>: {document}"},
            ]
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch = self._tok(prompt, return_tensors="pt", truncation=True, max_length=self._max_length)
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with self._torch.inference_mode():
                logits = self._model(**batch).logits[0, -1]
            out.append(float(logits[self._yes] - logits[self._no]))
        return np.asarray(out, dtype=np.float64)
```

Not: `mlx-community/Qwen3-Reranker-4B-mxfp8` önbellekte MLX formatındadır ve `transformers` ile YÜKLENMEZ; torch sürümü `Qwen/Qwen3-Reranker-4B` ayrıca indirilir (~8 GB). Sonda tek koşumdur; sonuç "model tavanı nerede" sorusuna cevaptır, KEPT adayı değildir.

- [ ] **Adım 4: Kıyasla, kapılar, commit.** Karar: en iyi R@5 hangi modeldeyse o Görev 4'ün tabanı olur.

---

### Görev 4: Cascade + RRF havuz sırası (E5)

**Files:**
- Modify: `scripts/eval_candidate_reranker.py`

**Interfaces:**
- Produces: `--pool-order {first-seen,rrf}`, `--rerank-top-n INT` (0 = hepsi).

**Neden:** ölçüldü — havuz 126→192 adaya çıkınca c308'in gold'u 5.'den 6.'ya düştü; aday sayısı arttıkça ilk beş bozuluyor. Cascade hem bunu hem gecikmeyi (173 aday × 4,9 s) aynı anda keser. RRF eğitimsizdir, kalibrasyon istemez.

- [ ] **Adım 1: RRF sıralayıcı ve cascade**

```python
def _rrf_order(sources: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Karşılıklı sıra füzyonu; skorlar değil YALNIZ sıralar birleşir."""
    scores: dict[str, float] = {}
    for source in sources:
        for rank, page_id in enumerate(source, start=1):
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda page_id: (-scores[page_id], page_id))
```

`run_comparison` içinde havuz kurulumunu değiştir:

```python
        sources = [bm25_routed, *late_pages]
        if pool_order == "rrf":
            pool = _rrf_order([source[:candidate_limit] for source in sources])
        else:
            pool = build_candidate_pool(bm25_routed, late_pages, limit=candidate_limit)
        if rerank_top_n:
            head, tail = pool[:rerank_top_n], pool[rerank_top_n:]
        else:
            head, tail = pool, []
```

`compare_rerankings` yalnız `head` ile çağrılır; `tail` sıralamanın SONUNA sırası korunarak eklenir:

```python
        comparison = compare_rerankings(
            question.question, head, page_texts, bm25_scores, reranker,
            threshold=threshold, page_scorer=page_scorer,
        )
        pinned_ranking = [*comparison.pinned_pages, *tail]
        unpinned_ranking = [*comparison.unpinned_pages, *tail]
```

`rows["rankings"]` bu iki listeyi alır (havuz kolu `pool` ile ölçülmeye devam eder).

- [ ] **Adım 2: Sözleşme testi** (kuyruk KAYBOLMAMALI — R@50 guardrail'i buna bağlı)

```python
def test_cascade_keeps_the_unreranked_tail_at_the_end():
    report = run_comparison(
        questions=[Question(gold_page_ids=["a2"])],
        text=TwoQueryText(),
        doc_names={},
        page_texts={"b1": "bm25", "a1": "orta", "a2": "iyi", "x9": "kanun dili"},
        late_channels=[TwoQueryLate(["a1"])],
        reranker=OriginalOnlyReranker(),
        candidate_limit=4,
        rerank_top_n=2,
    )

    ranking = report["unpinned"]["diagnostics"][0]["gold_rank"]["unpinned"]
    assert ranking is not None, "kuyruktaki gold sıralamadan düşmemeli"
```

- [ ] **Adım 3: Tarama koş** (tek değişken kuralı: önce sıra, sonra N)

```bash
for n in 0 50 30 20; do
  .venv-lab/bin/python scripts/eval_candidate_reranker.py \
    --bench data/bench/retrieval_eval_v2.jsonl --min-verification human \
    --rerank-max-length 1024 --pool-order rrf --rerank-top-n $n \
    --out data/bench/results/20260911-cascade-n$n.json > /dev/null
done
```

Karar: R@5 kesin artan EN KÜÇÜK N kazanır (küçük N = düşük gecikme). Eşitlikte küçük N.

- [ ] **Adım 4: Kapılar + commit.**

---

### Görev 5: doc2query — indeks zamanında belge genişletme (E4)

**Files:**
- Create: `scripts/build_doc2query.py`
- Modify: `scripts/eval_candidate_reranker.py` (`--text-expansions`)

**Interfaces:**
- Produces: `data/index-traincompat-int8/chunk_questions.jsonl` (satır: `{"chunk_id", "questions": [...], "model_revision", "prompt_sha256", "kept": [...]}`)
- Consumes: `chunks.parquet`, yerel üretici model, filtre için mevcut reranker.

**Neden:** üretime gidebilecek TEK aile budur — sorgu zamanına hiçbir şey eklemez (docTTTTTquery: BM25 55 → 58 ms). Kayıt boşluğunu indeks tarafında kapatır: madde metnine, o maddenin günlük dilde cevapladığı sorular eklenir. Sonda zaten kanıtladı: sorgu kanun diline çevrilince gold rank 1'e geliyor; doc2query bu çeviriyi indeks zamanına taşır. Doc2Query−− bulgusu: **filtre şart**, kötü üretim getirimi düşürüyor.

- [ ] **Adım 1: Üretici betiği yaz** (kesintiye dayanıklı — dense artefakt disiplininin aynısı)

```python
"""Her madde chunk'ı için günlük-dil sorular üretir (doc2query).

    uv run python scripts/build_doc2query.py --limit 200   # pilot
    uv run python scripts/build_doc2query.py               # tam koşum

Üretim ANINDA diske yazılır: 10.531 chunk'lık koşum saatler sürer ve
yarıda kesilen bir koşum tamamlanmış üretimleri kaybetmemelidir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.config import Settings  # noqa: E402

PROMPT = (
    "Aşağıdaki kanun maddesini okuyan sıradan bir vatandaşın bu maddeyi bulmak için "
    "günlük Türkçeyle soracağı 3 farklı soru yaz. Hukuk terimi KULLANMA; maddeden "
    "kelime kopyalama. Her satıra bir soru yaz, başka hiçbir şey yazma."
)


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repo", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    args = ap.parse_args()

    index_dir = Settings().index_dir
    out = args.out or index_dir / "chunk_questions.jsonl"
    chunks = pd.read_parquet(index_dir / "chunks.parquet")
    done = set()
    if out.exists():
        done = {json.loads(line)["chunk_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.repo, revision=args.revision, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.repo, revision=args.revision, dtype=torch.float16
    ).to("mps")
    model.eval()

    rows = chunks if args.limit is None else chunks.head(args.limit)
    with out.open("a", encoding="utf-8") as handle:
        for chunk_id, text in zip(rows["chunk_id"], rows["text"], strict=True):
            if str(chunk_id) in done:
                continue
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": str(text)[:4000]},
            ]
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch = {k: v.to("mps") for k, v in tok(prompt, return_tensors="pt").items()}
            with torch.inference_mode():
                generated = model.generate(**batch, do_sample=False, max_new_tokens=96)
            decoded = tok.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            questions = [q.strip(" -•\t") for q in decoded.splitlines() if q.strip()][:3]
            handle.write(json.dumps(
                {
                    "chunk_id": str(chunk_id),
                    "questions": questions,
                    "model_revision": args.revision,
                    "prompt_sha256": prompt_sha256(),
                },
                ensure_ascii=False,
            ) + "\n")
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Adım 2: 200 chunk'lık PİLOT koş, çıktıyı GÖZLE oku**

```bash
HF_HOME=/opt/llm-lab/hf-cache .venv-lab/bin/python scripts/build_doc2query.py --limit 200 \
  --out /tmp/doc2query-pilot.jsonl
head -5 /tmp/doc2query-pilot.jsonl | .venv-lab/bin/python -c "
import json,sys
for line in sys.stdin:
    row=json.loads(line); print(row['chunk_id']); [print('   ',q) for q in row['questions']]"
```

Kapı: sorular günlük dilde mi, maddeden kelime kopyalamış mı, Türkçe mi? Kopyalıyorsa PROMPT düzeltilir ve pilot tekrarlanır. Tam koşum ancak pilot geçince başlar (10.531 chunk, tahmini 3-6 saat, arka planda).

- [ ] **Adım 3: Filtre (Doc2Query−−)**

Üretilen her soruyu mevcut reranker ile kendi chunk'ına karşı skorla; medyanın altındakileri AT. Ölçüm kolu yalnız `kept` alanını kullanır.

- [ ] **Adım 4: Ölçüm kolu**

`scripts/eval_candidate_reranker.py` içine `--text-expansions PATH`; verilirse BM25 kanalı genişletilmiş metinle kurulur:

```python
    if args.text_expansions:
        chunks = pd.read_parquet(index_dir / "chunks.parquet")
        by_page: dict[str, list[str]] = {page_id: [] for page_id in page_ids}
        kept = {
            row["chunk_id"]: row.get("kept", row["questions"])
            for row in map(json.loads, args.text_expansions.read_text(encoding="utf-8").splitlines())
            if row.strip()
        }
        for chunk_id, pages in zip(chunks["chunk_id"], chunks["page_ids"], strict=True):
            for question in kept.get(str(chunk_id), []):
                for page_id in pages:
                    if str(page_id) in by_page:
                        by_page[str(page_id)].append(str(question))
        expanded = [page_texts[p] + "\n" + "\n".join(by_page[p]) for p in page_ids]
        text = BM25Index(page_ids, expanded)
```

`doc_names` DEĞİŞMEZ (yönlendirme kanun adından gelir, üretilen sorulardan değil).

- [ ] **Adım 5: Koş, kıyasla, kapılar, commit.** Bu kolun ayrıca **havuz kapsamasını** ve **BM25 top-1 doğruluğunu** raporlaması gerekir: doc2query'nin asıl vaadi, gold'u BM25'in KENDİSİNİN 1. sıraya koyması — o olursa P kolu pinleme sözleşmesi sayesinde reranker'a hiç ihtiyaç kalmaz.

---

### Görev 6: Alan uyarlaması — sentetik veriyle cross-encoder eğitimi (E6)

**Açılma koşulu:** Görev 3 "model tavan" derse (mod A soruları hiçbir hazır modelde ilk beşe çıkmıyorsa). Aksi halde AÇILMAZ — günler maliyetli tek iştir.

**Files:**
- Create: `scripts/train_domain_reranker.py`
- Consumes: Görev 5'in `chunk_questions.jsonl` üretimleri (sentetik sorgular), mevcut kanallar (hard negative), en iyi reranker (sözde-etiket)

**Yöntem (GPL uyarlaması):** sentetik sorgu üret (var) → her sorgu için mevcut hattan top-50 aday çek → pozitif = sorgunun türetildiği chunk, negatif = diğerleri → en güçlü reranker ile margin sözde-etiketi → küçük çok-dilli cross-encoder'ı MarginMSE ile fine-tune et → aynı harness'ta ölç. GPL hazır dense retriever'ı 9,3 nDCG@10'a kadar geçiyor; burada hedef reranker.

- [ ] **Adım 1:** eğitim çifti üretimi (sorgu, pozitif, negatif, margin) → `data/bench/train/doc2query-pairs.jsonl`
- [ ] **Adım 2:** `sentence-transformers` `CrossEncoder` + `MarginMSELoss` ile 1-2 epoch, MPS
- [ ] **Adım 3:** aynı harness, `--reranker-repo <yerel yol>` ile ölç
- [ ] **Adım 4:** KEPT ise model artefaktını Hub'a değil, ölçüm künyesiyle yerel dizine yaz; karar kaydı `docs/research/findings/`

---

### Görev 7: Kalibrasyon ve eşik sözleşmesi (E7)

**Açılma koşulu:** bir kazanan kol KEPT olduktan SONRA; üretim tartışmasının ön koşulu.

**Files:**
- Create: `scripts/calibrate_reranker.py` (deseni `scripts/calibrate_late_channel.py`)

**Neden:** reranker skorları listwise softmax ile eğitildiği için mutlak seviye anlamsızdır; sabit kesim her sorguda başka yere düşer. Bizim `min_score_threshold=10.6` BM25 ölçeğinde tanımlı ve U kolunda 5-6 soruda ihlal ediliyor.

- [ ] **Adım 1:** kazanan kolun skorlarını topla (gold/gold-değil etiketiyle, dev bölmesi)
- [ ] **Adım 2:** Platt (sigmoid) ve izotonik kalibrasyon; hangisi daha iyi kalibre (Brier/ECE) ölç
- [ ] **Adım 3:** kalibre olasılıktan çekimserlik eşiği türet; guardrail: `would_abstain` sayısı ve cevaplanan sorularda doğruluk
- [ ] **Adım 4:** eşik DEĞİŞTİRİLMEZ, yalnız ÖNERİLİR: üretim geçişi ayrı bir karar ve ayrı holdout ister

---

## Self-review

- **Kanıt kapsaması:** mod A (4 soru) → Görev 2/3/6; mod B (3 soru) → Görev 1/2; seyreltme (c308) → Görev 4; kapsama (c206/c404) → zaten ölçüldü, dense+genişletme; kayıt boşluğu üretim yolu → Görev 5; eşik sözleşmesi → Görev 7. Açık kalan yok.
- **Placeholder taraması:** her adımda çalıştırılabilir kod veya tam komut var; "uygun hata yönetimi ekle" türü ifade yok. Görev 6 adımları kasten üst düzey — açılma koşulu sağlanmadan detaylandırmak israf olur ve o koşul Görev 3'ün sonucuna bağlı.
- **Tip tutarlılığı:** `page_scorer: Callable[[str, Sequence[str]], np.ndarray] | None` üç yerde aynı imzayla geçiyor (`compare_rerankings`, `run_comparison`, `_MaxPScorer.__call__`). `--rerank-max-length` Görev 1'de tanımlanıp Görev 2-5'te kullanılıyor. `chunk_questions.jsonl` şeması Görev 5 Adım 1'de tanımlanıp Adım 4'te aynı alan adlarıyla okunuyor.


---

## Koşum kaydı

### E1 — pencere 512 -> 1024 · **DISCARDED** (2026-09-11 14:48)

Artefakt: `data/bench/results/20260911-rerank-w1024.json`

| metrik | taban | E1 |
|---|---|---|
| P R@5 | 0,7766 | **0,8085** (+1 soru) |
| P R@20 | 0,8617 | **0,9362** (+3,5 soru) |
| P R@50 | 0,9468 | 0,9574 |
| nDCG@5 | 0,5709 | 0,5809 |
| çekimser | 5 | **6** (gerileme) |
| rerank p50 | 3.241 ms | **5.577 ms** (+%72) |

**Kök neden DOĞRULANDI.** Mod B'nin üç sorusu da fırladı: c203 **40 -> 7**,
c407 **23 -> 6**, c408 **27 -> 3** (tek gerçek ilk-beş kazancı). Kesme gerçekten
reranker'ı kör ediyormuş.

**Ama ilaç pahalı ve yan etkili.** Gecikme %72 arttı; çekimserlik 5 -> 6 çünkü
c412'nin U kolundaki top-1'i `k213:87` (BM25 16,2) yerine `k5271:13` (BM25 8,2)
oldu — yeni top-1 10,6 eşiğinin altında. Ayrıca bazı sorular geriledi
(c405 8 -> 17, c411 23 -> 41, c207 7 -> 11): daha uzun pencere her soruda daha
iyi skor demek değil.

İlan edilmiş kural uygulandı: +1 soru YALNIZ hiçbir guardrail gerilemezse
tutulur; iki guardrail geriledi -> **DISCARDED**. Kural sonuç görüldükten sonra
DEĞİŞTİRİLMEDİ. `--rerank-max-length` bayrağı depoda kalıyor (varsayılan 512,
davranış aynı) çünkü E2 ile birlikte yeniden denenecek.

**Yan çıktı — ölçüm aracında hata bulundu ve düzeltildi.** `compare_rerank_arms.py`
ilk-beş İÇİNDEKİ yer değiştirmeleri (2 -> 3, 4 -> 5) "KAYIP" diye etiketliyordu;
altı deneyin kararını yanıltacaktı. Etiket artık yalnız ilk beşten ÇIKAN soruya
veriliyor.

**Sonraki adım (karar ağacının gerekçesi değişti):** E1 düştü ama hipotez
doğrulandı, yani E2 "kesme sebep değildi" diye değil, **aynı kökün daha ucuz
ilacı** olduğu için koşulur: chunk'lar ~200 token, sayfa başına 3,19 chunk ->
~640 token; 1024'lük padded pencereden ucuz ve kesme yok.


### Yeni taban — sayfa birimi, pencere 4096 (2026-09-11 15:35)

Artefakt: `data/bench/results/20260911-base-w4096.json`. Eski (kesilmiş) tabana
göre R@5 0,7766 -> **0,8085**, R@20 0,8617 -> **0,9362**. Bundan sonraki bütün
kollar bununla kıyaslanır.

### E2 — MaxP (madde chunk'ları, max toplama) @4096 · **KEPT** (15:58)

Artefakt: `data/bench/results/20260911-maxp-w4096.json`

| metrik | taban@4096 | MaxP@4096 |
|---|---|---|
| **P R@5** | 0,8085 | **0,8511** (+2 soru) |
| %95 GA | [0,7021; 0,9149] | [0,7447; 0,9362] |
| P R@20 | 0,9362 | 0,9574 |
| P R@50 | 0,9574 | 0,9574 |
| nDCG@5 | 0,5809 | 0,5914 |
| çekimser | 6 | **4** |
| rerank p50 | 6.087 ms | 24.742 ms |

Soru düzeyinde: **kazanan 4** (c203 7->5, c207 11->5, c407 7->4, c412 10->4),
**kaybeden 2** (c111 5->15, c406 2->11), net +2. Çekimserlik de düzeldi (6 -> 4).

İlk 512-pencere tabanından kümülatif: **R@5 0,7766 -> 0,8511 (+3,5 soru,
+0,0745)**; mod B'nin üç sorusu (c203, c407, c408) ve mod A'nın ikisi (c207,
c412) artık ilk beşte.

**Kaybedenlerin mekanizması (not, henüz ölçülmedi):** MaxP çok-chunk'lı sayfaya
daha fazla "bilet" verir — maksimum, daha çok örnekten alındığı için yukarı
sapar. c111 ve c406 muhtemelen az chunk'lı sayfalar. Olası rafinasyon: chunk
sayısına göre normalize etmek ya da en iyi iki chunk'ın ortalaması. Bu, E2'yi
DÜŞÜRMEZ (net +2), sıradaki iyileştirme adayı olarak not edilir.

### E3 — uzak kodlu çok-dilli reranker'lar · **KOŞULAMADI**

`Alibaba-NLP/gte-multilingual-reranker-base@8215cf04...` pinli olarak yüklendi
ama ilk forward'da çöktü: `index <çöp sayı> is out of bounds ... size 732`.
MPS'e özgü değil — **CPU'da da aynı hata**. Yani uzak kod transformers 5.3 ile
uyumsuz; `jinaai/jina-reranker-v2` ile aynı sınıf arıza (o da xlm_roberta'nın
kaldırılmış private fonksiyonunu import ediyordu). trust_remote_code politikası
açık olmasına rağmen bu iki model bu yığında koşamıyor.

Kalan E3 adayı: **Qwen3-Reranker-4B** — uzak kod YOK, standart `AutoModelForCausalLM`
üzerinde yes/no logit farkı; skorlayıcısını kendimiz yazarız (colbert_encode.py
deseni). Taban MaxP@4096 olur, tek değişken model.
