# Aday havuzu BGE reranker teşhisi — development sonucu

Tarih: 2026-09-03  
Karar: **Üretim değişikliği yok; BM25 top-1 sabit P kolu güvenlik açısından U'dan üstündür, ancak kazanan seçimi için yeni holdout gerekir.**

## Koşum

`retrieval_eval_v2`den yalnız insan doğrulanmış, cevaplanabilir 47 soru seçildi.
Her sorguda yönlendirilmiş BM25'ten ve iki ColBERT kanalından ilk 50 sayfa
skor füzyonu olmadan birleştirildi. Sabit checkpoint
`BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, MPS,
`max_length=512`, `batch_size=8` ile bu havuzu yeniden sıraladı.

- **P:** BM25 top-1 yerinde kalır; 2–N BGE sırasındadır.
- **U:** BGE tüm adayları sıralar; BM25 top-1'in yeni yeri ve U top-1'in eski
  BM25 skoru yalnız tanı olarak kaydedilir.

Rapor: `data/bench/results/candidate-reranker-dev-v1.json`.

## Sonuç

| Kol | R@5 | R@20 | R@50 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Havuzun ilk 50 sırası | 0,6277 | 0,7660 | 0,8085 | 0,4242 | 0,4621 |
| **P — BM25 top-1 sabit** | **0,7766** | 0,8617 | 0,9468 | 0,5185 | 0,5709 |
| U — tam serbest | 0,7553 | 0,8617 | 0,9468 | **0,6937** | **0,6915** |

Havuz kapsaması (sırası değil, tüm aday kümesi): **0,9574**; paraphrase
kapsaması **0,9048**. Bu, ilk 50'nin R@50'sinden ayrıdır: P/U, havuzdaki
50'den sonraki adayları üst sıralara taşıyabilir.

P, U'dan R@5'te bir soru (**+0,0213**) üstündür; R@20 ve R@50 eşittir. U'nun
daha yüksek MRR/nDCG'si top-1'leri daha cesurca yer değiştirmesinden gelir,
fakat bu cevap yolunun güven sözleşmesini ihlal eder: BM25 top-1'in U'daki
medyan olmayan ortalama sırası **13,98**, en kötü sıra **84** oldu. U top-1'i
5/47 soruda 10,6 BM25 eşiğinin altında kaldı (`c104`, `c205`, `c405`, `c409`,
`c411`); U bugün `/ask`e bağlansaydı sistem bulduğu sayfayı yanlış ölçekli
eşikle çekimserliğe çevirebilirdi.

Yalnız BGE bölümü p50 **8.690 ms**, p95 **12.534 ms** sürdü. Bu maliyet,
üretim isteğine eklenmedi.

## Karar sınırı

Bu koşum **geliştirme teşhisidir**, üretim seçimi değildir:

1. `retrieval_eval_v2` bu projede önceki tasarımları yönlendirmek için görülmüş bir
   kümedir; burada yeniden kazanan seçmek veri sızıntısı olur.
2. Bu deney tanım gereği kaynak başına 50 aday kullanır. Bugünkü üretim geç
   aday örmesi 200 derinlikte çalışır; dolayısıyla P'nin R@20'sini üretimdeki
   0,9149 ile doğrudan “reranker kaybı” diye yorumlamak geçerli değildir.
3. U kolu, ayrı uçtan uca çekimserlik kalibrasyonu ve onun için yeni, insan
   doğrulanmış, hukuk-grubu-ayrık kilitli test olmadan offline dışında kalır.

Sonraki karar verisi: daha önce görülmemiş insan-doğrulanmış hukuk-grubu
holdout üzerinde, aynı 50-derinlikli P/U tanısı ile mevcut üretim örme
topolojisinin eşleştirilmiş baseline'ı. Bu veri olmadan bayrak, eşik veya
`HybridRetriever` sırası değiştirilmeyecek.

## Yeniden oynatma

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/eval_candidate_reranker.py \
  --bench data/bench/retrieval_eval_v2.jsonl \
  --min-verification human \
  --out data/bench/results/candidate-reranker-dev-v1.json
```

`--final`, yalnız açık `--yes-final-gate` eşliğiyle kabul edilir; bu mekanik
onay yeni holdout ya da insan doğrulamasının yerine geçmez.
