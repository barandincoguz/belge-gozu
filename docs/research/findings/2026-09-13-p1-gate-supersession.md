# P1 / G1 kapı kararı — 2026-09-13 supersession

Bu belge, `2026-08-31-p1-gate.md` içindeki eski v1/n=43 hükümlerini silmez;
aynı kapıları `retrieval_eval_v2.jsonl` ve insan doğrulamalı n=47 ölçümüyle
yeniden değerlendirir. Reranker artefaktları offline geliştirme koşumudur; üretim
varsayılanı hâlâ hibrittir ve bu sonuçlardan otomatik değiştirilmez.

## Koşum kimliği

- benchmark: `data/bench/retrieval_eval_v2.jsonl`
- benchmark SHA-256: `81e2ce2b90a25166cdc81594e7e063d73fbb72a1cb211dd3be222b971acaa818`
- seçim: `only_verified=true`, `min_verification=human`, answerable n=47
- indeks revision: `133444d8c235/train-compat-v1/int8`
- page text SHA-256: `796ff4dfdea9225409948e4191324e54732b62e5eae41485e1793da768b07978`
- reranker: `BAAI/bge-reranker-v2-m3`, revision
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, candidate limit 50

## Yedi kapı

| Kapı | Ölçüt | Güncel hüküm | Kanıt |
|---|---|---|---|
| G1.1 | Candidate-union Recall@50 ≥ %95 | **PASS** | `20260911-base-w4096.json` candidate-pool coverage **0.9574** (45/47). Bu kez ölçüm late-candidate union'dır; eski v1 `0.9302` tek BM25 sırasıydı. |
| G1.2 | Dört kritik dilimin her birinde R@50 ≥ %90 | **PASS** | `20260911-base-w4096.json`: `dogrudan-madde` 1.0000 (13), `madde-numarali` 1.0000 (6), `ayni-kanun-hard-negative` 1.0000 (5), `paraphrase` **0.9048 (21)**. |
| G1.3 | Reranker kazancı paired bootstrap CI alt sınırı > 0 | **FAIL** | `base-w4096` → `maxp-w4096`: binary first-5 hit 38→40 (4 kazanım, 2 kayıp), delta **0.0426**, deterministic bootstrap %95 CI **[-0.0426, 0.1489]**. Alt sınır sıfırın altında. |
| G1.4 | Uzun sorgunun dayanak sayfası final top-5'te | **PASS** | gerçek-model `tests/retrieval/test_retrieval_regression.py`: rank cırcırı ve short/accentless checks geçti; güncel hibrit rank ≤2. |
| G1.5 | Kalite farkı + gecikme/bellek bütçesi | **KISMİ** | MaxP R@5 `0.8085→0.8511`, fakat offline rerank p50/p95 yaklaşık **24.7/52.2 s**; canlı production p50/p95 ve peak RSS yok. Bu latency, `/ask` production süresi diye sunulamaz. |
| G1.6 | Kazanç göstermeyen katman default kapalı | **PASS** | Dense aday kolu ilk sıra metriklerinde kazanç göstermedi; MaxP yalnız offline artefakt. Production default ve BM25 top-1 korunuyor. |
| G1.7 | Canlı Space boyut/RAM/cold-start/p50/p95 | **ÖLÇÜLMEDİ** | HF Space yok; Docker/Hub/hosting issue’ları açık. |

## Reranker kararı

MaxP, aynı v2/index/recipe üzerinde point estimate olarak iyileşme sağlıyor ve
R@20/R@50/nDCG5 guardrail'lerini düşürmüyor. Ancak paired CI alt sınırı `>0`
koşulunu karşılamadığı için G1.3 kapı hükmü **FAIL** kalır. Production aktivasyonu
başka bir issue/kapı kararıdır; bu belge MaxP'yi default'a almaz.

## Sonraki kilitler

1. #23 artifact validator ile her yeni G1/G2 raporunu doğrula (tamamlandı).
2. #1 bu supersession'ı referanslayan tek güncel G1 raporu ve ASCII/holdout kararını
   yayımla.
3. #17 reranker için paired holdout veya daha geniş insan bench'i; latency/RAM
   ölçümü olmadan production'a alma.
4. #12/#13 benchmark büyüklüğü ve insan doğrulaması tamamlanmadan G1 sayıları
   portfolyo manşeti olarak genelleme.

Kaynak artefaktlar: `20260910-rerank-nodense.json`, `20260911-base-w4096.json`,
`20260911-maxp-w4096.json`, `20260910-1837-semantic-coverage.json`.
