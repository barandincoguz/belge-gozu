# 2026-09-17 — Üretim geç adaylarıyla dev getirim karşılaştırması

## Koşum künyesi

`bench run` önceden geç kanallar etkin olsa da onları yüklemiyor ve
`HybridDiagnosticAdapter` yalnız BM25/yönlendirme sırasını raporluyordu.
`3da1ae0` üretim ve bench yollarını aynı aday örme metoduna bağladı;
`bc7315b` seçim doğrulamasını ve yan indekslerin dosya hash'lerini ekledi.
Bu karşılaştırma **yalnız geliştirme bölmesinin tanısıdır**; test bölmesi
koşulmadı ve G1/G2 final kapı sayısı değildir.

- Kaynak: `data/bench/retrieval_eval_v2.jsonl`, SHA-256
  `81e2ce2b90a25166cdc81594e7e063d73fbb72a1cb211dd3be222b971acaa818`.
- Bölme: `data/bench/splits_v1.json`, SHA-256
  `c0841f7bab8bbb48bcf9bc573765f861330bb76976481110bfa6a236168a7650`.
- Seçim: `verification_kind=human`, `verification_status=verified`,
  `answerable=true`, `question_split(q, splits)=dev`: **24 soru**. Geçici
  JSONL (`/tmp/belge-gozu-retrieval-v2-human-dev.jsonl`) SHA-256
  `234aff68947a7aa3ddb1d700c3c5b0aa642d51fc610e47320acdf73fc1fef9c2`.
- Kod: `bc7315b`. Ana indeks revizyonu
  `133444d8c235/train-compat-v1/int8`, BM25 `recipe_fingerprint`
  `7b56eeeb7327`; iki kolda da aynı.
- Açık koldaki yan indeksler: Mogan-ColBERT-TR
  `ad90b4f64135e4db75a6453feee85fd7b44b33a1` ve ColmmBERT-small-TR
  `3b5dd416a29c8f3abff1c9274f0b08ba69de5232`. Dört dosyanın ayrı
  SHA-256 değeri her yan indeks için raporun `config.late_index_evidence`
  alanında bulunur.

Kaynak kümeden geçici seçimin üretimi:

```python
from pathlib import Path
from belge_gozu.bench.dataset import load_bench, load_splits, question_split

rows = load_bench("data/bench/retrieval_eval_v2.jsonl", min_verification="human")
splits = load_splits("data/bench/splits_v1.json")
dev = [q for q in rows if q.answerable and question_split(q, splits) == "dev"]
Path("/tmp/belge-gozu-retrieval-v2-human-dev.jsonl").write_text(
    "".join(q.model_dump_json() + "\n" for q in dev), encoding="utf-8"
)
```

Her iki koşum gerçek ColSmol modelini ve yerel int8 indeksi kullandı; Hub
erişimine izin verildi. Komutlar:

```bash
BG_LATE_CHANNEL_ENABLED=true HF_HUB_ETAG_TIMEOUT=10 HF_HUB_DOWNLOAD_TIMEOUT=20 .venv-lab/bin/belge-gozu bench run --pipeline hybrid --bench /tmp/belge-gozu-retrieval-v2-human-dev.jsonl --min-verification human --out /tmp/belge-gozu-retrieval-v2-human-dev-bc7315b-late.json
BG_LATE_CHANNEL_ENABLED=false HF_HUB_ETAG_TIMEOUT=10 HF_HUB_DOWNLOAD_TIMEOUT=20 .venv-lab/bin/belge-gozu bench run --pipeline hybrid --bench /tmp/belge-gozu-retrieval-v2-human-dev.jsonl --min-verification human --out /tmp/belge-gozu-retrieval-v2-human-dev-bc7315b-no-late.json
uv run python scripts/verify_evaluation_report.py /tmp/belge-gozu-retrieval-v2-human-dev-bc7315b-late.json --kind retrieval
uv run python scripts/verify_evaluation_report.py /tmp/belge-gozu-retrieval-v2-human-dev-bc7315b-no-late.json --kind retrieval
```

İki rapor da doğrulayıcıda `OK` verdi. Rapor SHA-256 değerleri sırasıyla
geç adaylı kol için `214b670a3b43189ddf653623af73641149f89a45f6b33d9bc70f7ed38f32b520`,
kapalı kol için `e2a3e0c2c6c50feeaa6bc7fd14156ceb5428232a9658497a4e17c669bba91f5c`.
Raporlar ve seçili JSONL şimdilik `/tmp` altındadır; bu kayıt onları
kalıcı/taşınabilir artefakt olarak sunmaz.

## Dev sonuçları

| Ölçüt | Geç kanal kapalı | Geç kanal açık | Fark |
|---|---:|---:|---:|
| R@1 | 0,2292 | 0,2292 | 0 |
| R@5 | 0,6875 | 0,8125 | +0,1250 |
| R@20 | 0,8750 | 0,9583 | +0,0833 |
| R@50 | 0,8750 | 0,9583 | +0,0833 |
| MRR | 0,4405 | 0,5123 | +0,0718 |
| nDCG@5 | 0,4861 | 0,5769 | +0,0909 |

R@5'te **5 soru kazanıldı, 2 soru kaybedildi**, 17 soru aynı kaldı;
net fark **+3/24**. Kazananlar `c101`, `c104`, `c207`, `c401`, `c407`;
kaybedenler `c208`, `c408`. Son iki soru `paraphrase` dilimindedir.
Geç kanal aşamasının ölçülen süresi medyan **354 ms**, en yakın sıra p95
**591 ms** (`n=24`); bu, uçtan uca yanıt gecikmesi değildir.

Güven aralıkları geniştir ve örtüşür: R@5 bootstrap aralığı kapalı kolda
`[0,5000, 0,8542]`, açık kolda `[0,6458, 0,9375]`. Bu dev farkı bir
istatistiksel/nihai kapı zaferi olarak yorumlanmaz. `c208` ve `c408` için
aday örmenin gold sayfayı neden ilk beşten ittiği #17'deki sıralayıcı
kararından önce incelenmelidir. Cevap kalitesi, atıf doğruluğu ve
cevaplanamaz sorularda güvenlik bu koşumun kapsamı dışındadır; #1, #2,
#9, #10 ve #13 açık kalır.
