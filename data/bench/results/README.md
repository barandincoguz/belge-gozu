# Koşum artefaktları doğrulama durumu

Bu dizindeki JSON dosyaları ölçüm kanıtıdır; dosya adı veya tek başına aggregate
alanı bir raporu self-audited yapmaz. Canonical retrieval/answer raporlarını şu
araçla doğrulayın:

```bash
uv run python scripts/verify_evaluation_report.py <report.json>
```

## Legacy custom reranker artefaktları

Aşağıdaki dosyalar eski custom reranker şemasını kullanır. Yalnız aggregate
metrik ve gold minimum rank sakladıkları için nDCG ve paired per-question delta
tam yeniden üretilemez; validator bunları bilerek `per_question` eksikliğiyle
reddeder. Bu metrikler [2026-09-13 G1 supersession raporunda](../../../docs/research/findings/2026-09-13-p1-gate-supersession.md)
tarihsel geliştirme kanıtı olarak tutulur, canonical self-audit sonucu olarak
alıntılanmaz:

- `candidate-reranker-dev-v1.json`
- `20260910-rerank-*.json`
- `20260911-*.json`

`eval_candidate_reranker.py` ile üretilen yeni dosyalar `per_question` altında
candidate pool, pinned ve unpinned tam ranking’lerini taşımalıdır. Yeni dosya
önce validator’dan geçirilmeli, sonra bir G1/G2 raporuna alınmalıdır.
