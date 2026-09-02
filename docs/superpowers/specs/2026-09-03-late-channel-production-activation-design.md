# Geç Kanal Üretim Aktivasyonu — Tasarım

Tarih: 2026-09-03

## Amaç

İnsan doğrulamalı canary ölçümünde en iyi sonucu veren sayfa-BM25 + iki
chunk-ColBERT aday birleşimini varsayılan hibrit üretim yoluna bağlamak:
R@5 `0,6277 -> 0,7766`, R@50 `0,8085 -> 0,9362`, paraphrase R@50
`0,5714 -> 0,8571`.

## Karar

- BM25 sayfa düzeyinde kalır ve her zaman ilk adayı korur.
- Mogan-ColBERT-TR ve ColmmBERT-small-TR chunk üzerinde çalışır.
- Kanalların skorları karıştırılmaz; sıralamalar `union_candidates` ile
  sırayla örülür ve chunk adayları sayfaya indirgenir.
- `PageHit.score` BM25 ölçeğinde kalır. Eski `min_score_threshold=10.6` yalnız
  BM25'in sabit tuttuğu top-1'e uygulanır.
- Başarısız `late-channel-v1` kalibratörü açılmaz: kilitli testte kapsamı
  `0/175`tir. Geç kanalın top-1'i devralmasına veya geç skorla cevap kapısı
  kurulmasına izin verilmez.
- İki geç indeks eksik, bozuk, yinelenen kimlikli veya chunk eşlemesiyle
  uyumsuzsa uygulama başlangıçta açık hata verir.

## Yapılandırma ve veri akışı

`Settings`, geç kanal etkinliği ile iki indeks yolunu taşır. Varsayılan hibrit
üretim yapılandırmasında kanal açıktır; ablasyon ve testler açıkça kapatabilir.
`build_retriever`, ana indeksin `chunks.parquet` artefaktından `chunk_id ->
page_ids` eşlemesini kurar, iki geç indeksi ve sabit model revizyonlarını yükler
ve kanalları `HybridRetriever`a verir.

`HybridRetriever.search` BM25 ve mevcut doküman yönlendirmesini hesapladıktan
sonra iki geç kanalın sayfa adaylarını sırayla örer. Sonuç listesinin ilk öğesi
BM25 top-1 olarak kalır; dolayısıyla çekimserlik ölçeği değişmez. Geç kanal
özetleri telemetri künyesine ayrı alanlarla yazılır.

## Hata davranışı

- Geç kanal yalnız `hybrid` pipeline ile kullanılabilir.
- Gerekli `colbert.json`, `embs.npy`, `offsets.npy`, `chunk_ids.json` veya
  `chunks.parquet` yoksa başlangıç durur.
- Model revizyonu ve indeks boyutları sidecar sözleşmesiyle uyuşmazsa başlangıç
  durur.
- Sabit-BM25-top-1 değişmezi bozulursa test başarısız olur; kalibrasyonsuz geç
  kanalın cevap kapısını sahiplenmesine izin verilmez.

## Doğrulama

- TDD: yapılandırma, indeks yükleme, aday örme, top-1/score ölçeği, başlangıç
  korkulukları ve `/healthz` görünürlüğü.
- Tam `pytest`, Ruff ve Pyright.
- `scripts/eval_late_channel.py` ile üretim sınıflarının ölçülen
  `0,7766 / 0,9149 / 0,9362 / 0,8571` sonucunu yeniden üretmesi.

## Temizlik sınırı

Yalnız yeniden üretilebilir `graphify-out/`, `.pytest_cache/`, `.ruff_cache/`
ve `__pycache__/` dizinleri silinir. Geç indeksler, kalibrasyon/bench verileri,
`.agents/`, `skills-lock.json` ve kökeni belirsiz `st.bin` korunur.
