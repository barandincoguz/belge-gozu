# Anlamsal Kapsama Deneyleri — Tasarım

Tarih: 2026-09-03

## Amaç

Saf anlamsal paraphrase sorularında altın sayfanın aday havuzuna girmesi için
iki offline kolu ardışık test etmek:

1. Çok dilli dense sayfa kanalı.
2. Dense kanalı üzerinde deterministik yerel sorgu genişletme.

Birincil metrik, insan-doğrulanmış cevaplanabilir `canary_v2`de havuzun TAMAMI
üzerindeki fractional R@50 kapsamasıdır. Paraphrase R@50 ayrı zorunlu kırılım
olarak raporlanır. Bu veri daha önce görülmüştür; sonuçlar yalnız teşhis ve
tasarım seçimi içindir, üretim kararı değildir.

## Değişmezler ve kapsam dışı

- `retrieval/text.py`, `recipe_fingerprint()`, BM25 sayfa reçetesi,
  `min_score_threshold=10.6`, `HybridRetriever.search()`, `/search` ve `/ask`
  değişmeyecek.
- Dense ve genişletme kolları önce sadece bench/CLI yolunda yaşar; production
  bayrağı veya varsayılanı eklenmez.
- Sayısal skor füzyonu, RRF ve canary sonucuna göre elle kural ekleme yoktur.
- Yeni insan-doğrulanmış, hukuk-grubu-ayrık holdout olmadan hiçbir sonuç
  üretime sevk edilmez.

## Kol A — dense sayfa adayları

İki sabit checkpoint aynı sayfa metinlerinden, ayrı ayrı dense indeks kurar:

| Model | Sabit revision |
|---|---|
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| `intfloat/multilingual-e5-large` | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |

Her sorguda dense kanal en fazla 50 sayfa döndürür. Aday havuzu, mevcut
yönlendirilmiş BM25 ilk 50 sayfası, Mogan ilk 50 chunk'ından indirgenen sayfalar,
Colmm ilk 50 chunk'ından indirgenen sayfalar ve dense ilk 50 sayfasının ilk
görülme sırasını koruyan tekrarsız birleşimidir. Hiçbir kanalın skoru başka bir
kanalla aritmetik olarak birleştirilmez.

Model seçimi development raporunda en yüksek paraphrase kapsamasıyla yapılır;
eşitlikte daha yüksek genel kapsama, sonra daha küçük indeks diski, sonra daha
düşük sorgu gecikmesi seçer. Bu seçim holdout sonucu değildir ve üretim seçimi
sayılmaz.

## Kol B — yerel sorgu genişletme

Kol A'nın development seçicisiyle kazanan dense modeli kullanılır. Sorgu,
değişmez aşağıdaki modelle tek bir Türkçe hukukî arama varyantına dönüştürülür:

`Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1`.

Sistem istemi yalnız sorgudaki anlamı koruyan, kısa Türkçe hukukî anahtar
terimler/olası kanun adı içeren tek varyant ister; cevap, kanıt, madde numarası
veya gerekçe uydurmasını yasaklar. Çözümleme `do_sample=False` ile deterministiktir.
Boş, özgün sorguya eşit veya parse edilemeyen çıktı açık hatadır; sessiz fallback
ya da üretim sorgusuna müdahale yoktur.

Özgün ve genişletilmiş sorgu; BM25, seçilen dense kanal, Mogan ve Colmm üzerinde
ayrı çalışır. Sekiz aday listesi aynı ilk-görülen/tekrarsız kuralla birleşir;
başka bir sıralama uygulanmaz. Genişletme çıktıları prompt parmak izi, model
revision, soru kimliği ve SHA-256 ile JSONL önbelleğinde saklanır; aynı künyeyle
tekrar koşum önbelleği yeniden kullanır, künyesi farklıysa yeniden üretir.

## Ölçüm ve rapor

Koşucu en az şu kolları aynı seçili sorularda raporlar:

- mevcut üç-kaynak baseline (BM25 + Mogan + Colmm),
- her dense model eklenmiş havuz,
- seçilen dense model + yerel genişletme havuzu.

Her kol için genel ve dilim bazında havuz-kapsaması R@50, R@5/R@20/R@50 ilk sıra
tanıları, MRR/nDCG@5, kanal başına benzersiz gold katkısı, `c206`/`c404`nin aday
durumu, aday havuzu boyutu, p50/p95 sorgu gecikmesi ve dense indeks disk boyutu
kaydedilir. Model, revision, bench SHA-256, indeks/reçete kimlikleri, genişletme
prompt parmak izi ve çıktı önbelleği SHA-256'sı rapor künyesine yazılır.

JSON, atomik yazılan makine kaydıdır. Aynı JSON'dan tek dosyalık HTML karar
raporu üretilir: özet KPI'lar, kol karşılaştırma tablosu, paraphrase kırılımı,
soru-bazlı aday kaynakları, gecikme/disk grafikleri ve görünür “development
only — holdout gerekir” bandı taşır. Rapor hiçbir modeli üretim kazananı diye
etiketlemez.

## Hata davranışı ve doğrulama

- Model/artefakt revisionı, embedding boyutu, page-id hizası, skor sonluluğu
  veya cache künyesi uyuşmazsa koşum rapor yazmadan durur.
- Dense indeks sayfa sırasını `page_texts.parquet` ile birebir doğrular.
- TDD ile dense top-k hizası, skor-füzyonsuz aday birleşimi, deterministik
  genişletme cache anahtarı, geçersiz genişletme reddi, metrik/HTML şeması ve
  atomik çıktı sınanır.
- Tam `pytest`, Ruff, Pyright ve iki gerçek model koşumu sonunda HTML dosyası
  görsel olarak denetlenir.

## Sonraki karar kapısı

Developmentta paraphrase R@50, mevcut 0,9048 kapsamasının üstüne çıkmalı;
genel kapsama ve kritik dilim kapsaması gerilememelidir. Bu koşul yalnız
holdout denemesine aday olmayı sağlar. Yeni insan-doğrulanmış hukuk-grubu-ayrık
holdout aynı protokolle tek sefer koşulmadan dense veya genişletme kanalı
online/productiona bağlanmaz.
