# Aday Havuzu Reranker Deneyi — Tasarım

Tarih: 2026-09-03

## Amaç

Mevcut sayfa-BM25 + Mogan + Colmm adaylarının taşıdığı fakat sıralamanın ilk
beşine giremediği kanıt sayfalarını, skor füzyonu yapmadan bir cross-encoder ile
yeniden sıralamak. Deney iki açık kolu karşılaştırır:

- **P (pinned):** BM25 top-1 ilk sırada kalır; reranker yalnız 2--50 arasını
  sıralar.
- **U (unpinned):** Aynı havuzun tamamı, BM25 top-1 dahil, reranker tarafından
  sıralanır. Rapor BM25 top-1'in yeni sırasını ve seçilen ilk sayfanın BM25
  skoruyla mevcut 10,6 kapısından geçip geçmeyeceğini ayrıca yazar.

Bu deney bir üretim aktivasyonu değildir. U kolu, ayrı bir cevap-kapısı
kalibrasyonu geçmeden üretime bağlanmaz.

## Gerekçe

İnsan-doğrulanmış retrieval_eval'de sevk edilen sıralama paraphrase R@50'de 0,8571'dir;
buna karşılık BM25, Mogan ve Colmm'nin derinlik-50 aday kümesi 0,9048 kapsama
taşır. `c411` Mogan'da sayfa 29, Colmm'de 44 iken mevcut ardışık örgüde 86.
Bu, en az bir ıskanın getirimden değil sıralama politikasından kaynaklandığını
gösterir. `c206` ve `c404` ise aynı havuzda yoktur; bunlar ilerideki sorgu
varyantı/alan uyarlama deneyinin kapsamındadır.

## Kapsam ve değişmezler

- Donmuş BM25 reçetesi, `recipe_fingerprint`, sayfa düzeyi getirim ve
  `min_score_threshold=10.6` değişmez.
- Mogan ve Colmm skorları BM25 veya birbirleriyle sayısal olarak birleştirilmez.
- Havuz, BM25'in yönlendirilmiş ilk 50 sayfası ile her ColBERT kanalının ilk
  50 chunk'ından indirgenen sayfaların tekrarsız birleşimidir.
- Reranker yalnız `query, page_text` çiftlerini skorlar; görüntü veya LLM
  cevap metni görmez.
- Üretim `HybridRetriever.search()` ve `/ask` varsayılanı deney tamamlanana
  kadar değişmez; yeni yol önce yalnız bench/CLI deneyi olarak yaşar.
- `retrieval_eval_v2` daha önce görüldüğü için teşhis/dev verisidir. Model, havuz
  derinliği veya karar kuralı bunun sonucuna göre seçilemez.

## Bileşen sınırları

`retrieval/candidates.py`, sıralama kaynaklarından bağımsız saf bir
`build_candidate_pool(bm25_pages, late_page_lists, limit=50)` fonksiyonu
sağlar. Çıktı, ilk görülen sırayı koruyan tekrarsız `list[str]`dir; bu katman
skor taşımaz.

`retrieval/rerank.py`, `PageReranker` protokolü ile sınırlandırılır:
`score(query: str, documents: Sequence[str]) -> np.ndarray`. İlk somut kol,
Türkçe destekli, yerel Transformers cross-encoder olarak yapılandırılır.
Model girdisi `page_texts.parquet`teki aynı metindir; hiçbir ikinci OCR ya da
metin çıkarım yüzeyi kurulmaz.

`bench/rerank_experiment.py`, havuzu kurar ve iki kolu üretir. P kolu
`[bm25_top1, *reranked_without_top1]`; U kolu `reranked_all`dır. Her iki kolda
hit skoru, seçilen sayfanın mevcut BM25 skorudur. Böylece U için raporlanan
`would_abstain` mevcut kapının gerçekte nasıl davranacağını gösterir, fakat
üretim eşiğini değiştirmez.

## Veri ve değerlendirme protokolü

1. Yeni, insan-doğrulanmış `rerank_holdout_v1` hazırlanır. Kanun kimliği,
   dev/model seçimi için kullanılan satırlarla kesişmez; özellikle paraphrase
   sorular içerir. Altın sayfa, kanıt alıntısı ve insan doğrulama künyesi
   zorunludur.
2. `retrieval_eval_v2` yalnız mekanizma teşhisi ve hata ayıklama için kullanılır.
3. Holdout tek seferlik nihai koşumdur. Koşumdan sonra model, prompt, havuz
   derinliği veya karar kuralı holdout'a bakılarak ayarlanmaz.
4. Rapor P/U için R@5, R@20, R@50, MRR, nDCG@5, paraphrase dilimi, üç R@50
   guardrail, BM25-top1 sıra dağılımı, `would_abstain` sayısı, p50/p95
   gecikme ve artefakt kimliklerini yazar.

## Karar kuralı

- **P kolu sevke adaydır** ancak holdout'ta paraphrase R@50 >= 0,90, üç
  guardrail R@50 = 1,0, genel R@5 mevcut üretim değerinin altına düşmez ve
  p95 gecikme bütçesi raporlanmışsa.
- **U kolu**, P'den kalitece iyi görünse bile doğrudan sevk edilmez. Seçilen
  ilk sayfanın BM25 skoru kapı için yeni bir dağılım oluşturduğu için, önce
  yeni insan-doğrulanmış cevaplanabilir/cevaplanamaz veride ayrı kalibrasyon
  ve kilitli test gerekir. Bu kapı geçilmezse U yalnız deney sonucu kalır.
- Hiçbir kol koşulları geçmezse üretim yolu aynen kalır; sonraki çalışma
  sorgu varyantı veya alan uyarlamalı retriever olur.

## Hata ve geri alma

- Reranker modeli, `page_texts.parquet` veya havuz kimlikleri yok/uyumsuzsa
  deney açık hatayla durur.
- Boyut, NaN veya sayfa-id hizası uyuşmazsa rapor üretmeden durur.
- Deney kodu üretim bayrağı eklemez; dolayısıyla geri alma yalnız deney
  artefaktını kullanmamaktır.

## Doğrulama

- TDD: havuzun kararlı tekrarsız birleşimi, P'nin top-1 sabitlemesi, U'nin
  BM25 top-1 sırasını raporlaması, skor hizası ve eksik artefakt hataları.
- Aynı girdiyle iki koşumda aynı rapor; model revizyonu, korpus checksum'u,
  bench SHA-256'sı ve ayarlar rapora yazılır.
- Tam `pytest`, Ruff, Pyright; ardından dev teşhisi ve bir kez holdout
  koşumu.
