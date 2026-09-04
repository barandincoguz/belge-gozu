# Colab ile sürümlü dense artefakt üretimi

**Tarih:** 2026-09-04

## Karar

Uzun süren Qwen dense sayfa embedding üretimi Colab GPU üzerinde yapılacak. Ortaya
çıkan deney artefaktları üretim indeks deposundan ayrı
`barandincoguz/belge-gozu-semantic-artifacts` Hugging Face Dataset deposuna
yüklenecek. Yerel depo bu artefaktları yalnız açık bir Hub commit SHA'sı ile,
doğruladıktan sonra indirecek. Üretim indeksi, cevap kanalı ve kalibrasyon bu
çalışmanın dışında kalır.

Bu, yalnız offline `retrieval_eval_vN` kapsam ölçümünü hızlandırır; bir sonucu
üretime alma kararı veya skor eşiği değiştirmez.

## Seçenekler

1. **Ayrı Dataset deposu (seçilen).** Deney verisi ile üretim indeksinin yaşam
   döngüsü, erişimi ve silme kuralları ayrılır. Ek bir repo kimliği gerekir.
2. Üretim Dataset deposunda `experiments/` dizini. Tek depo kolaylığı sağlar,
   ancak artefakt temizliği veya yanlış bir indirme üretim sözleşmesine temas eder.
3. Yalnız JSON/HTML raporu yüklemek. Hızlıdır ama vektörler tekrar kullanılamaz;
   her ölçümde yeniden model çalıştırmak gerekir.

## Artefakt sözleşmesi

Her model, Dataset kökünde kendi dizinine yüklenir:

```
artifacts/<model-key>/
  embeddings.npy
  dense.json
```

`dense.json` şunları zorunlu taşır:

- şema sürümü ve model `repo` / sabit `revision`;
- dense kodlama protokolü kimliği (sorgu talimatı, max length, normalizasyon ve
  sayfa-kodlama biçimini kapsayan fingerprint);
- kaynak indeks Dataset repo kimliği ile 40 karakterli kaynak commit SHA'sı;
- sıralı `page_id` listesinin SHA-256'sı ve `page_texts.parquet` SHA-256'sı;
- `embeddings.npy` SHA-256'sı, satır sayısı, boyut ve `float32` dtype;
- artefaktı üreten bu depo Git commit'i.

`embeddings.npy` tam yazılmadan veya hash'i doğrulanmadan Hub'a yüklenmez.
Yerel indirici, sabit bir Hub commit SHA'sını yeniden çözer; manifest, dosya
hash'i, şekil, dtype ve beklenen sayfa kimliği uyuşmadıkça artefaktı görünür
hedef dizine atomik olarak taşımaz.

## Colab akışı

Notebook kaynak depoyu kullanıcı tarafından verilen sabit Git commit'inde
çalıştırır ve üretim Dataset'indeki yalnız gerekli sayfa metnini sabit kaynak
commit'inden indirir. Aynı `TransformerDenseEncoder` ve mevcut sayfa sırasını
kullanır; model kodunu notebook içinde kopyalamaz. Modeller teker teker
işlenir; böylece iki model aynı anda GPU belleğini kullanmaz.

Notebook GPU ön kontrolü yapar. 8B embedding modeli için 24 GB'ın altındaki
GPU belleği açık bir uyumsuzluk hatasıdır; model gizlice küçültülmez,
nicemlenmez veya başka revision'a düşürülmez. 4B ve 8B sonuçları hangi GPU,
kod ve kaynak indeksle üretildiklerini manifestte kaydeder. Colab oturumu
kesilirse yerel/Drive kontrol noktasıyla aynı modele devam edilir; yalnız
tamamlanan model dizini Hub'a yüklenir.

Hugging Face belirteci notebook dosyasında tutulmaz. Colab Secrets'tan
`HF_TOKEN` okunur; token yalnız Dataset oluşturma/yükleme çağrılarında kullanılır.

## Yerel indirme ve değerlendirme

Yeni açık bir çekme komutu `--repo`, `--revision`, `--model` ve hedef dense
artefakt dizinini ister. Revision için dal adı kabul edilmez: 40 karakterli
değişmez Hub commit SHA zorunludur. İndirici geçici dizine alır, doğrular ve
ancak başarıda ilgili model dizinini atomik olarak değiştirir. Hatalı veya
yarım indirme mevcut başarılı artefaktı bozmaz.

`eval_semantic_coverage.py`, dense artefaktı kullanmadan önce aynı manifest
doğrulamasını çağırır. Böylece Colab'da üretilen sayfa sırası veya model
revisionı yerel indeksle farklıysa ölçüm durur; yanlış hizalı vektörlerle
başarım sayısı üretilmez.

Colab yalnız artefakt üretir. Getirim ölçümü, genişletme deneyi ve HTML raporu
bu repoda, açıkça indirilen artefaktlarla çalışmaya devam eder.

## Hata davranışı

- Eksik/yanlış Hub revisionı, eksik dosya, SHA-256, dtype, şekil, model veya
  kaynak sayfa kimliği uyuşmazlığı: indirme başarısız olur ve mevcut hedef kalır.
- Yetersiz GPU belleği: notebook modeli `skipped_oom` diye raporlar; başka
  model veya düşük doğruluklu bir mod seçmez.
- Colab kesintisi: sadece kontrol noktası sürer; tamamlanmamış dizin Hub
  artefaktı sayılmaz.
- Hub yükleme başarısızlığı: tamamlanmış yerel/Drive artefaktı saklanır; yeni
  commit SHA oluşmadan yerel indirme komutu çalıştırılamaz.

## Doğrulama

Davranış değişiklikleri TDD ile test edilir. Birim testleri manifestin geçerli
olduğu durumu ve yanlış SHA, sayfa kimliği, model revisionı, şekil/dtype ve
yarım indirme durumlarını kapsar; atomik geri alma doğrulanır. API çağrıları
mock'lanır; testler ağ veya gerçek model gerektirmez.

Notebook, başlık hücresinde sabit kod commit'ini, kaynak indeks commit'ini ve
oluşan artefakt Hub commit SHA'sını yazar. Gerçek GPU koşumu sonunda yerelde
indirilen artefaktla `eval_semantic_coverage.py` çalıştırılır; sonuç JSON ve
HTML raporunda artefakt kimliği kaydedilir.

## Kapsam dışı

- Üretim indeks Dataset deposunu değiştirmek.
- Online sıralama, cevap üretimi veya çekimserlik eşiğini değiştirmek.
- Ölçüm geliştirme verisiyle bir üretim eşiği kalibre etmek.
