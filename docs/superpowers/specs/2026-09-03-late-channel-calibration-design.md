# Geç Kanal Çekimserlik Kalibrasyonu — Tasarım

## Amaç

BM25 sayfa sıralamasını ve `recipe_fingerprint()` sözleşmesini değiştirmeden,
Mogan-ColBERT-TR ile ColmmBERT-small-TR adaylarının birleştiği üretim yolunda
"getirilen ilk beş sayfada yanıtın dayanağı var mı?" olasılığını kalibre etmek.
Deney yalnız geliştirme verisinde eşik seçer; kilitli test bölmesini bir kez
değerlendirir ve geç kanalın açılmaya uygun olup olmadığını makine-okur bir
kararla kaydeder.

## İncelenen yaklaşımlar

### 1. Ham MaxSim eşiği

Her ColBERT kanalının en yüksek MaxSim toplamı doğrudan eşiklenir. En küçük
değişiklik budur fakat toplam, etkin sorgu tokenı sayısıyla ölçeklenir. Ön
tasarım ölçümünde Mogan ham top-1 skoru ile sorgu tokenı sayısı arasındaki
korelasyon `0,965`, güven etiketi AUC'si `0,492` bulundu. Bu kol yalnız negatif
kontrol olarak raporlanacak, artefakt üretmeyecek.

### 2. Tek kanal, token başına MaxSim

Mogan top-1 toplamı etkin sorgu tokenı sayısına bölünür. Sıralama değişmez;
yalnız çekimserlik ekseni sorgu uzunluğundan ayrılır. Ön ölçümde korelasyon
`0,023`, AUC `0,856` oldu. Sağlam bir taban ve yorumlanabilir bir teşhis
koludur, ancak ikinci kanalın farklı isabetlerini kullanmaz.

### 3. Normalize iki-kanallı kalibratör — seçilen

Her kanal için sayfa düzeyindeki normalize top-1 ve top-1/top-2 marjı çıkarılır:

1. `mogan_top1_mean`
2. `mogan_margin_mean`
3. `colmm_top1_mean`
4. `colmm_margin_mean`

Bu dört özellik mevcut saf-NumPy lojistik kalibratöre verilir. BM25 özellikleri
bilerek eklenmez: deney geç kanalın güvenini ölçer, donmuş BM25 reçetesine yeni
bir kalibrasyon bağımlılığı kurmaz. Ön ölçümde iki-kanallı normalize model AUC
`0,868` verdi; BM25 özelliklerini de eklemek AUC'yi `0,930`a taşısa da aynı
%5 geliştirme riski noktasında kapsamayı artırmadı. Eşit sonuçta daha küçük ve
daha taşınabilir model seçilir.

## Veri ve sızıntı sınırı

- Cevaplanabilir sınıf: `data/bench/retrieval_eval_v2.jsonl` içindeki
  `verification_kind == "human"` ve `answerable == true` satırları.
- Cevaplanamaz sınıf: `data/bench/abstention_eval_v1.jsonl` içindeki
  `verification_status == "verified"` satırları. Bu setin büyük kısmı insan
  onaylı değildir; rapor bunu istatistiksel güvenceyle karıştırmayacak.
- Etiket: `1` yalnız soru cevaplanabilir ve altın sayfalardan en az biri üretim
  birleşiminin top-5'inde ise; diğer bütün durumlar `0`.
- Dış bölme: mevcut `splits_v1.json` ve `assign_split()`. Test belgeleri eşik
  veya özellik seçimine hiçbir biçimde girmez.
- İç geliştirme bölmesi: hukuk grubu anahtarının
  `sha256("late-calibration-v1:" + group_key)` özetiyle deterministik
  `fit`/`calibration` ayrımı. Aynı kanun iki iç yakaya sızamaz.
- Test değerlendirmesi açık `--yes-final-gate` onayı olmadan çalışmaz.

## Üretimle aynı özellik yolu

`LateInteractionChannel.search_with_scores()` tek sorgu kodlamasından şunları
döndürür:

- chunk sırası ve bundan türeyen tekrarsız sayfa sırası,
- etkin sorgu tokenı sayısı,
- sayfa düzeyinde top-1/top-2 ham skor,
- token başına normalize top-1 ve marj.

`candidate_pages()` bu sonucu kullanır. Kalibrasyon betiği aynı üretim metodunu
çağırır; skorlama formülünü yeniden yazmaz. Genişletme tokenları kodlamada
korunur fakat `encode_query_vectors()` sözleşmesi gereği MaxSim toplamına
katılmaz.

## Fit, eşik ve rapor

Kalibratör yalnız iç `fit` yakasında eğitilir. `tau`, ayrı iç `calibration`
yakasında `risk <= 0,05` koşulu altında en yüksek kapsamayı veren nokta olarak
seçilir. Artefakt şu kimlikleri taşır:

- ana görsel indeks revizyonu,
- değişmemiş BM25 `recipe_fingerprint`,
- iki ColBERT model repo/revizyonu ve yan dosya SHA-256'ları,
- `late-score-v1:content-token-mean+page-max+sequential-interleave` reçete kimliği,
- benchmark ve split dosyalarının SHA-256/git-blob kimlikleri,
- fit/calibration sayımları ve seçilen eşikte belirsizlik alanları.

Rapor ham ve normalize tek-değişkenli AUC'leri, sorgu uzunluğu korelasyonlarını,
risk-kapsama eğrisini, soru-bazlı özellikleri ve karar gerekçesini içerir.

## Kilitli test ve açılma kararı

Test koşumu artefaktı değiştirmez; aynı kalibratör ve `tau` ile sonuç üretir.
`eligible_to_enable` yalnız aşağıdakilerin tamamı sağlanırsa `true` olur:

1. test seçici riski nokta tahmini `<= 0,05`,
2. cevaplanamazlarda yanlış-kabul nokta oranı `<= 0,02`,
3. cevaplanamaz yanlış-kabul oranının %95 Clopper-Pearson üst sınırı `<= 0,05`,
4. top-5'te altını bulunan cevaplanabilir soruların en az `%80`i eşikten geçer,
5. artefakt/veri/reçete kimliklerinin tümü eşleşir.

Bu koşullardan biri bile başarısızsa deney tamamlanmış sayılır fakat kanal
açılmaz. Düşük kapsamayı "güvenli" diye sevk etmek yasaktır.

## Hata davranışı

- Eksik/geçersiz indeks yan dosyası, yinelenen chunk kimliği, boş sınıf veya
  kimlik uyuşmazlığı fail-fast eder.
- NaN/sonsuz özellik artefakta giremez.
- Test kapısı örtük çalışmaz.
- Rapor yazımı atomiktir; yarım JSON geçerli sonuç gibi kalmaz.
- `graphify-out/` ve diğer kullanıcı değişiklikleri kapsam dışıdır.

## Test stratejisi

- Saf birim testleri: sayfa düzeyi skor özeti, token normalizasyonu, tek sorgu
  kodlama, deterministik hukuk-gruplu iç bölme, artefakt round-trip ve kimlik
  uyuşmazlığı.
- CLI testleri: fit raporu/artefaktı, test kapısı reddi ve sentetik başarılı
  final-gate yolu; gerçek model veya ağ kullanılmaz.
- Gerçek deney: iki yerel fp16 indeks ve üretim sınıflarıyla dev fit/calibration,
  ardından kilitli test koşumu.
- Son doğrulama: hedefli testler, tüm pytest paketi, Ruff ve Pyright.

