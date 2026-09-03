# Geç kanal çekimserlik kalibrasyonu — sonuç

Tarih: 2026-09-03  
Karar: **BLOCKED — ColBERT skoru cevap kapısına bağlanmadı**

## Kısa hüküm

ColBERT MaxSim toplamına ham eşik koymak geçersiz çıktı; skor neredeyse sorgu
uzunluğunu ölçüyor. Etkin sorgu tokenı başına normalizasyon bu yanlılığı büyük
ölçüde kaldırdı ve sıralama kalitesi taşıyan bir güven sinyali üretti. Buna
rağmen ayrı geliştirme kalibrasyon yakasında seçilen eşik, hukuk-gruplu kilitli
test yakasına taşınmadı: testte **hiçbir soru eşiği geçmedi**.

Bu sonuç güvenli ama yararsız bir kapıdır. Cevaplanamazlarda sıfır yanlış kabul,
tam çekimserlikten geliyor; paraphrase getirim kazancını ürüne taşıyamıyor.
Mevcut `require_calibrated_late_channel` korkuluğu bu yüzden yerindedir:
ColBERT skoru cevap kapısını belirleyecekse kanal açılmaz.

## Sevk edilen yol (ayrı karar)

2026-09-03'te ölçülmüş konfigürasyon, **BM25 sayfa sırası birincil ve top-1
sabit** kalacak biçimde aday-örme olarak sevk edildi. Mogan ve Colmm ColBERT
kanalları yalnız BM25 listesindeki sayfaları birleştirir; `PageHit.score` ve
10,6 çekimserlik eşiği BM25 ölçeğinde kalır. Bu yüzden bu aktivasyon yeni bir
ColBERT güven eşiği gerektirmez ve ölçülen sonuçları taşır: insan-doğrulanmış
n=47'de R@5 **0,7766**, R@20 **0,9149**, R@50 **0,9362**; paraphrase n=21'de
R@50 **0,8571**. Önceki yalnız BM25 üretim yolu sırasıyla 0,6277 / 0,7660 /
0,8085 / 0,5714'tü.

Başlangıçta geç indeks yan-dosyaları ile ana `chunks.parquet` eşleşmesi zorunlu
doğrulanır; eksik veya uyumsuz indeks uygulamayı durdurur. Container imajı bu
~1,1 GB artefaktı henüz dağıtmadığı için `Dockerfile` kanalı açıkça kapalı
tutar; artefaktlar mount edilip bayrak açılmadan container yolu değişmez.

## Protokol

- Cevaplanabilir kaynak: `retrieval_eval_v2`, yalnız `answerable=true` ve
  `verification_kind=human`.
- Cevaplanamaz kaynak: `abstention_eval_v1`, yalnız `verification_status=verified`.
- Etiket `safe_to_answer=1`: soru cevaplanabilir ve altın sayfalardan en az biri
  iki ColBERT kanalının BM25 sayfa sırasına ardışık örüldüğü üretim top-5'inde.
- Dış split: dondurulmuş `splits_v1`, hukuk gruplu.
- Dış dev içinde ikinci, deterministik hukuk-gruplu split: model fit'i ile eşik
  seçimi farklı satırlarda yapıldı.
- Özellikler: Mogan ve Colmm için sayfa düzeyi, etkin sorgu tokenı başına top-1
  MaxSim ile top-1/top-2 marjı. BM25 özellikleri modele alınmadı.
- Eşik: iç calibration yakasında seçici risk `<=0,05` altında en yüksek kapsama.
- Test: artefakt değişmeden, `--yes-final-gate` ile bir kez koşuldu.

`abstention_eval_v1` hakkındaki sınır önemlidir: bu set büyük ölçüde model
çapraz-kontrollü veya manifest-yokluğu ile mekanik doğrulanmıştır; insan
doğrulaması değildir. Aşağıdaki istatistiksel aralık etiketlerin bağımsız insan
doğruluğunu kanıtlamaz.

## Ham skor hipotezinin çöküşü

Dev'in tamamındaki tek-değişkenli teşhis:

| Sinyal | güven etiketi AUC | sorgu tokenı korelasyonu |
|---|---:|---:|
| Mogan ham top-1 toplamı | 0,4922 | 0,9697 |
| Mogan token-başına top-1 | **0,8556** | **0,0399** |
| Colmm ham top-1 toplamı | 0,5180 | 0,9429 |
| Colmm token-başına top-1 | 0,7227 | 0,4858 |

Ham MaxSim özellikle Mogan'da rastgele sıralama düzeyinde; sorgu uzunluğu
korelasyonu ise neredeyse birebir. Normalizasyon sıralamayı değiştirmeden bu
karışmayı kaldırdı. Bu nedenle “ColBERT ölçeğinde eşik”, ham toplam üzerinde
tanımlanamaz; en azından etkin token sayısına göre normalize edilmelidir.

## Geliştirme sonucu

Toplam dev: **181** satır; üretim birleşiminin top-5'inde altını bulunan
**20** pozitif, **161** negatif.

| Yaka | toplam | pozitif | negatif |
|---|---:|---:|---:|
| fit | 114 | 10 | 104 |
| calibration | 67 | 10 | 57 |

İç calibration yakasında seçilen eşik:

- `tau = 0,6724892781547004`
- AUC `0,9333`
- kapsama `4/67 = %5,97`
- seçici risk `0/4 = %0`
- güvenli cevap kabulü `4/10 = %40`
- cevaplanamaz yanlış kabul `0/56`; CP95 üst sınır `%5,21`
- seçici risk CP95 üst sınır `%52,71`; istatistiksel güvence `none`

Aynı eşik fit yakasına geri uygulandığında yalnız `1/10` güvenli pozitif geçti.
Bu, eşik test açılmadan önce bile zayıf taşınabilirlik işaretiydi.

## Kilitli test sonucu

Test: **175** satır; 17 güvenli pozitif, 6 cevaplanabilir retrieval ıskası ve
152 cevaplanamaz soru.

| Metrik | Sonuç |
|---|---:|
| AUC | 0,8466 |
| Brier | 0,0604 |
| ECE | 0,0385 |
| kapsama | **0/175 = %0** |
| güvenli cevap kabulü | **0/17 = %0** |
| cevaplanamaz yanlış kabul | 0/152 = %0 |
| cevaplanamaz CP95 üst sınır | %1,95 |
| en yüksek test olasılığı | 0,664569 |
| eşik | 0,672489 |

Testin en yüksek olasılığı bile kalibrasyon eşiğinin altında kaldı. Dolayısıyla
seçici risk tanımsızdır: cevaplanan örnek yoktur.

## Açılma kapısı

| Koşul | Sonuç |
|---|---|
| seçici risk nokta tahmini `<= %5` | **FAIL** — cevap yok, tanımsız |
| cevaplanamaz yanlış kabul `<= %2` | PASS — 0/152 |
| cevaplanamaz CP95 üst sınır `<= %5` | PASS — %1,95 |
| güvenli cevap kabulü `>= %80` | **FAIL** — 0/17 |
| artefakt/indeks/reçete kimliği eşleşiyor | PASS |

Artefakt anahtarı:
`late-channel-v1__d9f6e2179917291d`.

Artefakt ana görsel indeks revizyonunu, değişmemiş BM25 reçete parmak izini,
iki ColBERT model revizyonunu ve her geç indeksin `colbert.json`,
`chunk_ids.json`, `offsets.npy`, `embs.npy` SHA-256 değerlerini taşır. Aynı fit
koşumu ikinci kez çalıştırıldı; zaman/yol alanları çıkarıldığında raporlar ve
artefakt bayt düzeyinde aynı çıktı.

## Yorum ve sonraki deney sınırı

Normalize skorun AUC'si yararlı bir sıralama sinyali olduğunu gösteriyor;
başarısız olan şey mutlak çalışma noktasının gruplar arası taşınması. Fit'te 10,
eşik yakasında 10 güvenli pozitif olması, dört özellikli olasılık ölçeği için
çok küçük. Bu veriyle eşiği gevşetmek, kilitli teste bakarak ayar yapmak olur ve
ölçüm aracını deney nesnesine çevirir.

Bu test yakası artık görülmüştür; sonraki bir `v2` kalibrasyonunu doğrulamak için
yeni ve kilitli bir holdout gerekir. Makul devam yolu daha fazla insan-onaylı
cevaplanabilir soru toplamak, fit/calibration pozitiflerini büyütmek ve yeni
holdout üzerinde tek seferlik kapıyı yeniden kurmaktır. BM25 özelliklerini veya
uzunluk-kovalı eşikleri bu test sonucuna göre eklemek bu koşumda yasaktır.

## Yeniden oynatma

Geliştirme raporu:
`data/bench/results/late-channel-calibration-dev-v1.json`

Kilitli test raporu:
`data/bench/results/late-channel-calibration-test-v1.json`

Koşucu:

```bash
/Users/barandincoguz/Desktop/project-delta/.venv/bin/python \
  scripts/calibrate_late_channel.py fit \
  --index-dir /Users/barandincoguz/Desktop/project-delta/data/index-traincompat-int8 \
  --late-index /Users/barandincoguz/Desktop/project-delta/data/index-colbert-mogan-f16 \
  --late-index /Users/barandincoguz/Desktop/project-delta/data/index-colbert-colmm-f16 \
  --artifact-dir data/calibration/late-channel-v1 \
  --out data/bench/results/late-channel-calibration-dev-v1.json
```

`eval` aynı argümanlarla ve açık `--yes-final-gate` ile çalışır. Bu rapordaki
test koşumu tek sefer yapılmıştır; yeniden koşmak yeni kanıt üretmez.
