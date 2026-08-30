# `unans_v1.jsonl` — köken ve doğrulama künyesi

**Bu set insan-doğrulanmış DEĞİLDİR. 300 satırın 0'ı insan onayından geçmiştir.**
Sorular bir model ajanı tarafından yazıldı; hiçbiri bir insan tarafından
okunup onaylanmadı. Bu yüzden 300 satırın tamamı
`source_type: "ajan-taslak"` taşır — mevcut `ajan-taslak-insan-onayli`
değerinden bilerek ayrı bir değer, aksi halde onaysız satırlar künyede
onaylı gibi sayılırdı.

`canary_v1.jsonl` için yazılan dürüstlük rejimi burada da geçerlidir
(bkz. `canary_v1.README.md`): `verification_status: "verified"` gören biri
bunu insan onayı sanmasın diye, ne tür bir doğrulamadan geçtiği
`verification_kind` alanında ayrıca kayıtlıdır.

Künye, **2026-08-30 çapraz-kontrol turundan sonraki** hâliyle (bkz. §3):

| Dilim | Satır | `verification_status` | `verification_kind` | `verified_by` |
|---|---|---|---|---|
| `korpus-disi` | 200 | 195 `verified` / 5 `rejected` | `mechanical:manifest-absence` (+40'ı çapraz-kontrollü) | `script:validate_unans` |
| `anlamsiz-ood` | 60 | 60 `verified` | `model-cross-check` | `model-cross-check:claude-fable-5-checker` |
| `eksik-kanit` | 40 | 31 `verified` / 9 `rejected` | `model-cross-check` | `model-cross-check:claude-fable-5-checker` |
| **toplam** | **300** | 286 verified / **14 rejected** | | |

`rejected` satırlar dosyadan SİLİNMEDİ; tüketiciler onları
`verification_status` üzerinden dışlar (`load_bench(..., only_verified=True)`
zaten dışlıyor). Silinmemelerinin nedeni, reddedilme gerekçelerinin
(§3) etiket-gürültüsü ölçümünün kanıtı olması.

## 1. Neden bu set var

Kalibre edilmiş seçici cevaplama (bir soruya cevap vermek yerine
"bilmiyorum" demeyi öğrenmek) ölçülebilmesi için cevaplanamaz veri ister.
Canary'de yalnız 5 cevaplanamaz soru vardı — bu sayıyla ölçülen bir hata
oranının güven aralığı işe yaramayacak kadar geniştir.

Hedef sayı aritmetikten gelir: test kümesinde **n=150** cevaplanamaz soruda
**0 hata** gözlenirse, Clopper-Pearson %95 üst sınırı ≈ **%2.0**'dir. G2.1
kapısının istediği eşik budur; 150 bu eşiğin **aritmetik asgarisidir**, keyfi
bir yuvarlama değil. 300 üretilip ~yarısı test'e düştüğünde bu sayı sağlanır
(fiilî: **test'te 151 cevaplanamaz**, bkz. §5; çapraz-kontrol sonrası 144, bkz. §3).

## 2. Dilimler ve her birinin ne kadarına güvenilebilir

### `korpus-disi` (200) — mekanik doğrulanmış
Her soru, korpusta **bulunmayan** gerçek bir Türk kanununa çapalanmıştır
(117 farklı kanun). Satırlar `_anchor_law` (kanun numarası) ve
`_anchor_name` (kanun adı) alt çizgili alanlarını taşır; alt çizgili alanlar
`BenchQuestion` tarafından yok sayılır, yani yükleyiciyi kirletmezler.

**Mekanik etiketin İDDİA ETTİĞİ tam olarak şudur:** sorunun dayandığı kanun,
korpus manifestinde (`data/state.json` ∩ `data/manifest/v0_manifest.csv`,
56 belge / 50 kanun) **yoktur**. `scripts/validate_unans.py` bunu her koşumda
yeniden türeterek sınar — betik ezberden kanun listesi taşımaz.

**İDDİA ETMEDİĞİ şey:** sorunun cevaplanamaz olduğu. Hiç kimse (insan ya da
model) korpusta cevap **aramamıştır**.

> **Bilinen artık risk.** Korpusta olmayan bir kanuna dayanan soru, korpustaki
> BAŞKA bir kanundan cevaplanabilir olabilir. Örneğin "Gümrük Kanunu'na göre
> tebliğ zamanaşımı" sorusu, Vergi Usul Kanunu'nun genel zamanaşımı hükmüyle
> yanlışlıkla cevaplanmış sayılabilir.
>
> **Azaltma (üretim kuralı):** her sorunun CEVAP ÖZÜ o absent kanuna hastır —
> yalnız o kanunda geçen bir süre, tutar, usul ya da şart sorulur; genel hukuk
> bilgisiyle karşılanabilecek sorular elenmiştir. Üretim sırasında bu gerekçeyle
> düşürülen somut örnekler: dernek kurmak için gereken kişi sayısı (TMK m.56
> korpusta), vakıf kuruluşunda asgari mal varlığı (TMK m.101+), terör
> suçlarında gözaltı süresi (CMK korpusta), abonelik sözleşmesinin feshi
> (6502 m.52 korpusta), kamu görevlilerinde yetkili sendika tespiti (6356
> korpusta).
>
> **Kalan riskin büyüklüğü ÖLÇÜLDÜ (2026-08-30, §3.2):** 40 satırlık
> deterministik örneklemde **5 hit = %12.5**, Wilson %95 [%5.5, %26.1] —
> 200 satıra ölçeklenirse ~25 satır [11, 52]. Örneklenen 5 hatalı satır
> `rejected` yapıldı; örneklem DIŞINDA kalan 160 satır hâlâ yalnız
> mekanik etiketlidir ve tahmini ~%12'lik bir gürültü taşır. Bu dilim
> "manifest-yokluk doğrulanmış" diye anılmaya devam etmelidir;
> "cevaplanamazlığı doğrulanmış" DEĞİLDİR.

### `anlamsiz-ood` (60) — taslak
Zırva, kategori hatası (`"Zamanaşımı süresi kaç santimetredir?"`), hukuk-dışı
soru, hukuk kelimesi serpiştirilmiş alan-ötesi saçmalık, uydurma kanun
numarası ve talimat enjeksiyonu denemeleri. Bu dilimi soruları yazan tur
doğrulayamazdı, çünkü aynı turun "bu anlamsızdır" kararı bağımsız bir ölçüm
değildir. **2026-08-30'da bağımsız bir denetleyici turu 60 satırın tamamını
okudu ve tamamını onayladı** (§3.4); dilim artık `verified` /
`model-cross-check`'tir — insan onayı DEĞİL.

### `eksik-kanit` (40) — taslak, sınırdaki sınıf
Sorular **korpustaki** bir kanun hakkındadır ama aranan somut ayrıntı korpus
metninde yoktur: yönetmeliğe devredilmiş bir usul, kanunun yazmadığı bir
tutar, ya da mülga bir hükme yapılan atıf. Satırlar `_subject_doc` alanını
taşır (20'si test, 20'si dev belgelerinden).

Bu, kalibrasyonun **en zor** durumudur: retrieval doğru belgeyi getirir, model
"kanıt var" sanır ve uydurma yapmaya en yatkın olduğu yer burasıdır.

Her satırın `verification_note` alanı **gerçek bir yokluk kanıtı** taşır:
`data/research/page_texts.parquet` üzerinde o belgenin tüm sayfalarında
yapılan grep sayımları. Bu sayımlar elle yazılmadı — üretici betik onları
ölçtü ve sıfır olmayan bir "yok" iddiası bulursa üretimi **durdurdu** (bir
kez durdu: `k197` için "1600" terimi metinde geçiyordu, satır yeniden
yazıldı). Örnek not:

```
k4857 (55 sayfa) tam metninde 'kıdem tavan'=0, 'kıdem tazminatı tavan'=0,
'tavanını aşamaz'=0; buna karşılık 'kıdem tazminatı'=25, '1475'=8 —
m.120 kıdem tazminatını mülga 1475 sayılı Kanunun 14 üncü maddesine
devreder; tavan rakamı metinde yok.
```

Grep yokluğu, "sayfa görüntüsünde de yok" demek DEĞİLDİR (metin çıkarma
kusurlu olabilir). Bu satırlar ayrıca ileride PPI (prediction-powered
inference) çiftleri olarak kullanılmak üzere saklanmaktadır.

> **Bu yöntem 40 satırın 9'unda YANILDI.** Çapraz-kontrol (§3.3) grep
> desenlerinin kanun metninin ifadesini ıskaladığı 9 satır buldu: `u262`,
> `u264`, `u268`, `u276`, `u282`, `u283`, `u284`, `u291`, `u299` artık
> `rejected`. Kalan 31 satırın notu grep sayımı yerine **hükmün birebir
> alıntısıyla** yeniden yazıldı.

## 3. Çapraz-kontrol (2026-08-30) — bağımsız denetleyici turu

Setin taslak dilimlerini **yazan turdan farklı** bir model turu (drafter ≠
checker; canary'nin model-çapraz-kontrol turuyla aynı rejim, bkz.
`canary_v1.README.md`) 140 satırı tek tek `page_texts.parquet` üzerinde
denetledi. Denetleyicinin tek kanıt aracı korpus metniydi: ağ yok, Gemini yok,
ezber yok; her `rejected` kararı bir `page_id` + birebir alıntıyla satırın
`verification_note` alanında duruyor.

### 3.1 Dilim başına sonuç

| Dilim | Denetlenen | uygun | düzelt | **reddedildi** | Red oranı |
|---|---|---|---|---|---|
| `anlamsiz-ood` | 60 / 60 (tamamı) | 60 | 0 | 0 | %0 |
| `eksik-kanit` | 40 / 40 (tamamı) | 31 | 0 | **9** | **%22.5** |
| `korpus-disi` | 40 / 200 (örneklem) | 35 | 0 | **5** | **%12.5** |

Sıfır "düzelt": denetleyici doğru etiketlenmiş bir satırı yeniden yazmayı
uygun bulmadı; yanlış olanlar düzeltilmek yerine reddedildi (etiket
gürültüsünü yeniden yazarak gizlemek yerine kayda geçirmek).

### 3.2 `korpus-disi` artık riskinin niceliği (§2'deki açık soru)

Örneklem **deterministik**: her 5. satır (`u005, u010, …, u200`), n=40.
Ölçülen şey mekanik etiketin kendisi değil — o zaten doğru — **artık risk**:
*çapa kanun korpusta yok ama sorunun cevap özü korpustaki BAŞKA bir kanundan
üretilebiliyor mu?*

- Gözlenen: **5 / 40 = %12.5**
- Wilson %95 iki yanlı aralık: **[%5.5, %26.1]**
- 200 satıra ölçeklenirse: **~25 satır [11, 52]**

Yani §2'nin "kalan riskin büyüklüğü ölçülmemiştir" uyarısının cevabı:
*ölçüldü, sıfır değil, ~%12* — ve aralık geniş, çünkü n=40. Duyarlılık:
5 hitten 2'si sınırda (`u110`, `u140`); yalnız net 3'ü sayarsak %7.5
[%2.6, %19.9], 4'ünü sayarsak %10.0 [%4.0, %23.1].

Reddedilen 5 satır ve bulunan cevap:

| id | Soru çapası | Cevabı taşıyan korpus hükmü |
|---|---|---|
| `u015` | 7201 Tebligat K. | `k213:31` VUK m.94 — "ikametgah adresinde bulunanlardan veya işyerlerinde memur ya da müstahdemlerinden birine … 18 yaşından aşağı olmaması" |
| `u110` | 4915 Kara Avcılığı K. | `k492:64` Harçlar K. (9) sayılı tarife — avcılık belgeleri "(her yıl için)" |
| `u130` | 6802 Gider Vergileri K. | `k213:72` VUK m.204 — "banka …, banker ve sigorta şirketleri banka ve sigorta muameleleri vergisinin mevzuuna giren işlemleri …" |
| `u135` | 6085 Sayıştay K. | `k5018:40-41` 5018 m.68 — dış denetim türleri (malî/uygunluk + performans) sayılı |
| `u140` | 3071 Dilekçe Hakkı K. | `k2577:4` İYUK m.10/2 — "Otuz gün içinde bir cevap verilmezse istek reddedilmiş sayılır" |

### 3.3 `eksik-kanit`: 40'ta 9 yanlış etiket

Bu dilimin taslak notları grep sayımlarına dayanıyordu ve **grep deseni ile
kanun metninin ifadesi tutmadığında yokluk yanlış çıkıyor**. İki sistematik
kusur:

1. **Konsolide metne işlenmiş güncel tutarlar.** mevzuat.gov.tr metinleri
   kanunî tutarın yanına parantez içinde *uygulanan* (tebliğ/CB kararıyla
   güncellenmiş, çoğu 1/1/2026'dan geçerli) tutarı da taşıyor. "2026 tutarı
   kanunda yazmaz" varsayımı bu yüzden çöktü: `u282` (VUK 1 sayılı cetvel,
   "Sermaye şirketleri … 20.000 **(35.000)** TL"), `u283` (GVK m.103,
   "18.000 TL'ye **(190.000 TL)** kadar"), `u284` (Harçlar (6) sayılı tarife,
   pasaport "**(2.806,50 TL.)**"), `u276` (6183 m.51, "%4 **(%3,7)**"),
   `u268` (Kamu İhale K. metin sonundaki karşılaştırmalı tablo,
   "1/2/2025–31/1/2026 döneminde uygulanacak eşik değerler … 14.673.866").
2. **Kanunun tutarı formülle tanımlaması.** "Tutar yönetmeliğe bırakılmıştır"
   sanılan yerde kanun somut bir gösterge çarpımı veriyor: `u264`
   (2828 ek m.7, "(10.000) gösterge rakamı ile memur aylık katsayısının
   çarpımı"), `u262` (Koop. K. m.87, "(1200) gösterge … geçmemek üzere").
   Ayrıca `u291` (MTV m.5 kasko değerlerini kimin ilan ettiğini yazıyor) ve
   `u299` (İYUK m.31 → HMK m.87, "teminatın tutarını … hâkim serbestçe tayin
   eder") aynı şekilde cevaplanabilir çıktı.

En öğretici örnek `u284`: not "kanun yalnız kanunî miktarları taşır" diyordu;
`k492:55` ise "Uygulanan Miktar" sütununu ve 31/12/2025 tarihli 98 Seri No.'lu
Genel Tebliğe yapılan dipnotu taşıyor — yani soruda istenen 2026 pasaport
harcı doğrudan metinde.

Kalan 31 satırın notu yeniden yazıldı: grep sayıları yerine **hükmün kendisi**
alıntılanıyor (ör. `u265`: KTK m.50 şehirlerarası 90 / bölünmüş 110 / otoyol
120 km/s **verir**, sorulan şehir içi değerini vermez — taslağın "kanun
sayısal değer vermez" gerekçesi yanlıştı, sonucu doğruydu).

### 3.4 `anlamsiz-ood`: 60/60 uygun

Her satır için "bunun kazara bir hukuki okuması var mı ve o okuma korpustan
cevaplanır mı?" sorusu ayrıca sınandı. Üç bilinçli tuzak korunmuştur ve
notlarında işaretlidir: `u244` (Atlantis Medeni Kanunu ↔ `k4721:25` TMK m.124
gerçek evlenme yaşını taşır), `u203` (birim yok sayılırsa TBK m.146'nın
"on yıllık zamanaşımı"na kayma riski), `u224` (KMK m.34 yönetici değişikliği).
Üçünde de doğru davranış çekimserliktir; etiket ayakta. Dilimin en zayıf
üyesi `u253`: kategori hatası değil, kurgusal bir meta istem — yine de
korpustan cevaplanamaz olduğu için tutuldu.

### 3.5 Yöntem

- Kanıt: `data/research/page_texts.parquet` (4222 sayfa), `page_id` bazında
  tam metin. Arama tr-duyarlı küçültme + aksan düzleştirme (`şğüöçı→sguoci`)
  ve düzenli ifade ile yapıldı; her iddia için birden çok ifade, eş anlamlı ve
  yazım varyantı denendi (ör. `tellaliye` **ve** `dellâliye`).
- `eksik-kanit`'te yalnız `_subject_doc` değil, cevabı taşıyabilecek diğer
  korpus kanunları da tarandı (`u274` için 3095'e atıf yapan 5 kanun,
  `u296` için Harçlar Kanunu ve RG taramaları, `u299` için HMK).
- Karar kuralı: **şüphede reddet.** Yanlış red yalnız n'i küçültür; yanlış
  kabul G2.1 metriğini bozar.

### 3.6 Bu turun sonuçları — iki tanesi can sıkıcı

**(a) Test kümesindeki cevaplanamaz sayısı n=150 asgarisinin ALTINA düştü.**
14 reddin 7'si test yakasına düşüyor. Yeni bileşim (canary dahil,
`rejected` hariç):

| | cevaplanamaz | canary cevaplanabilir |
|---|---|---|
| **dev** | 147 (101 korpus-dışı + 29 anlamsız + 15 eksik-kanıt + 2 canary) | 26 |
| **test** | **144** (94 korpus-dışı + 31 anlamsız + 16 eksik-kanıt + 3 canary) | **17** |

n=144'te 0 hata için Clopper-Pearson %95 üst sınırı **%2.06**'dır; §1'in
istediği %2.0 eşiği artık **karşılanmıyor** (gereken asgari n = 149). Yani
G2.1 kapısı ya 5+ yeni cevaplanamaz test sorusuyla beslenmeli ya da eşik
gerekçesi yeniden yazılmalıdır. Bu, çapraz-kontrolün *yarattığı* bir sorun
değil; zaten var olan ve şimdi görünür hâle gelen bir sorundur.

**(b) `scripts/validate_unans.py` şu an KIRMIZI (105 ihlal) — ve bu beklenen
bir kırmızı.** Betiğin 7 numaralı kontrolü dilim başına doğrulama künyesini
sabit kodluyor:

```python
VERIF_EXPECT = {
    "korpus-disi": ("verified", "script:validate_unans", "mechanical:manifest-absence"),
    "anlamsiz-ood": ("draft", "", "model-cross-check"),  # <- çapraz-kontrol ÖNCESİ
    "eksik-kanit": ("draft", "", "model-cross-check"),  # <- çapraz-kontrol ÖNCESİ
}
```

Çapraz-kontrol turu tam da bu `draft` durumunu kaldırmak için koştuğundan,
100 satır artık beklenen üçlüyü taşımıyor; ayrıca 5 `rejected` korpus-dışı
satır da künye kontrolüne takılıyor (105 = 60 + 40 + 5). **İhlallerin
tamamı bu tek kontroldendir**; şema, kimlik, dilim sayıları,
cevaplanamazlık değişmezleri, çapa yokluğu, `_subject_doc` varlığı,
yakın-tekrar ve split türetimi kontrollerinin hepsi temiz geçiyor
(bellekte beklenti güncellenip `rejected` satırlar muaf tutulduğunda kalan
ihlal sayısı **0**).

Denetleyici betiği bilerek DEĞİŞTİRMEDİ (kendi dosyası değil). Sahibinin
yapması gereken düzeltme, `VERIF_EXPECT`'i çapraz-kontrol sonrası duruma
güncellemek ve `rejected` satırları künye kontrolünden muaf tutmaktır.

### 3.7 Bu turun sınırları (dürüstlük kaydı)

- **Denetleyici bir modeldir, insan değildir.** Bu tur `verification_kind`
  değerini `human` yapmaz ve hiçbir rakam "insan-doğrulanmış benchmark
  üzerinde ölçüldü" diye sunulamaz.
- **Yazarla kör noktaları ortaktır.** Drafter ve checker aynı model ailesinden;
  ikisinin de aynı yerde yanılma olasılığı bağımsız iki insanınkinden
  yüksektir. Bulunan 14 hata bir alt sınırdır, gerçek sayı değil.
- **Kanıt yalnız metin katmanıdır.** `page_texts.parquet` OCR/metin çıkarma
  ürünüdür; bir hüküm sayfa görüntüsünde olup metne düşmemişse denetleyici de
  göremez. Özellikle RG taramaları (`rg*`) için bu risk yüksektir.
- **Örneklem küçüktür.** `korpus-disi` için n=40; %12.5 tahmininin aralığı
  [%5.5, %26.1] ve bu aralık karar için fazla geniştir. Kalan 160 satır
  denetlenmemiştir — **bireysel satır bazında hâlâ yalnız mekanik
  etiketlidirler**.
- **"Cevaplanabilir" eşiği bir yargı kararıdır.** `u110` ve `u140` gibi
  sınırda vakalarda "cevap korpusta var" demek yorum gerektirir; farklı bir
  denetleyici 3 ile 6 arasında bir sayı bulabilirdi.

## 4. Soru kalitesi kuralları

- Doğal Türkçe, bir vatandaşın ya da avukatın gerçekten soracağı biçimde.
- `query_style` çeşitlemesi: 79 `dogal` / 86 `hukuki` / 35 `madde-referansli`
  (korpus-dışı dilimde). `madde-referansli` sorular gerçekten bir madde
  numarası anar — canary incelemesinde yakalanan etiket hatası tekrarlanmasın
  diye.
- Zorluk dağılımı: 49 kolay / 161 orta / 90 zor.
- Tekrar yok: set içinde ve **canary'ye karşı** normalize edilmiş token kümesi
  örtüşmesi (Jaccard ≥ 0.8) ile taranır. Doğrulayıcı bir kez gerçek bir tekrar
  yakaladı (u108/u109 aynı metne düşmüştü) ve satır yeniden yazıldı.

## 5. Split şeması

`data/bench/splits_v1.json` — **hukuk-gruplu (law-grouped)**: bölme birimi soru
değil BELGEDİR. Bir kanunun tüm sayfaları ve o kanuna bağlı tüm sorular aynı
yakaya düşer; aksi halde test ölçümü dev'de görülmüş bir kanunu geri okur.

- `seed`: `belge-gozu-splits-v1`, 22 test belgesi / 34 dev belgesi.
- 4 belge test'e **sabitlenmiştir** (`k6098`, `k5237`, `k6698`, `rg1935a`);
  toplamları tam 17 canary cevaplanabilir soru eder, yani hedeflenen
  26 dev / 17 test bölünmesini tek seçimle verir.
- Kalan 18 yer, canary sorusu OLMAYAN belgeler arasından
  `sha256(seed|doc_id)` artan sırayla doldurulur; ≥2 RG tarama belgesi
  garantilidir (fiilî: `rg1935a`, `rg1945a`).
- Cevaplanamaz atama kuralı `belge_gozu.bench.dataset.assign_split` içinde saf
  fonksiyon olarak uygulanmıştır ve splits dosyasında da yazılıdır:
  korpus-dışı `sha256("anchor:<kanun no>")` ile kanun-gruplu 50/50; anlamsız
  `sha256("qid:<id>")` ile 50/50; eksik-kanıt `_subject_doc`'un belge
  bölmesini takip eder.

Ortaya çıkan bileşim:

| | cevaplanamaz | canary cevaplanabilir |
|---|---|---|
| **dev** | 154 (103 korpus-dışı + 29 anlamsız + 20 eksik-kanıt + 2 canary) | 26 |
| **test** | **151** (97 korpus-dışı + 31 anlamsız + 20 eksik-kanıt + 3 canary) | **17** |

Bu tablo çapraz-kontrol ÖNCESİ hâldir. 14 satır `rejected` olduktan sonra
test'teki cevaplanamaz sayısı **144**'e düşer ve §1'deki n=150 asgarisi artık
KARŞILANMAZ — bkz. §3.6(a).

## 6. Doğrulamayı yeniden koşmak

```bash
uv run python scripts/validate_unans.py
```

Betik korpus kümesini repodan yeniden türetir, 300 satırın tamamını şemadan
geçirir, çapaların korpusta olmadığını numara VE ad-token bazında sınar,
soruların çapalarını andığını kontrol eder, dilim sayılarını, doğrulama
künyesini ve yakın-tekrarları denetler, split'i künyedeki kuralla yeniden
türetip dosyayla karşılaştırır. İhlalde çıkış kodu 1'dir.

> **Şu an 105 ihlalle KIRMIZI dönüyor ve bu beklenen bir kırmızıdır** — 7
> numaralı kontrolün `VERIF_EXPECT` sabiti çapraz-kontrol öncesi künyeye
> kilitli. İhlallerin tamamı bu tek kontroldendir; nedeni ve düzeltmesi
> §3.6(b)'de. Diğer sekiz kontrol temiz geçiyor.

Yükleme kontrolü:

```bash
uv run python -c "from belge_gozu.bench.dataset import load_bench; \
  print(len(load_bench('data/bench/unans_v1.jsonl', only_verified=False)))"   # 300
uv run python -c "from belge_gozu.bench.dataset import load_bench; \
  print(len(load_bench('data/bench/unans_v1.jsonl')))"                        # 286 (rejected hariç)
```

Saf mantığın testleri: `tests/test_validate_unans.py` (her kontrol için kasıtlı
bozuk bir satır da geçirilir — "TEMİZ" çıktısının bir şey kanıtlaması için) ve
`tests/bench/test_dataset.py` (sözlük genişlemeleri + `assign_split`).

## 7. Bu setin `verified` olması için ne gerekiyor (açık kapılar)

1. ~~**`anlamsiz-ood` (60):** bağımsız bir denetleyici turu her satırı
   okuyup gerçekten anlamsız/alan-dışı olduğunu onaylamalı.~~ **KAPANDI**
   (2026-08-30, §3.4) — 60/60 onaylandı. İnsan onayı hâlâ yok.
2. ~~**`eksik-kanit` (40):** denetleyici~~ + **insan**. Denetleyici turu koştu
   (§3.3) ve 9 satırı reddetti; kalan 31'i `verified`. **İnsan ayağı hâlâ
   açık:** grep/metin yokluğu, sayfa görüntüsünde de bulunmadığını kanıtlamaz.
3. ~~**`korpus-disi` (200):** artık riskin örneklemle nicelenmesi.~~ **KAPANDI**
   (2026-08-30, §3.2) — %12.5 [%5.5, %26.1]. Ama örneklem dışı 160 satır
   birey bazında hâlâ yalnız mekanik etiketlidir; dilimin künyesi
   `mechanical:manifest-absence` olarak kalır.
4. **YENİ — n=150 açığı:** test kümesinde `rejected` sonrası 144 cevaplanamaz
   soru kaldı; G2.1'in %2.0 eşiği için en az 149 gerekiyor (§3.6a). Ya 5+ yeni
   test-yakası cevaplanamaz soru üretilmeli ya da eşik gerekçesi
   yeniden yazılmalı.
5. **YENİ — doğrulayıcı künye beklentisi:** `scripts/validate_unans.py`
   içindeki `VERIF_EXPECT` çapraz-kontrol öncesi duruma sabitlenmiş; 105
   ihlalin tamamı bundan kaynaklanıyor (§3.6b). Betiğin sahibi güncellemeli.
6. **Tümü:** bu set üzerinde ölçülen hiçbir rakam "insan-doğrulanmış benchmark
   üzerinde ölçüldü" ifadesiyle sunulamaz. Çapraz-kontrol turu da bir MODEL
   turudur (§3.7).

---

## 8. Yedek parti u301–u330 (2026-08-30)

**Ne:** `korpus-disi` dilimine 30 yeni satır (u301–u330), 30 farklı gerçek
Türk kanununa çapalı. Dilim 200 → **230**, dosya 300 → **330** satır.

### 8.1 Neden

İki gerekçe:

1. **Çapraz-kontrol retleri** (§3): 5/40 örneklenen `korpus-disi` + 9/40
   `eksik-kanit` reddedildi. Retlerin 7'si test yakasındaydı.
2. **G2.1 aritmetiği** (§7.4): test yakasındaki cevaplanamaz sayısı 151 → 144'e
   düşmüştü; %2.0 eşiği için asgari n = 149 gerekiyor, yani kapı 5 satır
   açıktı. Hedef yalnız 149'u kapatmak değil, **gelecek tam test-yakası
   doğrulamasının daha fazla reddi soğurabilmesi için tampon** bırakmaktı.

Yeni bileşim (`rejected` hariç, doğrulayıcının bastığı satır):

| | cevaplanamaz | canary cevaplanabilir | rejected |
|---|---|---|---|
| dev | 159 (113 korpus-dışı + 29 anlamsız + 15 eksik-kanıt + 2 canary) | 26 | 7 |
| **test** | **162** (112 + 31 + 16 + 3) | 17 | 7 |

Test yakası 144 → **162**: G2.1'in 149 asgarisi karşılandı, ≥155 tampon hedefi
7 satır aşıldı. Hash-split 30 satırı iki kümeye dağıttığı için artış tek
yakada toplanmadı.

### 8.2 Tuzak-karşıtı rejim (retlerden çıkarılan dersler)

Bu parti, §3.2'deki 5 `korpus-disi` reddinin **kusur türlerine karşı**
tasarlandı. Her satır için, sonlandırmadan önce `data/research/page_texts.parquet`
(4222 sayfa / 56 belge) tr-duyarlı küçültme + aksan düzleştirme ile tarandı ve
`verification_note` alanına **bir satır negatif kanıt** yazıldı (kaç sayfa, hangi
belgede, hangi bağlamda). Uygulanan kurallar:

- **Ücret/tutar sorusu yok.** `k492` (Harçlar) ve `k488` (Damga) tarifeleri
  somut tutar taşır (u284 pasaport reddi). (8) sayılı tarife tek başına eczane,
  avukatlık, hastane, özel okul, banka, sigorta, gümrük, avcılık, silah, turizm
  ve havayolu ruhsat harçlarını içerir.
- **Genel hüküm testi.** Cevabın özü korpustaki bir kanunun *genel*
  hükmünden çıkarılabiliyorsa satır atıldı — retlerin kaynağı buydu
  (TMK kişilik, CMK usul, 6502, 6356).
- **Kök seviyesinde arama.** Yalnız çoğul/uzun biçim aramak yetmiyor;
  tekil kök taranmadan bir satır sonlandırılmadı.
- **Atıf ≠ içerik, ama atıf uyarıdır.** Korpus bir kanunun ADINI anabilir
  (5779, 7258, 6772, 5488 böyle); satır ancak aranan içerik o atıflarda da
  yoksa tutuldu.

**Taslak aşamasında elenen 6 çapa** (kanıtıyla) — bu partinin asıl ürünü:

| Çapa | Neden elendi |
|---|---|
| 1262 İspençiyari ve Tıbbi Müstahzarlar K. | `k492:59` (8) sayılı tarife ruhsatı verecek makamı **adıyla** yazıyor: "Tıbbi ve ispençiyari müstahzarların ticarete çıkarılması için **Sağlık Bakanlığınca** verilecek ruhsatnameler". Ayrıca `rg1928a:2-3` kanunun 2 nci maddesini fiilen uyguluyor. |
| 1774 Kimlik Bildirme K. | `k5490:27` (Nüfus Hiz. K. m.72) 1774 m.6'yı DEĞİŞTİRİRKEN metni taşıyor: "…**üç gün içinde genel kolluk örgütüne** verilmesi zorunludur". Bildirim süresi korpusta. |
| 2876 Atatürk Kültür, Dil ve Tarih Yüksek Kurumu K. | `k2709:45` (Anayasa m.134) kurumun bileşenlerini sayıyor. |
| 7036 İş Mahkemeleri K. | `k4857:13` (m.20) arabuluculuk zorunluluğunu ve "iki hafta içinde dava" süresini veriyor; `k6102:3` "altı hafta + iki hafta" formülünü taşıyor. u140 tipi sınır ihlali. |
| 2886 Devlet İhale K. | `k4734` ihale usullerinin kavramsal karşılığını taşıyor. |
| 4922 Denizde Can ve Mal Koruma K. | `k6102:249` "denize elverişli"yi tanımlıyor, `k492:57` belgeyi tarifeye almış. |

### 8.3 Künye ve statü

30 satırın tamamı dilimin mevcut rejimini taşır:
`source_type="ajan-taslak"`, `verified_by="script:validate_unans"`,
`verification_status="verified"`,
`verification_kind="mechanical:manifest-absence"`.

> **Bu satırlar da insan onaylı DEĞİLDİR** ve `korpus-disi` diliminin geri
> kalanı gibi **ikinci denetleyici (checker-2) turunu beklemektedir.**
> Negatif kanıt notları taslağı yazan ajanın kendi taramasıdır; drafter ≠
> checker ilkesi gereği bağımsız bir tur yerine geçmez. §1'deki ~%12'lik
> etiket-gürültüsü tahmini bu 30 satır için de geçerli sayılmalıdır — tuzak
> rejimi oranı düşürmeyi amaçlar, sıfırladığını iddia etmez.

### 8.4 Açık kapı: doğrulayıcı dilim sayısı sabiti

`scripts/validate_unans.py` **şu an tek ihlalle kırmızı**:

```
İHLAL: 1
  - dilim korpus-disi: 230 satır, beklenen 200
```

Sebep, 55. satırdaki modül düzeyi sabittir:

```python
SLICE_EXPECT = {"korpus-disi": 200, "anlamsiz-ood": 60, "eksik-kanit": 40}
```

Sayı dosyadan TÜRETİLMİYOR; 190-192. satırlarda tam eşitlikle sınanıyor ve
bunu geçersiz kılacak bayrak/ortam değişkeni yok. Betik bu turda bilerek
DEĞİŞTİRİLMEDİ (`scripts/` paralel bir ajanın dosyası). Sabit bellekte 230'a
çekildiğinde kalan ihlal **0**'dır ve betik `TEMİZ` basar — yani diğer sekiz
kontrolün (şema, kimlik, cevaplanamazlık, çapa yokluğu, künye, yakın-tekrar,
split türetimi) tamamı bu 30 satır için temiz geçiyor. Sahibinin yapması
gereken tek satır:

```python
SLICE_EXPECT = {"korpus-disi": 230, "anlamsiz-ood": 60, "eksik-kanit": 40}
```

---

## 9. Checker-2: test yakası tam doğrulama (2026-08-30)

§3'teki çapraz-kontrol turu `korpus-disi` diliminin yalnız **40/200**'ünü
örneklemişti ve %12.5 etiket gürültüsü ölçmüştü. Bu artık risk G2.1 için
ölçülemezlik yaratır: n=155'te **tek** yanlış etiketli test satırı, kusursuz
çalışan bir sistemi bile ≤%2 kapısında düşürür. Ledger R33 bu yüzden
**test yakasının tam doğrulanmasını, dev yakasındaki gürültünün ise
tolere edilip belgelenmesini** karara bağladı. Bu tur o kararın icrasıdır.

**Kapsam (mekanik olarak hesaplandı, tahmin değil):** `slice=="korpus-disi"`
**ve** `assign_split(satır, splits_v1)=="test"` **ve**
`verified_by=="script:validate_unans"` (yani checker-1'in künyesini henüz
taşımayan, salt mekanik etiketli satırlar) → **112 satır**, 75 ayrık çapa
kanunu. Bunun 94'ü eski `u001–u200` partisinden, 18'i `u301–u330` yedek
partisindendir. `anlamsiz-ood` ve `eksik-kanit` test satırları checker-1
tarafından zaten 100% denetlenmişti; bu tur onlara dokunmadı.

### 9.1 Sonuç

| | adet | oran |
|---|---|---|
| denetlenen | **112** | — |
| uygun | **105** | %93.75 |
| **reddedildi** | **7** | **%6.25** |

Red oranı checker-1'in örneklem tahmininin (%12.5, Wilson [%5.5, %26.1])
alt yarısına düştü; iki ölçüm birbiriyle çelişmiyor (aralıklar örtüşüyor).
`u301–u330` partisinin tuzak-karşıtı rejimi (§8.2) işe yaramış görünüyor:
18 yeni test satırının **hiçbiri** reddedilmedi; 7 reddin tamamı eski
partiden geldi.

### 9.2 Reddedilen 7 satır

| id | çapa | tek satırlık gerekçe (kanıt) |
|---|---|---|
| `u002` | 5901 Türk Vatandaşlığı K. | `k5490:2` "Mavi Kart"ı **adıyla** tanımlıyor: "çıkma izni almak suretiyle Türk vatandaşlığını kaybedenler ... altsoylarına verilen ... resmi belge". Sorulan belge korpusta. |
| `u076` | 5737 Vakıflar K. | `k3065:18` "**Vakıflar Genel Müdürlüğünün yönettiği ve temsil ettiği** mazbut vakıflar" — hem yönetim mercii hem tanımın özü korpusta (+`k4721:22`). |
| `u105` | 4207 Tütün Ürünleri K. | `k5326:12` (Kabahatler m.39) yasağın kapsamını üç fıkrada sayıyor: kamu hizmet binalarının kapalı alanları / toplu taşıma araçları / herkesin girebileceği özel binalar. |
| `u124` | 5520 Kurumlar Vergisi K. | `k193:27` (GVK m.41/1-5) tanımın **ayna metnini** taşıyor ("ilişkili kişilerle emsallere uygunluk ilkesine aykırı ... bedel veya fiyatlar"); `k193:28` açıkça "5520 ... 13 üncü maddesi hükmü uygulanır" diyor; `k3065:27` terimi adıyla anıyor. |
| `u142` | 6216 AYM K. | `k2709:50` (Anayasa m.149) yeter sayıyı birebir veriyor: "siyasî partilerin kapatılmasına ... karar verilebilmesi için toplantıya katılan üyelerin **üçte iki oy çokluğu** şarttır". |
| `u167` | 6413 TSK Disiplin K. | `k657:67` (DMK m.126) "**uyarma**, kınama ve aylıktan kesme cezaları **disiplin amirleri** tarafından ... verilir" — 6413 m.20'nin cevabı farklı kanundan aynen çıkıyor (checker-1'in `u140` deseni). |
| `u196` | 5378 Engelliler K. | `k2828:20` (Sos. Hiz. K. ek m.7) yararlanma şartını birebir veriyor: "hane içinde kişi başına düşen ortalama aylık gelir ... asgarî ücretin aylık net tutarının **2/3'ünden daha az** olan bakıma ihtiyacı olan engellilere ... **evde bakımına destek** verilmesi sağlanır". |

**En öğretici olan `u124`**: çapa kanunu (5520) korpusta yok, ama tanımın
metni gelir vergisindeki eşdeğer hükümde (GVK m.41/1-5) neredeyse kelimesi
kelimesine duruyor ve o hüküm çapa kanununa **adıyla yollama** yapıyor.
"Çapa kanunu korpusta yok" ile "cevap korpusta yok" arasındaki fark bu
satırda en keskin hâlini alıyor — mekanik etiketin yapısal kör noktası.

Ortak desen: **7 reddin 6'sı** (`u002`, `u076`, `u105`, `u124`, `u142`,
`u167`) "başka bir korpus kanunu aynı maddi cevabı taşıyor" tipidir;
yalnız `u196` doğrudan konu örtüşmesidir. Anayasa (`k2709`), DMK (`k657`),
GVK (`k193`) ve Kabahatler K. (`k5326`) en sık sızdıran kaynaklar.

### 9.3 Korunan tuzaklar (reddedilmedi ama sınırdaydı)

Bu satırlarda korpus **yakın ama yanlış** bir cevap taşıyor; etiket doğru
olduğu için tutuldular ve kalibrasyon için değerlidirler:

- `u106` (4207, işletmeye ceza): `k5326:12` **kişiye** 50 TL kesiyor,
  işletmeye değil. Model "50 TL" derse yanlış cevap verir.
- `u109` (6136, ruhsatsız silah): `k5326:14` (m.43) yalnız "kanuna göre
  **yasak olmayan** silahlar" için 50 TL öngörür; ateşli silahın hapis
  cezası korpusta yok.
- `u128` (7338, beyanname süresi): `k213:133` (VUK m.342) yalnız **ek**
  süreyi (15+15 gün) verir; asıl 4 aylık süre korpusta yok.
- `u147` (6325 m.18/A, süre): `k6102:3` (TTK m.5/A) ticari davalar için
  **altı hafta + iki hafta** der; 18/A'nın üç hafta + bir haftası korpusta yok.
- `u120` (4925, zorunlu sigorta): `k2918:71` (KTK m.91) araç **işleteninin**
  ZMSS'ini düzenler; taşımacının taşımacılık sigortası ayrı rejimdir.
- `u162` (3218, vergi istisnaları): korpus serbest bölgelerle ilgili KDV,
  emlak, damga ve harç istisnalarını taşır ama bunlar **ilgili vergi
  kanunlarının** istisnalarıdır; 3218'in kendi gelir/kurumlar ve ücret
  istisnası korpusta yok.
- `u326` (7258, bahis yetkisi): `k3065:59` **Millî Piyango**'yu anar; sorulan
  sabit ihtimalli spor bahisleri teşkilatı korpusta adlandırılmıyor.

### 9.4 Ortaya çıkan garanti

> **Test yakasındaki tüm `korpus-disi` satırları model-çapraz-kontrolden
> geçmiştir** (115 test satırının 112'si bu turda, 3'ü checker-1 turunda).
> **Dev yakasında ~%12.5'lik mekanik-etiket gürültüsü belgelidir** ve
> bilerek tolere edilmiştir (ledger R33).

Yani G2.1'in ≤%2 kapısı yalnız **test** yakasında ölçülebilir sayılmalıdır;
dev yakasındaki cevaplanamaz sayıları geliştirme sinyalidir, kapı değildir.

### 9.5 Künye ve doğrulayıcı

Kararlar şu üçlülerle yazıldı — **ikisi de `VERIF_EXPECT`'te zaten izinliydi**,
betik değiştirilmedi:

```python
uygun: ("verified", "model-cross-check:claude-fable-5-checker", "mechanical:manifest-absence")
reddet: ("rejected", "model-cross-check:claude-fable-5-checker", "model-cross-check")
```

Checker-2 turu **checker-1'in künye dizgisini yeniden kullanmıştır**
(`...-checker`, `...-checker2` değil): aynı rejim, aynı kanıt tabanı, aynı
sınırlar — ve doğrulayıcının izinli kümesini genişletmeye gerek kalmıyor.
Turları bu bölümün tarihi ayırır; `uygun` satırların notu ` +checker2`
soneki taşır, red notları `[checker2 reddi]` ile biter.

`uv run python scripts/validate_unans.py` → **TEMİZ**. §8.4'teki
`SLICE_EXPECT` açığı bu tur öncesinde kapatılmıştır (sabit 230'dur).

### 9.6 Doğrulama sonrası bileşim

| | cevaplanamaz (`rejected` hariç) | canary cevaplanabilir | `rejected` |
|---|---|---|---|
| dev | 159 (113 korpus-dışı + 29 anlamsız + 15 eksik-kanıt + 2 canary) | 26 | 7 |
| **test** | **155** (105 + 31 + 16 + 3) | 17 | 14 |

n=155'te 0 hata → Clopper-Pearson tek-yanlı %95 üst sınırı **%1.914** < %2.0
(§3.6'daki hesapla aynı yöntem: `1 - 0.05^(1/n)`).
**G2.1'in asgari n=149 eşiği karşılanmıştır**; tampon 6 satırdır.

### 9.7 Bu turun sınırları (dürüstlük kaydı)

- Denetleyici bir **modeldir, insan değil**. `verification_kind` bu turda da
  `human` OLMADI; hiçbir rakam "insan-doğrulanmış benchmark" diye sunulamaz.
- Drafter, checker-1 ve checker-2 **aynı model ailesindendir**; eğitim-verisi
  kör noktaları ortaktır. Bulunan 7 hata bir **alt sınırdır**.
- Kanıt yalnız **metin katmanıdır** (`data/research/page_texts.parquet`,
  4222 sayfa / 56 belge); sayfa görüntüsünde olup metne düşmemiş bir hükmü
  bu tur da göremez — özellikle `rg*` taramalarında.
- "Cevap korpusta var" eşiği bir **yargı kararıdır**. §9.3'teki yedi sınır
  satırının bir kısmı başka bir denetleyicide reddedilebilirdi; ret sayısı
  5 ile 12 arasında oynayabilirdi.
- **Dev yakası bu turda denetlenmedi.** Oradaki 113 `korpus-disi` satırının
  ~14'ünün (%12.5) yanlış etiketli olması beklenir; §3.7 hâlâ geçerlidir.
