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

| Dilim | Satır | `verification_status` | `verification_kind` | `verified_by` |
|---|---|---|---|---|
| `korpus-disi` | 200 | `verified` | `mechanical:manifest-absence` | `script:validate_unans` |
| `anlamsiz-ood` | 60 | `draft` | `model-cross-check` (BEKLİYOR) | boş |
| `eksik-kanit` | 40 | `draft` | `model-cross-check` (BEKLİYOR) | boş |
| **toplam** | **300** | 200 verified / 100 draft | | |

## 1. Neden bu set var

Kalibre edilmiş seçici cevaplama (bir soruya cevap vermek yerine
"bilmiyorum" demeyi öğrenmek) ölçülebilmesi için cevaplanamaz veri ister.
Canary'de yalnız 5 cevaplanamaz soru vardı — bu sayıyla ölçülen bir hata
oranının güven aralığı işe yaramayacak kadar geniştir.

Hedef sayı aritmetikten gelir: test kümesinde **n=150** cevaplanamaz soruda
**0 hata** gözlenirse, Clopper-Pearson %95 üst sınırı ≈ **%2.0**'dir. G2.1
kapısının istediği eşik budur; 150 bu eşiğin **aritmetik asgarisidir**, keyfi
bir yuvarlama değil. 300 üretilip ~yarısı test'e düştüğünde bu sayı sağlanır
(fiilî: **test'te 151 cevaplanamaz**, bkz. §4).

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
> **Kalan riskin BÜYÜKLÜĞÜ ölçülmemiştir.** Denetleyici turunun örneklem
> alarak nicelemesi gerekir; o sayı gelene kadar bu dilim "manifest-yokluk
> doğrulanmış", "cevaplanamazlığı doğrulanmış" DEĞİL diye anılmalıdır.

### `anlamsiz-ood` (60) — taslak
Zırva, kategori hatası (`"Zamanaşımı süresi kaç santimetredir?"`), hukuk-dışı
soru, hukuk kelimesi serpiştirilmiş alan-ötesi saçmalık, uydurma kanun
numarası ve talimat enjeksiyonu denemeleri. Bu dilim **kendi kendine
doğrulanmamıştır**: soruları yazan tur onları doğrulayamaz, çünkü aynı turun
"bu anlamsızdır" kararı bağımsız bir ölçüm değildir. `verified_by` bilerek
boştur ve ayrı bir denetleyici turu beklemektedir.

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
kusurlu olabilir). Bu yüzden dilim `draft`'tır. Bu satırlar ayrıca ileride
PPI (prediction-powered inference) çiftleri olarak kullanılmak üzere
saklanmaktadır.

## 3. Soru kalitesi kuralları

- Doğal Türkçe, bir vatandaşın ya da avukatın gerçekten soracağı biçimde.
- `query_style` çeşitlemesi: 79 `dogal` / 86 `hukuki` / 35 `madde-referansli`
  (korpus-dışı dilimde). `madde-referansli` sorular gerçekten bir madde
  numarası anar — canary incelemesinde yakalanan etiket hatası tekrarlanmasın
  diye.
- Zorluk dağılımı: 49 kolay / 161 orta / 90 zor.
- Tekrar yok: set içinde ve **canary'ye karşı** normalize edilmiş token kümesi
  örtüşmesi (Jaccard ≥ 0.8) ile taranır. Doğrulayıcı bir kez gerçek bir tekrar
  yakaladı (u108/u109 aynı metne düşmüştü) ve satır yeniden yazıldı.

## 4. Split şeması

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

Test'teki 151 cevaplanamaz soru §1'deki n=150 asgarisini karşılar.

## 5. Doğrulamayı yeniden koşmak

```bash
uv run python scripts/validate_unans.py
```

Betik korpus kümesini repodan yeniden türetir, 300 satırın tamamını şemadan
geçirir, çapaların korpusta olmadığını numara VE ad-token bazında sınar,
soruların çapalarını andığını kontrol eder, dilim sayılarını, doğrulama
künyesini ve yakın-tekrarları denetler, split'i künyedeki kuralla yeniden
türetip dosyayla karşılaştırır. İhlalde çıkış kodu 1'dir.

Yükleme kontrolü:

```bash
uv run python -c "from belge_gozu.bench.dataset import load_bench; \
  print(len(load_bench('data/bench/unans_v1.jsonl', only_verified=False)))"   # 300
```

Saf mantığın testleri: `tests/test_validate_unans.py` (her kontrol için kasıtlı
bozuk bir satır da geçirilir — "TEMİZ" çıktısının bir şey kanıtlaması için) ve
`tests/bench/test_dataset.py` (sözlük genişlemeleri + `assign_split`).

## 6. Bu setin `verified` olması için ne gerekiyor (açık kapılar)

1. **`anlamsiz-ood` (60):** bağımsız bir denetleyici turu her satırı okuyup
   gerçekten anlamsız/alan-dışı olduğunu onaylamalı.
2. **`eksik-kanit` (40):** denetleyici + insan. Grep yokluğu makine kanıtıdır;
   sayfa görüntüsünde de bulunmadığının teyidi ayrıca gerekir.
3. **`korpus-disi` (200):** artık riskin (absent kanun sorusunun korpustaki
   başka bir kanunla cevaplanabilmesi) örneklemle **nicelenmesi**. Bu sayı
   üretilene kadar dilimin mekanik etiketi olduğu gibi okunmalıdır.
4. **Tümü:** bu set üzerinde ölçülen hiçbir rakam "insan-doğrulanmış benchmark
   üzerinde ölçüldü" ifadesiyle sunulamaz.
