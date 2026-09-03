# RetrievalEval Ön-Kontrol Raporu

Oluşturulma: 2026-09-02T14:00:52.617502+00:00  ·  toplam soru: 66

Normalizasyon: karşılaştırmadan önce boşluk/satır sonları tek boşluğa indirgenir, ardından Türkçe'ye duyarlı küçültme uygulanır (İ->i, I->ı, sonra str.casefold()); hem sayfa metni hem alıntı AYNI fonksiyondan geçer.

## Özet

- **TEMİZ**: 59
- **ŞÜPHELİ**: 7
- **MANUEL**: 0

## TEMİZ (59)

- `c001` (dogrudan-madde, orta)
- `c002` (dogrudan-madde, kolay)
- `c003` (korpus-disi, orta) — cevaplanamaz soru: kontrol edilecek kanıt yok
- `c004` (korpus-disi, orta) — cevaplanamaz soru: kontrol edilecek kanıt yok
- `c005` (korpus-disi, orta) — cevaplanamaz soru: kontrol edilecek kanıt yok
- `c006` (anlamsiz-ood, kolay) — cevaplanamaz soru: kontrol edilecek kanıt yok
- `c007` (anlamsiz-ood, kolay) — cevaplanamaz soru: kontrol edilecek kanıt yok
- `c101` (dogrudan-madde, kolay)
- `c102` (ayni-kanun-hard-negative, orta)
- `c103` (dogrudan-madde, kolay)
- `c104` (paraphrase, orta)
- `c105` (dogrudan-madde, orta)
- `c106` (ayni-kanun-hard-negative, zor)
- `c107` (dogrudan-madde, kolay)
- `c108` (dogrudan-madde, zor)
- `c109` (dogrudan-madde, kolay)
- `c112` (ayni-kanun-hard-negative, orta)
- `c201` (dogrudan-madde, kolay)
- `c202` (dogrudan-madde, kolay)
- `c203` (dogrudan-madde, kolay)
- `c204` (dogrudan-madde, orta)
- `c205` (dogrudan-madde, kolay)
- `c206` (paraphrase, orta)
- `c207` (paraphrase, orta)
- `c210` (madde-numarali, orta)
- `c211` (madde-numarali, kolay)
- `c212` (madde-numarali, orta)
- `c213` (madde-numarali, zor)
- `c214` (madde-numarali, zor)
- `c215` (madde-numarali, kolay)
- `c301` (tablo-layout, orta)
- `c302` (tablo-layout, kolay)
- `c303` (tablo-layout, zor)
- `c304` (tablo-layout, orta)
- `c305` (tarihi-tarama, zor)
- `c306` (tarihi-tarama, zor)
- `c309` (capraz-kanun-terim, orta)
- `c310` (capraz-kanun-terim, zor)
- `c311` (capraz-kanun-terim, orta)
- `c312` (capraz-kanun-terim, orta)
- `c313` (ayni-kanun-hard-negative, zor)
- `c401` (paraphrase, orta)
- `c402` (paraphrase, kolay)
- `c403` (paraphrase, orta)
- `c404` (paraphrase, orta)
- `c405` (paraphrase, kolay)
- `c406` (paraphrase, orta)
- `c407` (paraphrase, orta)
- `c408` (paraphrase, orta)
- `c409` (paraphrase, kolay)
- `c410` (paraphrase, kolay)
- `c411` (paraphrase, orta)
- `c412` (paraphrase, orta)
- `c413` (paraphrase, orta)
- `c414` (paraphrase, orta)
- `c415` (paraphrase, orta)
- `c416` (paraphrase, kolay)
- `c417` (paraphrase, orta)
- `c418` (paraphrase, orta)

## ŞÜPHELİ (7)

### `c110` — Onbir yaşındaki bir çocuk suç sayılan bir eylemde bulunursa hakkında ceza davası açılabilir mi?
- dilim: paraphrase, zorluk: orta
- gold_page_ids: ['k5237:8']
- not: paraphrase dilimi ama sözlüksel örtüşme %55 (azami %50) — soru sözcükleri gold sayfada birebir geçiyor, dilim etiketi yanlış olabilir

### `c111` — Kanunun kendisine tanıdığı bir yetkiyi kullanan kişi, bunun sonucunda oluşan bir suçtan dolayı cezalandırılır mı?
- dilim: paraphrase, zorluk: kolay
- gold_page_ids: ['k5237:7']
- not: paraphrase dilimi ama sözlüksel örtüşme %58 (azami %50) — soru sözcükleri gold sayfada birebir geçiyor, dilim etiketi yanlış olabilir

### `c208` — İşyeri başka bir şirkete satıldığında, yeni işveren sadece bu satış nedeniyle çalışanların iş sözleşmesini sona erdirebilir mi?
- dilim: paraphrase, zorluk: orta
- gold_page_ids: ['k4857:5']
- not: paraphrase dilimi ama sözlüksel örtüşme %64 (azami %50) — soru sözcükleri gold sayfada birebir geçiyor, dilim etiketi yanlış olabilir

### `c209` — Gözaltına alınan bir kişiye, neden gözaltına alındığı derhal bildirilmek zorunda mıdır?
- dilim: paraphrase, zorluk: orta
- gold_page_ids: ['k2709:6']
- not: paraphrase dilimi ama sözlüksel örtüşme %57 (azami %50) — soru sözcükleri gold sayfada birebir geçiyor, dilim etiketi yanlış olabilir

### `c307` — 10 Kasım 1975 tarihli Resmî Gazete'de yayımlanan 7/10445 sayılı Kararname hangi konudadır?
- dilim: tarihi-tarama, zorluk: zor
- gold_page_ids: ['rg1975a:1']
- eşleşmeyen alıntı (rg1975a:1): "Türk Silâhlı Kuvvetleri Kıyafet Kararının Bazı Maddelerinin Değiştirilmesi ve Bazı Maddelerine Fıkralar Eklenmesi Hakkında Karar"
  en yakın satır: "min Değiştirilmesi ve Bazı Maddelerine Fıkralar Eklenmesi"

### `c308` — 619 sayılı 1965 Yılı Bütçe Kanunu'nun 35 inci maddesine göre, bu kanun ne zaman yürürlüğe girmiştir?
- dilim: tarihi-tarama, zorluk: zor
- gold_page_ids: ['rg1965a:3']
- eşleşmeyen alıntı (rg1965a:3): "Bu kanun 1 Mart 1965 tarihinde yürürlüğe girer."
  en yakın satır: "Madde 35 — B u kanun 1 Mart 1965 tarihinde yürürlüğe girer."

### `c314` — Ömür boyu gelir gibi dönemsel edimlerde, borçlar hukukundaki zamanaşımı ne zaman işlemeye başlar?
- dilim: ayni-kanun-hard-negative, zorluk: zor
- gold_page_ids: ['k6098:30']
- eşleşmeyen alıntı (k6098:30): "Ömür boyu gelir ve benzeri dönemsel edimlerde, alacağın tamamı için zamanaşımı, ifa edilmemiş ilk dönemsel edimin muaccel olduğu günde işlemeye başlar."
  en yakın satır: "için zamanaşımı, ifa edilmemiş ilk dönemsel edimin muaccel olduğu günde işlemeye başlar."


## MANUEL (0)

