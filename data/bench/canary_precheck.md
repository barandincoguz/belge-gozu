# Canary Ön-Kontrol Raporu

Oluşturulma: 2026-08-29T16:34:12.897687+00:00  ·  toplam soru: 48

Normalizasyon: karşılaştırmadan önce boşluk/satır sonları tek boşluğa indirgenir, ardından Türkçe'ye duyarlı küçültme uygulanır (İ->i, I->ı, sonra str.casefold()); hem sayfa metni hem alıntı AYNI fonksiyondan geçer.

## Özet

- **TEMİZ**: 45
- **ŞÜPHELİ**: 3
- **MANUEL**: 0

## TEMİZ (45)

- `c001` (paraphrase, orta)
- `c002` (paraphrase, kolay)
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
- `c108` (paraphrase, zor)
- `c109` (dogrudan-madde, kolay)
- `c110` (paraphrase, orta)
- `c111` (paraphrase, kolay)
- `c112` (ayni-kanun-hard-negative, orta)
- `c201` (dogrudan-madde, kolay)
- `c202` (dogrudan-madde, kolay)
- `c203` (dogrudan-madde, kolay)
- `c204` (dogrudan-madde, orta)
- `c205` (dogrudan-madde, kolay)
- `c206` (paraphrase, orta)
- `c207` (paraphrase, orta)
- `c208` (paraphrase, orta)
- `c209` (paraphrase, orta)
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

## ŞÜPHELİ (3)

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

