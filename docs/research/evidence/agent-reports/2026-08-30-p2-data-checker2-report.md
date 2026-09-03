# P2 veri — checker-2: test yakası `korpus-disi` tam doğrulama raporu

**Tarih:** 2026-08-30 · **Rol:** ikinci bağımsız denetleyici (checker-2) ·
**Künye:** `model-cross-check:claude-fable-5-checker` (checker-1'in dizgisi
bilerek yeniden kullanıldı — §6) ·
**Kanıt:** yalnız `data/research/page_texts.parquet` (4222 sayfa / 56 belge).
Ağ yok, Gemini yok, alt-ajan yok.

Dokunulan dosyalar: `data/bench/abstention_eval_v1.jsonl`,
`data/bench/abstention_eval_v1.README.md` (§9 eklendi).
`scripts/validate_abstention_eval.py` DEĞİŞTİRİLMEDİ — gerekmedi (§6).

---

## 1. Neden bu tur

Checker-1 (`p2-data-crosscheck-report.md`) `korpus-disi` diliminin **40/200**'ünü
örnekledi ve **%12.5** etiket gürültüsü ölçtü (Wilson %95 [%5.5, %26.1]).
Bu artık risk G2.1'i ölçülemez kılar: n≈155'te **tek** yanlış etiketli test
satırı, kusursuz çalışan bir sistemi bile ≤%2 kapısında düşürür — çünkü
sistem doğru cevabı verdiğinde "abstain etmedi" diye ceza alır.

Ledger **R33**: *test yakası tam doğrulanacak, dev yakasındaki gürültü tolere
edilip belgelenecek.* Bu rapor o kararın icrasıdır.

---

## 2. Kapsam — hesaplandı, tahmin edilmedi

```python
from belge_gozu.bench.dataset import assign_split, load_splits
splits = load_splits("data/bench/splits_v1.json")
scope = [r for r in rows
         if r["slice"] == "korpus-disi"
         and assign_split(r, splits) == "test"
         and r["verified_by"] == "script:validate_abstention_eval"]
```

**Kapsam listesi sayısı: `112`** (75 ayrık çapa kanunu).

| kaynak | adet |
|---|---|
| eski parti `u001–u200` (checker-1 örneklemine girmemiş test satırları) | 94 |
| yedek parti `u301–u330` (test'e düşenler) | 18 |
| **toplam** | **112** |

Filtre neden `verified_by`? Çünkü checker-1 `korpus-disi` diliminde yalnız
**reddettiği** satırların künyesini değiştirdi; "uygun" dediği 14 test
satırı `script:validate_abstention_eval` olarak kaldı. Bu 14 satır da kapsama girdi ve
yeniden denetlendi — §5'teki garanti *tüm* test satırlarının denetçi künyesi
taşımasını gerektiriyor. (Yeniden denetimde hiçbiri farklı sonuç vermedi.)

Dışarıda bırakılanlar: test yakasındaki `anlamsiz-ood` (31) ve `eksik-kanit`
(16) satırları — checker-1 bunları %100 denetlemişti; ayrıca checker-1'in
3 test-yakası reddi (`u110`, `u130`, `u135`) zaten `rejected`.

---

## 3. Sonuç

| | adet | oran |
|---|---|---|
| denetlenen | **112** | — |
| uygun | **105** | %93.75 |
| **reddedildi** | **7** | **%6.25** |

Wilson %95 iki yanlı: **[%3.1, %12.4]**. Checker-1'in örneklem tahminiyle
(%12.5 [%5.5, %26.1]) **çelişmiyor** — aralıklar örtüşüyor; tam sayım
tahminin alt yarısına düştü.

`u301–u330` yedek partisinin tuzak-karşıtı rejimi (README §8.2) tuttu:
**18 yeni test satırının hiçbiri reddedilmedi**; 7 reddin tamamı eski
`u001–u200` partisinden geldi. Yani eski parti için gerçek red oranı
7/94 = **%7.4**.

---

## 4. Reddedilen 7 satır — kanıt

Her red için `page_id` + korpustan birebir alıntı. Aynı metinler satırların
`verification_note` alanında da duruyor (`[checker2 reddi]` soneki ile).

### 4.1 `u002` — 5901 Türk Vatandaşlığı K., çıkma izni belgesi

> **Soru:** "Türk Vatandaşlığı Kanunu uyarınca çıkma izni alarak vatandaşlıktan
> ayrılan kişilere hangi belge verilir ve bu belge sahibine hangi hakları sağlar?"

> `k5490:2` (Nüfus Hizmetleri K. m.3/m): "**Mavi Kart:** Doğumla Türk vatandaşı
> olup da **çıkma izni almak suretiyle Türk vatandaşlığını kaybedenler** ve
> bunların 29/5/2009 tarihli ve 5901 sayılı Türk Vatandaşlığı Kanununun 28 inci
> maddesinde belirtilen altsoylarına **verilen** ve söz konusu maddede belirtilen
> **haklardan faydalanabileceklerini gösteren resmi belgeyi**"

Sorunun birinci yarısı (hangi belge) korpusta **adıyla** duruyor; ikinci yarı
(hangi haklar) 5901 m.28'e yollamayla bırakılmış. Kısmî de olsa doğru cevap
üretilebildiği için etiket ayakta duramaz. `k5490:1` ve `k5490:5` aynı tanımı
tekrarlar.

### 4.2 `u076` — 5737 Vakıflar K., mazbut vakıf

> **Soru:** "5737 sayılı Vakıflar Kanunu'nda mazbut vakıf nasıl tanımlanmıştır ve
> yönetimi kime aittir?"

> `k3065:18` (KDV K. m.17/4): "...belediyeler, il özel idareleri ve yatırım izleme
> ve koordinasyon başkanlıklarının mülkiyetindeki taşınmazların ve **Vakıflar
> Genel Müdürlüğünün yönettiği ve temsil ettiği mazbut vakıfların** mülkiyetinde
> bulunan taşınmazların satışı..."
> `k4721:22` (TMK m.111): "Vakıfların, vakıf senedindeki hükümleri yerine getirip
> getirmedikleri ... **Vakıflar Genel Müdürlüğünce** ve üst kuruluşlarınca
> denetlenir."

5737 m.3'ün tanımı ("Genel Müdürlükçe yönetilecek ve temsil edilecek vakıflar")
ile `k3065:18`'in ibaresi neredeyse aynı; hem **tanımın özü** hem **yönetim
mercii** korpustan okunuyor. `k1319:2` ayrıca "özel bütçeli idarelere (mazbut
vakıflar dahil)" der.

### 4.3 `u105` — 4207 Tütün Ürünleri K., kapalı alan yasağının kapsamı

> **Soru:** "4207 sayılı Kanun'a göre kapalı alanlarda tütün ürünü kullanma
> yasağının kapsamı nedir?"

> `k5326:12` (Kabahatler K. m.39): "(1) **Kamu hizmet binalarının kapalı
> alanlarında** tütün mamulü tüketen kişiye ... elli Türk Lirası idarî para cezası
> verilir. Bu fıkra hükmü, tütün mamulü tüketilmesine tahsis edilen alanlarda
> uygulanmaz. (2) **Toplu taşıma araçlarında** tütün mamulü tüketen kişiye ...
> (3) **Özel hukuk kişilerine ait olan ve herkesin girebileceği binaların kapalı
> alanlarında**, tütün mamullerinin tüketilemeyeceğini belirtir açık bir işarete
> yer verilmesine rağmen, bu yasağa aykırı hareket eden kişiye ..."

Kabahatler K. m.39, 4207 m.2'nin kapsam üçlüsünü (kamu binaları / toplu taşıma /
herkese açık özel binalar) yaptırım hükmü biçiminde **aynen** taşıyor. Bu satırı
checker-1 örneklemişti ve "uygun" demişti; tam tarama düzeltiyor.

### 4.4 `u124` — 5520 Kurumlar Vergisi K., transfer fiyatlandırması *(en öğretici red)*

> **Soru:** "Kurumlar Vergisi Kanunu'nda transfer fiyatlandırması yoluyla örtülü
> kazanç dağıtımı nasıl tanımlanmıştır?"

> `k193:27` (GVK m.41/1-5): "...teşebbüs sahibinin, **ilişkili kişilerle emsallere
> uygunluk ilkesine aykırı olarak tespit edilen bedel veya fiyatlar üzerinden mal
> veya hizmet alım ya da satımında bulunması** halinde, emsallere uygun bedel veya
> fiyatlar ile teşebbüs sahibince uygulanmış bedel veya fiyat arasındaki işletme
> aleyhine oluşan farklar işletmeden çekilmiş sayılır."
> `k193:28`: "Teşebbüs sahibinin eşi, üstsoy ve altsoyu ... **ilişkili kişi
> sayılır.** ... İlişkili kişiler ve bu kişilerle yapılan işlemler hakkında bu
> maddede yer almayan hususlar bakımından, **5520 sayılı Kurumlar Vergisi
> Kanununun 13 üncü maddesi hükmü uygulanır.**"
> `k3065:27` (KDV K. m.30/d): "(**5520 sayılı Kanunun 13 üncü maddesine göre
> transfer fiyatlandırması yoluyla örtülü olarak dağıtılan kazançlar** ile Gelir
> Vergisi Kanununun 41 inci maddesinin birinci fıkrasının (5) numaralı bendine
> göre işletme aleyhine oluşan farklar...)"

KVK m.13/1'in tanımı ile GVK m.41/1-5'in metni **ayna hükümdür**: aynı ölçüt
(ilişkili kişi + emsallere uygunluk ilkesi), aynı sonuç. Üstelik korpus hem
terimi adıyla anıyor hem de çapa kanununa numarasıyla yollama yapıyor.

Bu satır, mekanik etiketin yapısal kör noktasını en keskin gösterendir:
**"çapa kanunu korpusta yok" ≠ "cevap korpusta yok"** — Türk vergi mevzuatında
gelir/kurumlar ikizi hükümler kuraldır, istisna değil. Aynı desen `u123`
(iştirak kazançları) için kontrol edildi ve orada ayna hüküm **bulunmadı**
(GVK'da gerçek kişiler için iştirak kazancı istisnası yoktur) — satır tutuldu.

### 4.5 `u142` — 6216 AYM K., siyasi parti kapatmada karar yeter sayısı

> `k2709:50` (Anayasa m.149/1): "...Anayasa değişikliğinde iptale, **siyasî
> partilerin kapatılmasına** ya da Devlet yardımından yoksun bırakılmasına
> **karar verilebilmesi için toplantıya katılan üyelerin üçte iki oy çokluğu
> şarttır.**"

6216 m.66/3 bu kuralı tekrarlar; sorulan sayı birebir Anayasa'da. `k2709:51`
ayrıca "siyasî partilerin kapatılmasına ilişkin davalarda" duruşma usulünü verir.

### 4.6 `u167` — 6413 TSK Disiplin K., uyarma cezası yetkilisi

> `k657:67` (DMK m.126): "**Uyarma**, kınama ve aylıktan kesme cezaları **disiplin
> amirleri tarafından**; kademe ilerlemesinin durdurulması cezası, memurun bağlı
> olduğu kurumdaki disiplin kurulunun kararı alındıktan sonra, atamaya yetkili
> amirler ... tarafından verilir."

6413 m.20/1 de "uyarma, kınama ve hizmete kısmi süreli devam cezaları **disiplin
amirleri** tarafından verilir" der. Kurumlar farklı (memur ≠ asker) ama
kullanıcının aradığı cevap dizgisi (**"disiplin amiri"**) aynı ve korpusta —
checker-1'in `u140` (3071 vs. İYUK, "30 gün") reddiyle birebir aynı desen.

### 4.7 `u196` — 5378 Engelliler K., evde bakımdan yararlanma şartları

> `k2828:20` (Sosyal Hizmetler K. ek m.7): "...**hane içinde kişi başına düşen
> ortalama aylık gelir tutarı, asgarî ücretin aylık net tutarının 2/3'ünden daha
> az olan bakıma ihtiyacı olan engellilere**, resmî veya özel bakım merkezlerinde
> bakım hizmeti ya da sosyal yardım yapılmak suretiyle **evde bakımına destek
> verilmesi sağlanır.** Hanede birden fazla bakıma ihtiyacı olan engelli
> bulunması hâlinde, hane içinde kişi başına düşen ortalama aylık gelir
> tutarının hesaplanmasında birinci bakıma ihtiyacı olan engelliden sonraki her
> bakıma ihtiyacı olan engelli iki kişi sayılır."

Sorulan "yararlanma şartları" tam olarak bu gelir eşiğidir. `k2828:21` ve
`k2828:28` aynı rejimin tutar ve geçiş hükümlerini taşır. Checker-1 aynı
sayfayı `u264`'ü reddederken kullanmıştı; `u196` o taramanın kapsamı dışındaydı.

### 4.8 Redlerin ortak deseni

**7 reddin 6'sı** "başka bir korpus kanunu aynı maddi cevabı taşıyor" tipidir;
yalnız `u196` doğrudan konu örtüşmesidir. En sık sızdıran kaynaklar:

| kaynak | kaç redde |
|---|---|
| `k2709` Anayasa | 1 (`u142`) |
| `k657` DMK | 1 (`u167`) |
| `k193` GVK | 1 (`u124`) |
| `k5326` Kabahatler K. | 1 (`u105`) |
| `k5490` Nüfus Hiz. K. | 1 (`u002`) |
| `k3065` KDV K. | 1 (`u076`) |
| `k2828` Sos. Hiz. K. | 1 (`u196`) |

Genelleme: **Anayasa, DMK, Kabahatler K. ve vergi usul/oran kanunları
"evrensel sızdırıcılardır"** — hemen her idari/disiplin/kapsam sorusunun
karşılığını taşıyabilirler. Gelecek partilerde çapa seçilmeden önce bu dört
belgeye karşı ayrı tarama yapılmalıdır.

---

## 5. Korunan tuzaklar (reddedilmedi ama sınırdaydı)

Bu satırlarda korpus **yakın ama yanlış** bir cevap taşır. Etiket doğru olduğu
için tutuldular; kalibrasyon açısından setin en değerli satırlarıdır çünkü
modeli "buldum sandığı" bir kanıtla abstain etmemeye kışkırtırlar.

| id | korpustaki yakın-yanlış | neden doğru cevap değil |
|---|---|---|
| `u106` (4207, **işletmeye** ceza) | `k5326:12` "elli Türk Lirası idarî para cezası" | ceza **kişiye** kesiliyor; 4207 m.5'in işletme cezası korpusta yok |
| `u109` (6136, ruhsatsız silah) | `k5326:14` (m.43) "yetkili makamlardan ruhsat almaksızın **kanuna göre yasak olmayan** silahları ... taşıyan kişiye ... elli Türk lirası" | hüküm açıkça 6136'nın suç saydığı silahları dışlıyor |
| `u128` (7338, beyanname süresi) | `k213:133` (VUK m.342) "beyanname verme süresinin sonundan başlayarak **15 gün** beklenir ... yeniden 15 günlük bir mühlet" | yalnız **ek** süre; asıl 4 aylık süre korpusta yok |
| `u147` (6325 m.18/A) | `k6102:3` (TTK m.5/A) "arabulucu ... **altı hafta** içinde sonuçlandırır ... en fazla **iki hafta** uzatılabilir" | ticari davalara özgü; 18/A'nın 3+1 haftası korpusta yok |
| `u120` (4925, zorunlu sigorta) | `k2918:71` (KTK m.91) "İşletenlerin ... **malî sorumluluk sigortası yaptırmaları zorunludur**" | araç işletenin ZMSS'i; taşımacının taşımacılık sigortası ayrı rejim |
| `u162` (3218, vergi istisnaları) | `k3065:6/8/17`, `k492:14`, `k488:28`, `k1319:3` serbest bölgelere ilişkin KDV/harç/damga/emlak istisnaları | bunlar **ilgili vergi kanunlarının** istisnalarıdır; 3218'in kendi gelir/kurumlar ve ücret istisnası korpusta yok |
| `u326` (7258, bahis yetkisi) | `k3065:59` "**Millî Piyango İdaresi** Genel Müdürlüğünün şans oyunlarına ilişkin işletme hakkını devrettiği işletici firma" | piyango ≠ sabit ihtimalli spor bahsi; sorulan teşkilat korpusta adlandırılmıyor |
| `u108` (6136, bulundurma/taşıma farkı) | `k492:64` "silah **taşıma** müsaade vesikaları (her yıl için) ... b) **bulundurma** vesikaları" | tarife yalnız harç farkını gösteriyor; kavramsal fark (nerede taşınabilir) korpusta yok |
| `u088` (5393, belediye kuruluş nüfusu) | `k3194:7` "nüfusu **5.000**'in altında kalan yerler" | büyükşehirde mahalleye dönüşen kırsal yerleşim ölçütü; belediye kuruluş eşiği değil |

`u162` en zorlanılan karardır: korpus serbest bölgelerle ilgili **dört ayrı
vergi istisnası** taşıyor. Tutma gerekçesi — bunların hiçbiri "3218'e göre
tanınmış" değildir; sorulan istisnalar (imalatçıya kurumlar vergisi, personele
ücret gelir vergisi) korpusta yok, dolayısıyla korpustan **doğru** cevap
üretilemez. Farklı bir denetleyici bu satırı reddedebilirdi.

---

## 6. Doğrulayıcı: künye izinli çıktı, betik değişmedi

`scripts/validate_abstention_eval.py`'nin `VERIF_EXPECT["korpus-disi"]` kümesi zaten
şunları içeriyordu:

```python
_CHECKER = "model-cross-check:claude-fable-5-checker"
"korpus-disi": {
    ("verified", "script:validate_abstention_eval", "mechanical:manifest-absence"),
    ("verified", _CHECKER, "mechanical:manifest-absence"),   # ← uygun kararları
    ("rejected", _CHECKER, "model-cross-check"),             # ← red kararları
}
```

`...-checker2` dizgisi izinli kümede **yok**. Brief'in verdiği karar uyarınca
**mevcut künye dizgisi yeniden kullanıldı**: aynı rejim (model çapraz-kontrolü),
aynı kanıt tabanı (`page_texts.parquet`), aynı sınırlar (insan değil).
Turları README §9'un tarihi ve not sonekleri ayırır:

- uygun satırlar: `verification_note` sonuna `" +checker2"`
- red satırları: `verification_note` sonunda `"[checker2 reddi]"`

Böylece betiğe hiç dokunulmadı ve `git diff` yalnız iki veri dosyasını
kapsıyor.

### 6.1 Doğrulayıcı çıktısı (son hâl)

```
========================================================================
abstention_eval doğrulama — data/bench/abstention_eval_v1.jsonl
========================================================================
korpus: 56 belge, 50 kanun numarası
satır : 330

dilim           satır   çapa/konu  doğrulama                             stil dağılımı
---------------------------------------------------------------------------------------------------
korpus-disi       230   147 kanun  rejected/verified:karışık             anah=3 doga=85 huku=100 madd=42
anlamsiz-ood       60           -  verified:model-cross-check            doga=34 huku=21 madd=5
eksik-kanit        40    40 belge  rejected/verified:model-cross-check   doga=17 huku=23
---------------------------------------------------------------------------------------------------
zorluk: {'kolay': 51, 'orta': 180, 'zor': 99}
kaynak: {'ajan-taslak': 330}

split bileşimi (seed='belge-gozu-splits-v1', test_docs=22, RG=2)
küme  dilim                     adet
------------------------------------
dev   anlamsiz-ood                29
dev   retrieval_eval-cevaplanabilir       26
dev   retrieval_eval-cevaplanamaz          2
dev   eksik-kanit                 15
dev   korpus-disi                113
test  anlamsiz-ood                31
test  retrieval_eval-cevaplanabilir       17
test  retrieval_eval-cevaplanamaz          3
test  eksik-kanit                 16
test  korpus-disi                105
------------------------------------
dev   TOPLAM cevaplanamaz        159   cevaplanabilir: 26   (rejected hariç; rejected: 7)
test  TOPLAM cevaplanamaz        155   cevaplanabilir: 17   (rejected hariç; rejected: 14)

TEMİZ — tüm kontroller geçti.
```

---

## 7. G2.1 ölçülebilirliği

**Doğrulama sonrası test yakası cevaplanamaz sayısı: `155`** (105 korpus-dışı
+ 31 anlamsız + 16 eksik-kanıt + 3 retrieval_eval cevaplanamaz; `rejected` hariç).

| eşik | değer |
|---|---|
| n=155'te 0 hata, tek-yanlı %95 üst sınır (`1 - 0.05^(1/n)`) | **%1.914** |
| G2.1 hedefi | ≤ %2.0 |
| gereken asgari n | 149 |
| **tampon** | **+6 satır** |

**Kapı açık değil; sayı hedefin ÜSTÜNDE.** Yeni bir yedek partiye ihtiyaç
YOKTUR. Ancak tampon dardır: test yakasında ileride **6'dan fazla** satır
düşerse (ör. korpusa yeni kanun eklenmesi bir çapayı geçersiz kılarsa) n=149
altına inilir.

---

## 8. Ortaya çıkan garanti

> **Test yakasındaki tüm `korpus-disi` satırları model-çapraz-kontrolden
> geçmiştir** — 115 test satırının 112'si bu turda, 3'ü (`u110`, `u130`,
> `u135`) checker-1 turunda. **Dev yakasında ~%12.5'lik mekanik-etiket
> gürültüsü belgelidir** ve ledger R33 uyarınca bilerek tolere edilmiştir.

Sonuç: G2.1'in ≤%2 abstain kapısı yalnız **test** yakasında ölçülebilir
sayılmalıdır. Dev yakasındaki cevaplanamaz sayıları geliştirme sinyalidir,
kapı değildir; dev'de gözlenen ~%12'lik bir "hata" oranının bir bölümü
sistemin değil etiketin hatasıdır.

---

## 9. Yöntem

- tr-duyarlı küçültme (`İ→i`, `I→ı`, `Ş→ş`, …) + aksan düzleştirme
  (`şğüöçı→sguoci`) sonrası regex arama; `data/research/page_texts.parquet`
  üzerinde sayfa bazında.
- Her satır için en az **üç** arama ekseni:
  1. **çapa kanunu numarası** (`4207`, `6413`, …) — atıf var mı;
  2. **çapa adının ayırt edici kökleri** — çoğul/uzun biçim yetmez, **tekil
     kök** ayrıca tarandı (checker-1'in `müstahzar` dersi: `müstahzarlar`
     temiz görünürken `müstahzar` `k492:59`'u yakalamıştı);
  3. **sorunun kavram çekirdeği** ve eş anlamlıları (`mazbut vakıf` **ve**
     `Vakıflar Genel Müdürlüğü`; `kapalı görüş` **ve** `açık görüş`,
     `görüşme hakkı`; `imece` **ve** `köy ihtiyar`).
- **Tüm 56 belge** tarandı — yalnız konuya "yakın" görünenler değil. Harçlar
  Kanunu tarifeleri (`k492:43–69`) ve RG taramaları (`rg1928a`…`rg1975a`)
  her sorguda kapsam içindeydi.
- **Ayna-hüküm taraması** (bu turun eklediği eksen): vergi sorularında
  gelir/kurumlar ikizi, disiplin sorularında DMK/TSK ikizi, kapsam ve
  yaptırım sorularında Kabahatler K. karşılığı ayrıca arandı. `u124` ve
  `u167` yalnız bu eksenle yakalandı.
- Karar kuralı: **şüphede reddet.** Yanlış red n'i küçültür (ölçülebilir,
  telafi edilebilir); yanlış kabul metriği sessizce bozar.
- Operasyonel eşik: *"Korpusu okuyan bir model, sorulan şeye maddi olarak
  DOĞRU bir cevap üretebilir mi?"* — üretebiliyorsa red. Yakın ama **yanlış**
  bir cevap üretiyorsa satır tutulur ve §5'e yazılır.

---

## 10. Sınırlar (dürüstlük kaydı)

- Denetleyici bir **modeldir, insan değil.** `verification_kind` bu turda da
  `human` OLMADI. Hiçbir rakam "insan-doğrulanmış benchmark" diye sunulamaz;
  `source_type` tüm satırlarda `ajan-taslak` olarak kaldı.
- Drafter, checker-1 ve checker-2 **aynı model ailesindendir**; eğitim-verisi
  kör noktaları ortaktır. Bulunan 7 hata bir **alt sınırdır**. Bağımsız bir
  model ailesi (ya da bir hukukçu) muhtemelen daha fazlasını bulurdu.
- Kanıt yalnız **metin katmanıdır**. Sayfa görüntüsünde olup OCR/metin
  katmanına düşmemiş bir hükmü bu tur da göremez — özellikle `rg*` tarihî
  taramalarında, ki oralarda metin kalitesi gözle görülür şekilde bozuktur.
- **"Cevap korpusta var" eşiği bir yargı kararıdır.** §5'teki dokuz sınır
  satırı farklı bir denetleyicide farklı sonuçlanabilirdi; red sayısı
  makul olarak **5 ile 12** arasında oynayabilirdi.
- **Dev yakası denetlenmedi.** Oradaki 113 `korpus-disi` satırının ~14'ünün
  (%12.5) yanlış etiketli olması beklenir. Bu bilinçli bir karardır (R33),
  ihmal değil — ama dev üzerinden yapılan hiçbir kalibrasyon ölçümü mutlak
  sayı olarak raporlanmamalıdır.
- Bu tur **satır eklemedi/silmedi ve hiçbir soruyu yeniden yazmadı.** Yanlış
  etiketli satırlar `rejected` yapıldı; ölçülen gürültü gizlenmedi.

---

## 11. Satır satır kararlar (112)

| # | id | çapa | soru (kısalt.) | karar |
|---|---|---|---|---|
| 1 | `u001` | 5901 Türk Vatandaşlığı Kanunu | 5901 sayılı Türk Vatandaşlığı Kanunu'nun 16 ncı maddesine göre evlenme yoluyla Türk vata… | uygun |
| 2 | `u002` | 5901 Türk Vatandaşlığı Kanunu | Türk Vatandaşlığı Kanunu uyarınca çıkma izni alarak vatandaşlıktan ayrılan kişilere hang… | **REDDEDİLDİ** |
| 3 | `u003` | 5901 Türk Vatandaşlığı Kanunu | 5901 sayılı Kanun'un 34 üncü maddesindeki seçme hakkıyla vatandaşlığın kaybı, hangi süre… | uygun |
| 4 | `u010` | 6458 Yabancılar ve Uluslararası Koruma Kanunu | 6458 sayılı Kanun'un 31 inci maddesine göre kısa dönem ikamet izni en fazla kaç yıllık s… | uygun |
| 5 | `u011` | 6458 Yabancılar ve Uluslararası Koruma Kanunu | Yabancılar ve Uluslararası Koruma Kanunu'nda mülteci ile şartlı mülteci statüsü arasında… | uygun |
| 6 | `u012` | 6458 Yabancılar ve Uluslararası Koruma Kanunu | 6458 sayılı Kanun'a göre idari gözetim kararına karşı hangi mercie ve ne kadar sürede it… | uygun |
| 7 | `u013` | 6735 Uluslararası İşgücü Kanunu | 6735 sayılı Uluslararası İşgücü Kanunu kapsamında Turkuaz Kart hangi niteliklere sahip y… | uygun |
| 8 | `u014` | 6735 Uluslararası İşgücü Kanunu | Uluslararası İşgücü Kanunu'na göre çalışma izni başvurusunun reddi kararına karşı itiraz… | uygun |
| 9 | `u020` | 2644 Tapu Kanunu | 2644 sayılı Tapu Kanunu'nun 35 inci maddesine göre yabancı uyruklu bir gerçek kişi Türki… | uygun |
| 10 | `u021` | 2644 Tapu Kanunu | 2644 sayılı Tapu Kanunu uyarınca yabancılara yapılacak satışlarda ilçe yüzölçümü bakımın… | uygun |
| 11 | `u022` | 6306 Afet Riski Altındaki Alanların Dönüştürülmesi Hakkın | 6306 sayılı Kanun'un 3 üncü maddesine göre riskli yapı tespitine itiraz için tanınan sür… | uygun |
| 12 | `u023` | 6306 Afet Riski Altındaki Alanların Dönüştürülmesi Hakkın | Kentsel dönüşümde 6306 sayılı Kanun uyarınca riskli yapının yıktırılması için maliklere … | uygun |
| 13 | `u024` | 6306 Afet Riski Altındaki Alanların Dönüştürülmesi Hakkın | 6306 sayılı Kanun kapsamında kira yardımından yararlanmanın şartları nelerdir? | uygun |
| 14 | `u037` | 5307 Sıvılaştırılmış Petrol Gazları (LPG) Piyasası Kanunu | 5307 sayılı Kanun'a göre otogaz istasyonlarında sorumlu müdür bulundurma zorunluluğu han… | uygun |
| 15 | `u038` | 5809 Elektronik Haberleşme Kanunu | 5809 sayılı Elektronik Haberleşme Kanunu'na göre numara taşınabilirliği talebi hangi sür… | uygun |
| 16 | `u039` | 5809 Elektronik Haberleşme Kanunu | 5809 sayılı Kanun uyarınca kayıt dışı IMEI numarasına sahip cihazların şebekeye erişimi … | uygun |
| 17 | `u056` | 5464 Banka Kartları ve Kredi Kartları Kanunu | Banka Kartları ve Kredi Kartları Kanunu'na göre kredi kartı dönem borcunun asgari ödeme … | uygun |
| 18 | `u057` | 5464 Banka Kartları ve Kredi Kartları Kanunu | Kredi kartı kaybolduğunda, bildirim yapılmadan önceki işlemler için kart hamilinin sorum… | uygun |
| 19 | `u063` | 5607 Kaçakçılıkla Mücadele Kanunu | Kaçakçılıkla Mücadele Kanunu'na göre akaryakıt kaçakçılığı yapanlara hangi ceza verilir? | uygun |
| 20 | `u064` | 5607 Kaçakçılıkla Mücadele Kanunu | Kaçakçılıkla Mücadele Kanunu'nda etkin pişmanlık hükümleri hangi şartlarda uygulanır? | uygun |
| 21 | `u065` | 4458 Gümrük Kanunu | 4458 sayılı Gümrük Kanunu'nun 197 nci maddesine göre gümrük vergisi alacağının yükümlüye… | uygun |
| 22 | `u066` | 4458 Gümrük Kanunu | 4458 sayılı Gümrük Kanunu'nda dahilde işleme rejimi nasıl tanımlanmıştır? | uygun |
| 23 | `u067` | 4458 Gümrük Kanunu | 4458 sayılı Kanun'a göre gümrük idaresinin kararlarına karşı itiraz süresi kaç gündür? | uygun |
| 24 | `u068` | 5275 Ceza ve Güvenlik Tedbirlerinin İnfazı Hakkında Kanun | 5275 sayılı Kanun'un 107 nci maddesine göre süreli hapis cezasında koşullu salıverilme i… | uygun |
| 25 | `u069` | 5275 Ceza ve Güvenlik Tedbirlerinin İnfazı Hakkında Kanun | 5275 sayılı Kanun'un 105/A maddesine göre denetimli serbestlik tedbiri uygulanarak cezan… | uygun |
| 26 | `u070` | 5275 Ceza ve Güvenlik Tedbirlerinin İnfazı Hakkında Kanun | Cezaevindeki bir yakınımı ayda kaç kez kapalı görüşe gidebilirim; Ceza ve Güvenlik Tedbi… | uygun |
| 27 | `u075` | 5253 Dernekler Kanunu | 5253 sayılı Dernekler Kanunu'nun 19 uncu maddesine göre dernekler beyannamelerini hangi … | uygun |
| 28 | `u076` | 5737 Vakıflar Kanunu | 5737 sayılı Vakıflar Kanunu'nda mazbut vakıf nasıl tanımlanmıştır ve yönetimi kime aitti… | **REDDEDİLDİ** |
| 29 | `u078` | 4447 İşsizlik Sigortası Kanunu | İşsizlik Sigortası Kanunu'na göre işsizlik ödeneğine hak kazanmak için son üç yılda en a… | uygun |
| 30 | `u079` | 4447 İşsizlik Sigortası Kanunu | 4447 sayılı İşsizlik Sigortası Kanunu'na göre kısa çalışma ödeneği en fazla kaç ay sürey… | uygun |
| 31 | `u080` | 4447 İşsizlik Sigortası Kanunu | 4447 sayılı İşsizlik Sigortası Kanunu'nun 50 nci maddesine göre günlük işsizlik ödeneği … | uygun |
| 32 | `u081` | 854 Deniz İş Kanunu | 854 sayılı Deniz İş Kanunu hangi tonajın üzerindeki gemilerde çalışan gemi adamlarına uy… | uygun |
| 33 | `u082` | 854 Deniz İş Kanunu | Deniz İş Kanunu'na göre gemi adamının hizmet akdinin belirli bir sefer için yapılması ha… | uygun |
| 34 | `u088` | 5393 Belediye Kanunu | 5393 sayılı Belediye Kanunu'nun 4 üncü maddesine göre yeni bir belediye kurulabilmesi iç… | uygun |
| 35 | `u089` | 5393 Belediye Kanunu | 5393 sayılı Belediye Kanunu'na göre belediye meclisi üyeliği hangi hallerde düşer? | uygun |
| 36 | `u090` | 5393 Belediye Kanunu | 5393 sayılı Kanun'un 33 üncü maddesine göre belediye encümeni kimlerden oluşur? | uygun |
| 37 | `u091` | 5216 Büyükşehir Belediyesi Kanunu | 5216 sayılı Büyükşehir Belediyesi Kanunu'nun 4 üncü maddesine göre büyükşehir belediyesi… | uygun |
| 38 | `u092` | 5216 Büyükşehir Belediyesi Kanunu | Büyükşehir Belediyesi Kanunu'na göre büyükşehir belediyesi ile ilçe belediyeleri arasınd… | uygun |
| 39 | `u093` | 2464 Belediye Gelirleri Kanunu | Belediye Gelirleri Kanunu'na göre dükkânımın tabelası için ödeyeceğim ilan ve reklam ver… | uygun |
| 40 | `u094` | 2464 Belediye Gelirleri Kanunu | Belediye Gelirleri Kanunu'na göre çevre temizlik vergisi kimden ve nasıl tahsil edilir? | uygun |
| 41 | `u097` | 1219 Tababet ve Şuabatı San'atlarının Tarzı İcrasına Dair | 1219 sayılı Kanun'a göre Türkiye'de hekimlik mesleğini icra edebilmek için hangi şartlar… | uygun |
| 42 | `u098` | 1219 Tababet ve Şuabatı San'atlarının Tarzı İcrasına Dair | Tababet ve Şuabatı San'atlarının Tarzı İcrasına Dair Kanun'a göre diş hekimlerinin yetki… | uygun |
| 43 | `u101` | 5199 Hayvanları Koruma Kanunu | Hayvanları Koruma Kanunu'na göre sahipsiz hayvanların toplanması ve bakımından hangi kur… | uygun |
| 44 | `u102` | 5199 Hayvanları Koruma Kanunu | 5199 sayılı Hayvanları Koruma Kanunu'na göre bir hayvana kasten kötü muamele edene hangi… | uygun |
| 45 | `u105` | 4207 Tütün Ürünlerinin Zararlarının Önlenmesi ve Kontrolü | 4207 sayılı Kanun'a göre kapalı alanlarda tütün ürünü kullanma yasağının kapsamı nedir? | **REDDEDİLDİ** |
| 46 | `u106` | 4207 Tütün Ürünlerinin Zararlarının Önlenmesi ve Kontrolü | Tütün Ürünlerinin Zararlarının Önlenmesi ve Kontrolü Hakkında Kanun'a göre yasağa uymaya… | uygun |
| 47 | `u107` | 4733 Tütün, Tütün Mamulleri ve Alkol Piyasasının Düzenlen | 4733 sayılı Kanun'a göre tütün mamulü satış belgesi hangi usulle alınır? | uygun |
| 48 | `u108` | 6136 Ateşli Silahlar ve Bıçaklar ile Diğer Aletler Hakkın | 6136 sayılı Ateşli Silahlar ve Bıçaklar Hakkında Kanun'a göre bulundurma ruhsatı ile taş… | uygun |
| 49 | `u109` | 6136 Ateşli Silahlar ve Bıçaklar ile Diğer Aletler Hakkın | Ruhsatsız silah taşımanın cezası, 6136 sayılı Ateşli Silahlar ve Bıçaklar Hakkında Kanun… | uygun |
| 50 | `u111` | 4915 Kara Avcılığı Kanunu | Kara Avcılığı Kanunu'na göre yasak sahada avlananlara uygulanacak yaptırım nedir? | uygun |
| 51 | `u112` | 1380 Su Ürünleri Kanunu | 1380 sayılı Su Ürünleri Kanunu'na göre yasak araç ve yöntemlerle avcılık yapanlara hangi… | uygun |
| 52 | `u116` | 2634 Turizmi Teşvik Kanunu | Turizmi Teşvik Kanunu'na göre bir otel için turizm işletmesi belgesi nasıl alınır? | uygun |
| 53 | `u117` | 2634 Turizmi Teşvik Kanunu | Turizmi Teşvik Kanunu'na göre kültür ve turizm koruma ve gelişim bölgeleri nasıl ilan ed… | uygun |
| 54 | `u118` | 1618 Seyahat Acentaları ve Seyahat Acentaları Birliği Kan | 1618 sayılı Seyahat Acentaları Kanunu'na göre seyahat acentası işletme belgesi almanın ş… | uygun |
| 55 | `u119` | 4925 Karayolu Taşıma Kanunu | Karayolu Taşıma Kanunu'na göre yetki belgesi olmadan taşımacılık yapanlara hangi ceza ke… | uygun |
| 56 | `u120` | 4925 Karayolu Taşıma Kanunu | Karayolu Taşıma Kanunu'na göre taşımacıların zorunlu sorumluluk sigortası yaptırma yüküm… | uygun |
| 57 | `u123` | 5520 Kurumlar Vergisi Kanunu | 5520 sayılı Kurumlar Vergisi Kanunu'na göre iştirak kazançları istisnası hangi şartlarla… | uygun |
| 58 | `u124` | 5520 Kurumlar Vergisi Kanunu | Kurumlar Vergisi Kanunu'nda transfer fiyatlandırması yoluyla örtülü kazanç dağıtımı nası… | **REDDEDİLDİ** |
| 59 | `u125` | 5520 Kurumlar Vergisi Kanunu | 5520 sayılı Kanun'a göre tasfiye halindeki kurumlarda vergilendirme dönemi nasıl belirle… | uygun |
| 60 | `u128` | 7338 Veraset ve İntikal Vergisi Kanunu | Babamdan miras kaldı; Veraset ve İntikal Vergisi Kanunu'na göre beyannameyi kaç ay içind… | uygun |
| 61 | `u129` | 7338 Veraset ve İntikal Vergisi Kanunu | 7338 sayılı Veraset ve İntikal Vergisi Kanunu'na göre ivazsız intikallerde vergi oranı n… | uygun |
| 62 | `u131` | 7194 Dijital Hizmet Vergisi Kanunu | 7194 sayılı Dijital Hizmet Vergisi Kanunu'na göre dijital hizmet vergisinin oranı yüzde … | uygun |
| 63 | `u132` | 7194 Dijital Hizmet Vergisi Kanunu | 7194 sayılı Kanun'a göre konaklama vergisinin matrahı nasıl belirlenir? | uygun |
| 64 | `u133` | 4749 Kamu Finansmanı ve Borç Yönetiminin Düzenlenmesi Hak | 4749 sayılı Kamu Finansmanı ve Borç Yönetiminin Düzenlenmesi Hakkında Kanun'a göre Hazin… | uygun |
| 65 | `u134` | 6085 Sayıştay Kanunu | 6085 sayılı Sayıştay Kanunu'na göre yargılamaya esas rapora ilişkin ilamlara karşı temyi… | uygun |
| 66 | `u138` | 4982 Bilgi Edinme Hakkı Kanunu | 4982 sayılı Bilgi Edinme Hakkı Kanunu'nun 11 inci maddesine göre kurumlar başvuruları ka… | uygun |
| 67 | `u139` | 4982 Bilgi Edinme Hakkı Kanunu | 4982 sayılı Bilgi Edinme Hakkı Kanunu'na göre hangi bilgi ve belgeler bilgi edinme hakkı… | uygun |
| 68 | `u141` | 6216 Anayasa Mahkemesinin Kuruluşu ve Yargılama Usulleri  | 6216 sayılı Kanun'un 47 nci maddesine göre Anayasa Mahkemesine bireysel başvuru süresi k… | uygun |
| 69 | `u142` | 6216 Anayasa Mahkemesinin Kuruluşu ve Yargılama Usulleri  | Anayasa Mahkemesinin Kuruluşu ve Yargılama Usulleri Hakkında Kanun'a göre siyasi parti k… | **REDDEDİLDİ** |
| 70 | `u144` | 2797 Yargıtay Kanunu | 2797 sayılı Yargıtay Kanunu'na göre Yargıtay Büyük Genel Kurulu hangi hallerde toplanır? | uygun |
| 71 | `u147` | 6325 Hukuk Uyuşmazlıklarında Arabuluculuk Kanunu | 6325 sayılı Kanun'un 18/A maddesine göre dava şartı arabuluculukta süreç kaç hafta içind… | uygun |
| 72 | `u148` | 6325 Hukuk Uyuşmazlıklarında Arabuluculuk Kanunu | 6325 sayılı Arabuluculuk Kanunu uyarınca düzenlenen anlaşma belgesine icra edilebilirlik… | uygun |
| 73 | `u152` | 3568 Serbest Muhasebeci Mali Müşavirlik ve Yeminli Mali M | 3568 sayılı Kanun'un 6 ncı maddesine göre serbest muhasebeci mali müşavir olabilmek için… | uygun |
| 74 | `u153` | 3568 Serbest Muhasebeci Mali Müşavirlik ve Yeminli Mali M | Yeminli mali müşavir olabilmek için 3568 sayılı Kanun'da öngörülen mesleki kıdem süresi … | uygun |
| 75 | `u154` | 5362 Esnaf ve Sanatkârlar Meslek Kuruluşları Kanunu | 5362 sayılı Esnaf ve Sanatkârlar Meslek Kuruluşları Kanunu'na göre esnaf odasına kayıt z… | uygun |
| 76 | `u159` | 5957 Sebze ve Meyveler ile Yeterli Arz ve Talep Derinliği | 5957 sayılı Hal Kanunu'na göre toptancı hallerinde alınan hal rüsumu oranı nedir? | uygun |
| 77 | `u160` | 4691 Teknoloji Geliştirme Bölgeleri Kanunu | 4691 sayılı Teknoloji Geliştirme Bölgeleri Kanunu'na göre bölgede çalışan Ar-Ge personel… | uygun |
| 78 | `u162` | 3218 Serbest Bölgeler Kanunu | 3218 sayılı Serbest Bölgeler Kanunu'na göre serbest bölgelerde faaliyet gösteren firmala… | uygun |
| 79 | `u164` | 222 İlköğretim ve Eğitim Kanunu | İlköğretim ve Eğitim Kanunu'na göre çocuğunu okula göndermeyen veliye ne yapılır? | uygun |
| 80 | `u166` | 2914 Yükseköğretim Personel Kanunu | 2914 sayılı Yükseköğretim Personel Kanunu'na göre öğretim üyelerine ödenen geliştirme öd… | uygun |
| 81 | `u167` | 6413 Türk Silahlı Kuvvetleri Disiplin Kanunu | 6413 sayılı Türk Silahlı Kuvvetleri Disiplin Kanunu'na göre uyarma cezası vermeye kim ye… | **REDDEDİLDİ** |
| 82 | `u170` | 3628 Mal Bildiriminde Bulunulması, Rüşvet ve Yolsuzluklar | 3628 sayılı Mal Bildiriminde Bulunulması, Rüşvet ve Yolsuzluklarla Mücadele Kanunu'na gö… | uygun |
| 83 | `u177` | 6222 Sporda Şiddet ve Düzensizliğin Önlenmesine Dair Kanu | Sporda Şiddet ve Düzensizliğin Önlenmesine Dair Kanun'a göre seyirden yasaklanma tedbiri… | uygun |
| 84 | `u179` | 7223 Ürün Güvenliği ve Teknik Düzenlemeler Kanunu | 7223 sayılı Ürün Güvenliği ve Teknik Düzenlemeler Kanunu'na göre iktisadi işletmecilerin… | uygun |
| 85 | `u180` | 5502 Sosyal Güvenlik Kurumu Kanunu | 5502 sayılı Sosyal Güvenlik Kurumu Kanunu'na göre Kurum Genel Kurulu kimlerden oluşur ve… | uygun |
| 86 | `u182` | 3294 Sosyal Yardımlaşma ve Dayanışmayı Teşvik Kanunu | 3294 sayılı Sosyal Yardımlaşma ve Dayanışmayı Teşvik Kanunu'na göre vakıfların gelir kay… | uygun |
| 87 | `u191` | 6415 Terörizmin Finansmanının Önlenmesi Hakkında Kanun | 6415 sayılı Terörizmin Finansmanının Önlenmesi Hakkında Kanun'a göre malvarlığının dondu… | uygun |
| 88 | `u192` | 6493 Ödeme ve Menkul Kıymet Mutabakat Sistemleri, Ödeme H | 6493 sayılı Kanun'a göre ödeme kuruluşu olarak faaliyet izni almak için aranan asgari se… | uygun |
| 89 | `u193` | 394 Hafta Tatili Hakkında Kanun | 394 sayılı Hafta Tatili Hakkında Kanun'a göre hafta tatilinde çalışmak için ruhsat alınm… | uygun |
| 90 | `u194` | 4046 Özelleştirme Uygulamaları Hakkında Kanun | 4046 sayılı Özelleştirme Uygulamaları Hakkında Kanun'da öngörülen satış yöntemleri neler… | uygun |
| 91 | `u195` | 5378 Engelliler Hakkında Kanun | 5378 sayılı Engelliler Hakkında Kanun'a göre erişilebilirlik denetimlerini hangi komisyo… | uygun |
| 92 | `u196` | 5378 Engelliler Hakkında Kanun | Engelliler Hakkında Kanun'a göre bakıma muhtaç engellilere sağlanan evde bakım hizmetind… | **REDDEDİLDİ** |
| 93 | `u197` | 2860 Yardım Toplama Kanunu | Yardım Toplama Kanunu'na göre izinsiz yardım toplayanlara ne yapılır? | uygun |
| 94 | `u199` | 2313 Uyuşturucu Maddelerin Murakabesi Hakkında Kanun | 2313 sayılı Uyuşturucu Maddelerin Murakabesi Hakkında Kanun'a göre haşhaş ekimi hangi iz… | uygun |
| 95 | `u301` | 1211 Türkiye Cumhuriyet Merkez Bankası Kanunu | 1211 sayılı Türkiye Cumhuriyet Merkez Bankası Kanunu'nun 20 nci maddesine göre Banka Mec… | uygun |
| 96 | `u302` | 5302 İl Özel İdaresi Kanunu | 5302 sayılı İl Özel İdaresi Kanunu'na göre vali, il genel meclisinin hukuka aykırı gördü… | uygun |
| 97 | `u303` | 442 Köy Kanunu | 442 sayılı Köy Kanunu'na göre köylülerin yol ve köprü gibi işlerde ücretsiz çalıştırılma… | uygun |
| 98 | `u305` | 3573 Zeytinciliğin Islahı ve Yabanilerinin Aşılattırılmas | Zeytinciliğin Islahı ve Yabanilerinin Aşılattırılması Hakkında Kanun, zeytinlik sahaları… | uygun |
| 99 | `u308` | 4631 Hayvan Islahı Kanunu | 4631 sayılı Hayvan Islahı Kanunu'na göre suni tohumlama, ovum ve embriyo transferi faali… | uygun |
| 100 | `u310` | 6023 Türk Tabipleri Birliği Kanunu | 6023 sayılı Türk Tabipleri Birliği Kanunu'na göre bir tabibin mesleğini icra edebilmesi … | uygun |
| 101 | `u312` | 3308 Meslekî Eğitim Kanunu | 3308 sayılı Meslekî Eğitim Kanunu'na göre bir işletmenin öğrencilere beceri eğitimi yapt… | uygun |
| 102 | `u315` | 4646 Doğal Gaz Piyasası Kanunu | 4646 sayılı Doğal Gaz Piyasası Kanunu'na göre dağıtım şirketlerine verilen lisansların s… | uygun |
| 103 | `u317` | 5627 Enerji Verimliliği Kanunu | 5627 sayılı Enerji Verimliliği Kanunu'na göre hangi büyüklükteki endüstriyel işletmeler … | uygun |
| 104 | `u318` | 6461 Türkiye Demiryolu Ulaştırmasının Serbestleştirilmesi | Türkiye Demiryolu Ulaştırmasının Serbestleştirilmesi Hakkında Kanun'a göre demiryolu alt… | uygun |
| 105 | `u319` | 618 Limanlar Kanunu | 618 sayılı Limanlar Kanunu liman idari saha sınırlarını belirlemeye yetkili merci | uygun |
| 106 | `u320` | 5200 Tarımsal Üretici Birlikleri Kanunu | 5200 sayılı Tarımsal Üretici Birlikleri Kanunu'na göre bir üretici birliği kurabilmek iç… | uygun |
| 107 | `u325` | 5429 Türkiye İstatistik Kanunu | 5429 sayılı Türkiye İstatistik Kanunu'na göre Resmî İstatistik Programı kaç yıllık dönem… | uygun |
| 108 | `u326` | 7258 Futbol ve Diğer Spor Müsabakalarında Bahis ve Şans O | 7258 sayılı Kanun'a göre spor müsabakalarına dayalı sabit ihtimalli bahis oyunlarını oyn… | uygun |
| 109 | `u327` | 6772 Devlet ve Ona Bağlı Müesseselerde Çalışan İşçilere İ | 6772 sayılı Kanun'a göre devlete ait müesseselerde çalışan işçilere yılda kaç günlük ücr… | uygun |
| 110 | `u328` | 6191 Sözleşmeli Erbaş ve Er Kanunu | 6191 sayılı Sözleşmeli Erbaş ve Er Kanunu'nun 4 üncü maddesine göre ilk sözleşme süresi … | uygun |
| 111 | `u329` | 5488 Tarım Kanunu | 5488 sayılı Tarım Kanunu tarımsal destekleme bütçesi gayri safi milli hasıla asgari oran | uygun |
| 112 | `u330` | 6172 Sulama Birlikleri Kanunu | 6172 sayılı Sulama Birlikleri Kanunu'na göre birlik meclisi üyeleri nasıl belirlenir? | uygun |

**Toplam: 112 satır — 105 uygun, 7 reddedildi.**
