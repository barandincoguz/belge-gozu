# `retrieval_eval_v2.jsonl` — köken ve doğrulama künyesi

**66 satırın 47'si insan doğrulamasından geçmiştir.** G1.2'nin üzerinde hüküm
verdiği dört karar diliminin (`paraphrase`, `dogrudan-madde`, `madde-numarali`,
`ayni-kanun-hard-negative`) tamamı insan onaylıdır — v1'de bu oran 3/48'di.
Kalan 19 satır (dört yeni taslak + karar dışı dilimler) model çapraz-kontrolüyle
sınırlıdır. Bu dosya, v1'den farkını ve sayıların neyi ifade edip etmediğini
kayda geçirir.

## 1. v2, v1'in ÜST KÜMESİDİR

v1'in 48 satırı v2'ye **birebir** taşındı — `question_id`'ler, altın sayfalar,
kanıt alıntıları, doğrulama künyeleri değişmedi. `retrieval_eval_v1.jsonl` dondurulmuş
tarihsel referans olarak repoda durur.

Bu yapı bilinçli bir karardır: v2 v1'i tamamen içerdiği için, v1 alt kümesinde
yayımlanmış her sayı yeniden üretilebilir. D1 kurulumunda bu bir **saf kontrol**
olarak koşuldu — v2 henüz v1'in birebir kopyasıyken ölçüm yapıldı ve
yayımlanmış her kapı sayısı aynen çıktı:

| Metrik | Yayımlanan | v2 kontrolü (48 satır) |
|---|---|---|
| fractional R@5 | 0,8488 | `0.8488372093023255` |
| R@20 / R@50 (G1.1) | 0,9302 | `0.9302325581395349` |
| `paraphrase` R@5 (G1.2) | 0,2857 | `0.2857142857142857` |
| `paraphrase` R@50 (G1.2) | 0,5714 | `0.5714285714285714` |

Kanıt: `data/bench/results/d1-v2-control.json`.

## 2. Eklenen satırlar

| Dilim | v1 | v2 | Not |
|---|---|---|---|
| `paraphrase` | 7 | **25** | 18 yeni satır (`c401`–`c418`) |
| diğer sekiz dilim | 41 | 41 | değişmedi |
| **toplam** | **48** | **66** | |

Yeni 18 satır, korpusta o güne dek **hiç altın sayfası olmayan** on kanundan
üretildi (`k6502`, `k6331`, `k634`, `k5941`, `k6284`, `k4054`, `k5490`,
`k5846`, `k5651`, `k5188`) — set 11 belgeye yığılmışken belge çeşitliliğini
artırmak için.

`c401`–`c414` insan onayından geçti; `c415`–`c418` hâlâ taslaktır.

## 3. Yeni makine kapısı: paraphrase sözlüksel örtüşme

Çapraz-kontrol turunun v1'de bulduğu altı kusurun üçü (`c001`, `c002`, `c108`)
tek bir hataydı: soru, altın sayfanın KENDİ sözcüklerini tekrarlıyor ama
`paraphrase` etiketli. Bu, dilimi ölçtüğü şeyden koparır — "yeniden ifade
edilmiş sorgu" dilimi aslında birebir kelime eşleşmesini ölçer ve BM25 haksız
yere iyi görünür.

`verify_retrieval_eval.py --report` artık bunu ölçer: soru ile altın sayfa, ÜRETİM
tokenizer'ıyla (`retrieval.text.tokenize`) tokenize edilir ve sorunun içerik
sözcüklerinin kaçta kaçının sayfada birebir geçtiği hesaplanır. `paraphrase`
diliminde %50'yi aşan satır ŞÜPHELİ'ye düşer. Kapı **etiket değiştirmez**,
yalnız insan incelemesine taşır.

İlk koşuda dört mevcut satır yakalandı:

| Satır | Örtüşme | Örtüşen sözcükler |
|---|---|---|
| `c110` | %55 | ceza, cocuk, hakki, sayil, suc, yasin |
| `c111` | %58 | cezal, dolay, kanun, kendi, kulla, sonuc, yetki |
| `c208` | %64 | baska, calis, is, isver, isyer, neden, sirke, sona, sozle |
| `c209` | %57 | alina, bildi, derha, zorun |

**İnsan incelemesi dördünü de `paraphrase` olarak DOĞRULADI.** Gerekçeler
satırların `verification_note` alanında: `c110` yaş eşiğini "onbir yaşındaki
çocuk" diyerek kendi ifadesiyle kuruyor; `c111` madde terimi "hak" yerine
"kanunun tanıdığı yetki" diyor. Yani örtüşme, Türkçe'nin eklemeli yapısı ve
hukuk sözlüğünün darlığından geliyor, kopyalamadan değil.

**Kapının bu turdaki bilançosu dürüstçe: 4 yanlış pozitif, 0 gerçek bulgu.**
Eşik yine de 0,5'te BIRAKILDI ve veriye uydurulmadı — ölçüm aracını deneyin
nesnesi yapmak bu projede yasak. Kapı bir *reddedici* değil bir *yüzeye
çıkarıcıdır*: maliyeti 4 ek inceleme oldu, karşılığında o dört satırın neden
paraphrase sayıldığı artık yazılı kayıtta duruyor. v1'i düzelten üç kusur
(`c001`, `c002`, `c108`) %100 örtüşmedeydi, yani kapının yakalamak için var
olduğu sınıf eşiğin çok üstünde.

Kapı yeni taslakları da denetledi: `c404` ilk yazımda %57, `c417` %64 ile
takıldı ve ikisi de yeniden yazıldı (%18 ve %25). Yani kural yalnız eski veriye
değil, üretim sürecine de uygulanıyor.

## 4. Ara adım: 14 satır eklemek ölçümü NASIL değiştirdi

| | 48 satır (v1≡v2) | 62 satır | |
|---|---|---|---|
| genel R@5 | 0,8488 (n=43) | **0,6930** (n=57) | −0,156 |
| genel R@50 | 0,9302 | **0,8421** | G1.1 eşiği 0,95'ten daha da uzak |
| `paraphrase` R@5 | 0,2857 (n=7) | 0,2381 (n=21) | |
| `paraphrase` R@50 | 0,5714 (n=7) | 0,5714 (n=21) | aynı sayı, **3× örneklem** |

**Genel sayıdaki düşüş sistemin kötüleşmesi DEĞİLDİR.** Set artık zor dilime
daha çok ağırlık veriyor: `paraphrase` payı %16'dan (7/43) %37'ye (21/57)
çıktı. Bu, yayımlanan 0,8488'in bir sistem özelliği kadar bir **dilim karışımı
özelliği** olduğunu gösterir — D1'in ortaya çıkarmak için var olduğu şey tam
olarak budur. İki set arasındaki genel sayılar bu yüzden doğrudan
karşılaştırılamaz; karşılaştırılabilir olan dilim-içi sayılardır.

`paraphrase` R@50'nin 0,5714'te sabit kalması ise iyi haberdir: n=7'lik tahmin
kabaca doğruymuş, ama artık n=21 ile çok daha dar hata payıyla biliniyor.

Kanıt: `data/bench/results/d1-v2-62row.json`.

## 4b. İnsan doğrulamasından sonra — D1'in asıl çıktısı

44 satırlık kuyruk incelendi: **0 ret, 0 içerik düzeltmesi**, 44 satırda yalnız
künye alanları değişti. Karar dilimleri artık %100 insan onaylı ve ilk kez
insan-doğrulanmış bir getirim ölçümü mümkün oldu.

| Ölçüm | R@5 | n | Not |
|---|---|---|---|
| Yayımlanan (v1) | 0,8488 | 43 | 3 satır insan onaylı |
| v2, tüm satırlar | 0,6930 | 57 | taslaklar dahil |
| **v2, yalnız insan** | **0,6277** | **47** | `--min-verification human` |

Üç sayı arasındaki fark büyük ölçüde **dilim karışımıdır**, sistem kalitesi
değil: insan altkümesi `tablo-layout` ve `capraz-kanun-terim` gibi R@5=1,0000
alan kolay dilimleri dışarıda bırakır ve `paraphrase` payını daha da artırır.
Genel sayılar bu yüzden setler arası karşılaştırmaya UYGUN DEĞİLDİR.

Karşılaştırılabilir olan dilim-içi sayılardır ve asıl sonuç budur:

| Karar dilimi (G1.2) | R@50, insan-doğrulanmış | n | Hüküm |
|---|---|---|---|
| `dogrudan-madde` | 1,0000 | 13 | 1,0000 iddiası insan denetiminden SAĞ ÇIKTI |
| `madde-numarali` | 1,0000 | 6 | ayakta |
| `ayni-kanun-hard-negative` | 1,0000 | 5 | ayakta |
| `paraphrase` | **0,5714** | **21** | n=7'deki değerin BİREBİR aynısı |

`paraphrase` R@50'nin üç kat örneklemde rakama kadar aynı çıkması, o dilimdeki
başarısızlığın örneklem gürültüsü olmadığının en güçlü kanıtıdır. **G1.2 hükmü
değişmedi ama artık savunulabilir.** G1.1 tarafında R@50 (tüm satırlar) 0,9302
yerine 0,8421 — 0,95 eşiğinden daha da uzak.

Kanıt: `data/bench/results/d1-final-all.json`, `d1-final-human.json`.

## 5. Bilinen kısıtlar

1. **Hangi sayının insan-doğrulanmış olduğuna dikkat.** `--all` ile alınan
   genel sayı 19 model-onaylı satırı da içerir ve "insan-doğrulanmış benchmark
   üzerinde ölçüldü" diye sunulamaz. İnsan iddiası yalnız
   `--min-verification human` çıktısı ve dört karar dilimi için geçerlidir
   (47/66). `--status` iki sayıyı ayrı ayrı basar.
2. **Korelasyonlu kör nokta karar dilimlerinde KAPANDI, dışında sürüyor.** Yeni
   18 satırı yazan model, v1'i çapraz kontrol eden modelle aynı ailedendir
   (`claude-opus-5`). Dört karar diliminin %100 insan onayı bu riski oralarda
   ortadan kaldırır; `tablo-layout`, `capraz-kanun-terim`, `tarihi-tarama` ve
   cevaplanamaz dilimlerde risk aynen durur.
3. **Dört satır hâlâ taslak.** `c415`–`c418` insan onayı bekliyor; genel
   sayılar `--all` ile alındığında bu dördünü de içerir.
4. **`tarihi-tarama` hâlâ n=4.** D1 planındaki genişletme yapılmadı; o dilim
   D4 (RG yeniden OCR) deneyinin guardrail'idir ve ayrıca ele alınacaktır.

## 6. İnceleme nasıl yapılır

```bash
uv run python scripts/review_server.py --by baran          # HTML arayüz, karar dilimleri
uv run python scripts/verify_retrieval_eval.py --status            # sayım
uv run python scripts/verify_retrieval_eval.py --report            # makine ön-kontrolü
```

Arayüz yalnız `127.0.0.1`'e bağlanır. Her karar diske ATOMİK yazılır; sekme
kapansa da veri bütün kalır.
