# `canary_v2.jsonl` — köken ve doğrulama künyesi

**Bu set insan-doğrulanmış bir benchmark DEĞİLDİR.** 62 satırın 3'ü insan
doğrulamasından geçmiştir. Bu dosya, v1'den farkını ve sayıların neyi ifade
edip etmediğini kayda geçirir.

## 1. v2, v1'in ÜST KÜMESİDİR

v1'in 48 satırı v2'ye **birebir** taşındı — `question_id`'ler, altın sayfalar,
kanıt alıntıları, doğrulama künyeleri değişmedi. `canary_v1.jsonl` dondurulmuş
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
| `paraphrase` | 7 | **21** | 14 yeni taslak (`c401`–`c414`) |
| diğer sekiz dilim | 41 | 41 | değişmedi |
| **toplam** | **48** | **62** | |

Yeni 14 satır, korpusta o güne dek **hiç altın sayfası olmayan** yedi kanundan
üretildi (`k6502`, `k6331`, `k634`, `k5941`, `k6284`, `k4054`, `k5490`) — set
11 belgeye yığılmışken belge çeşitliliğini artırmak için.

Hepsi `verification_status: "draft"`, `source_type: "ajan-taslak"`. **Hiçbiri
insan onayından geçmemiştir.**

## 3. Yeni makine kapısı: paraphrase sözlüksel örtüşme

Çapraz-kontrol turunun v1'de bulduğu altı kusurun üçü (`c001`, `c002`, `c108`)
tek bir hataydı: soru, altın sayfanın KENDİ sözcüklerini tekrarlıyor ama
`paraphrase` etiketli. Bu, dilimi ölçtüğü şeyden koparır — "yeniden ifade
edilmiş sorgu" dilimi aslında birebir kelime eşleşmesini ölçer ve BM25 haksız
yere iyi görünür.

`verify_canary.py --report` artık bunu ölçer: soru ile altın sayfa, ÜRETİM
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

`c209` maddenin operatif sözcüklerini ("derhal", "bildirilmek", "zorunda")
taşıyor; `c208` İş K. m.6'nın terimlerini. Bunlar insan incelemesini bekliyor;
`paraphrase` dışına çıkarılırlarsa dilim 21'den 17'ye iner.

Kapı yeni taslakları da denetledi: `c404` ilk yazımda %57 ile takıldı ve
yeniden yazıldı (%18). Yani kural yalnız eski veriye değil, üretim sürecine de
uygulanıyor.

## 4. 14 satır eklemek ölçümü NASIL değiştirdi

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

## 5. Bilinen kısıtlar

1. **Alıntılanamaz.** Bu set üzerinde ölçülen hiçbir sayı "insan-doğrulanmış
   benchmark üzerinde ölçüldü" diye sunulamaz. `--status` hedefi iki ayrı
   sayıyla basar; yalnız-insan sayısı **3**'tür.
2. **Korelasyonlu kör nokta sürüyor.** Yeni 14 satırı yazan model, v1'i çapraz
   kontrol eden modelle aynı ailedendir (`claude-opus-5`). Sözlüksel örtüşme
   kapısı bu riski `paraphrase` diliminde ölçülebilir biçimde azaltır ama
   ortadan kaldırmaz; asıl çare karar dilimlerinin %100 insan onayıdır.
3. **Taslaklar ölçüme dahil.** Yukarıdaki 62 satırlık sayı `--all` ile
   alınmıştır, yani 14 taslağı da içerir. İnsan incelemesi bittiğinde
   `--min-verification human` ile ayrıca ölçülecektir.
4. **`tarihi-tarama` hâlâ n=4.** D1 planındaki genişletme yapılmadı; o dilim
   D4 (RG yeniden OCR) deneyinin guardrail'idir ve ayrıca ele alınacaktır.

## 6. İnceleme nasıl yapılır

```bash
uv run python scripts/review_server.py --by baran          # HTML arayüz, karar dilimleri
uv run python scripts/verify_canary.py --status            # sayım
uv run python scripts/verify_canary.py --report            # makine ön-kontrolü
```

Arayüz yalnız `127.0.0.1`'e bağlanır. Her karar diske ATOMİK yazılır; sekme
kapansa da veri bütün kalır.
