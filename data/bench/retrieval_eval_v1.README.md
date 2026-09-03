# `retrieval_eval_v1.jsonl` — köken ve doğrulama künyesi

**Bu set insan-doğrulanmış bir benchmark DEĞİLDİR.** 48 satırın 3'ü insan
doğrulamasından geçmiştir; 45'i bağımsız bir model turunun çapraz-kontrolünden.
Bu dosya, sayıların nasıl üretildiğini ve neyi ifade edip etmediğini kayda
geçirmek için vardır — `verification_status: "verified"` gören biri bunu insan
onayı sanmasın diye.

| Doğrulama türü (`verification_kind`) | Satır | `verified_by` |
|---|---|---|
| `human` | 3 (`c307`, `c308`, `c314`) | `baran` |
| `model-cross-check` | 45 | `model-cross-check:claude-opus-5` |
| **toplam** | **48** | |

## 1. Set nasıl üretildi

Sorular model ajanları tarafından **sayfa görüntülerinden taslak olarak**
yazıldı (`data/images/<doc_id>/<sayfa:04d>.webp`; kaynak PDF'ler `data/pdf/`).
Ajan her soru için altın belge/sayfa/madde kimliklerini, minimal kanıt
alıntılarını ve bir referans cevap önerdi. Bu aşamanın çıktısı **taslaktı**:
48/48 satır `verification_status: "draft"`, `verified_by` boş.

## 2. Nasıl doğrulandı

- **3 satır (`c307`, `c308`, `c314`) — insan.** Proje sahibi sayfa görüntüsünü
  kendisi açıp okudu. Bu satırlar bu turda içerik olarak DEĞİŞTİRİLMEDİ; yalnız
  `verification_kind: "human"` etiketi eklendi.
- **45 satır — model çapraz-kontrolü.** Taslağı yazan turdan ayrı, dört bağımsız
  model ajanı aynı sayfa görüntülerini **yeniden okudu** ve her satır için üç
  karardan birini verdi: `dogru` (39), `duzeltme` (5), `hatali` (1). Kararlar
  satır satır gerekçe notu taşır; not, satırın `verification_note` alanına
  yazılmıştır (düzeltilen satırlarda `"düzeltildi: "` önekiyle). Bu satırların
  `verified_by` alanı bilerek `model-cross-check:claude-opus-5` — bir insan adı
  değil.

## 3. Sayılar için ne anlama geliyor (kısıtlar)

1. **Bu set insan-doğrulanmış diye alıntılanamaz.** Üzerinde ölçülen hiçbir
   recall/precision rakamı "insan-doğrulanmış benchmark üzerinde ölçüldü"
   ifadesiyle sunulamaz. `scripts/verify_retrieval_eval.py --status` bu yüzden hedefi
   iki ayrı sayıyla basar: birleşik ve yalnız-insan. Yalnız-insan sayısı **3**,
   yani P0 planındaki ">=25 cevaplanabilir + >=5 cevaplanamaz insan
   doğrulaması" kapısı **hâlâ açıktır**.
2. **Doğrulayan model, taslağı yazan modelle aynı aileden.** Bağımsız bir tur
   olsa da bağımsız bir *ölçüm aracı* değil: her iki tur da aynı model
   ailesinin görme/okuma davranışını paylaşıyor, dolayısıyla **korelasyonlu kör
   noktalar** mümkün. Tipik olarak bir insanın yakalayacağı ama bu kurulumun
   yapısal olarak kaçırabileceği hata sınıfları: her iki turun da aynı şekilde
   yanlış okuduğu tablo hücreleri, taranmış/düşük kaliteli sayfalardaki OCR
   benzeri hatalar, ve "sayfada geçiyor ama soruyu hukuken karşılamıyor" türü
   hukuki muhakeme hataları.
3. **Alıntı eşleşmesi ayrı ve makine tarafından kontrol edilebilir.**
   `scripts/verify_retrieval_eval.py --report` her kanıt alıntısını PDF metnine karşı
   birebir arar ve altın sayfaların üretim indeksinde bulunduğunu doğrular; bu
   kontrol modelden bağımsızdır ve bu turun bulgularının üstünde ayrı bir
   güvenlik ağıdır. Bu tur sonrası koşumda **45 model çapraz-kontrollü satırın
   45'i de TEMİZ** (`c213` dahil — düzeltilen üçüncü şart artık `k213:9`
   metniyle birebir eşleşiyor, yani düzeltme makinece de teyitli). Kalan 3
   ŞÜPHELİ satır tam olarak insan doğrulamalı olanlardır (`c307`, `c308`,
   `c314`) ve nedenleri gerçek hata değil PDF metin çıkarma artefaktlarıdır
   (satır kaydırma; eski Resmî Gazete taramalarında `"B u kanun"` gibi araya
   giren boşluklar).

## 4. Turun bulduğu somut kusurlar (rubber stamp olmadığının kanıtı)

Tur 45 satırın 6'sında (%13) gerçek kusur buldu. Onay makinesi olsaydı bu sayı
0 olurdu.

| Satır | Karar | Kusur | Uygulanan düzeltme |
|---|---|---|---|
| `c001` | `duzeltme` | Soru maddenin kendi terimini ("yerleşim yeri") aynen kullanıyor; `paraphrase` dilimi yanlış etiket | `slice`: `paraphrase` → `dogrudan-madde` |
| `c002` | `duzeltme` | Aynı hata: "Yerleşim yeri nedir?" madde terimini birebir tekrarlıyor | `slice`: `paraphrase` → `dogrudan-madde` |
| `c108` | `duzeltme` | Soru TBK m.28'in anahtar terimlerini ("zor durum", "yararlanılarak") aynen taşıyor | `slice`: `paraphrase` → `dogrudan-madde` |
| `c203` | `duzeltme` | Tek başına `"e) Ev hizmetlerinde,"` bendi cevabı kanıtlamıyor — istisna cümlesi (madde başı) olmadan bendin ne anlama geldiği belirsiz | `minimal_evidence_spans`'a madde başı cümlesi eklendi |
| `c312` | `duzeltme` | Soru hiçbir madde numarası anmıyor, ama `query_style` `madde-referansli` etiketliydi | `query_style`: `madde-referansli` → `hukuki` |
| `c213` | `hatali` | **Sayfa-aralığı hatası.** VUK m.17'nin üç şartından üçüncüsü ("3. Mühletin verilmesi halinde verginin alınması tehlikeye girmemelidir.") altın sayfada (`k213:8`) değil, SONRAKİ sayfada. Doğrulayıcı bunu `data/images/k213/0009.webp`'i açarak teyit etti. Soru altın sayfadan eksiksiz cevaplanamıyordu ve referans cevap üç şarttan ikisini veriyordu | `gold_page_ids`: `["k213:8"]` → `["k213:8", "k213:9"]`; üçüncü şart kanıt alıntılarına ve referans cevaba eklendi. `k213:9`'un üretim indeksinde (`page_ids.json`) bulunduğu yazmadan önce teyit edildi |

`c213` bu turun en değerli bulgusu: dilim etiketi hatası ölçümü hafifçe kaydırır,
ama eksik altın sayfa **retrieval'ı haksız yere cezalandıran** bir hatadır —
doğru sayfayı getiren bir sistem yine de kısmi cevap üretmek zorunda kalırdı.

## 5. Bunu insan-doğrulanmış hale getirmek için (bilinen araç boşluğu)

Kapı, yalnız-insan sayısı >=25 cevaplanabilir + >=5 cevaplanamaz olduğunda
kapanır. O ana kadar bu setin üzerindeki her rakam "model çapraz-kontrollü set
ölçümü" etiketiyle anılmalıdır.

**Ancak `--review` bugün bu işi yapamaz — önce iki küçük değişiklik gerekiyor:**

1. `cmd_review`'ın kuyruğu yalnız `verification_status == "draft"` satırları
   alıyor. Bu tur 45 satırı `verified` yaptığı için kuyruk şu an **boş**;
   `--review --by baran` "gözden geçirilecek taslak soru yok" der. Kuyruk,
   `verification_kind == "model-cross-check"` satırlarını da kapsayacak şekilde
   genişletilmeli (insan onayı bekleyen satırlar tam olarak bunlar).
2. `apply_decision` `verification_status`/`verified_by` yazıyor ama
   `verification_kind`'a dokunmuyor. Genişletilmiş kuyrukta bir insan onayı,
   satırı `model-cross-check` etiketiyle bırakırdı — yani insan emeği sayıma
   girmezdi. `--review` insan yolu olduğuna göre `apply_decision` onay/ret
   kararında `verification_kind`'ı `"human"` yapmalı.

Bu ikisi yapılana kadar insan doğrulaması yalnız dosyayı elle düzenleyerek
kaydedilebilir. Bu boşluk bilerek burada yazılıdır: aksi halde "insan
doğrulaması yapılabilir" varsayımı sessizce yanlış kalırdı.
