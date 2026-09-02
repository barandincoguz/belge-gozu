# Belge-Gözü

[![ci](https://github.com/barandincoguz/belge-gozu/actions/workflows/ci.yml/badge.svg)](https://github.com/barandincoguz/belge-gozu/actions/workflows/ci.yml)

**4.222 sayfalık Türk mevzuatı üzerinde, dayanağı gösterilen soru-cevap — ölçümle inşa edildi.**
*Grounded question answering over 4,222 pages of Turkish legislation — built measurement-first.*

**[Türkçe](#türkçe) · [English](#english)**

| | |
|---|---|
| **Korpus / Corpus** | 4.222 sayfa · 56 belge (50 kanun + 6 taranmış *Resmî Gazete*, 1928–1975) |
| **Getirim kalitesi / Retrieval** | Recall@5 **0.8605** (37/43) · Recall@20 0.930 — aksanlı ve aksansız yazımda aynı |
| **Başlangıç / Starting point** | Aynı kıyas kümesinde Recall@5 **0.116** |
| **Gecikme / Latency** | metin getirimi 2–5 ms · uçtan uca cevap 6–24 sn (LLM'e bağlı) |
| **Mühendislik / Engineering** | 707 test · CI hem süiti koşar hem dağıtım imajını derler · her sayı tarihli bir koşum artefaktına bağlı |
| **Yığın / Stack** | Python 3.12 · FastAPI · PyTorch · Transformers · ColPali-class vision encoder · BM25 (hand-written) · SQLite · Prometheus · Grafana · Docker · GitHub Actions · pytest · ruff · pyright · uv |

İndeks ve tüm sayfa görüntüleri Hugging Face Datasets üzerinde herkese açık:
**[barandincoguz/belge-gozu-index](https://huggingface.co/datasets/barandincoguz/belge-gozu-index)**
Üretim pull'u, içinde int8 vektörler ve hizalı metin kanalı bulunan değişmez
[`700ac324...`](https://huggingface.co/datasets/barandincoguz/belge-gozu-index/tree/700ac324fffefb22de02c8e90347b31185547948)
revizyonuna sabitlenmiştir.

Getirim kazancı bütün P1 kapılarını geçmiş sayılmaz. Resmî
[P1/G1 kararı](docs/research/findings/2026-08-31-p1-gate.md) **FAIL**'dir:
Recall@50 ve paraphrase dilimi hedefi kaçırdı; reranker ve canlı dağıtım
bütçeleri henüz ölçülmedi.

---

# Türkçe

## 1. Amaç

Türk mevzuatı kamuya açıktır ama pratikte aranabilir değildir. Metin, düzeni anlam taşıyan
PDF'lerin içinde durur: madde numaraları, tarife tabloları, kenar notları, 1928'den
taranmış gazeteler. Anahtar kelime araması kanunu bulur ama *sayfayı* bulmaz; genel amaçlı
bir sohbet modeli ise var olmayan bir madde numarasını akıcı bir dille söyler.

Belge-Gözü tam ters arıza moduna göre kuruldu: **madde uydurmaktansa "bulamadım" demeyi
tercih eder.** Cevaplar yalnızca getirilen sayfalardan üretilir, her iddia sayfa atıfı
taşır ve sayfa görüntüsü ekranda gösterilir — insan kendi gözüyle doğrulayabilsin diye.

Proje aynı zamanda yöntem üzerine bir iddiadır. Hiçbir şey sayı olmadan öne sürülmez, her
sayı künyesini taşır (hangi indeks, hangi korpus özeti, hangi commit) ve reddedilen
deneyler ölçümleriyle birlikte depoda kalır.

## 2. Problem

İlk çalışan sürüm sayfa *görüntülerini* ColPali sınıfı bir görsel-dil modeliyle getiriyordu:
sayfa ekran görüntüleri üzerinde geç etkileşimli MaxSim, OCR yok, düzen ayrıştırma yok.
Doğru görünüyordu; ölçümde yanlış çıktı.

```mermaid
flowchart TB
    Q["Query: annual paid leave<br/>under the Labour Act?"] --> V["Visual retrieval<br/>4,222 pages"]
    V --> D["Right document<br/>cover page, rank 1"]
    V --> P["Right page<br/>art. 53 table, rank 137"]
    P --> A["'I could not find it<br/>in the given pages'"]
    classDef ok fill:#e8f3ec,stroke:#2e7d4f,color:#14321f
    classDef bad fill:#fceceb,stroke:#a61c2c,color:#3d1015
    classDef neutral fill:#eef2f7,stroke:#2c5b8a,color:#12283d
    class D ok
    class P,A bad
    class Q,V neutral
```

Model kanunun *kimliğini* — başlık sayfasını — yakalıyor, maddeyi kaybediyordu. Dürüst
"bulamadım" cevabı, yanlış kanıt üzerinde doğru davranıştı. Ölçüldüğünde arıza netti:
**Recall@5 = 0.116** ve kullanıcının ilk sırada beklediği yerleşim yeri tanımı sayfası
4.222 sayfa içinde 3.127. sıradaydı.

Kök neden, tahminle değil ölçümle: kodlayıcı ağırlıklı olarak İngilizce eğitilmiştir ve
uzun bir Türkçe hukuk sorgusu, maddenin gövde metnine değil dokümanın *adına* çok daha
güçlü hizalanır.

## 3. Mühendislik haritası

Her adım tek bir kontrollü deneydir: tek değişken, tek birincil metrik (43 soruluk kıyas
kümesinde Recall@5), donmuş bir ölçüm düzeneği ve tut-ya-da-geri-al kararı. **Kesikli
dallar ölçüldü ve reddedildi** — portfolyoların çoğunun sildiği kısım.

```mermaid
flowchart TB
    B["visual only<br/>0.233"] --> T["+ BM25 text<br/>0.674"]
    B -.->|rejected| R1["equal RRF<br/>0.395"]
    T --> F["+ prefix<br/>0.767"]
    T -.->|rejected| BG["bigrams<br/>0.628"]
    F --> S["+ stopwords<br/>0.767"]
    S --> W["+ routing w20<br/>0.814"]
    S -.->|rejected| AP["hard partition<br/>0.791"]
    W --> W5["window 50<br/>0.837"]
    W5 -.->|rejected| WR["window RRF<br/>0.535"]
    W5 --> FD["+ folding<br/>0.8605"]
    FD -.->|rejected| DF["dual-form<br/>0.837"]
    classDef kept fill:#e8f3ec,stroke:#2e7d4f,color:#14321f
    classDef gone fill:#faf3e4,stroke:#a8741a,color:#3a2a08
    class B,T,F,S,W,W5,FD kept
    class R1,BG,AP,WR,DF gone
```

Diyagram render edilmiyorsa (ör. GitHub mobil uygulaması Mermaid çizmez) aynı zincir tablo hâlinde:

| Adım | R@5 | Karar |
|---|---|---|
| yalnız görsel kanal | 0.233 | taban |
| + BM25 metin kanalı | **0.674** | tutuldu |
| eşit ağırlıklı RRF füzyonu | 0.395 | reddedildi |
| + 5 karakterlik ön ek | **0.767** | tutuldu |
| + bigram parçacıkları | 0.628 | reddedildi |
| + işlev kelimesi listesi | 0.767 | tutuldu (derin sıralar düzeldi) |
| + kanun-adı yönlendirmesi (pencere 20) | **0.814** | tutuldu |
| mutlak doküman bölümleme | 0.791 | reddedildi (korkuluk vetosu) |
| pencere 50 | **0.837** | tutuldu |
| pencere içi RRF | 0.535 | reddedildi |
| + aksan katlama | **0.8605** | tutuldu — yayında |
| çift-biçim token | 0.837 | reddedildi (iki korkuluk düştü) |

Üç sonuç açıkça yazılmayı hak ediyor, çünkü üçü de bariz yaklaşımı çürütüyor:

**Reciprocal Rank Fusion işi kötüleştirdi — üç kez.** Ders kitabı hamlesi görsel ve metin
sıralamalarını birleştirmektir. Eşit ağırlıklı RRF, Recall@5'i 0.674'ten 0.395'e düşürdü;
mutlak doküman bölümleme bir korkuluğu deldi; pencere içi RRF 0.535 verdi. Zayıf kanalın
başlık sayfasına olan çekimi, denenen her granülarite düzeyinde güçlü kanalın gerçek
isabetlerini eziyor. Ayakta kalan çözüm sözcüksel-birincil sıralama artı kural tabanlı
yeniden düzenlemedir.

**Görsel kanalın ilk 5'e benzersiz katkısı sıfırdır** — metin kanalı ayarlandıktan sonra.
Kanal yine de her sorguda çalışır: telemetriyi ve kalibrasyon veri kümesini besler, ayrıca
metin katmanı zayıf olan 16 sayfa için ayrı bir ölçüm sinyali sağlar. Ama artık
sıralamıyor. Bu, ilk sunuma uyan sonuç değil; ölçümün söylediği sonuçtur.
**Üretimde iki kanal koşar; sıralamayı yalnız BM25 metin kanalı belirler.**

**Aksan işaretleri bir üretim hatasıydı, incelik değil.** Türkçe klavye rutin olarak devre
dışı bırakılır; kullanıcı *"yillik ucretli izin"* yazar. Ölçüldüğünde bu, Recall@5'i
0.837'den 0.581'e düşürüyordu. Aksanları hem indeks hem sorgu tarafında katlamak sistemi
**yazım-değişmez** yapıyor: iki koşulda da 0.8605.

## 4. Mimari

```mermaid
flowchart TB
    PDF["56 PDFs"] --> IMG["page images"]
    PDF --> TXT["text layer"]
    IMG --> EMB["ColSmol-500M<br/>embeddings"]
    EMB --> Q8["int8 index<br/>476 MB"]
    TXT --> BM["BM25 index<br/>fold + prefix"]
    QRY["Turkish question"] --> TOK["tokenise"]
    TOK --> SC["BM25 scoring<br/>2-5 ms"]
    SC --> ROUTE["law-name routing<br/>top 50 reorder"]
    ROUTE --> GATE{"above<br/>threshold?"}
    GATE -->|no| ABS["abstain"]
    GATE -->|yes| LLM["VLM answerer<br/>pages + markers"]
    LLM --> CITE["answer<br/>+ citations"]
    BM -.-> SC
    Q8 -.-> VIS["visual MaxSim<br/>telemetry only"]
    TOK -.-> VIS
    classDef store fill:#eef2f7,stroke:#2c5b8a,color:#12283d
    classDef act fill:#f7f4ec,stroke:#8a6d2c,color:#2f2510
    classDef out fill:#e8f3ec,stroke:#2e7d4f,color:#14321f
    class Q8,BM store
    class TOK,SC,ROUTE,VIS,LLM,EMB,IMG,TXT act
    class CITE,ABS out
```

Üstteki üç kutu çevrimdışı kurulum (bir kez), alttaki zincir her istekte koşar.

**Etkileşimli şemalar.** Yukarıdaki şema özettir. Gezilebilir hâlleri
[şema galerisindedir](https://barandincoguz.github.io/belge-gozu/): [gözlem
katmanı](https://barandincoguz.github.io/belge-gozu/observability.architecture.html) ·
[getirim yolculuğu](https://barandincoguz.github.io/belge-gozu/retrieval.sequence.html) ·
[indeksleme hattı](https://barandincoguz.github.io/belge-gozu/indexing.dataflow.html) ·
[P1/G1 kapı döngüsü](https://barandincoguz.github.io/belge-gozu/p1-gate.lifecycle.html) ·
[görsel-only → hibrit geçişi](https://barandincoguz.github.io/belge-gozu/retrieval-delta.html).
Her figürün tipli JSON kaynağı `docs/diagrams/` altında izlenir; şemalar o kaynaktan
deterministik olarak derlenir, elle çizilmez.

Sistemi ayakta tutan iki kural var:

**Kimlik veriyle birlikte yolculuk eder.** Her indeks bir künye taşır: model revizyonu,
sorgu biçimi, doküman istemi özeti, nicemleme ve korpus özeti. Sunucu, kimliği kendi
yapılandırmasıyla uyuşmayan bir indekse karşı açılmayı reddeder; kalibrasyon artefaktı da
`index_revision × pipeline × recipe_fingerprint` anahtarına bağlıdır. Bu disiplin, bir
değerin kendisine anlam veren bağlamdan koptuğu 139 yeri saptayan bir denetimden doğdu — en
kötüsü, sessizce tek bir nicemleme şemasına bağlanmış bir skor eşiğiydi.

**Bayraklar ve geri alma.** Yeni karar katmanları (kalibre edilmiş kapı, kanıt
doğrulayıcı) varsayılanı kapalı bayraklarla gelir; bayrak kapalıyken sunulan davranışın
bayt düzeyinde aynı kaldığını doğrulayan bir test vardır.

## 5. Teknik detaylar

### Getirim

| Yapılandırma | Recall@5 | Recall@20 | MRR | Gold sayfa sırası, sorgu A / B |
|---|---|---|---|---|
| yalnız görsel, 1-bit (ilk sürüm) | 0.116 | — | — | 3127 / — |
| yalnız görsel, int8 | 0.233 | 0.302 | 0.149 | 664 / 137 |
| hibrit, katlama öncesi | 0.837 | 0.930 | 0.655 | 2 / 2 |
| **hibrit + katlama (yayında)** | **0.8605** | **0.930** | 0.632 | **2 / 2** |

Sağlamlık taraması: BM25 `k1` ∈ [0.9, 1.8] × `b` ∈ [0.5, 0.9] kombinasyonlarının tamamı
0.814–0.837 aralığında kalıyor; ön ek uzunluğu 4–7 bir plato. Reçete bıçak sırtında değil.
Bir soru daha kazandıracak bir ayar bilinçli olarak **alınmadı**; çünkü bu, problemi değil
kıyas kümesini optimize etmek olurdu.

### Nicemleme

int8, her k değerinde float16 ile aynı sıralama kalitesini verir, 1-bit'ten 4.3 kat daha hızlıdır
(CPU'da sorgu başına 0.24 sn'ye karşı 1.08 sn) ve 58 MB yerine 476 MB yer kaplar. 1-bit
hem Recall@20'de 7 puan kaybettirir hem de daha yavaştır — bit paketleme numaraları burada
BLAS'a yeniliyor. Yayında olan int8'dir; diğerleri yeniden üretilebilir ablasyon olarak durur.

### Cevap yolu

Sayfa etiketleri görüntülerin arasına serpiştirilir (`[S1]`, görüntü, `[S2]`, görüntü, …),
böylece bir atıf konuma dayalı tahmine değil belirli bir sayfaya bağlanır. Otomatik atıf
yedeği yoktur: model etiket üretmezse cevap atıfsız kalır. Hatalar sınıflandırılır
(`timeout`, `http_5xx`, `http_429`, `auth`, `safety_block`, `parse`) ve toplam süre bütçesi
bir değişmez olarak uygulanır — kalan bütçe karşılamıyorsa yeniden deneme başlatılamaz.
İki API anahtarı dönüşümlüdür: taşıma düzeyindeki herhangi bir hata isteği diğer anahtara
taşır ve çalışan anahtar yapışkan hale gelir.

### Seçici cevaplama (devam ediyor)

Çekimserlik eşiği, önceki bir çalışma noktasının BM25 ölçeğine **mekanik taşınmasıdır**,
kalibrasyon değildir — ve cevaplanabilir ile cevaplanamaz soruları ayırmaz. Ölçüldüğünde,
sayıyı oynatmak bunu düzeltmiyor:

- Getirim tarafındaki beş özellik üzerine kurulu bir güven modeli, geliştirme bölmesinde AUROC
  0.782'ye ulaşıyor; ama %5 risk bütçesinde soruların yalnız %2.2'sine cevap veriyor.
- 5 cevaplanamaz soruya karşı güçlü görünen sinyaller (AUC 0.94), 151 gerçekçi
  cevaplanamaz soruya karşı 0.68'e düştü — yeni olumsuz örnekler sözcüksel olarak akla yatkın.

Plan değil, bulgu olarak: **getirim tarafı güven tek başına seçici cevaplamayı taşıyamaz.**
İddia düzeyinde bir kanıt doğrulayıcı — cevabı iddialara böl, her iddiayı atıf verdiği
sayfanın metnine karşı sına, desteklenmeyen iddia varsa cevabı düşür — yazıldı ve bayrak
arkasında test edildi. `bench answers`, cevaplanabilir ve cevaplanamaz dev sorularını
aynı künyeli raporda koşturur; atıf precision/completeness ve false-supported-answer
oranını tek taraflı %95 Clopper–Pearson sınırlarıyla yazar. Test yakası ayrı bir
`--yes-final-gate` kilidi taşır. Kapılar kütüphane varsayılanında kapalı kalır.

### Kıyas kümeleri ve künyeleri

- **Canary**: 48 soru (43'ü cevaplanabilir). Yukarıdaki her getirim kararının arkasında bu var.
- **Cevaplanamaz küme**: üç sınıfta 330 soru — korpus dışı, anlamsız ve zor olanı: korpustaki
  bir kanun *hakkında* ama sorulan ayrıntı gerçekten metinde yok.
- Etiketler taslakçı ≠ denetçi rejiminde üretildi. Mekanik etiketler ("çapa kanun 56 belgelik
  künyede yok") CI'da koşan bir betikle yeniden doğrulanır. Örneklemli çapraz kontrol kalıntı
  etiket gürültüsünü %12.5 ölçtü; ardından test yakasının tamamı satır satır, her ret için
  birebir alıntıyla doğrulandı.
- Bölme hukuk-gruplu: 56 belgenin 22'si yalnız test tarafında. Test yakasında 155 cevaplanamaz
  soru var — sıfır hatanın %95 güvenle ≤%2 iddiasını taşıyabildiği büyüklük.

**Bu sayıların geçtiği her yerde tekrarlanan dürüstlük notu:** canary'nin 48 satırından 3'ü
insan tarafından doğrulandı; diğer 45'i ve cevaplanamaz kümenin tamamı model çapraz
kontrolüyle doğrulandı. **Bunlar insan onaylı kıyas kümeleri değildir.**

### İşletim

İstek başına 29 alanlı bir SQLite olay kaydı (hangi işlem hattı, hangi skor ölçeği, hangi API
anahtarı servis etti, model dürüst ıska bildirdi mi), bir Prometheus uç noktası ve hazır
sağlanmış bir Grafana panosu. Girdi doğrulaması boş, aşırı uzun ve bozuk sorguları reddeder;
IP başına tahliyeli hız sınırı ve ham sorgu metnini diske yazmayan gizlilik varsayılanı
konteyner imajında etkindir.
`/ask` ve `/search`, aynı sunucu eşik kararından türetilen `no_match` alanını taşır;
arayüz eşik-altı sonuçları geçerli sayfa kartları gibi göstermez. Etkin sıralama kanalı,
skor etiketi ve eşik `/healthz` tarafından sahiplenilir.

CI; lint, tip denetimi, 707 test ve kıyas bütünlüğü doğrulayıcısını koşar. Ayrı Docker
işi imajı derler; UID 1000, yazılabilir `/data`, CPU-only PyTorch, eksik revizyonda
fail-fast ve `/healthz` smoke sözleşmelerini denetler. İlk iki koşumu kırmızıydı ve 147 yerel commit'in yakalayamadığı iki
taşınabilirlik hatasını yakaladı: terminal rengine bağımlı CLI kontrolleri ve doğrulayıcının
ihtiyaç duyduğu, hiç izlenmemiş korpus künyesi.

## 6. Kapanış

**Bugün çalışan.** Getirim, kalan hataların sözcüksel değil anlamsal olduğu noktaya kadar
çözüldü: çözülmemiş altı kıyas sorusu, hedef sayfalarıyla hiçbir kelime paylaşmayan saf
paraphrase (başka sözcüklerle sorulmuş) sorulardır. Sistem atıflı cevap veriyor, anlamsız sorularda çekimser kalıyor,
aksan eksikliğine dayanıklı ve tükenen API kotasını anahtar değiştirerek atlatıyor.

**Dürüstçe bitmemiş olan.**

| Alan | Durum |
|---|---|
| Seçici cevaplama | Güven modeli kuruldu ve ölçüldü; açılamayacak kadar temkinli. Doğrulayıcı yazıldı, bayrak arkasında. |
| Resmî kapı raporları | Faz 0 geçti. Faz 1, Recall@50 0.930 ve paraphrase 0.571 nedeniyle resmî raporda **FAIL** olarak hükme bağlandı; reranker ve canlı dağıtım bütçeleri ölçülmedi. |
| İnsan doğrulaması | 48 satırın 3'ü. İnsan kapılı bir kıyas kümesi, dürüst olan bir sonraki adımdır. |
| Herkese açık dağıtım | Pinli, kendi kendine yeterli Hub indeksi ile root olmayan CPU Docker/CI sözleşmesi hazır; barındırma ücretli katman istiyor ve uygulama henüz canlıya dağıtılmadı. |
| Madde yapısı ve OCR | Madde düzeyinde hiyerarşi tasarlandı ama hiç kurulmadı; 16 sayfanın metin katmanı zayıf ve OCR yedeği yok. |

Hepsi bu depoda issue olarak izleniyor — yukarıdaki başarısızlıklar dahil, dipnot değil
kayıt olarak.

**Bu projenin iddiası.** İlginç olan kısım model değildi. Kendi tasarımını çürütecek kadar
dürüst bir ölçüm düzeneği kurmaktı: görsel getirim projesi olarak başlayıp ölçümleri
"görsel kanal artık sıralamamalı" diyen bir sistem, kanıtla üç kez reddedilen bir füzyon
stratejisi ve kendi sayıları "yayına hazır değilim" diyen bir güven modeli. Bu kararların
arkasındaki koşumlar `docs/research/findings/` altında, tarihli, ham artefaktlarıyla birlikte.

### Çalıştırma

```bash
uv sync --all-extras
BG_HF_DATASET_REPO=barandincoguz/belge-gozu-index \
BG_HF_REVISION=700ac324fffefb22de02c8e90347b31185547948 \
  uv run belge-gozu index pull
BG_DEVICE=cpu uv run belge-gozu serve --port 7860  # http://localhost:7860
```

`.env` içinde `GOOGLE_API_KEY` (rotasyon için isteğe bağlı `GOOGLE_API_KEY_2`) gerekir.
Testler: `make test` · lint ve tipler: `make lint` · panolar: `make obs-up`

G2 dev ölçümü gerçek API denemesi bütçesini açıkça ister:

```bash
uv run belge-gozu bench answers --split dev --max-llm-attempts 40
```

Docker aynı pinli artefaktı kullanır ve UID/GID 1000 ile çalışır:

```bash
docker build -t belge-gozu:p0 .
docker run --rm -p 7860:7860 -v belge-gozu-data:/data \
  -e GOOGLE_API_KEY="$GOOGLE_API_KEY" belge-gozu:p0
```

### Depo haritası

| Yol | İçeriği |
|---|---|
| `src/belge_gozu/retrieval/` | BM25 metin kanalı, kanun-adı yönlendirmesi, hibrit getirici |
| `src/belge_gozu/index/` | kodlama, nicemleme, künyeler, uyumluluk fail-fast'i |
| `src/belge_gozu/answer/` | cevaplayıcı, anahtar rotasyonu, kalibrasyon, iddia doğrulayıcı |
| `src/belge_gozu/telemetry/` | olay kaydı, Prometheus metrikleri, aşama zamanlaması |
| `docs/research/findings/` | tarihli ölçüm notları — buradaki her sayının gerekçesi |
| `docs/diagrams/` | etkileşimli şemaların tipli JSON kaynağı ve derlenmiş HTML'i |
| `research/` | deney döngüsü: günlük, düzenek, sonuçlar |
| `data/bench/` | kıyas kümeleri, bölmeler ve künye dosyaları |

---

# English

## 1. Purpose

Turkish legislation is public but practically unsearchable. The text lives in PDFs whose
layout carries meaning — article numbers, tariff tables, marginal notes, gazettes scanned
from 1928. Keyword search returns the law but not the *page*; a general chatbot returns a
fluent article number that does not exist.

Belge-Gözü is built for the opposite failure mode: **it would rather say "I could not find
it" than invent an article.** Answers are produced only from retrieved pages, each claim
carries a page citation, and the page image is shown so a human can check it.

The project is also an argument about method. Nothing is claimed without a number, every
number carries its provenance (which index, which corpus checksum, which commit), and
rejected experiments stay in the repository with their measurements intact.

## 2. Problem

The first working version retrieved page *images* with a ColPali-class vision-language
model — late-interaction MaxSim over page screenshots, no OCR, no layout parsing. It
looked right and measured wrong.

```mermaid
flowchart TB
    Q["Query: annual paid leave<br/>under the Labour Act?"] --> V["Visual retrieval<br/>4,222 pages"]
    V --> D["Right document<br/>cover page, rank 1"]
    V --> P["Right page<br/>art. 53 table, rank 137"]
    P --> A["'I could not find it<br/>in the given pages'"]
    classDef ok fill:#e8f3ec,stroke:#2e7d4f,color:#14321f
    classDef bad fill:#fceceb,stroke:#a61c2c,color:#3d1015
    classDef neutral fill:#eef2f7,stroke:#2c5b8a,color:#12283d
    class D ok
    class P,A bad
    class Q,V neutral
```

The model matched the law's *identity* — its title page — and lost the article. The honest
"could not find it" was correct behaviour on wrong evidence. Measured, the failure was
unambiguous: **Recall@5 = 0.116**, and the domicile-definition page a user would expect
first sat at rank 3,127 of 4,222.

Root cause, once measured rather than guessed: the encoder is trained predominantly on
English, and a long Turkish legal query aligns with a document's *name* far more strongly
than with an article's body text.

## 3. Engineering map

Each step is one controlled experiment: one variable changed, one primary metric
(Recall@5 over a 43-question benchmark), a frozen harness, and a keep-or-revert decision.
**Dashed branches were measured and rejected** — the part most portfolios delete.

```mermaid
flowchart TB
    B["visual only<br/>0.233"] --> T["+ BM25 text<br/>0.674"]
    B -.->|rejected| R1["equal RRF<br/>0.395"]
    T --> F["+ prefix<br/>0.767"]
    T -.->|rejected| BG["bigrams<br/>0.628"]
    F --> S["+ stopwords<br/>0.767"]
    S --> W["+ routing w20<br/>0.814"]
    S -.->|rejected| AP["hard partition<br/>0.791"]
    W --> W5["window 50<br/>0.837"]
    W5 -.->|rejected| WR["window RRF<br/>0.535"]
    W5 --> FD["+ folding<br/>0.8605"]
    FD -.->|rejected| DF["dual-form<br/>0.837"]
    classDef kept fill:#e8f3ec,stroke:#2e7d4f,color:#14321f
    classDef gone fill:#faf3e4,stroke:#a8741a,color:#3a2a08
    class B,T,F,S,W,W5,FD kept
    class R1,BG,AP,WR,DF gone
```

If the diagram does not render (the GitHub mobile app draws no Mermaid), the same chain as a table:

| Step | R@5 | Decision |
|---|---|---|
| visual channel only | 0.233 | baseline |
| + BM25 text channel | **0.674** | kept |
| equal-weight RRF fusion | 0.395 | rejected |
| + 5-character prefix | **0.767** | kept |
| + bigram shingles | 0.628 | rejected |
| + function-word list | 0.767 | kept (deep ranks fixed) |
| + law-name routing (window 20) | **0.814** | kept |
| absolute document partition | 0.791 | rejected (guardrail veto) |
| window 50 | **0.837** | kept |
| within-window RRF | 0.535 | rejected |
| + diacritic folding | **0.8605** | kept — shipped |
| dual-form tokens | 0.837 | rejected (two guardrails down) |

Three results worth stating plainly, because each contradicts the obvious approach:

**Reciprocal Rank Fusion made things worse — three times.** The textbook move is to fuse
the visual and text rankings. Equal-weight RRF dropped Recall@5 from 0.674 to 0.395;
absolute document partitioning failed a guardrail; within-window RRF gave 0.535. The weak
channel's title-page attraction outranks the strong channel's real hits at every
granularity tried. What survived is lexical-primary ranking with a rule-based re-order.

**The visual channel contributes zero unique top-5 hits** once the text channel is tuned.
It still runs on every query — it feeds telemetry and the calibration dataset, and it is
an independent measurement signal for the 16 pages with a weak text layer — but it no
longer ranks. That is the honest result, not the one that fits the original pitch.
**Both channels run in production; BM25 alone determines the ranking.**

**Diacritics were a production bug, not a nicety.** Turkish keyboards are routinely
bypassed: users type *"yillik ucretli izin"*. Measured, that collapsed Recall@5 from 0.837
to 0.581. Folding diacritics on both the index and the query side makes the system
**writing-invariant** — 0.8605 in both conditions.

## 4. Architecture

```mermaid
flowchart TB
    PDF["56 PDFs"] --> IMG["page images"]
    PDF --> TXT["text layer"]
    IMG --> EMB["ColSmol-500M<br/>embeddings"]
    EMB --> Q8["int8 index<br/>476 MB"]
    TXT --> BM["BM25 index<br/>fold + prefix"]
    QRY["Turkish question"] --> TOK["tokenise"]
    TOK --> SC["BM25 scoring<br/>2-5 ms"]
    SC --> ROUTE["law-name routing<br/>top 50 reorder"]
    ROUTE --> GATE{"above<br/>threshold?"}
    GATE -->|no| ABS["abstain"]
    GATE -->|yes| LLM["VLM answerer<br/>pages + markers"]
    LLM --> CITE["answer<br/>+ citations"]
    BM -.-> SC
    Q8 -.-> VIS["visual MaxSim<br/>telemetry only"]
    TOK -.-> VIS
    classDef store fill:#eef2f7,stroke:#2c5b8a,color:#12283d
    classDef act fill:#f7f4ec,stroke:#8a6d2c,color:#2f2510
    classDef out fill:#e8f3ec,stroke:#2e7d4f,color:#14321f
    class Q8,BM store
    class TOK,SC,ROUTE,VIS,LLM,EMB,IMG,TXT act
    class CITE,ABS out
```

The top three boxes are the offline build (run once); the chain below runs per request.

**Interactive figures.** The diagram above is a summary. The explorable versions live in the
[diagram gallery](https://barandincoguz.github.io/belge-gozu/):
[observability architecture](https://barandincoguz.github.io/belge-gozu/observability.architecture.html) ·
[retrieval journey](https://barandincoguz.github.io/belge-gozu/retrieval.sequence.html) ·
[indexing pipeline](https://barandincoguz.github.io/belge-gozu/indexing.dataflow.html) ·
[P1/G1 gate loop](https://barandincoguz.github.io/belge-gozu/p1-gate.lifecycle.html) ·
[visual-only → hybrid delta](https://barandincoguz.github.io/belge-gozu/retrieval-delta.html).
Each figure's typed JSON source is tracked under `docs/diagrams/`; the diagrams are compiled
from that source deterministically rather than drawn by hand.

Two rules hold the system together:

**Identity travels with data.** Every index carries a manifest — model revision, query
format, document-prompt hash, quantisation, corpus checksum. The server refuses to start
against an index whose identity does not match its configuration, and the calibration
artefact is keyed by `index_revision × pipeline × recipe_fingerprint`. This discipline came
out of an audit that found 139 places where a value had drifted from the context that gave
it meaning — the worst being a score threshold silently bound to one quantisation scheme.

**Flags and rollback.** New decision layers (calibrated gate, evidence verifier) ship behind
flags that default to off, with a test asserting the served behaviour is byte-identical
while they are off.

## 5. Technical detail

### Retrieval

| Configuration | Recall@5 | Recall@20 | MRR | Gold-page rank, query A / B |
|---|---|---|---|---|
| visual only, 1-bit (original) | 0.116 | — | — | 3127 / — |
| visual only, int8 | 0.233 | 0.302 | 0.149 | 664 / 137 |
| hybrid, before folding | 0.837 | 0.930 | 0.655 | 2 / 2 |
| **hybrid + folding (shipped)** | **0.8605** | **0.930** | 0.632 | **2 / 2** |

Robustness sweep: BM25 `k1` ∈ [0.9, 1.8] × `b` ∈ [0.5, 0.9] all land in 0.814–0.837, and
prefix length 4–7 is a plateau — the recipe is not balanced on a knife edge. One tuning
setting would have added a further question and was deliberately **not** taken: that is
fitting the benchmark, not the problem.

### Quantisation

int8 matches float16 ranking quality at every k, runs 4.3× faster than 1-bit (0.24 s vs
1.08 s per query on CPU), and costs 476 MB against 58 MB. 1-bit loses 7 points of Recall@20
*and* is slower — bit-packing tricks lose to BLAS here. int8 ships; the others stay as
reproducible ablations.

### Answer path

Page markers are interleaved with images (`[S1]`, image, `[S2]`, image, …) so a citation
binds to a specific page rather than a positional guess. There is no auto-citation
fallback: if the model emits no marker, the answer carries none. Failures are classified
(`timeout`, `http_5xx`, `http_429`, `auth`, `safety_block`, `parse`), and a total time
budget is enforced as an invariant — a retry may not start if the remaining budget cannot
cover it. Two API keys rotate: any transport-level error moves the request to the other key,
and the working key becomes sticky.

### Selective answering (in progress)

The abstain threshold is a **mechanical transfer** of a prior operating point onto the BM25
scale, not a calibration — and it does not separate answerable from unanswerable questions.
Measured, moving the number does not fix that:

- A confidence model over five retrieval-side features reaches AUROC 0.782 on the
  development split, but at a 5% risk budget it answers only 2.2% of questions.
- Signals that looked strong against 5 unanswerable questions (AUC 0.94) fell to 0.68
  against 151 realistic ones — the new negatives are lexically plausible.

Stated as a finding rather than a plan: **retrieval-side confidence alone cannot carry
selective answering here.** A claim-level evidence verifier — segment the answer, check each
claim against its cited page text, demote the answer if any claim is unsupported — is built
and tested behind a flag. `bench answers` evaluates answerable and unanswerable dev
questions in one provenance-rich report, with claim-level citation precision/completeness,
false-supported-answer rate, and one-sided 95% Clopper–Pearson bounds. The test split is
locked behind `--yes-final-gate`; library defaults remain off.

### Benchmarks and their provenance

- **Canary**: 48 questions (43 answerable), behind every retrieval decision above.
- **Unanswerable set**: 330 questions in three classes — out-of-corpus, nonsense, and the
  hard one: *about* a corpus law, but the specific detail genuinely is not in the text.
- Labels come from a drafter ≠ checker regime. Mechanical labels ("the anchored law is
  absent from the 56-document manifest") are re-verified by a script that runs in CI. A
  sampled cross-check put residual label noise at 12.5%, after which the entire test side
  was verified row by row with an evidence quote for every rejection.
- The split is law-grouped: 22 of 56 documents are test-only. The test side holds 155
  unanswerable questions — the size at which a zero-error result supports a ≤2% claim at
  95% confidence.

**Honesty note, repeated wherever these numbers appear:** 3 of the 48 canary rows were
verified by a human; the other 45 and the whole unanswerable set were verified by model
cross-check. **These are not human-validated benchmarks.**

### Operations

A SQLite event log (29 fields per request — pipeline, score scale, which API key served,
whether the model reported an honest miss), a Prometheus endpoint and a provisioned Grafana
dashboard. Input validation rejects empty, overlong and malformed queries; a per-IP rate
limiter with eviction and a privacy default that keeps raw query text off disk are both
enabled in the container image.
`/ask` and `/search` expose `no_match` from the same server-side threshold decision, so
the UI does not present below-threshold diagnostics as valid page cards. `/healthz` owns
the active ranking-channel, score-label, and threshold presentation contract.

CI runs lint, type-check, 707 tests and the benchmark-integrity validator. A separate
Docker job builds the image and checks UID 1000, writable `/data`, CPU-only PyTorch,
missing-revision fail-fast, and a `/healthz` smoke. Its first two runs were red — catching two portability bugs
that 147 local commits had not: CLI assertions that depended on terminal colour, and a
corpus manifest the validator needs that was never tracked.

## 6. Closing

**What works today.** Retrieval is solved to the point where the remaining errors are
semantic rather than lexical: the six unsolved benchmark questions are pure paraphrases
sharing no vocabulary with their target pages. The system answers with citations, abstains
on nonsense, tolerates missing diacritics, and survives an exhausted API quota by rotating
keys.

**What is honestly unfinished.**

| Area | Status |
|---|---|
| Selective answering | Confidence model built and measured; too conservative to enable. Verifier built, behind a flag. |
| Formal gate reports | Phase 0 passed. Phase 1 is formally adjudicated **FAIL** on Recall@50 0.930 and paraphrase 0.571; reranker and live-deployment budgets remain unmeasured. |
| Human validation | 3 of 48 rows. A human-gated benchmark is the honest next step. |
| Public deployment | A pinned self-contained Hub index and non-root CPU Docker/CI contract are ready; hosting needs a paid tier and the app has not been deployed live. |
| Article structure & OCR | Article-level hierarchy was specified but never built; 16 pages have a weak text layer and no OCR fallback. |

All of it is tracked as issues in this repository — the failures included, filed rather
than footnoted.

**What this project argues.** The interesting part was not the model. It was building a
measurement apparatus honest enough to overturn its own design: a visual-retrieval project
whose measurements said the visual channel should stop ranking, a fusion strategy rejected
three times on evidence, and a confidence model whose own numbers said it was not ready to
ship. The runs behind those calls are in `docs/research/findings/`, dated, with the raw
artefacts beside them.

### Run it

```bash
uv sync --all-extras
BG_HF_DATASET_REPO=barandincoguz/belge-gozu-index \
BG_HF_REVISION=700ac324fffefb22de02c8e90347b31185547948 \
  uv run belge-gozu index pull
BG_DEVICE=cpu uv run belge-gozu serve --port 7860  # http://localhost:7860
```

Requires `GOOGLE_API_KEY` (optionally `GOOGLE_API_KEY_2` for rotation) in `.env`.
Tests: `make test` · lint and types: `make lint` · dashboards: `make obs-up`

Answer-gate evaluation requires an explicit real-attempt budget:

```bash
uv run belge-gozu bench answers --split dev --max-llm-attempts 40
```

The container uses the same pinned artifact and runs as UID/GID 1000:

```bash
docker build -t belge-gozu:p0 .
docker run --rm -p 7860:7860 -v belge-gozu-data:/data \
  -e GOOGLE_API_KEY="$GOOGLE_API_KEY" belge-gozu:p0
```

### Repository map

| Path | What is in it |
|---|---|
| `src/belge_gozu/retrieval/` | BM25 text channel, law-name routing, hybrid retriever |
| `src/belge_gozu/index/` | encoding, quantisation, manifests, compatibility fail-fast |
| `src/belge_gozu/answer/` | answerer, key rotation, calibration, claim verifier |
| `src/belge_gozu/telemetry/` | event log, Prometheus metrics, stage timing |
| `docs/research/findings/` | dated measurement notes — the reasoning behind every number here |
| `docs/diagrams/` | typed JSON sources and compiled HTML for the interactive figures |
| `research/` | the experiment loop: journal, harness, results |
| `data/bench/` | benchmarks, splits, and their provenance READMEs |
