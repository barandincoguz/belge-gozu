# P2 autoresearch programı — semantik metin kanalı (D2)

> Skill: `.claude/skills/autoresearch/SKILL.md`. Bu dosya P1'in `program.md`'sinin
> muadilidir ve onu **değiştirmez**: P1 döngüsü (`research/retrieve.py`,
> `research/evaluate.py`) donuk kalır, tarihsel referanstır.

## Hedef

`paraphrase` dilimini kapatmak. Bugün ölçülen durum, ve neden başka hiçbir şeyin
öncelikli olmadığı:

- Hiçbir kanalın top-50'de bulamadığı **8 sorunun 8'i de `paraphrase`** (%100).
- BM25 yedi dilimin altısında R@20 = **1,0000** — keyword kanalı fiilen çözülmüş.
- Görsel kanalın 57 soruda benzersiz katkısı **1 soru** — getirim kaldıracı değil.

Yani sistemde eksik olan tek şey **semantik metin eşleştirmesi**, ve D2 tam olarak
odur.

Sayısal hedef: **`paraphrase` R@50 ≥ 0,90** (G1.2 eşiği; taban 0,5714). Ara hedef
0,75 — oraya ulaşan bir deney tutulur, ama kapı 0,90'da kapanır.

## Birincil metrik (TEK)

**`paraphrase` dilimi R@50**, `data/bench/retrieval_eval_v2.jsonl` üzerinde,
`--min-verification human` altkümesinde (n=21, tamamı insan-doğrulanmış).

Dilim seçildi, genel sayı değil: genel R@5 bir **dilim karışımı** özelliğidir ve
setler arası karşılaştırmaya uygun değildir (bkz. `data/bench/retrieval_eval_v2.README.md`
§4). Hedef dilimin kendi içindeki sayı karşılaştırılabilir olandır.

Guardrail'ler (karar metriği DEĞİL; kesin gerileme bir deneyi veto edebilir):

- `dogrudan-madde`, `madde-numarali`, `ayni-kanun-hard-negative` R@50 = 1,0000 —
  üçü de insan-doğrulanmış ve **hiçbiri gerilemeyecek**.
- **Eşit karakter bütçesinde kapsama** (4k / 10k / 25k / 60k). Bu ölçüm P2'de
  birincil guardrail'dir çünkü reranker ve LLM token-sınırlıdır; sabit-k
  karşılaştırması ince chunk'ın aleyhine yanlıdır (ölçüldü: chunk sabit k'da
  kaybederken 4k bütçede +0,1721 kazanıyor).
- Sorgu gecikmesi. Metin kanalı bugün 2–5 ms; semantik kanal bu bandı büyütecek
  ve G1.5'in ölçülmemiş gecikme kalemini ölçmeye zorlayacak — bu bir yan fayda,
  ama bütçe raporlanmadan deney kapanmaz.

## Taban (2026-09-02, chunk'lı korpus, insan-doğrulanmış set)

| | değer |
|---|---|
| `paraphrase` R@50 | **0,5714** (n=21) |
| `paraphrase` R@5 | 0,2381 |
| genel R@5 (yalnız insan) | 0,6277 (n=47) |
| eşit bütçe kapsama 4k / 10k / 25k | 0,5410 / 0,5902 / 0,6885 |

Korpus: 10.531 chunk (9.532 madde + 999 sayfa), medyan 539 karakter, %87,6'sında
kenar başlığı. Chunk üretimi `src/belge_gozu/corpus/chunking.py` (23 test).

Kanıt: `data/bench/results/d1-final-human.json`, `chunking-arms.json`.

## Mimari kararlar (D2 öncesi verildi, deney nesnesi DEĞİL)

1. **Birleşim aday düzeyinde, skor düzeyinde DEĞİL.** BM25 top-N ∪ semantik top-N
   birleşimi reranker'a gider. G1.6'da üç **skor** füzyonu ölçülüp reddedildi
   (küresel RRF 0,674→0,395; doküman bölümleme 0,907→0,837; pencere-içi RRF
   0,837→0,535); o olumsuz sonuç aday-düzeyi birleşime uygulanmaz, çünkü skorlar
   hiç karışmaz.
2. **Getirim için chunk, kanıt için sayfa.** Chunk iç temsildir; `page_ids` taşır,
   skorlama ve VLM cevaplayıcı sayfaya indirger. Bench'in altın verisi sayfa bazlı
   kalır.
3. **Görsel kanal bayrak arkasında, varsayılan kapalı.** Projenin kendi G1.6
   ilkesi ("kazanç göstermeyen katman varsayılan kapalı") kendi ayırt edici
   özelliğine uygulandı: 57 soruda 1 benzersiz katkı, 476 MB bellek.
4. **BM25 reçetesi DONUK.** `k1=1.5, b=0.75` + ASCII katlama + stopword + F5
   kırpma + kanun-adı yönlendirmesi. Chunk'lı korpusta yeniden tarandı; bütçe
   metriğinde en iyi konfigürasyonla fark 0,0054 (bir sorunun üçte biri) —
   değiştirmek gürültüye uydurmak olurdu.

## Kapsam — tek değiştirilebilir yüzey

`research/retrieve_v2.py` — imza:
`rank_chunks(q: QueryContext) -> list[str]` (sıralı `chunk_id` listesi).

DONUK: `research/program.md`, `research/evaluate.py`, `research/retrieve.py`,
`data/bench/**`, `src/belge_gozu/corpus/chunking.py`, `src/belge_gozu/retrieval/text.py`.

P1'den farklı olarak **model yükleme SERBESTTİR** — D2'nin bütün amacı bu.

## Yasaklar

1. Bench'e dokunulmaz. Ölçüm aracı deney nesnesi olamaz. (D1'de örtüşme kapısının
   eşiği veriye uydurulmadı; aynı disiplin.)
2. BM25 reçetesi değiştirilmez. Semantik kanal onun **yanına** eklenir, yerine değil.
3. Skor düzeyinde füzyon yasak (yukarıda karar 1).
4. Guardrail dilimlerinde gerileme veto sebebidir — birincil metrik yükselse bile.
5. **(2026-09-02'de DEĞİŞTİRİLDİ — mimari karar, gerekçe aşağıda.)** Orijinal
   sorgu her zaman kendi kanalında KORUNUR. Yeniden yazılmış bir sorgu yalnız
   EK bir kanal olarak eklenebilir; orijinali hiçbir yerde İKAME EDEMEZ.

   Kuralın ilk hâli P1'den miras alınmıştı ("sorgu yeniden yazımı yok") ve
   oradaki gerekçe geçerliydi: tek kanallı bir çözümde sorguyu yeniden yazmak
   ölçtüğün şeyi değiştirir. Ama P2'nin mimarisi çok kanallı; orijinali koruyup
   yanına bir kanal eklemek o gerekçeyi ihlal etmez.

   **Sıra dürüstçe kayda geçsin:** bu kuralı, sondanın sonucunu BİLEREK
   değiştirdim. Elle kanun diline çevrilen üç sorgu rank 1 bulmuştu (300 /
   bulunamadı / 88 iken). Yani bu, "deneyi kurtarmak için kapı taşını oynatmak"
   görünümü taşıyor ve öyle değerlendirilmeli. Savunmam: sonda bir MEKANİZMA
   kanıtıdır (boşluk sözlükseldir, model/chunking değil), ve sorgu anlama bu
   ürünün gerçek bir bileşenidir — kullanıcı günlük Türkçe yazar, mevzuat kanun
   dilinde yazılmıştır. Kuralı değiştirmemek, ölçümün gösterdiği tek çözümü
   yasaklamak olurdu. Karar okuyucuya açık bırakılıyor.
6. Bir seferde bir değişken. Model seçimi, chunk metni biçimi (başlık öneki var/yok)
   ve birleşim derinliği ayrı deneylerdir.

## Bağımlılık ve depolama kararı

Aday modeller (hepsi HF'de doğrulandı):

| model | param | boyut |
|---|---|---|
| `newmindai/ColmmBERT-small-TR` | 140M | 597 MB |
| `moganai/Mogan-ColBERT-TR` | 148,9M | 599 MB |
| `newmindai/ColmmBERT-base-TR` | 310M | 1.263 MB |
| `ytu-ce-cosmos/turkish-colbert` | 100M | 444 MB |

Ortam: torch 2.13.0, transformers 5.15.1, MPS mevcut, 14 CPU / 24 GB.
`pylate`, `sentence-transformers`, `qdrant-client` **kurulu değil**.

**Qdrant kararı: REDDEDİLDİ — ve bu bir ölçüm sonucu, tercih değil.**

Bu bölümün ilk hâli Qdrant'ı seçiyordu ("vector store'u elle yazmıyoruz").
Araştırma turu o kararı çürüttü; gerekçe aynen kayda geçiyor çünkü karar
ölçümle değişti:

| | Qdrant gömülü | mevcut FloatIndex (fp16) |
|---|---|---|
| sorgu gecikmesi | 323 ms | **29 ms** (11× hızlı) |
| RSS | 1.034 MB | **352 MB** |
| yeni bağımlılık | 7 paket / ~74 MB (grpcio, protobuf 7.x) | yok |

Üç bağımsız ret sebebi:

1. **Her eksende kaybediyor.** Gecikme, bellek ve bağımlılık ağırlığı — üçü birden.
2. **Gömülü mod zaten kaba kuvvet.** Qdrant local mode saf Python + numpy; HNSW
   yok. Yani kendi yazacağımız algoritmanın daha yavaş bir uygulamasına bağımlılık
   ödemiş olurduk. Üstelik dokümantasyonu onu "geliştirme/test, ~20.000 noktaya
   kadar" diye tanımlıyor; 10.531 chunk o tavanın %53'ü.
3. **Yaklaşık indeks D2'nin amacına ters.** PLAID ve MUVERA kayıplıdır; MUVERA'nın
   0,54 ms'i gerçek ama D2 bir **recall** açığını kapatmak için var. Kesin MaxSim,
   verilen model için recall TAVANIDIR — her yaklaşık indeks oradan aşağı iner.
   İhtiyacımız olmayan 28 ms için recall takas etmek tam tersi yöndür.

Genel ilke değişmedi ("vector store elle yazılmaz"); değişen şey, **bu ölçekte
yazılacak bir vector store olmadığı**. Depo katmanı `embs.npy` + `offsets.npy`;
`FloatIndex.load` onu zaten mmap'liyor. Tam yeniden inşa ~2 dakika, yani indeks
bir veritabanı değil **önbellek**.

**fp16, int8 DEĞİL.** Sayaltı sonuç: bu iş yükünde fp16 int8'den ~3× HIZLI
(28 ms vs 87 ms), üstelik nicemleme hatası sıfır. int8 görsel kanalda doğru
karardı çünkü orada 1,9 GB fp32 bağlayıcı kısıttı; metin kanalı ölçeğinde o kısıt
yok, yani int8'i miras almak 3× gecikme + recall kaybı demek olurdu. 1-bit de
yasak (repo zaten ölçmüş: retrieval_eval top-1 int8 0,6250 vs 1-bit 0,4953).

**pylate: HAYIR.** Kurulumu torch 2.13.0 → 2.11.0 ve transformers 5.15.1 → 5.3.0
**düşürüyor** (pylate 1.6.0 `transformers<=5.3.0` ve `sentence-transformers==5.3.0`
pinliyor; uv 4 paket kaldırıp 30 paket kuruyor). Ölçülmüş ve dondurulmuş bir
yığında bu kabul edilemez risk. Düz `transformers` ile ~60 satırlık bir yeniden
yazım `pylate.models.ColBERT.encode()`'u bit düzeyinde üretiyor (max_abs_diff
4,7e-07, kosinüs 1,000000, MaxSim skorları birebir). Yeni bağımlılık: **sıfır**.

**Model: `moganai/Mogan-ColBERT-TR` (A kolu).** ColmmBERT TurkColBERT'te önde ama
belge penceresi 180 token'da sabit ve bizim korpusumuzda **chunk'ların %50,3'ünü
kesiyor, token'ların %61,1'ini atıyor** — Türk mevzuatında işletici koşul sıklıkla
uzun bir maddenin SONUNDA durur, yani tam olarak silinen yarıda. Mogan'ın Türkçe
tokenizer'ı %33 daha verimli ve native penceresi 512; kesme oranı %12,7.
B kolu: `ytu-ce-cosmos/turkish-colbert` (dim 256, legacy colbert-ai formatı).
C kolu / yedek: `lightonai/mLateOn`.

**Kodlama sözleşmesi — tahmin edilmeyecek, config'den okunacak.** Beş kural:
(1) `max_length - 1`'e kadar tokenize et, işaret bir slot yiyor; (2) işaret
token'ı metin olarak DEĞİL, id olarak index 1'e (CLS'ten sonra) yerleştirilir;
(3) sorgu tarafı `[MASK]` ile `query_length`e doldurulur ve 32 vektörün TAMAMI
korunur; (4) belge tarafı doldurulmaz, noktalama skiplist ile atılır;
(5) `Linear(128, bias=False)` → **sonra** L2 normalize.

Bir tuzak kayda geçsin: `pylate.ColBERT.encode()`'un `is_query` varsayılanı
**True**. Belgeleri bayrağı vermeden kodlamak her chunk'a [Q] işareti verir,
32 token'a keser ve [MASK] ile doldurur — hata yok, uyarı yok. Bu, görsel
kodlayıcıda R@5 0,233 vs 0,093'e mal olan hata sınıfının aynısı. Bu yüzden
çağrı yüzeyi `encode_documents()` / `encode_queries()` olacak ve indeksleyici ham
`encode(..., is_query=...)` ÇAĞIRAMAYACAK; ayrıca inşa zamanında "chunk başına
ortalama vektör > 40" doğrulaması konacak (sorgu olarak kodlanmış bir belge tam
32 verir).

## Karar kuralı

`paraphrase` R@50 kesin artarsa ve hiçbir guardrail gerilemezse → commit
(`exp(<ad>): paraphrase R@50 <eski>-><yeni> KEPT`). Eşitlik, gerileme ya da
guardrail ihlali → `git checkout -- research/retrieve_v2.py`.

Durma: hedefe ulaşıldı; art arda 5 verimsiz deney; ya da bütçe bitti.
