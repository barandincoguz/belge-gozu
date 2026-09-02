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

**`paraphrase` dilimi R@50**, `data/bench/canary_v2.jsonl` üzerinde,
`--min-verification human` altkümesinde (n=21, tamamı insan-doğrulanmış).

Dilim seçildi, genel sayı değil: genel R@5 bir **dilim karışımı** özelliğidir ve
setler arası karşılaştırmaya uygun değildir (bkz. `data/bench/canary_v2.README.md`
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
5. Sorgu yeniden yazımı yok; türetilmiş temsiller (tokenizasyon, öneki, [Q]/[D]
   işaretleri) serbest.
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

**Qdrant kararı.** Vector store'u elle yazmıyoruz — BM25'i elle yazmak ölçülmüş
bir Türkçe reçetesi kazandırdığı için savunulabilirdi, çok-vektörlü depolamada
öyle bir kazanç yok, orada tekerleği yeniden icat etmek olurdu. Qdrant'ın native
late-interaction (multi-vector) desteği bu iş için doğru araç.

Tek kısıt bağlayıcı: **tek konteyner.** Ayrı bir Qdrant sunucusu HF Space
topolojisini kırar ve G1.5/G1.7 bellek bütçesi zaten ölçülmemiş. Bu yüzden karar
şu iki koşula bağlıdır ve araştırma turu bunları cevaplıyor:

- Qdrant **gömülü (in-process) modda** çalışabiliyorsa → kullan.
- 10.531 chunk × ~100–300 token vektörü ölçeğinde kaba-kuvvet MaxSim yeterince
  hızlıysa (mevcut görsel kanal 4.222 sayfada 0,24 sn/sorgu yapıyor) → önce onunla
  ölç, Qdrant'ı ölçüm kazancı gösterdiğinde devreye al.

> Bu bölümün kesin hâli araştırma turunun bellek ve gecikme sayılarıyla
> tamamlanacaktır.

## Karar kuralı

`paraphrase` R@50 kesin artarsa ve hiçbir guardrail gerilemezse → commit
(`exp(<ad>): paraphrase R@50 <eski>-><yeni> KEPT`). Eşitlik, gerileme ya da
guardrail ihlali → `git checkout -- research/retrieve_v2.py`.

Durma: hedefe ulaşıldı; art arda 5 verimsiz deney; ya da bütçe bitti.
