# Anlamsal kapsama ölçüldü: dense kanal 1 soru, genişletme 1 soru daha getirdi

Tarih: 2026-09-10
Ölçüm: `data/bench/results/20260910-1837-semantic-coverage.json` (+ `.html` rapor)
Karar: **Dense kanal ÜRETİME GİRMEZ.** Ölçüm bir şeyi çürüttü ("daha büyük
gömme modeli gerek"), bir şeyi kanıtladı (kalan ıskalar sözlüksel) ve 12 gün
kotaya takılı kalan exp5 ölçümünü kapattı.

## Kurulum

Şartname `docs/superpowers/plans/2026-09-03-semantic-coverage-experiments.md`:
kollar yalnız ilk-görülme tekilleştirmesiyle birleşir (füzyon YOK), dense tam
ilk 50 sayfayı ekler, taban BM25-50 + Mogan-50 + Colmm-50. Küme
`retrieval_eval_v2`, insan-doğrulanmış, cevaplanabilir **n=47**
(`--min-verification human`). Karar metriği `coverage`: birleşim havuzunda gold
sayfadan en az birinin bulunma oranı. Havuz kırpılmadığı için (medyan 112 →
dense ile 138 sayfa) bu metrik kanal eklemeyle **düşemez**; soru "artar mı"
değil, "ne kadar ve hangi bedelle".

Ortam: Mac Studio M3 Ultra, 96 GB. Metal bütçesi **77,8 GiB** — 24 GiB'lık
makinede 17,8 GiB olan kapı burada iki modeli de geçiriyor. Yeniden üretilen
önbellekler eski kayıtla birebir aynı: chunks 10.531; Mogan 1.918.277 vektör /
491 MB; Colmm 2.530.392 vektör / 648 MB. Dense kodlama tek seferde bitti
(4B 64 dk, 8B 95 dk, kesinti/OOM yok).

## Sayılar (n=47)

| kol | coverage | paraphrase (n=21) | R@5 | R@20 | R@50 | MRR |
|---|---|---|---|---|---|---|
| taban BM25+Mogan+Colmm | 0,9574 (45/47) | 0,9048 (19/21) | 0,6277 | 0,7660 | 0,8085 | 0,4242 |
| + dense 4B | **0,9787** (46/47) | **0,9524** | 0,6277 | 0,7660 | 0,8085 | 0,4244 |
| + dense 8B | 0,9787 (46/47) | 0,9524 | 0,6277 | 0,7660 | 0,8085 | 0,4244 |
| + dense 4B + genişletme | **1,0000** (47/47) | **1,0000** | 0,6277 | 0,7660 | 0,8085 | 0,4245 |

Diğer bütün dilimler (`dogrudan-madde` 13, `madde-numarali` 6,
`ayni-kanun-hard-negative` 5, `tarihi-tarama` 2) tabanda **zaten 1,000**.
Kazanım yalnız `paraphrase` diliminde.

**Ortam sağlaması (asıl kanıt).** Taban satırı, 2026-09-03 reranker teşhisinin
aynı 47 soruda ölçtüğü havuzla **birebir** aynı: kapsama 0,9574, paraphrase
0,9048, havuzun ilk-50 sırası R@5 0,6277 · R@20 0,7660 · R@50 0,8085 · MRR
0,4242 (`docs/research/findings/2026-09-03-candidate-pool-reranker-experiment.md`).
Yeni makinede sıfırdan üretilen chunk/sidecar/indeks zinciri eski ölçümü dört
haneye kadar tekrar üretiyor.

Atıf, soru düzeyinde:

- Tabanın ıskaladığı iki soru: **c206** (KVKK saklama süresi) ve **c404**
  (ayıplı hizmet) — ikisi de `paraphrase`/`dogal`.
- **c206'yı yalnız dense getirdi**: `k6698:3`, bütün koşumda `gold_sources`
  değeri tek başına `["dense"]` olan tek sayfa.
- **c404'ü genişletme getirdi** (Qwen3-8B varyantı, kanalların tümü varyantla
  yeniden sorgulandı).

## Üç sonuç

**1. 8B, 4B'yi hiçbir yerde geçmiyor.** Aynı coverage, aynı R@k, MRR dördüncü
haneye kadar aynı. Bedeli: 66 MB'a karşı 41 MB artefakt, sorgu başına 188 ms'e
karşı 127 ms (p50, model yerleşik), 95 dk'ya karşı 64 dk kodlama, 16 GB'a karşı
8 GB yerleşik ağırlık. Önceden ilan edilmiş `select_dense_arm` eşitlik bozucusu
da 4B'yi seçti. Dense bir gün üretime girerse **4B girer**; 8B artefaktının
ölçülmüş bir değeri yok. Kapsam notu: 2 soruluk pay varken metrik model
kalitesini ayırt edemez — bu "8B kötü" değil, "bu küme bu metrikte 8B'yi
haklı çıkaramaz" demektir.

**2. Kazanım havuzun DERİNİNDE, ürünün gördüğü yerde değil.** R@5/R@20/R@50 üç
kolda da rakamı rakamına aynı; MRR dördüncü hanede oynuyor. Sebep tasarımsal:
şartname füzyonu yasaklıyor, dense adayları BM25 ve geç kanallardan SONRA
ekleniyor, yani ilk sıralar hiç değişmiyor. Ürünün asıl arızası (gold top-5'e
girmiyor) bu ölçümle **hiç dokunulmadan** kalıyor. Kapsama artışını sıralamaya
çevirecek şey füzyon + yeniden sıralamadır; bu döngünün dışı.

**3. İstatistik: yön doğru, büyüklük ölçülemez.** n=47'de dense +1 soru,
genişletme +1 soru. Eşleşmiş McNemar (b=1, c=0) p≈1,0; 0,9574 ile 0,9787'nin
Wilson %95 aralıkları neredeyse tümüyle örtüşüyor. Bir soruluk fark bu örneklem
büyüklüğünde gürültüden ayrılamaz. Ayrıca aynı 47 soru üzerinde yinelenen
iterasyon var (program.md'nin kendi aşırı-uyum uyarısı burada da geçerli).

## exp5'in kapanışı (asıl kazanç)

`research/journal-p2.md` exp5 — "LLM sorgu yeniden yazımı" — **ÖLÇÜLEMEDİ**
olarak kapanmıştı: 47 sorgunun 45'i Gemini'den 429 `RESOURCE_EXHAUSTED` aldı,
yarım artefakt yanıltıcı olmasın diye silindi ve kayda "eksik olan tek şey
kota" yazıldı. Sonda mekanizmayı kanıtlamıştı: üç ıska elle kanun diline
çevrilince sıra 300/bulunamadı/88 → 1/1/1.

Bu koşum o boşluğu **yerel** modelle kapattı: kota yok, ağ yok, 47 sorunun
41'i gerçekten genişletildi ve kapsama 1,0000'e çıktı. Yani kalan `paraphrase`
ıskalarının kök nedeni gerçekten **kayıt uyuşmazlığı** (günlük dil ↔ kanun
dili); indeks, chunking, gold ve model masum. "Daha güçlü gömme modeli" hipotezi
ise aynı ölçümde çürüdü (madde 1).

Genişletmenin bedeli dürüstçe: sorgu başına 15 GB'lık checkpoint'ten bir
`generate` **artı** bütün kanalların varyantla ikinci kez sorgulanması, ve
47 varyantın **6'sı bozuk** çıktı (özgün sorgunun aynısı; c107, c203, c205,
c208, c313, c407). O altı soru özgün sorgusuyla ölçüldü ve kimlikleri raporda
`invalid_expansions` altında duruyor — sessiz düzeltme yok.

## Harness düzeltmesi (ölçümden önce zorunlu oldu)

İlk koşum 18 dakika sonra düştü ve **hiçbir çıktı yazmadı**: genişletme kolu
24 GiB'lık makinede hep `skipped_oom` olduğu için bu kod yolu ilk kez burada
gerçekten koştu, bir soruda model özgün sorgunun aynısını üretti ve
`validate_expansion`ın haklı `ValueError`ı bütün ölçümü (dense kolları dahil)
çöpe attı. İki dayanıklılık düzeltmesi (TDD, 2 yeni test):

1. Bozuk varyant artık kolu düşürmez; o soru özgün sorgusuyla ölçülür ve
   `question_id` raporda kalır. Sözleşme sıkı kaldı: geçersiz varyant
   KULLANILMAZ.
2. Üretilen her varyant anında diske yazılır. Toplu yazım, 40'ıncı soruda düşen
   bir koşumda tamamlanmış 39 üretimi de kaybediyordu.

## Uçtan uca deney: dense'li havuz + BGE yeniden sıralama (aynı gün, n=47)

Yukarıdaki bölüm "sıradaki dar deney" diye tarif ediyordu; deney aynı gün
koştu. `scripts/eval_candidate_reranker.py`'a dense opsiyonel dördüncü havuz
kaynağı olarak eklendi (`--dense-model`; `run_comparison` DEĞİŞMEDİ, dense
zaten kanal protokolüne uyuyor). İki kol, tek fark dense:

| kol | havuz kapsaması | P R@5 | P R@20 | P R@50 | P nDCG@5 | rerank p50 |
|---|---|---|---|---|---|---|
| A — dense yok | 0,9574 | 0,7766 | 0,8617 | 0,9468 | 0,5709 | 3.319 ms |
| B — dense 4B | **0,9787** | **0,7766** | **0,8830** | **0,9681** | **0,5709** | 4.038 ms |

**A kolu 2026-09-03 koşumunu beş metrikte de dört haneye kadar tekrar üretti**
(havuz 0,6277/0,7660/0,8085, P 0,7766/0,8617/0,9468, U 0,7553/…/0,6937,
kapsama 0,9574). Farklı Python (3.11 ↔ 3.12), farklı torch (2.11 ↔ 2.13),
farklı numpy (2.4 ↔ 2.5) ve sıfırdan üretilmiş indeks/sidecar zinciriyle: hat
deterministik. Tek fark donanım — yeniden sıralama p50 8.690 → **3.319 ms**
(M3 Ultra, 2,6×).

**Sonuç: dense ilk beşi HİÇ kıpırdatmıyor.** R@5 iki kolda da 0,7766 ve %95
güven aralığı bile aynı: [0,6596, 0,8830]. nDCG@5 dördüncü haneye kadar aynı —
sunulan liste değişmiyor. Dense'in getirdiği tek sayfa (c206 → `k6698:3`;
havuzda A'da YOK, B'de VAR, havuz 127 → 159) yeniden sıralamada yalnız
**6–20 arasına** çıkabildi: R@20 ve R@50 tam +0,0213 (= 1/47) arttı, R@5 sıfır.

**Üstelik bir guardrail geriledi.** Serbest (U) kolunda dense, `c407`'ye yeni
bir aday soktu: BGE `k5411:84`'ü eski top-1 `k5941:8`'in üstüne koydu, ama yeni
top-1'in BM25 skoru **8,9** — 10,6 eşiğinin altında. `would_abstain` sayısı
5 → **6**: sistem daha önce cevapladığı bir soruda çekimserliğe düşüyor.
(BM25 top-1'i sabitleyen P kolu bu arızayı yapısal olarak önlüyor; 2026-09-03
kaydının P'yi güvenli bulması burada da doğrulanıyor.)

Uçtan uca bilanço: **+0 R@5, +1 soru R@20/R@50'de, +%22 yeniden sıralama
gecikmesi (3,32 → 4,04 s), +8 GB yerleşik gömme modeli, +127 ms sorgu
kodlama, +41 MB artefakt, 64 dk üretim, ve U kolunda +1 çekimserlik.**

## Karar ve sıradaki adım

Dense kanal üretime girmez. Bu artık proxy metrik değil, uçtan uca ölçüm:
ilk beşe etkisi tam olarak sıfır (R@5 ve GA'sı iki kolda birebir aynı),
karşılığında +%22 yeniden sıralama gecikmesi, 8 GB yerleşik ağırlık ve U
kolunda bir çekimserlik daha. Getirdiği tek soru (c206) rank 6–20 bandında
kalıyor ve o sorunun kök nedeni için envanterde çıkarım maliyeti sıfır olan bir
aday zaten var (exp9: başlıktan türetilmiş kısaltma alias'ı).

Dense için soru kapandı (yukarıdaki uçtan uca deney). Açık kalan tek umut
verici kol **genişletme**: kapsamayı 1,0000'e çıkaran o, ve rescue ettiği soru
(c404) yeniden sıralamada ilk beşe çıkıyor mu, ÖLÇÜLMEDİ. Deney aynı harness'la
yapılabilir ama bir tasarım kararı gerektirir — reranker özgün soruyu mu yoksa
genişletilmiş sorguyu mu skorlasın? Program-p2 kuralı "yeniden yazım EK
kanaldır, orijinali İKAME EDEMEZ" diyor; sadık kurulum, havuzu iki sorguyla da
besleyip **özgün soruyla** yeniden sıralamaktır ve bu, `run_comparison`'a
sorgu-başına ek aday listesi geçirmeyi gerektirir. Karar kullanıcıya bırakıldı.

Not: vasıta `research/retrieve.py` DEĞİLDİR — o döngünün `QueryContext`i yalnız
`query_text`, `page_ids`, `visual_scores`, `page_texts` taşıyor (dense/ColBERT
adayı yok) ve deney bütçesi "saniyeler, model yükü yok" diyor.
