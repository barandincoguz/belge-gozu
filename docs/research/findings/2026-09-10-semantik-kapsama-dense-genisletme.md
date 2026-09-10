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

## Karar ve sıradaki adım

Dense kanal üretime girmez: ürünün gördüğü ilk beşe etkisi ÖLÇÜLEN sıfır,
tek getirdiği soru (c206) için sorgu başına GPU'da yerleşik bir gömme modeli
gerekiyor — üretim ise CPU int8 + BM25 üzerinde koşuyor — ve o sorunun kök
nedeni için envanterde daha ucuz aday var (exp9: başlıktan türetilmiş kısaltma
alias'ı, çıkarım maliyeti sıfır).

Kapsamayı sıralamaya çeviren mekanizma zaten ÖLÇÜLDÜ ve dense'siz hâliyle
çalışıyor: 2026-09-03 koşumunda BGE reranker havuzun derinini yukarı taşıyıp
R@5'i 0,6277 → **0,7766**'ya çıkardı (P kolu, BM25 top-1 sabit; +7 soru). Yani
"havuzda olmak" ile "ilk beşte görünmek" arasındaki köprü var — ama bedeli
sorgu başına p50 **8.690 ms**'dir ve o koşum da üretim isteğine eklenmedi.

Bu, dense sorusunu tek ve dar bir deneye indiriyor: **dense'li havuz + aynı BGE
P kolu**, n=47, bootstrap GA ile dense'siz kola karşı. Dense'in getirdiği tek
sayfa (`k6698:3`) yeniden sıralamada ilk beşe tırmanıyorsa dense'in değeri
"8 GB yerleşik ağırlık + 127 ms karşılığında bir soru"dur; tırmanmıyorsa
uçtan uca ölçülebilir katkısı **sıfırdır**. Vasıta `research/retrieve.py`
DEĞİLDİR: o döngünün `QueryContext`i yalnız `query_text`, `page_ids`,
`visual_scores` ve `page_texts` taşıyor (dense/ColBERT adayı yok) ve deney
bütçesi "saniyeler, model yükü yok" diyor. Doğru vasıta, üç kanallı havuzu
zaten kuran `scripts/eval_candidate_reranker.py`dir (bench-only, şartnamenin
izin verdiği yüzey).
