# P2 deney günlüğü — semantik metin kanalı

Kurallar `research/program-p2.md`'de. Her deney: hipotez → değişiklik → sayılar
→ karar → öğrenilen. Başarısız deney de kayıttır.

---

## exp1 — Türkçe geç-etkileşim kanalı (Mogan-ColBERT-TR) · **KEPT**

**Hipotez.** BM25 sözlükseldir; `paraphrase` dilimi tam olarak sözlüksel
örtüşmenin olmadığı yerdir. Türkçe'de geç-etkileşim modelleri yoğun modelleri
alan-özgü görevlerde açık ara geçiyor (TurkColBERT, arXiv 2511.16528) ve MaxSim
altyapısı bu repoda zaten var — yanlış modele bağlı.

**Değişiklik.** `moganai/Mogan-ColBERT-TR` @ `ad90b4f`, `document_length=512`,
10.531 chunk üzerinde fp16 indeks. Aday düzeyinde örülmüş birleşim; skor füzyonu
yok. Yeni bağımlılık yok.

**Sayılar** (insan-doğrulanmış set, n=47, 60 ms/sorgu):

| kanal | R@1 | R@5 | R@20 | R@50 |
|---|---|---|---|---|
| BM25 | 0,4681 | 0,5745 | 0,7021 | 0,7872 |
| ColBERT | 0,5532 | 0,6809 | 0,8723 | 0,9149 |
| BİRLEŞİM | 0,4681 | 0,6809 | 0,7872 | **0,9149** |

**Birincil metrik: `paraphrase` R@50 0,5714 → 0,8095 (+0,2381), n=21.**

Guardrail'ler: `ayni-kanun-hard-negative`, `madde-numarali`, `tarihi-tarama`
1,0000'de kaldı; `dogrudan-madde` 0,9231 → 1,0000 **iyileşti**. Gerileme yok.

**Karar: KEPT.** Hedef 0,90'a ulaşılmadı, ara hedef 0,75 aşıldı.

**Öğrenilenler.**

1. **ColBERT tek başına birleşimden daha iyi sıralıyor** (R@20 0,8723 vs
   0,7872). Birleşim BM25'e ilk sırayı bıraktığı için R@1/R@20'de ondan geri
   kalıyor, R@50'de eşitleniyor. Reranker eklendiğinde bu fark kapanmalı —
   ama eğer kapanmazsa "BM25 ilk sırayı korusun" kararı yeniden ölçülmeli.
2. **Ölçüm hattındaki bir hata kazancı tamamen gizlemişti.** İlk birleşim
   uygulaması BM25'in ~400 sayfalık listesini yazıp ColBERT'i arkasına
   ekliyordu; birleşim BM25'e ÖZDEŞ çıkıyor, +0,238 hiç görünmüyordu. Ders:
   birleşim bir DERİNLİK kümesidir, liste birleştirme değil.
3. **Sözleşme modelden okunmalı.** `sentence_bert_config.json` Mogan'da
   `max_seq_length: 31` gönderiyor; miras alınsaydı her belge 31 token'a
   kesilirdi ve kanal sessizce çöp üretirdi.

**Kanıt.** `data/bench/results/d2-colbert-mogan.json`,
`data/index-colbert-mogan-f16/colbert.json`.

**Sıradaki.** exp2: reranker (`BAAI/bge-reranker-v2-m3`) — birleşimi sıralayıp
R@5'i R@50'ye yaklaştırmak, ve G1.3'ü ölçülmüşe çevirmek.

---

## exp2 — ikinci nöral kanal (ColmmBERT-small-TR) · **KEPT**

**Hipotez.** R@50 bir KÜME kapsama metriğidir; ikinci bir nöral kanal onu
matematiksel olarak düşüremez. Farklı tokenizer (Türkçe-native 50k vs
çok-dilli mmBERT 256k) ve farklı eğitim, farklı sorularda isabet etmeli.

**Değişiklik.** `newmindai/ColmmBERT-small-TR` @ `3b5dd41`, `document_length`
180 → **512 override** (native pencere korpusun %50,3'ünü kesiyordu). Aday
düzeyinde üçlü birleşim. 2.530.392 vektör, 648 MB fp16, 170 sn kodlama.

**Sayılar** (insan-doğrulanmış, n=47; paraphrase n=21):

| kol | R@5 | R@20 | R@50 | paraphrase R@50 |
|---|---|---|---|---|
| BM25 | 0,5745 | 0,7021 | 0,7872 | 0,5714 |
| A Mogan | 0,6809 | 0,8723 | 0,9149 | 0,8095 |
| B ColmmBERT | 0,6064 | 0,7660 | 0,8298 | 0,8095 |
| **A+B** | **0,7234** | **0,8936** | **0,9362** | **0,8571** |
| BM25+A+B | 0,7021 | 0,8723 | 0,9362 | 0,8571 |

**Birincil metrik: `paraphrase` R@50 0,8095 → 0,8571 (+0,0476), 18/21.**
Hedef 0,90'a **bir soru** kaldı.

**Karar: KEPT.** Guardrail gerilemesi yok.

**Öğrenilenler.**

1. **İki model tek modelden iyi, çünkü FARKLI soruları ıskalıyorlar.** İkisi de
   tek başına 0,8095 veriyor ama birleşimleri 0,8571 — yani ıskaları örtüşmüyor.
   Ensemble çeşitliliği gerçek.
2. **BM25 artık R@50'ye HİÇBİR benzersiz katkı yapmıyor** (A+B ile BM25+A+B
   ikisi de 0,9362). Yine de TUTULDU: `dogrudan-madde` R@5'inde 0,8462 → 0,9231
   kazandırıyor ve 2–5 ms'e mal oluyor. Sıralamayı reranker çözecek; karar
   reranker ölçüldükten sonra yeniden bakılmalı.
3. **ColmmBERT'in native 180 penceresi bir tuzaktı** ve override edilmeseydi
   kanal korpusun yarısını görmeyecekti. Sözleşme okunur ama sorgulanır.

**Kanıt.** `data/bench/results/d2-multiarm.json`,
`data/index-colbert-colmm-f16/colbert.json`.

**Kalan üç ıska:** c206 (KVKK saklama süresi), c404 (ayıplı hizmet),
c411 (rekabet muafiyeti). c412 exp2 ile çözüldü.

---

## exp3 — genişletme token'larını MaxSim toplamından çıkar · **KEPT**

**Hipotez.** Sorgu genişletmesi 32 vektörün ~13'ünü `[MASK]` yapıyor. Bu
vektörler her belgeyle bir şeye eşleşiyor ve katkıları belge uzunluğuyla
korelasyonlu — yani sıralamaya uzunluk yanlılığı ekliyorlar.

**Değişiklik.** MaxSim toplamı yalnız GERÇEK sorgu token'ları üzerinden
(`attention_mask == 1`). İndeks değişmiyor, yeniden kodlama yok.

| kol | R@5 | R@20 | R@50 | paraphrase R@50 |
|---|---|---|---|---|
| taban | 0,7021 | 0,8723 | 0,9362 | 0,8571 |
| **genişletmesiz** | **0,7447** | **0,9149** | 0,9362 | 0,8571 |

**Karar: KEPT.** R@5 +0,0426, R@20 +0,0426, birincil metrik ve guardrail'ler
değişmedi. Not: eğitim rejiminden sapma (pylate 32 vektörün hepsini tutar), ama
ölçüm iki metrikte iyileşme, hiçbirinde gerileme gösteriyor.

---

## exp4 — başlık kanalı (dördüncü birleşim üyesi) · **DISCARDED**

**Hipotez.** Madde başlıkları ("Avukatlığa kabul şartları:") insan yazımı, kanun
dilinde, kısa etiketler — kayıt boşluğuna köprü olabilirler.

**Değişiklik.** 10.531 başlık+kanun-adı metni ayrı ColBERT indeksi (22 sn,
123.521 vektör, 32 MB), dördüncü birleşim üyesi.

| kol | R@5 | R@20 | R@50 | paraphrase R@50 |
|---|---|---|---|---|
| BM25+A+B | 0,7447 | 0,9149 | 0,9362 | 0,8571 |
| +BAŞLIK | 0,7234 | 0,8511 | 0,9149 | **0,8095** |

**Karar: DISCARDED.** Her metrikte gerileme; üstelik exp2'de kazanılan `c412`
tekrar kayboldu.

**BU DENEY BENİM İKİ KEZ SÖYLEDİĞİM BİR ŞEYİ ÇÜRÜTTÜ.** "R@50 bir küme
kapsama metriğidir, kanal eklemek onu düşüremez" dedim — **yanlış**. Doğrusu:
birleşim k'da KIRPILIYOR, yani örgüde her yeni kanal diğerlerinin slot'unu
yiyor. Zayıf bir kanal eklemek güçlü kanalların adaylarını top-k'nın dışına
iter. Kanal eklemek ancak TAM birleşim kümesi alınırsa zararsızdır; kırpılmış
bir listede değil.

---

## Kalan tıkanma ve teşhisi

`paraphrase` R@50 = 0,8571 (18/21). Kalan üç ıska: `c206`, `c404`, `c411`.

Workflow teşhisi (4 ajan, her biri gerçek korpusta kod koşarak) üçü için de aynı
kök nedeni buldu: **kayıt uyuşmazlığı (vocabulary gap)**. Gold'ların üçü de
doğru, chunk'lar bütün, kesme yok, ilgili madde tek bir chunk'ın içinde.

Korpus kanun dilinde yazılmış ("veri sorumlusu", "işlenmesini gerektiren
sebepler", "muafiyet şartları"), `paraphrase` soruları günlük dilde ("şirket",
"müşteri verisi", "cezadan kurtulma").

**SONDA (üst sınır ölçümü, dağıtılabilir sonuç DEĞİL).** Üç soruyu elle kanun
diline çevirip aynı kanallara sordum:

| soru | özgün sıra | çeviri sıra |
|---|---|---|
| c206 | 300 | **1** |
| c404 | bulunamadı | **1** |
| c411 | 88 | **1** |

Yani boşluk tamamen sözlükseldir. Model, chunking, kesme ve gold masum.

---

## exp5 — LLM sorgu yeniden yazımı (kanun diline çeviri kanalı) · **ENGELLENDİ**

**Hipotez.** Sonda kanıtladı: üç ıskanın da kök nedeni kayıt uyuşmazlığı ve elle
çevrilen sorgular rank 1 buluyor. Bir LLM kanalı bunu otomatikleştirirse
`paraphrase` R@50 0,8571 → ~1,0000 olmalı.

**Ön koşul — kural değişikliği.** `program-p2.md` kural 5 "sorgu yeniden yazımı
yok" diyordu. Değiştirdim: yeniden yazım artık EK kanal olarak serbest,
orijinali ikame edemez. **Sıra dürüstçe kayda geçti: kuralı sondanın sonucunu
bilerek değiştirdim** ve program dosyası bunu o şekilde yazıyor.

**Bench hijyeni.** İstem, dört sorunun konusunu hiçbir şekilde adlandırmıyor;
yalnız kayıt çevirisi ilkesini tarif ediyor. Yeniden yazımlar ÖLÇÜMDEN ÖNCE
dosyaya yazılacak ve commit'lenecekti.

**Sonuç: ÖLÇÜLEMEDİ.** Gemini kotası tükendi — 47 sorgunun 45'i 429
`RESOURCE_EXHAUSTED` aldı ve betik güvenli biçimde orijinal sorguya düştü.
Gerçekten çevrilen 2 satırla ölçüm yapmak, çeyreği dolu bir kanalı "ölçtüm"
diye raporlamak olurdu. Üretilen dosya SİLİNDİ; yanıltıcı bir artefakt repoda
durmasın.

**Bilinen:** mekanizma kanıtlı (sonda: 300/bulunamadı/88 → 1/1/1), kural
hazır, kod yolu hazır. Eksik olan tek şey kota.

**Sıradaki denemede:** ya kota yenilendiğinde bu deney koşulur, ya da LLM'siz
bir alternatif ölçülür — korpusun kendi madde başlıklarından üretilmiş
günlük-dil → kanun-dili sözlüğü (exp4'ün başlık kanalı BİRLEŞİM üyesi olarak
başarısızdı, ama SORGU GENİŞLETME kaynağı olarak denenmedi).
