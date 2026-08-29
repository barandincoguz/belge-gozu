# P1 hibrit üretimde + autoresearch round 2: R@5 0.116 → 0.8372'ye giden ölçüm zinciri

- **Tarih:** 2026-08-30 (oturum 2026-08-29 akşamı → gece, tam otonom)
- **Talimat:** "P1 entegrasyonunu başlat, tam otonom; sonra ölçümle autoresearch'ü tekrar başlat, deep dive; kapsamlı rapor."
- **Commit zinciri:** `ded732b` (P1 hibrit üretim) → `69e2dd0` (research lint) → exp8/round2 kapanış → `de2cc04` (inceleme düzeltmeleri + pencere-50)
- **SDD zinciri:** implementer (opus) → review (opus, APPROVE with fixes: 0 Critical/2 High/4 Medium/10 Low) → fix round (16/16 + addendum, itiraz 0) → re-review (sonnet, **ALL RESOLVED 17/17, 0 yeni bulgu** — sayılar bench artefaktından bağımsız yeniden hesaplandı)
- **Kanıt:** `research/journal.md` (#8-#10), `research/results.jsonl`, `data/research/robustness.json`, `data/bench/results/20260829-2115-3a031ca-hybrid.json`, `.superpowers` rapor kopyaları `docs/research/evidence/agent-reports/` altında

## 1. Yönetici özeti

İki vitrin sorusunun "bulamadım" alması şikâyetiyle başlayan zincir, üretimde çalışan
ölçülmüş bir hibrit retrieval hattıyla kapandı:

| Aşama | Canary R@5 (43 answerable, binary any-gold@5) | chip1 (TMK uzun) | chip2 (İş K. izin) |
|---|---|---|---|
| 1-bit görsel (eski üretim) | 0.116 | rank ~1221 | — |
| int8 görsel (b790f6c) | 0.2326 | 664 | 137 |
| Hibrit v1: BM25+F5+stop+pencere-20 (ded732b) | 0.8140 (35/43) | 2 | 2 |
| **Hibrit v2: pencere-50 (de2cc04, ÜRETİM)** | **0.8372 (36/43)** | **2** | **2** |

R@20 0.9302 · MRR 0.655 · requires_visual alt-kümesi 8/8 · fractional recall 0.8256
(aynı koşum, farklı metrik tanımı — her ikisi de raporda açık). Canlıda iki vitrin
sorusu da gerçek cevap + doğru madde atıfı veriyor; anlamsız sorgu (top1 4.23 < 10.6)
abstain'e düşüyor.

## 2. P1 üretim entegrasyonu (ded732b + de2cc04)

**Mimari:** `corpus/text.py` (PDF metin çıkarımı), `retrieval/text.py` (tokenizasyon:
tr_lower + ≥2 harf + sabit işlev-kelime listesi + F5 kırpma; BM25 k1=1.5 b=0.75;
doküman-adı çıkarımı — her dokümanın 1. sayfa büyük-harf başlığından, elle tablo yok;
`route_window`), `retrieval/hybrid.py` (HybridRetriever: görsel kanal telemetri/P2
verisi için hesaplanmaya devam eder, sıralama BM25-birincil + pencere-içi yönlendirme),
CLI `index build-text` (eksik-PDF reddi + `--allow-missing`), `<index_dir>/page_texts.parquet`
artefaktı (page_id eşitliği fail-fast). Varsayılan pipeline `hybrid`.

**Eşik:** BM25 ölçeğinde **10.6** — binary@60/int8@0.58'in çalışma noktasının (42/43 + 4/5)
mekanik taşıması; bant `(10.528, 10.712]` SUNULAN (yönlendirme sonrası) top-1 skorlarından
yeniden üretildi. Dağılımlar BM25 ölçeğinde de örtüşüyor (answerable min 10.53 / med 26.05;
unanswerable 4.23–23.53) → ayrım hâlâ YOK, kalibrasyon P2 (ilke korunuyor). Ölçek
korkulukları pipeline-farkında: hybrid (0,1.5] ve >200 reddeder, görsel hatlar >1.5;
negatif eşik (test kancası) her hatta serbest; ölçek haritası tek kaynaktan türetilir.

**Doğruluk güvencesi:** inceleme, üretim portunun araştırma reçetesiyle **bit-identical**
olduğunu mekanik kanıtladı (BM25 skorları birebir, tokenize 20k fuzz + Türkçe kenar
vakaları, `rank_pages == rank_all` uçtan uca). Eşik bandı ve bench sayıları bağımsız
yeniden üretildi. 296 test + lint yeşil; slow süit 4 passed + abstain kilidi xfail
(XPASS yok — BM25 ölçeğinde de korpus-dışılar eşik üstünde: 23.53/12.96/17.86).

**Gözlemlenebilirlik:** `bg_retrieval_top_score_bm25` + `bg_retrieval_score_margin_bm25`
(ölçek kimliği seri adında), Grafana'da BM25 paneli, `text_bm25`/`route_fuse` aşamaları,
healthz `pipeline` alanı. Gecikme: BM25 sorgu ~2-5 ms, başlangıç inşası 0.40 s
(ölçüldü); toplam gecikmeyi görsel kanal (sorgu encode) belirlemeye devam ediyor.

## 3. Autoresearch round 2 (derin dalış — kontrolcü, model'siz/ucuz)

| # | Deney | R@5 | Karar | Ders |
|---|---|---|---|---|
| 8 | pencere 20→50 | **0.8372**, R@20 0.9302↑ | KEPT → üretime | Yapısal garanti yerine ölçüm; yönlendirme gold'u top-20'ye de çekti |
| 9 | ayırt-edici-tek-token yönlendirme | 0.8372, R@1 −2, MRR↓ | DISCARDED | Fazla ateşliyor; hedef soru zaten ad içermiyormuş — hipotez veriden önce kurulmuştu |
| 10 | pencere-içi asimetrik RRF (görsel) | 0.5349 | DISCARDED | **Üçüncü füzyon biçimi de ölçümle reddedildi** |

**Füzyon üçlemesi (güçlü negatif sonuç):** küresel eşit-RRF 0.395 → mutlak doküman
bölümleme guardrail vetosu → pencere-içi RRF 0.535. Görsel kanalın başlık-sayfası
çekimi her granülaritede metin gold'larını eziyor; bu korpusta hayatta kalan tek
birleşim sözcüksel-birincil + kural yönlendirme.

**Sağlamlık (robustness.json):** k1∈[0.9,1.8]×b∈[0.5,0.9] → R@5 0.814–0.837 dar bandı;
F5∈{4..7} platosu (3 zarar, kırpmasız 0.767); pencere 10-20 eş, 30-50 +1. Reçete bıçak
sırtında değil. k1=1.8'in +1 sorusu bilinçli ALINMADI (canary'ye ayar = overfit; R24).
Bootstrap ΔR@5 (görsel taban → reçete) %95 GA **[0.42, 0.74]** — kesin.

**Dilim kırılımı (reçete vs görsel):** ayni-kanun-hard-negative 5/5 vs 1/5 ·
çapraz-kanun 4/4 vs 0/4 · madde-numaralı 6/6 vs 0/6 (pencere-50 ile) · tablo-layout
4/4 vs 1/4 · tarihi-tarama 4/4 vs 2/4 · dogrudan-madde 11/13 vs 6/13 ·
**paraphrase 2/7 vs 0/7 — sözcüksel tavan.**

**Kalan 7 ıska** (c101, c104, c111, c202, c206, c207, c209): TAMAMI kanun adı/kısaltma
içermeyen saf anlamsal paraphrase. Kural-tabanlı iyileştirme uzayı tükendi; sonraki
sıçrama **dense metin kanalı** ister (P1 backlog T7 — artık ölçülmüş gerekçesiyle:
hedef tam olarak paraphrase dilimi). Metin katmanı denetimi: 4222 sayfadan 4221'i
metinli (RG taramaları kaynak-OCR'lı, medyan 5.2k karakter); 16 zayıf sayfa (<200 kr)
görsel kanalın kalıcı gerekçesi.

## 4. Sınırlar (dürüstlük)

- Canary 48'in 45'i model-cross-check doğrulamalı (insan 3) — sayılar insan-doğrulanmış
  SAYILMAZ; benchmark v2 (İNSAN kapılı, plan T12) hâlâ backlog'da.
- 43 soruya yinelenen iterasyon = geliştirme-kümesi uyum riski; karşı tedbirler:
  ayarsız/kural-tabanlı değişiklikler, sabit listelerin canary'den bağımsız seçimi,
  duyarlılık platoları, k1=1.8 reddi. Yine de nihai doğrulama tutulmuş (held-out)
  set ister.
- Eşik hiçbir ölçekte ayırmıyor (tasarım gereği mekanik taşıma) — "halüsinasyon freni"
  hâlâ LLM'in dürüst-ıska davranışına yaslanıyor; kalibre selective answering P2.
- dogrudan-madde diliminin sözcüksel örtüşmesi BM25 lehine yapısal avantaj.

## 5. Sıradaki işler (öncelik sırasıyla, hepsi ölçülmüş gerekçeli)

1. **Dense metin kanalı (T7):** paraphrase 2/7'yi hedefler; model seçim koşumu planda.
2. **P2 kalibrasyon:** iki kanallı skor telemetrisi (görsel+bm25 birlikte kaydediliyor)
   kalibrasyon veri setini üretimde biriktiriyor.
3. **Benchmark v2 + insan doğrulama (T12)** — sayıların dış geçerliliği.
4. Bağlaşım denetimi hızlı dalgası (11 kalem) + hub/deploy P1 kalemleri (C3-C5, C11).
5. UI'da kanal şeffaflığı (görsel/bm25 skorlarını birlikte gösterme — veri zaten detail'de).
