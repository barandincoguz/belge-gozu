# Belge-Gözü — Tasarım Dokümanı

**Tarih:** 2026-08-25 · **Durum:** Kullanıcı onayı bekliyor · **Repo adı:** `belge-gozu`

> Visual document RAG for Turkish legal documents — sayfaları görüntü olarak indeksleyen,
> OCR/parse kullanmayan, kendi benchmark'ıyla ölçülen uçtan uca bir sistem.

## 1. Amaç ve bağlam

Bu proje bir vitrin (portfolio) projesidir. Birincil hedefi teknik değil stratejiktir:
kullanıcının işe alınma şansını artırmak ve GitHub'ına bakan teknik değerlendiriciye
"bu insan gerçekten kuruyor ve anlıyor" dedirtmek.

100 ajanlı derin araştırmanın doğrulanmış bulguları tasarımın zeminidir
(rapor: https://claude.ai/code/artifact/566a8c77-e385-4e4c-959f-c8178cfb328b):

- ML mülakatçıları tek projeyi seçip her kararın hesabını soruyor → her tasarım kararının
  bir cümlelik savunusu olmalı; bu doküman o savunuların kaynağıdır.
- Klişe projeler (MNIST/Titanic/jenerik RAG sarmalayıcı) anında eleme malzemesi →
  bu proje, metin-RAG klişesinin yapısal panzehiri olan görsel belge RAG alanında
  (ColPali sınıfı, ICLR 2025; NDCG@5 81.3 vs OCR hattı 67.0) kurulur.
- Kendi tanımladığın problem + kendi ürettiğin veri/benchmark en güçlü sahiplik sinyali →
  proje, Türkçe görsel-belge retrieval için ilk açık mini-benchmark'ı üretir.

## 2. Karar günlüğü (kullanıcıyla birlikte alındı)

| Karar | Seçim | Gerekçe |
|---|---|---|
| Korpus | Mevzuat + Resmî Gazete (tarihî taramalar dahil) | Kamu malı (FSEK m.31), bol, OCR'ın gerçekten çöktüğü taramalar farkı dramatize eder |
| Tempo | Yoğun (20+ saat/hafta), Claude Code ile eşli çalışma | Hafta 1 sonunda canlı v0; 2-3 haftada tamamı. 1 haftada "bitti" iddiası reddedildi: duvar saati (embedding/scraping), benchmark KK'sı ve mülakat içselleştirmesi kodlama hızıyla sıkışmaz |
| Yanıtlayıcı | Tak-çıkar (pluggable) — LocalVLM + FreeAPI (+ mümkünse ZeroGPU) | Demo hızlı yoldan çalışır, repo tam açık yolu belgeler, benchmark ikisini de ölçer |
| Bütçe | Bedava-öncelikli | Her şey ücretsiz katmanda tasarlanır; yalnızca demo UX gerçekten aksarsa 1-2 aylık HF PRO opsiyonu belgelenir |
| Mimari | A: Yekpare repo, gömülü indeks | Kendi yazılmış iki aşamalı MaxSim çekirdeği; dış servis yok; Qdrant sonradan takılabilir adaptör |
| RAG derinliği | Kademeli sıralama (VLM rerank) + sorgu yeniden yazımı + agentic mod | Kullanıcı isteği (2026-08-25): "düz RAG değil, kaliteli ve agentic RAG" sinyali. Şart: her katman benchmark'ta açık/kapalı ablasyonla ölçülür — süs katman yok |

## 3. Sistem mimarisi

İki dünya:

```
ÇEVRİMDIŞI (Colab ücretsiz GPU + lokal M4 Pro)
  mevzuat.gov.tr ─┐
  resmigazete.gov.tr ─┴→ indirici (rate-limited) → PDF→WebP sayfa görüntüleri (~150dpi)
      → embedder (ColQwen2-3B sınıfı + ColSmol-500M sınıfı, parti parti)
      → float16 embedding + binary kopya + metadata parquet
      → HF Datasets'e push (görüntüler / embeddingler / metadata — sürümlü)

ÇEVRİMİÇİ (HF Space, ücretsiz CPU, Docker)
  açılışta: binary indeks + metadata indir, mmap'le
  sorgu → sorgu yeniden yazımı (günlük dil → hukuk dili, çoklu sorgu; önbellekli)
        → sorgu kodlayıcı (küçük retriever, CPU)
        → Aşama 1: havuzlanmış sayfa vektörüyle Hamming eleme → top-200
        → Aşama 2: binary MaxSim (late interaction) yeniden sıralama → top-20
        → Aşama 3: VLM yeniden sıralayıcı (cross-encoder tarzı) → top-k sayfa
        → Hızlı mod: Answerer (tak-çıkar) → atıflı yanıt
          Derin mod: agentic döngü (ayrıştır → ara → öz-denetle → dayanaklılık) → atıflı yanıt
```

Veri sürümleme DVC ile değil HF Datasets revizyonlarıyla yapılır: tek araç, bedava
depolama/CDN, `revision=` ile tekrarlanabilirlik. (Savunu: araç sayısını azaltır,
ücretsiz katmanda büyük dosya problemini çözer.)

## 4. Korpus ve veri hattı

**Kapsam (tam korpus, hafta 2):**
- Yürürlükteki kanunlar (~700 adet) + seçilmiş temel yönetmelikler (ölçüt: temel kanunların
  uygulama yönetmelikleri + en bilinen ~100 yönetmelik; liste G1'de sabitlenir) → hedef 15-30 bin sayfa.
- Tarihî Resmî Gazete örneklemi: 1930'lar-1980'lerden taranmış sayılar → 2-5 bin sayfa
  (OCR-kırıcı dilim; benchmark'ın (c) dilimini besler).

**v0 dilimi (hafta 1):** ~2 bin sayfa — en bilinen ~50 kanun + ~100 tarihî sayfa.

**Boru hattı adımları** (her biri typer CLI komutu, idempotent, kaldığı yerden devam eder):
1. `corpus download` — nazik scraping: rate limit, retry, User-Agent, indirilenlerin manifesti.
2. `corpus render` — PDF→WebP sayfa görüntüleri (~150dpi, hedef ~100-200KB/sayfa).
3. `index build` — iki modelle embedding (Colab defteri repo içinde), float16 + binary çıktı.
4. `index push` / `index pull` — HF Datasets senkronizasyonu.

**Metadata şeması (parquet):** `page_id, belge_id, belge_adi, belge_turu, tarih,
rg_sayi, sayfa_no, kaynak_url, taranmis_mi, webp_yolu`.

**Hukuki zemin:** Resmî metinler FSEK m.31 gereği telif koruması dışındadır; scraping
nazik ve kimlikli yapılır. README'de veri kaynağı ve yasal dayanak açıkça belirtilir.

## 5. Retrieval çekirdeği

**Neden kendi çekirdeğimiz:** Ücretsiz CPU katmanının bellek/bağımlılık kısıtına göre
tasarım + mülakatta "late-interaction'ı kütüphanesiz kurdum" hikâyesi. ~50-100 satır.

- **Temsil:** Sayfa başına ~1030 token embedding'i (retriever modeline göre değişir),
  128 boyut, binarize edilmiş (1 bit/boyut) → ~16KB/sayfa. 30 bin sayfa ≈ ~500MB, mmap.
  float16 asıllar yalnızca HF Datasets'te durur (Space'e inmez).
- **Aşama 1 — eleme:** Sayfa token'larının ortalaması → binarize → tek "sayfa vektörü".
  Sorgu token'larının ortalamasıyla Hamming mesafesi (numpy uint64 XOR + popcount,
  vektörize). 30 bin sayfada milisaniyeler. → top-200 aday.
- **Aşama 2 — kesin sıralama:** Adayların tam token matrisleriyle binary MaxSim:
  her sorgu token'ı için aday sayfa token'larıyla benzerlik (Hamming→benzerlik),
  maksimumların toplamı. → top-20.
- **Aşama 3 — VLM yeniden sıralayıcı:** MaxSim top-20'sindeki sayfa görüntüleri, soruyla
  birlikte küçük VLM'e puanlatılır (pointwise, cross-encoder'ın görsel karşılığı) → top-k
  (varsayılan 5). Katkısı ve gecikme bedeli benchmark'ta ölçülür; config'ten kapatılabilir —
  klasik "retrieve → rerank" kademesinin (cascade) üçüncü basamağı.
- **Arayüz:** `Retriever.search(query: str, k: int) -> list[PageHit]` protokolü.
  İleride Qdrant adaptörü bu arayüzün ikinci uygulaması olur (kapsam dışı, bkz. §12).
- **Ablasyonlar (benchmark'ta sayıya bağlanır):** binarization kaybı (float16 vs binary),
  aday sayısı (50/200/500), retriever boyutu (3B vs 500M sınıfı), VLM rerank açık/kapalı,
  sorgu yeniden yazımı açık/kapalı, Hızlı vs Derin (agentic) mod.
- **Sorgu kodlayıcı:** Serviste küçük retriever (ColSmol-500M sınıfı) CPU'da koşar
  (hedef 1-3 sn); 3B model kalite referansı olarak yalnız benchmark'ta kullanılır.
  Kesin model adları/sürümleri inşa günü doğrulanır (araştırma uyarısı: alan hızlı eskiyor).

## 6. Yanıtlama ve RAG orkestrasyon katmanı

**Arayüz:** `Answerer.answer(question: str, pages: list[PageHit]) -> Answer{text, citations}`.

Uygulamalar (config ile seçilir, demo UI'da hangisinin aktif olduğu görünür):
1. **FreeAPI** — ücretsiz API katmanı (Gemini Flash sınıfı; inşa günü seçilir). Demo varsayılanı.
2. **LocalVLM** — quantized Qwen2.5-VL-3B / SmolVLM2 (llama.cpp). M4 Pro'da hızlı;
   Space CPU'sunda yavaş ama çalışır — README'de dürüstçe belgelenir.
3. **ZeroGPU** (opsiyonel) — HF ZeroGPU kotası inşa günü doğrulanır; uygunsa demo bu yola geçer.

**Korkuluklar:**
- Yanıt daima atıflıdır: kullanılan sayfaların küçük resimleri + belge adı + resmî kaynak linki.
- Retrieval güven eşiği: en iyi MaxSim skoru eşik altındaysa answerer çağrılmaz,
  "bu korpusta bulamadım" döner. Eşik, benchmark'ın halüsinasyon ölçümüyle ayarlanır.
- Prompt, yanıtı yalnızca verilen sayfalara dayandırmaya zorlar; sayfa dışı bilgi işaretlenir.

**Sorgu yeniden yazımı (retrieval öncesi):** Günlük dildeki soru hukuk terminolojisine
çevrilir ve gerekirse çoklu sorguya açılır (multi-query genişletme; HyDE ailesinden
teknikler değerlendirilir). Ör: "kira artışı en fazla ne olabilir" → "TBK kira bedeli
artışı TÜFE oranı sınırı". FreeAPI ile yapılır, sonuçlar önbelleğe alınır, config'ten
kapatılabilir; katkısı benchmark ablasyonuyla ölçülür.

**Agentic mod — "Derin arama":** Bir yönlendirici, soruyu basit/karmaşık diye sınıflar
(adaptive retrieval): basit soru tek atışta yanıtlanır; karmaşık/çok-adımlı soruda ReAct
tarzı döngü çalışır (en fazla 3 tur):

1. Soruyu alt sorulara ayır (query decomposition).
2. Her alt soru için retrieval kademesini çalıştır.
3. Kanıt yeterliliğini öz-denetle (Self-RAG tarzı refleksiyon) — eksikse yeni arama turu.
4. Dayanaklılık (groundedness) kontrolü: yanıttaki her iddia atıf sayfalarınca destekleniyor mu?
   Geçemeyen yanıt yerine "bulamadım" döner.

Tur sayısı ve API çağrısı sınırlıdır (ücretsiz katman bütçesi); UI, Derin modda adım adım
ajan izini (trace) gösterir — demonun vitrin özelliği. Hızlı mod her zaman ayaktadır.

## 7. Benchmark ve değerlendirme — `belge-gozu-bench`

**İçerik:** ~115 soru, dört dilim:
- (a) Güncel mevzuat metin soruları (~40) — düz metin sayfalarından.
- (b) Tablo/düzen soruları (~30) — tablodan değer okuma, çok sütunlu düzen.
- (c) Tarihî taranmış sayfa soruları (~30) — OCR-kırıcı dilim.
- (d) Çok adımlı (multi-hop) sorular (~15) — birden fazla belge/sayfa gerektiren;
  agentic modun sahası (Hızlı vs Derin karşılaştırması bu dilimde yapılır).

**Kayıt şeması:** `soru, dogru_sayfa_idleri (1+), referans_yanit, dilim, zorluk`.

**Üretim ve kalite kontrol:** Taslak soru-cevaplar sayfalardan yarı-otomatik üretilir
(Claude/VLM destekli), ardından **her çift kullanıcı tarafından elle doğrulanır/düzeltilir**
(~1-2 gün). Bu adım pazarlık konusu değildir: benchmark projenin güvenilirlik çekirdeğidir
ve mülakatta savunulacak kısımdır.

**Ölçülenler:**
- Retrieval: NDCG@5, Recall@5 — dilim bazında kırılımla.
- Uçtan uca: yanıt doğruluğu (LLM-judge + kullanıcı spot-check'i), "bulamadım" doğruluğu
  (cevapsız sorularda yanlış yanıt üretmeme).
- Operasyonel: sorgu gecikmesi (p50/p95), yanıtlayıcı başına maliyet/1000 soru, mod bazında.
- **Katman katkı tablosu (README'nin vitrini):** yeniden yazım / MaxSim / VLM-rerank /
  agentic mod — her katmanın açık/kapalı ablasyonla NDCG@5 ve uçtan uca doğruluğa katkısı,
  gecikme/maliyet bedeliyle yan yana. "Agentic RAG ne kazandırdı?" sorusunun sayısal cevabı.

**Baseline (dürüst rakip):** OCR (Türkçe için Tesseract/PaddleOCR — inşa günü seçilir)
→ metin chunk'lama → çok dilli dense embedding (e5/bge sınıfı) → aynı sorularla ölçüm.

**Yayın:** Benchmark HF Dataset olarak açık yayınlanır; README'de sonuç tabloları +
kısa analiz (nerede kazanıyoruz, nerede kaybediyoruz — kayıplar saklanmaz).

## 8. Web uygulaması ve deploy

- **Backend:** FastAPI. Uçlar: `POST /search`, `POST /ask`, `GET /stats`, `GET /healthz`.
- **UI:** Tek sayfa, sade-özenli (Gradio değil): arama kutusu, Hızlı/Derin mod anahtarı,
  sayfa küçük resimleri (skorlarıyla), atıf kartlı yanıt, Derin modda adım adım ajan izi
  (trace) paneli, aktif answerer rozeti. Türkçe/İngilizce dil anahtarı.
- **Deploy:** HF Space (Docker SDK, ücretsiz CPU). Space uykudan ziyaretçiyle uyanır —
  README ve UI'da "ilk açılış ~1 dk sürebilir" notu.
- **İzleme-lite:** İstek logları (sqlite/parquet): zaman, gecikme, retrieval skoru,
  answerer türü. `/stats` basit özet döner. Amaç production farkındalığı sinyali; abartısız.

## 9. Mühendislik hijyeni

- **Testler:** pytest. Kritik: binary MaxSim çekirdeği float referans uygulamayla
  özdeşlik/tolerans testinden geçer; scraper ve render idempotenlik testleri; API smoke.
- **Kalite:** ruff (lint+format), pyright (temel mod), pre-commit.
- **CI:** GitHub Actions — lint + test + küçük örneklem smoke (indeks fikstürüyle).
- **CLI:** typer — `corpus download|render`, `index build|push|pull`, `bench run`, `serve`.
- **Config:** pydantic-settings; tüm eşikler/model adları tek config dosyasında.
- **README (İngilizce):** mimari diyagram, canlı demo linki, benchmark tabloları,
  "Design Decisions" bölümü (bu dokümandaki savunuların özeti), veri/yasal not.
- **Dizin yapısı:**

```
belge-gozu/
  src/belge_gozu/{corpus, index, retrieval, answer, bench, app}/
  notebooks/ (Colab embedding defteri)
  tests/
  docs/superpowers/specs/
  .github/workflows/ci.yml
```

## 10. Kilometre taşları

**Hafta 1 → canlı v0:**
- G1: repo iskeleti, CI, config; scraper başlar.
- G1-2: v0 korpus (~2 bin sayfa) indirildi + render edildi.
- G2-3: Colab'da embedding; HF Datasets'e push.
- G3-4: retrieval çekirdeği + testleri.
- G4-5: FreeAPI answerer + UI + Space deploy → **canlı link**.

**Hafta 2 → asıl et:** tam korpus (~20-30 bin sayfa) + tarihî dilim; sorgu yeniden yazımı
+ VLM rerank kademesi; benchmark taslakları → kullanıcı doğrulaması; OCR baseline;
ilk eval koşuları; LocalVLM yolu.

**Hafta 3 → agentic katman + cila:** Derin mod (agentic döngü) + multi-hop dilimi;
katman-katkı ablasyonları ve tabloları; README/analiz; benchmark yayını; gecikme
iyileştirmeleri; opsiyonel ZeroGPU; duyuru taslağı (LinkedIn/blog).

Dürüst not: RAG-derinliği kararıyla hafta 3'ün tampon payı agentic katmana gitti.
Takvim hâlâ 3 hafta, ama sürpriz çıkarsa sarkma payı ~yarım haftadır — kapsam
genişletmenin bilinçli bedeli.

## 11. Riskler ve azaltmalar

| Risk | Azaltma |
|---|---|
| Scraping sürprizleri (site yapısı, engel) | Manifest + idempotent indirici; erken başla (G1); gerekirse kapsamı kanun PDF'lerine daralt |
| Colab kesintileri | Parti parti embedding + checkpoint; ~30 bin sayfa birkaç oturuma bölünür |
| Sorgu kodlayıcı CPU'da yavaş kalır | Küçük retriever birincil; olmazsa ZeroGPU; en kötü retrieval-only mod hızlı kalır |
| Ücretsiz API katmanı kota/politika değişimi | Answerer tak-çıkar; ikinci ücretsiz sağlayıcı yedekte; LocalVLM her zaman çalışır |
| Model adları/sürümleri eskimiş çıkar | İnşa günü (G1) HF'te güncel ColPali/ColSmol/VLM sürümleri doğrulanır; config'te izole |
| Benchmark kalitesi zayıf kalır | Kullanıcı doğrulaması zorunlu adım; ana dilimlerde ≥25, multi-hop diliminde ≥10 doğrulanmış soru olmadan sonuç yayınlanmaz |
| Agentic mod ücretsiz katmanın gecikme/kota bütçesini zorlar | Tur sınırı (≤3), yalnız Derin modda, yeniden yazım önbelleği; Hızlı mod her zaman ayakta |
| Space belleği yetmez | Binary + mmap tasarımı; gerekirse korpus dilimlenir (öncelik: benchmark bütünlüğü) |

## 12. Kapsam dışı (YAGNI)

- Model fine-tuning'i (retriever veya VLM) — gelecek iş olarak README'de not edilir.
- Qdrant adaptörü — arayüz hazır, uygulaması stretch goal.
- Kullanıcı hesabı/auth, çok kiracılılık, ödeme, mobil uygulama.
- Mevzuat dışı belge türleri (mahkeme kararları vb.) — benchmark'ı bulandırır.
- Otomatik güncel-mevzuat senkronu (günlük cron) — stretch; v1'de statik korpus.
- Hibrit metin+görsel füzyon (gömülü PDF metniyle BM25 + RRF birleştirme) — ilginç
  ablasyon ama tezi ("OCR'sız/parse'sız") bulandırır; stretch olarak README'de not edilir.

## 13. Başarı kriterleri (bitti'nin tanımı)

1. Canlı demo: HF Space'te çalışan, atıflı yanıt veren, herkese açık link — Hızlı ve
   Derin (agentic, izli) modlarıyla.
2. `belge-gozu-bench` HF'te yayında; ≥85 kullanıcı-doğrulamalı soru, 4 dilim.
3. README'de görsel-RAG vs OCR-baseline karşılaştırma tablosu (kayıplar dahil, dürüst).
4. README'de katman katkı tablosu: yeniden yazım / rerank / agentic modun ölçülmüş
   katkı ve bedelleri.
5. CI yeşil; çekirdek testleri geçiyor; `git clone` → `make demo` ile lokalde çalışıyor.
6. Kullanıcı, bu dokümandaki her kararı bir cümleyle savunabiliyor (mülakat provası yapılacak).
