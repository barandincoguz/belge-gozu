# Konfigürasyon–bağlaşım denetimi: eşik-temsil tuzağı sınıfının tam envanteri ve triyajı

- **Tarih:** 2026-08-29
- **Taban:** `feat/p0-retrieval-correctness` @ dd4f251 (denetim anı; int8 geçişi öncesi)
- **Tetik:** Kullanıcı talimatı — "eşiği int8 yap; bu ve bunun gibi ince gözden kaçabilecek bütün konfigürasyonları ve benzeri kodları incele."
- **Yöntem:** 3 bağımsız read-only denetim ajanı (opus) + kontrolcünün yerel doğrulamaları + canlı int8 eşik-taşıma ölçümü. Ham envanterler (139 bulgu satırı + verified-safe listeleri) birebir arşivde:
  - `docs/research/evidence/agent-reports/2026-08-29-audit-scale-units.md` (S1–S60)
  - `docs/research/evidence/agent-reports/2026-08-29-audit-config-env-deploy.md` (C1–C42)
  - `docs/research/evidence/agent-reports/2026-08-29-audit-duplicated-contracts.md` (D1–D37)
- **İlişkili:** `docs/research/findings/2026-08-29-e2e-review.md` (K1–K34; bu denetim K2/K3/K10 hattının derinleştirilmesidir), `data/bench/results/int8-threshold-transfer.json` (eşik taşıma ölçümü).

Not: S/C/D numaraları yukarıdaki üç arşiv dosyasının tablo satırlarına işaret eder. Satır numaraları dd4f251 anlık görüntüsüne aittir.

---

## 1. Çekirdek tuzak ve çözümü

**Tuzak:** `min_score_threshold=60.0` binary skor ölçeğine (per-token `128 − 2·Hamming`, aralık ≈[−128,128]) gömülüydü; int8/float yolları ise L2-normalize dot-product ortalaması (≈[−1,1]) döndürüyor. Config yalnız `index_dir` taşıyor, **temsil kimliği hiçbir değerle birlikte yolculuk etmiyordu.** Sonuç: indeksi int8'e çevirmek eşiği sessizce "her şeyden büyük" yapar → %100 abstain, `/healthz` "ok" demeye devam eder (C7). Üç denetim bu tuzağın *aynı sınıftan* 100+ akrabasını buldu: Prometheus kovaları (S6/S7), UI fallback'i ve ondalık formatı (S11/S14), cırcır beklentisi (S48), `xfail(strict=True)`'ın XPASS'e dönüp süiti ters yönde kırması (S26/C17), `index build`'in int8 dizinini sessizce 1-bit ile değiştirmesi (C6), hub pull'un karma dizin üretmesi (C4/S4)…

**Çözüm (int8 geçiş commit'i `b790f6c`, 2026-08-29):** iki parça —

1. **Tek ölçek:** Bütün skorlayıcılar (`PackedIndex`, `Int8Index`, `FloatIndex`) normalize per-query-token ortalama ≈[−1,1] döner; binary yol `/EMBED_DIM` (=128) ile aynı ölçeğe çekildi. Formül üç uygulamada zaten tutarlıydı (S denetimi verified-safe teyidi); yalnız aralık birleştirildi. Böylece "hangi temsil?" sorusu skorun *değerini* değil yalnız *kalitesini* etkiler.
2. **Kimlik + korkuluk:** `load_scorable_index` manifest.quantization'a göre dispatch eder; eşik 0.58'e **mekanik ölçek taşıması** ile geçirildi (kalibrasyon DEĞİL — ilke 21 korunuyor; binary@60.0'ın çalışma noktası 42/43 answerable + 4/5 unanswerable birebir); `create_app` `threshold > 1.5` görürse binary-kalıntı diye fail-fast; `/healthz` + Prometheus etiketi + `events.index_revision` temsil kimliğini dışarı taşır; cırcır dosyası temsil anahtarıyla kilitli.

**Ölçüm (eşik taşıma, int8, MPS):** answerable n=43 min 0.5767 / medyan 0.6250 / maks 0.7450; unanswerable n=5: 0.5679–0.6866. Dağılımlar int8'de de örtüşüyor → **ayrışmama temsilden bağımsızdır**; kalibrasyon P2'nin işi olarak değişmedi. Hedef sorgular: kısa sorgu gold rank 4 (top1 0.7450), uzun sorgu gold rank 664 (1-bit'te 1221 idi — int8 kalite kazancının cırcıra yansıması).

**Ek ölçüm (scoped review, 2026-08-29) — eşik temsiller arası TAŞINAMAZ:** b790f6c incelemesi, 1-bit indeksin AYNI normalize ölçekteki ([−1,1], /EMBED_DIM) canary top-1 dağılımını ölçtü: answerable min 0.4676 / medyan 0.4953 / maks 0.6133 → 0.58 eşiğini 1-bit'te yalnız **1/43** answerable geçer (int8'de 42/43). Yani **normalizasyon aralığı birleştirir, dağılımı birleştirmez** — eşik, ölçek değil temsil dağılımı üzerinde tanımlıdır ve her temsil değişiminde yeniden taşıma ölçümü gerekir (README/config bu uyarıyı taşır; temsil-başına eşik konfigürasyonu bilinçli olarak EKLENMEDİ — R19, kalibrasyon P2). İkinci incelik: 0.58 taşıması çalışma noktasını SAYICA korur (42/43 + 4/5) ama soru-kimliği bazında iki satır yer değiştirir — c306 binary@60'ta abstain iken int8@0.58'de geçer (0.5965), c211 tersi (0.5767) — çünkü int8 ile binary sıralamaları/skorları farklıdır. Monotonik taşıma sayıları korur, kimlikleri korumaz; makale için not edilmiştir. Düzeltme turunda bağımsız yeniden üretildi; 1-bit için eşdeğer çalışma noktası bandı `(0.4676, 0.4698]` ≈ **0.47** ölçüldü — yani 1-bit'e dönülecek olsa eşik de ~0.47'ye taşınmalıdır (README/config uyarısı bunu söyler).

---

## 2. Triyaj

### A. Bu geçişle kapatıldı (int8 geçiş commit'i)

| Konu | Denetim ref | Ne yapıldı |
|---|---|---|
| Eşik-temsil bağı | C7, S3, S25 | Tek ölçek + 0.58 mekanik taşıma + config yorumu ölçek künyeli |
| Ölçek korkuluğu | (sınıfın kendisi) | `create_app`: `threshold > 1.5` → fail-fast (eski binary değer kalıntısı reddi) |
| Loader dispatch | C1, S2, S3, C23, S49 | `load_scorable_index` (manifest.quantization → Packed/Int8/Float); serve + bench run + d1 + canary fixture aynı yolu kullanır |
| two-stage × int8 çapraz kontrolü | C22 | two-stage yalnız PackedIndex; aksi halde anlamlı IndexCompatibilityError |
| create_app ↔ canary fixture kopyası | D22, C18 | `build_retriever()` çıkarıldı; fixture üretim kablolamasının kendisini çağırır |
| `index build` karma-dizin tuzağı | C6 | `--out` yokken hedef manifest'inin quantization'ı uyuşmuyorsa red (mevcut format guard'ının genişletilmesi) |
| Quantization vokabüleri | D7 | `Quantization` StrEnum `index/manifest.py`'a taşındı + `float16` üyesi; çıplak literaller kapandı |
| Prometheus kovaları | S6, S7, C29 | SCORE/MARGIN kovaları normalize ölçeğe |
| Histogram temsil kimliği | S8 | `bg_retrieval_top_score` + `bg_retrieval_margin_1_2` histogramlarına `quantization` etiketi |
| UI fallback + format | S11, S14, S15, D16 kısmen | JS THRESHOLD fallback 0.58; skor/eşik 2 ondalık |
| UI negatif skor çubukları | S12, S13 | Genişlik/eksen `Math.max(0, …)` ile kırpıldı |
| UI metinleri | S16 kısmen, S29 | Dipnot temsil-nötr ("kalibre edilmemiş benzerlik ~[−1..1]"); "binary" ibaresi düştü |
| /healthz kimliği | S17 kısmen, D17, S30 | `quantization`, `index_revision`, `top_k` eklendi; UI "ilk 5" top_k'dan |
| EMBED_DIM / INT8_MAX | S37, S38, S39 | `EMBED_DIM=128`, `INT8_MAX=127` tek tanım; `_as_u64` şekil guard'ı; normalizasyon `/EMBED_DIM` |
| Cırcır temsil anahtarı | S48, C32, D33 kısmen | `canary_expectations.json` `quantization` alanı taşır; test yüklü manifest'le eşleştirir; 1221→664 |
| xfail XPASS riski | S26, C17 | Abstain kilidi int8 sayılarıyla yeniden yazıldı; 0.58'de hâlâ FAIL (xfail geçerli) — korpus-dışı top1'ler eşik üstünde |
| Config drift kilitleri | C21 | `test_defaults` artık index_dir + eşiği assert ediyor |
| colpali-engine pin | C13 kısmen | `==0.3.18` (format sözleşmesinin asıl kilidi) |
| FloatIndex katman ihlali | (devir notu D4) | `index/float_store.py`'a taşındı; oracle re-export; `score_all(chunk_tokens)` kazandı (S40 da kapandı) |
| README ölçek/temsil iddiaları | C24–C27, S29, D19 kısmen | Quickstart int8; skor tanımı/eşik/boyut/gecikme güncel; n_tokens 3.776.882→3.759.994 düzeltmesi (üretim = train-compat) |

### B. Hızlı dalga — önerilen küçük bağımsız düzeltmeler (davranış-nötr, ayrı commit)

| Konu | Ref | İş |
|---|---|---|
| `query_format_id` tipi | C8, D8 | `QueryFormatChoice`/Literal yap → bozuk env temiz ValidationError versin |
| CLI import-anı çökmesi | C9 | `_CLI_DEFAULTS = Settings()` etrafına ValidationError yakalama → okunabilir mesaj + exit 2 |
| `write-manifest --legacy` | C10 | Mevcut `manifest.json` varsa reddet (üretim manifest'ini ezme) |
| `index derive --out` dolu hedef | S5 | Hedefte bilinen indeks dosyaları varsa reddet |
| CI lock disiplini | C12 | `uv sync --locked`; (imaj build adımı P1) |
| Test env izolasyonu | C19 | autouse fixture: `BG_*` env temizliği + `_env_file=None` |
| Docker gizlilik varsayılanı | C38 | `ENV BG_LOG_QUERY_TEXT=false` |
| ABSTAIN_TEXT drift-lock | D1 ara adım | index.html içinde `ABSTAIN_TEXT` birebir geçiyor mu testi (kalıcı çözüm P1'de alanla) |
| Fikstür fabrikası kopyası | D21 | `q_dict` tek yerde |
| Ölü sabit | C42 | `DEFAULT_MANIFEST` sil |
| FloatIndex padding reddi | C41 | `build`'e all-zero kontrolü (store ile simetri) |

### C. P1'e devredilen (API/deploy/altyapı değişikliği gerektirir)

- **Hub/deploy zinciri:** push slot semantiği (C3), pull atomikliği (C4/S4 — loader dispatch tuzağı "sessiz yanlış skor"dan "bayat-ama-tutarlı ya da gürültülü hata"ya indirdi; atomik takas yine gerekli), Dockerfile `--pull` sessiz no-op (C5), `HF_KEY`→`HF_TOKEN` (C11), uzak repo bayatlığı (C28), göreli yol/CWD (C36).
- **Build yolu model pinleri:** `from_pretrained(revision=…)` (C15), `_commit_hash`→"unknown" sessiz atlaması (C16), engine_versions karşılaştırması (C14), `transformers`'ı açık bağımlılık yapma (C13 kalanı).
- **API sözleşmesi:** `/ask`'a `abstain_reason`/`status` alanı — ABSTAIN_TEXT/SERVICE_ERROR_TEXT string bağının kalıcı çözümü (S33, S34, D1, D2); `k`/`question` doğrulama sınırları (S51, S52); yanıt şeması tam-anahtar kilitleri (D30).
- **honest_miss sözleşmesi:** `HONEST_MISS_MARKER` tek sabit + `tr_lower` paylaşımı (S35, D3).
- **Vokabüler birleştirmeleri:** pipeline StrEnum + match/raise (D6, S36), telemetri StageName enum (D4), bench stage sözlüğü (D5), stage-duration temsil etiketi (S24), mask_policy enum (D25).
- **Telemetri/katalog:** metrics-catalog üreteci + 8 kalem sapmanın kapanışı (D12, S22, S23), şema üçlü-liste kilidi (D11), events `index_revision` NOT NULL/score_scale (S10), p95 pencere eşitliği (S55, D14), abstain payda tanımı (S21), dashboard eşik çizgisi/alert/panel onarımları (S18, S19, S20, D13), maliyet metriği model etiketi (S53), tps adlandırması (S54).
- **Bench altyapısı:** tam-korpus rank kaydı (S45), tek `BENCH_KS` (S46, D23), `record_top` kesmesi (S47), oracle rapor `score_scale` notu (S50), `DEFAULT_CANDIDATES` (D24).
- **Provenance:** render dpi/quality gerçeğinin manifest'e akması (S43, C30, D26), `ab_st_reference` render bağı notu (S58).
- **Yardımcı birleştirmeler:** `page_id`/görüntü yolu tek modül (S41, S42, D9, D10), encode determinizm garantisinin `encode_pages` içine taşınması (S44), `data_dir` türetmeleri (C34, C35), port/timeout tablosu (S56, S57, D32), örnek sorgular tek dosya (D36), UI künye alanlarının healthz'ten gelmesi — kalanlar: pipeline/model adı/tarih aralığı (S60, D18, D20), pacing/aşama seçimi (S31, S32), `[Sn]` sabitleri (D31), CPE prompt canlı doğrulaması (C39), `/healthz` answerer_ready (C37 — E2E öncelik #4 LLM-freni ile birlikte).
- **Canary süreci:** insan onayının `verification_kind`'ı yükseltmesi + `require_human` (D27), şema dokümanının modelden üretimi (D37), README rakam-kilidi testi (D19 kalanı, D18).

### D. P2'ye bağlı (kalibrasyonla birlikte anlamlı)

- Eşiğin kendisi: 0.58 ayrım YAPMIYOR (tasarım gereği — mekanik taşıma). Kalibre selective answering P2; xfail(strict=True) kilidi bunu bekliyor.
- Göreli margin metriği (S9) ve SCORE_BUCKETS'ın eşikten türetilmesi (D16 kalanı) — kalibrasyon eşiği oynatmaya başladığında.

### E. Reddedilen / farklı çözülen (gerekçeli kararlar)

| Öneri | Ref | Karar ve gerekçe |
|---|---|---|
| `Settings.expected_quantization` + compat'a quantization karşılaştırması | C2, S27 | **Reddedildi.** Manifest tek otoritedir; loader ona göre dispatch eder, kimlik /healthz + prom etiketi + `events.index_revision` ile görünür. İkinci bir "beklenen temsil" ayarı, dizin değiştirildiğinde bayatlayacak AYNI sınıftan yeni bir çift-kaynak tuzağı olurdu. Yanlış-temsil senaryosu artık "sessiz yanlış skor" değil, "farklı-ama-doğru skor + görünür kimlik" üretir. |
| `PageHit.score_scale` alanı | S1 | **Şimdilik reddedildi.** Kabul edilen tasarım kuralı: aynı anda TEK ölçek yaşar (≈[−1,1]); hit başına etiket gereksiz. Kimlik istek düzeyinde taşınır (`detail.retrieval.quantization`). Çok-ölçekli bir gelecek olursa yeniden açılır. |
| Tarihi telemetri satırları | S10 kısmen | **Kabul edildi (bilinçli).** Geçiş öncesi events/prom örnekleri binary ölçekte kalır; `index_revision` kolonu ve yeni `quantization` etiketi ayırt eder. Migrasyon yok; katalog notu düşülür. |
| Stage adına temsil gömme | S24 | **Şimdilik kabul.** Tek temsil canlı; seri karışımı deploy zaman çizgisi + bg_app_info ile çözülür. Kalıcı çözüm P1 vokabüler birleştirmesiyle. |
| Fikstür `"sign-1bit"` literalleri | C31, S28 | **Kabul.** Bunlar sözleşme-şekli testleri; üretim kimliği healthz testi + canlı doğrulama kilitliyor. Fikstürü Settings'e bağlamak fikstürün amacını bulanıklaştırırdı. |
| UI ondalığı maxV'den türetme | S14 önerisi | **Basitleştirildi:** tek ölçek kabulüyle sabit 2 ondalık yeterli; dinamik türetme gereksiz karmaşıklık. |

### F. Verified-safe — denetimlerin teyit ettiği sağlam eksenler

- Üç skorlayıcının per-query-token ortalama formülü tutarlı (`/max(1,n_q)`); kırık olan yalnız aralıktı.
- `QUERY_FORMATS`/`DOC_PROMPTS` tek sözlük; CLI + serve + fixture aynı kaynaktan (drift kilitli).
- `check_compatibility` model/revision/format/doc_prompt/mask_policy/checksum dallarının tamamı test kapsamında (tek delik quantization idi → loader dispatch ile kapandı).
- `/pages` allowlist'i traversal'a kapalı; `.gitignore` veri/anahtar sızdırmıyor; `binarize_pack`/`PackedIndex.build` guard'ları gürültülü.
- `index_revision` SQLite tarafında kimliği zaten taşıyordu (Prometheus tarafı bu geçişte eklendi).
- `bench oracle` çapraz kontrolleri (korpus sırası dahil) üç kol için tam.

---

## 3. Sınıf analizi: tuzağın anatomisi ve benimsenen kural

139 bulgunun büyük çoğunluğu tek desenin örnekleri: **bir değer (skor, eşik, kova, süre, sayı, metin) üretildiği bağlamın kimliğinden koparılıp çıplak olarak taşınıyor; tüketici kimliği "hatırlıyor" (hardcode ediyor).** Kimlik değişince değer sessizce yalan söylüyor — sistem çökmediği için kimse fark etmiyor (%100 abstain'de bile `/healthz` "ok").

Benimsenen kural — **"kimlik veriyle yolculuk eder"** — yeni bir eksen (temsil, format, model, render ayarı…) eklerken kontrol listesi:

1. **Manifest'e yaz:** eksenin kimliği artefaktın manifest'inde alan olur; tüketici dispatch'i oradan yapar (elle seçim yok).
2. **Fail-fast korkuluğu:** kimliksiz/uyumsuz değer reddedilir — "muhtemelen eski ölçek" sezgisi koda çevrilir (`threshold > 1.5` örneği).
3. **Her dışa açılan yüzey kimliği taşır:** /healthz, Prometheus etiketi, bench raporu, cırcır dosyası, events kolonu.
4. **Testler kimliği kaynaktan okur:** beklenti dosyaları hangi kimlikte ölçüldüğünü makine-doğrulanır biçimde söyler (literal değil).
5. **Dokümanlar sayıyı künyeyle verir:** hangi temsil, hangi indeks, hangi tarih, hangi artefakt — künyesiz sayı yayımlanmaz.

Bu denetimin kendisi de kuralın 5. maddesine tabidir: buradaki her sayı `data/bench/results/int8-threshold-transfer.json` ve üç arşiv raporuna işaret eder.

---

## 4. Sınırlar (dürüstlük notları)

- Denetimler dd4f251 anlık görüntüsü üzerinde statik incelemeydi; satır numaraları geçiş commit'iyle kaymıştır. C denetimi 3 bulguyu ampirik doğruladı; geri kalanların her biri tek tek yeniden üretilmedi — A grubundakiler geçişin test/canlı doğrulamasından geçer, B/C/D grubundakiler uygulanırken doğrulanmalıdır.
- İki ajanın örtüşen bulguları (ör. eşik, kovalar, hub) bağımsız keşif olarak değerlidir ama üç ajan da aynı koddan beslendi; örtüşme "kesin doğru" garantisi değildir.
- Uzak HF repo'sunun içeriği (C28) yerelden doğrulanamadı; P1 hub işinde ilk adım budur.
