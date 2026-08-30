# Yazım-değişmezlik + profesyonel vitrin sprinti: gerçek-kullanıcı kalitesi

- **Tarih:** 2026-08-30 · **Commit zinciri:** `6d5b345` (sprint) → `71e5860` (inceleme düzeltmeleri)
- **Tetik:** "gerçek bir kullanıcının hayatında kullanabileceği kalite" talimatı → edge-case
  sondajı (`2026-08-30-edge-case-probe.md`) + autoresearch round 3 (`research/journal.md` #11-#13)
- **SDD:** implementer (opus, frontend-design skill) → review (opus, **APPROVE**: 0 Critical/High,
  2 Medium, 5 Low, 3 Nit) → fix round (sonnet, 10/10 + 1 gerekçeli red) → re-review (sonnet, **ALL RESOLVED**; tek yeni bulgu bugün-erişilemez bir NIT — kabul edilen artık; bench artefaktı kendi teşhis verisinden bağımsız yeniden türetilip doğrulandı)

## 1. Ana kazanım: yazım-değişmez retrieval

Sondajın kritik bulgusu: aksansız yazan gerçek Türkçe kullanıcıda ("Is Kanunu'na gore
yillik ucretli izin suresi") R@5 **0.8372 → 0.5814** çöküyordu (exp11 ölçümü). Round 3'ün
KEPT deneyi **exp12 — aksan katlama** (çğıöşü+âîû→cgiosu+aiu, iki tarafta, stoplist
katlanmış uzayda, F5 sonra): sistem artık **iki yazım koşulunda da birebir aynı**:

| Koşul | Önce (exp8) | Sonra (exp12, üretim) |
|---|---|---|
| Aksanlı canary R@5 | 0.8372 | **0.8605 (37/43)** |
| AKSANSIZ canary R@5 | 0.5814 | **0.8605 (37/43)** |

R@20 0.9302, visual 8/8, chip'ler 2/2 aynen. Bedel: aksanlı MRR 0.655→0.632 (katlama
çakışmaları; düşenler sunulan top-5 İÇİNDE kalır) — program R26 istisnasıyla kabul;
alternatif exp13 (çift-biçim) İKİ guardrail'i düşürdüğü için reddedildi. Eşik yeniden
taşındı: fold sonrası sunulan-top1 bandı **(10.5265, 10.7115]** → **10.6 geçerli**
(42/43 + 4/5 birebir). Kapsam dürüstlüğü: değişmezlik aksan katlamasıyla sınırlı —
kesme işareti düşürme ("Kanunu'na"→"Kanununa") tokenizasyonu değiştirir (docstring +
slow testte belgeli). Üretim portu incelemede karakter-birebir doğrulandı; yeni slow
test yazım-değişmezliği üretim indeksinde kilitler. Bench: **binary R@5 0.8605 /
fractional 0.8488** (`data/bench/results/20260830-1611-6d5b345-hybrid.json`).

## 2. API sertleştirme (sondaj kusurlarının kapanışı)

- `k` doğrulaması `Field(ge=1, le=50)` (önce: k=-1→4221 sonuç, k=1e5→tüm korpus);
  `query/question` ≤500 karakter (önce: 3000 kr → BM25 1053, eşik anlamsız).
- Boş/yalnız-stopword sorgu → 422 `"sorgu boş ya da yalnız işlev kelimeleri içeriyor"`
  (önce: skor-0 ile rastgele 5 sayfa). Geçit ürün-düzeyi kuraldır, pipeline'dan
  bağımsız (R27).
- `/ask` yanıtına `status` alanı (answered/abstained/degraded) — UI artık ABSTAIN_TEXT
  string karşılaştırması yapmıyor (D1/S33 sınıfının kalıcı kapanışı); dürüst-ıska
  "answered" sayılır.
- Hit başına `visual_score` (hibritte; ≈[−1,1]) — kanal şeffaflığı + P2 verisi.
- Hız limiti (varsayılan KAPALI; Docker'da /ask 10/dk, /search 60/dk): kayan pencere,
  tahliye + 10k IP tavanı (M1), doğrulama-SONRASI sayım (L2), 429 + Retry-After +
  `bg_rate_limited_total` (L3); XFF bilinçli güvenilmez (yorum+test).
- Savunmacı encode semaforu (4) — ölçüm 40@c=8 sağlıklı; semafor sigorta. Kuyruk
  beklemesi encode süresine karışmaz (M2: semafor timer'dan önce).
- `query_format_id` enum tipi + CLI `Settings()` ValidationError zarfı (C8/C9 kapandı):
  bozuk env'de `--help` artık ham traceback değil.
- Docker: `BG_LOG_QUERY_TEXT=false` (kamu gizlilik varsayılanı).

## 3. Vitrin (frontend-design skill'iyle)

- **Durum-güdümlü arayüz:** mühür yalnız `status=abstained`; `degraded` için ayrı bant
  ("yanıt servisi geçici kapalı — arama sonuçları geçerli"; canlıda geçici bir Gemini
  503 bu yolu kendiliğinden doğruladı); 422/ağ hataları dostane Türkçe mesaj.
- **6 chip vitrini** (canary'den canlı-doğrulamalı, kategori etiketli): c001 uzun-TMK
  (rank 2) · İş K izin (2) · c302 tablo-layout (1) · c307 tarihî-tarama (2) ·
  c110 paraphrase (3) · aksansız varyant (2) — sistemin genişliği tek bakışta.
- Kanal şeffaflığı (BM25 + görsel mini gösterge), "nasıl çalışır" hibrit anlatımı
  künyeli sayılarla, meta/OG, odak halkaları + aria, 375px mobil doğrulaması,
  pipeline şeridi gerçek aşamalarla.

## 4. Kalan bilinen sınırlar

Yazım hatası (harf düşmesi) kısmi hasar (karakter n-gram gelecek işi) · çok-niyetli
tek soru karışık cevap (UI yönlendirmesi düşünülebilir) · paraphrase dilimi 2-3/7
(dense kanal T7) · `_GENERIC` aksanlı kaldı (research ile birebir; katlanmış-_GENERIC
gelecek deney adayı) · kesme-işareti normalizasyonu ölçülmemiş aday.
