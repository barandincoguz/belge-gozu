# Edge-case sondajı: hibrit üretim sisteminin uç-girdi davranış envanteri

- **Tarih:** 2026-08-30 · canlı :7860, hibrit pipeline (de2cc04 sonrası), eşik 10.6
- **Yöntem:** kontrolcü sondajı — 20 /search vakası + 5 /ask vakası + eşzamanlılık
  (tek atım c=2/4/8 + sürdürülmüş 40 istek @ c=8). Ham log scratchpad'de; kritik
  satırlar aşağıda. Bu doküman "vitrin sprinti"nin gerekçe tabanıdır.

## /search davranışları

| Vaka | Gözlem | Hüküm |
|---|---|---|
| Boş / yalnız boşluk / yalnız stopword | skor-0 ile İLK 5 rastgele sayfa döner (k6098:1,2,3…) | **KUSUR** → 422 gerekli |
| **AKSANSIZ chip2** ("Is Kanunu'na gore yillik ucretli izin suresi") | gold TAMAMEN kayıp; top1 alakasız k492:38 @ 11.0 | **KRİTİK gerçek-hayat kusuru** → round 3 |
| AKSANSIZ kısa ("yerlesim yeri nedir") | gold k4721:4 hâlâ top1 (kısmî token şansı) | kısmi hasar |
| BÜYÜK HARF (İ/I dahil) | gold rank 2 — tr_lower doğru | ✓ |
| Yazım hatası ("yıllk … izn") | doküman düzeyi doğru (İş K), sayfa kısmi | bilinen sınır (karakter n-gram gelecek işi) |
| İngilizce soru | top1 8.9 < 10.6 → /ask'ta abstain | ✓ kabul edilebilir |
| Emoji'li soru | etkisiz, gold rank 1 | ✓ |
| HTML/SQL injection dizgileri | retrieval etkilenmez (token filtresi) | ✓ (UI render yolu ayrıca sprint kontrolünde) |
| **3000 karakterlik sorgu** | BM25 top skoru ~1053 → eşik anlamsız; uzunluk sınırı YOK | **KUSUR** → max_length 422 |
| "madde 53" / "4857" yalnız | zayıf-orta eşleşmeler, çökme yok | bilinen sınır |
| Çift soru | iki niyet karışır (aşağıda /ask) | bilinen RAG sınırı |
| **k=0** | sessizce 5 | KUSUR → doğrulama |
| **k=-1** | **4221 sonuç** | **KUSUR** → Field(ge=1) |
| **k=100000** | **tüm korpus 4222 sayfa döner** | **KUSUR** → Field(le=50) |
| k=50 | 50 sonuç, tutarlı | ✓ |

## /ask davranışları

| Vaka | Gözlem | Hüküm |
|---|---|---|
| Boş sorgu | 0.5 s, temiz abstain + mühür (top-1 0 < 10.6) | ✓ (yine de 422 daha dürüst) |
| İngilizce | 0.3 s abstain, LLM çağrılmadı | ✓ |
| Korpus-dışı (vatandaşlık) | eşik geçti (23.5) → LLM → dürüst "bulamadım", atıf yok, halüsinasyon yok | ✓ dürüstlük korunuyor |
| Çift soru | kısmî cevap + 3 atıf (biri konu-dışı k657) — niyet karışımı | bilinen sınır; UI yönlendirmesi düşünülebilir |
| AKSANSIZ chip2 | "Verilen sayfalarda bulamadım" | **KRİTİK** → round 3 |

## Eşzamanlılık

- Tek atım c=2/4/8: hepsi OK (0.5/0.6/1.2 s duvar).
- **Sürdürülmüş 40 istek @ c=8: 40/40 OK, p50 1.34 s, maks 2.4 s, sunucu ayakta.**
- Eski telemetri-dönemi SIGSEGV (c=8) bu kod yolunda TEKRARLAMADI → "kritik kilit"
  ihtiyacı düştü; savunmacı encode sınırlayıcı (Semaphore) yeterli görüldü.

## Sprint'e devredilen düzeltmeler

k/query doğrulama (422), boş-sorgu 422, aksan katlama (round 3 exp12 — iki koşulda
R@5 0.8605), status alanıyla abstain/degraded ayrımı, hız limiti (varsayılan kapalı;
Docker'da açık), savunmacı encode semaforu, UI durum/kanal/chip vitrini.
