# P2 başlangıcı: ölçüm tesisatı, 330 soruluk cevaplanamaz seti ve kalibratör v1 — "verifier yük taşıyıcıdır" kanıtı

- **Tarih:** 2026-08-30 (tam otonom oturum devamı)
- **Talimat:** "geniş inceleme + P2 adımına başlayalım"
- **Commit zinciri:** `ece4620` (T7) → `3851c2c→d89cee7` (Faz-0) → `5d3bd83…c6b68c3` (veri katmanı, 6 commit) → `437ba6e→af6cc46` (T5+T6)
- **Resmî beyan (R29):** P2, kullanıcı talimatıyla başladı; P1 kapsamı R23 ile daraltıldığı için resmî G1 PASS koşumu YOKTUR — nihai p2-gate.md bunu açıkça taşıyacak.
- **Kanıt zinciri:** `docs/research/evidence/agent-reports/2026-08-30-p2-*.md` + `2026-08-30-e2e-review-2.md` (12 rapor, ~4.3k satır); künyeli kayıt `data/bench/results/p2-calibration-dev-v1.json` (+`-pin300` provenance)

## 1. Geniş inceleme (iki paralel denetim + sinyal ölçümü)

**Plan-gerçeklik denetimi:** P2 planının 12 görevinden 6'sı hiç yazılmamış P1 API'lerini
(EvidencePack/QueryFacets/rerank/bench_v2) hedefliyordu (STALE); 2 kısmen hazırdı
(G2.7 auto-citation zaten P0'da kalkmıştı), 2 veriye-bloke, 2 geçerliydi. Kritik yakalama
**B14**: Gemini çağrısında [Sk]↔görüntü bağı pozisyonel şanstı — G2.2 ölçülmeden düzeltilmesi
şarttı (Faz-0'da interleave ile kapandı; canlıda atıf top-1 yerine gerçek m.19 sayfasını
gösterdi). Test split'i BOŞTU; T6 sürüm anahtarı pipeline/reçete kimliği istiyordu.

**Taze E2E (44 bulgu):** üretim kritikleri (Gemini timeout yok; BM25 qtf şişmesi "ihbar"×80→667.5;
çift-gönderim; events %96 pipeline-NULL + üç uyumsuz skor ölçeği) + "P2 veri iştahı" listesi.
Eski 34 bulgudan 12 kapalı / 4 kısmi / 17 açık çıktı.

**Abstain-sinyal ölçümü (kontrolcü, 43v5):** en güçlü ayrıştırıcılar bedava metin-yanı
özelliklerdi (matched_terms AUC .937); görsel skor TERS (.34); tek özellik G2.1'i veremez →
iki-kapı mimarisi veriyle desteklendi. (Bu tabanın kendisi de sonra düzeltildi — bkz. §4.)

## 2. Faz-0: güvenlik + ölçüm tesisatı (3851c2c → d89cee7; review→fix→re-review ALL RESOLVED)

Gemini timeout/tek-retry/hata taksonomisi (bütçe aşımına deneme binmez — dürüst kapsam:
sert duvar iptal ister, belgeli); BM25 **qtf tavanı=2** (R30 — retrieval_eval bayt-birebir 37/43,
saldırı 667.5→16.69 = tam 2.00×); olay hijyeni (`pipeline` her satırda + `score_scale` +
`honest_miss` yalnız answered'da 0/1 [R32] + `error_type` + 422/429 `rejected` satırları +
`bg_rejected_total`); dürüst-ıska birinci-sınıf (tek-kaynak marker f-string'le prompt'a —
S35/D3 borcu kapandı; kalıntı: model marker'a uymayabilir, açık borç); UI çift-gönderim
kilidi + bilinmeyen-durum kartı; [Sk]↔görüntü interleave. /stats ortalamaları rejected-dışı.
443→ testler. Canlı kota gözlemi: bugün 2× http_429 — kota baskısı gerçek.

## 3. Veri katmanı: abstention_eval_v1 (330 satır) + hukuk-gruplu split — G2.1 ölçülebilirliği

Taslakçı≠denetçi rejimi, üç tur:
1. **Taslak (5d3bd83):** 200 korpus-dışı (MEKANİK etiket: çapa kanunu 56-belgelik manifestte
   yok; `script:validate_abstention_eval` künyesi) + 60 anlamsız + 40 eksik-kanıt (grep-yokluk kanıt
   notlu). İnşa korkulukları 2 gerçek hata yakaladı. Split: sha256-tabanlı, hukuk-gruplu
   (22/56 test; retrieval_eval 26/17 bölünümü birebir hedefte).
2. **Çapraz-kontrol (ad6e80d):** anlamsız 60/60 uygun; eksik-kanıt 9/40 RED (%22.5 — sınıf
   adına yakışır); korpus-dışı 40-örneklem 5 RED → **mekanik etiket gürültüsü %12.5
   Wilson [%5.5, %26.1]**. İki aritmetik sonuç: test una 151→144 < 149 asgari; ve
   doğrulanmamış test satırlarındaki gürültüyle mükemmel sistem bile G2.1'de ~%8 ölçülür →
   **R33: test yakası TAM doğrulanmadan kapı ölçülemez; dev gürültüsü belgelenerek tolere.**
3. **Yedek parti (6f26d76) + checker-2 (c6b68c3):** ret dersleri gömülü 30 yeni satır
   (6 çapa taslakta negatif-kanıtla elendi — u284 tarife tuzağının birebir tekrarı dahil);
   checker-2 test yakasındaki 112 korpus-dışının TAMAMINI doğruladı: 105 uygun / 7 red
   (%6.25, örneklem CI'ı içinde). Yapısal ders: retlerin 6/7'si "İKİZ HÜKÜM" tipi —
   başka bir korpus kanunu aynı özsel cevabı taşıyor (GVK m.41 ↔ KVK m.13 ayna metni).

**Sonuç: test 155 cevaplanamaz + 17 cevaplanabilir; 0 hata @ n=155 → CP %95 üst = %1.914 < %2.0
— G2.1 aritmetik olarak ölçülebilir.** (9 sınır-vakası bilinçli tutuldu; sertlik kaynağı.)

## 4. Kalibratör v1 (T5+T6, 437ba6e→af6cc46) ve YÜKSELTİLEN BULGU

Metin-yanı 5 özellik (served_top1, bm25_margin, matched_terms, matched_frac, routed —
q_len kurgu-artefaktı ve TÜM görsel özellikler ölçüm gerekçesiyle DIŞLANDI), saf-numpy
lojistik, sürüm-anahtarlı artefakt (`index_revision__pipeline__recipe_fp`; parmakizi
davranış-taşıyan TÜM sabitler üzerinde; anahtar/özellik-sırası fail-fast), test-split
CLI kilidi (`--yes-final-gate`), kendi-kendinden yeniden-hesaplanabilir künyeli kayıt
(per-question satırlar dahil). Etiket: `safe_to_answer = answerable ∧ gold∈top-5` (LLM'siz).
Review bağımsız yeniden-hesapla doğruladı (AUROC 16 hane, fit byte-identical).

**Dev sonuçları (n=185: 22 poz / 163 neg):** AUROC 0.782 · Brier 0.086 · ECE 0.034 ·
dev false-answer 0/159 (CP üst %1.87) · seçilen tau 0.504 → **kapsam %2.2 (4/185;
cevaplanabilirin %15'i), guarantee="none"** (n=4, CP üst %52.7 — yüksek sesli uyarıyla; R35).

**Yükseltilen bulgu (P2'nin yönünü belirler):** 5-una tabanında ölçülen özellik AUC'leri
gerçekçi 151+ una tabanında ÇÖKTÜ (.937→.677, .863→.722) — abstention_eval_v1 negatifleri sözcüksel
olarak cevaplanabilir-görünümlü sert negatifler. %5 risk bütçesinde retrieval-yanı kapsam
yalnız %2.2. **Retrieval-yanı güven tek başına işe yarar seçici cevaplama VEREMEZ; ikinci
kapı (LLM kanıt-doğrulayıcı, T1/T2) yük taşıyıcıdır** — planın iki-kapı tasarımı artık
zorunluluk olarak sayıyla kanıtlı. Eşik ayarıyla çıkış yok (ilke 21 zaten yasaklıyor).

## 5. Kısıtlar ve açık borçlar

- **Gemini kotası kritik yol:** bugün 2×429; verifier koşumları (yüzlerce çağrı) sha256
  önbellek + güne bölme + sayaçlı ön-sonda ister; ücretli katman kararı KULLANICININ.
- Dev yakasında ~%12.5 mekanik-etiket gürültüsü belgeli (kalibrasyon dayanıklılığı);
  test yakası temiz. İnsan doğrulaması hâlâ yok (model-checker rejimi; künyeler dürüst).
- Kalibratör eşdoğrusallık borcu (2 küçük negatif ağırlık); dev answerable n=26 küçük.
- honest-miss marker uyumu yumuşak (verifier gelince yapısal çözülür).

## 6. Sıradaki adımlar

1. **T1/T2 verifier + iki-kapı entegrasyonu** (kod+testler kota-sız, stub'lı; canlı smoke
   küçük bütçeyle; tam koşum runbook'la taze-kota gününe).
2. T8 servis entegrasyonu (kalibre kapı + güvenli fallback) — verifier'la birlikte.
3. G2 kapı koşumu: test split TEK SEFER, verifier sinyali + kota planı hazır olunca.
4. UI claim-citation (T9 kalanı) + T10 judge (PPI çiftleri: eksik-kanıt 31 doğrulanmış).

---

## Ek (2026-08-31): T1/T2 verifier + anahtar rotasyonu — yük taşıyıcı kapı kuruldu

- **T1/T2 (d051918 → 418b028):** Türkçe-farkında iddia bölütleme ([Sn]-bağlı; en-yakın-önceki
  marker kalıtımı), sayfa-metnine karşı LLM doğrulama, kalıcı sha256 önbellek (yalnız
  iyi-biçimli model yanıtı yazılır — zehirlenme kapalı, R37; restart-sonrası 0-çağrı kanıtlı),
  **API-denemesi birimli** sert bütçe (rotasyon çarpanı dahil, R36; canlı kanıt: tavan=3'te
  tam durma, kalan iddialar şüphede-reddet), iki kapı bayrak-KAPALI (üretim bayt-uyumlu
  kilitli). Canlı: gate-2 demote yolu gerçek cevapta çalıştı; gate-1 tüm vitrin chip'lerinde
  abstain (p 0.14-0.32 < tau 0.504) — §4'teki %2.2-kapsam bulgusunun canlı teyidi.
- **Anahtar rotasyonu (348fb63, kullanıcı direktifi):** tek istemci-fabrikası sarımı; HERHANGİ
  API hatasında (parse hariç) diğer anahtara yapışkan fallback; ≤3 deneme tek 35sn bütçede;
  ikisi de düşerse degraded + keys_tried. CANLI: key1 429 → key2 servis; yapışkanlık teyit;
  anahtar değeri hiçbir yüzeyde yok. Review: doğrudan APPROVE (API çağrısı kilit DIŞINDA —
  serileşme yok; anyio copy_context istek-izolasyonu kaynak-okumayla teyit).
- **Kota gerçeği ölçüldü:** free tier 20 çağrı/gün/model (429 gövdesinden); tipik cevap 6-7
  iddia → tam-doğrulamalı /ask ≈ 8 deneme. G2 kapı koşumu matematiği: gate-1 eleğinden
  geçenler × ~8 ≈ yüzlerce deneme → 2 anahtar × 20/gün ile ÇOK-GÜNLÜ takvim (önbellek
  birikimli — tekrar koşumlar bedava) YA DA ücretli katman (kullanıcı kararı).
- SDD: birleşik review (0 kritik; verifier bayrak-öncesi 15 bulgu) → fix (15/15; 1 itiraz
  KABUL — R38 tek-kaynak etiket sözlüğü) → re-review ALL RESOLVED (ContextVar izolasyonu +
  `is`-kimlik kanıtları). 666 test. Kanıt: evidence/agent-reports/2026-08-31-*.
