# Belge-Gözü autoresearch programı — retrieval kalite döngüsü

> Skill: `.claude/skills/autoresearch/SKILL.md` (Karpathy autoresearch uyarlaması).
> Bu dosya o skill'in `program.md` muadilidir: hedef, metrik, kapsam, yasaklar.

## Hedef

Vitrin sorgularının sınıfını düzeltmek: kanun adı içeren uzun/hukuki Türkçe
sorgularda gold sayfa top-5'e girmiyor (teşhis: doküman kimliği yakalanıyor —
TMK sorgusunda kapak sayfası rank 1 — ama madde içeriği sayfası kayboluyor;
`data/bench/results/showcase-queries-diagnosis.json`). Sayısal hedef:
**canary R@5 ≥ 0.30** (taban 0.116'nın ~2.5 katı) — P1 hibrit hedefiyle uyumlu.

## Birincil metrik (TEK)

**R@5** — doğrulanmış canary'nin 43 cevaplanabilir sorusunda, gold sayfalardan
en az birinin ilk 5'e girdiği soru oranı. Harness: `research/evaluate.py`
(DONUK). Kaynak veri: `data/bench/canary_v1.jsonl` (48 satır; 43 answerable).

Guardrail'ler (karar metriği DEĞİL; kesin gerileme bir deneyi veto edebilir):
- R@1, R@20, MRR (canary aynı küme)
- `requires_visual=true` alt-kümesinde R@5 (metin kanalı görsel soruları bozmasın)
- Vaka analizleri: chip1 (TMK uzun, gold k4721:4) ve chip2 (İş K. izin, gold
  k4857:28) gold sırası — raporlanır, karara girmez (2 soruya overfit yasak).

## Taban (2026-08-29, int8 üretim indeksi)

- **R@5 = 0.2326 (10/43)** — harness sağlaması GEÇTİ: A2 oracle'ın int8 kolu 0.233
  ile birebir (10/43), chip rank'ları teşhisle birebir (664/137). Programın ilk
  taslağındaki 0.116 ESKİ 1-bit üretimin sayısıydı; int8 geçişi (b790f6c) üretim
  R@5'ini zaten ikiye katladı. Düzeltme kaydı: journal #0.
- R@1 0.093 · R@20 0.3023 · MRR 0.1487 · visual_R@5 0.375 (n_visual=8)
- chip1 gold rank 664 · chip2 gold rank 137 · kısa sorgu referansı rank 4
- Görsel kanalın tek başına tavanı budur (oracle float=int8) — 0.30 hedefi ancak
  EK kanalla (metin) aşılabilir; bu P1 tasarımının da öngörüsü. Esnek üst hedef: 0.40.

## Kapsam — tek değiştirilebilir dosya

`research/retrieve.py` — imza: `rank_pages(q: QueryContext) -> list[str]`
(sıralı page_id listesi). İçine yardımcı fonksiyon/sınıf yazılabilir; başka
HİÇBİR dosyaya dokunulmaz. `prepare.py`, `evaluate.py`, `program.md` ve
`src/belge_gozu/**` DONUKTUR.

`QueryContext` (evaluate.py sağlar): `query_text`, `page_ids` (4222, sabit sıra),
`visual_scores` (np.float32[4222] — üretim int8 MaxSim, önceden hesaplanmış),
`page_texts` (list[str], page_ids hizalı; RG taramalarında boş).

## Yasaklar (proje ilkeleriyle hizalı)

1. İndeks yeniden inşası YOK; görsel skorlar önbellekten (deneyler görsel
   kanalın İÇİNİ değiştiremez, yalnız üstüne kanal ekleyip birleştirebilir).
2. Orijinal sorgu korunur: sorguyu yeniden yazan tek-kanal çözüm yasak;
   türetilmiş temsiller (tokenizasyon, n-gram) serbest.
3. Füzyonda önce RRF (ilke 24); öğrenilmiş ağırlık/reranker bu döngünün dışı.
4. Eşik/abstain'e dokunulmaz (P2); LLM çağrısı yok (retrieval-only döngü).
5. Canary soruları/gold'ları değiştirilemez (ölçüm aracı deney nesnesi olamaz).
6. Deney başına bütçe: tek `evaluate.py` koşusu (saniyeler). Model yeniden
   yükleme gerektiren fikirler (yeni sorgu formatı vb.) bu döngünün DIŞI — P1'e not düşülür.

## Karar kuralı

KEPT ⇔ R@5 kesin artar VE R@20 ile requires_visual-R@5 kesin gerilemez
(≥ taban − 0.001). Aksi → DISCARDED (`git checkout -- research/retrieve.py`).
Eşitlik → DISCARDED (basitlik kazanır) — TEK istisna (ikincil-kanıt kuralı,
deney #5'te eklendi): R@5 eşitken R@20 VE MRR ikisi birden kesin iyileşiyor ve
hiçbir guardrail gerilemiyorsa KEPT. Art arda 5 verimsiz deney → dur, raporla.

## Kayıt

- `research/journal.md`: hipotez → değişiklik → sayılar → karar → öğrenilen.
- `research/results.jsonl`: evaluate.py'nin makine-okur satırları (künyeli).
- KEPT deneyler: `exp(<ad>): R@5 <eski>-><yeni> KEPT — <tek cümle>` commit'i.
- Döngü çıktıları üretime OTOMATİK GEÇMEZ: kazanan reçete P1 planının F1/F2
  görevlerine ölçümüyle birlikte devredilir (SDD kapı düzeni orada işler).

## Round 2 (2026-08-29, P1 üretim entegrasyonu SONRASI) — derin dalış

Round 1 exp7'de hedefi aşarak durdu; kullanıcı talimatıyla round 2 açıldı.
Kurallar aynen geçerli; ek çerçeve:

- **Taban:** exp7 reçetesi (R@5 0.8140). retrieve.py üretim portu tamamlanana
  kadar DONDURULDU; round 2 deneyleri port bitince başlar.
- **Hedef:** kalan 8 ıskadan kural-tabanlı, ayarsız kazanımlar + reçetenin
  sağlamlık kanıtı. Sayısal hedef yok (açık uçlu derin dalış); karar kuralı ve
  art-arda-5 durması aynen geçerli.
- **Deney adayları (kanıt sırasına göre):** exp8 madde-numarası kanalı
  (sorguda "madde N" → "Madde N" başlıklı sayfalar pencere içinde öne; c214);
  exp9 başlıktan türetilmiş kısaltma alias'ı (baş harfler → "kvkk"; c206);
  exp10 ayırt-edici-tek-token yönlendirmesi (ad token'ı korpus çapında tek
  dokümana özgüyse tek eşleşmeyle yönlendir; c209 "Anayasa"); exp11 pencere içi
  BM25-eşitlik kırıcı olarak görsel skor (yalnız beraberliklerde).
- **Sağlamlık raporu (deney değil, ölçüm):** k1/b ve F5/pencere duyarlılık
  taraması (KEPT edilmez, yalnız rapora — reçete bıçak sırtında mı?);
  dilim-bazlı kırılım; 43 soruda bootstrap %95 GA (taban vs final);
  aşırı-uyum uyarısı (aynı 43 soruya yinelenen iterasyon) raporda açık.

## Round 3 (2026-08-30) — aksansız-Türkçe gerçek-kullanıcı koşulu

Edge-case sondajı gerçek-hayat kritik vakayı doğruladı: aksansız yazılan sorgular
("Is Kanunu'na gore yillik ucretli izin suresi") gold'u tamamen kaybediyor.
Çerçeve eki:

- **İkinci koşul:** her deney iki koşulda ölçülür — (1) standart canary (aksanlı;
  resmî evaluate.py) ve (2) AKSANSIZ türev (sorgular ASCII'ye katlanmış; analiz
  script'i — harness donuk kalır, sorular/gold'lar değişmez, yalnız yazım katlanır).
- **Round-3 karar kuralı (şeffaf ek):** KEPT ⇔ aksanlı R@5 GERİLEMEZ VE aksansız
  R@5 kesin artar; guardrail'ler (R@20, visual-R@5, aksanlı MRR) gerilemez.
  **R26 istisnası (exp12 kararında eklendi, gerekçeli):** aksanlı MRR'de ≤0.025
  gerileme, (a) birincil metrik İKİ koşulda da iyileşiyor, (b) R@20 ve visual-R@5
  aynen, (c) düşen sıralar sunulan top-5 içinde kalıyorsa kabul edilir — bedel,
  yazım-değişmezlik ürün özelliğinin karşılığıdır; alternatif (exp13 çift-biçim)
  denenmiş ve iki guardrail'i düşürdüğü için reddedilmiştir.
- Aday: tokenizasyonda aksan katlama (ç→c, ğ→g, ı/i→i, ö→o, ş→s, ü→u, â/î/û→a/i/u)
  iki tarafta (indeks+sorgu), stopword listesi de katlanarak eşlenir. Ayarsız,
  deterministik, Türkçe IR'de standart uygulama.
