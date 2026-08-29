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
