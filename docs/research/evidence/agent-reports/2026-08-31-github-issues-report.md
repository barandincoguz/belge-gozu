# GitHub issue açılışı — rapor

- **Tarih:** 2026-08-31
- **Repo:** `barandincoguz/belge-gozu` (PRIVATE)
- **Kapsam:** yalnız `gh issue` / `gh label` komutları; repoda hiçbir dosya değiştirilmedi, commit/push yapılmadı.
- **Ana kaynak:** `docs/research/findings/2026-08-31-proje-durum-envanteri.md`
- **Destek:** `2026-08-29-config-coupling-audit.md`, `2026-08-29-e2e-review.md`, `2026-08-30-p2-baslangic.md`, `docs/superpowers/plans/2026-08-26-belge-gozu-rag-quality-master.md`

## Oluşturulan etiketler (8)

`kapı` · `p1` · `p2` · `dağıtım` · `dürüstlük` · `teknik-borç` · `bilimsel-derinlik` · `hızlı-kazanım`

(Repoda hazır bulunan 10 GitHub varsayılan etiketi kullanılmadı.)

## Açılan issue'lar (20)

### P0-şimdi (6)

| # | Başlık | Etiketler | Özet |
|---|---|---|---|
| 1 | G1 kapı raporunu yaz ve iki ölçülen FAIL'i adjudike et | `kapı`, `dürüstlük` | `p1-gate.md` yok; R@50 0,9302 < 0,95 ve paraphrase 0,5714 < 0,90 ölçüldü ama hüküm verilmedi; ASCII-katlamanın R@50'yi 0,9535→0,9302 düşürdüğü ödün adjudike edilmemiş. |
| 2 | G2 ölçüm ayağını kur: `bench/answer_eval.py` + `bench answers` harness'ı | `kapı`, `p2` | Citation precision ve false supported-answer hesaplayacak kod hiç yok → G2.1/G2.2 ölçülemez, veri (43+330) hazır bekliyor. |
| 3 | `--only-verified` etkisiz: filtre `verification_kind`'a bakmıyor | `dürüstlük`, `hızlı-kazanım` | `dataset.py` yalnız `verification_status`'a bakıyor; 48/48 verified olduğu için bayrak hiçbir satırı elemiyor → p0-gate'in "doğrulanmış sette aynı" gözlemi totoloji. |
| 4 | Vitrin ve API sunum dürüstlüğü: çip kaynağı iddiası, "hibrit" adlandırması, boş sonuç | `dürüstlük`, `hızlı-kazanım` | Y30 (`index.html:347`) + Y33/Y34/Y35/Y36 + Y5 (`/search` sıfır-skorlu listeyi geçerli sonuç gibi döndürüyor) + ilke 14/9 adlandırma riski. |
| 5 | Docker dağıtım zincirini uçtan uca doğrula: `serve --pull` sessiz no-op + boot kusurları | `dağıtım`, `teknik-borç` | `BG_HF_DATASET_REPO` hiç set edilmediği için `--pull` sessizce atlanıyor; `USER`/`HF_HOME`/`.dockerignore`/`[tool.uv]` CPU torch yok; recorder `mkdir` guard'sız. |
| 6 | HF Hub indeks yayını: token, revision pinleme, atomik pull, taze int8+metin push | `dağıtım` | Uzaktaki indeks P1 öncesi (`page_texts.parquet` yok → hibrit boot etmez); `hub.py` token/revision/`delete_patterns` taşımıyor, pull atomik değil (K21/C11/C28). |

### P1-sonra (10)

| # | Başlık | Etiketler | Özet |
|---|---|---|---|
| 7 | HF Space / barındırma kararı: sistem hiçbir yerde canlı değil (G1.7 ölçülemiyor) | `dağıtım` | Space PRO gerektirdiği için (402) hiç oluşturulmadı; seçenekler (PRO / statik SDK / başka CPU host) tartılıp karara bağlanacak — KULLANICI KARARI. |
| 8 | Bench ve telemetri teşhis etiketleri sessizce yanlış rapor üretiyor | `dürüstlük`, `teknik-borç` | K9/K10 (`candidate_survival` ≠ survival, `gold_ranks=-1` sinyali yok ediyor), K18 (`stage1_ms`/`stage2_ms` daima NULL), `EvalReport.oracle_gap` yok, ilke 5 (oracle üretim hattını ölçmüyor). |
| 9 | Kapı bayraklarının üretim politikası + T8 kalanı (ilke 20 ihlali) | `kapı`, `p2` | `gate_verifier`/`gate_calibrated` default kapalı → serviste iddia doğrulaması yok; `/healthz`'de `calibrator`/`answerer_ready`, `/ask`'ta `abstain_reason` yok; açılma koşulu yazılı değil. |
| 10 | G2 kapı raporu, test-split final koşumu ve Gemini kota planı (P2 T12) | `kapı`, `p2` | `p2-gate.md` yok, test yakası hiç kullanılmadı; kota (2×20/gün) planı + runbook + tek seferlik final koşum + risk-coverage figürü. |
| 11 | Dense metin kanalı (P1 T7): paraphrase dilimindeki sözcüksel tavanı kır | `p1`, `bilimsel-derinlik` | `retrieval/dense.py` yok; paraphrase R@50 0,5714 / R@5 0,2857 — G1.2'nin tek FAIL'i ve G1.1 açığının en olası çıkışı. Füzyon şekli yeniden ölçülecek (RRF reddedilmişti). |
| 12 | Benchmark v2 (120 answerable + 30 unanswerable): 43 soruluk taban dar | `p1`, `bilimsel-derinlik` | `bench_v2.jsonl` yok; mevcut taban hedefin ~1/3'ü, 12 dilimden 3'ü boş, %0,8 insan onaylı; law-grouped split hazır. |
| 13 | İnsan doğrulama kapısı: 378 satırın yalnız 3'ü insan onaylı (K8 + `require_human`) | `dürüstlük`, `bilimsel-derinlik` | canary 3/48 human, unans 0/330 human (113 satır hiç denetlenmemiş); K8 kuyruk kusuru + `require_human` + README rakam-kilidi testi. |
| 14 | Bağlaşım denetimi B grubu: kalan hızlı dalga düzeltmeleri | `teknik-borç`, `hızlı-kazanım` | C10, S5, C19 (autouse env fixture hiç yok), C41, C42, D21, D1 — hepsi davranış-nötr, ayrı commit. (C8/C9/C12/C38 kapandığı doğrulandı.) |
| 15 | Gizlilik ve erişim varsayılanları güvensiz | `dağıtım`, `teknik-borç` | `log_query_text=True` ve hız sınırı 0 kütüphane varsayılanı; güvenli profil yalnız Dockerfile'da; `/metrics` + `/stats` kimlik doğrulamasız ve tam tablo taramalı (Y11). |
| 16 | Korpus yapı katmanı: madde hiyerarşisi (ilke 11 ihlali) + OCR fallback (ilke 10) | `p1`, `bilimsel-derinlik` | `corpus/articles.py` yok → retrieval atomu hâlâ sayfa, `gold_article_ids` ölü alan, `CitationRef.article_id` daima None; ayrıca 5 metinsiz sayfa için OCR yok. |

### P2-sonra (4)

| # | Başlık | Etiketler | Özet |
|---|---|---|---|
| 17 | Cross-encoder reranker (P1 T10): G1.3 artık ölçülebilir, katman yok | `p1`, `bilimsel-derinlik` | Recall ön koşulu (R@20 0,9302) sağlandı → ilke 2/16 uyumlu biçimde denenebilir; kazanç bootstrap CI alt sınırı > 0 değilse default kapalı kalır. |
| 18 | P2 T9: UI claim-citation + outcome telemetrisi + geri bildirim + drift raporu | `p2` | Verifier çalışıyor ama görünmüyor: UI'da iddia-atıf bağı yok, `/feedback` yok, outcome alanları yok, `scripts/drift_report.py` yok. |
| 19 | P2 T10+T11: insan-kalibreli LLM-judge (PPI) ve fine-tuning kapısının resmî hükmü | `p2`, `bilimsel-derinlik` | `bench/judge.py` yok (ön koşul: ≥30 insan çifti); T11 kapı koşulu zaten sağlanıyor (paraphrase R@5 0,2857) ama resmî hüküm yok — dense/reranker sonrası değerlendirilmeli. |
| 20 | Ertelenmiş borç: BM25 inverted index (Y2) + UI erişilebilirlik/dayanıklılık turu | `teknik-borç` | Y2 doğrusal tarama (bugünkü ölçekte acil değil), Y37/Y38/Y40/Y43/Y45, NEW-1 guard, `unans_v1.README.md` bayat tablo (300/286/14 → 330/309/21). |

## Bilinçli olarak issue açılmayanlar

**Bu oturumda kapandığı doğrulanan (kod üzerinde teyit edildi):**

- GitHub'a çıkış — `git remote -v` → `origin https://github.com/barandincoguz/belge-gozu.git`.
- UI'daki "insan-doğrulanmış canary" yanlış iddiası (Y29) — `index.html:454-461` artık doğrulama künyesini taşıyor ("48 satırdan 3'ü insan… insan-doğrulanmış sayılmaz").
- CI lock disiplini + docker build işi (C12) — `.github/workflows/ci.yml` `uv sync --locked` + ayrı `docker` işi + `validate_unans.py` adımı.
- LLM zaman bütçesi — `answer/gemini.py` `GEMINI_TIMEOUT_S=24.0`, `GEMINI_TOTAL_BUDGET_S=50.0`.

**Envanterin spot-check'inde zaten KAPALI işaretli olanlar:** Y1 (QTF_CAP=2), Y15/K33 (Gemini timeout + lazy client yarışı), K27 (`tr_lower`), Y17/Y31 (honest-miss API alanı), Y20 (`error_type`), Y28/K17 (çift-gönderim kilidi), C8, C9, C38, S51/S52, S33/S34/D1/D2, bağlaşım denetimi §2A'nın 21 satırı.

**Gruplama kararları (tek tek açılmayanlar):** ~40 açık kalem 20 issue'ya toplandı. En büyük birleştirmeler: UI/API sunum nit'leri → #4 ve #20; bench/telemetri teşhis kalemleri + `oracle_gap` + ilke 5 → #8; bağlaşım B grubunun 7 kalemi → #14; madde hiyerarşisi + OCR (aynı korpus katmanı, iki ilke) → #16; T10 judge + T11 fine-tuning kapısı (ikisi de "neye göre yargılıyoruz" sorusu) → #19.

**Kapsam dışı bırakılanlar:** envanter §7(d9) — korpus genişletmesi (1475 belge), LocalVLM, agentic derin arama. Gerekçe: spec §10 kapsam dışı ve ilke 23 gereği P2 sonrasına ait; issue açmak yol haritasına gürültü ekler.

## Doğrulama notu

Her issue'nun **Kanıt** bölümündeki dosya:satır ve sayı referansları envanterden birebir taşındı; ayrıca şu kalemler bugün kod üzerinde bağımsız olarak yeniden doğrulandı: `index/hub.py` (token/revision/`delete_patterns` yokluğu), `bench/harness.py` `EvalReport` alan listesi, `bench/dataset.py` `load_bench` filtresi, `app/main.py:544-545` ölü aşama sütunları ve `:713` düz `{"hits": hits}`, `config.py:176/192-193/215-216`, `Dockerfile`, `.dockerignore` yokluğu, `pyproject.toml`'de `[tool.uv]` yokluğu, `retrieval/` ve `corpus/` ve `bench/` dizin içerikleri, `index.html:347`.
