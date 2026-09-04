# P1 hibrit getirim — düzeltme turu sonrası kapsamlı yeniden inceleme

- **İncelenen commit:** `de2cc04` (fix round), önceki: `ded732b` (orijinal inceleme konusu)
- **Yeniden inceleme tarihi:** 2026-08-30
- **Girdi:** `p1-hybrid-review.md` (0 Critical / 2 High / 4 Medium / 10 Low) +
  `p1-hybrid-report.md`'nin "Fix round 1" bölümü + `review-fix-de2cc04.diff` (1851 satır, 18 dosya)
- **Verdict: TÜMÜ ÇÖZÜLDÜ (ALL RESOLVED).** 16 bulgu + pencere-50 addendum'un
  tamamı bağımsız olarak doğrulandı. Yeni bulgu yok.

---

## 0. Kendi koştuğum doğrulamalar

| Koşum | Sonuç |
|---|---|
| `uv run pytest -q -m "not slow"` | **296 passed, 5 deselected** (beklenen) |
| `make lint` (ruff check + format --check + pyright) | **All checks passed! / 100 files already formatted / 0 errors, 0 warnings** |
| `python3 -m json.tool observability/grafana/provisioning/dashboards/belge-gozu.json` | **VALID JSON** |
| Grafana panel grid taraması | id'ler tekil (1,2,3,4,5,10,6,7,8,9); id5 (x0,y16) ve id10 (x12,y32) mevcut panellerle ÇAKIŞMIYOR |
| `grep -rn "pipelines_on_scale\|BM25_SCALE_PIPELINES" src/` | `config.py:31` tanım, `prom.py:14,40,166` TEK kaynaktan tüketim — ikinci sabit yok |
| `GET /healthz` (canlı 7860) | `{"pipeline":"hybrid","threshold":10.6,"pages":4222,...}` |
| `POST /search` (query alanıyla) sonrası `GET /metrics` | `bg_retrieval_top_score_bm25_count 2.0` (>0); `bg_retrieval_top_score_bucket` (görsel seri) satırı YOK — hibrit varsayılanda beklenen ayrım canlıda doğrulandı |
| `grep -n "_rank(" src tests` / `grep -rn "build_text_channel"` | Sıfır kalıntı referans (rename/move temiz) |
| `grep -n "command(\"pull\"" src/belge_gozu/cli.py` | `index pull` gerçek, önceden var olan komut; `pull_index()` fonksiyonu `serve --pull` ile AYNI (satır 438-441 vs 774-776) |
| `data/bench/results/20260829-2115-3a031ca-hybrid.json` içinden bağımsız yeniden hesap | aşağıda §2 Addendum |

---

## 1. Bulgu bazlı tablo (16 + addendum)

| # | Bulgu (özet) | Durum | Kanıt (tek satır) |
|---|---|---|---|
| H1 | Grafana "Top skor" paneli varsayılan (hibrit) pipeline'da kalıcı boş | **RESOLVED** | id5 artık `bg_retrieval_top_score_bm25_bucket` sorguyor ve orijinal slotta (x0,y16) kaldı; eski görsel sorgu id10'a (x12,y32, boş slot) taşındı; canlı `/metrics`'te bm25 serisi dolu, görsel seri boş — grid çakışması yok |
| H2 | README quickstart yazıldığı sırayla çalışmıyor | **RESOLVED** | Sıra: `index pull` → `corpus download` → `index build-text` → `serve` (bloklayıcı komut EN SONDA); `index pull` önceden var olan komut, `serve --pull` ile birebir aynı `pull_index()`'i çağırıyor; "no local corpus needed" yanlış yorumu kaldırıldı, "neden zorunlu" paragrafı eklendi |
| M1 | `BM25_SCALE_PIPELINES` `PIPELINE_SCORE_SCALE`'den türemiyor (ikinci hakikat kaynağı) | **RESOLVED** | `config.pipelines_on_scale(BM25_SCALE)` ile tek satır türetim; `test_bm25_routing_set_is_derived_from_the_single_scale_map` kilitliyor; grep tek kaynak doğruladı |
| M2 | README 0.814 / bench raporu 0.8023 tutarsızlığı açıklanmıyor | **RESOLVED** | README'ye "Which Recall@5?" kutusu: ikili 36/43=0.8372 vs kesirli 0.8256, aynı koşumdan; JSON'dan bağımsız yeniden hesapladım — ikili 36/43=0.83721, `overall.recall_at["5"]=0.82558` — ikisi de birebir tutuyor |
| M3 | `index build-text` kısmi-boş (yarım korpus) artefaktı sessizce kabul ediyor | **RESOLVED** | Doküman bazlı PDF-varlık kontrolü artık listeyle REDDEDİYOR (`--allow-missing` kaçış yolu var); her koşumda doküman başına boş-sayfa kırılımı basılıyor; 3 yeni test (`refuses_partial_corpus`, `allow_missing_escape_hatch`, `reports_no_empty_docs_on_healthy_corpus`) geçiyor |
| M4 | UI aşama etiketi hâlâ "exhaustive MaxSim → ilk 5" diyor | **RESOLVED** | Statik yedek + `/healthz`'den sonra `h.pipeline`'a göre `SCAN_LABEL` sözlüğü (hybrid/exhaustive/two-stage) doğru metni yazıyor |
| L1 | Eşik bandı/medyan kanalın top-1'inden alıntılanmış (servis edileninden değil) | **RESOLVED** | README + `config.py` + metrics-catalog.md üçü de medyanı **24.02** (servis edilen) olarak düzeltti, kanal medyanı 26.05'i ayrı not etti; bağımsız hesapla `route_fuse` skorlarının medyanı = 24.0215 — birebir tutuyor |
| L2 | UI: eşik kararı `hits[0]`'da, alt satır renkleri kafa karıştırabilir | **RESOLVED** | `sec-sub` metni + 1. satır tooltip'i ("eşik bu satıra uygulanır") eklendi; liste-monoton-değil uyarısı yazıldı |
| L3 | Bir app testi totolojik (aynı listeden iki okuma karşılaştırılıyor) | **RESOLVED** | Test artık fikstürün parquet'inden BAĞIMSIZ bir `BM25Index` kurup beklenen skoru hesaplıyor + `> 1.5` ölçek denetimi ekliyor |
| L4 | Korkuluk sınır değerleri (1.5, 200, 0) test edilmiyor | **RESOLVED** | 9 durumluk parametrik sınır testi (`0, 0.0001, 1.5, 1.5001, 200, 200.0001` hibrit + `1.5/1.5001` görsel + `1.5001` two-stage); negatif-eşik testi `two-stage`'i de kapsıyor artık |
| L5 | Pencere-küme değişmezliği örnek testiyle kilitli, property testiyle değil | **RESOLVED** | 300 rastgele (sıralama, yönlendirilen küme, pencere) üçlüsüyle property testi; pencere ∈ {0,1,2,5,20,WINDOW,n,n+10} |
| L6 | Metin artefaktı VARLIK kontrolü VLM+476MB indeks yüklendikten SONRA koşuyor | **RESOLVED** | `app/main.py` kaynağını okudum: `require_text_artifact()` çağrısı satır 208-209'da, encoder oluşturmadan (212+) ve `build_retriever`/indeks yüklemeden (226) ÖNCE çalışıyor |
| L7 | Bench adapter'ı sıra kompozisyonunu yeniden kuruyor + görsel gecikme encode'u içeriyor | **RESOLVED** | Adapter artık `HybridRetriever.rank()` (public yapıldı) çağırıyor, elle `route_window` kurmuyor; `t0` artık `encode_query`'den SONRA başlıyor (encode hariç) |
| L8 | `bench`/CLI artık `app.main`'e (FastAPI modülüne) bağımlı | **RESOLVED** | `build_text_channel` `require_text_artifact`+`load_text_channel` olarak `retrieval/hybrid.py`'ye taşındı; `grep -rn build_text_channel` sıfır kalıntı, `cli.py` artık `app.main` import ETMİYOR |
| L9 | `retrieval_regression_expectations.json`'da `pipeline` anahtarı hiç okunmuyor (ölü) | **RESOLVED** | `test_long_query_rank_ratchet` artık `block["pipeline"] == s.retrieval_pipeline` assert ediyor |
| L10 | `detail` kolonunun katalog satırı yeni anahtarları (retrieval, stages) saymıyor | **RESOLVED** | metrics-catalog.md `detail` satırı `hits`/`threshold`/`stages`/`retrieval` (+ hibritte `bm25_top1`/`visual_top1`/`routed_docs`) dahil yeniden yazıldı |
| **ADDENDUM** | Yönlendirme penceresi 20→50 (exp8, R@5 0.8372, R@20 0.9302) | **RESOLVED** | `WINDOW=50` TEK kaynak (`retrieval/text.py`); `HybridRetriever.window` parametresi ve `route_window` varsayılanı oradan miras alıyor (grep doğrulandı); testler `WINDOW==50`'yi ve sembolik `WINDOW` kullanımını kilitliyor; README 36/43=0.8372 (ikili) + 0.8256 (kesirli) tanım notuyla; eşik bandı `(10.528, 10.712]` **SERVİS EDİLEN** (`route_fuse`) skorlardan olduğu açıkça yazılı VE `data/bench/results/20260829-2115-3a031ca-hybrid.json`'dan bağımsız yeniden hesapla min=10.5284, 2.=10.7117, medyan=24.0215, maks=69.2982, ikili R@5=36/43=0.83721, kesirli R@5=0.82558, R@20 (ikili=kesirli)=0.93023, MRR=0.65496 — hepsi birebir tutuyor |

**Tally: 17/17 RESOLVED, 0 PARTIAL, 0 NOT RESOLVED.**

---

## 2. Addendum'un bağımsız yeniden hesabı (rapor metnine güvenmeden)

`data/bench/results/20260829-2115-3a031ca-hybrid.json`'un `diagnostics[].stages[stage="route_fuse"]`
alanlarından kendi yazdığım script ile:

```
n = 43
served top-1: min=10.528377532958984  second=10.711709022521973  median=24.02154541015625  max=69.2982177734375
ikili R@5  = 36/43 = 0.8372093023255814
ikili R@20 = 40/43 = 0.9302325581395349
```

`overall` bloğundan (üretim harness'ının kesirli tanımı):
`recall_at["5"]=0.8255813953488372`, `recall_at["20"]=0.9302325581395349`, `mrr=0.6549631887609911`.

Hepsi README/config.py/rapordaki sayılarla (0.8372, 0.8256, 0.9302, 0.655, min 10.53/medyan
24.02/maks 69.30) birebir örtüşüyor — rapor metnine güvenmeden, ham veriden doğrulandı.

## 3. Diff'in geri kalanının taranması — yeni bulgu / kapsam dışı değişiklik

18 dosyanın tamamı hunk-hunk incelendi. 16 bulgu + addendum dışında davranış
değiştiren hiçbir satır yok:

- `retrieval/hybrid.py`'de `_rank` → `rank` (public) rename'i L7'nin ZORUNLU bir
  yan etkisi (bench adapter'ının çağırabilmesi için) — kapsam dışı değil.
- README'deki "iki negatif sonuç" → "üç negatif sonuç" genişlemesi (pencere-içi
  RRF'nin de reddedildiği eklenmiş) addendum'un ölçüm provenance'ının bir
  parçası (`research/journal.md` #9/#10), 16 bulguya dahil değil ama regresyon
  da değil — projenin "ölçüm dürüstlüğü" duruşuyla tutarlı ek dokümantasyon.
- `research/*.py`'ye bu diff'te DOKUNULMADI (round-1 raporunun bıraktığı
  `make lint` kırmızısı, bu diff'ten ÖNCEKİ bir commit'te — `69e2dd0` — ayrıca
  giderilmiş; bu turun kapsamına girmiyor, sadece canlı doğrulamada teyit edildi).
- Test zayıflatma YOK: incelenen her test değişikliği (L3, L4, L5, L9) iddiaları
  SIKILAŞTIRIYOR, gevşetmiyor.

**Yeni bulgu: 0.**

Küçük bir gözlem (yeni bulgu DEĞİL, davranışı değiştirmiyor): M3'ün doküman
kontrolü yalnız PDF dosyasının VARLIĞINA bakıyor; teorik olarak "var ama
bozuk/0 bayt" bir PDF hard-fail tetiklemez. Ancak bu durum da her koşumda
basılan doküman-başına boş-sayfa kırılımında görünür kalır (`--allow-missing`
olmadan bile) — M3'ün asıl amacı olan "sessiz bozulmayı görünür kılma"
gereksinimini karşılıyor, bu yüzden bulgu olarak sayılmadı.

## 4. Sonuç

`de2cc04`, `p1-hybrid-review.md`'nin 16 bulgusunu ve koordinatörün pencere
20→50 addendum'unu eksiksiz ve doğru şekilde çözdü. Testler (296 passed),
lint (temiz), canlı servis (`pipeline=hybrid`, `threshold=10.6`, bm25 serisi
akıyor) ve bench artefaktının ham verisi tüm sayısal iddiaları destekliyor.
Birleştirmeye engel yok.
