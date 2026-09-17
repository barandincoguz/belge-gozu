# 2026-09-13 — Ölçüm, telemetri ve uçtan uca durum denetimi

## Kapsam ve yürütme künyesi

Bu denetim `main` dalını güvenli biçimde senkronladıktan sonra yürütüldü.
`origin/main` ve yerel `main` son doğrulamada aynı commit'tedir. Üretim kodu,
testler ve raporlar üzerinde yapılan her davranış değişikliği ayrı commit olarak
gönderildi:

- `e15d51d` — citation precision yalnız atıflı claim'leri sayar.
- `8059df5` — geçersiz calibration metric girdileri reddedilir.
- `530f301` — reranker kararındaki soru farkı diagnostics'ten sayılır.
- `18d1820` — tam gold rank ve gerçek candidate survival korunur.
- `2db6c17` — özel Prometheus registry'sine runtime kolektörleri eklenir.
- `000e163` — bootstrap CI geçersiz/sonlu olmayan girdileri reddeder.
- `9208c91` — GitHub/triage/domain agent sözleşmesi kaydedilir.
- `f5a7f9c` — bu ölçüm/observability audit ve P/G issue haritası kaydedilir.
- `3789e6e` — legacy telemetry stage alanlarının null/deprecation sözleşmesi netleştirilir.
- `b75c8d9` — verifier LLM kullanım metadata'sı, süre, toplam maliyet ve amaç-bazlı Prometheus serisi eklenir.
- `ea220b0` — telemetri yazma kaybı ve rejected trafik monitoring panelleri eklenir.
- `d0a0b63` — eşzamanlı recorder hatalarının Prometheus’ta çağrı başına sayılması sağlanır.
- `8d6d690` — `/ask` response ve UI gerçek stage sürelerini kullanır.
- `306b14d` — `/ask`, Prometheus ile aynı event kararından `abstain_reason` yayınlar.
- `e02c768` — `/healthz`, answerer/calibrator/verifier hazır oluşunu ve gate künyesini yayınlar.
- `90b3fed` — persisted answer/retrieval raporları için tamper/aggregate doğrulayıcı eklenir.
- `fc148bd` — custom reranker raporlarına per-question ranked evidence ve doğrulama desteği eklenir.

Veri ve artefakt iddiaları, koşumun kendi künye alanlarıyla birlikte okunmalıdır;
model cache'i veya kullanıcı artefaktları silinmemiştir.

## Doğrulanmış kapılar

| Alan | Kanıt | Sonuç |
|---|---|---|
| Ağsız regresyon | `uv run --extra dev pytest tests -q -m "not slow"` | **884 passed, 2 skipped, 6 deselected** |
| Lab ağsız regresyon | `make test` | **896 passed, 6 deselected** |
| Statik kalite | `uv run ruff check .`, `uv run pyright`, `git diff --check` | **yeşil** |
| Gerçek retrieval yolu | `.venv-lab/bin/pytest tests/retrieval/test_retrieval_regression.py -q -rx` | **4 passed, 1 beklenen strict xfail** |
| Beklenen xfail | BM25 10.6 eşiği cevaplanabilir/cevaplanamazı ayırmıyor | P2 kalibrasyonu bekleniyor; xfail kaldırılmadı |
| Abstention veri doğrulaması | `validate_abstention_eval.py` | 330 satır, tüm mekanik kontroller **TEMİZ** |
| Retrieval doğrulama | `verify_retrieval_eval.py --status` | v2: 62 verified; 47 human, 15 model-cross-check; 4 draft |

Gerçek retrieval regression, model ve indeks önbelleği kullanılarak çalıştı;
`4 passed / 1 xfailed` sonucu ölçülmüş bir sonuçtur. Gerçek Gemini answer smoke'u
çalıştırılmadı: yapılandırılmış Gemini anahtarı yok. Bu nedenle G2.1/G2.2 için
quota-backed sayı iddiası yapılmamıştır.

## Metrik sözleşmesi düzeltmeleri

- Citation precision: `supported_cited / all_cited`; uncited claim yalnız
  completeness paydasına girer. Sıfır atıf precision'ı `null`dır.
- Calibration helpers: NaN/sonsuz/out-of-range confidence, NaN eşik, yanlış
  method ve düşük örneklemli garanti kurulumu sessiz sayı üretmez.
- Ranking metrics: duplicate page ID veya `k <= 0` değerlendirmeyi açıkça
  durdurur; nDCG'nin 1'in üzerine çıkması engellenir.
- Reranker karar metriği: kesirli R@5 raporlanır; “+N soru” yalnız diagnostics
  içindeki binary first-5 hit farkıdır. `retrieval_eval_v2`, insan n=47:
  `base-w4096` 38 → `maxp-w4096` 40; R@5 `0.8085 → 0.8511`, R@20
  `0.9362 → 0.9574`, R@50 sabit `0.9574`, nDCG5 `0.5809 → 0.5914`.
- Runtime monitoring: custom registry process/platform/GC kolektörlerini açıkça
  kaydeder. macOS'ta `/proc` serileri yoktur; Docker/Linux'ta RSS/CPU serileri
  üretilir.
- Verifier LLM kullanımı: provider metadata'sı `detail.llm_usage` içinde amaç
  bazında tutulur; toplam token/maliyet ve `verifier_ms` ayrı kanıtlanır. Missing
  metadata bilinmeyen kalır, sıfır diye uydurulmaz.
- Telemetry loss counter: recorder artık her çağrı için başarı/başarısızlık
  sonucu döndürür; global failure counter farkı eşzamanlı istekler arasında
  paylaştırılmaz.

## Uçtan uca kalite resmi

| Yol/konfigürasyon | Veri ve kimlik | Ölçüm | Hüküm |
|---|---|---:|---|
| Eski görsel-only taban | int8, retrieval_eval_v1, n=43 | R@5 `0.2326` | görsel model tavanı |
| Üretim hibrit + late aday örme | retrieval_eval_v1/v2 | v1 R@5 `0.8488`, R@20 `0.9302`; v2 insan altkümesi için aday pool R@5 `0.6277` | BM25 + aday kapsamı, reranker yok |
| Dense aday kanalı | `20260910-1837-semantic-coverage.json`, v2 human n=47 | R@5/R@20/R@50 `0.6277/0.7660/0.8085` ile aynı | aday coverage artıyor, ilk sıra kazancı kanıtlanmadı |
| Cross-encoder page, 4096 | `20260911-base-w4096.json`, v2 human n=47 | P R@5 `0.8085`, rerank p50 ~6.1 s | taban |
| Cross-encoder MaxP chunk, 4096 | `20260911-maxp-w4096.json`, aynı bench/index/recipe | P R@5 `0.8511`, rerank p50 ~24.7 s | offline **KEPT**; üretime alınmadı |

Bu sayılar farklı benchmark altkümeleri veya farklı katmanlar arasında doğrudan
tek bir “program başarı yüzdesi” olarak ortalanamaz. Karşılaştırma yalnız aynı
benchmark, aynı index revision, aynı recipe fingerprint ve aynı seçim filtresi
üzerinde yapılmalıdır.

## P/G kapısı ve issue haritası

| Alan | Mevcut durum | Issue |
|---|---|---|
| P0/G0 | indeks, cevap kapısı ve runtime regression mevcut; G1 kapı raporu yok | [#1](https://github.com/barandincoguz/belge-gozu/issues/1) |
| G2 answer harness | kod/test mevcut; gerçek quota-backed dev/test artefaktı yok | [#2](https://github.com/barandincoguz/belge-gozu/issues/2) |
| Verification filter | `--min-verification human` gerçek çalışıyor; #3 kapandı | [#3](https://github.com/barandincoguz/belge-gozu/issues/3) |
| UI/API dürüstlüğü | per-stage UI zamanı tamamlandı; kalan a11y/güvenilirlik nit'leri açık | [#4](https://github.com/barandincoguz/belge-gozu/issues/4), [#20](https://github.com/barandincoguz/belge-gozu/issues/20) |
| Docker/Hub/hosting | uçtan uca taze pull ve canlı URL yok | [#5](https://github.com/barandincoguz/belge-gozu/issues/5), [#6](https://github.com/barandincoguz/belge-gozu/issues/6), [#7](https://github.com/barandincoguz/belge-gozu/issues/7) |
| Bench diagnostics | K9/K10 ve görsel oracle kimlik/kapsam etiketi kapandı; legacy stage SQL, `EvalReport.oracle_gap` ve hibrit oracle açık | [#8](https://github.com/barandincoguz/belge-gozu/issues/8) |
| Gate policy/API visibility | API görünürlüğü tamamlandı; yalnız sayısal production açılma politikası insan kararı olarak açık | [#9](https://github.com/barandincoguz/belge-gozu/issues/9) |
| P2 final gate | test split, G2 raporu ve quota planı yok | [#10](https://github.com/barandincoguz/belge-gozu/issues/10) |
| Dense/retrieval | dense aday coverage ölçüldü, kalite kazancı yok; production fusion yok | [#11](https://github.com/barandincoguz/belge-gozu/issues/11) |
| Benchmark gücü | v2 insan n=47; spec'in 120+30 hedefi ve 12 dolu dilim yok | [#12](https://github.com/barandincoguz/belge-gozu/issues/12) |
| İnsan doğrulaması | retrieval v2 insan 47; abstention insan 0 ve draft satırlar var | [#13](https://github.com/barandincoguz/belge-gozu/issues/13) |
| Teknik borç/güvenlik | #14'ün env/indeks guard ve fikstür işleri kapandı; BM25 karmaşıklığı #20'de, erişim ve gizlilik #15'te açık | [#14](https://github.com/barandincoguz/belge-gozu/issues/14) (kapatıldı), [#20](https://github.com/barandincoguz/belge-gozu/issues/20), [#15](https://github.com/barandincoguz/belge-gozu/issues/15) |
| Korpus yapısı | madde hiyerarşisi ve OCR fallback yok | [#16](https://github.com/barandincoguz/belge-gozu/issues/16) |
| Reranker | offline MaxP kazancı var; production config/G1.3 hükmü yok | [#17](https://github.com/barandincoguz/belge-gozu/issues/17) |
| Outcome/drift | UI claim verdict, feedback ve drift raporu yok | [#18](https://github.com/barandincoguz/belge-gozu/issues/18) |
| Judge/fine-tuning | insan PPI önkoşulu ve resmi FT kararı yok | [#19](https://github.com/barandincoguz/belge-gozu/issues/19) |
| Yeni ölçüm artefaktı doğrulama | **canonical raporlarda tamamlandı**; custom reranker raporları tam yeniden üretim için eksik | [#23](https://github.com/barandincoguz/belge-gozu/issues/23) (kapatıldı), [#25](https://github.com/barandincoguz/belge-gozu/issues/25) |
| Yeni verifier telemetry | **tamamlandı**; verifier token/süre/maliyet amaç bazında ve toplamda izleniyor | [#22](https://github.com/barandincoguz/belge-gozu/issues/22) (kapatıldı) |
| Yeni monitoring | **tamamlandı**; telemetry write loss ve rejected trafik ayrı Prometheus/dashboard popülasyonları olarak izleniyor | [#24](https://github.com/barandincoguz/belge-gozu/issues/24) (kapatıldı) |
| Reranker artefaktı | **tamamlandı**; yeni raporlar per-question üç kol ranking’i taşıyor, eski JSON’lar açıkça legacy/unverifiable işaretli | [#25](https://github.com/barandincoguz/belge-gozu/issues/25) (kapatıldı) |

## Uygulama sırası

1. **Ölçüm güvenliği:** canonical artifact verifier (#23), verifier usage (#22)
   ve telemetry loss/rejected monitoring (#24) kapandı. Yeni G2/G1 sayıları
   yine yalnız doğrulanmış veri/identity ile yayınlanmalı; custom reranker
   artefaktlarının eski dosyalara uygulanması/legacy işareti #25'te.
2. **Kanıt kapıları:** #2 quota-backed dev smoke → #1 güncel G1 gate raporu →
   #8 K18/oracle sınırı → #9 gate policy → #10 tek seferlik test final gate.
3. **Kalite katmanı:** #13 insan doğrulama → #12 bench v2 → #11 dense/fusion
   veya #17 production reranker. Her deney aynı index/recipe kimliği ve holdout
   ile ölçülmeli; mevcut MaxP sonucu üretime otomatik taşınmamalı.
4. **Çalıştırılabilir ürün:** #5 Docker smoke → #6 pinli Hub artefaktı → #7
   hosting kararı/G1.7.
5. **Ürün güveni ve derinlik:** #18 claim/outcome/drift → #16 madde/OCR → #19
   judge/fine-tuning → #14/#20 teknik borç.

Her adımda başarısız deney de commit/artefakt/journal kaydıdır; test yakası,
geliştirme sırasında eşik seçimi için kullanılmayacaktır.

## 2026-09-17 devam kaydı

- #14, `5f3b42c`, `bfc4b0f`, `58fae42`, `14279b8`, `6409413` ve
  `766dfe4` commit'leriyle kapandı. D1'in geçici `ABSTAIN_TEXT` kopya testi,
  sunucu sahipli `status`/`abstain_reason` API sözleşmesi ve UI testiyle
  geçersizleşti; UI'da karşılaştırılacak metin kopyası kalmadı.
- #8'in `e9dedb4` dilimi, `bench oracle` kollarının kaynak manifest kimliğini
  karşılaştırıyor. Aynı `page_id` listesine sahip ama farklı `corpus_checksum`
  taşıyan iki kol artık model yüklenmeden reddediliyor. Rapor ve CLI yardımı
  kapsamı `exhaustive-visual` olarak belirtiyor; bu BM25 yönlendirmeli hibrit
  üretim hattının oracle sonucu değildir.
- Doğrulama: `uv run pytest tests/bench/test_dataset.py
  tests/test_verify_retrieval_eval.py -q` → 88 passed; #8'in üç CLI testi →
  3 passed; `make test` → 902 passed, 6 deselected; `make lint`,
  `uv run pyright` ve `git diff --check` yeşil. Bu turda yeni gerçek-model
  koşumu veya kalite sayısı üretilmedi; veri kümesi ve indeks/recipe kimliği
  gerektiren önceki ölçümler yukarıdaki tarihli kayıtlarında kalır.
- G0.4'ün master planındaki `EvalReport oracle-gap` araç ifadesi gerçek ayrı
  `bench oracle` raporuna göre düzeltildi. Hibrit hatta bir sayısal gap'in
  anlamı henüz kararlaştırılmadı: referans tanımı [#26](https://github.com/barandincoguz/belge-gozu/issues/26),
  kimlik kontrollü rapor uygulaması [#27](https://github.com/barandincoguz/belge-gozu/issues/27)
  (karara bağlı). Legacy `stage1_ms`/`stage2_ms` hibritte `NULL` ve
  `detail.stages` kanoniktir; SQL tüketici/deprecation işi
  [#28](https://github.com/barandincoguz/belge-gozu/issues/28) olarak ayrıldı.
  #28, `d0a4250` ile API→SQLite→Parquet ve two-stage sözleşme testleri,
  tüketici envanteri ve metrics katalog kararı eklenerek kapandı (`make test`:
  904 passed, 6 deselected; `make lint` yeşil).
- `88ddd9d`, cevaplanabilir satırı olmayan retrieval koşumlarının eskiden
  `n=0` ve sıfır Recall/MRR/nDCG üretmesini engelledi. `bench run` ve
  `bench oracle` seçim kontrolünü model yüklemeden yapıyor; doğrudan harness
  çağrısı da açık hata veriyor. Yeniden üretim:
  `uv run pytest tests/bench/test_harness.py::test_retrieval_eval_rejects_a_selection_without_answerable_questions tests/test_cli.py::test_retrieval_cli_refuses_zero_answerable_questions_before_loading_model -q`
  → 2 passed; `make test` → 906 passed, 6 deselected; `make lint` yeşil.
  Bu testin verisi tek cevaplanamaz sentetik satır ve tek sayfalık sentetik
  indeks çiftidir; üretim indeksi/recipe'si veya gerçek model kullanılmadı,
  yeni bir kalite sayısı üretilmedi. Gerçek `retrieval_eval_v2` seçimi
  `verify_retrieval_eval.py --status --bench data/bench/retrieval_eval_v2.jsonl`
  ile kontrol edildi: 62 verified, bunların 47'si insan onaylı cevaplanabilir,
  insan onaylı cevaplanamaz satır 0. Bu sayımın sınırı #13'te açık.
- `fb1e028`, CLI testlerindeki üçüncü `q_dict` kopyasını ortak fabrikaya
  bağladı; ilgili 139 test, `make lint` ve `git diff --check` geçti.
