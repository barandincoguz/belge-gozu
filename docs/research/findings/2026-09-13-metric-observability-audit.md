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
- `90b3fed` — persisted answer/retrieval raporları için tamper/aggregate doğrulayıcı eklenir.
- `fc148bd` — custom reranker raporlarına per-question ranked evidence ve doğrulama desteği eklenir.

Veri ve artefakt iddiaları, koşumun kendi künye alanlarıyla birlikte okunmalıdır;
model cache'i veya kullanıcı artefaktları silinmemiştir.

## Doğrulanmış kapılar

| Alan | Kanıt | Sonuç |
|---|---|---|
| Ağsız regresyon | `uv run --extra dev pytest tests -q -m "not slow"` | **880 passed, 2 skipped** |
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
| UI/API dürüstlüğü | bazı eski bulgular düzeltildi; per-stage UI zamanı ve kalan a11y açık | [#4](https://github.com/barandincoguz/belge-gozu/issues/4), [#20](https://github.com/barandincoguz/belge-gozu/issues/20) |
| Docker/Hub/hosting | uçtan uca taze pull ve canlı URL yok | [#5](https://github.com/barandincoguz/belge-gozu/issues/5), [#6](https://github.com/barandincoguz/belge-gozu/issues/6), [#7](https://github.com/barandincoguz/belge-gozu/issues/7) |
| Bench diagnostics | K9/K10 kapandı; legacy stage SQL ve oracle sınırı açık | [#8](https://github.com/barandincoguz/belge-gozu/issues/8) |
| Gate policy/API visibility | `/healthz` calibrator ve `abstain_reason` eksikleri açık | [#9](https://github.com/barandincoguz/belge-gozu/issues/9) |
| P2 final gate | test split, G2 raporu ve quota planı yok | [#10](https://github.com/barandincoguz/belge-gozu/issues/10) |
| Dense/retrieval | dense aday coverage ölçüldü, kalite kazancı yok; production fusion yok | [#11](https://github.com/barandincoguz/belge-gozu/issues/11) |
| Benchmark gücü | v2 insan n=47; spec'in 120+30 hedefi ve 12 dolu dilim yok | [#12](https://github.com/barandincoguz/belge-gozu/issues/12) |
| İnsan doğrulaması | retrieval v2 insan 47; abstention insan 0 ve draft satırlar var | [#13](https://github.com/barandincoguz/belge-gozu/issues/13) |
| Teknik borç/güvenlik | BM25 karmaşıklığı, env izolasyonu, erişim ve gizlilik açık | [#14](https://github.com/barandincoguz/belge-gozu/issues/14), [#15](https://github.com/barandincoguz/belge-gozu/issues/15) |
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
