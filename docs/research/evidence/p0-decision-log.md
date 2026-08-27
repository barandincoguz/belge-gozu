# SDD ledger — plan: docs/superpowers/plans/2026-08-26-belge-gozu-p0-retrieval-correctness.md

Spec: docs/superpowers/specs/2026-08-26-belge-gozu-rag-quality-v2-design.md
Branch: feat/p0-retrieval-correctness (base d964b9c; plan docs commit 59223b7)
Model politikası: implementer taban = sonnet (kullanıcı geri bildirimi: haiku YASAK),
review = sonnet, eskalasyon (round 4-5) = opus, final review = opus/fable.

## Pre-flight scan (2026-08-26)

Paylaşılan dosya/arayüz çiftleri:
| Çift | Üretilen ↔ Tüketilen | Bulgu |
|---|---|---|
| T1↔T2/T3/T4/T9/T12/T13 | QueryFormat/IndexManifest/CPE_0_3_18 | adlar birebir, temiz |
| T2↔T3 | maskeli encoder ↔ store zero-row reddi | tutarlı (yeni encoder sıfır satır üretmez; FakeEncoder rastgele) |
| T3↔T4 | manifest'li PackedIndex ↔ tiny_corpus fixture güncellemesi | T4 notu fail-fast testinde manifest.json unlink — tutarlı |
| T4↔T5 | app/main.py compat check ↔ pipeline seçimi | sıralı görevler, çakışma yok |
| T5↔T8 | score_all / search_embedding ↔ Diagnostic adapter'lar | imzalar eşleşiyor |
| T8↔T9 | EvalReport/rank_of ↔ bench oracle CLI | eşleşiyor |
| T9↔T12 | FloatIndex ↔ derive_packed/Int8Index | eşleşiyor |
| T13↔T14 | schema/app alanları ↔ README/rapor | sıralı, temiz |

Görev-içi tutarlılık: T4 fail-fast testi + fixture notu (yukarıda); T5 chunk testi
CHUNK_TOKENS'ı instance attr olarak override ediyor — score_all bounds'u çağrı anında
hesaplıyor, tutarlı; T9 f16 roundtrip toleransı sınırda (R4); T3/T4/T8 testleri
tests.* çapraz import kullanıyor (R1).

## Rulings (pre-flight)

- Ruling R0: Worktree yerine mevcut çalışma ağacında yeni branch
  (feat/p0-retrieval-correctness). Neden: slow testler + runbook'lar untracked
  data/ (753MB görüntü + indeks) gerektiriyor; worktree'de olmazdı. Kullanıcının
  commit'lenmemiş Makefile/observability/ değişiklikleri yerinde korunuyor.
  Yanlışsa maliyeti: main'e görece dağınık geçmiş — squash/merge ile telafi.
- Ruling R1: Çapraz test importları için T1'de boş `tests/__init__.py` eklenecek
  (tests paketi importable olur). Yanlışsa maliyeti: pytest import modu sürprizi; düşük.
- Ruling R2: T5 candidates aktarımı — AskService.ask `candidates: int | None = None`
  tutar; None değilse `search(q, k, candidates=...)`, None ise `search(q, k)` çağırır;
  app iki-aşamalı modda `s.stage1_candidates` geçirir (plan'daki functools.partial
  taslağının yerine — tip-güvenli). Maliyeti: yok; plan niyetiyle aynı.
- Ruling R3: CLI index build'de corpus_checksum dosyalar yazıldıktan SONRA hesaplanır;
  manifest `write_manifest()` ile save sonrası yazılır (PackedIndex.build(manifest=...)
  bellek-içi/test kullanımına kalır). Float indeks dizinine de meta.parquet kopyalanır.
- Ruling R4: T9 f16 roundtrip testinde atol=1e-3 f16 aralık hatasına takılırsa
  atol=2e-3'e genişletilir (yorum satırıyla). Maliyeti: biraz gevşek tolerans.
- Ruling R5: Commit'lerde YALNIZ görev dosyaları stage edilir (açık yollar;
  `git add -A` yasak). Kullanıcının Makefile/observability/ değişiklikleri hiçbir
  commit'e girmez.
- Ruling R6: Görev sırası: 1,2,3,4,5,6,7,8,9,13,15 → 10 (taslak+İNSAN kapısı) →
  11,12 → 14. Neden: T10'un kullanıcı doğrulaması plan gereği bloklayıcı; kod
  görevleri önce biter, uzun model koşumları (T11) canary onayı beklerken başlar.

## Görevler

(başlangıç: hiçbiri tamamlanmadı)

Task 1: minor (deferred): manifest.py `dir: Path` builtin gölgeleme (brief kaynaklı); corpus_checksum eksik-dosya testi yok
Task 1: complete (commits 59223b7..bb09374, review clean)

Task 2: Ruling R7: batch-vs-single slow testinin ==1.0 iddiası MPS'te kalıcı FAIL üretir
(ölçüm 0.9990/0.9989). Test, karar sonrası invariant'a dönüştürülür: agreement >= 0.995
eşiği + karar/ölçüm yorum satırı; gerçek bit-exact kilit T10 canary + batch=1 build'te.
Yanlışsa maliyeti: 0.995-1.0 arası gerileme yakalanmaz — baseline raporu ham değerleri tutar.

Ruling R8: b38be99 (docs/telemetry, BAŞKA bir Claude oturumunun commit'i — session
01G4ezdAg4Mq8SwQXsrFWegb) Task 2 sırasında paylaşılan çalışma ağacında branch'imize
düştü. Karar: history yeniden yazılmaz (başka oturumun işi yok edilemez); commit
zararsız içerik (telemetri bulgu notu) ve merge'de main'e taşınır. Task review
aralıkları bu yabancı commit'i DIŞLAR (Task 2 base = b38be99). Yanlışsa maliyeti:
branch geçmişinde konu-dışı bir commit.

Task 2: minor (deferred): R7 yorumunda 3. ölçüm değeri (1.0) eksik; FakeTorchLike ölü sınıf (brief kaynaklı); model_revision/doc_prompt attr'ları için doğrudan birim test yok (T4/T11 dolaylı kapsar)
Task 2: complete (commits b38be99..2cd099e, review clean; MPS batch nondeterminizmi ölçüldü -> build batch=1 kararı)

Task 3: complete (commits 2cd099e..d458e0d, review clean)
Ruling R9: Eş oturumun (telemetri) küçük düzeltme dalgasına pencere açıldı (app/main.py,
cli.py, docs/research; commit'ler branch'imize düşecek — R8 gereği kabul). Bu sırada
YALNIZ disjoint-yeni-dosyalı Task 6 (src/belge_gozu/bench/, tests/bench/) paralel koşar;
app/main.py-cli.py'ye dokunan Task 4-5, eş oturum "pencere kapandı" diyene kadar bekler.
Yanlışsa maliyeti: commit sıralamasında iç içe geçme — içerik çakışması yok.

Task 6: Ruling R10: data/ gitignore'lu; benchmark artefaktları (data/bench/) plan gereği
commit'lenmek zorunda -> .gitignore'a `!data/bench/` istisnası eklenir (data/manifest
deseninde). Yanlışsa maliyeti: büyük sonuç dosyaları repoya girebilir — results/ boyutu
küçük JSON; kabul.

Task 6: fix round 1/5 (6 addressed, 0 open — 5 negatif-yol validator testi + split fallback yorum/test; commit b4cfe1e)
Task 6: complete (commits d458e0d..b4cfe1e + b4cfe1e fix, review clean; R10 .gitignore bench istisnası dahil)
Not: Eş oturum (telemetri) 252d12f ile işini bitirdi ve commit yetkisini bıraktı; Task 4-5 serbest.

Task 4: complete (commits b4cfe1e..a09ee75, review clean)
Task 4: minor (deferred): compat mismatch dallarının (model_name/mask_policy/checksum) doğrudan birim testi yok (brief kaynaklı)
Task 4: NOT (T9'a taşınan yükümlülük): `index build` şu an manifest YAZMIYOR; taze v1 indeks compat'ten geçemez.
T9 dispatch'i packed VE f16 yollarında save-sonrası write_manifest'i (R3 sırası: dosyalar → checksum → manifest) zorunlu kılacak.

Task 5: Ruling R11: plan'ın karşı-örnek testi (self-query) ayrıştırıcı değil — self-query'de
mean-sign stage-1 daima doğru sayfayı bulur. Test, karışık-sorgu (iki sayfanın token'ları)
taramasıyla değiştirilir: stage-1 top-1 != exhaustive argmax olan çift bulunur ve
stage-1-kısıtlı sonucun exhaustive'den saptığı assert edilir. Yanlışsa maliyeti: yok
(daha güçlü test). Minor #4 (CHUNK bellek ~450MB transient) G1.7 bütçe ölçümüne not edildi.

Task 5: fix round 1/5 (6 addressed, 0 open — detail.stages telemetri + ayrıştırıcı karşı-örnek (a=0,b=2) + int32 sim + docstring/pyright/invariant; commit de1b68b)
Task 5: complete (commits a09ee75..de1b68b, review clean; ÜRETİM YOLU ARTIK EXHAUSTIVE)
Task 5: minor (deferred): prom _STAGE_COLS exhaustive_maxsim serisini bilmiyor -> T13 kapsamı; detail.stages iki-aşamalıda hafif yinelenme

Task 7: complete (commits de1b68b..223c40d, review clean)

Task 8: Ruling R12: StageRecord.gold_ranks record_top ile sınırlı kalır (-1 = ilk N'de yok);
tam-korpus sıra teşhisi (ör. 1768 karşı-örneği) T9 oracle çıktısının işidir — harness
sözleşmesi inceleme ortasında genişletilmez. Yanlışsa maliyeti: stage-1 tam sırası yalnız
oracle koşumlarında görünür; kapı kararları etkilenmez (survival + recall@candidate yeterli).

Task 8: fix round 1/5 (7 addressed, 0 open — TwoStage adapter üretim-skor testi, Pipeline StrEnum, record_top eşleme, dürüst latency notu, git_commit public, tam metrik çıktısı; commit 440d8c6)
Task 8: complete (commits 223c40d..440d8c6, review clean)

Task 9: fix round 1/5 (4 addressed, 0 open — oracle missing-gold guard, bench run manifest-format, ölü import, atol yorumu; commit 6e9864e)
Task 9: complete (commits 440d8c6..6e9864e, review clean; index build artık manifest yazıyor — T4 yükümlülüğü kapandı)
Task 9: minor (deferred): FloatIndex.build padding-satır invariantı yok (packed eşi yakalar); _chunk_bounds/CHUNK_TOKENS kopyası drift riski; bench oracle corpus_checksum çapraz kontrolü yok; test_cli fixture yinelenmesi; binary rank tie-yorumu raporda not edilmeli
Task 13: Ruling R13: detail["retrieval"] canlı serviste yalnız kimlik alanları taşır
(query_format, quantization); top-20 candidate kaydı plan metnindeki haliyle k-şişirme
gerektirirdi — derin aday teşhisi bench harness'ın işi (top-200 kaydediyor). Prom:
detail.stages'ten YALNIZ _STAGE_COLS'ta olmayan aşamalar bg_stage_duration_seconds'a
observe edilir (çifte sayım yok). Yanlışsa maliyeti: canlı istek başına aday listesi
yalnız top-k; kapı kararları etkilenmez.

Task 13: complete (commits 6e9864e..0e898c4, review clean; exhaustive_maxsim artık /metrics'te)

Task 15: complete (commits 0e898c4..67b01be, review clean)
Task 15: minor (deferred): Starlette deprecation uyarısı dar filtreyle bastırıldı; kalıcı çözüm bağımlılık güncellemesi (backlog)

Task 10: canary_v1 TASLAK hazır (commit 88e0a1a): 48 soru (43 answerable + 5 unanswerable),
dilim dağılımı plan hedefiyle uyumlu, tüm gold sayfalar korpusta doğrulandı, iki hedef sorgu
(c001/c002) dahil. verification_status=draft — İNSAN KAPISI kullanıcıda.
Task 11 (Step 1-2): complete (commit cca153d, review Approved).
  BULGU: sorgu formatı zaten doğruymuş; DOKÜMAN prompt'u farklıymış —
  eğitim: "<|im_start|>User: Describe the image.<image><end_of_utterance>" (871 token/sayfa)
  mevcut: "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:" (875)
  ST çapraz kontrol: token sayıları birebir, sign uyumu 0.9935/0.9922/0.9925 (fp16 kaynaklı).
Task 11 açık bulgular (Step 4 ÖNCESİ kapatılacak): (1) ab_st_reference.py'ye ids-identical +
871/875 negatif kontrol eklenmeli; (2) bench oracle guard doc_prompt_sha256'yı da karşılaştırmalı.
Ölçüm: encode 0.71 s/sayfa (MPS, batch=1) -> f16 build ~50 dk/varyant.
Build başlatıldı (arka plan): data/index-cpe0318-f16 (A1) + data/index-traincompat-f16 (A2).

Task 12: complete-pending-fixes (commit 5b63b52, review Approved + 3 Important düzeltme Step 5 öncesi)
Task 12: Ruling R14: C2 ablasyonunun int8 kolu koşulamıyor (hiçbir bench girişi Int8Index'i
skorlamıyor). Karar: `bench oracle`'a opsiyonel `--int8-index DIR` üçüncü kolu eklenir
(binary/float/int8 aynı koşumda, aynı sorgu embedding'i). Alternatif (ayrı script) reddedildi:
üç oracle'ın aynı koşum künyesinde raporlanması C2 karşılaştırmasının ön şartı.
Yanlışsa maliyeti: bench oracle biraz daha kalabalık bir komut olur.

Task 11: fix round 1/5 (5 addressed — self-verifying script (ids-identical + 871/875 negatif
kontrol), bench oracle doc_prompt_sha256 guard, --device, Protocol/type-ignore, repo-root path;
commit 2710779)
Task 11 BULGU (p0-gate riskine taşınacak): fp16 hipotezi ÇÜRÜTÜLDÜ — CPU/fp32'de de sign uyumu
0.9944/0.9912/0.9920 (MPS-fp16: 0.9935/0.9922/0.9925). Fark dtype değil, implementasyon
düzeyinde (colpali-engine ColIdefics3 vs ST modül yığını; ST yüklerken linear.weight/bias
UNEXPECTED log'u). Mean cosine >= 0.9995. Format kararı ETKİLENMİYOR (kanıt: input_ids birebir).
Açık soru: sign-1bit paketleme referans implementasyona göre ~%0.8 bit belirsizliği taşıyor.
A/B'yi confound ETMEZ (iki kol da aynı encoder).
Task 11 (Step 1-2): complete (commits cca153d..2710779, review clean)
Task 12: fix round 1/5 (6 addressed, 0 open — per-token-scale testi (global-scale/truncation/3-bit
artık FAIL), chunk'lı derive (peak ~6GB -> ~1.9GB), bench oracle --int8-index kolu (R14),
_chunk_bounds geç bağlama, derive guard'ları, roundtrip tam doğrulama; commit e5b1283)
Task 12: complete (commits cca153d..e5b1283, review clean)
Task 14 (Step 1-2): fix round 1/5 (3 addressed — UI footer + pipeline şeridi artık exhaustive
hattı anlatıyor, README kuantizasyon iddiası C1/C2'ye atıfla kesinleşti; commit acf7432)
Task 14 (Step 1-2): complete (commits e5b1283..acf7432, review clean)

DURUM: bütün KOD görevleri bitti (T1-T9, T11 s1-2, T12, T13, T14 s1-2, T15).
Kalan: T11 s3-6 (build koşuyor) + T12 s5-6 (C1/C2) + T14 s3-4 (raporlar) + T10 insan kapısı.
D1 aracı: complete (commit 6a18209, review clean) — scripts/d1_augmentation.py + testi.

A1 f16 build TAMAM (17:05-18:02, 57 dk): data/index-cpe0318-f16, 922MB, 4222 sayfa,
3.776.882 token. KANIT (G0.6): eski v0 indeksi 3.780.842 token idi; fark TAM 3.960 —
yani ölçülen padding satırı sayısı birebir. Yeni indekste 0 all-zero satır.
model_revision=650243e9... (gerçek sha), mask_policy=drop-padding, quantization=float16.
A2 build başladı 18:02.
Ruling R15: `bench run` ve `bench oracle` load_bench'i only_verified=True ile çağırıyor;
canary taslak (hiç verified yok) ile ValueError veriyor. Karar: her iki komuta
--only-verified/--all seçeneği eklenir (d1_augmentation deseninde), varsayılan --all +
aktif modun basılması. Gerekçe: insan kapısı her ölçümü bloke etmemeli; kapı sayıları
yine yalnız verified set üzerinden alınır. Yanlışsa maliyeti: taslak üstünde koşulan
ölçümün kapı sayısı sanılması — rapor modu açıkça yazacak.

B1/B2 BASELINE (v0 indeksi, canary TÜMÜ n=43 answerable, CPU, 18:13-18:17):
  two-stage (v0 üretim hattı): Recall@5=0.000  MRR=0.004  nDCG@5=0.000
  exhaustive              : Recall@5=0.070  MRR=0.049  nDCG@5=0.050  CI(0.0-0.163)
  -> Stage-1 kaldırma kararının sayısal kanıtı: eski hat canary'de HİÇBİR soruyu
     top-5'e sokamıyor. Aynı zamanda %7'lik tavan, spec'in "görsel kanal tek başına
     yetmez, hibrit şart (P1)" tezini destekliyor.
  Raporlar: data/bench/results/baseline-v0idx-{exhaustive,twostage}.json

C1/C2 (A1 = cpe-0.3.18 formatı, canary n=43, CPU, 18:18-18:24):
  arm      R@1    R@5    R@20   R@50   R@200
  1-bit    0.023  0.070  0.070  0.174  0.360
  float16  0.070  0.093  0.186  0.209  0.349
  int8     0.070  0.093  0.186  0.209  0.349
  -> int8, float16 ile HER k'da BİREBİR aynı (kayıp 0.0 puan).
  -> 1-bit'in R@20 kaybı 11.6 puan; karar kuralı eşiği (<=2 puan) 5.8x aşılıyor.
  Hedef sorgular (gold k4721:4): c001 uzun -> 1bit 1687 / float 2325 / int8 2348
                                 c002 kısa -> 1bit 2    / float 6    / int8 6
  GERİLİM: genel kalite int8 lehine (R@20 2.7x), ama G0.8'in kilit sorgusu (c002 top-5)
  yalnız 1-bit'te sağlanıyor. Karar A2 (train-compat) sonuçlarından SONRA verilecek.
  Boyutlar: 1bit 58MB / int8 476MB / f16 922MB.
  LATENCY (4222 sayfa, 40-token sorgu, CPU, skorlama-only):
    1-bit  2.34 s/sorgu (58MB)  |  int8 0.51 s (476MB)  |  float16 0.15 s (922MB)
  -> 1-bit hem kalitede hem HIZDA kaybediyor (BLAS matmul vs popcount+geçici dizi).
     Tek avantajı disk. Plan kuralı ("en küçük yeterli temsil") int8'i seçiyor;
     float16 daha da hızlı ve aynı kaliteli, HF Space disk bütçesine göre yeniden
     değerlendirilebilir (rapora not).

A/B KARARI (canary n=43): train-compat-v1 sorgu formatı + train-compat doküman prompt'u
KAZANDI, her metrikte:
  float  R@5 0.093->0.233 (+14.0 puan, 2.5x) | R@20 0.186->0.302 | R@200 0.349->0.535
  1-bit  R@5 0.070->0.116 (+4.7)             | R@20 0.070->0.233 (+16.3)
  Hedef sorgular (A2): c001 uzun 1bit=1221 float=661 int8=664 (v0: 3127 Stage-1 / 1576 exh.)
                       c002 kısa 1bit=4 float/int8=4  -> G0.8 (top-5) SAĞLANDI
C2 KARARI (A2 formatında): int8 = float16 ile BİREBİR aynı (kayıp 0.0 her k'da);
  1-bit'in R@20 kaybı 7.0 puan (eşik <=2) -> 1-bit "tek production truth" DEĞİL.
Ruling R16: int8 üretim yoluna BAĞLANAMIYOR — ExhaustiveBinaryRetriever yalnız PackedIndex
tüketiyor, Int8Index için retriever/serve entegrasyonu yok. Karar: (a) üretim indeksi
train-compat 1-bit olur (mevcut retriever'la uyumlu, format kazancını hemen alır),
(b) C2 sonucu p0-gate'te KARAR olarak kaydedilir (G0.7 "kayıp sayılandırıldı" şartı sağlanır),
(c) int8/float üretim entegrasyonu P1'in ilk işine devredilir (master §8'e eklenecek).
Gerekçe: int8 entegrasyonu correctness değil performans işi; P1 zaten retrieval mimarisini
değiştiriyor. Yanlışsa maliyeti: R@20'de ~7 puanlık kazanç P1'e kadar ertelenir.

D1 (augmentation, kazanan indekste): with-aug R@5=0.116 / no-aug R@5=0.116 -> BERABERE.
  ColPali makalesinin İngilizce bulgusuyla tutarlı; Fransızca'daki kazanç Türkçe'de yok.
  Karar: eğitim formatı gereği n_suffix=10 korunur, değişiklik yok.
T11 Step 6: complete (commit 6dd7099, review Approved) — üretim varsayılanları:
  index_dir=data/index-traincompat-1bit, query_format_id=train-compat-v1, doc_prompt_id=train-compat.
  CANLI DOĞRULAMA: create_app + TestClient /search "Yerleşim yeri nedir?" -> k4721:4 rank 4,
  skor 73.17. Kaybeden formata işaret edince IndexCompatibilityError (fail-fast çalışıyor).
Ruling R17: inceleme boşluk buldu — check_compatibility doc_prompt_sha256'yı KARŞILAŞTIRMIYOR
(bench oracle'a eklendi ama serve'e eklenmedi). Doc prompt artık bağımsız kalite ekseni
(A/B'de 14 puan). G0.5 "uyumsuzluk fail-fast" bunu kapsamalı -> compat.py'ye eklenecek.

DÜZELTME (gate incelemesi, 2026-08-27): A2 c001 int8 sırası 664'tür (661 değil; 661 float'ın
sırasıdır). Kaynak: data/bench/results/a2-traincompat-oracle.json.
LATENCY ham artefaktı üretildi: data/bench/results/latency-by-representation.json — kazanan
formatta (train-compat), makine BOŞTA: 1-bit 1.083 s / int8 0.243 s / float16 0.079 s.
Ledger'daki önceki değerler (2.34/0.51/0.15) MPS build'i koşarken alınmıştı; oranlar aynı
(int8 1-bit'ten 4.5x, float16 13.7x hızlı), mutlak değerler makine yükü nedeniyle ~2x yüksekti.

B2 EKSİK ABLASYON TAMAMLANDI (2026-08-27 19:53-19:56, kazanan train-compat indeks, MPS):
  aday        R@5    R@20   MRR    nDCG@5  gold candidate survival
  200         0.023  0.047  0.013  0.012   9.3%
  500         0.047  0.093  0.018  0.022   20.9%
  1000        0.116  0.163  0.051  0.064   34.9%
  exhaustive  0.116  0.233  0.068  0.067   44.2% (record_top=200 içinde)
  -> G0.3'ün ASIL ölçümü: Stage-1'in gold candidate survival'ı hiçbir aday sayısında
     %98'e yaklaşmıyor (c=200'de %9.3). Aday havuzunu korpusun %24'üne (1000) çıkarmak
     bile MaxSim sıralamasını exhaustive'e yetiştirmiyor (MRR 0.051 vs 0.068).
     Stage-1'i üretim dışında tutma kararının en güçlü sayısal kanıtı budur.
  Ham: data/bench/results/b2-traincompat-twostage-c{200,500,1000}.json + -exhaustive-ref.json
