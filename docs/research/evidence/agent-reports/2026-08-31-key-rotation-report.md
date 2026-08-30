# Gemini anahtar rotasyonu — uygulama raporu

**Tarih:** 2026-08-30/31 · **Dal:** `feat/p0-retrieval-correctness` · **BASE:** `d051918`
**Commit:** `feat(llm): anahtar rotasyonu — herhangi API hatasında otomatik ikinci anahtara yapışkan fallback`

**GÜVENLİK NOTU:** bu raporda, kodda, loglarda, telemetride ve testlerde anahtar
DEĞERİ hiçbir biçimde geçmez (parça, uzunluk, parmakizi dahil). Anahtarlar
hakkında yazılan tek şey `"key1"` / `"key2"` etiketleridir.

---

## 1. Ne yapıldı

Kullanıcı direktifi: `.env`'e ikinci bir ücretsiz-kota anahtarı (`GOOGLE_API_KEY_2`)
eklendi; **herhangi bir LLM API hatasında** aynı istek otomatik olarak öbür
anahtarla yeniden denensin, geri kalan her şey aynı kalsın.

Rotasyon **tek sarmalama noktasına** kondu: `answer/gemini.py::build_gemini_client`
— yanıtlayıcının (`GeminiAnswerer`) ve P2 kanıt doğrulayıcısının
(`answer/verify.py::GeminiVerifierClient`) İKİSİNİN DE geçtiği tek fabrika. Fabrika
artık 1-2 slotluk bir anahtar havuzu üzerinde `RotatingGeminiClient` döndürür;
doğrulayıcı tarafına ekstra tek satır yazılmadı, entegrasyon fabrikanın
tekilliğinden geldi.

| Dosya | Değişiklik |
|---|---|
| `src/belge_gozu/config.py` | `google_api_key_2` alanı (alias: `BG_GOOGLE_API_KEY_2`, `GOOGLE_API_KEY_2`, `GEMINI_API_KEY_2`) |
| `src/belge_gozu/answer/gemini.py` | `KEY_LABELS`, `NON_ROTATABLE_ERROR_TYPES`, `StickyKeyIndex`, `KeySlot`, `RotatingGeminiClient`, `GeminiClient.budget_fits`, `_generate(started=, max_attempts=)`, fabrika + `GeminiAnswerer(api_key_2=)` |
| `src/belge_gozu/answer/verify.py` | `GeminiVerifierClient(api_key_2=)`, `build_gates` ikinci anahtarı geçirir |
| `src/belge_gozu/app/main.py` | `GeminiAnswerer(..., api_key_2=s.google_api_key_2)`; olay künyesine `detail.llm` |
| `src/belge_gozu/cli.py` | `verify run` harness'ı da ikinci anahtarı geçirir (üretimle aynı parçalar) |
| `src/belge_gozu/telemetry/collect.py` | `note()` + `merge_note()` (aynı notu iki üreten ezmesin) |
| `src/belge_gozu/telemetry/prom.py` | `bg_llm_key_rotations_total{from_key}` |
| `docs/research/metrics-catalog.md` | katalog satırı (katalog testi zaten zorunlu kılıyor) |
| `tests/answer/test_gemini.py`, `tests/telemetry/test_prom.py`, `tests/test_config.py` | +23 test |

## 2. Deneme merdiveni (istek başına EN FAZLA 3 çağrı)

Hepsi **TEK** duvar-saati bütçesinin (`GEMINI_TOTAL_BUDGET_S = 35 sn`) altında:
rotasyon denemesi ikinci bir 35 sn açmaz, `started` iki anahtar arasında
paylaşılır.

| # | anahtar | hangi koşulda |
|---|---------|----------------|
| 1 | geçerli (yapışkan) | HER ZAMAN |
| 2 | öbürü | 1 hata verdi VE sınıf `parse` DEĞİL VE ikinci anahtar VAR VE kalan bütçe bir deneme daha kaldırıyor |
| 3 | öbürü | 2 hata verdi VE sınıf retry'lenebilir (`timeout`/`http_5xx`) VE bütçe yetiyor (`GeminiClient`'in mevcut aynı-anahtar retry'si) |

Kararlar:

* **Aynı-anahtar retry hakkı SON anahtara bırakıldı.** İlk anahtarda "öbür
  anahtarı dene" her zaman daha iyi bir ikinci hamledir (429 aynı anahtarda
  umutsuz, öbüründe tam olarak umut vaat eden hatadır); rotasyondan sonra
  gidecek yer kalmadığı için mevcut retry semantiği orada anlamlı.
* **`parse` rotasyon YAPMAZ** (tek istisna). Kullanıcının "herhangi bir hata"
  kuralı TAŞIMA/API düzeyinde birebir uygulanır; `parse` ise BAŞARILI bir
  yanıtın sınıflandırmasıdır (HTTP 200 geldi, gövde okunamadı) — aynı istek
  başka bir anahtarla aynı gövdeyi geri getirir, rotasyon yalnız ikinci bir
  çağrının parasını harcardı. Bu sapma kodda ve testte açıkça yazılı.
* **`safety_block` rotasyon YAPAR.** Sonucu değiştirmesi beklenmez ama
  zararsızdır, kullanıcının kuralına uyar ve tek rotasyonla sınırlıdır.
* **İki anahtar da düştüyse:** `degraded` + **SON** hatanın taksonomisi +
  `detail.llm.keys_tried = ["key1","key2"]` + istisna mesajında
  `[denenen anahtarlar: key1, key2]`.
* **Bütçe rotasyonu da kapsar:** rotasyondan ÖNCE mevcut retry ön-kontrolünün
  aynısı çalışır (`budget_fits`, tek formül iki yerde kullanılır); yetmiyorsa
  rotasyon YAPILMAZ ve `gemini_rotation_skipped_budget` notu düşer.

## 3. Yapışkanlık ve iş parçacığı güvenliği

`StickyKeyIndex` **süreç düzeyinde** tek göstergedir (`_STICKY`): rotasyondan
sonra yeni anahtar geçerli anahtar olur ve sonraki istekler doğrudan onunla
başlar. Alternatif (her istek key1 ile başlar) kotası bitmiş bir anahtarda
**istek başına garantili bir başarısız çağrı** demekti — canlıda ölçülen bedeli
~4,6 sn/istek (§5). Yanıtlayıcı ve doğrulayıcı aynı göstergeyi paylaşır (ikisi
de aynı fabrikadan geçer).

Gösterge `threading.Lock` ile korunur; senkron uç noktalar Starlette iş parçacığı
havuzunda koştuğu için bu gerçekten paylaşılan mutable durumdur. Eşzamanlı iki
istek birbirini **iyi huylu** biçimde ezebilir (ikisi de aynı sıradaki anahtara
gider); garanti edilen, göstergenin her an GEÇERLİ bir slot numarası olmasıdır.
İki iş parçacıklı duman testi bunu kilitler ve **iki meşru sıralanışı da** kabul
eder (aksi halde test yarışı değil, bir zamanlamayı kilitlerdi).

**Süreç yeniden başladığında gösterge sıfırlanır** (kalıcı değil): her `serve`
başlangıcında ilk `/ask` bir kez key1'e çarpar, rotasyon yapar ve orada kalır.
Canlıda ölçülen davranış budur (§5); kalıcılaştırmak ayrı bir karardır, bu
fazın kapsamında değil.

## 4. Telemetri

* Olay künyesi `detail.llm` — YALNIZ gerçekten bir LLM çağrısı olduğunda yazılır
  (`/search` ve eşik-altı `/ask` satırları taşımaz):
  * `key` — hangi anahtar **servis etti** (`"key1"`/`"key2"`),
  * `rotations` — `[{"from": "key1", "error_type": "http_429"}, ...]` SIRAYLA,
  * `keys_tried` — iki anahtar da düştüyse.
* `rotations[].error_type` bilinçli olarak eklendi (canlı sondajın çıkardığı
  boşluk): rotasyon BAŞARILI olduğunda istek `answered` biter ve
  `events.error_type` NULL kalır — yani "key1 kotası doldu" (`http_429`,
  kendiliğinden geçer) ile "key1 iptal edildi" (`auth`, insan müdahalesi ister)
  başka hiçbir yerde ayırt edilemezdi.
* Prometheus: `bg_llm_key_rotations_total{from_key}` — etiket kümesi
  `answer/gemini.KEY_LABELS`'ten gelir (kopya yok, kardinalite kapalı; bozuk bir
  olay satırı yeni etiket sızdıramaz). Hata sınıfı **etiket değil** olay alanıdır.
  Mevcut `bg_llm_*` serileri DEĞİŞMEDİ.
* `/ask` **gövdesindeki** `detail` DEĞİŞMEDİ (yalnız `gate1`/`gate2`): istemci
  sözleşmesi genişletilmedi.

## 5. Canlı sondaj (:7860, ÇALIŞIR BIRAKILDI)

Sunucu yeniden başlatıldı: `BG_DEVICE=mps nohup uv run belge-gozu serve --port 7860`
`healthz`: `{"status":"ok","pages":4222,"threshold":10.6,"top_k":5,"pipeline":"hybrid","index":{"quantization":"int8","revision":"133444d8c235/train-compat-v1/int8"}}`

**1. çip `/ask`** ("İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır?"):

```
status: answered | honest_miss: False | citations: ['k4857:28']
LOG: WARNING:belge_gozu.answer.gemini:gemini key1 ile başarısız (http_429) — istek key2 ile bir kez yeniden deneniyor
/metrics: bg_llm_key_rotations_total{from_key="key1"} 1.0
events id=2909 status=answered error_type=None answer_ms=10179
        detail.llm={'rotations': [{'from': 'key1', 'error_type': 'http_429'}], 'key': 'key2'}
```

Yani kontrolcünün bildirdiği canlı olgu (key1'in günlük kotası tükendi) sondajda
**birebir doğrulandı**: key1 → `http_429` → rotasyon → **key2 servis etti**, kullanıcı
yanıtı gördü, istek `degraded` OLMADI.

**2. `/ask` (yapışkanlık kontrolü)** ("Türk Medeni Kanunu'na göre yerleşim yeri..."):

```
status: answered | citations: ['k4721:4']
/metrics: bg_llm_key_rotations_total{from_key="key1"} 1.0   <-- DEĞİŞMEDİ
events id=2910 status=answered error_type=None answer_ms=5606
        detail.llm={'key': 'key2'}                          <-- rotasyon YOK
```

İkinci istek doğrudan key2 ile başladı: yeni rotasyon yok, ölü anahtara ikinci
kez çarpılmadı. Yapışkanlığın ölçülen bedeli/kazancı: `answer_ms` 10 179 → 5 606
ms (~4,6 sn, tam olarak atlanan başarısız key1 denemesi).

Sunucu **:7860'ta çalışır durumda bırakıldı** (final kod ile).

## 6. Doğrulama

```
uv run pytest -q -m "not slow"   ->  650 passed, 6 deselected      (BASE: 627)
make lint                        ->  ruff check + ruff format + pyright: All checks passed / 0 errors
```

Yeni 23 test — kapsananlar: key1 429 → key2 servis + `detail.llm` + sayaç;
yapışkanlık (2. istekte key1'e SIFIR çağrı); iki anahtar da 429 → `degraded` +
`keys_tried` + `http_429`; `parse` → rotasyon YOK; bütçe dolu → rotasyon YOK
(tek deneme, `timeout`); bütçe yetiyor → rotasyon VAR; merdivenin 3. basamağı ve
`<= 3` deneme tavanı; `generate_json` (doğrulayıcı yolu) aynı merdivenden geçer;
doğrulayıcı istemcisi aynı havuzu paylaşır + önbellek anahtarı ANAHTARDAN
BAĞIMSIZ (`cache_key` imzası: model + istem sürümü + iddia + kanıt sha);
tek-anahtar havuzu (etiket slota bağlı, keyless boot, aynı-anahtar retry
BİREBİR eskisi); `GOOGLE_API_KEY_2` ortam eşlemesi + varsayılan boş;
eşzamanlılık dumanı. Mevcut 627 test ELLENMEDİ ve yeşil — tek anahtarlı
davranışın değişmediğinin kilidi budur.

## 7. Dürüst sınırlar

* **Bütçe garantisi genişlemedi:** invariant retry/rotasyon KARARINI kapsar, tek
  bir denemenin İÇİNİ kapsamaz (httpx faz-başına sayaç; canlı ölçüm 16,2 sn /
  15 sn). Garanti "toplam <= 35 sn" değil, "bütçe aşılmışken üstüne bir deneme
  daha BİNMEZ"dir — rotasyon bu cümleyi değiştirmedi, kapsamına girdi.
* **Yapışkan gösterge kalıcı değil:** süreç yeniden başladığında key1'e bir kez
  daha çarpılır (§3).
* **İkinci anahtar da tükenirse** sistem `degraded`'a düşer; iki anahtar tek bir
  ücretsiz kotanın iki katıdır, sonsuz kota değil.
* **`auth` sınıfı da rotasyona girer:** iptal edilmiş bir key1, her istekte bir
  başarısız çağrıya mal olur (rotasyon sonrası yapışkanlık bunu istek başına
  DEĞİL, süreç başına bir kereye indirir). Ayrım artık telemetride görünür
  (`rotations[].error_type == "auth"`).
