# Güvenlik düzeltmesi: `/pages` sızıntısı + sahte atıf fallback'i

Tarih: 2026-08-29 · Dal: `feat/p0-retrieval-correctness`

## Özet

Üç bulgu kapatıldı:

1. **KRİTİK** — `/pages` tüm `data/` ağacını sunuyordu; telemetri veritabanı
   (`requests.sqlite`, ham kullanıcı sorguları) ve korpus PDF'leri indirilebiliyordu.
2. **KRİTİK** — aynı handler'da dizin yolu işlenmemiş `IsADirectoryError` ile 500 veriyordu.
3. **ÖNEMLİ** — atıf bulunmadığında top-1 sayfayı uydurma "dayanak" olarak ekleyen fallback.

## 1 + 2 — `/pages` yalnızca sayfa görüntülerini sunar

`src/belge_gozu/app/main.py` · `page_image()`

Önceki hâl `s.data_dir` altında çözüyor, `is_relative_to` (yol aşımı doğru şekilde
engelliydi) ve `exists()` kontrolü yapıp sabit `media_type="image/webp"` ile dosyayı
döndürüyordu. Uzantı allowlist'i ve `is_file()` kontrolü yoktu. `belge-gozu serve`
`0.0.0.0`'a bağlandığı için bu, ağdaki herkese açıktı.

Yeni kurallar — üçü de sağlanmazsa 404:

- Çözülen yol `s.data_dir / "images"` altında olmalı (yalnızca `data_dir` değil).
  Bu, aynı zamanda **yol aşımı korumasının kendisidir**: `data_dir`'den daha dar bir
  köktür, `..` ile dışarı çıkan ya da `images/` dışına işaret eden her yolu eler.
  `resolve()` sembolik bağları çözdüğü için `images/` içinden dışarı gösteren bir
  link de reddedilir.
- Uzantı `.webp` olmalı (`suffix.lower()`).
- `is_file()` — `exists()` değil. Dizin yolu artık `FileResponse`'a düşüp
  `IsADirectoryError` ile 500 üretmiyor.

URL biçimi korundu: UI `/pages/${image_path}` çağırıyor ve `meta.parquet` içindeki
`image_path` zaten `images/...` ile başlıyor, bu yüzden çözümleme `data_dir`'e göre
yapılmaya devam ediyor; kısıt sonuç yolunun `images/` altında olmasında.

Reddedilen her yol **404** döner (403/500 değil): uç nokta neyin var olup olmadığını
sızdırmaz.

### Testler — `tests/app/test_api.py` (+5)

- `test_page_image_served` (mevcut) — meşru sayfa görüntüsü 200 + `image/webp`.
- `test_pages_does_not_serve_telemetry_db` — gerçek `EventRecorder` DB'si oluşturulur,
  `/pages/requests.sqlite` → 404, gövdede `SQLite format 3` yok.
- `test_pages_rejects_paths_outside_images_dir` — `pdf/whatever.pdf`, `meta.parquet`,
  `index/meta.parquet` → 404.
- `test_pages_rejects_non_webp_inside_images` — `images/notlar.txt` → 404
  (images/ altında olmak yetmez).
- `test_pages_directory_is_404_not_500` — `images` ve `images/d0` → 404.
- `test_pages_blocks_traversal` — `..` yüzde-kodlu (`%2e%2e`) gönderilir ki HTTP
  istemcisi sadeleştirmesin. Handler'a bozulmadan ulaştığı ayrıca doğrulandı
  (`/pages/%2e%2e/x.webp` → path param `../x.webp`), yani testler boş yere geçmiyor.

## 3 — Sahte atıf fallback'i kaldırıldı

`src/belge_gozu/answer/gemini.py` — silinen iki satır:

```python
if not citations and pages:
    citations = [pages[0].page_id]
```

Model doğru biçimde "verilen sayfalarda bulamadım" dediğinde (metinde `[Sn]` işareti
yok) bu, top-1 sayfayı uydurma atıf olarak ekliyor ve UI bir yanıt-olmayanın altına
"dayanak" çipi basıyordu. Artık atıf yoksa `citations=[]`.

`tests/answer/test_gemini.py`: `test_citation_fallback_top1` →
`test_no_marker_means_no_citation`, `citations == []` iddiasıyla ters çevrildi.
Başka hiçbir iddia zayıflatılmadı.

**UI'da değişiklik gerekmedi.** `src/belge_gozu/app/static/index.html:439-454` zaten
`cites.innerHTML = ""` ile temizleyip `if (a.citations.length)` koşuluyla render
ediyor — atıfsız yanıt artık boş kutu değil, hiç çip göstermiyor. Yapı ve Türkçe ses
olduğu gibi bırakıldı.

## Doğrulama

```
uv run pytest -q -m "not slow"   → 211 passed, 5 deselected
make lint                        → ruff check: All checks passed!
                                   ruff format: 86 files already formatted
                                   pyright: 0 errors, 0 warnings, 0 informations
```

211 = önceki 206 + 5 yeni `/pages` testi.

### Çalışan sunucuya karşı (:7860)

Önce (eski kod canlıydı, PID 66857):

```
GET /pages/requests.sqlite -> 200
GET /pages/images          -> 500
```

Sunucu tam olarak istendiği gibi yeniden başlatıldı
(`BG_DEVICE=mps nohup uv run belge-gozu serve --port 7860 > /tmp/bg-serve.log 2>&1 &`),
yeni PID 73926, `/healthz` 200.

Sonra:

```
GET /pages/requests.sqlite            -> 404
GET /pages/images/k4721/0004.webp     -> 200   (content-type: image/webp)
GET /pages/images                     -> 404
GET /pages/pdf/k4721.pdf              -> 404
GET /pages/meta.parquet               -> 404
```

Uçtan uca `/search`:

```
POST /search {"query":"kira artışı sınırı nedir?"}  → HTTP 200 in 0.999s, 5 hit
  k6098:69  69.7  images/k6098/0069.webp
  k6098:64  69.0  images/k6098/0064.webp
  k6098:61  67.4  images/k6098/0061.webp
GET /pages/images/k6098/0069.webp (arama sonucundan) -> 200
```

Sunucu **çalışır durumda bırakıldı** (PID 73926).

## Kapsam dışı / notlar

- **`/pages` telemetriye görünmez.** `/pages`, `/healthz`, `/stats`, `/metrics` ve `/`
  hiçbir zaman `EventRecorder`'a kaydedilmiyor. İstendiği gibi dokunulmadı — yalnızca
  dizin yolunun 500 yerine 404 dönmesi düzeltildi. `/pages` erişim kaydı ileride ayrı
  ele alınmalı; şu anda görüntü uç noktasına gelen istekler için hiçbir denetim izi yok.
- Telemetri DB'si hâlâ varsayılan olarak `BG_LOG_QUERY_TEXT=True` ile ham sorgu
  metnini saklıyor. Artık HTTP üzerinden indirilemiyor, ama veri saklama politikası
  (ve varsayılanın doğru olup olmadığı) ayrı bir karar.
- Bu düzeltme dosyayı sunma yolunu daraltır; `data/` altındaki dosya izinlerine
  dokunulmadı ve hiçbir indeks yeniden kurulmadı.
