# Telemetri Runbook

## Stack'i kaldır
1. Sunucu: `uv run belge-gozu serve` (host'ta, :7860)
2. `make obs-up` → Prometheus http://localhost:9090 · Grafana http://localhost:3001
3. Doğrula: Prometheus Targets sayfasında `belge-gozu` UP; Grafana'da "Belge-Gözü" dashboard'u.

> **Port notu:** Spec/brief'in varsayılan Grafana portu `3000`'dir; bu repoda
> `observability/docker-compose.yml` Grafana'yı host `3001`'e eşler (container
> içi port hâlâ `3000`). Sebep: bu stack'in ilk canlı doğrulamasının yapıldığı
> geliştirme makinesinde host `:3000` başka (bu projeyle ilgisiz) bir süreç
> tarafından zaten tutuluyordu, ve `docker compose up` bu çakışmayı sessizce
> yutup Grafana'yı fiilen erişilemez bırakıyordu (bkz. `task-9-report.md`).
> Kendi makinende `:3000` boşsa, `docker-compose.yml`'deki `ports: ["3001:3000"]`
> satırını `["3000:3000"]`'e çevirip normal spec portuna dönebilirsin.

## Ölçüm oturumu koş
1. `uv run python scripts/loadgen.py --concurrency 8 --duration 60 --out docs/research/findings/raw/$(date +%F)-loadgen.json`
2. Gerçek yanıt yolu için EN FAZLA 2-3 `/ask` sorusu (kota: ≈20/gün): UI'dan ya da curl ile.
3. `uv run belge-gozu metrics summary` çıktısını not al.
4. `uv run belge-gozu metrics export --out data/exports/$(date +%F)-events.parquet`

> **p95 formülü notu:** Yukarıdaki `metrics summary` çıktısındaki ve
> loadgen'in `p95_ms`/`p99_ms`/`p50_ms` alanlarındaki yüzdelik hesaplaması
> **en-yakın-sıra (nearest-rank, ceil)** formülünü kullanır — sıralı listede
> `index = min(n-1, ceil(p·n) - 1)`. Bu formül `/stats` (`app/main.py`),
> `belge-gozu metrics summary` (`cli.py`) ve `scripts/loadgen.py`'de birebir
> aynıdır; üçü arasında rapor edilen p95 rakamları doğrudan karşılaştırılabilir.
> Grafana'daki `histogram_quantile(...)` panelleri ayrı bir yöntemdir
> (Prometheus bucket interpolasyonu) — yakın ama farklı bir tahmin üretir.

> **loadgen `requests` notu:** `scripts/loadgen.py` çıktısındaki `requests`
> alanı yalnızca **başarılı** (HTTP 200) çağrıları sayar; başarısız/timeout
> çağrılar ayrı `errors` alanında tutulur. Denenen toplam istek sayısı
> `requests + errors`'tır — bulgu notu yazarken RPS/hata oranı hesaplarken bu
> ayrımı koru (ör. "attempted" = `requests + errors`, "achieved RPS" yalnız
> `requests`'ten hesaplanır).

## Bulgu notu yaz
`docs/research/findings/YYYY-MM-DD-<konu>.md`: koşum künyesi (commit sha, config,
korpus boyutu, yük parametreleri) + gözlemler + figür referansları. Grafana panel
görüntüleri `docs/research/figures/` altına PNG olarak.

## Fiyat varsayımını doğrula
`BG_GEMINI_PRICE_IN_USD_PER_1M` / `BG_GEMINI_PRICE_OUT_USD_PER_1M` varsayılanları
tahminidir. Güncel Gemini Flash fiyatını resmî fiyat sayfasından kontrol et; farklıysa
`.env`'e yaz ve bulgu notunda belirt.

## Kota bütçesi
Gemini ≈20 çağrı/gün. Loadgen ASLA `/ask` ile koşulmaz (bayrak korumalı). CI hiç
çağrı yapmaz.
