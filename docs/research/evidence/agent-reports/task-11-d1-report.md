# Task 11 (D1) — sorgu artırma ablasyon koşucusu

## Ne yapıldı

`scripts/d1_augmentation.py` yazıldı: D1 ablasyonunun runbook betiği. Amaç —
doküman indeksi SABİT tutulurken sorgu tarafındaki augmentation token sayısı
(`QueryFormat.n_suffix`, normalde 10) getirim kalitesini etkiliyor mu, sorusunu
ölçmek.

### Betiğin davranışı
- `--index DIR` (zorunlu): paketli (`PackedIndex`) indeks dizini. `read_manifest`
  ile manifest okunur; manifest varsa `manifest.query_format` ve
  `manifest.model_name` kullanılır, yoksa `CPE_0_3_18` + `vidore/colSmol-500M`'e
  düşülür ve bir UYARI basılır. `manifest.doc_prompt_sha256` yalnız raporlamada
  kullanılır, encode'a hiç geçilmez.
- `--bench PATH` (varsayılan `data/bench/canary_v1.jsonl`, REPO_ROOT'a göre).
- `--only-verified`/`--all` karşılıklı dışlayan grup; varsayılan `--all`
  (taslak dahil) çünkü insan doğrulaması hâlâ sürüyor — aktif mod stdout'a
  basılıyor.
- `--device` (varsayılan `BG_DEVICE` env değişkeni, yoksa `"auto"`).
- `--out PATH` (varsayılan `data/bench/results/<tarih-saat>-<git-sha>-d1.json`).
- İki kol: `with-aug` (manifest formatı aynen) ve `no-aug`
  (`fmt.model_copy(update={"n_suffix": 0, "format_id": fmt.format_id+"-noaug"})`
  — prefix/suffix_token/trailing_newline korunur). TEK `ColSmolEncoder` örneği
  iki kol arasında `encoder.query_format` takas edilerek kullanılıyor (görev
  talimatındaki "swapping is fine and cheaper" seçeneği); takas kodda açıkça
  yorumlanmış durumda.
- Her kol+soru için: `retriever.score_all(q_emb)` ile tam skor, `argsort` ile
  tam sıralama, `rank_of` ile altın sayfa sıraları. İndekste olmayan altın
  sayfalar `missing_gold_pages`'e toplanıp o soru için atlanıyor (cli.py'deki
  `bench oracle` komutuyla aynı desen). `ExhaustiveBinaryRetriever` `encoder=None`
  ile kuruluyor çünkü embedding zaten elle hesaplanıp `score_all`'a geçiliyor
  (yine `bench oracle` deseniyle birebir).
- Çıktı JSON şeması istenen alanların tümünü içeriyor:
  `run_id, git_commit, index_dir, index_manifest, bench, arms.{with-aug,no-aug}.
  {summary:{recall_at,mrr,n}, per_question:[{question_id,gold_ranks}]},
  missing_gold_pages`.
- stdout'a kompakt karşılaştırma tablosu (recall@5, recall@20, mrr, delta) ve
  tek satır SONUÇ (recall@5'te kazanan kol) basılıyor.
- Ağır importlar (`ColSmolEncoder`, `ExhaustiveBinaryRetriever`) yalnız
  `main()` içinde, argparse/manifest/bench-yükleme aşamasından SONRA yapılıyor
  — `--help` model/torch'a hiç dokunmuyor (aşağıda kanıtlandı).

### Saf/testlenebilir kısımlar (model gerektirmiyor)
- `noaug_format(fmt: QueryFormat) -> QueryFormat`
- `summarize_arm(gold_sets: list[set[str]], rankings: list[list[str]], ks) -> dict`
  — Recall@k/MRR toplulaştırmasını `belge_gozu.bench.metrics` üzerinden yapan
  saf fonksiyon; `main()` bunu her kol için çağırıyor.

## Test

`tests/test_d1_augmentation.py` — `scripts/loadgen.py` testindeki desenle
(`sys.path.insert` + doğrudan modül importu) `d1_augmentation`'ı içe aktarıyor,
model/index'e hiç dokunmuyor:
- `test_noaug_format_derivation` (CPE_0_3_18 ve TRAIN_COMPAT_V1 için
  parametrize): `n_suffix==0`, `format_id` ayırt edilebilir ve
  `-noaug` sonekli, `prefix`/`suffix_token`/`trailing_newline` değişmemiş,
  `render()` çıktısı beklenenle eşleşiyor.
- `test_summarize_arm_basic`, `test_summarize_arm_empty`,
  `test_summarize_arm_length_mismatch_raises`.

## Doğrulama kanıtı

```
$ uv run pytest -q -m "not slow"
150 passed, 1 deselected in 1.32s

$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
78 files already formatted
0 errors, 0 warnings, 0 informations

$ uv run python scripts/d1_augmentation.py --help
(exit=0, tam argparse yardımı basıldı)

$ uv run python -X importtime scripts/d1_augmentation.py --help | grep -iE "torch|colpali"
(boş çıktı — --help sırasında torch/colpali_engine importu tetiklenmedi)
```

`git status --porcelain` çalışma boyunca `data/` altında hiçbir değişiklik
göstermedi; script hiç çalıştırılmadı (yalnız `--help`), model yüklenmedi,
index build'e dokunulmadı.

## Controller'ın koşacağı komut (build bitince)

```
uv run python scripts/d1_augmentation.py --index data/index
```

(Farklı bir indeks dizini/bench/cihaz gerekirse `--index`, `--bench`,
`--device`, `--out`, `--only-verified` bayraklarıyla özelleştirilebilir.)

## Endişeler / notlar
- `ExhaustiveBinaryRetriever` yapıcısı bir `meta: pd.DataFrame` bekliyor;
  script bunu `args.index / "meta.parquet"`'ten okuyor (mevcut `bench run`/
  `bench oracle` CLI komutlarıyla aynı varsayım: meta.parquet indeks dizininin
  içinde). Controller'ın gerçek indeks dizininde bu dosyanın bulunduğunu
  varsaydım — build script'i (`PackedIndex.save` + ayrı `meta.to_parquet`)
  bunu zaten üretiyor, bu yüzden risk düşük ama koşumda ilk kontrol edilmesi
  gereken nokta budur.
- `missing_gold_pages` her iki kol için ayrı değil TEK bir üst-seviye liste
  olarak hesaplandı (indeks aynı olduğu için iki kol arasında fark etmez);
  görev tanımındaki JSON şemasıyla birebir uyumlu.
- Gerçek model/index ile uçtan uca koşum yapılmadı (hard constraint gereği);
  bu yüzden sayısal sonuçlar (hangi kolun kazandığı) henüz bilinmiyor —
  yalnız kod yolu ve saf fonksiyonlar test edildi.

## İlgili dosyalar
- `/Users/barandincoguz/Desktop/project-delta/scripts/d1_augmentation.py`
- `/Users/barandincoguz/Desktop/project-delta/tests/test_d1_augmentation.py`
