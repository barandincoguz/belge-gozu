# Dense artefakt üretimi Colab'dan Apple Silicon'a taşındı

Tarih: 2026-09-10
Karar: **Colab notebook'u kaldırıldı; üretim yolu `scripts/build_dense_artifacts_local.py`.**

## Neden

Colab hattı ölçüm üretmeden tıkandı. T4 16 GiB'dir; `qwen3-embedding-8b` fp16
ağırlıkları oraya sığmıyor, notebook da bu yüzden 8B için ≥24 GiB istiyordu.
Yerel denemede 24 GiB'lık makinede PyTorch'un Metal bütçesi 17,8 GiB çıktı ve
koşum 4.222 sayfanın **192'sinde** yarım kaldı (`embeddings.partial.npy`).
Donanım 96 GB birleşik bellekli M3 Ultra'ya taşınınca Colab'ın tek gerekçesi —
yeterli GPU belleği — ortadan kalktı.

Colab'ın kalan maliyeti gerçekti: kodun ayrı bir commit'e klonlanması, Drive
checkpoint'i, Secrets üzerinden token, ve oturum kesintisi. Bunların hepsi
yerelde yok.

## Ne değişti

| Kaldırılan | Yerine gelen |
|---|---|
| `notebooks/build_dense_artifacts_colab.ipynb` | `scripts/build_dense_artifacts_local.py` |
| `tests/test_colab_dense_notebook.py` (7 iddia) | `tests/test_build_dense_artifacts_local.py` (11 test) |
| Colab GPU bellek kontrolü (`cuda.get_device_properties`) | `torch.mps.recommended_max_memory()` bütçe kapısı |
| Drive checkpoint | `--artifact-root` altındaki yerel checkpoint (sözleşme aynı) |

**Değişmeyen:** `build_dense_artifacts.py` tek-model primitifi, manifest
sözleşmesi, `validate_dense_artifact`, `push/pull_dense_artifact`, Hub Dataset
kimlikleri ve "yalnız tamamlanmış matris manifestlenir" kuralı. Bu taşıma bir
taşıyıcı değişikliğidir; artefakt kimliği ve doğrulama yolu aynen korundu.

## Taşınan üç ders

1. **Bütçe kapısı önce koşar.** Sığmayan model, tek sayfa kodlanmadan reddedilir.
   Kapıyı besleyen sayı toplam RAM değil, macOS'un PyTorch'a verdiği çalışma
   kümesidir — 24 GiB'lık makinede 17,8 GiB. Yerel yarım koşum tam bu farktan
   doğdu.
2. **Her model ayrı süreçte.** MPS, 8B ağırlıklarını `empty_cache()` sonrası bile
   süreç içinde tam bırakmıyor. Colab'da `subprocess` kullanılmasının nedeni de
   buydu; yerelde de korundu.
3. **Doğrulama yayımdan bağımsız.** `--push` kapalı olsa bile artefakt manifest
   sözleşmesine karşı doğrulanır, çünkü yerelde kalan artefakt da ölçüme girer.

## Sonraki adım

`--source-revision 700ac324fffefb22de02c8e90347b31185547948` ile iki modeli
üret, ardından `scripts/eval_semantic_coverage.py` ile anlamsal kapsama
kollarını ölç. Bu üretim üretim açılışı değildir: dense kanal hâlâ yalnız
bench/CLI yolunda yaşar.
