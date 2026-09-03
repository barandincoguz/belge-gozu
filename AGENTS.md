# Belge-Gözü çalışma rehberi

Belge-Gözü, Türk mevzuatı üzerinde kanıtlı soru-cevap üreten bir RAG sistemidir.
Bir değişikliğin doğru sayılması için yalnız kodun çalışması yetmez: ölçümün,
indeks kimliğinin ve cevap güvenlik kapısının da aynı yapılandırmayı temsil etmesi gerekir.

## Depo haritası

- `src/belge_gozu/`: uygulama kodu; `retrieval/` getirim, `answer/` cevap ve
  kalibrasyon, `index/` indeksleme, `bench/` kıyas veri modeli.
- `scripts/`: tekrar üretilebilir değerlendirme, doğrulama ve raporlama araçları.
- `tests/`: pytest testleri; gerçek model kullananlar `slow` işaretlidir.
- `data/bench/`: sürümlü değerlendirme kümeleri, split'ler ve koşum artefaktları.
- `docs/research/`: kararlar ve kanıtlar; `docs/superpowers/`: onaylanmış tasarım ve planlar.

## Günlük çalışma

- Önce ilgili çağrı yolunu, indeks/kayıt şemalarını ve mevcut testleri `rg` ile bulun.
- Küçük ve geri alınabilir değişiklikler yapın. Kullanıcının sahip olduğu ilgisiz,
  izlenmeyen dosyalara dokunmayın.
- Kaynak dosyalarını `apply_patch` ile düzenleyin. Biçimlendirme ve büyük mekanik
  dönüşümler bunun istisnasıdır.
- Davranış değişikliğinde önce başarısız bir hedef test yazın; ardından en dar
  uygulamayı yapın. Yalnız yeniden adlandırma veya dosya taşıma gibi mekanik işler
  bunun dışındadır.
- Bir değerlendirme sonucunu değiştiren işte sonuçla birlikte komutu, veri kümesi
  sürümünü, indeks/recipe kimliğini ve commit'i kaydedin.

## Değerlendirme adları — zorunlu sözleşme

Kısa, metaforik veya belirsiz takma adlar kullanmayın. Yeni adlar işlevi doğrudan
anlatmalı; CLI seçenekleri `kebab-case`, Python değişkenleri `snake_case` olmalıdır.

| Amaç | Onaylı ad |
| --- | --- |
| Cevaplanabilir soruların getirim değerlendirmesi | `retrieval_eval_vN` |
| Cevaplanamayan soruların çekimserlik değerlendirmesi | `abstention_eval_vN` |
| Gerçek-model sıralama regresyonu | `retrieval_regression` |
| Beklenen sıralama artefaktı | `retrieval_regression_expectations.json` |
| Getirim veri kümesi inceleme aracı | `verify_retrieval_eval.py` |
| Çekimserlik veri kümesi doğrulayıcısı | `validate_abstention_eval.py` |

Bir terimi veya dosyayı yeniden adlandırırken tüm çağrı yüzeyini güncelleyin:

1. `git mv` ile izlenen dosyayı taşıyın.
2. Kaynak, test, CLI yardımı, betikler, veri artefaktları ve dokümantasyondaki
   başvuruları güncelleyin.
3. Eski adın hem içerikte hem dosya yolunda kalmadığını `rg` ile doğrulayın.
4. İçe aktarmaları, varsayılan yolları ve CLI yardımını çalıştırın.

## Ölçüm ve üretim kuralları

- Üretimde bir eşiği, farklı skor ölçeğinde ölçülmüş başka bir kanaldan devralmayın.
  Yeni bir kanal, kendi ölçeğinde kalibre edilmiş eşik ve eşleşen indeks kimliği ister.
- `recipe_fingerprint`, indeks revizyonu ve kalibrasyon artefaktı birlikte ele alınır;
  bunlardan biri değişirse eski kalibrasyon geçerli sayılmaz.
- Değerlendirme aracını deney nesnesi hâline getirmeyin. Eşik seçimi geliştirme
  bölmesinde yapılır; nihai sayı yalnız ayrılmış test bölmesinden gelir.
- Gerçek model ve ağ kullanan komutları açıkça belirtin; model önbelleğini veya
  kullanıcı tarafından indirilen artefaktları silmeyin.

## Doğrulama

Değişikliğin yüzeyine uygun en dar testten başlayın; bitirmeden önce en az şunları çalıştırın:

```bash
uv run pytest <ilgili-testler> -q
uv run ruff check .
uv run pyright
git diff --check
```

Veri kümesi veya CLI değiştiyse ilgili doğrulayıcının `--help` çıktısını ve veri kümesi
doğrulamasını da çalıştırın. Gerçek model koşumları pahalı veya beklemedeyse bunu açıkça
raporlayın; tahmini sonucu ölçülmüş gibi sunmayın.
