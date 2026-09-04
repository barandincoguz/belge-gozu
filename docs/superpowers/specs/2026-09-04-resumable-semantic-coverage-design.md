# Kesintiye Dayanıklı Anlamsal Kapsama Koşumu — Tasarım

Tarih: 2026-09-04

## Sorun

Gerçek 8B dense koşumu yaklaşık bir saat sonra sonuç artefaktı yazmadan sonlandı.
macOS OOM/çökme kaydı yoktu; koşucu tüm 4.222 sayfanın embedding'ini bellekte
biriktirip yalnız kol bittiğinde yazdığı için, dış kesintinin ardından güvenli
bir ilerleme kaydı yoktu.

## Karar

Offline dense koşucu batch bazında devam edilebilir indeks üretir.

- Her biten batch, model dizinindeki geçici `.partial.npy` dosyasına flush edilir;
  eşleşen `progress.json` atomik yazılır.
- İlerleme kaydı model repo/revision, page-id SHA-256, toplam satır sayısı,
  embedding boyutu ve tamamlanan satır sayısını taşır.
- Yeniden başlatma ancak bu kimlikler eşleşirse devam eder; uyuşmazlıkta açık
  hata verir. Tamamlanmış `embeddings.npy` varsa yalnız doğrulanmış final indeks
  kullanılır.
- İstemciye görünür metrik satırı her batch'te tamamlanan/toplam, geçen süre ve
  kalan tahmini süreyi basar.
- Bir invocation için opsiyonel batch bütçesi vardır. Bütçe bitince kontrol noktası
  korunur, dense kolu `in_progress` raporlanır ve süreç başarılı biçimde biter.
  Böylece dış yürütücü süresi, model koşumunun tamamını kaybettirmez.
- Yalnız tamamlanmış dense kolu seçime ve genişletme aşamasına girebilir.

Bu değişiklik online getirim, BM25 reçetesi, eşik veya üretim bayrağına dokunmaz.

## Doğrulama

TDD ile şu durumlar sınanır: kesintiden sonra yalnız eksik batch'lerin
kodlanması; yanlış kimlikli kontrol noktasının reddi; final dosyanın ancak tüm
satırlar yazıldıktan sonra görünmesi; batch bütçesinde `in_progress` raporu.
