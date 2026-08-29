---
name: autoresearch
description: Karpathy'nin autoresearch döngüsü — tek metrik, dar kapsam, hızlı doğrulama, tut/geri-al, git=hafıza ile hedefe yönelik otonom deney iterasyonu. Ölçülebilir bir kalite hedefi verildiğinde kullan.
---

# Autoresearch — hedefe yönelik otonom deney döngüsü

Kaynak metodoloji: Andrej Karpathy, `github.com/karpathy/autoresearch` (2026).
Özgün kurulum: ajan yalnız `train.py`'ı değiştirir, her deney SABİT süre koşar,
TEK metrik (`val_bpb`) karşılaştırılır, iyileşme commit'lenir, gerisi atılır;
`program.md` deney yönergesidir, git geçmişi ajanın hafızasıdır. Bu skill o
kuralları herhangi bir ölçülebilir hedefe genelleştirir.

## Değişmez kurallar

1. **TEK birincil metrik.** Deneyler yalnız bu sayıyla kıyaslanır. Yan (guardrail)
   metrikler gerileme kontrolü içindir, karar metriği değildir.
2. **TEK değiştirilebilir yüzey.** Deney dosyası (train.py muadili) dışında hiçbir
   şeye dokunulmaz: harness, veri hazırlığı, test altyapısı, üretim kodu DONUK.
   Harness'ta hata bulunursa döngü DURUR, hata ayrı düzeltilir, taban yeniden ölçülür.
3. **Sabit değerlendirme bütçesi.** Her deney aynı harness'la, aynı veriyle, aynı
   bütçeyle ölçülür. Bütçeyi deney değiştiremez.
4. **Tut/geri-al.** Metrik kesin iyileşirse commit (git = hafıza); eşitlik veya
   gerileme → değişiklik atılır (`git checkout -- <deney dosyası>`). Guardrail
   gerilemesi varsa iyileşme bile atılabilir — eşiği program.md belirler.
5. **Günlük.** Her deney `journal.md`'ye yazılır: hipotez → değişiklik →
   sayılar → karar (KEPT/DISCARDED) → öğrenilen. Sonuç satırları makine-okur
   biçimde `results.jsonl`'a da eklenir. Başarısız deney de kayıttır.
6. **Bir seferde bir değişken.** Bileşik değişiklik yalnız, bileşenleri tek tek
   ölçüldükten sonra denenebilir.
7. **Durma koşulları:** hedef sayıya ulaşıldı; art arda N verimsiz deney
   (program.md belirler, varsayılan 5); ya da bütçe bitti. Durunca özet rapor.

## Döngü

```
0. KURULUM (bir kez): program.md yaz — hedef, metrik, taban, kapsam, yasaklar.
   prepare.py (bir kez koşar, dokunulmaz) + evaluate.py (sabit harness, dokunulmaz)
   + <deney>.py (tek değiştirilebilir dosya, tabanla başlar).
1. TABAN: evaluate.py koş, sayıyı journal'a yaz. Taban bilinen bir referansla
   tutarlı olmalı (harness sağlaması) — değilse harness'ı düzeltmeden deney YOK.
2. HİPOTEZ: journal'daki birikimden bir sonraki en bilgilendirici deneyi seç.
3. DEĞİŞTİR: yalnız deney dosyasını düzenle.
4. ÖLÇ: evaluate.py.
5. KARAR: kural 4. Commit mesajı: `exp(<ad>): <metrik eski>-><yeni> <KEPT|kısa neden>`.
6. → 2'ye dön.
```

## Proje entegrasyonu

Projede `research/program.md` varsa ONU oku ve uygula — hedef/metrik/kapsam/yasaklar
oradadır ve bu skill'in genel kurallarını proje kuralları TAMAMLAR (çelişkide proje
yönergesi + kullanıcı talimatı kazanır; üretim kod tabanının kendi kapı/ilke
düzenine research/ döngüsü dokunamaz). Yoksa: kullanıcıyla hedef ve metriği
netleştir, kurulum adım 0'ı uygula, tabanı ölç, sonra döngüye gir.
