# Autoresearch döngüsü: vitrin sorgusu teşhisi ve metin kanalı reçetesi (R@5 0.2326 → 0.8140)

- **Tarih:** 2026-08-29
- **Tetik:** Kullanıcı raporu — UI örnek chip'lerindeki iki soru ("TMK'ya göre yerleşim
  yeri nasıl tanımlanır?", "İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır?")
  canlı demoda "verilen sayfalarda bulamadım" alıyor.
- **Yöntem:** Karpathy autoresearch metodolojisi (`github.com/karpathy/autoresearch`) —
  proje skill'i `.claude/skills/autoresearch/SKILL.md`, program `research/program.md`.
  Tek metrik (retrieval_eval-answerable n=43 R@5), tek değiştirilebilir dosya
  (`research/retrieve.py`), donuk harness (`evaluate.py`), tut/geri-al, git=hafıza.
- **Artefaktlar:** `research/journal.md` (deney kayıtları), `research/results.jsonl`
  (künyeli sayılar), `data/bench/results/showcase-queries-diagnosis.json` (teşhis),
  `data/research/` (yeniden-üretilebilir hazırlık: sayfa metinleri + görsel skor matrisi).

## 1. Teşhis: sistem dürüst, retrieval ıskalıyor; cevaplar korpusta VAR

Canlı `/ask` yeniden üretimi: iki soruda da abstain YOK (top1 0.6682 / 0.6080 > 0.58),
Gemini çağrılıyor, top-5'te gold olmadığı için dürüstçe "bulamadım" diyor (sahte atıf
fallback'i kaldırıldığı için bu doğru davranış). Korpus kontrolü:

- TMK m.19 tanımı `k4721:4`'te birebir ("Yerleşim yeri bir kimsenin sürekli kalma
  niyetiyle oturduğu yerdir") — int8 görsel rank **664**; `k4721:1` (kapak) rank **1**.
- İş K. m.53 süre cetveli `k4857:28`'de birebir ("a) ... ondört günden, b) ... yirmi
  günden, c) ... yirmialtı günden") — görsel rank **137**; İş K.'nın ilk sayfası rank 17.

Desen: **model doğru dokümanı buluyor, doğru sayfayı bulamıyor** — kanun adı görsel
kanalda kapak/başlık sayfalarını çekiyor, madde içeriği kayboluyor (colSmol'un
İngilizce-ağırlıklı eğitiminin Türkçe uzun sorgudaki bilinen sınırı; P1 gerekçesi D3).

## 2. Döngü seyri (7 deney; karar kuralı: R@5 ↑ VE R@20/visual-R@5 gerilemez)

| # | Deney | R@5 | R@1 | R@20 | MRR | vis-R@5 | chip1 | chip2 | Karar |
|---|---|---|---|---|---|---|---|---|---|
| 0 | taban: yalnız görsel (int8 üretim) | 0.2326 | 0.093 | 0.302 | 0.149 | 0.375 | 664 | 137 | — |
| 1 | + BM25 (yalnız metin, PDF katmanı) | **0.6744** | 0.512 | 0.791 | 0.610 | 0.750 | 8 | 2 | KEPT |
| 2 | eşit-RRF(görsel, BM25) k=60 | 0.3953 | 0.209 | 0.744 | 0.316 | 0.500 | 17 | 2 | DISCARDED |
| 3 | BM25 + F5 ön-ek kırpması (Can vd.) | **0.7674** | 0.488 | 0.884 | 0.621 | 1.000 | 8 | 3 | KEPT |
| 4 | + bigram shingle | 0.6279 | 0.512 | 0.837 | 0.570 | 0.500 | 9 | 2 | DISCARDED |
| 5 | + sabit Türkçe işlev-kelime listesi | 0.7674 | 0.512 | **0.907** | 0.625 | 1.000 | **4** | 2 | KEPT* |
| 6 | + mutlak doküman-adı bölümleme | 0.7907 | 0.465 | 0.837 | 0.607 | 0.875 | 2 | 2 | DISCARDED (veto) |
| 7 | + pencere-içi (top-20) yönlendirme | **0.8140** | 0.512 | 0.907 | 0.652 | 1.000 | **2** | **2** | KEPT |

\* R@5 eşit; ikincil-kanıt kuralıyla tutuldu (R@20+MRR+chip1 birlikte iyileşti;
kural güncellemesi journal #5'te şeffaf).

Taban sağlaması: taban 0.2326 = A2 oracle int8 kolu 0.233 (10/43) birebir; programın
ilk taslağındaki 0.116 eski 1-bit üretimin sayısıydı (int8 geçişi tek başına 2× getirmişti).

## 3. Kazanan reçete (exp7, `research/retrieve.py` @ HEAD)

PDF metin katmanı üzerinde **BM25** (k1=1.5, b=0.75) + `tr_lower` + `\w+`/≥2 harf +
**sabit Türkçe işlev-kelime listesi** (retrieval_eval'ye ayarsız; "zaman"/"iş" gibi içerik-riskli
kelimeler bilinçli dışarıda) + **F5 ön-ek kırpması** (Türkçe eklemeli; Can vd. Turkish IR) +
**doküman-adı pencere-içi yönlendirmesi**: her dokümanın adı kendi 1. sayfa başlık
satırından türetilir (elle tablo yok → sızıntı yok), adın jenerik-dışı TÜM token'ları
sorguda geçiyorsa yalnız BM25 top-20 penceresinin İÇİ yeniden sıralanır (aday kümesi
değişmez → R@20 yapısal korunur).

## 4. Bilimsel bulgular (makale malzemesi)

1. **Metin kanalı görselin ~3 katı** (0.674 vs 0.233 R@5): Türkçe hukuk korpusunda
   birebir terim eşleşmesi, İngilizce-eğitimli görsel geç-etkileşimden çok daha güçlü.
   RG taramaları dahil 4222 sayfanın 4221'inde metin katmanı var (kaynak OCR'lı) —
   requires_visual soruların 8/8'i metinden bulunuyor.
2. **Eşit-ağırlık RRF zarar veriyor** (0.674 → 0.395): zayıf kanalın gürültüsü
   (kapak sayfaları) güçlü kanalın 21 tekil kazanımından 12'sini düşürüyor. "RRF önce"
   ilkesi ÖLÇÜLDÜ ve bu bench'te reddedildi; füzyon ancak asimetrik olabilir.
3. **F5 sonrası görselin @5 benzersiz katkısı SIFIR soru** (öncesinde 2: c202, c205);
   mükemmel-füzyon tavanı bile metin-only'nin +2'siydi. Görselin rolü P1'de "tablo/
   tarama fallback + metinsiz sayfalar" olarak yeniden çerçevelenmeli.
4. F5 kırpması +9.3 puan (recall tarafında); bigram karışımı −14 puan (F5-kırpık
   kalıp bigram'ları başlık/atıf sayfalarını şişiriyor).
5. Stoplist R@5'i değil derin sıraları düzeltiyor (R@20 0.884→0.907, chip1 8→4):
   soru-kalıbı kelimeleri en çok orta-sıra karışıklığı üretiyor.
6. Kural sinyalleri (doküman adı) küresel bölümleme yerine pencere-içi düzeltme
   olarak güvenli: guardrail'le hizalı pencere, gerileme sınıfını yapısal kapatıyor.

## 5. Kalan ıskalar (8/43) ve sınırlar

`c101(6) c104(17) c111(62) c202(12) c206(250) c207(18) c209(39) c214(27)`.
Notlar: c202 görselin eski tekil kazanımıydı (metin-only reçetede kayboldu — füzyon
tavanının maliyeti, bilinçli); c214 madde-numaralı sorgu ("madde 7") — madde-numarası
kanalı denenmedi (aday: pencere-içi "Madde N" başlık eşleşmesi); c206 (KVKK kısaltması
ad-yönlendirmesine yakalanmıyor) + c209 (Anayasa, ad token'ı kısmi) yönlendirme
kapsamı dışı. Sınırlar: retrieval_eval 45/48 model-cross-check (insan-doğrulanmış sayılmaz);
dogrudan-madde dilimi sözcüksel örtüşmeyle BM25 lehine; sayılar araştırma
harness'ından — üretim entegrasyonu ölçümü P1'de yeniden yapılır.

## 6. P1'e devir

Reçete, P1 planının F1/F2 görevlerine ölçülmüş girdi olarak devredilir: sayfa-metin
indeksi üretim artefaktı olur (manifest'li), serve'e metin kanalı + pencere-içi
yönlendirme entegre edilir (SDD kapı düzeniyle), görsel kanalın rolü bulgu 3'e göre
yeniden tanımlanır, bench run + canlı doğrulama üretim sayısını teyit eder. Üretim
bugün hâlâ yalnız-görsel: chip soruları demoda P1 entegrasyonuna kadar "bulamadım"
almaya devam eder.
