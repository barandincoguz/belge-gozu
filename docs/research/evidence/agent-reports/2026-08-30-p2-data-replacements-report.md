# P2 veri — `abstention_eval_v1.jsonl` yedek parti raporu (u301–u330)

**Tarih:** 2026-08-30 · **Rol:** taslak yazan ajan (drafter) ·
**Kanıt:** yalnız `data/research/page_texts.parquet` (4222 sayfa / 56 belge).
Ağ yok, Gemini yok, alt-ajan yok.

Dokunulan dosyalar: `data/bench/abstention_eval_v1.jsonl` (30 satır eklendi),
`data/bench/abstention_eval_v1.README.md` (§8 eklendi).
`scripts/validate_abstention_eval.py` bilerek DEĞİŞTİRİLMEDİ (bkz. §4).

---

## 1. Özet

| | önce | sonra |
|---|---|---|
| dosya satırı | 300 | **330** |
| `korpus-disi` dilimi | 200 | **230** |
| ayrık çapa (kanun) | 117 | **147** |
| test yakası cevaplanamaz (`rejected` hariç) | 144 | **162** |
| dev yakası cevaplanamaz | 147 | **159** |

G2.1'in %2.0 eşiği için gereken asgari n = 149 → **karşılandı**;
≥155 tampon hedefi **7 satır aşıldı**. Hash-split 30 satırı iki kümeye
dağıttı (test'e 15, dev'e 15 civarı), artış tek yakada toplanmadı.

**Statü:** 30 satırın tamamı `verification_kind="mechanical:manifest-absence"`
ve **insan onaylı değil**; `korpus-disi` diliminin geri kalanı gibi
**checker-2 turunu bekliyor**. Aşağıdaki negatif kanıtlar drafter'ın kendi
taramasıdır, bağımsız denetleyici turu yerine GEÇMEZ.

---

## 2. Tuzak-karşıtı rejim

Her satır sonlandırılmadan önce korpusun tamamı (56 belge) tr-duyarlı
küçültme (`İ→i`, `I→ı`, …) + aksan düzleştirme (`şğüöçı→sguoci`) sonrası
regex ile tarandı. Uygulanan kurallar:

1. **Ücret/tutar sorusu yasak.** `k492` (8) ve (9) sayılı tarifeleri tek
   başına eczane, avukatlık, hastane, laboratuvar, özel okul, sürücü kursu,
   turizm, banka, sigorta, döviz büfesi, bağımsız denetim, gümrük antrepo,
   avcılık, silah, ağız-diş sağlığı ve havayolu ruhsat harçlarını içeriyor
   (u284 pasaport reddinin kaynağı). Hiçbir satır bir bedel sormuyor.
2. **Genel hüküm testi.** Cevabın özü korpustaki bir kanunun genel
   hükmünden çıkarılabiliyorsa satır atıldı (u015/u130/u135 reddinin deseni).
3. **Kök seviyesinde arama.** Çoğul/uzun biçim aramak yetmiyor: `müstahzarlar`
   temiz görünürken `müstahzar` kökü `k492:59`'u yakaladı. Her çapa için
   tekil kök ayrıca tarandı.
4. **Atıf ≠ içerik, ama uyarıdır.** Korpus 5779, 7258, 6772, 5488, 6172,
   6191, 5429, 4646, 6461, 618, 1211, 5302, 442, 3308'in ADINI anıyor;
   bu satırlar ancak aranan içeriğin o atıflarda da bulunmadığı
   doğrulandıktan sonra tutuldu.

### 2.1 Taslakta elenen 6 çapa (bu partinin asıl ürünü)

| Çapa | Elenme kanıtı |
|---|---|
| **1262** İspençiyari ve Tıbbi Müstahzarlar K. | `k492:59` — "(8) SAYILI TARİFE … Müstahzar ruhsatnameleri: Tıbbi ve ispençiyari müstahzarların ticarete çıkarılması için **Sağlık Bakanlığınca** verilecek ruhsatnameler". Ruhsat mercii tarife tablosunda ADIYLA var. Ayrıca `rg1928a:2-3` kanunun "ikinci maddesinin son fıkrası" uyarınca serum/aşı şartlarını ilan ediyor. |
| **1774** Kimlik Bildirme K. | `k5490:27` (Nüfus Hiz. K. m.72) 1774 m.6'yı değiştirirken metni taşıyor: "…tarafından örneğine uygun kimlik belgesi doldurularak **üç gün içinde genel kolluk örgütüne** verilmesi zorunludur". Bildirim süresi korpusta. (m.71 ayrıca 1774'ün 5, 6/a, 6/d, 8, 16'ncı maddelerini kaldırıyor.) |
| **2876** Atatürk Kültür, Dil ve Tarih Yüksek Kurumu K. | `k2709:45` (Anayasa m.134) bileşenleri sayıyor: "Atatürk Araştırma Merkezi, Türk Dil Kurumu, Türk Tarih Kurumu ve Atatürk Kültür Merkezinden oluşan…". |
| **7036** İş Mahkemeleri K. | `k4857:13` (m.20) — "…bir ay içinde işe iade talebiyle, İş Mahkemeleri Kanunu hükümleri uyarınca arabulucuya başvurmak zorundadır. … **iki hafta içinde** iş mahkemesinde dava açılabilir"; `k6102:3` "**altı hafta** … en fazla **iki hafta** uzatılabilir". u140 tipi sınır ihlali riski. |
| **2886** Devlet İhale K. | `k4734` ihale usullerinin kavramsal karşılığını taşıyor; "açık teklif" 0 sayfa olsa da model 4734'ün "açık ihale usulü"nden cevap üretebilir. |
| **4922** Denizde Can ve Mal Koruma K. | `k6102:249` "denize elverişli"yi tanımlıyor; `k492:57` "Denize elverişlilik belgesi" tarife kalemi. |

---

## 3. 30 satır — çapa ve negatif kanıt

### `u301` — 1211 Türkiye Cumhuriyet Merkez Bankası Kanunu

- **soru** (madde-referansli / orta): 1211 sayılı Türkiye Cumhuriyet Merkez Bankası Kanunu'nun 20 nci maddesine göre Banka Meclisi kaç üyeden oluşur ve üyelerin görev süresi kaç yıldır?
- **negatif kanıt:** 'banka meclisi' 0 sayfa, 'para politikası kurulu' 0 sayfa; 1211'e atıf yapan k657:151, k5411:82/97/106 ve k6362:96 yalnız 'hükümleri saklıdır' der, organ yapısını vermez.

### `u302` — 5302 İl Özel İdaresi Kanunu

- **soru** (hukuki / zor): 5302 sayılı İl Özel İdaresi Kanunu'na göre vali, il genel meclisinin hukuka aykırı gördüğü bir kararını kaç gün içinde yeniden görüşülmek üzere meclise iade edebilir?
- **negatif kanıt:** 'vali … yedi gün' 0 sayfa, 'il genel meclisi … toplan' 0 sayfa; 'il genel meclisi' geçen 6 sayfa yalnız alakasız bağlamda (k1136:5 bağdaşan işler, k193:40 huzur hakkı, k3194:21 plan onayı, k657:42/51 kazanılmış hak); 5302'ye atıflar (k3194:1, k657:1, k492:1) yalnız 'aykırılık halinde 5302 uygulanır' dipnotudur.

### `u303` — 442 Köy Kanunu

- **soru** (dogal / orta): 442 sayılı Köy Kanunu'na göre köylülerin yol ve köprü gibi işlerde ücretsiz çalıştırılmasını ifade eden imece yükümlülüğü kimleri kapsar?
- **negatif kanıt:** 'imece' 0 sayfa; 442'ye korpusta yalnız aylık/haciz atfı var (k5510:66 ek m.16, k5510:139 m.74 güvenlik korucusu, k5510:166/167 yaş haddi, k6183:25 köylerde haciz) — mecburi çalışma yükümlülüğü hiçbirinde düzenlenmiyor.

### `u304` — 167 Yeraltı Suları Hakkında Kanun

- **soru** (hukuki / orta): 167 sayılı Yeraltı Suları Hakkında Kanun'a göre yeraltı suyu çıkarmak amacıyla kuyu açacak kişinin hangi kurumdan belge alması gerekir?
- **negatif kanıt:** 'kuyu açma' 0 sayfa ('kuyu' geçen k492:61 'kuyum ticareti'dir); 'yeraltı su' yalnız 3 sayfa ve hiçbiri idari izin rejimi değil (k4721:151 m.756 kaynak mülkiyeti, k2872:21 kirletme idari para cezası, rg1965a:23 kanun adı listesi).

### `u305` — 3573 Zeytinciliğin Islahı ve Yabanilerinin Aşılattırılması Hakkında Kanun

- **soru** (dogal / zor): Zeytinciliğin Islahı ve Yabanilerinin Aşılattırılması Hakkında Kanun, zeytinlik sahalarına yakın yerlerde kurulacak tesisler için en az ne kadar mesafe şartı arar?
- **negatif kanıt:** '3 km|üç kilometre' 0 sayfa; 'zeytinlik' geçen 4 sayfanın tamamı alakasız (k213:124 amortisman, k193:36 zirai işletme büyüklüğü ölçüsü, k2709:59 orman vasfını yitirmiş araziler, rg1965a:65 bütçe tertibi) — koruma mesafesi korpusta yok.

### `u306` — 5553 Tohumculuk Kanunu

- **soru** (anahtar-kelime / orta): 5553 sayılı Tohumculuk Kanunu bitki çeşidi tescili kayıt şartı ticarete arz
- **negatif kanıt:** 'tohumculuk' 0 sayfa, 'çeşit tescil' 0 sayfa; 'tohum' geçen 21 sayfanın tamamı vergi, haciz ve işçi bağlamı (k193, k2004, k4857, k488, k6183, rg1965a) — çeşit kayıt rejimi korpusta yok.

### `u307` — 2238 Organ ve Doku Alınması, Saklanması, Aşılanması ve Nakli Hakkında Kanun

- **soru** (hukuki / zor): 2238 sayılı Organ ve Doku Alınması, Saklanması, Aşılanması ve Nakli Hakkında Kanun'a göre kadavradan organ alınabilmesi için ölüm hali hangi uzmanlardan oluşan kurulca saptanır?
- **negatif kanıt:** 'hekimler kurulu' 0 sayfa, 'tıbbi ölüm' 0 sayfa, 'ölüm halinin tespit' 0 sayfa; k5237:36 (TCK m.91) yalnız organ/doku TİCARETİNİ suç sayar, ölüm tespiti usulünü düzenlemez; 'organ ve doku' geçen tek sayfa k2547:66 ek ödeme oranıdır.

### `u308` — 4631 Hayvan Islahı Kanunu

- **soru** (hukuki / orta): 4631 sayılı Hayvan Islahı Kanunu'na göre suni tohumlama, ovum ve embriyo transferi faaliyetinde bulunacak kişi ve kuruluşların hangi izni alması zorunludur?
- **negatif kanıt:** 'hayvan ıslah' 0 sayfa, 'embriyo' 0 sayfa, 'ovum' 0 sayfa; 'damızlık' geçen tek sayfa k193 (GVK zirai kazanç), 'suni tohumlama' geçen tek sayfa rg1965a:67 taşıt tahsis cetvelidir.

### `u309` — 5258 Aile Hekimliği Kanunu

- **soru** (dogal / orta): Aile Hekimliği Kanunu kapsamında görev yapan aile hekimleri hangi statüde istihdam edilir ve sözleşmeleri kim tarafından yapılır?
- **negatif kanıt:** 'aile hekimliği' geçen 5 sayfanın tamamı yan atıf (k488:31 damga istisnası, k5510:161 prim, k657:124/129/134 kadro) — istihdam statüsü ve sözleşme makamı korpusta yok.

### `u310` — 6023 Türk Tabipleri Birliği Kanunu

- **soru** (hukuki / orta): 6023 sayılı Türk Tabipleri Birliği Kanunu'na göre bir tabibin mesleğini icra edebilmesi için tabip odasına üye olması zorunlu mudur?
- **negatif kanıt:** 'tabip odası' 0 sayfa, 'oda üyeliği' 0 sayfa; 'tabipleri birliği' yalnız k5510 ve k6502'de anılıyor ve üyelik zorunluluğunu düzenlemiyor.

### `u311` — 6235 Türk Mühendis ve Mimar Odaları Birliği Kanunu

- **soru** (madde-referansli / orta): 6235 sayılı Türk Mühendis ve Mimar Odaları Birliği Kanunu'nun 12 nci maddesine göre Birliğin organları hangileridir?
- **negatif kanıt:** 'tmmob' 0 sayfa, 'oda üyeliği' 0 sayfa; Birliğin adı korpusta yalnız k3194:34'te 'görüşü alınacak kuruluş' olarak geçiyor — organ yapısı korpusta yok.

### `u312` — 3308 Meslekî Eğitim Kanunu

- **soru** (hukuki / zor): 3308 sayılı Meslekî Eğitim Kanunu'na göre bir işletmenin öğrencilere beceri eğitimi yaptırmakla yükümlü sayılması için en az kaç personel çalıştırması gerekir?
- **negatif kanıt:** 'beceri eğitimi' 0 sayfa, 'on ve daha fazla personel' 0 sayfa; 3308'e yapılan atıfların tamamı başka konuda (k5510:2/7 çırak sigortalılığı, k193:16 çırak ücreti istisnası, k6331:5 görevlendirme süresi, k3194:3 fen adamı tanımı, k2547:127 staj primi).

### `u313` — 4675 İnfaz Hâkimliği Kanunu

- **soru** (madde-referansli / orta): 4675 sayılı İnfaz Hâkimliği Kanunu'nun 5 inci maddesine göre infaz hâkimliğine şikâyet başvurusu hangi süre içinde yapılmalıdır?
- **negatif kanıt:** 'infaz hakim' 0 sayfa, 'ceza infaz kurumu … şikayet' 0 sayfa; 'şikayet … infaz' geçen tek sayfa k2004:102 (İİK, ihtiyati haczin infazına karşı icra mahkemesine şikayet) — farklı kurum ve farklı süre rejimi.

### `u314` — 4562 Organize Sanayi Bölgeleri Kanunu

- **soru** (hukuki / zor): 4562 sayılı Organize Sanayi Bölgeleri Kanunu'na göre bölgenin kuruluşunda müteşebbis heyet kaç üyeden oluşur?
- **negatif kanıt:** 'müteşebbis heyet' 0 sayfa, '4562' 0 sayfa; 'organize sanayi bölgesi' geçen 2 sayfa alakasız (k213:8 mücbir sebep bölgeleri, k2872:9 atıksu altyapı yönetimi).

### `u315` — 4646 Doğal Gaz Piyasası Kanunu

- **soru** (madde-referansli / orta): 4646 sayılı Doğal Gaz Piyasası Kanunu'na göre dağıtım şirketlerine verilen lisansların süresi en fazla kaç yıldır?
- **negatif kanıt:** 'doğal gaz dağıtım' 0 sayfa, 'lisans süresi' 0 sayfa; 4646'ya atıf yapan k3065:50 (KDV istisnası) ve k488:16 (damga tarifesi) lisans rejimini düzenlemiyor.

### `u316` — 5346 Yenilenebilir Enerji Kaynaklarının Elektrik Enerjisi Üretimi Amaçlı Kullanımına İlişkin Kanun

- **soru** (dogal / zor): Lisanslı bir rüzgâr santralinin Yenilenebilir Enerji Kaynaklarının Elektrik Enerjisi Üretimi Amaçlı Kullanımına İlişkin Kanun kapsamındaki destekleme mekanizmasından kaç yıl süreyle yararlanabileceği nasıl belirlenir?
- **negatif kanıt:** 'yekdem' 0 sayfa, '5346' 0 sayfa; 'yenilenebilir enerji kaynak' geçen tek sayfa k2872:5 (Çevre K., teşvik edilecek araçlar arasında genel anma) — destek süresi ve fiyat listesi korpusta yok.

### `u317` — 5627 Enerji Verimliliği Kanunu

- **soru** (hukuki / orta): 5627 sayılı Enerji Verimliliği Kanunu'na göre hangi büyüklükteki endüstriyel işletmeler enerji yöneticisi görevlendirmekle yükümlüdür?
- **negatif kanıt:** 'enerji yöneticisi' 0 sayfa; 'enerji verimliliği' yalnız k3194 ve k492'de anılıyor; 5627'ye tek atıf k634:18/32'de Kat Mülkiyeti K. m.42'yi DEĞİŞTİREN kanun künyesi olarak geçiyor, yükümlülük eşiği vermiyor.

### `u318` — 6461 Türkiye Demiryolu Ulaştırmasının Serbestleştirilmesi Hakkında Kanun

- **soru** (hukuki / zor): Türkiye Demiryolu Ulaştırmasının Serbestleştirilmesi Hakkında Kanun'a göre demiryolu altyapı işletmecisi olarak faaliyet göstermek isteyen bir şirketin hangi belgeyi alması gerekir?
- **negatif kanıt:** 'demiryolu altyapı' 0 sayfa, 'altyapı işletmecisi' 0 sayfa; 'demiryolu' geçen 23 sayfanın tamamı kamulaştırma/trafik/vergi bağlamı; 6461'e atıf k4734:6 ve k4734:77'de yalnız değiştiren kanun künyesidir.

### `u319` — 618 Limanlar Kanunu

- **soru** (anahtar-kelime / kolay): 618 sayılı Limanlar Kanunu liman idari saha sınırlarını belirlemeye yetkili merci
- **negatif kanıt:** 'liman idari sahası' 0 sayfa, 'liman … saha|sınır' 0 sayfa; 618'e atıf yapan k2872:27 ve k2872:32 yalnız deniz kirliliği CEZA hükümlerine yollama yapıyor, idari saha yetkisini düzenlemiyor.

### `u320` — 5200 Tarımsal Üretici Birlikleri Kanunu

- **soru** (hukuki / orta): 5200 sayılı Tarımsal Üretici Birlikleri Kanunu'na göre bir üretici birliği kurabilmek için en az kaç üreticinin bir araya gelmesi gerekir?
- **negatif kanıt:** 'üretici birliğ' 0 sayfa; 5200'e korpusta tek atıf k1163:23'te 'kurulmuş birlik ve merkez birlikleri de Türkiye Milli Kooperatifler Birliğine ortak olabilir' cümlesindedir — kuruluş için asgari üretici sayısı korpusta yok.

### `u321` — 1567 Türk Parasının Kıymetini Koruma Hakkında Kanun

- **soru** (hukuki / orta): 1567 sayılı Türk Parasının Kıymetini Koruma Hakkında Kanun'a göre kambiyo işlemlerine ilişkin kararları almaya hangi merci yetkilidir?
- **negatif kanıt:** 'kambiyo karar' 0 sayfa; 1567'ye yapılan atıfların tamamı 'hükümleri saklıdır' biçimindedir (k6098:68/69 döviz kirası, k5411:50 BDDK, k6362:26 kripto varlıklar, rg1965a:1/32 arşiv künyesi) — karar alma yetkisi korpusta yok.

### `u322` — 5779 İl Özel İdarelerine ve Belediyelere Genel Bütçe Vergi Gelirlerinden Pay Verilmesi Hakkında Kanun

- **soru** (madde-referansli / zor): 5779 sayılı İl Özel İdarelerine ve Belediyelere Genel Bütçe Vergi Gelirlerinden Pay Verilmesi Hakkında Kanun'un 2 nci maddesine göre belediyelere ayrılan pay, genel bütçe vergi gelirleri tahsilatının yüzde kaçıdır?
- **negatif kanıt:** 5779 korpusta üç belgede ADI ANILIYOR ama oran hiçbir yerde verilmiyor (k3194:5 ve k3194:26 yıkım maliyetinin paylardan kesilmesi, k492:26 matraha dâhil edilmeme, k1319:20 değerli konut vergisi); 'genel bütçe vergi gelirlerinden' geçen 4 sayfada da oran yok.

### `u323` — 5977 Biyogüvenlik Kanunu

- **soru** (dogal / orta): Genetiği değiştirilmiş bir ürünün ithaline izin verilip verilmeyeceğine 5977 sayılı Biyogüvenlik Kanunu'na göre hangi kurul karar verir?
- **negatif kanıt:** 'biyogüven' 0 sayfa, 'biyogüvenlik kurulu' 0 sayfa, 'gdo' 0 sayfa — konu korpusun 4222 sayfasının hiçbirinde geçmiyor.

### `u324` — 2565 Askeri Yasak Bölgeler ve Güvenlik Bölgeleri Kanunu

- **soru** (madde-referansli / zor): 2565 sayılı Askeri Yasak Bölgeler ve Güvenlik Bölgeleri Kanunu'nun 5 inci maddesine göre birinci derece kara askeri yasak bölgesi sınırdan itibaren en fazla kaç metre genişliğinde belirlenebilir?
- **negatif kanıt:** 'birinci derece askeri' 0 sayfa; 'askeri yasak bölge' geçen 3 sayfanın tamamı k3194 (k3194:9, k3194:41, k3194:42) ve hepsi imar/ruhsat istisnası olarak anıyor — bölge derecelerinin mesafe ölçüsü korpusta yok.

### `u325` — 5429 Türkiye İstatistik Kanunu

- **soru** (hukuki / orta): 5429 sayılı Türkiye İstatistik Kanunu'na göre Resmî İstatistik Programı kaç yıllık dönemler için hazırlanır?
- **negatif kanıt:** 'resmî istatistik programı' 0 sayfa, 'istatistiki gizlilik' 0 sayfa; 'resmi istatistik' geçen tek sayfa k6698:17 (KVKK istisnası); 5429'a atıf yapan k5490:19 ve k5490:29 yalnız 'hükümleri uygulanır' der, program dönemini vermez.

### `u326` — 7258 Futbol ve Diğer Spor Müsabakalarında Bahis ve Şans Oyunları Düzenlenmesi Hakkında Kanun

- **soru** (dogal / orta): 7258 sayılı Kanun'a göre spor müsabakalarına dayalı sabit ihtimalli bahis oyunlarını oynatma yetkisi hangi kuruluşa aittir?
- **negatif kanıt:** 'spor toto' 0 sayfa, 'spor müsabaka … bahis' 0 sayfa; 7258 korpusta yalnız SUÇ KATALOĞU olarak anılıyor (k5651:8 erişim engelleme katalog suçu, k5411:48 kart kullanımının önlenmesi) — yetkili teşkilat hiçbir yerde adlandırılmıyor.

### `u327` — 6772 Devlet ve Ona Bağlı Müesseselerde Çalışan İşçilere İlave Tediye Yapılması Hakkında Kanun

- **soru** (hukuki / orta): 6772 sayılı Kanun'a göre devlete ait müesseselerde çalışan işçilere yılda kaç günlük ücret tutarında ilave tediye ödenir?
- **negatif kanıt:** 'ilave tediye' 0 sayfa, 'elli iki gün|52 gün|yirmi altı gün' 0 sayfa; 6772'ye tek anlamlı atıf rg1965a:57'de 'bu statüye tabi personel hakkında 6772 sayılı kanun hükümleri uygulanmaz' cümlesidir — gün sayısı korpusta yok.

### `u328` — 6191 Sözleşmeli Erbaş ve Er Kanunu

- **soru** (madde-referansli / orta): 6191 sayılı Sözleşmeli Erbaş ve Er Kanunu'nun 4 üncü maddesine göre ilk sözleşme süresi en az ve en çok kaç yıldır?
- **negatif kanıt:** 'sözleşmeli erbaş' geçen 5 sayfanın tamamı sigortalılık ve kadro listesi (k5510:42/52/53/56, k657:2) ve 6191 oralarda yalnız DEĞİŞTİREN kanun dipnotu olarak anılıyor — sözleşme süresi korpusta yok.

### `u329` — 5488 Tarım Kanunu

- **soru** (anahtar-kelime / kolay): 5488 sayılı Tarım Kanunu tarımsal destekleme bütçesi gayri safi milli hasıla asgari oran
- **negatif kanıt:** 'gayri safi milli hasıla' 0 sayfa; 'tarımsal destek' geçen 7 sayfa yalnız vergi istisnası ve kooperatif aracılığı (k193:11/12/134, k488:23, k1163:4/17/22); 5488'e tek atıf k488:26 damga istisnasıdır — destekleme bütçesinin oranı korpusta yok.

### `u330` — 6172 Sulama Birlikleri Kanunu

- **soru** (hukuki / orta): 6172 sayılı Sulama Birlikleri Kanunu'na göre birlik meclisi üyeleri nasıl belirlenir?
- **negatif kanıt:** 'birlik meclisi' 0 sayfa; 6172'ye korpusta yalnız iki atıf var (k657:133 geçici madde kaynaklı kadro geçişi, k1163:23 Türkiye Milli Kooperatifler Birliğine ortaklık) — birliğin organ ve seçim rejimi korpusta yok.

---

## 4. Doğrulayıcı: tek ihlal, tek satırlık sabit

`uv run python scripts/validate_abstention_eval.py` şu an **tek ihlalle kırmızı**:

```
İHLAL: 1
  - dilim korpus-disi: 230 satır, beklenen 200
```

Sebep 55. satırdaki modül düzeyi sabit:

```python
SLICE_EXPECT = {"korpus-disi": 200, "anlamsiz-ood": 60, "eksik-kanit": 40}
```

Sayı **dosyadan türetilmiyor**; 190-192. satırlarda `!=` ile tam eşitlik
sınanıyor ve geçersiz kılacak bayrak/ortam değişkeni yok. Betik bu turda
bilerek DEĞİŞTİRİLMEDİ (`scripts/` paralel bir ajanın dosyası; brief de
"düzeltmeye zorlama, bildir" diyordu).

**Doğrulandı:** sabit yalnız BELLEKTE 230'a çekildiğinde kalan ihlal **0**'dır
ve betik `TEMİZ` basar. Yani şema, kimlik (`u301..u330` sıra kontrolü satır
numarasından türediği için sorunsuz), cevaplanamazlık değişmezleri, çapa
yokluğu (numara + ad-token), soru-çapa anma, doğrulama künyesi, yakın-tekrar
(0 çift; mevcut 300 + retrieval_eval'ye karşı) ve split türetimi kontrollerinin
**tamamı bu 30 satır için temiz geçiyor.**

Sahibinin yapması gereken tek satır:

```python
SLICE_EXPECT = {"korpus-disi": 230, "anlamsiz-ood": 60, "eksik-kanit": 40}
```

Yükleme kontrolü zaten geçiyor:

```
uv run python -c "from belge_gozu.bench.dataset import load_bench; \
  print(len(load_bench('data/bench/abstention_eval_v1.jsonl', only_verified=False)))"   # 330
```

---

## 5. Sınırlar (dürüstlük kaydı)

- Negatif kanıtları üreten, satırları YAZAN ajandır. **Drafter = checker**
  bu turda; §3'teki notlar bağımsız doğrulama değildir.
- Kanıt yalnız **metin katmanıdır**; sayfa görüntüsünde olup metne düşmemiş
  bir hüküm bu taramada görünmez.
- Tuzak rejimi (§2) ölçülen ~%12'lik etiket gürültüsünü **düşürmeyi**
  amaçlar; sıfırladığı iddia edilemez. Bu 30 satır için artık risk
  ölçülmemiştir ve dilimin geri kalanıyla aynı kabul edilmelidir.
- Elenen 6 çapa bir **alt sınırdır**: taramanın yakaladıkları. Aynı türde
  yakalanmamış tuzak kalmış olabilir.
