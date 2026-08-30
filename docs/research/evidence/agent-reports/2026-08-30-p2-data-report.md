# P2 veri raporu — `unans_v1` cevaplanamaz benchmark + dev/test bölmesi

Tarih: 2026-08-30 · Dal: `feat/p0-retrieval-correctness`

Üretim: model ajanı taslağı, **insan onayı YOK** (`source_type: ajan-taslak`, 300/300).

Künye ve dürüstlük kuralları: `data/bench/unans_v1.README.md`.


## 1. Doğrulayıcı çıktısı (`uv run python scripts/validate_unans.py`)

```
========================================================================
unans doğrulama — data/bench/unans_v1.jsonl
========================================================================
korpus: 56 belge, 50 kanun numarası
satır : 300

dilim           satır   çapa/konu  doğrulama                             stil dağılımı             
---------------------------------------------------------------------------------------------------
korpus-disi       200   117 kanun  verified:mechanical:manifest-absence  doga=79 huku=86 madd=35   
anlamsiz-ood       60           -  draft:model-cross-check               doga=34 huku=21 madd=5    
eksik-kanit        40    40 belge  draft:model-cross-check               doga=17 huku=23           
---------------------------------------------------------------------------------------------------
zorluk: {'kolay': 49, 'orta': 161, 'zor': 90}
kaynak: {'ajan-taslak': 300}

split bileşimi (seed='belge-gozu-splits-v1', test_docs=22, RG=2)
küme  dilim                     adet
------------------------------------
dev   anlamsiz-ood                29
dev   canary-cevaplanabilir       26
dev   canary-cevaplanamaz          2
dev   eksik-kanit                 20
dev   korpus-disi                103
test  anlamsiz-ood                31
test  canary-cevaplanabilir       17
test  canary-cevaplanamaz          3
test  eksik-kanit                 20
test  korpus-disi                 97
------------------------------------
dev   TOPLAM cevaplanamaz        154   cevaplanabilir: 26
test  TOPLAM cevaplanamaz        151   cevaplanabilir: 17

TEMİZ — tüm kontroller geçti.
```

Çıkış kodu: `0`


Yükleme kontrolü:

```
$ uv run python -c "from belge_gozu.bench.dataset import load_bench; qs=load_bench('data/bench/unans_v1.jsonl', only_verified=False); print(len(qs))"
300
```


## 2. Split bileşimi

| küme | korpus-dışı | anlamsız | eksik-kanıt | canary cevaplanamaz | **cevaplanamaz toplam** | **canary cevaplanabilir** |
|---|---|---|---|---|---|---|
| dev | 103 | 29 | 20 | 2 | **154** | **26** |
| test | 97 | 31 | 20 | 3 | **151** | **17** |

Hedef ≈150 cevaplanamaz + ≈17 cevaplanabilir → fiilî **151 + 17**. 
Canary cevaplanabilir bölünmesi hedeflenen **26 dev / 17 test** ile birebir.


22 test belgesi (2 RG taraması dahil): `k1136 k1163 k1512 k2828 k2918 k3194 k4054 k4734 k4735 k5188 k5237 k5411 k5490 k6098 k6102 k6183 k6284 k6331 k634 k6698 rg1935a rg1945a`


## 3. Örnek satırlar — `korpus-disi` (200'den 10)

| id | stil | zorluk | çapa | soru |
|---|---|---|---|---|
| `u083` | hukuki | zor | `5953` Basın Mesleğinde Çalışanlarla Çalıştıranlar Arasındaki Münasebetlerin Tanzimi Hakkında Kanun | 5953 sayılı Basın İş Kanunu'na göre gazetecinin kıdem tazminatı hangi esasa göre hesaplanır? |
| `u039` | hukuki | zor | `5809` Elektronik Haberleşme Kanunu | 5809 sayılı Kanun uyarınca kayıt dışı IMEI numarasına sahip cihazların şebekeye erişimi nasıl engellenir? |
| `u102` | hukuki | orta | `5199` Hayvanları Koruma Kanunu | 5199 sayılı Hayvanları Koruma Kanunu'na göre bir hayvana kasten kötü muamele edene hangi yaptırım uygulanır? |
| `u167` | hukuki | orta | `6413` Türk Silahlı Kuvvetleri Disiplin Kanunu | 6413 sayılı Türk Silahlı Kuvvetleri Disiplin Kanunu'na göre uyarma cezası vermeye kim yetkilidir? |
| `u013` | hukuki | orta | `6735` Uluslararası İşgücü Kanunu | 6735 sayılı Uluslararası İşgücü Kanunu kapsamında Turkuaz Kart hangi niteliklere sahip yabancılara verilir? |
| `u019` | hukuki | zor | `3402` Kadastro Kanunu | 3402 sayılı Kadastro Kanunu'na göre kadastro mahkemelerinin görev alanına hangi uyuşmazlıklar girer? |
| `u138` | madde-referansli | orta | `4982` Bilgi Edinme Hakkı Kanunu | 4982 sayılı Bilgi Edinme Hakkı Kanunu'nun 11 inci maddesine göre kurumlar başvuruları kaç iş günü içinde cevaplamak zorundadır? |
| `u025` | hukuki | zor | `4708` Yapı Denetimi Hakkında Kanun | 4708 sayılı Yapı Denetimi Hakkında Kanun'a göre yapı denetim kuruluşlarının sorumluluğu yapı kullanma izninden sonra kaç yıl devam eder? |
| `u094` | dogal | orta | `2464` Belediye Gelirleri Kanunu | Belediye Gelirleri Kanunu'na göre çevre temizlik vergisi kimden ve nasıl tahsil edilir? |
| `u150` | dogal | zor | `5718` Milletlerarası Özel Hukuk ve Usul Hukuku Hakkında Kanun | Milletlerarası Özel Hukuk ve Usul Hukuku Hakkında Kanun'a göre farklı vatandaşlıktaki eşlerin boşanmasında hangi hukuk uygulanır? |

## 4. Örnek satırlar — `anlamsiz-ood` (60'tan 10)

| id | stil | zorluk | — | soru |
|---|---|---|---|---|
| `u204` | hukuki | kolay | — | Bir sözleşmenin sıcaklığı kaç santigrat derece olmalıdır? |
| `u259` | hukuki | orta | — | Rüzgârın esme yönü hangi kanun hükmüne tabidir? |
| `u233` | dogal | kolay | — | Bebeğim 6 aylık, hangi mamayı önerirsiniz? |
| `u214` | dogal | kolay | — | Kanun kanun kanun kanun kanun kanun? |
| `u203` | hukuki | kolay | — | Zamanaşımı süresi kaç santimetredir? |
| `u206` | hukuki | orta | — | İcra müdürlüğünün frekansı kaç hertzdir? |
| `u228` | dogal | kolay | — | İstanbul'da yarın hava nasıl olacak? |
| `u227` | dogal | orta | — | Noterden kaç tane bulut tasdik ettirebilirim? |
| `u205` | dogal | kolay | — | Tapu senedi kaç kalori içerir? |
| `u216` | hukuki | orta | — | Sözleşmenin fotosentez hızı nasıl ölçülür? |

## 5. Örnek satırlar — `eksik-kanit` (40'tan 10)

| id | stil | zorluk | konu | soru |
|---|---|---|---|---|
| `u266` | hukuki | zor | `k3194` | İmar Kanunu'na göre konut parsellerinde uygulanacak TAKS ve KAKS (emsal) değerleri nedir? |
| `u296` | hukuki | orta | `k2004` | İcra ve İflas Kanunu'na göre taşınmaz satışlarında alınacak tellaliye harcı oranı nedir? |
| `u288` | dogal | orta | `k2872` | Çevre Kanunu'na göre gece saatlerinde konut alanlarında izin verilen azami çevresel gürültü kaç desibeldir? |
| `u264` | dogal | orta | `k2828` | Sosyal Hizmetler Kanunu kapsamında engelli yakınına ödenen evde bakım yardımının aylık tutarı ne kadardır? |
| `u268` | hukuki | orta | `k4734` | Kamu İhale Kanunu'na göre 2026 yılında geçerli olacak eşik değerler kaç TL'dir? |
| `u275` | hukuki | zor | `k6102` | Türk Ticaret Kanunu'na göre bir anonim şirketin bağımsız denetime tabi olması için aranan aktif toplamı ve çalışan sayısı ölçütleri nelerdir? |
| `u297` | hukuki | orta | `k6100` | Hukuk Muhakemeleri Kanunu'na göre bilirkişiye ödenecek ücretin tarifedeki tutarı nedir? |
| `u286` | dogal | orta | `k657` | Devlet Memurları Kanunu'na göre hâlen uygulanan memur aylık katsayısı kaçtır? |
| `u294` | dogal | orta | `k6502` | Tüketicinin Korunması Hakkında Kanun'a göre 2026 yılında il tüketici hakem heyetine başvuru için parasal sınır kaç TL'dir? |
| `u299` | hukuki | zor | `k2577` | İdari Yargılama Usulü Kanunu'na göre yürütmenin durdurulması kararı için istenecek teminatın miktarı nasıl belirlenir? |

### eksik-kanıt yokluk kanıtı (örnek notlar)

- **`u263`** — k1512 (71 sayfa) tam metninde 'maktu ücret'=0, 'türkiye noterler birliği tarifesi'=0; buna karşılık 'ücret tarifesi'=5 — kanun tarifeye atıf yapar; rakamlar yıllık Noterlik Ücret Tarifesinde.
- **`u296`** — k2004 (171 sayfa) tam metninde 'tellaliye'=0, 'tellaliye harcı'=0, 'satış bedeli üzerinden'=0 — tellaliye 2464 sayılı Belediye Gelirleri Kanununda düzenlenir, İİK metninde geçmez.
- **`u269`** — k4735 (23 sayfa) tam metninde 'fiyat farkı hesab'=0; buna karşılık 'fiyat farkı'=39, 'esas ve usuller'=9 — kanun fiyat farkı verilebileceğini söyler, formülü Cumhurbaşkanı kararına bırakır.
- **`u279`** — k634 (32 sayfa) tam metninde 'yönetici ücret'=0, 'ücretin alt sınırı'=0; buna karşılık 'yönetici'=73 — kanun ücret ödenebileceğini söyler, tutar/alt sınır belirlemez.
- **`u287`** — k2547 (149 sayfa) tam metninde 'katkı payı tutarları'=0; buna karşılık 'öğrenci katkı payı'=17, 'cumhurbaşkanı kararı'=6 — tutarlar kanunda değil, her yıl Cumhurbaşkanı kararıyla belirlenir.

## 6. Karşılanamayan / kısmen karşılanan kurallar

- **`unanswerable_reason` sözlüğü zaten `eksik-kanit` içeriyordu** (`Slice` de öyle); genişletme gerekmedi. Bunun yerine iki alan gerçekten genişletildi: `source_type` += `ajan-taslak` (satırlar bu değeri istiyordu ama şema kabul etmiyordu) ve `verification_kind` += `mechanical:manifest-absence` (istenen künye değeri şemada yoktu). Sözlük artık testle sabitli.
- **`load_bench`/`load_splits` `str` yol kabul etmiyordu**; istenen doğrulama komutu `Path` yerine `str` geçtiği için `Path(path)` sarmalayıcısı eklendi.
- **korpus-dışı dilim 117 kanuna dayanıyor** (istenen ~200 ayrı kanun değil); 200 soru bu kanunlara 1-3 soru olarak dağıtıldı. Daha fazla kanun, cevap-özü benzersizliğinden ödün vermeden bulunamadı.
- **Mekanik etiketin artık riski nicelenmedi**: absent kanuna dayanan bir sorunun korpustaki başka bir kanunla cevaplanabilme oranı ölçülmedi. Üretimde 8 soru bu gerekçeyle elendi ama kalan oran bilinmiyor — denetleyici turunun örneklemle ölçmesi gerekiyor (README §2, §6).
- **`anlamsiz-ood` ve `eksik-kanit` bilerek `draft`**: kendi kendini doğrulamadım.
