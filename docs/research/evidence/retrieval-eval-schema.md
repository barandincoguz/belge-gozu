# RetrievalEval taslak satırı şeması (JSONL — her satır tek JSON nesnesi)

Zorunlu 17 alan (hepsi her satırda bulunmalı):

| Alan | Tip/Değerler | Kural |
|---|---|---|
| question_id | str | sana verilen önekle: c101, c102... (benzersiz) |
| question | str | doğal Türkçe soru |
| query_style | "dogal" \| "hukuki" \| "madde-referansli" \| "anahtar-kelime" | sorunun üslubu |
| answerable | bool | taslaklarında her zaman true (cevaplanamazları controller yazar) |
| gold_doc_ids | list[str] | ör. ["k4721"] — gold_page_ids'in dok önekleriyle tutarlı OLMAK ZORUNDA |
| gold_page_ids | list[str] | "dok:sayfa" ör. ["k4721:4"] — YALNIZ gerçekten OKUDUĞUN ve cevabı içeren sayfa(lar) |
| gold_article_ids | list[str] | "k4721:m19" (madde) / "k4721:gm2" (geçici); sayfada madde no görünmüyorsa [] |
| minimal_evidence_spans | list[str] | sayfadan BİREBİR kısa alıntı (cevabı taşıyan cümle/ifade) |
| reference_answer | str | 1-2 cümlelik doğru cevap (madde referanslı) |
| slice | sana atanan dilim değeri | aşağıdaki tanımlara uy |
| difficulty | "kolay" \| "orta" \| "zor" | karışım hedefle |
| source_type | "ajan-taslak-insan-onayli" | sabit |
| requires_visual | bool | tablo/tarama sorularında true, düz metinde false |
| requires_multi_hop | bool | taslaklarında false |
| unanswerable_reason | null | taslaklarında daima null |
| verified_by | "" | boş bırak (insan doğrulayacak) |
| verification_status | "draft" | sabit |

## Dilim tanımları

- **dogrudan-madde**: cevabı tek maddede açıkça yazan, terimleri maddedekiyle örtüşen soru.
- **paraphrase**: günlük dille sorulmuş; maddenin kelimelerini KULLANMAYAN soru
  (ör. madde "yerleşim yeri...niyetiyle oturduğu yerdir" → soru "ikametgah neresi sayılır?").
- **madde-numarali**: soru kanun adı/numarası + madde numarası içerir
  (ör. "4857 sayılı İş Kanunu madde 17'ye göre ihbar süreleri nelerdir?") — o maddeyi
  gerçekten okuduğun sayfa gold olur.
- **ayni-kanun-hard-negative**: aynı kanunda birbirine ÇOK benzeyen 2+ madde varken
  yalnız birini hedefleyen soru (ör. TMK'da farklı yerleşim yeri maddeleri). Sorunun
  ayırt edici ayrıntısı, hedef maddeye özgü olmalı; benzer maddelerin sayfa/madde
  numaralarını reference_answer'a DEĞİL, soru satırının sonuna eklenmiş
  `"_hard_negatives": ["k4721:m20", ...]` alanına yaz (şema dışı ekstra alan — loader
  yok sayar, insan doğrulamada işe yarar).
- **capraz-kanun-terim**: aynı hukukî terim birden çok kanunda geçerken, sorunun
  bağlamı tek kanunu işaret etmeli (ör. "vergi hukukunda zamanaşımı süresi" → VUK).
  Bağlam kelimesi olmadan yazma.
- **tablo-layout**: cevap bir TABLO/tarife/cetvel hücresinden okunuyor; requires_visual=true.
- **tarihi-tarama**: taranmış Resmî Gazete sayfasından okunabilen somut bilgi
  (karar/kanun numarası, tarih, başlık); requires_visual=true; gold_article_ids=[].

## Mutlak kurallar

1. gold_page_ids'e görmediğin sayfa YAZMA. Cevap iki sayfaya yayılıyorsa ikisini de yaz.
2. minimal_evidence_spans sayfadaki metinle birebir olmalı (görüntüden oku).
3. Bir sayfa okunaksız/boşsa atla; o sayfadan soru üretme.
4. Sorular birbirinden bağımsız ve çeşitli olsun (aynı maddeye iki soru yazma).
5. Çıktın YALNIZ JSONL satırları olsun — açıklama, markdown, kod bloğu YOK.

## Örnek satırlar

{"question_id": "c000", "question": "Kayın hısımlığı evliliğin sona ermesiyle ortadan kalkar mı?", "query_style": "dogal", "answerable": true, "gold_doc_ids": ["k4721"], "gold_page_ids": ["k4721:4"], "gold_article_ids": ["k4721:m18"], "minimal_evidence_spans": ["Kayın hısımlığı, kendisini meydana getiren evliliğin sona ermesiyle ortadan kalkmaz."], "reference_answer": "Hayır; TMK m.18 uyarınca kayın hısımlığı, kendisini meydana getiren evliliğin sona ermesiyle ortadan kalkmaz.", "slice": "dogrudan-madde", "difficulty": "kolay", "source_type": "ajan-taslak-insan-onayli", "requires_visual": false, "requires_multi_hop": false, "unanswerable_reason": null, "verified_by": "", "verification_status": "draft"}
{"question_id": "c001", "question": "Bir öğrencinin okuduğu şehir ikametgahı sayılır mı?", "query_style": "dogal", "answerable": true, "gold_doc_ids": ["k4721"], "gold_page_ids": ["k4721:4"], "gold_article_ids": ["k4721:m22"], "minimal_evidence_spans": ["Bir öğretim kurumuna devam etmek için bir yerde bulunma", "yeni yerleşim yeri edinme sonucunu doğurmaz"], "reference_answer": "Hayır; TMK m.22'ye göre öğretim kurumuna devam için bir yerde bulunma yeni yerleşim yeri edinme sonucunu doğurmaz.", "slice": "paraphrase", "difficulty": "orta", "source_type": "ajan-taslak-insan-onayli", "requires_visual": false, "requires_multi_hop": false, "unanswerable_reason": null, "verified_by": "", "verification_status": "draft"}
