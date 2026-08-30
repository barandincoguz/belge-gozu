"""P2 T1: iddia bölümleme + doğrulayıcı (önbellek/bütçe/ayrıştırma) — AĞ YOK.

Her doğrulayıcı testi stub istemciyle koşar; hiçbir test Gemini'ye gitmez.
"""

import json

import pytest

from belge_gozu.answer.base import Answer, gate2_skip_reason, is_honest_miss
from belge_gozu.answer.verify import (
    PROMPT_VERSION,
    Claim,
    ClaimVerifier,
    EvidenceGate,
    VerifierBudget,
    VerifierCache,
    cache_key,
    parse_verdict,
    segment_claims,
    verify_claim,
)
from belge_gozu.retrieval.types import PageHit


class StubClient:
    """Sayaçlı sahte istemci: kaç kez ÇAĞRILDIĞI testin asıl iddiasıdır."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ['{"verdict": "supported", "gerekce": "kanıtta var"}']
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, schema=None) -> str:
        self.prompts.append(prompt)
        i = min(len(self.prompts) - 1, len(self.replies) - 1)
        return self.replies[i]


def hit(pid: str, n: int = 1) -> PageHit:
    return PageHit(
        page_id=pid,
        score=42.0,
        doc_name="Belge",
        page_no=n,
        image_path=f"images/{pid}.webp",
        source_url="https://example.org",
    )


# --- 1. Bölümleme: Türkçe-farkındalık -----------------------------------------


def test_segment_basic_strips_markers_and_numbers_claims():
    text = (
        "Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir [S1]. "
        "Bir kimsenin birden çok yerleşim yeri olamaz [S1]."
    )
    claims = segment_claims(text)
    assert [c.claim_id for c in claims] == ["c1", "c2"]
    assert "[S1]" not in claims[0].text
    assert claims[0].text == "Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir."
    assert claims[0].cited_sources == [1] and claims[1].cited_sources == [1]


@pytest.mark.parametrize(
    "text",
    [
        "TMK m.19 yerleşim yerini açıkça tanımlar [S1].",
        "TMK m. 19 yerleşim yerini açıkça tanımlar [S1].",
        "Bu kural 320. maddesinde düzenlenmiştir [S1].",
        "Kiracı toplam 2.806,50 TL ödemekle yükümlü tutulur [S1].",
        "Bkz. 4721 sayılı Kanun; hüküm oradadır [S1].",
    ],
)
def test_turkish_abbreviations_and_numbers_do_not_split_sentences(text):
    """Madde kısaltmaları, ordinaller ve tutarlar cümle SINIRI değildir."""
    assert len(segment_claims(text)) == 1, segment_claims(text)


def test_segment_merges_short_fragments():
    claims = segment_claims("Kural budur. Ancak. İstisna 320. maddededir.")
    assert len(claims) == 2  # "Ancak." tek başına iddia olmaz
    assert claims[0].text.endswith("Ancak.")


def test_segment_handles_multiple_markers_in_one_sentence():
    (claim,) = segment_claims("Yıllık izin süresi on dört gündür [S2] [S4].")
    assert claim.cited_sources == [2, 4] and claim.inherited_sources is False


def test_claim_without_own_marker_inherits_the_nearest_preceding_source():
    """review L4: devralma PARAGRAFIN BİRLEŞİMİ değil, EN YAKIN ÖNCEKİ atıftır.

    Birleşim kuralında kendi atfı olmayan — yani gerekçesi en zayıf — cümle
    paragrafın TÜM sayfalarına karşı yargılanıyordu: hem "supported" çıkması
    en kolay iddia o oluyordu (kapının yönüne ters), hem de istemi 3× büyütüp
    kırpma sınırına en çok yaklaşan iddia yine o oluyordu."""
    text = (
        "Birinci iddia yeterince uzun bir cümledir [S1].\n"
        "Bu ara cümle kendi atfını taşımıyor ama öncesine dayanıyor.\n"
        "Üçüncü iddia başka bir sayfaya dayanır [S2].\n"
        "Dördüncü cümle de kendi atfını taşımıyor."
    )
    c1, c2, c3, c4 = segment_claims(text)
    assert c1.cited_sources == [1] and c1.inherited_sources is False
    assert c2.cited_sources == [1] and c2.inherited_sources is True
    assert c3.cited_sources == [2] and c3.inherited_sources is False
    # [1, 2] BİRLEŞİMİ DEĞİL: yalnız en yakın önceki.
    assert c4.cited_sources == [2] and c4.inherited_sources is True


def test_claim_before_any_marker_stays_uncited():
    """Önünde işaretli cümle yoksa iddia ATIFSIZ kalır — kapı onu çağrı
    yapmadan `belirsiz` sayar (kanıtsız iddia doğrulanmış sayılamaz)."""
    text = "Bu giriş cümlesi hiçbir atıf taşımıyor. Asıl dayanak buradadır [S2]."
    c1, c2 = segment_claims(text)
    assert c1.cited_sources == [] and c1.inherited_sources is False
    assert c2.cited_sources == [2]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Evet. Yıllık ücretli izin süresi on dört gündür [S1].", "Evet. Yıllık"),
        ("Hayır. Böyle bir istisna kanunda düzenlenmemiştir [S1].", "Hayır. Böyle"),
        ("Yarg. HGK kararı bu yönde uygulanmaktadır [S1].", "Yarg. HGK"),
    ],
)
def test_leading_short_fragments_merge_forward(text, expected):
    """review H3: baştaki tek kelimelik parça KENDİ iddiası olmamalı.

    Ölçülen bedel: (a) anlamsız parçaya bir kota çağrısı gidiyordu,
    (b) yargıç izole "Evet."i kanıtta bulamayıp `belirsiz` diyor ve İÇERİĞİ
    TAMAMEN DESTEKLENEN yanıt düşüyordu."""
    claims = segment_claims(text)
    assert len(claims) == 1, [c.text for c in claims]
    assert claims[0].text.startswith(expected)
    assert claims[0].cited_sources == [1]


def test_two_word_short_sentence_still_stands_alone():
    """`_is_leading_fragment` ölçütü TEK KELİME + kısa; "Kural budur." iki
    kelimelik gerçek bir iddiadır ve kendi başına kalır."""
    claims = segment_claims("Kural budur. Ancak. İstisna 320. maddededir.")
    assert len(claims) == 2 and claims[0].text.startswith("Kural budur.")


def test_segment_splits_list_items_into_separate_claims():
    text = (
        "Şu koşullar aranır [S1]:\n"
        "- İkametgâh Türkiye'de olmalıdır [S1]\n"
        "- Yaş sınırı ayrıca aranır [S2]"
    )
    claims = segment_claims(text)
    assert len(claims) == 3
    assert claims[2].cited_sources == [2]


def test_segment_empty_and_marker_only_text():
    assert segment_claims("") == []
    assert segment_claims("   \n\n  ") == []
    assert segment_claims("[S1] [S2]") == []


# --- 2. Ayrıştırma dayanıklılığı ---------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"verdict": "supported", "gerekce": "m.19 metni"}', "supported"),
        ('```json\n{"verdict": "unsupported", "gerekce": "yok"}\n```', "unsupported"),
        ('[{"verdict": "belirsiz", "gerekce": "kesik"}]', "belirsiz"),
        ('kısa açıklama... "verdict": "unsupported" ... son', "unsupported"),
        ("{bozuk json", "belirsiz"),
        ("", "belirsiz"),
        ('{"verdict": "kesinlikle-dogru"}', "belirsiz"),
        ("supported", "belirsiz"),  # ÇIPLAK sözcük yetmez: anahtar aranır
    ],
)
def test_parse_verdict_never_crashes_and_never_invents_support(raw, expected):
    assert parse_verdict(raw).verdict == expected


# --- 3. Doğrulama: kanıt, önbellek, bütçe ------------------------------------


def test_claim_without_evidence_is_belirsiz_without_any_call():
    client = StubClient()
    v = verify_claim(Claim(claim_id="c1", text="iddia"), [], client, model="m")
    assert v.verdict == "belirsiz" and client.prompts == []


def test_blank_page_text_counts_as_no_evidence():
    """Metin katmanı boş (taranmış) sayfa: çağrı yapılmaz, karar `belirsiz`."""
    client = StubClient()
    v = verify_claim(Claim(claim_id="c1", text="iddia"), ["", "   "], client, model="m")
    assert v.verdict == "belirsiz" and client.prompts == []


def test_cache_hit_costs_zero_calls_and_returns_same_verdict(tmp_path):
    client = StubClient('{"verdict": "supported", "gerekce": "kanıtta aynen var"}')
    cache = VerifierCache(tmp_path)
    claim = Claim(claim_id="c1", text="Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir.")

    first = verify_claim(claim, ["madde metni"], client, model="m", cache=cache)
    second = verify_claim(claim, ["madde metni"], client, model="m", cache=cache)

    assert len(client.prompts) == 1, "ikinci çağrı önbellekten gelmeliydi"
    assert first.verdict == second.verdict == "supported"
    assert first.cached is False and second.cached is True
    assert second.gerekce == first.gerekce


def test_cache_file_is_kunyeli(tmp_path):
    client = StubClient('{"verdict": "unsupported", "gerekce": "kanıtta geçmiyor"}')
    cache = VerifierCache(tmp_path)
    verify_claim(Claim(claim_id="c1", text="iddia"), ["kanıt"], client, model="mdl", cache=cache)
    (path,) = list(tmp_path.glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"] == "mdl" and payload["prompt_version"] == PROMPT_VERSION
    assert payload["verdict"] == "unsupported" and payload["ts"]
    assert payload["evidence_sha256"] and payload["claim"] == "iddia"


def test_cache_key_separates_model_prompt_claim_and_evidence():
    """p2-reality-audit §4.4: model ve istem sürümü anahtarın PARÇASI."""
    base = dict(model="a", prompt_version="v1", claim_text="c", evidence_sha="e")
    k = cache_key(**base)
    assert k != cache_key(**{**base, "model": "b"})
    assert k != cache_key(**{**base, "prompt_version": "v2"})
    assert k != cache_key(**{**base, "claim_text": "c2"})
    assert k != cache_key(**{**base, "evidence_sha": "e2"})


def test_corrupt_cache_entry_is_ignored_not_fatal(tmp_path):
    cache = VerifierCache(tmp_path)
    claim = Claim(claim_id="c1", text="iddia")
    key = cache_key(model="m", prompt_version=PROMPT_VERSION, claim_text="iddia", evidence_sha="x")
    (tmp_path / f"{key}.json").write_text("{bozuk", encoding="utf-8")
    client = StubClient('{"verdict": "belirsiz", "gerekce": "-"}')
    assert verify_claim(claim, ["kanıt"], client, model="m", cache=cache).verdict == "belirsiz"


def test_budget_degrades_to_belirsiz_and_cache_hits_stay_free(tmp_path):
    """Bütçe dolunca FIRLATMAZ: kalan iddia `belirsiz` olur (şüphede-reddet).

    Fırlatmak `EvidenceGate.evaluate`i yarıda kesip `claims`/`api_attempts`
    alanlarını raporsuz bırakıyordu — `budget.used` ile rapor birbirini
    yalanlıyordu (review L3)."""
    client = StubClient('{"verdict": "supported", "gerekce": "var"}')
    cache = VerifierCache(tmp_path)
    budget = VerifierBudget(max_attempts=1)
    c1 = Claim(claim_id="c1", text="birinci iddia")
    c2 = Claim(claim_id="c2", text="ikinci iddia")

    verify_claim(c1, ["kanıt"], client, model="m", cache=cache, budget=budget)
    assert budget.used == 1 and budget.remaining == 0 and budget.exhausted

    # aynı iddia -> önbellek isabeti -> bütçe HARCANMAZ, tükenmiş olsa bile
    hit = verify_claim(c1, ["kanıt"], client, model="m", cache=cache, budget=budget)
    assert hit.cached and hit.verdict == "supported" and budget.used == 1

    out = verify_claim(c2, ["kanıt"], client, model="m", cache=cache, budget=budget)
    assert out.verdict == "belirsiz" and out.budget_exhausted is True
    assert out.llm_called is False and "bütçesi doldu" in out.gerekce
    assert len(client.prompts) == 1, "tavan GERÇEKTEN çağrıyı kesiyor"


def test_budget_counts_api_attempts_not_verifier_calls():
    """Rotasyon merdiveni bir çağrıyı 3 denemeye çıkarabilir; kota denemeyle
    tükenir, o yüzden bütçe DENEME sayar (review M1)."""

    class RotatingStub:
        """İki anahtar denedi gibi davranan istemci (`api_attempts` yeteneği var)."""

        def __init__(self):
            self._attempts = 0

        def api_attempts(self) -> int:
            return self._attempts

        def generate_json(self, prompt, schema=None):
            self._attempts += 2  # key1 hata + key2 başarılı
            return '{"verdict": "supported", "gerekce": "var"}'

    budget = VerifierBudget(max_attempts=3)
    client = RotatingStub()
    claim = Claim(claim_id="c1", text="iddia")
    v = verify_claim(claim, ["kanıt"], client, model="m", budget=budget)
    assert v.verdict == "supported" and v.attempts == 2
    assert budget.used == 2, "tek çağrı, İKİ deneme"
    assert budget.remaining == 1


def test_unparseable_response_is_not_cached_so_a_later_attempt_can_succeed(tmp_path):
    """review H2: ayrıştırılamayan 200 yanıtı KALICI `belirsiz` olamaz.

    Boş gövde (`finish_reason=MAX_TOKENS`), kesik akış ya da tanınmayan bir
    sarmalama bir YARGI değil ARIZADIR; TTL'siz bir önbellekte kalıcılaşırsa
    o iddiayı içeren her yanıt sonsuza dek kapı 2'de düşerdi."""
    cache = VerifierCache(tmp_path)
    claim = Claim(claim_id="c1", text="doğrulanabilir bir iddia")

    bad = StubClient("")  # boş gövde: ayrıştırılamaz
    first = verify_claim(claim, ["kanıt"], bad, model="m", cache=cache)
    assert first.verdict == "belirsiz" and first.cached is False
    assert list(tmp_path.glob("*.json")) == [], "arıza ÖNBELLEĞE YAZILMAMALI"

    good = StubClient('{"verdict": "supported", "gerekce": "kanıtta var"}')
    second = verify_claim(claim, ["kanıt"], good, model="m", cache=cache)
    assert second.verdict == "supported" and len(good.prompts) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1, "GERÇEK karar önbelleğe girer"


def test_transport_failure_is_not_cached_either(tmp_path):
    class BoomClient:
        def generate_json(self, prompt, schema=None):
            raise RuntimeError("429")

    cache = VerifierCache(tmp_path)
    claim = Claim(claim_id="c1", text="iddia")
    v = verify_claim(claim, ["kanıt"], BoomClient(), model="m", cache=cache)
    assert v.verdict == "belirsiz" and list(tmp_path.glob("*.json")) == []


def test_prompt_version_is_derived_from_the_prompt_text(monkeypatch):
    """review M3: istem/şema değişince önbellek anahtarı KENDİLİĞİNDEN değişir."""
    from belge_gozu.answer import verify as v

    base = v.prompt_fingerprint()
    assert v.PROMPT_VERSION.endswith(base) and v.PROMPT_VERSION.startswith("verify-v1-")
    monkeypatch.setattr(v, "VERIFIER_PROMPT", v.VERIFIER_PROMPT + " (düzenlendi)")
    assert v.prompt_fingerprint() != base


def test_gerekce_regex_keeps_turkish_apostrophes(monkeypatch):
    """review L8: `Kanun'un ...` gerekçesi `Kanun` diye kesilmemeli."""
    raw = 'notlar... "verdict": "supported", "gerekce": "Kanun\'un 19. maddesi bunu yazar" son'
    out = parse_verdict(raw)
    assert out.verdict == "supported" and out.gerekce == "Kanun'un 19. maddesi bunu yazar"


def test_client_failure_becomes_belirsiz_not_an_exception():
    class BoomClient:
        def generate_json(self, prompt, schema=None):
            raise RuntimeError("kota")

    v = verify_claim(Claim(claim_id="c1", text="iddia"), ["kanıt"], BoomClient(), model="m")
    assert v.verdict == "belirsiz" and "başarısız" in v.gerekce


def test_prompt_carries_claim_and_evidence():
    client = StubClient()
    verify_claim(Claim(claim_id="c1", text="İZİN ON DÖRT GÜN"), ["MADDE 53"], client, model="m")
    (prompt,) = client.prompts
    assert "İZİN ON DÖRT GÜN" in prompt and "MADDE 53" in prompt
    assert "supported" in prompt and "belirsiz" in prompt


# --- 4. Kanıt kapısı ----------------------------------------------------------


def _gate(*replies: str, page_texts=None, max_claims: int = 8, cache_dir=None):
    client = StubClient(*replies)
    verifier = ClaimVerifier(
        client=client, model="m", cache=VerifierCache(cache_dir) if cache_dir else None
    )
    texts = page_texts if page_texts is not None else {"a:1": "kanıt A", "b:2": "kanıt B"}
    return EvidenceGate(verifier, texts, max_claims=max_claims), client


def test_gate_passes_when_every_claim_is_supported():
    gate, client = _gate('{"verdict": "supported", "gerekce": "var"}')
    answer = Answer(
        text="Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir [S1].", citations=["a:1"]
    )
    detail = gate.evaluate(answer, [hit("a:1"), hit("b:2")])
    assert detail["demoted"] is False
    assert detail["n_claims"] == 1 and detail["n_supported"] == 1
    assert detail["llm_calls"] == 1 and detail["cache_hits"] == 0
    assert detail["claims"][0]["cited_sources"] == [1]
    assert "kanıt A" in client.prompts[0]


@pytest.mark.parametrize("bad", ["unsupported", "belirsiz"])
def test_a_single_unsupported_or_belirsiz_claim_demotes_the_answer(bad):
    """`belirsiz` DESTEKLENMEMİŞ sayılır (şüphede-reddet, G2.1 yönü)."""
    gate, _ = _gate(
        '{"verdict": "supported", "gerekce": "var"}',
        json.dumps({"verdict": bad, "gerekce": "yok"}),
    )
    answer = Answer(
        text=(
            "Yerleşim yeri sürekli kalma niyetiyle oturulan yerdir [S1]. "
            "Ayrıca kira bedeli her yıl otomatik olarak iki katına çıkar [S2]."
        ),
        citations=["a:1", "b:2"],
    )
    detail = gate.evaluate(answer, [hit("a:1"), hit("b:2")])
    assert detail["demoted"] is True and detail["n_supported"] == 1


def test_gate_maps_each_claim_to_its_own_cited_page_text():
    gate, client = _gate('{"verdict": "supported", "gerekce": "var"}')
    answer = Answer(
        text=(
            "Birinci iddia yeterince uzun bir cümledir [S1]. "
            "İkinci iddia da yeterince uzun bir cümledir [S2]."
        ),
        citations=["a:1", "b:2"],
    )
    gate.evaluate(answer, [hit("a:1"), hit("b:2")])
    assert "kanıt A" in client.prompts[0] and "kanıt B" not in client.prompts[0]
    assert "kanıt B" in client.prompts[1] and "kanıt A" not in client.prompts[1]


def test_truncated_claims_demote_rather_than_silently_pass():
    gate, client = _gate('{"verdict": "supported", "gerekce": "var"}', max_claims=1)
    answer = Answer(
        text=(
            "Birinci iddia yeterince uzun bir cümledir [S1]. "
            "İkinci iddia da yeterince uzun bir cümledir [S1]."
        ),
        citations=["a:1"],
    )
    detail = gate.evaluate(answer, [hit("a:1")])
    assert detail["truncated"] is True and detail["demoted"] is True
    assert len(client.prompts) == 1  # tavan gerçekten çağrıyı kesiyor


def test_gate_counts_cache_hits(tmp_path):
    gate, client = _gate('{"verdict": "supported", "gerekce": "var"}', cache_dir=tmp_path)
    answer = Answer(text="Tek bir yeterince uzun iddia cümlesi [S1].", citations=["a:1"])
    hits = [hit("a:1")]
    assert gate.evaluate(answer, hits)["llm_calls"] == 1
    second = gate.evaluate(answer, hits)
    assert second["llm_calls"] == 0 and second["cache_hits"] == 1
    assert len(client.prompts) == 1


def test_llm_calls_counts_actual_api_calls_not_just_cache_misses():
    """Canlı sondada ölçülen muhasebe hatası: atıfsız iddia ne önbellekten
    gelir ne API'ye gider — `not cached` ile saymak onu bir çağrı sanıyordu."""
    gate, client = _gate('{"verdict": "supported", "gerekce": "var"}')
    answer = Answer(
        text=(
            "Atıfsız ama yeterince uzun bir giriş cümlesi.\n\n"
            "Atıflı ve yeterince uzun ikinci cümle [S1]."
        ),
        citations=["a:1"],
    )
    detail = gate.evaluate(answer, [hit("a:1")])
    assert detail["n_verified"] == 2 and len(client.prompts) == 1
    assert detail["llm_calls"] == 1 and detail["cache_hits"] == 0
    assert detail["claims"][0]["verdict"] == "belirsiz" and detail["demoted"] is True


def test_leading_evet_no_longer_demotes_a_fully_supported_answer():
    """review H3 uçtan uca: "Evet." açılışı artık ne kota yakar ne yanıt düşürür.

    Düzeltmeden ÖNCE ölçülen: n_claims=2, n_supported=1, llm_calls=2,
    demoted=True — içeriği tamamen desteklenen yanıt düşüyordu."""
    gate, client = _gate('{"verdict": "supported", "gerekce": "kanıtta var"}')
    answer = Answer(text="Evet. Yıllık ücretli izin süresi on dört gündür [S1].", citations=["a:1"])
    detail = gate.evaluate(answer, [hit("a:1")])
    assert detail["n_claims"] == 1 and detail["n_supported"] == 1
    assert detail["demoted"] is False
    assert detail["llm_calls"] == 1 and len(client.prompts) == 1


def test_gate_creates_a_fresh_per_request_budget_by_default():
    """`budget` verilmezse HER `evaluate` kendi tavanını alır (serve yolu)."""
    gate, client = _gate('{"verdict": "supported", "gerekce": "var"}')
    gate.max_attempts = 1
    answer = Answer(
        text=(
            "Birinci iddia yeterince uzun bir cümledir [S1]. "
            "İkinci iddia da yeterince uzun bir cümledir [S1]."
        ),
        citations=["a:1"],
    )
    hits = [hit("a:1")]
    first = gate.evaluate(answer, hits)
    assert first["budget_exhausted"] is True and first["api_attempts"] == 1
    assert first["budget_used"] == 1 and first["budget_max_attempts"] == 1
    assert first["demoted"] is True
    # İkinci istek TAZE bütçeyle başlar (tavan istek başına, süreç başına değil).
    second = gate.evaluate(answer, hits)
    assert second["budget_used"] == 1 and second["cache_hits"] == 0
    assert len(client.prompts) == 2


def test_gate_shares_a_run_wide_budget_when_one_is_given():
    """`budget` verilirse ömrü çağırana aittir (CLI harness: koşum boyunca)."""
    from belge_gozu.answer.verify import VerifierBudget

    gate, client = _gate('{"verdict": "supported", "gerekce": "var"}')
    gate.budget = VerifierBudget(max_attempts=1)
    answer = Answer(text="Tek bir yeterince uzun iddia cümlesi [S1].", citations=["a:1"])
    hits = [hit("a:1")]
    assert gate.evaluate(answer, hits)["budget_exhausted"] is False
    second = gate.evaluate(answer, hits)
    assert second["budget_exhausted"] is True and second["demoted"] is True
    assert len(client.prompts) == 1, "paylaşılan bütçe koşum boyunca tükenmiş kalır"
    # L3: yarıda kesilmiyor -> `claims` ve muhasebe HER ZAMAN raporda.
    assert len(second["claims"]) == 1 and second["api_attempts"] == 0


def test_gate2_skip_reasons():
    from belge_gozu.answer.base import ABSTAIN_TEXT, HONEST_MISS_MARKER

    assert gate2_skip_reason(Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)) == "abstained"
    miss = Answer(text=f"Cevabı {HONEST_MISS_MARKER}.", citations=[])
    assert is_honest_miss(miss) and gate2_skip_reason(miss) == "honest_miss"
    assert gate2_skip_reason(Answer(text="atıfsız yanıt", citations=[])) == "no_citations"
    assert gate2_skip_reason(Answer(text="yanıt [S1].", citations=["a:1"])) is None
