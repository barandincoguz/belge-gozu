import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from belge_gozu.answer.base import ERROR_TYPES, HONEST_MISS_MARKER, AnswererError
from belge_gozu.answer.gemini import (
    GEMINI_MAX_ATTEMPTS,
    GEMINI_RETRY_BACKOFF_S,
    GEMINI_TIMEOUT_S,
    GEMINI_TOTAL_BUDGET_S,
    KEY_LABELS,
    NON_ROTATABLE_ERROR_TYPES,
    RETRYABLE_ERROR_TYPES,
    SYSTEM,
    GeminiAnswerer,
    GeminiClient,
    GenResult,
    KeySlot,
    RotatingGeminiClient,
    StickyKeyIndex,
    build_gemini_client,
    build_prompt,
    classify_error,
    source_marker,
)
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import collecting


def hit(pid: str) -> PageHit:
    return PageHit(
        page_id=pid,
        score=50.0,
        doc_name="Türk Borçlar Kanunu",
        page_no=12,
        image_path=f"images/{pid}.webp",
        source_url="https://example.org",
    )


def test_prompt_references_interleaved_markers_and_not_a_source_list():
    """B14: kaynak künyeleri artık istemde DEĞİL, görüntülerin arasında.

    İstem yalnız kuralı ("etiket kendinden SONRAKİ görüntüye aittir") ve sayfa
    sayısını söyler; `[S1] doküman, sayfa N` listesi kaldırıldı çünkü aynı
    bilgi `build_contents` ile her görüntünün ÖNÜNE konuyor. İstemde ikinci bir
    kopya kalsaydı iki künye sessizce ayrışabilirdi.
    """
    p = build_prompt("kira artışı sınırı nedir?", [hit("k6098:12"), hit("k6098:13")])
    assert "kira artışı" in p
    assert "[S1]" in p and "[S2]" in p  # etiket aralığı istemde ilan edilir
    assert "2 sayfa görüntüsü" in p
    assert "Türk Borçlar Kanunu" not in p  # künye listesi İSTEMDE DEĞİL


def test_citations_parsed_from_response():
    client = MagicMock()
    client.generate.return_value = GenResult(text="Kira artışı TÜFE ile sınırlıdır [S2].")
    ans = GeminiAnswerer("gemini-2.0-flash", "key", client=client)
    a = ans.answer("soru", [hit("k6098:12"), hit("k6098:13")], image_loader=lambda p: b"img")
    assert a.citations == ["k6098:13"] and not a.abstained


def test_no_marker_means_no_citation():
    # Metinde [Sn] işareti yoksa atıf da yoktur. Eskiden burada top-1 sayfayı
    # otomatik atıf yapan bir fallback vardı: model "verilen sayfalarda
    # bulamadım" dediğinde bile UI'da uydurma bir "dayanak" çipi çıkıyordu.
    client = MagicMock()
    client.generate.return_value = GenResult(text="Verilen sayfalarda bulamadım.")
    ans = GeminiAnswerer("gemini-2.0-flash", "key", client=client)
    a = ans.answer("soru", [hit("k6098:12")], image_loader=lambda p: b"img")
    assert a.citations == []


def test_answerer_constructs_without_key_no_sdk_touch():
    # Boş anahtarla bile kurulum patlamamalı (keyless boot): SDK'ya hiç dokunulmaz,
    # gerçek genai.Client yalnızca generate() çağrıldığında oluşturulur.
    GeminiAnswerer("gemini-2.0-flash", "")


def test_client_generate_raises_on_empty_key():
    # Boş anahtar hatası artık generate() zamanında patlar (ağa hiç çıkmadan) —
    # AskService'in degradation guard'ı bunu SERVICE_ERROR_TEXT'e çevirir. Kasıtlı
    # geniş yakalama: SDK'nın attığı tam exception sınıfı sürüme göre değişebilir,
    # guard zaten Exception'ı genel yakalıyor (base.py) — testin amacı sınıf değil,
    # "patlıyor mu" sorusu.
    client = GeminiClient("m", "")
    with pytest.raises(AnswererError) as ei:
        client.generate("p", [b"img"], ["[S1] X sayfa 1"])
    # Y15: artık yalnız "patlıyor" değil, NEDEN patladığı da taşınıyor.
    assert ei.value.error_type == "auth"


class StubClient:
    def generate(self, prompt, images, markers):
        return GenResult(text="cevap [S1]", tokens_in=1234, tokens_out=56)


class StubClientNoUsage:
    def generate(self, prompt, images, markers):
        return GenResult(text="cevap [S1]")


def test_answer_annotates_token_usage():
    ans = GeminiAnswerer("m", "k", client=StubClient())
    with collecting() as col:
        ans.answer("soru", [hit("k1:1")], lambda p: b"img")
    assert col.notes["tokens_in"] == 1234 and col.notes["tokens_out"] == 56


def test_answer_without_usage_annotates_nothing():
    ans = GeminiAnswerer("m", "k", client=StubClientNoUsage())
    with collecting() as col:
        ans.answer("soru", [hit("k1:1")], lambda p: b"img")
    assert "tokens_in" not in col.notes and "tokens_out" not in col.notes


# --- B14: [Sk] <-> görüntü AÇIK bağlama -------------------------------------
#
# Eskiden `contents=[*parts, prompt]` idi: önce 5 ETİKETSİZ görüntü, sonra tek
# metin bloğu. Modelin k'ıncı görüntüyü `[Sk]` ile eşlemesi tamamen KONUMSAL
# çıkarımdı ve hiçbir yerde doğrulanmıyordu — bu düzeltilmeden ölçülecek
# "citation precision" konumsal şansı ölçerdi (G2.2 önkoşulu). Testler yapı
# düzeyindedir: kurulan `contents` listesi incelenir, canlı çağrı YAPILMAZ.


class FakeSDKModels:
    """genai.Client().models yerine geçen kayıt tutucu."""

    def __init__(self, text="cevap [S1]", raises=None):
        self.calls = []
        self._text = text
        self._raises = raises

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        if self._raises:
            raise self._raises.pop(0)
        return SimpleNamespace(text=self._text, usage_metadata=None)


def _client_with(models: FakeSDKModels) -> GeminiClient:
    c = GeminiClient("m", "k")
    c._client = SimpleNamespace(models=models)  # SDK'ya hiç dokunulmaz
    return c


def test_contents_interleave_marker_then_image_then_prompt():
    c = GeminiClient("m", "k")
    contents = c.build_contents("İSTEM", [b"a", b"b"], ["[S1] X sayfa 1", "[S2] Y sayfa 9"])
    assert len(contents) == 5
    assert contents[0] == "[S1] X sayfa 1"
    assert contents[2] == "[S2] Y sayfa 9"
    assert contents[4] == "İSTEM"
    # 1. ve 3. konumlar görüntü part'ları — ve SIRA korunmuş.
    assert [p.inline_data.data for p in (contents[1], contents[3])] == [b"a", b"b"]
    assert all(p.inline_data.mime_type == "image/webp" for p in (contents[1], contents[3]))


def test_contents_rejects_marker_image_mismatch():
    c = GeminiClient("m", "k")
    with pytest.raises(ValueError, match="eşleşmiyor"):
        c.build_contents("p", [b"a", b"b"], ["[S1] tek"])


def test_answerer_binds_each_marker_to_its_own_page():
    """Uçtan uca yapı: `[Sk]` metni k'ıncı sayfanın künyesini taşır ve k'ıncı
    görüntünün HEMEN ÖNÜNDE durur."""
    models = FakeSDKModels()
    pages = [hit("k1:12"), hit("k2:7")]
    pages[1] = pages[1].model_copy(update={"doc_name": "İş Kanunu", "page_no": 7})
    loader = {p.image_path: f"img{i}".encode() for i, p in enumerate(pages)}
    ans = GeminiAnswerer("m", "k", client=_client_with(models))
    ans.answer("soru", pages, lambda p: loader[p])
    contents = models.calls[0]["contents"]
    assert contents[0] == "[S1] Türk Borçlar Kanunu sayfa 12"
    assert contents[1].inline_data.data == b"img0"
    assert contents[2] == "[S2] İş Kanunu sayfa 7"
    assert contents[3].inline_data.data == b"img1"
    assert isinstance(contents[4], str) and "Soru: soru" in contents[4]


def test_source_marker_shape():
    assert source_marker(3, hit("k1:12")) == "[S3] Türk Borçlar Kanunu sayfa 12"


def test_system_prompt_embeds_the_single_source_honest_miss_marker():
    """S35/D3 borcu: modele DAYATILAN ifade ile sunucunun ARADIĞI ifade tek
    sabitten gelir; ikisi elle yazılmış olsaydı sessizce ayrışabilirdi."""
    assert HONEST_MISS_MARKER in SYSTEM
    assert HONEST_MISS_MARKER in build_prompt("s", [hit("k1:1")])


# --- Y15: zaman aşımı, tek retry, hata taksonomisi ---------------------------


def test_timeout_budget_stays_under_the_declared_ceiling():
    """Bütçe aritmetiği kilitli: denemeler + backoff <= tavan."""
    worst = GEMINI_MAX_ATTEMPTS * GEMINI_TIMEOUT_S + (GEMINI_MAX_ATTEMPTS - 1) * (
        GEMINI_RETRY_BACKOFF_S
    )
    assert worst <= GEMINI_TOTAL_BUDGET_S
    assert GEMINI_MAX_ATTEMPTS == 2  # "TEK retry"


def test_client_passes_timeout_to_the_sdk(monkeypatch):
    """Zaman aşımı SDK'ya GERÇEKTEN geçiyor mu (Y15'in tek maddesi)."""
    captured = {}

    class FakeGenai:
        @staticmethod
        def Client(*, api_key, http_options):  # noqa: N802 - SDK adı
            captured["api_key"] = api_key
            captured["timeout_ms"] = http_options.timeout
            return SimpleNamespace(models=FakeSDKModels())

    import google.genai as real_genai

    monkeypatch.setattr(real_genai, "Client", FakeGenai.Client)
    GeminiClient("m", "anahtar", timeout_s=12.0)._ensure_client()
    assert captured["api_key"] == "anahtar"
    assert captured["timeout_ms"] == 12_000  # SDK sözleşmesi: MİLİSANİYE


@pytest.mark.parametrize(
    "exc,expected",
    [
        (TimeoutError("yavaş"), "timeout"),
        (ValueError("Missing key inputs argument! api_key"), "auth"),
        (RuntimeError("bilinmeyen"), "other"),
    ],
)
def test_classify_error_taxonomy(exc, expected):
    assert classify_error(exc) == expected
    assert classify_error(exc) in ERROR_TYPES


@pytest.mark.parametrize(
    "code,expected",
    [(429, "http_429"), (401, "auth"), (403, "auth"), (500, "http_5xx"), (503, "http_5xx")],
)
def test_classify_api_error_by_http_code(code, expected):
    from google.genai import errors as genai_errors

    exc = genai_errors.APIError(code, {"error": {"message": "x", "status": "S"}})
    assert classify_error(exc) == expected
    assert expected in ERROR_TYPES


def test_httpx_timeout_is_classified_as_timeout():
    import httpx

    assert classify_error(httpx.ReadTimeout("read")) == "timeout"
    assert classify_error(httpx.ConnectTimeout("connect")) == "timeout"


def test_timeout_is_retried_exactly_once_then_raises_taxonomy(monkeypatch):
    slept = []
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", slept.append)
    models = FakeSDKModels(raises=[TimeoutError("1"), TimeoutError("2")])
    with pytest.raises(AnswererError) as ei:
        _client_with(models).generate("p", [b"i"], ["[S1] X sayfa 1"])
    assert ei.value.error_type == "timeout"
    assert len(models.calls) == GEMINI_MAX_ATTEMPTS  # tam olarak TEK yeniden deneme
    assert slept == [GEMINI_RETRY_BACKOFF_S]


def test_retry_succeeds_on_the_second_attempt(monkeypatch):
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", lambda s: None)
    from google.genai import errors as genai_errors

    models = FakeSDKModels(raises=[genai_errors.ServerError(503, {"error": {"message": "down"}})])
    with collecting() as col:
        out = _client_with(models).generate("p", [b"i"], ["[S1] X sayfa 1"])
    assert out.text == "cevap [S1]"
    assert len(models.calls) == 2
    assert col.notes["gemini_retried"] is True


def test_rate_limit_is_not_retried(monkeypatch):
    """429 retry EDİLMEZ: kota aşımında ikinci istek durumu kötüleştirir ve
    faturayı büyütür."""
    slept = []
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", slept.append)
    from google.genai import errors as genai_errors

    models = FakeSDKModels(raises=[genai_errors.ClientError(429, {"error": {"message": "quota"}})])
    with pytest.raises(AnswererError) as ei:
        _client_with(models).generate("p", [b"i"], ["[S1] X sayfa 1"])
    assert ei.value.error_type == "http_429"
    assert len(models.calls) == 1 and slept == []
    assert "http_429" not in RETRYABLE_ERROR_TYPES


def test_safety_block_is_its_own_error_type():
    """Boş yanıt + blok sebebi = "cevaplamamalı" sınıfı; "cevaplayamadı" değil."""

    class BlockedModels(FakeSDKModels):
        def generate_content(self, *, model, contents):
            self.calls.append(contents)
            return SimpleNamespace(
                text="",
                prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name="SAFETY")),
            )

    with pytest.raises(AnswererError) as ei:
        _client_with(BlockedModels()).generate("p", [b"i"], ["[S1] X sayfa 1"])
    assert ei.value.error_type == "safety_block"


def test_ensure_client_is_locked():
    """K33: kilitsiz tembel kurulum iki eşzamanlı ilk /ask'te yavaş SDK
    import'unu istek yolunda ÇAKIŞARAK yapıyordu."""
    import threading

    c = GeminiClient("m", "k")
    assert isinstance(c._client_lock, type(threading.Lock()))


# --- review M1: toplam bütçe bir YORUM değil, ölçülen bir invariant ----------


class SlowFailingModels:
    """Her denemede sahte saati `cost_s` kadar ilerletip zaman aşımı fırlatır."""

    def __init__(self, clock: dict, cost_s: float):
        self.clock = clock
        self.cost_s = cost_s
        self.calls = 0

    def generate_content(self, *, model, contents):
        self.calls += 1
        self.clock["t"] += self.cost_s
        raise TimeoutError(f"deneme {self.calls} zaman aşımı")


def test_retry_is_skipped_when_the_budget_cannot_cover_another_attempt(monkeypatch):
    """httpx faz-başına sayaç tuttuğu için TEK deneme ilan edilen süreyi aşabilir
    (canlı ölçüm: 16,2 sn / 15 sn). Bütçe dolmuşken ÜSTÜNE ikinci bir deneme
    binmemeli — aksi halde "<= 35 sn" bir tahmin olarak kalırdı."""
    clock = {"t": 0.0}
    monkeypatch.setattr("belge_gozu.answer.gemini.time.monotonic", lambda: clock["t"])
    slept = []
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", slept.append)
    # 25 sn: 25 + 0.5 + 15 = 40.5 > 35 -> retry KALMIYOR (timeout retry'lenebilir bir sınıf).
    models = SlowFailingModels(clock, cost_s=25.0)
    with collecting() as col, pytest.raises(AnswererError) as ei:
        _client_with(models).generate("p", [b"i"], ["[S1] X sayfa 1"])
    assert ei.value.error_type == "timeout"
    assert models.calls == 1, "bütçe dolmuşken ikinci deneme yapılmamalı"
    assert slept == [], "retry atlandıysa backoff da uyumamalı"
    assert col.notes["gemini_retry_skipped_budget"] is True
    assert "gemini_retried" not in col.notes
    assert clock["t"] <= GEMINI_TOTAL_BUDGET_S  # duvar saati tavanın altında kaldı


def test_retry_still_happens_when_the_budget_allows_it(monkeypatch):
    """Invariant retry'yi TAMAMEN kapatmıyor: bütçe yetiyorsa tek deneme yapılır."""
    clock = {"t": 0.0}
    monkeypatch.setattr("belge_gozu.answer.gemini.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", lambda s: None)
    # 16.2 sn (canlı ölçüm): 16.2 + 0.5 + 15 = 31.7 <= 35 -> retry YAPILIR.
    models = SlowFailingModels(clock, cost_s=16.2)
    with collecting() as col, pytest.raises(AnswererError):
        _client_with(models).generate("p", [b"i"], ["[S1] X sayfa 1"])
    assert models.calls == GEMINI_MAX_ATTEMPTS
    assert col.notes["gemini_retried"] is True
    assert "gemini_retry_skipped_budget" not in col.notes


def test_slow_client_degrades_with_timeout_taxonomy_within_budget(monkeypatch):
    """Uçtan uca: yavaş istemci -> tek deneme -> degraded + error_type='timeout',
    ve gerçek duvar saati bütçenin çok altında (testte hiç uyunmuyor)."""
    from belge_gozu.answer.base import AskService

    clock = {"t": 0.0}
    monkeypatch.setattr("belge_gozu.answer.gemini.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", lambda s: None)
    models = SlowFailingModels(clock, cost_s=30.0)
    answerer = GeminiAnswerer("m", "k", client=_client_with(models))

    class OneHit:
        def search(self, query, k=5, candidates=200):
            return [hit("k1:1")]

    svc = AskService(OneHit(), answerer, min_score=-1e9, image_loader=lambda p: b"img")
    real_t0 = time.monotonic()
    with collecting() as col:
        answer, hits = svc.ask("soru", k=5, candidates=200)
    assert time.monotonic() - real_t0 < GEMINI_TOTAL_BUDGET_S
    assert col.notes["degraded"] is True and col.notes["error_type"] == "timeout"
    assert models.calls == 1 and answer.abstained and hits


# --- review M2: `parse` taksonomi değeri artık ULAŞILABİLİR ------------------


def test_unknown_api_response_error_is_classified_as_parse():
    """`UnknownApiResponseError` `ValueError`dan türer, `APIError`dan DEĞİL —
    kontrol APIError dalının içindeyken `parse` ÖLÜ KODDU ve SDK ham
    JSONDecodeError'ı bu sınıfa sarmaladığı için üretimde asla üretilemezdi."""
    from google.genai import errors as genai_errors

    assert not issubclass(genai_errors.UnknownApiResponseError, genai_errors.APIError)
    exc = genai_errors.UnknownApiResponseError("beklenmeyen gövde")
    assert classify_error(exc) == "parse"
    assert "parse" in ERROR_TYPES


def test_parse_error_is_not_misclassified_as_auth():
    """İkincil risk: bu istisna ham gövdeyi mesajına gömer, "API key" geçen bir
    hata sayfası `_API_KEY_MSG` desenine takılıp `auth` raporlardı."""
    from google.genai import errors as genai_errors

    exc = genai_errors.UnknownApiResponseError("<html>Invalid API key supplied</html>")
    assert classify_error(exc) == "parse"


def test_parse_error_reaches_the_degraded_row_end_to_end():
    models = FakeSDKModels()

    def boom(*, model, contents):
        from google.genai import errors as genai_errors

        raise genai_errors.UnknownApiResponseError("bozuk gövde")

    models.generate_content = boom
    with pytest.raises(AnswererError) as ei:
        _client_with(models).generate("p", [b"i"], ["[S1] X sayfa 1"])
    assert ei.value.error_type == "parse"


# --- ANAHTAR ROTASYONU -------------------------------------------------------
#
# Kural (kullanıcı direktifi): ikinci bir ücretsiz-kota anahtarı var; API
# katmanındaki HERHANGİ bir hatada aynı istek öbür anahtarla BİR KEZ yeniden
# denenir. Testlerin tamamı STUB'dur (ağ YOK) ve anahtar DEĞERLERİ apaçık
# sahtedir; hiçbir iddia anahtar değeri üzerinde değil, YALNIZ "key1"/"key2"
# etiketleri üzerinedir — gerçek bir anahtar dizesi test dosyasına giremez.

FAKE_KEY_1 = "sahte-birincil-anahtar"
FAKE_KEY_2 = "sahte-yedek-anahtar"


def rate_limit() -> Exception:
    from google.genai import errors as genai_errors

    return genai_errors.ClientError(429, {"error": {"message": "quota"}})


class KeyModels:
    """Tek bir anahtarın sahte SDK'sı (`genai.Client().models` yerine).

    `always` verilirse her çağrıda o istisna; yoksa `errors` sırayla tüketilir
    ve liste bitince yanıt döner. Sayaç KİLİTLİ: eşzamanlılık dumanı testinde
    iki iş parçacığı aynı anda çağırıyor.
    """

    def __init__(
        self,
        *,
        always: Exception | None = None,
        errors: list[Exception] | None = None,
        text: str = "cevap [S1]",
    ) -> None:
        self.calls = 0
        self._always = always
        self._errors = list(errors or [])
        self._text = text
        self._lock = threading.Lock()

    def generate_content(self, *, model, contents, config=None):
        with self._lock:
            self.calls += 1
            err = self._always or (self._errors.pop(0) if self._errors else None)
        if err is not None:
            raise err
        return SimpleNamespace(text=self._text, usage_metadata=None)


def _slot(label: str, models) -> KeySlot:
    client = GeminiClient("m", f"sahte-{label}")
    client._client = SimpleNamespace(models=models)  # SDK'ya hiç dokunulmaz
    return KeySlot(label, client)


def _rot(*models, sticky: StickyKeyIndex | None = None) -> RotatingGeminiClient:
    """N slotlu rotasyon istemcisi; her slot KENDİ sahte SDK'sıyla.

    `sticky` her testte YENİ: süreç düzeyindeki gerçek gösterge (`_STICKY`)
    testler arasında sızmasın.
    """
    slots = [_slot(label, m) for label, m in zip(KEY_LABELS, models, strict=True)]
    return RotatingGeminiClient(slots, sticky=sticky or StickyKeyIndex())


def _generate(client) -> GenResult:
    return client.generate("p", [b"i"], ["[S1] X sayfa 1"])


def test_rate_limit_on_key1_rotates_to_key2_and_serves():
    """Merdivenin 1->2. basamağı: 429 AYNI anahtarda umutsuz, ÖBÜR anahtarda
    tam olarak umut vaat eden hatadır — kota anahtar başına sayılır."""
    k1, k2 = KeyModels(always=rate_limit()), KeyModels()
    with collecting() as col:
        out = _generate(_rot(k1, k2))
    assert out.text == "cevap [S1]"
    assert k1.calls == 1, "ilk anahtarda AYNI anahtar retry'si yok — sıra öbürüne geçer"
    assert k2.calls == 1
    assert col.notes["llm"] == {
        "rotations": [{"from": "key1", "error_type": "http_429"}],
        "key": "key2",
    }


def test_rotation_is_sticky_next_request_starts_on_the_new_key():
    """Yapışkanlık: rotasyondan sonra ikinci istek doğrudan key2 ile başlar.

    Aksi (her istek key1 ile başlasın) kotası bitmiş bir anahtarda istek
    BAŞINA garantili bir başarısız çağrı demektir."""
    sticky = StickyKeyIndex()
    k1, k2 = KeyModels(always=rate_limit()), KeyModels()
    rot = _rot(k1, k2, sticky=sticky)
    _generate(rot)  # 1. istek: key1 -> 429 -> key2
    with collecting() as col:
        out = _generate(rot)  # 2. istek
    assert out.text == "cevap [S1]"
    assert k1.calls == 1, "ölü anahtara istek başına bir kez daha çarpılmamalı"
    assert k2.calls == 2
    assert col.notes["llm"] == {"key": "key2"}  # 2. istekte rotasyon YOK
    assert sticky.current(2) == 1


def test_both_keys_failing_raises_the_last_taxonomy_with_both_labels():
    k1, k2 = KeyModels(always=rate_limit()), KeyModels(always=rate_limit())
    with pytest.raises(AnswererError) as ei:
        _generate(_rot(k1, k2))
    assert ei.value.error_type == "http_429"
    assert "key1, key2" in str(ei.value)  # yalnız ETİKETLER, anahtar değeri DEĞİL
    assert k1.calls == 1 and k2.calls == 1


def test_both_keys_failing_reaches_the_degraded_row_with_keys_tried():
    """Uçtan uca: iki anahtar da düştü -> degraded + SON hatanın taksonomisi +
    `detail.llm.keys_tried`. Hiçbir anahtar SERVİS ETMEDİĞİ için `key` yok."""
    from belge_gozu.answer.base import AskService

    k1, k2 = KeyModels(always=rate_limit()), KeyModels(always=rate_limit())
    answerer = GeminiAnswerer("m", FAKE_KEY_1, client=_rot(k1, k2))

    class OneHit:
        def search(self, query, k=5, candidates=200):
            return [hit("k1:1")]

    svc = AskService(OneHit(), answerer, min_score=-1e9, image_loader=lambda p: b"img")
    with collecting() as col:
        answer, hits = svc.ask("soru", k=5, candidates=200)
    assert col.notes["degraded"] is True and col.notes["error_type"] == "http_429"
    assert col.notes["llm"] == {
        "rotations": [{"from": "key1", "error_type": "http_429"}],
        "keys_tried": ["key1", "key2"],
    }
    assert "key" not in col.notes["llm"]
    assert answer.abstained and hits


def test_parse_error_does_not_rotate():
    """`parse` BAŞARILI bir yanıtın sınıflandırmasıdır (200 geldi, gövde
    okunamadı): aynı istek başka bir anahtarla aynı gövdeyi geri getirir.
    Kullanıcının "her hatada dene" kuralı TAŞIMA düzeyinde uygulanır."""
    from google.genai import errors as genai_errors

    k1 = KeyModels(always=genai_errors.UnknownApiResponseError("bozuk gövde"))
    k2 = KeyModels()
    with collecting() as col, pytest.raises(AnswererError) as ei:
        _generate(_rot(k1, k2))
    assert ei.value.error_type == "parse"
    assert k1.calls == 1 and k2.calls == 0
    assert "llm" not in col.notes  # hiçbir anahtar servis etmedi, rotasyon da yok
    assert NON_ROTATABLE_ERROR_TYPES == frozenset({"parse"})


def test_rotation_is_skipped_when_the_budget_cannot_cover_another_attempt(monkeypatch):
    """Toplam bütçe invariantı rotasyonu da KAPSAR: "öbür anahtarı da dene"
    ikinci bir zaman aşımı penceresi açar ve tavan aşılırdı."""
    clock = {"t": 0.0}
    monkeypatch.setattr("belge_gozu.answer.gemini.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", lambda s: None)
    # 25 sn: 25 + 15 = 40 > 35 -> rotasyon KALMIYOR.
    k1, k2 = SlowFailingModels(clock, cost_s=25.0), KeyModels()
    with collecting() as col, pytest.raises(AnswererError) as ei:
        _generate(_rot(k1, k2))
    assert ei.value.error_type == "timeout"
    assert k1.calls == 1 and k2.calls == 0, "bütçe dolmuşken ikinci anahtar denenmemeli"
    assert col.notes["gemini_rotation_skipped_budget"] is True
    assert "llm" not in col.notes
    assert clock["t"] <= GEMINI_TOTAL_BUDGET_S


def test_rotation_still_happens_when_the_budget_allows_it(monkeypatch):
    """Invariant rotasyonu TAMAMEN kapatmıyor: 16,2 sn (canlı ölçüm) + 15 sn
    tavanın altında kalır, öbür anahtar denenir."""
    clock = {"t": 0.0}
    monkeypatch.setattr("belge_gozu.answer.gemini.time.monotonic", lambda: clock["t"])
    k1, k2 = SlowFailingModels(clock, cost_s=16.2), KeyModels()
    with collecting() as col:
        out = _generate(_rot(k1, k2))
    assert out.text == "cevap [S1]" and k1.calls == 1 and k2.calls == 1
    assert col.notes["llm"] == {
        "rotations": [{"from": "key1", "error_type": "timeout"}],
        "key": "key2",
    }


def test_the_last_key_keeps_the_same_key_retry_and_the_ladder_caps_at_three(monkeypatch):
    """Merdivenin 3. basamağı: rotasyondan sonra gidecek yer kalmadığı için
    mevcut AYNI ANAHTAR retry hakkı SON anahtarda kullanılır (toplam <= 3)."""
    slept = []
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", slept.append)
    k1 = KeyModels(always=rate_limit())
    k2 = KeyModels(errors=[TimeoutError("yavaş")])  # 1 hata, sonra yanıt
    with collecting() as col:
        out = _generate(_rot(k1, k2))
    assert out.text == "cevap [S1]"
    assert (k1.calls, k2.calls) == (1, 2) and k1.calls + k2.calls == 3
    assert slept == [GEMINI_RETRY_BACKOFF_S]
    assert col.notes["gemini_retried"] is True
    assert col.notes["llm"] == {
        "rotations": [{"from": "key1", "error_type": "http_429"}],
        "key": "key2",
    }


def test_the_ladder_never_exceeds_three_attempts(monkeypatch):
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", lambda s: None)
    k1 = KeyModels(always=rate_limit())
    k2 = KeyModels(always=TimeoutError("hep yavaş"))
    with pytest.raises(AnswererError) as ei:
        _generate(_rot(k1, k2))
    assert ei.value.error_type == "timeout"  # SON hatanın taksonomisi
    assert k1.calls + k2.calls == 3


def test_generate_json_goes_through_the_same_rotation_ladder():
    """Doğrulayıcı yolu (yalnız metin + JSON şeması) da AYNI havuzdan geçer —
    tek fabrikanın (`build_gemini_client`) karşılığı tam olarak budur."""
    k1, k2 = KeyModels(always=rate_limit()), KeyModels(text='{"verdict": "supported"}')
    with collecting() as col:
        out = _rot(k1, k2).generate_json("istem", None)
    assert out.text == '{"verdict": "supported"}'
    assert k1.calls == 1 and k2.calls == 1
    assert col.notes["llm"]["key"] == "key2"


def test_verifier_client_shares_the_rotating_pool_and_a_key_agnostic_cache():
    import inspect

    from belge_gozu.answer.verify import GeminiVerifierClient, cache_key

    vc = GeminiVerifierClient("m", FAKE_KEY_1, api_key_2=FAKE_KEY_2)
    assert isinstance(vc._client, RotatingGeminiClient)
    assert vc._client.labels == ["key1", "key2"]
    # Önbellek ANAHTARDAN BAĞIMSIZ olmalı: hangi API anahtarının servis ettiği
    # yargıyı değiştirmez, dolayısıyla rotasyon bir isabeti geçersizleştirmez.
    assert set(inspect.signature(cache_key).parameters) == {
        "model",
        "prompt_version",
        "claim_text",
        "evidence_sha",
    }


# --- tek anahtarlı havuz: BUGÜNKÜ davranışın kilidi ---------------------------


def test_empty_second_key_means_a_single_key_pool():
    assert build_gemini_client("m", FAKE_KEY_1, "").labels == ["key1"]
    assert build_gemini_client("m", FAKE_KEY_1, FAKE_KEY_2).labels == ["key1", "key2"]


def test_slot_label_follows_the_slot_not_the_position():
    """Yalnız yedek anahtar doluysa havuz tek elemanlıdır ama etiketi "key2"
    kalır: etiket SLOTU adlandırır, havuzdaki konumu değil."""
    assert build_gemini_client("m", "", FAKE_KEY_2).labels == ["key2"]


def test_keyless_boot_still_builds_a_pool_and_fails_with_auth_at_call_time():
    """Anahtarsız `serve` çökmemeli (tembel kurulum); hata ilk çağrıda `auth`."""
    client = build_gemini_client("m", "", "")
    assert client.labels == ["key1"]
    with pytest.raises(AnswererError) as ei:
        _generate(client)
    assert ei.value.error_type == "auth"


def test_single_key_pool_retries_on_the_same_key_exactly_like_before(monkeypatch):
    """Havuz tek anahtarlıyken rotasyon dalı HİÇ girilmez: aynı anahtarda tek
    yeniden deneme, aynı backoff, aynı taksonomi (bu katman eklenmeden önceki
    davranışın ta kendisi)."""
    slept = []
    monkeypatch.setattr("belge_gozu.answer.gemini.time.sleep", slept.append)
    k1 = KeyModels(always=TimeoutError("yavaş"))
    rot = RotatingGeminiClient([_slot("key1", k1)], sticky=StickyKeyIndex())
    with collecting() as col, pytest.raises(AnswererError) as ei:
        _generate(rot)
    assert ei.value.error_type == "timeout"
    assert k1.calls == GEMINI_MAX_ATTEMPTS and slept == [GEMINI_RETRY_BACKOFF_S]
    assert "llm" not in col.notes and "gemini_rotation_skipped_budget" not in col.notes


def test_single_key_success_still_records_which_key_served():
    k1 = KeyModels()
    with collecting() as col:
        _generate(RotatingGeminiClient([_slot("key1", k1)], sticky=StickyKeyIndex()))
    assert col.notes["llm"] == {"key": "key1"}


# --- eşzamanlılık: gösterge paylaşılan mutable durumdur -----------------------


def test_concurrent_rotations_leave_the_sticky_index_uncorrupted():
    """Senkron uç noktalar Starlette'in iş parçacığı havuzunda koşar: iki istek
    aynı anda rotasyon yapabilir. Yarış İYİ HUYLUDUR (ikisi de aynı sıradaki
    anahtara gider), ama gösterge her an GEÇERLİ bir slot numarası olmalı."""
    sticky = StickyKeyIndex()
    k1, k2 = KeyModels(always=rate_limit()), KeyModels()
    rot = _rot(k1, k2, sticky=sticky)
    start = threading.Barrier(2)
    out: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            start.wait(timeout=5)
            out.append(_generate(rot).text)
        except BaseException as exc:  # pragma: no cover - yalnız arıza yolunda
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors and out == ["cevap [S1]"] * 2
    assert sticky.current(2) == 1  # gösterge GEÇERLİ ve yeni anahtarı işaret ediyor
    assert k2.calls == 2, "iki istek de servis edilmeli"
    # İKİ MEŞRU SIRALANIŞ var ve testin ikisini de kabul etmesi ŞART (aksi
    # halde test yarışı değil, yalnız bir zamanlamayı kilitlerdi):
    #   * ikisi de key1 ile başladı (ikisi de 429 aldı, ikisi de rotasyon
    #     yaptı)                                     -> k1.calls == 2
    #   * biri önce rotasyonu bitirdi, öteki göstergeyi ZATEN key2'de buldu
    #                                                -> k1.calls == 1
    assert k1.calls in (1, 2)
