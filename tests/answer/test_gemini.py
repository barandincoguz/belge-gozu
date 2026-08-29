from unittest.mock import MagicMock

import pytest

from belge_gozu.answer.gemini import GeminiAnswerer, GeminiClient, GenResult, build_prompt
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


def test_prompt_mentions_sources():
    p = build_prompt("kira artışı sınırı nedir?", [hit("k6098:12"), hit("k6098:13")])
    assert "[S1]" in p and "[S2]" in p and "kira artışı" in p
    assert "Türk Borçlar Kanunu" in p and "sayfa 12" in p


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
    with pytest.raises(Exception):  # noqa: B017
        client.generate("p", [])


class StubClient:
    def generate(self, prompt, images):
        return GenResult(text="cevap [S1]", tokens_in=1234, tokens_out=56)


class StubClientNoUsage:
    def generate(self, prompt, images):
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
