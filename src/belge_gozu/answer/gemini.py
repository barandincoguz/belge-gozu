import re
from collections.abc import Callable
from dataclasses import dataclass

from belge_gozu.answer.base import Answer
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import annotate

SYSTEM = (
    "Sen Türk mevzuatı üzerine bir asistansın. YALNIZCA sana verilen sayfa "
    "görüntülerindeki bilgiye dayanarak Türkçe yanıt ver. Her iddianın sonuna "
    "dayandığı kaynağı [S1] gibi işaretle. Sayfalarda yanıt yoksa açıkça "
    "'verilen sayfalarda bulamadım' de. Sayfa dışı bilgi ekleme."
)


@dataclass
class GenResult:
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None


def build_prompt(question: str, pages: list[PageHit]) -> str:
    src_lines = [f"[S{i + 1}] {p.doc_name}, sayfa {p.page_no}" for i, p in enumerate(pages)]
    return f"{SYSTEM}\n\nKaynaklar:\n" + "\n".join(src_lines) + f"\n\nSoru: {question}"


class GeminiClient:
    """google-genai ince sarmalayıcısı. SDK yüzeyi Task 13'te canlı doğrulanır.

    Tembel kurulum: __init__ yalnızca model+api_key saklar, SDK'ya dokunmaz —
    böylece anahtarsız `serve` çökmez (keyless boot). Gerçek genai.Client, ve
    onunla birlikte boş-anahtar hatası, yalnızca ilk generate() çağrısında
    oluşur; AskService'in degradation guard'ı bunu SERVICE_ERROR_TEXT'e çevirir.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, images: list[bytes]) -> GenResult:
        from google.genai import types

        client = self._ensure_client()
        parts = [types.Part.from_bytes(data=b, mime_type="image/webp") for b in images]
        resp = client.models.generate_content(model=self.model, contents=[*parts, prompt])
        usage = getattr(resp, "usage_metadata", None)
        return GenResult(
            text=resp.text or "",
            tokens_in=getattr(usage, "prompt_token_count", None),
            tokens_out=getattr(usage, "candidates_token_count", None),
        )


class GeminiAnswerer:
    def __init__(self, model: str, api_key: str, client=None):
        self._client = client or GeminiClient(model, api_key)

    def answer(
        self, question: str, pages: list[PageHit], image_loader: Callable[[str], bytes]
    ) -> Answer:
        prompt = build_prompt(question, pages)
        images = [image_loader(p.image_path) for p in pages]
        gen = self._client.generate(prompt, images)
        text = gen.text
        if gen.tokens_in is not None:
            annotate("tokens_in", gen.tokens_in)
        if gen.tokens_out is not None:
            annotate("tokens_out", gen.tokens_out)
        idxs = {int(m) for m in re.findall(r"\[S(\d+)\]", text)}
        # Atıf YOKSA atıf yok. Eskiden burada top-1 sayfayı otomatik atıf olarak
        # ekleyen bir fallback vardı; model doğru biçimde "verilen sayfalarda
        # bulamadım" dediğinde bile uydurma bir "dayanak" üretiyordu (P2 / G2.7).
        citations = [pages[i - 1].page_id for i in sorted(idxs) if 0 < i <= len(pages)]
        return Answer(text=text, citations=citations)
