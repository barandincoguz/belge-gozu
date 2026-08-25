import re
from collections.abc import Callable

from belge_gozu.answer.base import Answer
from belge_gozu.retrieval.types import PageHit

SYSTEM = (
    "Sen Türk mevzuatı üzerine bir asistansın. YALNIZCA sana verilen sayfa "
    "görüntülerindeki bilgiye dayanarak Türkçe yanıt ver. Her iddianın sonuna "
    "dayandığı kaynağı [S1] gibi işaretle. Sayfalarda yanıt yoksa açıkça "
    "'verilen sayfalarda bulamadım' de. Sayfa dışı bilgi ekleme."
)


def build_prompt(question: str, pages: list[PageHit]) -> str:
    src_lines = [f"[S{i + 1}] {p.doc_name}, sayfa {p.page_no}" for i, p in enumerate(pages)]
    return f"{SYSTEM}\n\nKaynaklar:\n" + "\n".join(src_lines) + f"\n\nSoru: {question}"


class GeminiClient:
    """google-genai ince sarmalayıcısı. SDK yüzeyi Task 13'te canlı doğrulanır."""

    def __init__(self, model: str, api_key: str):
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, images: list[bytes]) -> str:
        from google.genai import types

        parts = [types.Part.from_bytes(data=b, mime_type="image/webp") for b in images]
        resp = self.client.models.generate_content(model=self.model, contents=[*parts, prompt])
        return resp.text or ""


class GeminiAnswerer:
    def __init__(self, model: str, api_key: str, client=None):
        self._client = client or GeminiClient(model, api_key)

    def answer(
        self, question: str, pages: list[PageHit], image_loader: Callable[[str], bytes]
    ) -> Answer:
        prompt = build_prompt(question, pages)
        images = [image_loader(p.image_path) for p in pages]
        text = self._client.generate(prompt, images)
        idxs = {int(m) for m in re.findall(r"\[S(\d+)\]", text)}
        citations = [pages[i - 1].page_id for i in sorted(idxs) if 0 < i <= len(pages)]
        if not citations and pages:
            citations = [pages[0].page_id]
        return Answer(text=text, citations=citations)
