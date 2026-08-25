import logging
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from belge_gozu.retrieval.types import PageHit

logger = logging.getLogger(__name__)

ABSTAIN_TEXT = "Bu soruya korpustaki belgelerde dayanak bulamadım."
SERVICE_ERROR_TEXT = (
    "Yanıt servisi şu anda kullanılamıyor (kota veya servis hatası). "
    "Bulunan sayfalar aşağıda listeleniyor."
)


class Answer(BaseModel):
    text: str
    citations: list[str]
    abstained: bool = False


class Answerer(Protocol):
    def answer(
        self, question: str, pages: list[PageHit], image_loader: Callable[[str], bytes]
    ) -> Answer: ...


class AskService:
    def __init__(
        self, retriever, answerer: Answerer, min_score: float, image_loader: Callable[[str], bytes]
    ):
        self.retriever = retriever
        self.answerer = answerer
        self.min_score = min_score
        self.image_loader = image_loader

    def ask(self, question: str, k: int, candidates: int) -> tuple[Answer, list[PageHit]]:
        hits = self.retriever.search(question, k=k, candidates=candidates)
        if not hits or hits[0].score < self.min_score:
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True), hits
        try:
            return self.answerer.answer(question, hits, self.image_loader), hits
        except Exception:
            logger.exception("answerer failed")
            return Answer(text=SERVICE_ERROR_TEXT, citations=[], abstained=True), hits
