from pydantic import BaseModel


class PageHit(BaseModel):
    page_id: str
    score: float
    doc_name: str
    page_no: int
    image_path: str
    source_url: str
