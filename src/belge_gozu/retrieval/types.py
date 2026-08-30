from pydantic import BaseModel


class PageHit(BaseModel):
    """Servis edilen bir sayfa — `score` SIRALAMA ölçeğindedir.

    `score` etkin pipeline'ın ölçeğinde gelir (hibrit: BM25 birimi, üst
    sınırsız; exhaustive/two-stage: normalize [-1,1]) ve eşik BU alanla
    karşılaştırılır.

    `visual_score` İKİNCİ bir ölçektir ve asla `score` ile karıştırılmaz:
    görsel MaxSim kanalının aynı sayfa için ürettiği normalize ~[-1,1] skor.
    Yalnız hibrit yolda doludur (görsel kanal orada sıralamaya girmeden koşar);
    görsel kollarda kanalın kendisi zaten `score`'u ürettiği için tekrar
    edilmez ve None kalır. Amacı GÖSTERİM ve ŞEFFAFLIK: arayüz iki kanalın aynı
    sayfaya ne dediğini yan yana gösterebilsin, "hibrit" iddiası kullanıcı
    tarafında da görünür olsun. Sıralamaya ETKİSİ YOKTUR (ölçüm: her füzyon
    denemesi geriledi — bkz. retrieval/hybrid.py).
    """

    page_id: str
    score: float
    doc_name: str
    page_no: int
    image_path: str
    source_url: str
    visual_score: float | None = None
