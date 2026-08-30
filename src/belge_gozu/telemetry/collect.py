import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class StageCollector:
    """İstek boyunca aşama sürelerini (ms) ve notları biriktirir."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.notes: dict[str, object] = {}


_collector: ContextVar[StageCollector | None] = ContextVar("bg_collector", default=None)


@contextmanager
def collecting() -> Iterator[StageCollector]:
    col = StageCollector()
    token = _collector.set(col)
    try:
        yield col
    finally:
        _collector.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    col = _collector.get()
    if col is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        col.stages[name] = (time.perf_counter() - t0) * 1000.0


def annotate(key: str, value: object) -> None:
    col = _collector.get()
    if col is not None:
        col.notes[key] = value


def note(name: str, /) -> object | None:
    """Bu istekte o ana kadar yazılmış notu okur (yoksa None)."""
    col = _collector.get()
    return None if col is None else col.notes.get(name)


def merge_note(name: str, /, **fields: object) -> None:
    """Sözlük değerli bir notu ÜZERİNE YAZMADAN günceller.

    `annotate` son yazanı kazandırır; aynı notu bir istekte BİRDEN FAZLA üreten
    olduğunda bu sessiz bir veri kaybıdır. Somut hâli: `llm` künyesini hem
    yanıtlayıcı hem de (bayrak açıkken, iddia başına) kanıt doğrulayıcısı
    yazar — `annotate` ile doğrulayıcının son çağrısı yanıtlayıcının anahtar
    rotasyonu kaydını siler ve olay satırı rotasyonun HİÇ olmadığını söylerdi.

    Not ADI KONUMSALDIR (`/`): alan adları `**fields`ten geldiği için `key`
    gibi bir alan adı, adlandırılmış bir parametreyle ÇAKIŞIRDI — ve
    `detail.llm.key` tam olarak yazmamız gereken alan.
    """
    col = _collector.get()
    if col is None:
        return
    cur = col.notes.get(name)
    merged = dict(cur) if isinstance(cur, dict) else {}
    merged.update(fields)
    col.notes[name] = merged
