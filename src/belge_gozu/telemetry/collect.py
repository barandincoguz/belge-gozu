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
