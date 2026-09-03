"""Skor-füzyonsuz aday havuzu oluşturma."""

from collections.abc import Sequence


def build_candidate_pool(
    bm25_pages: Sequence[str],
    late_page_lists: Sequence[Sequence[str]],
    *,
    limit: int = 50,
) -> list[str]:
    """Her kaynaktan en fazla ``limit`` sayfayı ilk-görülme sırasıyla birleştir."""
    if limit < 1:
        raise ValueError(f"candidate limit pozitif olmalı: {limit}")
    seen: set[str] = set()
    out: list[str] = []
    for source in (bm25_pages, *late_page_lists):
        for page_id in source[:limit]:
            if page_id not in seen:
                seen.add(page_id)
                out.append(page_id)
    return out
