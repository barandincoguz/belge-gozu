import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from belge_gozu.index.manifest import IndexManifest, read_manifest, write_manifest

CHUNK_TOKENS = 500_000


@dataclass
class FloatIndex:
    """f16 master token embedding'leri: embs.npy (toplam_token, 128) float16 +
    offsets.npy + page_ids.json + manifest.json (quantization="float16").

    T5'teki PackedIndex ile aynı sayfa-hizalı chunk deseni kullanılır; tek fark
    Hamming yerine gerçek dot-product MaxSim (native_float_scores)."""

    embs: np.ndarray
    offsets: np.ndarray
    page_ids: list[str]
    manifest: IndexManifest | None = None

    @classmethod
    def build(
        cls,
        page_ids: list[str],
        embs: list[np.ndarray],
        manifest: IndexManifest | None = None,
    ) -> "FloatIndex":
        if len(page_ids) != len(embs):
            raise ValueError(
                f"page_ids ({len(page_ids)}) ve embs ({len(embs)}) uzunlukları eşleşmiyor"
            )
        if not embs:
            raise ValueError("boş korpus: en az bir sayfa embedding'i gerekli")
        for pid, e in zip(page_ids, embs, strict=True):
            if e.shape[0] == 0:
                raise ValueError(f"sıfır token'lı sayfa: {pid}")
        offsets = np.zeros(len(embs) + 1, dtype=np.int64)
        np.cumsum([e.shape[0] for e in embs], out=offsets[1:])
        stacked = np.vstack(embs).astype(np.float16)
        return cls(stacked, offsets, list(page_ids), manifest)

    def page_tokens(self, i: int) -> np.ndarray:
        return self.embs[self.offsets[i] : self.offsets[i + 1]]

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / "embs.npy", self.embs)
        np.save(dir / "offsets.npy", self.offsets)
        (dir / "page_ids.json").write_text(json.dumps(self.page_ids, ensure_ascii=False))
        if self.manifest is not None:
            write_manifest(dir, self.manifest)

    @classmethod
    def load(cls, dir: Path, mmap: bool = True) -> "FloatIndex":
        mode = "r" if mmap else None
        return cls(
            embs=np.load(dir / "embs.npy", mmap_mode=mode),
            offsets=np.load(dir / "offsets.npy"),
            page_ids=json.loads((dir / "page_ids.json").read_text()),
            manifest=read_manifest(dir),
        )


def _chunk_bounds(offsets: np.ndarray) -> list[int]:
    bounds = [0]
    for i in range(1, len(offsets)):
        last = bounds[-1]
        if offsets[i] - offsets[last] >= CHUNK_TOKENS:
            bounds.append(i)
    if bounds[-1] != len(offsets) - 1:
        bounds.append(len(offsets) - 1)
    return bounds


def native_float_scores(findex: FloatIndex, q_emb: np.ndarray) -> np.ndarray:
    """(n_pages,) — float MaxSim (per-query-token ortalama), sayfa-hizalı chunk'lı.

    T5'teki `ExhaustiveBinaryRetriever.score_all` deseniyle aynı: sayfa-hizalı
    chunk'lar üstünde `np.maximum.reduceat`; tek fark Hamming yerine `q @
    chunk.T` gerçek dot-product'ı (f16 saklamadan float32'ye açılarak)."""
    q = np.asarray(q_emb, dtype=np.float32)
    offsets = np.asarray(findex.offsets)
    n_pages = len(findex.page_ids)
    out = np.empty(n_pages, dtype=np.float64)
    bounds = _chunk_bounds(offsets)
    for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
        t0, t1 = int(offsets[b0]), int(offsets[b1])
        chunk = np.asarray(findex.embs[t0:t1], dtype=np.float32)
        sim = q @ chunk.T  # (n_q, chunk_tokens)
        starts = (offsets[b0:b1] - t0).astype(np.int64)
        # offsets kesin artan (FloatIndex.build sıfır-token sayfayı reddeder) ->
        # reduceat boş segment göremez.
        out[b0:b1] = np.maximum.reduceat(sim, starts, axis=1).sum(axis=0)
    return out / max(1, q.shape[0])


def rank_of(scores: np.ndarray, page_ids: list[str], target: str) -> int:
    """1-tabanlı sıra: eşitlikte iyimser değil (stable argsort pozisyonu)."""
    order = np.argsort(-scores, kind="stable")
    idx = page_ids.index(target)
    return int(np.flatnonzero(order == idx)[0]) + 1
