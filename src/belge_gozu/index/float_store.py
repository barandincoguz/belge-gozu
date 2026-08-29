"""f16 master indeks (`FloatIndex`) — indeks katmanının kendi evi.

Eskiden `bench/oracle.py` içindeydi; T14'te buraya taşındı (ruling D4):
`index/quantize.py` üretim kodu bench paketinden import ediyordu, yani
katman ters çevrilmişti (index -> bench). Artık üç indeks sınıfı da
(`PackedIndex`, `Int8Index`, `FloatIndex`) aynı katmanda ve aynı
`score_all(q_emb, chunk_tokens=None)` sözleşmesini paylaşıyor.
`bench/oracle.py` geriye dönük uyumluluk için buradan re-export eder.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from belge_gozu.index.chunking import chunk_bounds
from belge_gozu.index.manifest import IndexManifest, read_manifest, write_manifest


@dataclass
class FloatIndex:
    """f16 master token embedding'leri: embs.npy (toplam_token, 128) float16 +
    offsets.npy + page_ids.json + manifest.json (quantization="float16").

    T5'teki PackedIndex ile aynı sayfa-hizalı chunk deseni kullanılır; tek fark
    Hamming yerine gerçek dot-product MaxSim (bkz. `score_all`)."""

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

    def score_all(self, q_emb: np.ndarray, chunk_tokens: int | None = None) -> np.ndarray:
        """(n_pages,) — float MaxSim, sorgu jetonu başına ortalama (~[-1,1]).

        T5'teki `PackedIndex.score_all` deseniyle aynı: sayfa-hizalı chunk'lar
        üstünde `np.maximum.reduceat`; tek fark Hamming yerine `q @ chunk.T`
        gerçek dot-product'ı (f16 saklamadan float32'ye açılarak). Ölçek zaten
        normalize: birim-norm token'larda kosinüs benzerliği, yani ~[-1,1] —
        PackedIndex'in 128'e bölünmüş Hamming skoruyla AYNI bantta.

        `chunk_tokens=None` -> `chunk_bounds`'ın çağrı anında okuduğu ortak
        `index.chunking.CHUNK_TOKENS` (chunk sınırı sonucu değiştirmez).
        """
        q = np.asarray(q_emb, dtype=np.float32)
        offsets = np.asarray(self.offsets)
        n_pages = len(self.page_ids)
        out = np.empty(n_pages, dtype=np.float64)
        bounds = chunk_bounds(offsets, chunk_tokens)
        for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
            t0, t1 = int(offsets[b0]), int(offsets[b1])
            chunk = np.asarray(self.embs[t0:t1], dtype=np.float32)
            sim = q @ chunk.T  # (n_q, chunk_tokens)
            starts = (offsets[b0:b1] - t0).astype(np.int64)
            # offsets kesin artan (FloatIndex.build sıfır-token sayfayı reddeder) ->
            # reduceat boş segment göremez.
            out[b0:b1] = np.maximum.reduceat(sim, starts, axis=1).sum(axis=0)
        return out / max(1, q.shape[0])

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
