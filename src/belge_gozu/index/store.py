import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def binarize_pack(emb: np.ndarray) -> np.ndarray:
    if emb.ndim != 2 or emb.shape[1] != 128:
        raise ValueError(f"beklenen (n,128), gelen {emb.shape}")
    return np.packbits((emb > 0).astype(np.uint8), axis=1)


@dataclass
class PackedIndex:
    tokens: np.ndarray
    offsets: np.ndarray
    page_vecs: np.ndarray
    page_ids: list[str]

    @classmethod
    def build(cls, page_ids: list[str], embs: list[np.ndarray]) -> "PackedIndex":
        if len(page_ids) != len(embs):
            raise ValueError(
                f"page_ids ({len(page_ids)}) ve embs ({len(embs)}) uzunlukları eşleşmiyor"
            )
        if not embs:
            raise ValueError("boş korpus: en az bir sayfa embedding'i gerekli")
        for pid, e in zip(page_ids, embs, strict=True):
            if e.shape[0] == 0:
                raise ValueError(f"sıfır token'lı sayfa: {pid}")
        packed = [binarize_pack(e) for e in embs]
        offsets = np.zeros(len(embs) + 1, dtype=np.int64)
        np.cumsum([p.shape[0] for p in packed], out=offsets[1:])
        page_vecs = np.vstack([binarize_pack(e.mean(axis=0, keepdims=True)) for e in embs])
        return cls(np.vstack(packed), offsets, page_vecs, list(page_ids))

    def page_tokens(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i] : self.offsets[i + 1]]

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / "tokens.npy", self.tokens)
        np.save(dir / "offsets.npy", self.offsets)
        np.save(dir / "page_vecs.npy", self.page_vecs)
        (dir / "page_ids.json").write_text(json.dumps(self.page_ids, ensure_ascii=False))

    @classmethod
    def load(cls, dir: Path, mmap: bool = True) -> "PackedIndex":
        mode = "r" if mmap else None
        return cls(
            tokens=np.load(dir / "tokens.npy", mmap_mode=mode),
            offsets=np.load(dir / "offsets.npy"),
            page_vecs=np.load(dir / "page_vecs.npy"),
            page_ids=json.loads((dir / "page_ids.json").read_text()),
        )
