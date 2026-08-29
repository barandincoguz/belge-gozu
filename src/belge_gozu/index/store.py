import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from belge_gozu.index.chunking import CHUNK_TOKENS, EMBED_DIM, chunk_bounds
from belge_gozu.index.manifest import IndexManifest, read_manifest, write_manifest

PACKED_BYTES = EMBED_DIM // 8  # 128 bit/token -> 16 bayt


def binarize_pack(emb: np.ndarray) -> np.ndarray:
    if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
        raise ValueError(f"beklenen (n,{EMBED_DIM}), gelen {emb.shape}")
    return np.packbits((emb > 0).astype(np.uint8), axis=1)


def as_u64(packed: np.ndarray) -> np.ndarray:
    """(n,16) uint8 -> (n,2) uint64 görünümü (popcount'u 16 yerine 2 kelimede yapar).

    Bitişik (contiguous) uint8 dizilerinde kopya YOKTUR; mmap'li bir
    `tokens.npy` de bitişik olduğu için burada da kopya çıkmaz.

    Şekil kontrolü sessiz bir sözleşmeyi açık hale getirir: yanlış genişlikte
    bir dizi `.view(np.uint64)`'te ya anlamsız bir sonuç ya da okuması zor bir
    numpy hatası üretirdi."""
    if packed.ndim != 2 or packed.shape[1] != PACKED_BYTES:
        raise ValueError(f"beklenen (n,{PACKED_BYTES}) uint8, gelen {packed.shape}")
    return np.ascontiguousarray(packed).view(np.uint64)


@dataclass
class PackedIndex:
    tokens: np.ndarray
    offsets: np.ndarray
    page_vecs: np.ndarray
    page_ids: list[str]
    manifest: IndexManifest | None = None

    # index.chunking.CHUNK_TOKENS ile aynı varsayılan; test override'ı için
    # instance üstünde değiştirilebilir (bkz. index/quantize.py'deki aynı desen).
    CHUNK_TOKENS: ClassVar[int] = CHUNK_TOKENS

    @classmethod
    def build(
        cls,
        page_ids: list[str],
        embs: list[np.ndarray],
        manifest: IndexManifest | None = None,
    ) -> "PackedIndex":
        if len(page_ids) != len(embs):
            raise ValueError(
                f"page_ids ({len(page_ids)}) ve embs ({len(embs)}) uzunlukları eşleşmiyor"
            )
        if not embs:
            raise ValueError("boş korpus: en az bir sayfa embedding'i gerekli")
        for pid, e in zip(page_ids, embs, strict=True):
            if e.shape[0] == 0:
                raise ValueError(f"sıfır token'lı sayfa: {pid}")
            if (np.abs(e).sum(axis=1) == 0).any():
                raise ValueError(f"padding satırı sızmış: {pid}")
        packed = [binarize_pack(e) for e in embs]
        offsets = np.zeros(len(embs) + 1, dtype=np.int64)
        np.cumsum([p.shape[0] for p in packed], out=offsets[1:])
        page_vecs = np.vstack([binarize_pack(e.mean(axis=0, keepdims=True)) for e in embs])
        return cls(np.vstack(packed), offsets, page_vecs, list(page_ids), manifest)

    def page_tokens(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i] : self.offsets[i + 1]]

    def score_all(self, q_emb: np.ndarray, chunk_tokens: int | None = None) -> np.ndarray:
        """(n_pages,) — binary MaxSim, sorgu jetonu başına ortalama (~[-1,1]).

        T14 (tek skor ölçeği): çekirdek eskiden
        `ExhaustiveBinaryRetriever.score_all` içindeydi; buraya taşındı ki
        getirici HERHANGİ bir indeks tipini (packed/int8/float) aynı
        sözleşmeyle skorlayabilsin. Tek matematik değişikliği en sondaki
        **EMBED_DIM'e bölme**: jeton başına ham skor `EMBED_DIM - 2*hamming`
        [-EMBED_DIM, EMBED_DIM] bandındaydı; bölününce Int8Index/FloatIndex'in
        dot-product skorlarıyla AYNI normalize [-1,1] bandına oturur. (Bu
        kuantizasyon temsilleri arasında karşılaştırılabilirlik içindir;
        kalibrasyon DEĞİLDİR — bkz. data/bench/results/int8-threshold-transfer.json.)

        `chunk_tokens=None` -> `self.CHUNK_TOKENS` (instance üstünde override
        edilebilir); chunk sınırı sonucu değiştirmez.
        """
        q_packed = binarize_pack(q_emb)
        qa = as_u64(q_packed)
        ta = as_u64(np.asarray(self.tokens))
        offsets = np.asarray(self.offsets)
        n_pages = len(self.page_ids)
        out = np.empty(n_pages, dtype=np.float64)
        resolved = chunk_tokens if chunk_tokens is not None else self.CHUNK_TOKENS
        bounds = chunk_bounds(offsets, resolved)
        for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
            t0, t1 = int(offsets[b0]), int(offsets[b1])
            ham = np.bitwise_count(qa[:, None, :] ^ ta[None, t0:t1, :]).sum(axis=2, dtype=np.int32)
            sim = EMBED_DIM - 2 * ham
            starts = (offsets[b0:b1] - t0).astype(np.int64)
            # offsets kesin artan (PackedIndex.build sıfır-token sayfayı reddeder) ->
            # reduceat boş segment göremez.
            out[b0:b1] = np.maximum.reduceat(sim, starts, axis=1).sum(axis=0)
        return out / max(1, q_emb.shape[0]) / EMBED_DIM

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / "tokens.npy", self.tokens)
        np.save(dir / "offsets.npy", self.offsets)
        np.save(dir / "page_vecs.npy", self.page_vecs)
        (dir / "page_ids.json").write_text(json.dumps(self.page_ids, ensure_ascii=False))
        if self.manifest is not None:
            write_manifest(dir, self.manifest)

    @classmethod
    def load(cls, dir: Path, mmap: bool = True) -> "PackedIndex":
        mode = "r" if mmap else None
        return cls(
            tokens=np.load(dir / "tokens.npy", mmap_mode=mode),
            offsets=np.load(dir / "offsets.npy"),
            page_vecs=np.load(dir / "page_vecs.npy"),
            page_ids=json.loads((dir / "page_ids.json").read_text()),
            manifest=read_manifest(dir),
        )
