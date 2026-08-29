import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from belge_gozu.index.chunking import CHUNK_TOKENS, EMBED_DIM, INT8_MAX, chunk_bounds
from belge_gozu.index.float_store import FloatIndex
from belge_gozu.index.manifest import IndexManifest, read_manifest, write_manifest
from belge_gozu.index.store import PackedIndex


def derive_packed(findex: FloatIndex) -> PackedIndex:
    """f16 master -> sign-1bit (T3 `PackedIndex`).

    Her sayfanın f16 token'ları float32'ye açılıp `binarize_pack` uygulanır;
    all-zero satır f16 master'da zaten yok (T2 mask policy padding'i düşürür),
    bu yüzden `PackedIndex.build`'in o kontrolü burada da güvenle çalışır.
    Manifest varsa `quantization="sign-1bit"` ile taşınır, yoksa None kalır."""
    embs = [
        np.asarray(findex.page_tokens(i), dtype=np.float32) for i in range(len(findex.page_ids))
    ]
    manifest = (
        findex.manifest.model_copy(update={"quantization": "sign-1bit"})
        if findex.manifest is not None
        else None
    )
    return PackedIndex.build(list(findex.page_ids), embs, manifest=manifest)


@dataclass
class Int8Index:
    """Per-token simetrik ölçekli int8 kuantizasyon: scale_t = max|x_t| / INT8_MAX,
    q_t = round(x_t / scale_t). Skor: chunk'lar float32'ye açılıp T5/oracle
    desenindeki aynı sayfa-hizalı MaxSim (`np.maximum.reduceat`) uygulanır --
    saklama küçülür, hesap yine float."""

    codes: np.ndarray  # (toplam_token, EMBED_DIM) int8
    scales: np.ndarray  # (toplam_token,) float32
    offsets: np.ndarray
    page_ids: list[str]
    manifest: IndexManifest | None = None

    # index.chunking.CHUNK_TOKENS ile aynı varsayılan; test override'ı için
    # instance üstünde değiştirilebilir (bkz. retrieval/core.py'deki aynı desen).
    CHUNK_TOKENS: ClassVar[int] = CHUNK_TOKENS

    @classmethod
    def derive(cls, findex: FloatIndex, chunk_tokens: int | None = None) -> "Int8Index":
        """Sayfa-hizalı chunk'lar halinde işler (bkz. `chunk_bounds`): her
        chunk yalnız kendi float32 kopyasını (chunk_tokens*128*4 byte tepe)
        tutar, tüm korpusu float32'ye açan ~4 tam kopya YERİNE (review R1
        IMPORTANT-2 — 4222 sayfa x ~871 token'da bu tepe belleği 5.5-6.5 GB'a
        çıkarıyordu). Her satırın kuantizasyonu kendi içinde bağımsız olduğu
        için chunk sınırları sonucu etkilemez (score_all'daki reduceat'ten
        farklı olarak burada satırlar arası indirgeme yok)."""
        offsets = np.asarray(findex.offsets)
        total_tokens = int(offsets[-1])
        codes = np.empty((total_tokens, EMBED_DIM), dtype=np.int8)
        scales = np.empty(total_tokens, dtype=np.float32)
        resolved_chunk = chunk_tokens if chunk_tokens is not None else cls.CHUNK_TOKENS
        bounds = chunk_bounds(offsets, resolved_chunk)
        for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
            t0, t1 = int(offsets[b0]), int(offsets[b1])
            chunk = np.asarray(findex.embs[t0:t1], dtype=np.float32)  # kopya, yalnız bu chunk
            abs_max = np.abs(chunk).max(axis=1)
            chunk_scale = np.maximum(abs_max / np.float32(INT8_MAX), np.float32(1e-8)).astype(
                np.float32
            )
            np.divide(chunk, chunk_scale[:, None], out=chunk)
            np.round(chunk, out=chunk)
            np.clip(chunk, -INT8_MAX, INT8_MAX, out=chunk)
            codes[t0:t1] = chunk.astype(np.int8)
            scales[t0:t1] = chunk_scale
        manifest = (
            findex.manifest.model_copy(update={"quantization": "int8"})
            if findex.manifest is not None
            else None
        )
        return cls(codes, scales, offsets, list(findex.page_ids), manifest)

    def page_tokens(self, i: int) -> np.ndarray:
        return self.codes[self.offsets[i] : self.offsets[i + 1]]

    def score_all(self, q_emb: np.ndarray, chunk_tokens: int | None = None) -> np.ndarray:
        """(n_pages,) — dequantize edilmiş chunk'larla float MaxSim, T5/oracle
        ile birebir aynı sayfa-hizalı chunk + reduceat deseni.

        Skorlar sorgu jetonu başına ortalamadır (~[-1,1]) — PackedIndex ve
        FloatIndex ile AYNI ölçek; burada matematik değişmedi (T14'te
        normalize edilen taraf binary koldu).

        `chunk_tokens=None` -> `self.CHUNK_TOKENS` (instance üstünde override
        edilebilir — bkz. tests/index/test_quantize.py çoklu-chunk testi)."""
        q = np.asarray(q_emb, dtype=np.float32)
        offsets = np.asarray(self.offsets)
        n_pages = len(self.page_ids)
        out = np.empty(n_pages, dtype=np.float64)
        resolved = chunk_tokens if chunk_tokens is not None else self.CHUNK_TOKENS
        bounds = chunk_bounds(offsets, resolved)
        for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
            t0, t1 = int(offsets[b0]), int(offsets[b1])
            chunk = self.codes[t0:t1].astype(np.float32) * self.scales[t0:t1, None]
            sim = q @ chunk.T  # (n_q, chunk_tokens)
            starts = (offsets[b0:b1] - t0).astype(np.int64)
            out[b0:b1] = np.maximum.reduceat(sim, starts, axis=1).sum(axis=0)
        return out / max(1, q.shape[0])

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / "codes.npy", self.codes)
        np.save(dir / "scales.npy", self.scales)
        np.save(dir / "offsets.npy", self.offsets)
        (dir / "page_ids.json").write_text(json.dumps(self.page_ids, ensure_ascii=False))
        if self.manifest is not None:
            write_manifest(dir, self.manifest)

    @classmethod
    def load(cls, dir: Path, mmap: bool = True) -> "Int8Index":
        mode = "r" if mmap else None
        return cls(
            codes=np.load(dir / "codes.npy", mmap_mode=mode),
            scales=np.load(dir / "scales.npy", mmap_mode=mode),
            offsets=np.load(dir / "offsets.npy"),
            page_ids=json.loads((dir / "page_ids.json").read_text()),
            manifest=read_manifest(dir),
        )
