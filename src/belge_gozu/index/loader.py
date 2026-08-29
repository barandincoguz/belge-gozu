"""Temsil-farkında indeks yükleyici (T14).

Serve/bench yolları eskiden doğrudan `PackedIndex.load` çağırıyordu: hangi
kuantizasyonun diskte olduğuna bakılmaksızın `tokens.npy` bekleniyordu, yani
int8/float16 indeksler üretimde HİÇ yüklenemiyordu (ruling R16/D1 —
ölçümlerde kazanan int8'in eksik olan tek parçası buydu).

`load_scorable_index` manifest'teki `quantization` alanına bakar ve doğru
sınıfı döner. Üç sınıf da aynı `ScorableIndex` sözleşmesini sağlar:
`page_ids` + `score_all(q_emb, chunk_tokens=None)` -> normalize [-1,1]
skorlar (bkz. `index/store.py::PackedIndex.score_all` — binary kol T14'te
128'e bölünerek aynı banda taşındı).
"""

from pathlib import Path
from typing import Protocol

import numpy as np

from belge_gozu.index.compat import IndexCompatibilityError
from belge_gozu.index.float_store import FloatIndex
from belge_gozu.index.manifest import Quantization, read_manifest
from belge_gozu.index.quantize import Int8Index
from belge_gozu.index.store import PackedIndex

# manifest.quantization -> yükleyici. `index derive --quant` ve
# `index build --precision` bu değerleri yazar (bkz. cli.py). Anahtarlar
# ham dize değil `Quantization` üyeleri: yeni bir temsil enum'a eklenip
# buraya eklenmezse `_LOADERS.get` None döner ve açık hata verilir.
_LOADERS = {
    Quantization.sign_1bit: PackedIndex.load,
    Quantization.int8: Int8Index.load,
    Quantization.float16: FloatIndex.load,
}
_SUPPORTED = sorted(q.value for q in _LOADERS)


class ScorableIndex(Protocol):
    """Getiricinin bir indeksten ihtiyaç duyduğu HER ŞEY.

    `ExhaustiveRetriever` yalnız bu iki üyeyi kullanır; böylece packed/int8/
    float indeksler getirici tarafında ayrıştırılmadan çalışır."""

    page_ids: list[str]

    def score_all(self, q_emb: np.ndarray, chunk_tokens: int | None = None) -> np.ndarray:
        """(n_pages,) — normalize [-1,1] skorlar (sorgu jetonu başına ortalama MaxSim)."""
        ...


def load_scorable_index(dir: Path) -> PackedIndex | Int8Index | FloatIndex:
    """Manifest'teki `quantization`'a göre doğru indeks sınıfını yükler.

    Manifest yoksa: `tokens.npy` varsa eski (v0, manifest'siz) paketli indeks
    kabul edilir — `PackedIndex.load` ile yüklenir ki manifest'siz indeksler
    `allow_index_mismatch` altında hâlâ servis edilebilsin. Yoksa hata."""
    manifest = read_manifest(dir)
    if manifest is None:
        if (dir / "tokens.npy").exists():
            return PackedIndex.load(dir)  # v0 legacy: manifest'siz sign-1bit
        raise IndexCompatibilityError(
            f"indeks yüklenemedi: {dir} — manifest.json yok ve eski paketli indeks "
            "işareti olan tokens.npy de yok. Beklenen: manifest.json (quantization "
            f"∈ {_SUPPORTED}) ya da legacy tokens.npy. Çözüm: "
            "`belge-gozu index build` (ya da f16 master'dan `belge-gozu index derive`)."
        )
    try:
        quant = Quantization(manifest.quantization)
    except ValueError:
        raise IndexCompatibilityError(
            f"indeks yüklenemedi: {dir} — manifest'teki quantization="
            f"{manifest.quantization!r} tanınmıyor; desteklenen: {_SUPPORTED}"
        ) from None
    return _LOADERS[quant](dir)
