"""`load_scorable_index` — manifest'teki temsile göre doğru sınıf (T14).

Bu yükleyici olmadan serve/bench sabit `PackedIndex.load` çağırıyordu: int8
indeks (ölçümün kazananı) üretimde HİÇ yüklenemiyordu. Testler üç temsili de
gerçek dosyalardan (tmp_path) yükler — sınıf seçimi disk düzeninden değil
manifest'ten çıkmalı."""

from pathlib import Path

import numpy as np
import pytest

from belge_gozu.index.compat import IndexCompatibilityError
from belge_gozu.index.float_store import FloatIndex
from belge_gozu.index.loader import load_scorable_index
from belge_gozu.index.manifest import write_manifest
from belge_gozu.index.quantize import Int8Index, derive_packed
from belge_gozu.index.store import PackedIndex
from tests.index.test_manifest import make_manifest


def _embs(n_pages=4, tokens=6, seed=9):
    """BİRİM NORMLU token satırları — gerçek ColPali çıktısı gibi.

    Ölçek iddiası (normalize ~[-1,1]) bu varsayıma dayanır: dot-product
    MaxSim birim vektörlerde kosinüs benzerliğidir. (Binary kol
    `(128-2*ham)/128` ile zaten yapı gereği bu banttadır; float/int8 kolları
    ise girdinin normlu olmasına bağlıdır — bu yüzden fikstür de normlu.)"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_pages):
        e = rng.standard_normal((tokens, 128)).astype(np.float32)
        out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    return out


def _write(index, dir: Path, quantization: str) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    index.save(dir)
    write_manifest(
        dir,
        make_manifest(
            quantization=quantization,
            n_pages=len(index.page_ids),
            n_tokens=int(index.offsets[-1]),
        ),
    )
    return dir


def _fixture_dirs(tmp_path: Path) -> dict[str, Path]:
    embs = _embs()
    ids = [f"d{i}:1" for i in range(len(embs))]
    findex = FloatIndex.build(ids, embs)
    return {
        "float16": _write(findex, tmp_path / "f16", "float16"),
        "sign-1bit": _write(derive_packed(findex), tmp_path / "packed", "sign-1bit"),
        "int8": _write(Int8Index.derive(findex), tmp_path / "int8", "int8"),
    }


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [("sign-1bit", PackedIndex), ("int8", Int8Index), ("float16", FloatIndex)],
)
def test_loads_class_named_by_manifest(tmp_path: Path, quantization: str, expected: type):
    dirs = _fixture_dirs(tmp_path)
    idx = load_scorable_index(dirs[quantization])
    assert isinstance(idx, expected)
    assert idx.manifest is not None and idx.manifest.quantization == quantization


def test_every_representation_satisfies_the_scoring_contract(tmp_path: Path):
    """Sözleşme (ScorableIndex): `page_ids` + normalize [-1,1] `score_all`.

    Üç temsil de AYNI bandı döndürmeli — `min_score_threshold` tek ve ortak
    olduğu için ölçek ayrışması sessiz bir abstain/uydurma hatası olurdu."""
    dirs = _fixture_dirs(tmp_path)
    q = _embs()[1]
    # f16 saklama (ve int8 kuantizasyon) birim normu tam korumaz: kendi
    # sayfasıyla eşleşen sorgu 1.0'ı ~1e-5 aşabilir. Bant iddiası "~[-1,1]",
    # bu yüzden tolerans saklama hatası mertebesinde tutuluyor.
    eps = 1e-4
    for quantization, dir in dirs.items():
        idx = load_scorable_index(dir)
        scores = idx.score_all(q, chunk_tokens=None)
        assert len(scores) == len(idx.page_ids) == 4, quantization
        assert scores.argmax() == 1, quantization  # kendi sayfası top-1
        assert (scores >= -1.0 - eps).all(), quantization
        assert (scores <= 1.0 + eps).all(), quantization


def test_legacy_manifestless_packed_dir_loads(tmp_path: Path):
    """v0 kalıntısı: manifest yok ama tokens.npy var -> PackedIndex.

    (create_app bu indeksi uyumluluk kontrolünde ayrıca reddeder; yükleyici
    onu yüklemekten sorumludur, yargılamaktan değil.)"""
    PackedIndex.build([f"d{i}:1" for i in range(4)], _embs()).save(tmp_path)
    assert not (tmp_path / "manifest.json").exists()
    assert isinstance(load_scorable_index(tmp_path), PackedIndex)


def test_empty_dir_raises_compatibility_error(tmp_path: Path):
    with pytest.raises(IndexCompatibilityError, match="manifest.json yok"):
        load_scorable_index(tmp_path)


def test_unknown_quantization_raises(tmp_path: Path):
    """Manifest tanınmayan bir temsil söylüyorsa sessizce packed'e DÜŞÜLMEZ."""
    dir = _write(PackedIndex.build([f"d{i}:1" for i in range(4)], _embs()), tmp_path, "fp4")
    with pytest.raises(IndexCompatibilityError, match="tanınmıyor"):
        load_scorable_index(dir)


def test_manifest_data_disagreement_raises_readable_error(tmp_path: Path):
    """Manifest "int8" diyor ama diskte yalnız tokens.npy var (yarım kopya /
    elle düzenlenmiş manifest): çıplak `FileNotFoundError: codes.npy` yerine
    ne olduğunu söyleyen Türkçe hata (review M10)."""
    dir = _write(PackedIndex.build([f"d{i}:1" for i in range(4)], _embs()), tmp_path, "int8")
    assert (dir / "tokens.npy").exists() and not (dir / "codes.npy").exists()
    with pytest.raises(IndexCompatibilityError, match="codes.npy"):
        load_scorable_index(dir)
