import numpy as np
import pytest

from belge_gozu.index.manifest import (
    CPE_0_3_18_DOC_PROMPT,
    TRAIN_COMPAT_DOC_PROMPT,
    TRAIN_COMPAT_V1,
)


class FakeTorchLike:
    """_run'ın maske kırpma sözleşmesini gerçek model olmadan sınamak için
    ColSmolEncoder._trim_by_mask saf fonksiyonu test edilir."""


def test_trim_by_mask_drops_padding_rows():
    from belge_gozu.index.encode import trim_by_mask

    emb = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    mask = np.array([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=np.int64)  # sol-pad
    out = trim_by_mask(emb, mask)
    assert [o.shape for o in out] == [(4, 3), (2, 3)]
    np.testing.assert_array_equal(out[1], emb[1, 2:])


def test_query_format_render_used(monkeypatch):
    """encode_query, processor'a QueryFormat.render çıktısını vermeli."""
    from belge_gozu.index import encode as enc_mod

    captured = {}

    class StubSelf:
        query_format = TRAIN_COMPAT_V1

        class processor:  # noqa: N801
            @staticmethod
            def process_texts(texts):
                captured["texts"] = texts
                return {"batch": True}

        def _run(self, batch):
            return [np.zeros((1, 128), dtype=np.float32)]

    out = enc_mod.ColSmolEncoder.encode_query(StubSelf(), "yerleşim yeri")
    assert captured["texts"] == [TRAIN_COMPAT_V1.render("yerleşim yeri")]
    assert out.shape == (1, 128)


class StubProcessor:
    """colpali-engine ColIdefics3Processor'ın ilgili yüzeyi: ClassVar prompt +
    onu okuyan process_images."""

    visual_prompt_prefix = CPE_0_3_18_DOC_PROMPT

    def __init__(self):
        self.seen: list[str] = []

    def process_images(self, images):
        self.seen = [self.visual_prompt_prefix] * len(images)
        return {"batch": True}


class StubEncoderSelf:
    def __init__(self, processor):
        self.processor = processor

    def _run(self, batch):
        return [np.zeros((1, 128), dtype=np.float32)]


def test_visual_prompt_override_reaches_process_images():
    """override verilince process_images'a giden text değişmeli; ClassVar bozulmamalı."""
    from belge_gozu.index.encode import ColSmolEncoder, apply_visual_prompt_override

    proc = StubProcessor()
    effective = apply_visual_prompt_override(proc, TRAIN_COMPAT_DOC_PROMPT)
    assert effective == TRAIN_COMPAT_DOC_PROMPT

    ColSmolEncoder.encode_pages(StubEncoderSelf(proc), [object()])
    assert proc.seen == [TRAIN_COMPAT_DOC_PROMPT]
    # örnek attr yalnız bu örneği gölgeler
    assert StubProcessor.visual_prompt_prefix == CPE_0_3_18_DOC_PROMPT


def test_visual_prompt_override_none_keeps_processor_default():
    from belge_gozu.index.encode import ColSmolEncoder, apply_visual_prompt_override

    proc = StubProcessor()
    assert apply_visual_prompt_override(proc, None) == CPE_0_3_18_DOC_PROMPT
    ColSmolEncoder.encode_pages(StubEncoderSelf(proc), [object()])
    assert proc.seen == [CPE_0_3_18_DOC_PROMPT]


@pytest.mark.slow
def test_batch_vs_single_sign_determinism():
    """Batch içinde (padding'li) ve tek başına encode edilen sayfa, maske
    kırpması sonrası SIGN düzeyinde büyük ölçüde aynı olmalı.

    ==1.0 MPS'te tutmadı (ölçüm: 0.9990/0.9989, 2026-08-26); karar: index
    build batch_size=1 (cli.py). Bu eşik yalnız kaba gerilemeleri (ör.
    maskeleme bozulması) yakalar; bit-exact kilit T10 canary testlerinde."""
    from pathlib import Path

    from PIL import Image

    from belge_gozu.index.encode import ColSmolEncoder

    enc = ColSmolEncoder("vidore/colSmol-500M", "auto")
    root = Path("data")
    # k6098:134 v0 indeksinde padding satırı olan sayfalardan biri (bulgu 20)
    paths = ["images/k6098/0134.webp", "images/k4721/0004.webp", "images/rg1965a/0001.webp"]
    imgs = [Image.open(root / p).convert("RGB") for p in paths]
    batch_out = enc.encode_pages(imgs)  # tek batch (karışık boyut)
    single_out = [enc.encode_pages([im])[0] for im in imgs]
    for b, s, p in zip(batch_out, single_out, paths, strict=True):
        assert b.shape == s.shape, f"{p}: sekans uzunluğu batch'e bağlı olmamalı"
        agree = float(((b > 0) == (s > 0)).mean())
        assert agree >= 0.995, f"{p}: sign uyuşması {agree:.4f} < 0.995 -> build batch=1 kararı"
