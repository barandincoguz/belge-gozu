import numpy as np
from PIL import Image

from belge_gozu.index.encode import FakeEncoder


def make_img(color: int) -> Image.Image:
    return Image.new("RGB", (32, 32), (color, 0, 0))


def test_fake_encoder_shapes_and_determinism():
    enc = FakeEncoder(tokens_per_item=8)
    a1, b1 = enc.encode_pages([make_img(10), make_img(20)])
    a2, _ = enc.encode_pages([make_img(10), make_img(20)])
    assert a1.shape == (8, 128) and a1.dtype == np.float32
    np.testing.assert_array_equal(a1, a2)  # aynı girdi → aynı embedding
    assert not np.array_equal(a1, b1)  # farklı girdi → farklı embedding
    q1 = enc.encode_query("kira artışı")
    q2 = enc.encode_query("kira artışı")
    np.testing.assert_array_equal(q1, q2)
    assert q1.shape == (8, 128)
