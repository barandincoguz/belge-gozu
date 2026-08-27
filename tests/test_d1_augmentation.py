import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from d1_augmentation import noaug_format, summarize_arm  # noqa: E402

from belge_gozu.index.manifest import CPE_0_3_18, TRAIN_COMPAT_V1  # noqa: E402


@pytest.mark.parametrize("base", [CPE_0_3_18, TRAIN_COMPAT_V1])
def test_noaug_format_derivation(base):
    fmt = noaug_format(base)
    assert fmt.n_suffix == 0
    assert fmt.format_id == f"{base.format_id}-noaug"
    assert fmt.format_id != base.format_id
    assert fmt.prefix == base.prefix
    assert fmt.suffix_token == base.suffix_token
    assert fmt.trailing_newline == base.trailing_newline
    # render augmentation suffix'i eklemiyor artık, geri kalanı aynı.
    expected = base.prefix + "soru"
    if base.trailing_newline:
        expected += "\n"
    assert fmt.render("soru") == expected


def test_summarize_arm_basic():
    gold_sets = [{"a"}, {"b"}]
    rankings = [["a", "x", "y"], ["x", "b", "y"]]
    summary = summarize_arm(gold_sets, rankings, ks=(1, 2))
    assert summary["n"] == 2
    assert summary["recall_at"][1] == 0.5  # q1 rank1 isabet, q2 rank1 ıska
    assert summary["recall_at"][2] == 1.0  # ikisi de ilk 2'de
    assert summary["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_summarize_arm_empty():
    summary = summarize_arm([], [], ks=(1, 5))
    assert summary == {"recall_at": {1: 0.0, 5: 0.0}, "mrr": 0.0, "n": 0}


def test_summarize_arm_length_mismatch_raises():
    with pytest.raises(ValueError):
        summarize_arm([{"a"}], [["a"], ["b"]])
