import math

import numpy as np
import pytest

from belge_gozu.bench.calibration_metrics import (
    abstain_precision_recall,
    auroc,
    brier,
    clopper_pearson_upper_bound,
    conformal_threshold,
    ece,
    false_answer_rate_on_unanswerable,
    risk_coverage,
    risk_coverage_auc,
    selective_accuracy,
    wilson_upper_bound,
)

# ---------------------------------------------------------------------------
# brier
# ---------------------------------------------------------------------------


def test_brier_perfect_and_partial():
    assert brier(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert brier(np.array([0.9, 0.2]), np.array([1.0, 0.0])) == pytest.approx(0.025)


def test_brier_worst_case_is_one():
    assert brier(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_brier_rejects_empty_mismatched_and_out_of_range():
    with pytest.raises(ValueError):
        brier(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        brier(np.array([0.1, 0.2]), np.array([1.0]))
    with pytest.raises(ValueError):
        brier(np.array([0.1]), np.array([0.5]))  # label 0/1 dışı
    with pytest.raises(ValueError):
        brier(np.array([1.5]), np.array([1.0]))  # prob [0,1] dışı


# ---------------------------------------------------------------------------
# ece
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration_is_zero():
    probs = np.array([1 / 3, 1 / 3, 1 / 3])
    labels = np.array([1.0, 0.0, 0.0])
    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-12)


def test_ece_hand_computed_two_bins():
    # bin(0.1) idx=1: conf=0.1, acc=mean([0,0])=0.0 -> fark 0.1, ağırlık 2/4
    # bin(0.8) idx=8: conf=0.8, acc=mean([1,0])=0.5 -> fark 0.3, ağırlık 2/4
    # ece = 0.5*0.1 + 0.5*0.3 = 0.2
    probs = np.array([0.1, 0.1, 0.8, 0.8])
    labels = np.array([0.0, 0.0, 1.0, 0.0])
    assert ece(probs, labels) == pytest.approx(0.2)


def test_ece_rejects_bad_n_bins_and_empty():
    with pytest.raises(ValueError):
        ece(np.array([0.1]), np.array([1.0]), n_bins=0)
    with pytest.raises(ValueError):
        ece(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# auroc
# ---------------------------------------------------------------------------


def test_auroc_perfect_separation():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(probs, labels) == pytest.approx(1.0)


def test_auroc_all_tied_scores_is_one_half():
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert auroc(probs, labels) == pytest.approx(0.5)


def test_auroc_requires_both_classes_present():
    with pytest.raises(ValueError):
        auroc(np.array([0.1, 0.2]), np.array([1, 1]))
    with pytest.raises(ValueError):
        auroc(np.array([0.1, 0.2]), np.array([0, 0]))


def test_auroc_rejects_empty():
    with pytest.raises(ValueError):
        auroc(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# risk_coverage / risk_coverage_auc
# ---------------------------------------------------------------------------


def test_risk_coverage_hand_computed_points():
    probs = np.array([0.2, 0.4, 0.6])
    labels = np.array([1, 0, 1])
    points = risk_coverage(probs, labels)
    assert len(points) == 4  # 3 benzersiz prob + 1 sentinel (all-abstain ucu)

    taus = [t for t, _, _ in points]
    coverages = [c for _, c, _ in points]
    assert taus == sorted(taus)
    # tau artık ⇒ coverage kesin artmaz
    assert all(c1 >= c2 for c1, c2 in zip(coverages, coverages[1:], strict=False))

    # tau=0.2 (all-answer): coverage=1.0, risk = 1/3 (yalnız orta eleman yanlış)
    assert points[0][1:] == pytest.approx((1.0, 1 / 3))
    # tau=0.4: coverage=2/3, risk=1/2 (iki yanıttan biri yanlış)
    assert points[1][1:] == pytest.approx((2 / 3, 0.5))
    # tau=0.6: coverage=1/3, risk=0.0 (tek yanıt doğru)
    assert points[2][1:] == pytest.approx((1 / 3, 0.0))


def test_risk_coverage_all_abstain_endpoint_is_finite_not_inf_or_nan():
    probs = np.array([0.2, 0.4, 0.6])
    labels = np.array([1, 0, 1])
    last_tau, last_cov, last_risk = risk_coverage(probs, labels)[-1]
    assert math.isfinite(last_tau)  # sentinel sonlu: JSON'a yazılabilir (math.inf DEĞİL)
    assert last_tau > 0.6
    assert (last_cov, last_risk) == (0.0, 0.0)


def test_risk_coverage_handles_single_class_without_nan():
    probs = np.array([0.3, 0.6, 0.9])
    labels = np.array([1.0, 1.0, 1.0])  # hiç hata yok
    points = risk_coverage(probs, labels)
    assert all(not math.isnan(r) for _, _, r in points)
    assert all(r == 0.0 for _, _, r in points)


def test_risk_coverage_rejects_empty():
    with pytest.raises(ValueError):
        risk_coverage(np.array([]), np.array([]))


def test_risk_coverage_auc_hand_computed_trapezoid():
    # coverage 0 -> 1/3 -> 2/3 -> 1 (eşit aralık 1/3); risk 0 -> 0 -> 0.5 -> 1/3
    # alan = (0+0)/2*(1/3) + (0+0.5)/2*(1/3) + (0.5+1/3)/2*(1/3) = 2/9
    points = [
        (0.6000000000000001, 0.0, 0.0),
        (0.6, 1 / 3, 0.0),
        (0.4, 2 / 3, 0.5),
        (0.2, 1.0, 1 / 3),
    ]
    assert risk_coverage_auc(points) == pytest.approx(2 / 9)


def test_risk_coverage_auc_matches_manual_sweep():
    probs = np.array([0.2, 0.4, 0.6])
    labels = np.array([1, 0, 1])
    assert risk_coverage_auc(risk_coverage(probs, labels)) == pytest.approx(2 / 9)


def test_risk_coverage_auc_is_order_independent():
    points = [(0.2, 1.0, 1 / 3), (0.4, 2 / 3, 0.5), (0.6, 1 / 3, 0.0), (0.7, 0.0, 0.0)]
    shuffled = [points[2], points[0], points[3], points[1]]
    assert risk_coverage_auc(points) == pytest.approx(risk_coverage_auc(shuffled))


def test_risk_coverage_auc_rejects_fewer_than_two_points():
    with pytest.raises(ValueError):
        risk_coverage_auc([])
    with pytest.raises(ValueError):
        risk_coverage_auc([(0.5, 1.0, 0.1)])


# ---------------------------------------------------------------------------
# selective_accuracy / abstain_precision_recall
# ---------------------------------------------------------------------------


def test_selective_accuracy_known_values():
    probs = np.array([0.2, 0.4, 0.6])
    labels = np.array([1, 0, 1])
    assert selective_accuracy(probs, labels, 0.4) == pytest.approx(0.5)
    assert selective_accuracy(probs, labels, 0.6) == pytest.approx(1.0)


def test_selective_accuracy_all_abstain_raises_not_nan():
    probs = np.array([0.2, 0.4, 0.6])
    labels = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        selective_accuracy(probs, labels, 0.7)  # max(probs)=0.6 altında hiç yanıt yok


def test_selective_accuracy_rejects_empty():
    with pytest.raises(ValueError):
        selective_accuracy(np.array([]), np.array([]), 0.5)


def test_abstain_precision_recall_hand_computed():
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1, 1])
    assert abstain_precision_recall(probs, labels, 0.5) == pytest.approx((1.0, 1.0))
    assert abstain_precision_recall(probs, labels, 0.6) == pytest.approx((2 / 3, 1.0))


def test_abstain_precision_recall_all_answer_raises_not_nan():
    probs = np.array([0.2, 0.4, 0.6])
    labels = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        abstain_precision_recall(probs, labels, 0.2)  # min(probs)=0.2 -> kimse abstain etmiyor


def test_abstain_precision_recall_no_negative_labels_raises_not_nan():
    probs = np.array([0.2, 0.4])
    labels = np.array([1, 1])  # hiç label==0 yok -> recall tanımsız
    with pytest.raises(ValueError):
        abstain_precision_recall(probs, labels, 0.3)


def test_abstain_precision_recall_rejects_empty():
    with pytest.raises(ValueError):
        abstain_precision_recall(np.array([]), np.array([]), 0.5)


# ---------------------------------------------------------------------------
# conformal_threshold
# ---------------------------------------------------------------------------


def test_conformal_threshold_hand_computed_quantile():
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    labels = np.zeros(9)
    # rank = ceil((9+1)*0.8) = 8 -> 8. sıra istatistiği (1-indeksli) = 0.8
    assert conformal_threshold(errors, labels, alpha=0.2) == pytest.approx(0.8)


def test_conformal_threshold_saturates_to_max_when_n_too_small():
    errors = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.zeros(4)
    # rank = ceil(5*0.95) = 5, n=4'e kırpılır -> en büyük hata confidence'ı
    assert conformal_threshold(errors, labels, alpha=0.05) == pytest.approx(0.4)


def test_conformal_threshold_ignores_correct_examples():
    # label==1 (doğru) örnekler, confidence'ları ne olursa olsun kalibrasyon
    # kümesine karışmamalı: yalnız label==0 alt kümesi hesaba katılır.
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.99, 0.99, 0.99, 0.99, 0.99])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert conformal_threshold(probs, labels, alpha=0.05) == pytest.approx(0.4)


def test_conformal_threshold_rejects_no_errors_bad_alpha_and_empty():
    with pytest.raises(ValueError):
        conformal_threshold(np.array([0.5, 0.6]), np.array([1.0, 1.0]))  # hiç hata yok
    with pytest.raises(ValueError):
        conformal_threshold(np.array([0.5]), np.array([0.0]), alpha=0.0)
    with pytest.raises(ValueError):
        conformal_threshold(np.array([0.5]), np.array([0.0]), alpha=1.0)
    with pytest.raises(ValueError):
        conformal_threshold(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# wilson_upper_bound / clopper_pearson_upper_bound
# ---------------------------------------------------------------------------


def test_clopper_pearson_zero_errors_n150_closed_form_matches_rule_of_three():
    # successes=0 kapalı formu: 1 - alpha^(1/n) = 1 - 0.05**(1/150)
    expected = 1 - 0.05 ** (1 / 150)
    assert clopper_pearson_upper_bound(0, 150) == pytest.approx(expected, abs=1e-9)
    assert clopper_pearson_upper_bound(0, 150) == pytest.approx(0.0198, abs=5e-4)  # G2.1 ~%2


def test_wilson_zero_errors_n150_closed_form():
    # successes=0 kapalı formu: z^2 / (n + z^2), z = Φ^{-1}(0.95) (tek yönlü)
    z = 1.6448536269514722
    expected = z * z / (150 + z * z)
    assert wilson_upper_bound(0, 150) == pytest.approx(expected, abs=1e-9)


def test_clopper_pearson_more_conservative_than_wilson():
    assert clopper_pearson_upper_bound(0, 150) > wilson_upper_bound(0, 150)
    assert clopper_pearson_upper_bound(1, 10) > wilson_upper_bound(1, 10)


def test_binomial_bound_saturates_to_one_when_all_successes():
    assert wilson_upper_bound(10, 10) == pytest.approx(1.0)
    assert clopper_pearson_upper_bound(10, 10) == pytest.approx(1.0)


def test_binomial_bound_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_upper_bound(-1, 10)
    with pytest.raises(ValueError):
        wilson_upper_bound(11, 10)
    with pytest.raises(ValueError):
        wilson_upper_bound(1, 0)
    with pytest.raises(ValueError):
        clopper_pearson_upper_bound(1, 10, confidence=1.0)
    with pytest.raises(ValueError):
        clopper_pearson_upper_bound(1, 10, confidence=0.0)


# ---------------------------------------------------------------------------
# false_answer_rate_on_unanswerable (G2.1)
# ---------------------------------------------------------------------------


def test_false_answer_rate_zero_errors_n150_upper_bound_near_two_percent():
    confidence = np.full(150, 0.1)
    answerable = np.zeros(150, dtype=bool)
    result = false_answer_rate_on_unanswerable(confidence, answerable, tau=0.5)
    assert result[:3] == pytest.approx((0.0, 150, 0))
    assert result[3] == pytest.approx(0.0198, abs=5e-4)


def test_false_answer_rate_counts_only_unanswerable_subset():
    # 150 cevaplanamaz (3'ü yüksek confidence'la yanlışlıkla "yanıtlanmış") +
    # 10 cevaplanabilir (confidence'ları ne olursa olsun sayılmamalı)
    confidence = np.concatenate([np.full(147, 0.1), np.full(3, 0.9), np.full(10, 0.95)])
    answerable = np.concatenate([np.zeros(150, dtype=bool), np.ones(10, dtype=bool)])
    rate, n, errors, upper = false_answer_rate_on_unanswerable(confidence, answerable, tau=0.5)
    assert (rate, n, errors) == pytest.approx((0.02, 150, 3))
    assert upper > rate


def test_false_answer_rate_method_wilson_vs_clopper_pearson():
    confidence = np.full(150, 0.1)
    answerable = np.zeros(150, dtype=bool)
    cp = false_answer_rate_on_unanswerable(confidence, answerable, 0.5, method="clopper_pearson")
    wilson = false_answer_rate_on_unanswerable(confidence, answerable, 0.5, method="wilson")
    assert wilson[3] < cp[3]


def test_false_answer_rate_rejects_no_unanswerable_shape_mismatch_and_empty():
    with pytest.raises(ValueError):
        false_answer_rate_on_unanswerable(np.array([0.5, 0.6]), np.array([True, True]), 0.5)
    with pytest.raises(ValueError):
        false_answer_rate_on_unanswerable(np.array([0.5, 0.6]), np.array([True]), 0.5)
    with pytest.raises(ValueError):
        false_answer_rate_on_unanswerable(np.array([]), np.array([]), 0.5)
