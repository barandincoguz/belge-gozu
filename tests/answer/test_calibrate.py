"""Güven özellikleri + kalibratör (P2 T5+T6).

Sentetik korpus ELDE HESAPLANABİLİR seçildi: token kümeleri, eşleşme sayıları
ve yönlendirmenin hangi sayfayı öne aldığı tek tek yazılıdır. Kalibratör
tarafında ise ayrılabilir sentetik veri kullanılır — "fit çalışıyor mu?"
sorusunun gerçek veriye ihtiyacı yoktur ve olmamalıdır.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from belge_gozu.answer.calibrate import (
    FEATURE_ORDER,
    GUARANTEE_CP,
    GUARANTEE_NONE,
    CalibrationArtifact,
    CalibrationKeyMismatch,
    Calibrator,
    build_rows,
    calibration_dir,
    calibration_key,
    choose_threshold,
    class_counts,
    conformal_candidate,
    evaluate,
    extract_features,
    feature_matrix,
    fit_calibration,
    load_calibrator,
    load_rows,
    per_question_rows,
    retrieval_context,
    to_vector,
    univariate_auc,
)
from belge_gozu.cli import app
from belge_gozu.index.manifest import CPE_0_3_18, IndexManifest, RenderConfig, write_manifest
from belge_gozu.retrieval.text import BM25Index, extract_doc_name_tokens, recipe_fingerprint

runner = CliRunner()

# --- elde hesaplanan sentetik korpus ---------------------------------------
#
# Sayfa token kümeleri (tokenize: tr_lower -> >=2 harf -> katlama -> stopword
# -> F5 kırpma), reçetenin kendisiyle doğrulandı:
#   k1:1 {kanun, meden}        k1:2 {19, madde, tanim, yeri, yerle}
#   k2:1 {icra, kanun}         k2:2 {53, madde, yeri, yerle}
# Doküman adları (1. sayfa başlığından, _GENERIC çıkarılmış):
#   k1 -> {"meden"},  k2 -> {"icra"}
IDS = ["k1:1", "k1:2", "k2:1", "k2:2"]
TEXTS = [
    "MEDENİ KANUNU\n",
    "MADDE 19 yerleşim yeri tanımı\n",
    "İCRA KANUNU\n",
    "MADDE 53 yerleşim yeri yerleşim yeri yerleşim yeri\n",
]


@pytest.fixture
def channel():
    return BM25Index(IDS, TEXTS), extract_doc_name_tokens(IDS, TEXTS)


def test_feature_order_is_the_measured_five():
    """Ölçülmüş beş özellik; sıra artefaktın sözleşmesi (değişirse anahtar da değişmeli)."""
    assert FEATURE_ORDER == (
        "served_top1",
        "bm25_margin",
        "matched_terms_top1",
        "matched_frac",
        "routed",
    )
    # Reddedilen sinyaller SIZMAMALI: q_len (veri artefaktı) ve görsel (AUC .34).
    assert not [n for n in FEATURE_ORDER if "q_len" in n or "vis" in n]


def test_features_without_routing_are_hand_computed(channel):
    """ "yerleşim yeri" -> hiçbir doküman adı eşleşmiyor; top-1 = k2:2 (3 tekrar)."""
    text, names = channel
    f = extract_features("yerleşim yeri", text, names)
    scores = dict(zip(IDS, text.scores("yerleşim yeri").tolist(), strict=True))

    assert f["routed"] == 0.0
    # sorgu token'ları {yerle, yeri}; k2:2'de İKİSİ de var -> 2/2
    assert f["matched_terms_top1"] == 2.0
    assert f["matched_frac"] == 1.0
    # yönlendirme yokken servis edilen top-1 = kanal top-1
    assert f["served_top1"] == pytest.approx(scores["k2:2"])
    assert f["served_top1"] == pytest.approx(max(scores.values()))
    assert f["bm25_margin"] == pytest.approx(scores["k2:2"] - scores["k1:2"])


def test_served_top1_is_post_routing_and_can_be_below_channel_top1(channel):
    """YÜK TAŞIYAN AYRIM: yönlendirme servis edilen sayfayı değiştirir.

    "Medeni yerleşim yeri" -> ad token'ı {"meden"} sorguda geçtiği için k1
    yönlendirilir; ama saf BM25 top-1'i k2:2'dir (yerle/yeri üç kez geçiyor).
    Eşiğin ve kalibratörün gördüğü skor SERVİS EDİLEN sayfanınkidir — kanalın
    en yüksek skoru değil (config.py review L1).
    """
    text, names = channel
    q = "Medeni yerleşim yeri"
    scores = dict(zip(IDS, text.scores(q).tolist(), strict=True))
    assert max(scores, key=lambda k: scores[k]) == "k2:2"  # kanal top-1 k2 dokümanında

    f = extract_features(q, text, names)
    assert f["routed"] == 1.0
    assert f["served_top1"] == pytest.approx(scores["k1:1"])  # yönlendirilen doküman
    assert f["served_top1"] < max(scores.values())  # kanal top-1'inin ALTINDA

    # margin ve eşleşme sayıları YÖNLENDİRME ÖNCESİ top-1'den (k2:2) okunur:
    # sorgu {meden, yerle, yeri}, k2:2 {53, madde, yeri, yerle} -> 2/3
    assert f["matched_terms_top1"] == 2.0
    assert f["matched_frac"] == pytest.approx(2 / 3)
    assert f["bm25_margin"] == pytest.approx(scores["k2:2"] - scores["k1:1"])


def test_matched_counts_use_distinct_query_tokens(channel):
    """Tek terimli sorgu: 1/1; margin = top1 - 0 (eşleşmeyen ikinci sayfa)."""
    text, names = channel
    f = extract_features("tanımı", text, names)
    scores = text.scores("tanımı")
    assert f["matched_terms_top1"] == 1.0
    assert f["matched_frac"] == 1.0
    assert f["routed"] == 0.0
    assert f["bm25_margin"] == pytest.approx(float(scores.max()))


def test_repeated_query_terms_do_not_inflate_matched_counts(channel):
    """`matched_terms_top1` BENZERSİZ token sayar (QTF_CAP ile aynı gerekçe)."""
    text, names = channel
    once = extract_features("yerleşim yeri", text, names)
    thrice = extract_features("yerleşim yerleşim yerleşim yeri", text, names)
    assert once["matched_terms_top1"] == thrice["matched_terms_top1"] == 2.0
    assert once["matched_frac"] == thrice["matched_frac"] == 1.0


def test_empty_query_yields_zero_fraction_not_nan(channel):
    """Tümü stopword olan sorgu -> matched_frac 0.0 (NaN kalibratöre sızmaz)."""
    text, names = channel
    f = extract_features("ve bu ile", text, names)
    assert f["matched_terms_top1"] == 0.0
    assert f["matched_frac"] == 0.0
    assert all(np.isfinite(v) for v in f.values())


def test_precomputed_scores_give_identical_features(channel):
    """Çalışma anında `search()` zaten skorladı: onu geçmek özellikleri BEDAVA yapar."""
    text, names = channel
    q = "Medeni yerleşim yeri"
    assert extract_features(q, text, names, bm25=text.scores(q)) == extract_features(q, text, names)


def test_misaligned_scores_fail_fast(channel):
    text, names = channel
    with pytest.raises(ValueError, match="hizalı olmalı"):
        retrieval_context("x", text, names, bm25=np.zeros(3))


def test_to_vector_follows_feature_order_and_rejects_missing(channel):
    text, names = channel
    f = extract_features("Medeni yerleşim yeri", text, names)
    vec = to_vector(f)
    assert vec.shape == (5,)
    assert list(vec) == [f[n] for n in FEATURE_ORDER]
    del f["routed"]
    with pytest.raises(KeyError, match="routed"):
        to_vector(f)


# --- kalibratör -------------------------------------------------------------


def _separable(n: int = 60):
    """Ayrılabilir sentetik veri: ilk özellik sınıfı tek başına belirliyor."""
    x = np.linspace(0.0, 1.0, n)
    y = (x > 0.5).astype(np.float64)
    X = np.column_stack([x, np.linspace(-1.0, 1.0, n), x * 2, x / 2, (x > 0.5).astype(float)])
    return X, y


def test_fit_recovers_separable_boundary():
    from belge_gozu.bench.calibration_metrics import auroc

    X, y = _separable()
    cal = Calibrator.fit(X, y)
    probs = cal.predict_proba(X)
    assert auroc(probs, y) == 1.0
    assert cal.fit_info["converged"] or cal.fit_info["n_iter"] == cal.fit_info["max_iter"]
    # ayırıcı özelliklerin ağırlıkları POZİTİF (etiket onlarla artıyor)
    w = dict(zip(cal.feature_names, cal.weights, strict=True))
    assert w["served_top1"] > 0.0
    assert w["routed"] > 0.0


def test_fit_is_deterministic():
    """Tohum YOK: başlangıç sıfır + tam-toplu GD -> bit-birebir aynı ağırlıklar."""
    X, y = _separable()
    a, b = Calibrator.fit(X, y), Calibrator.fit(X, y)
    assert a.weights == b.weights
    assert a.bias == b.bias
    assert a.mean == b.mean and a.std == b.std
    assert a.fit_info["rng"] == "none"


def test_standardization_roundtrip():
    X, y = _separable()
    cal = Calibrator.fit(X, y)
    Z = cal.standardize(X)
    back = Z * np.array(cal.std) + np.array(cal.mean)
    np.testing.assert_allclose(back, X, rtol=0, atol=1e-12)
    # z-skorları gerçekten standart (ortalama 0, std 1)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-12)


def test_constant_feature_gets_unit_std_instead_of_division_by_zero():
    X, y = _separable()
    X[:, 4] = 7.0  # sabit sütun
    cal = Calibrator.fit(X, y)
    assert cal.std[4] == 1.0
    assert np.all(np.isfinite(cal.predict_proba(X)))


def test_fit_requires_both_classes():
    X, _ = _separable()
    with pytest.raises(ValueError, match="her iki sınıf"):
        Calibrator.fit(X, np.ones(X.shape[0]))


def test_predict_one_uses_feature_names_order():
    X, y = _separable()
    cal = Calibrator.fit(X, y)
    feats = dict(zip(FEATURE_ORDER, X[0].tolist(), strict=True))
    assert cal.predict_one(feats) == pytest.approx(float(cal.predict_proba(X[:1])[0]))


# --- eşik seçimi ------------------------------------------------------------


def test_choose_threshold_maximizes_coverage_within_risk_budget():
    # 10 örnek: en yüksek 5 olasılık temiz (label 1), altındakiler karışık.
    probs = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3])
    labels = np.array([1.0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    ch = choose_threshold(probs, labels, max_risk=0.05)
    assert ch.name == "risk_budget"
    assert ch.value == pytest.approx(0.75)  # ilk 5'i alan en düşük tau
    assert ch.coverage == pytest.approx(0.5)
    assert ch.risk == 0.0


def test_choose_threshold_falls_back_to_all_abstain():
    """Hiçbir çalışma noktası bütçeyi sağlamıyorsa bütçe GEVŞETİLMEZ."""
    probs = np.array([0.9, 0.8, 0.7])
    labels = np.array([0.0, 1.0, 0.0])
    ch = choose_threshold(probs, labels, max_risk=0.05)
    assert ch.name == "abstain_all"
    assert ch.coverage == 0.0
    assert "hiçbir çalışma noktası yok" in ch.rationale
    # tam çekimserde "risk kanıtlanmış 0" YAZILMAZ: hiçbir şey dışlanmadı
    assert ch.n_answered == 0
    assert ch.risk_cp_upper_95 == 1.0
    assert ch.statistical_guarantee == GUARANTEE_NONE


def test_small_n_threshold_is_labelled_as_having_no_guarantee():
    """review J1: `risk=0.0` az satırdan ölçüldüyse artefakt bunu SÖYLEMELİ.

    0/4 hatanın %95 Clopper-Pearson üst sınırı ~0.527 — yani nokta tahmini
    "sıfır risk" ile %50 risk istatistiksel olarak ayırt edilemez. Modül bu
    korumayı conformal dalında zaten uyguluyordu; seçilen eşikte de uygular.
    """
    probs = np.array([0.9, 0.85, 0.8, 0.75, 0.4, 0.3, 0.2, 0.1])
    labels = np.array([1.0, 1, 1, 1, 0, 0, 0, 0])
    ch = choose_threshold(probs, labels, max_risk=0.05)
    assert ch.risk == 0.0  # nokta tahmini gerçekten sıfır...
    assert ch.n_answered == 4  # ...ama yalnız 4 satırdan
    assert ch.errors == 0
    assert ch.risk_cp_upper_95 == pytest.approx(0.5271, abs=1e-3)
    assert ch.statistical_guarantee == GUARANTEE_NONE
    # gerekçe dizesi de uyarıyı taşır (artefaktı okuyan tek satırda görsün)
    assert "İSTATİSTİKSEL GÜVENCE YOK" in ch.rationale
    d = ch.to_dict()
    assert d["risk_point"] == d["risk"] == 0.0
    assert set(d) >= {"n_answered", "errors", "risk_cp_upper_95", "statistical_guarantee"}


def test_large_n_zero_error_threshold_earns_the_cp_guarantee():
    """Aynı 0.0 riski YETERİNCE satırdan ölçülünce bayrak değişir.

    Negatiflerin TAMAMI tek bir olasılıkta toplanıyor, yani bir sonraki
    çalışma noktası 40 hatayı birden içeri alıp bütçeyi (0.0909 > 0.05)
    aşıyor. Böylece kapsamayı en büyükleyen nokta sıfır-hatalı 400 satır
    oluyor ve CP üst sınırı (~0.0075) bütçenin altına düşüyor.
    """
    n = 400
    probs = np.concatenate([np.linspace(0.6, 0.99, n), np.full(40, 0.5)])
    labels = np.concatenate([np.ones(n), np.zeros(40)])
    ch = choose_threshold(probs, labels, max_risk=0.05)
    assert ch.errors == 0
    assert ch.n_answered == n
    assert ch.risk_cp_upper_95 == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-6)
    assert ch.risk_cp_upper_95 <= 0.05
    assert ch.statistical_guarantee == GUARANTEE_CP
    assert "GÜVENCE YOK" not in ch.rationale


def test_conformal_reports_insufficient_n_instead_of_a_vacuous_number():
    """alpha=0.05 -> en az ceil(1/alpha)-1 = 19 hata gerekli; altında SAYI yazılmaz."""
    probs = np.linspace(0.1, 0.9, 20)
    labels = np.ones(20)
    labels[:5] = 0.0  # yalnız 5 hata
    out = conformal_candidate(probs, labels, alpha=0.05)
    assert out["available"] is False
    assert out["value"] is None
    assert out["n_errors"] == 5
    assert out["n_required"] == 19
    assert "n yetersiz" in out["note"]


def test_conformal_returns_a_threshold_when_n_is_adequate():
    from belge_gozu.bench.calibration_metrics import conformal_threshold

    probs = np.linspace(0.01, 0.99, 60)
    labels = np.ones(60)
    labels[:25] = 0.0  # 25 hata >= 19
    out = conformal_candidate(probs, labels, alpha=0.05)
    assert out["available"] is True
    assert out["n_errors"] == 25
    assert out["value"] == pytest.approx(conformal_threshold(probs, labels, alpha=0.05))


# --- versiyonlu artefakt ----------------------------------------------------


def test_calibration_key_includes_pipeline_and_recipe():
    """P2 denetimi T6 bulgusu: anahtar YALNIZ index_revision olamaz."""
    fp = recipe_fingerprint()
    key = calibration_key("abc123def456/train-compat-v1/int8", "hybrid")
    assert key.endswith(f"__hybrid__{fp}")
    assert "/" not in key  # dizin adı olarak güvenli
    # reçete bileşeni gerçekten ayırt ediyor
    assert calibration_key("r", "hybrid", "aaaaaaaaaaaa") != calibration_key(
        "r", "hybrid", "bbbbbbbbbbbb"
    )
    # boru hattı da (skor ölçeğini o belirliyor)
    assert calibration_key("r", "hybrid", "f") != calibration_key("r", "exhaustive", "f")


def _tiny_artifact(tmp_path: Path) -> tuple[CalibrationArtifact, Path]:
    X, y = _separable()
    rows = _rows_from_matrix(X, y)
    art = fit_calibration(rows, index_revision="rev/x/int8", pipeline="hybrid")
    d = calibration_dir(tmp_path, art.key)
    art.save(d)
    return art, d


def _rows_from_matrix(X, y):
    from belge_gozu.answer.calibrate import LabeledRow

    return [
        LabeledRow(
            question_id=f"q{i}",
            split="dev",
            answerable=bool(lab),
            label=int(lab),
            features=dict(zip(FEATURE_ORDER, row.tolist(), strict=True)),
            gold_in_topk=bool(lab),
            unanswerable_reason=None if lab else "korpus-disi",
            source="synthetic",
        )
        for i, (row, lab) in enumerate(zip(X, y, strict=True))
    ]


def test_artifact_roundtrip_gives_identical_predictions(tmp_path: Path):
    art, d = _tiny_artifact(tmp_path)
    loaded = load_calibrator(d, art.key)
    X, _ = _separable()
    np.testing.assert_array_equal(
        art.calibrator.predict_proba(X), loaded.calibrator.predict_proba(X)
    )
    assert loaded.tau == art.tau
    assert loaded.to_dict() == art.to_dict()
    # dosyanın kendisi de kabul edilir (dizin ya da calibrator.json)
    assert load_calibrator(d / "calibrator.json", art.key).key == art.key


def test_load_calibrator_fails_fast_on_key_mismatch(tmp_path: Path):
    art, d = _tiny_artifact(tmp_path)
    with pytest.raises(CalibrationKeyMismatch) as e:
        load_calibrator(d, "baska/anahtar__hybrid__000000000000")
    msg = str(e.value)
    assert art.key in msg and "baska/anahtar" in msg
    assert "calibrate fit" in msg  # çözüm yolu mesajda


def test_load_calibrator_fails_fast_on_feature_order_change(tmp_path: Path):
    art, d = _tiny_artifact(tmp_path)
    path = d / "calibrator.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibrator"]["feature_names"] = list(reversed(FEATURE_ORDER))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationKeyMismatch, match="özellik sırası"):
        load_calibrator(d, art.key)


def test_load_calibrator_missing_artifact_names_the_fix(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="calibrate fit"):
        load_calibrator(tmp_path / "yok", "k")


def test_two_fits_produce_identical_artifacts_modulo_timestamp(tmp_path: Path):
    X, y = _separable()
    rows = _rows_from_matrix(X, y)
    a = fit_calibration(rows, index_revision="rev/x/int8", pipeline="hybrid").to_dict()
    b = fit_calibration(rows, index_revision="rev/x/int8", pipeline="hybrid").to_dict()
    a["kunye"].pop("created_at")
    b["kunye"].pop("created_at")
    assert a == b


# --- etiket + veri kümesi ---------------------------------------------------


def _q(qid: str, **over) -> dict:
    base = {
        "question_id": qid,
        "question": "yerleşim yeri nedir",
        "query_style": "dogal",
        "answerable": True,
        "gold_doc_ids": ["k1"],
        "gold_page_ids": ["k1:2"],
        "gold_article_ids": [],
        "minimal_evidence_spans": [],
        "reference_answer": "cevap",
        "slice": "dogrudan-madde",
        "difficulty": "kolay",
        "source_type": "insan",
        "requires_visual": False,
        "requires_multi_hop": False,
        "unanswerable_reason": None,
        "verified_by": "test",
        "verification_status": "verified",
    }
    base.update(over)
    return base


def _unans(qid: str, question: str, **over) -> dict:
    return _q(
        qid,
        question=question,
        answerable=False,
        gold_doc_ids=[],
        gold_page_ids=[],
        reference_answer="",
        slice="eksik-kanit",
        unanswerable_reason="eksik-kanit",
        **{"_subject_doc": "k1", **over},
    )


SPLITS = {"dev_docs": {"k1", "k2"}, "test_docs": {"kx"}}


def test_label_is_one_only_for_answerable_with_gold_in_top5(channel):
    """Etiket: cevaplanabilir VE gold top-5'te. Iska ve cevaplanamaz -> 0."""
    text, names = channel
    # 4 sayfalık korpusta top-5 = her şey; ıskayı k=1 ile kurarız.
    raw = [
        _q("hit", question="tanımı", gold_page_ids=["k1:2"]),
        _q("miss", question="tanımı", gold_page_ids=["k2:1"], gold_doc_ids=["k2"]),
        _unans("u1", "hiçbiryerdeolmayankelime"),
    ]
    rows = build_rows(raw, SPLITS, text, names, k=1)
    by_id = {r.question_id: r for r in rows}
    assert by_id["hit"].label == 1 and by_id["hit"].gold_in_topk
    assert by_id["miss"].label == 0 and by_id["miss"].answerable  # ıska: kanıtsız yanıt riski
    assert by_id["u1"].label == 0 and not by_id["u1"].answerable

    counts = class_counts(rows)
    assert counts == {
        "total": 3,
        "positive_safe_to_answer": 1,
        "negative": 2,
        "answerable": 2,
        "answerable_gold_in_top5": 1,
        "answerable_retrieval_miss": 1,
        "unanswerable": 1,
    }


def test_build_rows_assigns_law_grouped_splits(channel):
    text, names = channel
    raw = [_q("d", gold_doc_ids=["k1"]), _q("t", gold_doc_ids=["kx"], gold_page_ids=["kx:1"])]
    rows = {r.question_id: r.split for r in build_rows(raw, SPLITS, text, names)}
    assert rows == {"d": "dev", "t": "test"}


def test_build_rows_rejects_window_smaller_than_k(channel):
    text, names = channel
    with pytest.raises(ValueError, match="window"):
        build_rows([_q("a")], SPLITS, text, names, k=5, window=3)


def test_load_rows_drops_draft_and_rejected(tmp_path: Path):
    """`only_verified` `load_bench` ile birebir: draft VE rejected dışarıda."""
    p = tmp_path / "b.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [
                _q("ok"),
                _q("dr", verification_status="draft", verified_by=""),
                _q("rj", verification_status="rejected", verified_by=""),
            ]
        ),
        encoding="utf-8",
    )
    assert [r["question_id"] for r in load_rows(p)] == ["ok"]
    assert len(load_rows(p, only_verified=False)) == 3


def test_load_rows_keeps_underscore_fields_split_needs(tmp_path: Path):
    """`_subject_doc` pydantic'te düşer; hukuk-gruplu bölme onu OKUMAK zorunda."""
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps(_unans("u1", "soru"), ensure_ascii=False), encoding="utf-8")
    assert load_rows(p)[0]["_subject_doc"] == "k1"


def test_load_rows_reports_the_bad_line_number(tmp_path: Path):
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps(_q("ok")) + "\n{bozuk\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bench satır 2"):
        load_rows(p)


# --- CLI ---------------------------------------------------------------------


def _fixture_env(
    tmp_path: Path, monkeypatch, test_docs: tuple[str, ...] = ("kx",)
) -> tuple[Path, Path, Path]:
    """Kendi kendine yeten kurulum: indeks dizini = page_ids + manifest + metin.

    `test_docs` varsayılanı korpusta olmayan bir doküman ("kx") — yani TÜM
    sorular dev'e düşer. Kapı (`--split test`) testi bunu `("k2",)` yaparak
    kendi küçük test bölmesini kurar; gerçek `splits_v1.json` hiç okunmaz.
    """
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "page_ids.json").write_text(json.dumps(IDS), encoding="utf-8")
    pd.DataFrame({"page_id": IDS, "text": TEXTS}).to_parquet(
        index_dir / "page_texts.parquet", index=False
    )
    write_manifest(
        index_dir,
        IndexManifest(
            model_name="test",
            model_revision="r",
            engine_versions={},
            query_format=CPE_0_3_18,
            doc_prompt_sha256="d" * 64,
            quantization="int8",
            mask_policy="drop-padding",
            render=RenderConfig(),
            corpus_checksum="c" * 64,
            n_pages=len(IDS),
            n_tokens=1,
            built_at="2026-08-30T00:00:00+00:00",
            git_commit="deadbee",
        ),
    )
    canary = tmp_path / "canary.jsonl"
    canary.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [
                _q("a1", question="Medeni yerleşim yeri", gold_page_ids=["k1:1"]),
                _q("a2", question="tanımı", gold_page_ids=["k1:2"]),
                _q(
                    "a3", question="İcra yerleşim yeri", gold_doc_ids=["k2"], gold_page_ids=["k2:2"]
                ),
                _q("a4", question="yerleşim yeri", gold_doc_ids=["k2"], gold_page_ids=["k2:2"]),
            ]
        ),
        encoding="utf-8",
    )
    unans = tmp_path / "unans.jsonl"
    unans.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [
                _unans("u1", "zzzqqq wwwxxx"),
                _unans("u2", "aaabbb cccddd"),
                _unans("u3", "eeefff ggghhh"),
                _unans("u4", "iiijjj kkklll"),
                _unans("u5", "mmmnnn oooppp", verification_status="rejected", verified_by=""),
                # `_subject_doc="k2"`: varsayılan bölmede dev'e, `test_docs=["k2"]`
                # varyantında test'e düşer — kapı testine negatif sınıfı verir.
                _unans("u6", "qqqrrr ssstttt", _subject_doc="k2"),
            ]
        ),
        encoding="utf-8",
    )
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "version": "vt",
                "seed": "s",
                "dev_docs": ["k1", "k2"],
                "test_docs": list(test_docs),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BG_INDEX_DIR", str(index_dir))
    return canary, unans, splits


def _fit_args(canary, unans, splits, out, extra=()):
    return [
        "calibrate",
        "fit",
        "--canary",
        str(canary),
        "--unans",
        str(unans),
        "--splits",
        str(splits),
        "--out",
        str(out),
        *extra,
    ]


def test_cli_calibrate_fit_writes_artifact_and_report(tmp_path: Path, monkeypatch):
    canary, unans, splits = _fixture_env(tmp_path, monkeypatch)
    out = tmp_path / "report.json"
    res = runner.invoke(app, _fit_args(canary, unans, splits, out))
    assert res.exit_code == 0, res.output

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["split"] == "dev"
    # rejected satır (u5) sayıma girmemeli: 4 cevaplanabilir + 5 cevaplanamaz
    assert report["kunye"]["counts"]["total"] == 9
    assert report["kunye"]["counts"]["unanswerable"] == 5
    assert report["key"].endswith(f"__hybrid__{recipe_fingerprint()}")
    assert report["artifact_committed"] is False
    assert [f["name"] for f in report["kunye"]["data_files"]] == ["canary", "unans"]
    assert all(len(f["sha256"]) == 64 for f in report["kunye"]["data_files"])
    # review M4: içeriği GERİ GETİREN referans da künyede; re-review N1: tmp
    # fikstürleri commit'lenmemiş olduğundan dürüst "-uncommitted" son eki taşır
    assert all(
        len(f["git_blob"]) == 40 or f["git_blob"].endswith("-uncommitted")
        for f in report["kunye"]["data_files"]
    )
    assert report["calibrator"]["feature_names"] == list(FEATURE_ORDER)
    assert "risk_coverage" in report["metrics"]

    # review M3: her sayı yalnız bu dosyadan yeniden hesaplanabilmeli
    rows = report["per_question"]
    assert len(rows) == 9
    assert {r["qid"] for r in rows} == {"a1", "a2", "a3", "a4", "u1", "u2", "u3", "u4", "u6"}
    assert all(set(r["features"]) == set(FEATURE_ORDER) for r in rows)
    assert sum(r["label"] for r in rows) == report["kunye"]["counts"]["positive_safe_to_answer"]
    # tau'da yanıtlanan satır sayısı eşik kaydıyla tutarlı
    ch = report["thresholds"]["chosen"]
    assert sum(r["answered_at_tau"] for r in rows) == ch["n_answered"]
    # review J1: belirsizlik alanları artefaktta VE raporda
    assert set(ch) >= {"risk_point", "n_answered", "errors", "risk_cp_upper_95"}
    assert ch["statistical_guarantee"] in {"none", "cp_upper<=target"}
    # review M3: §5.3/§5.4'ün dayanağı künyede
    assert all("auc" in v for v in report["kunye"]["feature_stats"].values())
    assert report["kunye"]["feature_correlations"]["served_top1"]["served_top1"] == pytest.approx(
        1.0
    )

    artifact = Path(report["artifact_path"])
    assert artifact.exists()
    assert load_calibrator(artifact, report["key"]).key == report["key"]
    assert "risk-coverage" in res.output
    assert "guarantee=" in res.output


def test_cli_calibrate_fit_records_the_data_pin_note(tmp_path: Path, monkeypatch):
    """Bench verisi aktif taslakta: koşumun kimliği yol değil İÇERİK + not."""
    canary, unans, splits = _fixture_env(tmp_path, monkeypatch)
    out = tmp_path / "report.json"
    res = runner.invoke(
        app, _fit_args(canary, unans, splits, out, ["--note", "unans @ commit abc1234"])
    )
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text(encoding="utf-8"))["kunye"]["note"] == "unans @ commit abc1234"

    # not verilmezse künyede alan HİÇ olmamalı (boş dize gürültüsü yok)
    out2 = tmp_path / "report2.json"
    assert runner.invoke(app, _fit_args(canary, unans, splits, out2)).exit_code == 0
    assert "note" not in json.loads(out2.read_text(encoding="utf-8"))["kunye"]


def test_cli_calibrate_eval_recomputes_from_artifact(tmp_path: Path, monkeypatch):
    canary, unans, splits = _fixture_env(tmp_path, monkeypatch)
    out = tmp_path / "report.json"
    assert runner.invoke(app, _fit_args(canary, unans, splits, out)).exit_code == 0
    fit_metrics = json.loads(out.read_text(encoding="utf-8"))["metrics"]

    ev_out = tmp_path / "eval.json"
    res = runner.invoke(
        app,
        [
            "calibrate",
            "eval",
            "--canary",
            str(canary),
            "--unans",
            str(unans),
            "--splits",
            str(splits),
            "--out",
            str(ev_out),
        ],
    )
    assert res.exit_code == 0, res.output
    # fit ve eval AYNI `evaluate` kodunu çağırır -> sayılar birebir aynı
    assert json.loads(ev_out.read_text(encoding="utf-8"))["metrics"] == fit_metrics


def test_cli_calibrate_test_split_needs_explicit_gate(tmp_path: Path, monkeypatch):
    canary, unans, splits = _fixture_env(tmp_path, monkeypatch)
    out = tmp_path / "report.json"
    res = runner.invoke(app, _fit_args(canary, unans, splits, out, ["--split", "test"]))
    assert res.exit_code != 0
    combined = res.output + (res.stderr or "")
    assert "--yes-final-gate" in combined
    assert "KAPI KOŞUMU DIŞINDA KULLANMAYIN" in combined


def test_cli_calibrate_test_split_proceeds_with_the_gate_flag(tmp_path: Path, monkeypatch):
    """review m7: kapının MUTLU YOLU da kilitli olmalı.

    Reddi test etmek yetmiyordu — kapıyı koşulsuz `raise`e çeviren bir refactor
    suite'ten geçer ve ancak faz sonu kapı koşumunda, yani TEKRARLANAMAYAN tek
    koşumda ortaya çıkardı.

    TAMAMEN SENTETİK bir test bölmesi kullanılır (`test_docs=("k2",)` ile üç
    satır); gerçek `data/bench/splits_v1.json` test yakası HİÇ okunmaz.
    """
    canary, unans, splits = _fixture_env(tmp_path, monkeypatch, test_docs=("k2",))
    out = tmp_path / "gate.json"
    res = runner.invoke(
        app, _fit_args(canary, unans, splits, out, ["--split", "test", "--yes-final-gate"])
    )
    assert res.exit_code == 0, res.output + (res.stderr or "")
    # banner yine basılır (onay, uyarıyı susturmaz)
    assert "KAPI KOŞUMU DIŞINDA KULLANMAYIN" in (res.output + (res.stderr or ""))
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["split"] == "test"
    # a3 + a4 (cevaplanabilir, k2 gold) + u6 (cevaplanamaz, _subject_doc=k2)
    assert {r["qid"] for r in report["per_question"]} == {"a3", "a4", "u6"}
    assert report["kunye"]["counts"] == {
        "total": 3,
        "positive_safe_to_answer": 2,
        "negative": 1,
        "answerable": 2,
        "answerable_gold_in_top5": 2,
        "answerable_retrieval_miss": 0,
        "unanswerable": 1,
    }


def test_cli_calibrate_fit_rejects_non_hybrid_pipeline(tmp_path: Path, monkeypatch):
    canary, unans, splits = _fixture_env(tmp_path, monkeypatch)
    monkeypatch.setenv("BG_RETRIEVAL_PIPELINE", "exhaustive")
    res = runner.invoke(app, _fit_args(canary, unans, splits, tmp_path / "r.json"))
    assert res.exit_code != 0
    assert "yalnız hybrid" in (res.output + (res.stderr or ""))


def test_cli_calibrate_fit_is_reproducible(tmp_path: Path, monkeypatch):
    """Aynı girdi -> aynı ağırlıklar/eşikler (zaman damgası hariç)."""
    canary, unans, splits = _fixture_env(tmp_path, monkeypatch)
    reports = []
    for name in ("r1.json", "r2.json"):
        out = tmp_path / name
        assert runner.invoke(app, _fit_args(canary, unans, splits, out)).exit_code == 0
        reports.append(json.loads(out.read_text(encoding="utf-8")))
    for r in reports:
        r.pop("created_at")
        r.pop("run_id")
        r["kunye"].pop("created_at")
    assert reports[0] == reports[1]


# --- katman disiplini -------------------------------------------------------


def test_runtime_import_does_not_pull_the_bench_package():
    """Üretim yolu bench'e bağlanmaz (provenance.py'nin ayrılma gerekçesi).

    `bench` importları fit/eval fonksiyonlarının İÇİNDEDİR; modülü içe aktarmak
    onları çekmemelidir.
    """
    code = (
        "import belge_gozu.answer.calibrate as c, sys;"
        "assert c.FEATURE_ORDER;"
        "leaked=[m for m in sys.modules if m.startswith('belge_gozu.bench')];"
        "assert not leaked, leaked"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_evaluate_labels_the_dev_false_answer_rate_as_not_the_gate(tmp_path: Path):
    art, _ = _tiny_artifact(tmp_path)
    X, y = _separable()
    metrics = evaluate(art, _rows_from_matrix(X, y))
    far = metrics["false_answer_on_unanswerable"]
    assert "G2.1 KAPI SAYISI DEĞİLDİR" in far["note"]
    assert far["method"] == "clopper_pearson"
    assert 0.0 <= far["rate"] <= far["upper_bound_95"] <= 1.0


def test_feature_matrix_rejects_empty_input():
    with pytest.raises(ValueError, match="boş girdi"):
        feature_matrix([])


def test_univariate_auc_works_on_raw_unscaled_features():
    """review M3: raporun §5.3 AUC'leri depoda hesaplanabilmeli.

    `bench.calibration_metrics.auroc` ham özellikte ValueError fırlatır (girdiyi
    [0,1] KALİBRE OLASILIK sayar) — doğru davranış, ama bu iş için yanlış araç.
    AUC monoton dönüşümlere duyarsız olduğundan ölçek serbest bırakılır.
    """
    from belge_gozu.bench.calibration_metrics import auroc

    raw = np.array([4.36, 10.0, 25.0, 66.68])  # BM25 ölçeği, [0,1] DEĞİL
    y = np.array([0.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        auroc(raw, y)
    assert univariate_auc(raw, y) == 1.0
    # monoton dönüşüm AUC'yi değiştirmez (ölçekten bağımsızlık)
    assert univariate_auc(raw / 100.0, y) == 1.0
    assert univariate_auc(-raw, y) == 0.0  # ters yön
    # beraberlikler 0.5 kredisiyle
    assert univariate_auc(np.array([1.0, 1.0]), np.array([1.0, 0.0])) == 0.5
    with pytest.raises(ValueError, match="hem pozitif hem negatif"):
        univariate_auc(raw, np.ones(4))


def test_per_question_rows_are_the_recomputable_base(tmp_path: Path):
    """Kayıt kendi kendini kanıtlamalı: toplu metrikler satırlardan türetilebilir."""
    art, _ = _tiny_artifact(tmp_path)
    X, y = _separable()
    rows = _rows_from_matrix(X, y)
    out = per_question_rows(art, rows)

    assert len(out) == len(rows)
    assert [r["qid"] for r in out] == [r.question_id for r in rows]
    assert [r["label"] for r in out] == [r.label for r in rows]
    # olasılıklar artefaktın kendi tahminleriyle birebir
    np.testing.assert_array_equal(
        np.array([r["prob"] for r in out]), art.calibrator.predict_proba(X)
    )
    # tau'da yanıtlanan sayısı eşik kaydıyla tutarlı
    assert sum(r["answered_at_tau"] for r in out) == art.thresholds["chosen"]["n_answered"]
    # satırlardan AUROC yeniden hesaplanabiliyor
    from belge_gozu.bench.calibration_metrics import auroc

    recomputed = auroc(
        np.array([r["prob"] for r in out]), np.array([float(r["label"]) for r in out])
    )
    assert recomputed == pytest.approx(art.kunye["dev_metrics"]["auroc"])


def test_fit_info_counts_updates_not_gradient_evaluations():
    """review m8: künyeye yazılan provenans sayısı 'yaklaşık' olamaz."""
    X, y = _separable()
    cal = Calibrator.fit(X, y)
    fi = cal.fit_info
    assert fi["converged"] is True
    # yakınsamada tolerans kontrolü güncellemeden ÖNCE gelir -> bir fazla değerlendirme
    assert fi["n_gradient_evals"] == fi["n_iter"] + 1


def test_feature_stats_std_matches_the_calibrator_std_exactly(tmp_path: Path):
    """review m9: aynı büyüklük iki yerde son ulp'te ayrışmamalı."""
    X, y = _separable()
    rows = _rows_from_matrix(X, y)
    art = fit_calibration(rows, index_revision="rev/x/int8", pipeline="hybrid")
    stats = art.kunye["feature_stats"]
    for name, std in zip(art.calibrator.feature_names, art.calibrator.std, strict=True):
        assert stats[name]["std"] == std  # yaklaşık değil, BİREBİR


def test_threshold_candidate_is_keyed_by_its_own_name():
    """review m10: anahtar içeriğiyle çelişmemeli."""
    X, y = _separable()
    rows = _rows_from_matrix(X, y)
    art = fit_calibration(rows, index_revision="r/x/i", pipeline="hybrid")
    chosen_name = art.thresholds["chosen"]["name"]
    assert chosen_name in art.thresholds
    assert art.thresholds[chosen_name]["name"] == chosen_name
    assert "abstain_all" not in art.thresholds or chosen_name == "abstain_all"


def test_git_blob_sha_marks_uncommitted_content(tmp_path):
    """re-review N1: commit'lenmemiş içerik '-uncommitted' son ekiyle işaretlenir."""
    import uuid

    from belge_gozu.answer.calibrate import git_blob_sha

    f = tmp_path / "dirty.jsonl"
    f.write_text(f"benzersiz-{uuid.uuid4()}\n")
    sha = git_blob_sha(f)
    assert sha.endswith("-uncommitted") and len(sha) == 40 + len("-uncommitted")
