"""Güven özellikleri + kalibratör (P2 T5+T6) — metin yanının 5 sinyali.

Modül İKİ YARIMDIR ve ayrım bilinçlidir:

1. **ÇALIŞMA ANI (online, T8'de `/ask`e takılacak)** — `extract_features`,
   `Calibrator.predict_one`, `load_calibrator`. Yalnız `numpy` +
   `retrieval/text.py` ister; `belge_gozu.bench`'e ve modele/ağa DOKUNMAZ.
2. **OFFLINE (fit/eval, yalnız CLI'den çağrılır)** — `build_rows`,
   `fit_calibration`, `evaluate`. Bench veri şemasını ve
   `bench/calibration_metrics.py`'yi kullanır; bu importlar FONKSİYON
   İÇİNDEDİR, modül düzeyinde değil.

`bench` importlarının tembel olması bir stil tercihi değil: üretim yolunun
bench paketine bağlanmaması bu projede yerleşik bir disiplindir
(`provenance.py` tam bu yüzden `bench/harness.py`'den ayrılmıştı — final review
IMPORTANT-5). `tests/answer/test_calibrate.py` bunu alt süreçte doğrular.

ÖZELLİK SEÇİMİ ÖLÇÜMLEDİR, TAHMİN DEĞİL (`data/research/abstain-signals.json`,
43 cevaplanabilir + 5 cevaplanamaz canary sorusu üzerinde tek tek AUC):

| özellik | AUC | karar |
|---|---|---|
| `matched_terms_top1` | .937 | ALINDI (en güçlü tek sinyal) |
| `matched_frac` | .863 | ALINDI |
| `bm25_top1` (kanal) | .856 | `served_top1` olarak ALINDI (aşağıya bak) |
| `served_top1` | .819 | ALINDI |
| `bm25_margin` | .679 | ALINDI (zayıf ama bağımsız eksen) |
| `q_len_toks` | — | REDDEDİLDİ: veri kümesi artefaktı (cevaplanamazlar elle uzun yazılmış) |
| TÜM görsel özellikler | .34 | REDDEDİLDİ: ölçülmüş TERS yön (`retrieval/hybrid.py` bulgu 3) |

`served_top1` ile kanal `bm25_top1` ayrımı (config.py review L1): pencere-içi
yönlendirme sıralamanın BİRİNCİSİNİ skora göre değil sorguda adı geçen kanuna
göre seçebilir. Eşiğin/kalibratörün gördüğü sayfa SERVİS EDİLENDİR, kanalın en
yüksek skorlusu değil — bu yüzden özellik `served_top1`dir. Buna karşılık
`bm25_margin` ve `matched_terms_top1` YÖNLENDİRME ÖNCESİ top-1'den ölçülür:
ikisi de "sözcüksel kanal ne kadar emin?" sorusunu sorar ve yönlendirme
kuralının kararıyla karıştırılmamalıdır; yönlendirmenin kendisi ayrı bir
özelliktir (`routed`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from belge_gozu.provenance import git_commit
from belge_gozu.retrieval.text import (
    WINDOW,
    BM25Index,
    rank_order,
    recipe_fingerprint,
    route_window,
    routed_docs,
    tokenize,
)

# ---------------------------------------------------------------------------
# 1. ÖZELLİKLER (T5) — çalışma anında da kullanılabilir
# ---------------------------------------------------------------------------

# Vektörleştirme sırası. SABİT ve VERSİYONLU: artefakt bu sırayı `feature_names`
# olarak yazar ve `load_calibrator` yüklerken karşılaştırır, yani sıra
# değişirse eski ağırlıklar yeni sıraya sessizce hizalanamaz.
FEATURE_ORDER: tuple[str, ...] = (
    "served_top1",
    "bm25_margin",
    "matched_terms_top1",
    "matched_frac",
    "routed",
)


@dataclass(frozen=True)
class RetrievalContext:
    """Tek bir getirim geçişinin ham malzemesi — bir `scores()` çağrısı yeniden kullanılır.

    Hem özellikler hem de (offline) etiket bundan türetilir; iki ayrı geçiş
    yapmak korpusu iki kez skorlamak demekti (4222 sayfa × 173 soru).
    """

    bm25: np.ndarray
    """Yönlendirme ÖNCESİ BM25 skorları — `text.page_ids` ile hizalı."""
    order: np.ndarray
    """BM25 skorunun azalan sırasındaki indeksler (üretimle aynı `stable` argsort)."""
    routed: set[str]
    """Yönlendirmeyi tetikleyen doküman kimlikleri (boşsa yönlendirme yok)."""
    window_ranking: list[str]
    """İlk `window` sayfa, YÖNLENDİRME SONRASI sıra — nihai sıranın başlangıcı."""


def retrieval_context(
    query: str,
    text: BM25Index,
    doc_names: Mapping[str, frozenset[str]],
    *,
    bm25: np.ndarray | None = None,
    window: int = WINDOW,
) -> RetrievalContext:
    """Sorguyu BİR KEZ skorlayıp reçetenin sıra kompozisyonunu kurar.

    `bm25` verilirse yeniden skorlanmaz: çalışma anında `HybridRetriever.search`
    zaten `text.scores(query)`'yi hesaplamıştır ve onu geçmek özellik çıkarımını
    BEDAVA yapar (kalibrasyon istek başına ek bir korpus taraması eklememelidir).

    Sıra `HybridRetriever.rank` ile aynı iki adımdır (stable argsort ->
    `route_window`); `route_window` burada YENİDEN YAZILMAZ, çağrılır.
    """
    scores = text.scores(query) if bm25 is None else np.asarray(bm25)
    n = len(text.page_ids)
    if scores.shape != (n,):
        raise ValueError(f"bm25 skorları page_ids ile hizalı olmalı: {scores.shape} != ({n},)")
    order = rank_order(scores)
    routed = routed_docs(query, doc_names)
    win = [text.page_ids[int(i)] for i in order[:window]]
    return RetrievalContext(
        bm25=scores,
        order=order,
        routed=routed,
        window_ranking=route_window(win, routed, window),
    )


def features_from_context(query: str, text: BM25Index, ctx: RetrievalContext) -> dict[str, float]:
    """`RetrievalContext`ten 5 özelliği okur (yeniden skorlama yok)."""
    scores = ctx.bm25
    top1_idx = int(ctx.order[0])
    top1 = float(scores[top1_idx])
    top2 = float(scores[int(ctx.order[1])]) if scores.size > 1 else 0.0

    # SERVİS EDİLEN top-1: yönlendirme sonrası sıranın birincisi. Konumu
    # pencere içinde aranır — `window_ranking` `order[:window]`ın bir
    # PERMÜTASYONUDUR (route_window sözleşmesi: küme değişmez).
    served_pid = ctx.window_ranking[0]
    win_pos = [text.page_ids[int(i)] for i in ctx.order[: len(ctx.window_ranking)]]
    served_idx = int(ctx.order[win_pos.index(served_pid)])

    q_toks = set(tokenize(query))
    # `doc_freqs[i]` sayfanın token -> frekans sayacı; anahtarları sayfanın
    # token KÜMESİDİR. "matched" = sorgunun BENZERSİZ token'larından kaçının
    # bu sayfada geçtiği (tekrar sayılmaz — QTF_CAP ile aynı gerekçe).
    top1_tokens = set(text.doc_freqs[top1_idx])
    matched = len(q_toks & top1_tokens)
    return {
        "served_top1": float(scores[served_idx]),
        "bm25_margin": top1 - top2,
        "matched_terms_top1": float(matched),
        # Boş sorgu (tümü stopword / <2 harf) -> 0.0. NaN DÖNDÜRÜLMEZ:
        # kalibratör NaN'i sessizce yayar, 0.0 ise "hiçbir terim eşleşmedi"
        # anlamını doğru taşır.
        "matched_frac": (matched / len(q_toks)) if q_toks else 0.0,
        "routed": 1.0 if ctx.routed else 0.0,
    }


def extract_features(
    query: str,
    text: BM25Index,
    doc_names: Mapping[str, frozenset[str]],
    *,
    bm25: np.ndarray | None = None,
    window: int = WINDOW,
) -> dict[str, float]:
    """Sorgu + metin kanalı -> `FEATURE_ORDER` anahtarlı 5 güven özelliği.

    ÇALIŞMA ANINDA DA ÇAĞRILABİLİR (T8): bench'e özgü hiçbir girdi almaz —
    yalnız sorgu metni ve serve'ün zaten kurduğu BM25 indeksi + doküman-adı
    sözlüğü. Fit ile serve'ün AYNI fonksiyonu çağırması sözleşmedir; iki ayrı
    çıkarım yolu, kalibratörün ölçtüğünden başka bir şeyi tahmin etmesi demek
    olurdu.
    """
    ctx = retrieval_context(query, text, doc_names, bm25=bm25, window=window)
    return features_from_context(query, text, ctx)


def to_vector(features: Mapping[str, float], names: Sequence[str] = FEATURE_ORDER) -> np.ndarray:
    """Özellik sözlüğü -> (d,) float64 vektör, `names` sırasında (eksik anahtar = hata)."""
    missing = [n for n in names if n not in features]
    if missing:
        raise KeyError(f"özellik eksik: {missing} (beklenen sıra: {list(names)})")
    return np.array([float(features[n]) for n in names], dtype=np.float64)


def feature_matrix(
    rows: Sequence[Mapping[str, float]], names: Sequence[str] = FEATURE_ORDER
) -> np.ndarray:
    """Özellik sözlükleri listesi -> (n, d) float64 matris."""
    if not rows:
        raise ValueError("boş girdi: en az bir özellik sözlüğü gerekli")
    return np.vstack([to_vector(r, names) for r in rows])


# ---------------------------------------------------------------------------
# 2. KALİBRATÖR (T6) — saf numpy lojistik regresyon
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sayısal olarak kararlı lojistik fonksiyon (büyük |z|'de exp taşmaz)."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# Fit varsayılanları. sklearn YOKTUR (plan `eval` extra'sı öneriyordu; P2
# gerçeklik denetimi sonrası vazgeçildi): 5 özellikli bir lojistik regresyon
# için 30 satırlık tam-toplu gradyan inişi yeterli ve BAĞIMLILIK EKLEMİYOR —
# artefaktı yükleyip tahmin eden çalışma anı zaten saf numpy olmak zorundaydı,
# fit'i de aynı kodla yapmak "eğitilen ile servis edilen aynı mı?" sorusunu
# tamamen ortadan kaldırır.
DEFAULT_LR = 0.5
DEFAULT_L2 = 1.0
DEFAULT_MAX_ITER = 50_000
DEFAULT_TOL = 1e-9


@dataclass(frozen=True)
class Calibrator:
    """Standartlaştırılmış özellikler üzerinde lojistik regresyon: p = sigmoid(w·z + b).

    DETERMİNİZM TOHUMA DEĞİL YAPIYA DAYANIR: başlangıç w=0, b=0 ve tam-toplu
    (stochastic değil) gradyan inişi kullanılır, yani fit hiçbir rastgele sayı
    üretici çağırmaz. Aynı girdi -> bit-birebir aynı ağırlıklar; "tohumu
    kaydettik" demeye gerek kalmaz çünkü kullanılan bir tohum yoktur.
    """

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    fit_info: dict[str, Any]

    def __post_init__(self) -> None:
        d = len(self.feature_names)
        for name, vec in (("mean", self.mean), ("std", self.std), ("weights", self.weights)):
            if len(vec) != d:
                raise ValueError(f"{name} boyu feature_names ile eşleşmeli: {len(vec)} != {d}")
        if any(s <= 0.0 for s in self.std):
            raise ValueError(
                f"std pozitif olmalı (sabit özellik fit'te 1.0'a sabitlenir): {self.std}"
            )

    def standardize(self, X: np.ndarray) -> np.ndarray:
        """(n, d) ham özellikler -> z-skorları (fit'te ölçülen mean/std ile)."""
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim != 2 or Xa.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X (n, {len(self.feature_names)}) biçiminde olmalı, gelen: {Xa.shape}"
            )
        return (Xa - np.array(self.mean)) / np.array(self.std)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, d) -> (n,) olasılık. Saf numpy; hiçbir opsiyonel bağımlılık gerekmez."""
        z = self.standardize(X) @ np.array(self.weights) + self.bias
        return _sigmoid(z)

    def predict_one(self, features: Mapping[str, float]) -> float:
        """Tek bir özellik SÖZLÜĞÜ -> olasılık (T8'in çalışma anı yüzeyi)."""
        vec = to_vector(features, self.feature_names)
        return float(self.predict_proba(vec.reshape(1, -1))[0])

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str] = FEATURE_ORDER,
        lr: float = DEFAULT_LR,
        l2: float = DEFAULT_L2,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
    ) -> Calibrator:
        """Deterministik tam-toplu gradyan inişi; L2 (bias hariç) 1/n ile ölçeklenir.

        L2 KAPALI DEĞİL çünkü etiket neredeyse ayrılabilir olabilir
        (`matched_terms_top1` tek başına AUC .937): ayrılabilir veride
        düzenlileştirmesiz lojistik regresyonun ağırlıkları sınırsız büyür,
        olasılıklar 0/1'e yapışır ve risk-coverage taraması anlamsız bir tek
        noktaya çöker. `l2/n` ölçeklemesi cezayı n'den bağımsız yorumlanır
        kılar. Sıralama (dolayısıyla AUROC) düzenlileştirmeden etkilenmez.
        """
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        names = tuple(feature_names)
        if Xa.ndim != 2 or Xa.shape[1] != len(names):
            raise ValueError(f"X (n, {len(names)}) biçiminde olmalı, gelen: {Xa.shape}")
        if ya.shape != (Xa.shape[0],):
            raise ValueError(f"y boyu X ile eşleşmeli: {ya.shape} != ({Xa.shape[0]},)")
        if not np.all((ya == 0.0) | (ya == 1.0)):
            raise ValueError("y yalnız 0/1 olabilir")
        if not np.all(np.isfinite(Xa)):
            raise ValueError("X sonsuz veya NaN içeremez")
        n_pos, n_neg = int(np.sum(ya == 1.0)), int(np.sum(ya == 0.0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"fit için her iki sınıf da gerekli (pozitif={n_pos}, negatif={n_neg})"
            )

        mean = Xa.mean(axis=0)
        std = Xa.std(axis=0)
        # Sabit özellik (std=0) 1.0'a sabitlenir: z hep 0 olur, ağırlığı L2 ile
        # 0'a çeker. Bölme hatası yerine "bu özellik bilgi taşımıyor".
        std = np.where(std <= 0.0, 1.0, std)
        Z = (Xa - mean) / std

        n = Xa.shape[0]
        w = np.zeros(Xa.shape[1], dtype=np.float64)
        b = 0.0
        # review m8: `n_iter` GERÇEKTEN YAPILAN GÜNCELLEME sayısıdır. Tolerans
        # kontrolü güncellemeden ÖNCE geldiği için gradyan değerlendirmesi bir
        # fazladır (yakınsandığında); ikisi ayrı ayrı künyeye yazılır — artefakta
        # yazılan bir provenans sayısı "yaklaşık" olamaz.
        n_updates, n_grad_evals, converged = 0, 0, False
        for _ in range(max_iter):
            resid = _sigmoid(Z @ w + b) - ya
            gw = Z.T @ resid / n + (l2 / n) * w
            gb = float(resid.mean())
            n_grad_evals += 1
            if max(float(np.max(np.abs(gw))), abs(gb)) < tol:
                converged = True
                break
            w -= lr * gw
            b -= lr * gb
            n_updates += 1

        p = _sigmoid(Z @ w + b)
        eps = 1e-12
        nll = float(-np.mean(ya * np.log(p + eps) + (1 - ya) * np.log(1 - p + eps)))
        return cls(
            feature_names=names,
            mean=tuple(float(v) for v in mean),
            std=tuple(float(v) for v in std),
            weights=tuple(float(v) for v in w),
            bias=float(b),
            fit_info={
                "solver": "full-batch-gd",
                "lr": lr,
                "l2": l2,
                "max_iter": max_iter,
                "tol": tol,
                "n_iter": n_updates,
                "n_gradient_evals": n_grad_evals,
                "converged": converged,
                "init": "zeros",
                "rng": "none",
                "n": n,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "final_nll": nll,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "std": list(self.std),
            "weights": list(self.weights),
            "bias": self.bias,
            "fit_info": self.fit_info,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Calibrator:
        return cls(
            feature_names=tuple(d["feature_names"]),
            mean=tuple(float(v) for v in d["mean"]),
            std=tuple(float(v) for v in d["std"]),
            weights=tuple(float(v) for v in d["weights"]),
            bias=float(d["bias"]),
            fit_info=dict(d.get("fit_info", {})),
        )


# ---------------------------------------------------------------------------
# 3. EŞİK SEÇİMİ (T6)
# ---------------------------------------------------------------------------

# Hukuk alanında yanlış kesin yanıt, gereksiz çekimserlikten pahalıdır: dev
# risk bütçesi %5'te sabitlenir ve o bütçe İÇİNDE kapsama en büyüklenir.
DEFAULT_RISK_BUDGET = 0.05
DEFAULT_ALPHA = 0.05


# İstatistiksel güvence etiketleri (review J1). Conformal dalının "n yetersiz"
# kaydının SEÇİLEN eşikteki karşılığı: nokta-tahmini bir riskin kaç satırdan
# ölçüldüğü artefaktın kendisinde durmalı.
GUARANTEE_NONE = "none"
GUARANTEE_CP = "cp_upper<=target"


@dataclass(frozen=True)
class ThresholdChoice:
    """Seçilen eşik + BELİRSİZLİĞİ.

    `risk` bir NOKTA TAHMİNİDİR ve tek başına yanıltıcıdır: 0/4 ile 0/400 aynı
    `0.0`ı yazar. Bu modül aynı ilkeyi conformal dalında zaten uyguluyordu
    (`conformal_candidate` n yetersizken SAYI YAZMAZ) — ama koruma, seçilmeyen
    adaydaydı; SERVİSE ÇIKAN eşikte yoktu (review J1). Artık her seçim
    `n_answered`/`errors`/`risk_cp_upper_95` ve bir `statistical_guarantee`
    bayrağı taşır.

    `statistical_guarantee`:
      * ``"cp_upper<=target"`` — %95 Clopper-Pearson ÜST SINIRI da bütçenin
        altında; risk iddiası tek bir örneklemin şansına bağlı değil.
      * ``"none"`` — yalnız nokta tahmini bütçeyi sağlıyor. Sayı doğru ama
        GÜVENCE YOK; okuyan taraf `risk_cp_upper_95`e bakmak zorunda.

    SEÇİM ÖLÇÜTÜ DEĞİŞMEDİ (bilinçli, v1): eşik hâlâ nokta-tahmini riske göre
    seçilir. CP-üst-sınırını ölçüt yapmak bugünkü n'de feasible kümeyi
    boşaltır ve sistemi tam-çekimsere iter; o karar beş metin özelliğiyle
    değil, verifier sinyali geldiğinde verilmelidir. Buradaki düzeltme
    SEÇİM değil, DÜRÜST ETİKETLEMEdir.
    """

    name: str
    value: float
    coverage: float
    risk: float
    rationale: str
    n_answered: int
    errors: int
    risk_cp_upper_95: float
    statistical_guarantee: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "coverage": self.coverage,
            # nokta tahmini; `risk_point` aynı sayının kendini açıklayan adı
            "risk": self.risk,
            "risk_point": self.risk,
            "n_answered": self.n_answered,
            "errors": self.errors,
            "risk_cp_upper_95": self.risk_cp_upper_95,
            "statistical_guarantee": self.statistical_guarantee,
            "rationale": self.rationale,
        }


def _threshold_uncertainty(
    probs: np.ndarray, labels: np.ndarray, tau: float, max_risk: float
) -> tuple[int, int, float, str]:
    """(n_answered, errors, %95 CP üst sınırı, güvence bayrağı) — seçilen eşikte."""
    from belge_gozu.bench.calibration_metrics import clopper_pearson_upper_bound

    answered = np.asarray(probs) >= tau
    n_answered = int(np.sum(answered))
    if n_answered == 0:
        # Tam çekimser: hiç yanıt yok -> hata da yok, ama bu bir ÖLÇÜM değil.
        # Üst sınır 0.0 yazmak "risk kanıtlanmış sıfır" gibi okunurdu; 1.0
        # (hiçbir şey dışlanmadı) dürüst olan.
        return 0, 0, 1.0, GUARANTEE_NONE
    errors = int(np.sum(np.asarray(labels)[answered] == 0.0))
    upper = clopper_pearson_upper_bound(errors, n_answered, confidence=0.95)
    return (
        n_answered,
        errors,
        upper,
        GUARANTEE_CP if upper <= max_risk else GUARANTEE_NONE,
    )


def choose_threshold(
    probs: np.ndarray, labels: np.ndarray, *, max_risk: float = DEFAULT_RISK_BUDGET
) -> ThresholdChoice:
    """dev risk-coverage taramasında `risk <= max_risk` iken KAPSAMAYI en büyükleyen tau.

    Seçim NOKTA TAHMİNİNE göredir (v1 kararı, bkz. `ThresholdChoice`), ama
    dönen kayıt belirsizliği de taşır: `n_answered`, `errors`,
    `risk_cp_upper_95` ve `statistical_guarantee`.

    Bütçeyi sağlayan hiçbir nokta yoksa TAM ÇEKİMSER eşik döner (coverage=0) —
    sessizce bütçe gevşetilmez; gerekçe dizesi bunu açıkça yazar.
    """
    from belge_gozu.bench.calibration_metrics import risk_coverage

    points = risk_coverage(probs, labels)
    feasible = [(t, c, r) for t, c, r in points if c > 0.0 and r <= max_risk]
    if not feasible:
        tau, cov, risk = max(points, key=lambda p: p[0])
        n_ans, errs, upper, guarantee = _threshold_uncertainty(probs, labels, tau, max_risk)
        return ThresholdChoice(
            name="abstain_all",
            value=tau,
            coverage=cov,
            risk=risk,
            n_answered=n_ans,
            errors=errs,
            risk_cp_upper_95=upper,
            statistical_guarantee=guarantee,
            rationale=(
                f"dev'de risk<={max_risk:.3f} sağlayan hiçbir çalışma noktası yok "
                "(kapsama>0 olan her tau bütçeyi aşıyor); tam çekimser eşik seçildi"
            ),
        )
    # `risk_coverage` BENZERSİZ olasılıkları tarar, yani her tau farklı bir
    # kapsama verir (kapsama tau'da kesin azalan). Bu yüzden "en yüksek kapsama"
    # tek bir noktayı seçer ve ek bir eşitlik-bozma kuralı ULAŞILAMAZDIR
    # (review m12: eski `-tau` dalı ölü koddu).
    tau, cov, risk = max(feasible, key=lambda p: p[1])
    n_ans, errs, upper, guarantee = _threshold_uncertainty(probs, labels, tau, max_risk)
    caveat = (
        ""
        if guarantee == GUARANTEE_CP
        else (
            f"; DİKKAT: bu bir NOKTA TAHMİNİDİR (n={n_ans}, hata={errs}), "
            f"%95 CP üst sınırı {upper:.3f} > bütçe {max_risk:.3f} — İSTATİSTİKSEL GÜVENCE YOK"
        )
    )
    return ThresholdChoice(
        name="risk_budget",
        value=tau,
        coverage=cov,
        risk=risk,
        n_answered=n_ans,
        errors=errs,
        risk_cp_upper_95=upper,
        statistical_guarantee=guarantee,
        rationale=(
            f"dev taramasında risk<={max_risk:.3f} kısıtı altında kapsamayı en büyükleyen tau "
            f"(coverage={cov:.3f}, risk={risk:.3f}){caveat}"
        ),
    )


def conformal_candidate(
    probs: np.ndarray, labels: np.ndarray, *, alpha: float = DEFAULT_ALPHA
) -> dict[str, Any]:
    """Split-conformal eşik + N-YETERLİLİĞİ KONTROLÜ (T7 review nit'i).

    `conformal_threshold` n küçükken sessizce "mümkün en ihtiyatlı eşik"e
    düşer ve bir SAYI döndürür — ama o sayının alpha garantisi YOKTUR:
    ``ceil((n+1)(1-alpha)) > n`` olduğunda sıra istatistiği kalibrasyon
    kümesinin dışına düşer. Sınır ``n >= ceil(1/alpha) - 1``
    (alpha=0.05 -> 19 hata). Altındaysa burada SAYI DEĞİL "n yetersiz" kaydı
    üretilir: boş bir garantiyi eşik diye artefakta yazmak, tam da kalibrasyonun
    önlemesi gereken sessiz yanlışlıktır.
    """
    from belge_gozu.bench.calibration_metrics import conformal_threshold

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha (0,1) aralığında olmalı: {alpha!r}")
    y = np.asarray(labels, dtype=np.float64)
    n_errors = int(np.sum(y == 0.0))
    n_required = math.ceil(1.0 / alpha) - 1
    if n_errors < n_required:
        return {
            "available": False,
            "value": None,
            "alpha": alpha,
            "n_errors": n_errors,
            "n_required": n_required,
            "note": (
                f"conformal: n yetersiz (hata n={n_errors} < gerekli {n_required}); "
                f"alpha={alpha} garantisi BOŞTUR, eşik olarak yazılmadı"
            ),
        }
    tau = conformal_threshold(np.asarray(probs, dtype=np.float64), y, alpha=alpha)
    return {
        "available": True,
        "value": tau,
        "alpha": alpha,
        "n_errors": n_errors,
        "n_required": n_required,
        "note": f"split-conformal, alpha={alpha}, hata n={n_errors}",
    }


# ---------------------------------------------------------------------------
# 4. VERSİYONLU ARTEFAKT (T6) — anahtar = indeks + boru hattı + REÇETE
# ---------------------------------------------------------------------------

CALIBRATOR_FILENAME = "calibrator.json"
ARTIFACT_SCHEMA_VERSION = 1
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class CalibrationKeyMismatch(RuntimeError):
    """Artefakt anahtarı beklenen anahtarla uyuşmuyor (fail-fast)."""


def calibration_key(index_revision: str, pipeline: str, recipe_fp: str | None = None) -> str:
    """`<index_revision-güvenli>__<pipeline>__<recipe_fp>` — artefakt dizin adı.

    ÜÇ BİLEŞEN de zorunludur (P2 gerçeklik denetimi, T6 bulgusu). Planın
    orijinal anahtarı yalnız `index_revision`dı; o dize getirim REÇETESİNİ
    kodlamaz, oysa eşiğin bağlı olduğu eksen tam olarak odur — reçete
    değişince eski eşik geçersizleşmez, sessizce YANLIŞ kalır. `pipeline`
    ayrıca skor ÖLÇEĞİNİ belirler (`config.PIPELINE_SCORE_SCALE`):
    hybrid BM25 birimi, exhaustive/two-stage normalize [-1,1].
    """
    fp = recipe_fingerprint() if recipe_fp is None else recipe_fp
    return f"{_UNSAFE.sub('-', index_revision)}__{_UNSAFE.sub('-', pipeline)}__{fp}"


def calibration_dir(base: Path | str, key: str) -> Path:
    """`<base>/<key>/` — taban varsayılanı `data/calibration` (gitignore'da, yeniden üretilir)."""
    return Path(base) / key


@dataclass(frozen=True)
class CalibrationArtifact:
    key: str
    index_revision: str
    pipeline: str
    recipe_fingerprint: str
    calibrator: Calibrator
    thresholds: dict[str, Any]
    kunye: dict[str, Any]
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "key": self.key,
            "index_revision": self.index_revision,
            "pipeline": self.pipeline,
            "recipe_fingerprint": self.recipe_fingerprint,
            "calibrator": self.calibrator.to_dict(),
            "thresholds": self.thresholds,
            "kunye": self.kunye,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CalibrationArtifact:
        return cls(
            schema_version=int(d.get("schema_version", ARTIFACT_SCHEMA_VERSION)),
            key=str(d["key"]),
            index_revision=str(d["index_revision"]),
            pipeline=str(d["pipeline"]),
            recipe_fingerprint=str(d["recipe_fingerprint"]),
            calibrator=Calibrator.from_dict(d["calibrator"]),
            thresholds=dict(d["thresholds"]),
            kunye=dict(d["kunye"]),
        )

    @property
    def tau(self) -> float:
        """Seçilen çalışma eşiği (`thresholds.chosen.value`)."""
        return float(self.thresholds["chosen"]["value"])

    def save(self, directory: Path | str) -> Path:
        path = Path(directory) / CALIBRATOR_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=1, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def load_calibrator(path: Path | str, expected_key: str) -> CalibrationArtifact:
    """Artefaktı yükler ve ANAHTARI DOĞRULAR — uyuşmazlıkta fail-fast.

    `path` ya `calibrator.json`ın kendisi ya da onu içeren dizindir.

    Anahtar kontrolü bu modülün varlık sebebidir: yanlış bir indekse/reçeteye
    ait bir eşik, "çalışan" ama ölçülmemiş bir sisteme dönüşür ve hiçbir yerde
    hata vermez. Uyuşmazlıkta yükleme yerine `CalibrationKeyMismatch` fırlar.
    """
    p = Path(path)
    if p.is_dir():
        p = p / CALIBRATOR_FILENAME
    if not p.exists():
        raise FileNotFoundError(
            f"kalibrasyon artefaktı yok: {p} — `uv run belge-gozu calibrate fit` ile üretin "
            "(data/calibration/ gitignore'dadır; artefaktlar yeniden üretilebilir)"
        )
    artifact = CalibrationArtifact.from_dict(json.loads(p.read_text(encoding="utf-8")))
    if artifact.key != expected_key:
        raise CalibrationKeyMismatch(
            f"kalibrasyon anahtarı uyuşmuyor: artefakt={artifact.key!r} beklenen={expected_key!r} "
            f"({p}). Anahtar <index_revision>__<pipeline>__<recipe_fp> biçimindedir; "
            "bileşenlerden biri değiştiyse eşik ARTIK ÖLÇÜLMEMİŞ bir boru hattına aittir. "
            "Çözüm: `uv run belge-gozu calibrate fit` ile yeniden fit edin."
        )
    if tuple(artifact.calibrator.feature_names) != FEATURE_ORDER:
        raise CalibrationKeyMismatch(
            f"özellik sırası uyuşmuyor: artefakt={list(artifact.calibrator.feature_names)} "
            f"kod={list(FEATURE_ORDER)} ({p}) — ağırlıklar yanlış özelliklere hizalanırdı"
        )
    return artifact


# ---------------------------------------------------------------------------
# 4b. ÇALIŞMA ANI KAPISI (T2, kapı 1) — `AskService`e takılan yüzey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibratedRetrievalGate:
    """Kalibre güven + artefaktın SEÇTİĞİ eşik: `p < tau` ise çekimser.

    Eşik BURADA SEÇİLMEZ, artefakttan OKUNUR (`thresholds.chosen.value`).
    Servis tarafında ikinci bir eşik seçimi olsaydı "hangi tau ölçüldü?"
    sorusunun iki cevabı olurdu; G2.5'in versiyonlama fikri tam olarak bunu
    engellemek için var.

    DÜRÜSTLÜK: bugünkü artefaktın `statistical_guarantee` alanı `"none"` —
    seçilen tau'nun riski n=4 üzerinde ölçülmüş bir NOKTA TAHMİNİDİR (CP üst
    sınırı %52.7). Kapı bunu saklamaz; künye `detail.gate1.guarantee` olarak
    her olaya yazılır.
    """

    artifact: CalibrationArtifact
    text: BM25Index
    doc_names: Mapping[str, frozenset[str]]
    window: int = WINDOW

    @property
    def tau(self) -> float:
        return self.artifact.tau

    def evaluate(self, question: str, *, bm25: np.ndarray | None = None) -> dict:
        """`{p, tau, passed, features, key}` — `bm25` verilirse YENİDEN skorlamaz."""
        feats = extract_features(question, self.text, self.doc_names, bm25=bm25, window=self.window)
        p = self.artifact.calibrator.predict_one(feats)
        tau = self.tau
        return {
            "p": p,
            "tau": tau,
            "passed": bool(p >= tau),
            "key": self.artifact.key,
            "guarantee": self.artifact.thresholds["chosen"].get("statistical_guarantee"),
            "features": {k: float(v) for k, v in feats.items()},
        }


# ---------------------------------------------------------------------------
# 5. OFFLINE: veri kümesi, fit, değerlendirme (yalnız CLI'den çağrılır)
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha(path: Path | str) -> str:
    """Dosya içeriğinin git BLOB kimliği (`git hash-object`), ya da "unknown".

    review M4: künyedeki `sha256` içeriği kimliklendirir ama içeriği GERİ
    GETİRMEZ. Blob sha'sı getirir — içerik git nesne veritabanındaysa
    `git cat-file -p <blob>` onu aynen üretir, dosya sonradan değişmiş olsa
    bile. Yani koşum girdisi prosa değil, YENİDEN OYNATILABİLİR bir referans
    olur. git yoksa/başarısızsa "unknown" (künye yine sha256 taşır).

    re-review N1: `hash-object` commit'lenmemiş içerik için de sözdizimsel
    olarak geçerli ama nesne veritabanında ÇÖZÜLEMEYEN bir sha üretir — bu,
    "yeniden oynatılabilir referans" amacını sessizce boşa çıkarır. Bu yüzden
    sha, `git cat-file -e` ile erişilebilirlik kontrolünden geçirilir;
    erişilemiyorsa `"<sha>-uncommitted"` döner ki künye dürüst kalsın.
    """
    try:
        out = subprocess.run(
            ["git", "hash-object", str(path)], capture_output=True, text=True, check=True
        )
        sha = out.stdout.strip()
        if not sha:
            return "unknown"
        reachable = subprocess.run(["git", "cat-file", "-e", sha], capture_output=True, text=True)
        return sha if reachable.returncode == 0 else f"{sha}-uncommitted"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def load_rows(path: Path | str, *, only_verified: bool = True) -> list[dict[str, Any]]:
    """Bench JSONL -> HAM sözlükler (şema `BenchQuestion` ile yine de doğrulanır).

    Ham sözlük gerekli çünkü hukuk-gruplu bölme (`assign_split`) alt çizgili
    `_anchor_law` / `_subject_doc` alanlarını okur; `BenchQuestion` (pydantic)
    onları düşürür ve fonksiyon sessizce 50/50 hash kuralına düşerdi.

    `only_verified=True` filtresi `load_bench` ile BİREBİR AYNIDIR:
    `verification_status != "verified"` olan her satır atılır — yani `draft`
    ile birlikte `rejected` de dışarıda kalır (unans_v1'de 14 rejected satır
    vardır ve bunlar veri kümesine girmemelidir).
    """
    from pydantic import ValidationError

    from belge_gozu.bench.dataset import BenchQuestion

    out: list[dict[str, Any]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            BenchQuestion(**rec)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as e:
            raise ValueError(f"bench satır {i}: {e}") from e
        if only_verified and rec.get("verification_status") != "verified":
            continue
        out.append(rec)
    if not out:
        raise ValueError(f"bench boş: {path} içinde yüklenecek soru yok")
    return out


@dataclass(frozen=True)
class LabeledRow:
    question_id: str
    split: str
    answerable: bool
    label: int
    features: dict[str, float]
    gold_in_topk: bool
    unanswerable_reason: str | None
    source: str


def build_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    splits: Mapping[str, set[str]],
    text: BM25Index,
    doc_names: Mapping[str, frozenset[str]],
    *,
    k: int = 5,
    window: int = WINDOW,
    source: str = "",
) -> list[LabeledRow]:
    """Ham bench satırları -> (özellik, etiket) satırları. LLM YOK, model YOK, ağ YOK.

    ETİKET (`safe_to_answer`): 1 YALNIZ soru cevaplanabilir VE gold sayfa
    BM25+yönlendirme top-k'sında ise. 0 ise:
      * cevaplanamaz her soru (asıl risk sınıfı), VE
      * cevaplanabilir ama getirimin ıskaladığı sorular — dayanağı elde
        olmayan bir soruya kesin yanıt vermek DE riskli bölgedir; etiketi 1
        yapmak kalibratöre "kanıtsız yanıtla" demek olurdu.

    Bu etiket LLM'siz ve DÜRÜSTTÜR: modelin ne söylediğini değil, yanıtın
    dayanağının önüne konup konmadığını ölçer. Ölçtüğü şey "yanıt doğru mu"
    değil "yanıt vermek güvenli miydi"dir; ayrım raporda açıkça yazılır.
    """
    from belge_gozu.bench.dataset import assign_split

    if window < k:
        raise ValueError(f"window (>{k}) etiket için yeterli olmalı: window={window}, k={k}")
    rows: list[LabeledRow] = []
    for rec in raw_rows:
        question = str(rec["question"])
        ctx = retrieval_context(question, text, doc_names, window=window)
        feats = features_from_context(question, text, ctx)
        answerable = bool(rec["answerable"])
        gold = set(rec.get("gold_page_ids") or [])
        hit = answerable and bool(gold & set(ctx.window_ranking[:k]))
        rows.append(
            LabeledRow(
                question_id=str(rec["question_id"]),
                split=assign_split(rec, dict(splits)),
                answerable=answerable,
                label=1 if hit else 0,
                features=feats,
                gold_in_topk=hit,
                unanswerable_reason=rec.get("unanswerable_reason"),
                source=source,
            )
        )
    return rows


def class_counts(rows: Sequence[LabeledRow]) -> dict[str, int]:
    """Sınıf/ dilim başına sayım — rapor künyesinin DÜRÜSTLÜK kısmı."""
    answerable = [r for r in rows if r.answerable]
    return {
        "total": len(rows),
        "positive_safe_to_answer": sum(r.label for r in rows),
        "negative": sum(1 for r in rows if r.label == 0),
        "answerable": len(answerable),
        "answerable_gold_in_top5": sum(1 for r in answerable if r.gold_in_topk),
        "answerable_retrieval_miss": sum(1 for r in answerable if not r.gold_in_topk),
        "unanswerable": sum(1 for r in rows if not r.answerable),
    }


def univariate_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """Tek bir HAM özelliğin ayırt etme gücü (rank-tabanlı AUC, beraberlik düzeltmeli).

    `bench.calibration_metrics.auroc` burada KULLANILAMAZ: o fonksiyon girdiyi
    [0,1] KALİBRE OLASILIK olarak doğrular (`_check_probs_labels`) ve
    `served_top1` gibi ham BM25 skorlarında `ValueError` fırlatır — doğru
    davranış, çünkü orada girdi bir olasılıktır. AUC ise sıralama ölçüsüdür ve
    monoton dönüşümlere duyarsızdır; o yüzden ham özellikler için ayrı ve
    açıkça adlandırılmış bir yol (review M3: rapordaki §5.3 sayıları depoda
    yeniden üretilebilir olmalı, ad-hoc bir betikte değil).
    """
    from belge_gozu.bench.calibration_metrics import _average_ranks

    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n_pos, n_neg = int(np.sum(y == 1.0)), int(np.sum(y == 0.0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("univariate_auc için hem pozitif hem negatif etiket gerekli")
    ranks = _average_ranks(v)
    return (float(np.sum(ranks[y == 1.0])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def feature_stats(rows: Sequence[LabeledRow]) -> dict[str, dict[str, float]]:
    """Özellik başına genel + sınıf kırılımlı istatistikler + tek-değişkenli AUC."""
    X = feature_matrix([r.features for r in rows])
    y = np.array([r.label for r in rows], dtype=np.float64)
    # review m9: std ile `Calibrator.fit`in std'si AYNI çağrıdan gelmeli.
    # Sütun görünümünde `col.std()` ile 2B `X.std(axis=0)` farklı toplama
    # sırası kullanıyor ve son ulp'lerde ayrışıyordu; artefaktı çapraz
    # kontrol eden okuyucu bunu bir tutarsızlık sanırdı.
    means, stds = X.mean(axis=0), X.std(axis=0)
    both_classes = bool(np.any(y == 1.0)) and bool(np.any(y == 0.0))
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(FEATURE_ORDER):
        col = X[:, j]
        stats = {
            "mean": float(means[j]),
            "std": float(stds[j]),
            "min": float(col.min()),
            "max": float(col.max()),
        }
        for cls_name, mask in (("pos", y == 1.0), ("neg", y == 0.0)):
            if bool(np.any(mask)):
                stats[f"mean_{cls_name}"] = float(col[mask].mean())
        if both_classes:
            stats["auc"] = univariate_auc(col, y)
        out[name] = stats
    return out


def feature_correlations(rows: Sequence[LabeledRow]) -> dict[str, dict[str, float]]:
    """Özellikler arası Pearson korelasyonu — ağırlık İŞARETLERİNİ okumak için şart.

    Eşdoğrusal özelliklerde çok değişkenli bir ağırlığın işareti tek-değişkenli
    yönü YANSITMAZ (kısmi etki). Bu matris olmadan "neden `matched_terms_top1`
    ağırlığı negatif?" sorusu depodaki veriyle cevaplanamazdı.
    """
    X = feature_matrix([r.features for r in rows])
    if X.shape[0] < 2:
        return {}
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X.T)
    return {
        a: {
            b: (float(C[i, j]) if np.isfinite(C[i, j]) else 0.0)
            for j, b in enumerate(FEATURE_ORDER)
        }
        for i, a in enumerate(FEATURE_ORDER)
    }


def evaluate(
    artifact: CalibrationArtifact,
    rows: Sequence[LabeledRow],
    *,
    tau: float | None = None,
) -> dict[str, Any]:
    """Artefaktı verilen satırlar üzerinde ölçer — `fit` ve `eval` AYNI kodu çağırır.

    İki komutun ayrı hesap yolları olsaydı, "fit raporundaki sayı ile eval
    çıktısı neden farklı?" sorusu kaçınılmazdı.
    """
    from belge_gozu.bench.calibration_metrics import (
        auroc,
        brier,
        ece,
        false_answer_rate_on_unanswerable,
        risk_coverage,
        risk_coverage_auc,
        selective_accuracy,
    )

    X = feature_matrix([r.features for r in rows])
    y = np.array([r.label for r in rows], dtype=np.float64)
    probs = artifact.calibrator.predict_proba(X)
    t = artifact.tau if tau is None else tau
    points = risk_coverage(probs, y)
    curve = [{"tau": a, "coverage": b, "risk": c} for a, b, c in points]

    n_pos, n_neg = int(np.sum(y == 1.0)), int(np.sum(y == 0.0))
    metrics: dict[str, Any] = {
        "n": int(y.size),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "tau": float(t),
        "brier": brier(probs, y),
        "ece": ece(probs, y),
        "aurc": risk_coverage_auc(points),
        "risk_coverage": curve,
    }
    metrics["auroc"] = auroc(probs, y) if (n_pos and n_neg) else None
    cov = float(np.mean(probs >= t))
    metrics["coverage_at_tau"] = cov
    metrics["risk_at_tau"] = float(np.mean(y[probs >= t] == 0.0)) if cov > 0.0 else None
    metrics["selective_accuracy_at_tau"] = selective_accuracy(probs, y, t) if cov > 0.0 else None

    answerable = np.array([r.answerable for r in rows], dtype=bool)
    if bool(np.any(~answerable)):
        rate, n_unans, n_err, upper = false_answer_rate_on_unanswerable(probs, answerable, t)
        metrics["false_answer_on_unanswerable"] = {
            "rate": rate,
            "n": n_unans,
            "errors": n_err,
            "upper_bound_95": upper,
            "method": "clopper_pearson",
            "note": (
                "DEV ÖLÇÜMÜ — G2.1 KAPI SAYISI DEĞİLDİR. Kapı, test bölmesinde "
                "faz sonunda TEK koşumla ölçülür; bu satır eşik seçimi dev'de "
                "yapıldığı için iyimserdir (aynı veride seçilip aynı veride ölçüldü)."
            ),
        }
    return metrics


def fit_calibration(
    rows: Sequence[LabeledRow],
    *,
    index_revision: str,
    pipeline: str,
    recipe_fp: str | None = None,
    max_risk: float = DEFAULT_RISK_BUDGET,
    alpha: float = DEFAULT_ALPHA,
    data_kunye: Mapping[str, Any] | None = None,
) -> CalibrationArtifact:
    """dev satırlarından kalibratör + iki aday eşik + seçilen eşik + künye üretir."""
    X = feature_matrix([r.features for r in rows])
    y = np.array([r.label for r in rows], dtype=np.float64)
    cal = Calibrator.fit(X, y)
    probs = cal.predict_proba(X)

    chosen = choose_threshold(probs, y, max_risk=max_risk)
    conformal = conformal_candidate(probs, y, alpha=alpha)
    fp = recipe_fingerprint() if recipe_fp is None else recipe_fp
    key = calibration_key(index_revision, pipeline, fp)

    artifact = CalibrationArtifact(
        key=key,
        index_revision=index_revision,
        pipeline=pipeline,
        recipe_fingerprint=fp,
        calibrator=cal,
        thresholds={
            # review m10: aday, ADIYLA anahtarlanır. Sabit "risk_budget" anahtarı
            # bütçe sağlanamadığında içinde `name: "abstain_all"` taşıyordu —
            # anahtar içeriğiyle çelişiyordu.
            chosen.name: {"max_risk": max_risk, **chosen.to_dict()},
            "conformal": conformal,
            "chosen": chosen.to_dict(),
        },
        kunye={
            "git_commit": git_commit(),
            "created_at": datetime.now(UTC).isoformat(),
            "counts": class_counts(rows),
            "feature_stats": feature_stats(rows),
            "feature_correlations": feature_correlations(rows),
            **(dict(data_kunye) if data_kunye else {}),
        },
    )
    # dev metrikleri seçilen eşikle birlikte künyeye gömülür: artefakt tek
    # başına "hangi veride, hangi çalışma noktasında ölçüldü"yü taşımalı.
    artifact.kunye["dev_metrics"] = evaluate(artifact, rows)
    return artifact


def per_question_rows(
    artifact: CalibrationArtifact, rows: Sequence[LabeledRow]
) -> list[dict[str, Any]]:
    """Satır satır `(qid, split, label, prob, features, ...)` — koşum kaydının ham tabanı.

    review M3: rapordaki HER sayı (AUC'ler, korelasyonlar, risk-coverage eğrisi,
    seçilen eşikteki hata sayısı) yalnız depodaki dosyalardan yeniden
    hesaplanabilmeli. Toplu metrikler bunu tek başına sağlamıyordu —
    değerlendirmeyi tekrarlamak için indeksi kurup özellikleri yeniden çıkarmak
    gerekiyordu. 173 satır × ~10 alan JSON'a rahatça sığar ve kayıt
    KENDİ KENDİNİ KANITLAR hale gelir.
    """
    X = feature_matrix([r.features for r in rows])
    probs = artifact.calibrator.predict_proba(X)
    tau = artifact.tau
    return [
        {
            "qid": r.question_id,
            "source": r.source,
            "split": r.split,
            "answerable": r.answerable,
            "unanswerable_reason": r.unanswerable_reason,
            "gold_in_topk": r.gold_in_topk,
            "label": r.label,
            "prob": float(p),
            "answered_at_tau": bool(p >= tau),
            "features": {k: float(v) for k, v in r.features.items()},
        }
        for r, p in zip(rows, probs.tolist(), strict=True)
    ]
