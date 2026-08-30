"""Kalibrasyon + seçici yanıtlama (selective answering) ölçütleri (P2 T7).

Saf numpy/stdlib: model yok, ağ yok, yalnızca ölçüm. İki girdi biçimi vardır:

- ``(probs, labels)`` çiftleri — ``probs`` [0,1] aralığında (kalibre) güven
  skorları, ``labels`` aynı boyda {0,1}: 1 = "bu yanıtı sunmak güvenliydi"
  (T6'daki fit etiketiyle birebir), 0 = yanlış/güvenilmez. ``brier``/``ece``/
  ``auroc``/``risk_coverage``/``risk_coverage_auc``/``selective_accuracy``/
  ``abstain_precision_recall``/``conformal_threshold`` bu biçimi kullanır.
- ``(confidence, answerable)`` çifti — G2.1 (cevaplanamaz sorularda
  yanlışlıkla kesin yanıt) ölçümü için: ``false_answer_rate_on_unanswerable``.

Tasarım ilkesi: hiçbir fonksiyon sessizce NaN DÖNDÜRMEZ. 0/0 durumunda ya
(a) eğri bütünlüğü için belgeli tek bir yakınsama uygulanır (yalnız
``risk_coverage``'ın coverage=0 ucu: risk=0.0 — çekimser kalındığında hata da
yoktur), ya da (b) çağıran tek bir eşik/alt küme seçtiği ve sonuç gerçekten
tanımsız olduğu için ``ValueError`` fırlatılır.
"""

import math
from typing import Literal

import numpy as np


def _check_probs_labels(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ortak girdi doğrulaması: 1B, eşit boy, boş değil, labels ∈ {0,1}, sonlu, probs ∈ [0,1]."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.ndim != 1 or y.ndim != 1:
        raise ValueError(f"probs/labels 1 boyutlu olmalı: {p.shape}, {y.shape}")
    if p.shape != y.shape:
        raise ValueError(f"probs ({p.shape}) ve labels ({y.shape}) boyu eşleşmiyor")
    if p.size == 0:
        raise ValueError("boş girdi: en az bir (prob, label) çifti gerekli")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
        raise ValueError("probs/labels sonsuz veya NaN içeremez")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("labels yalnız 0/1 (veya bool) olabilir")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probs [0,1] aralığında olmalı")
    return p, y


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier skoru: mean((p-y)^2) — mükemmel tahminde 0.0, en kötüde 1.0."""
    p, y = _check_probs_labels(probs, labels)
    return float(np.mean((p - y) ** 2))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Beklenen kalibrasyon hatası: örnek-ağırlıklı sum(|acc_bin - conf_bin|).

    Bin kuralı: ``i = min(floor(p * n_bins), n_bins - 1)`` — yani binler
    ``[0, 1/n_bins), ..., [(n_bins-1)/n_bins, 1]`` (son bin p=1.0'ı da
    kapsayacak şekilde kapalıdır). Boş binler toplama katkı yapmaz.
    """
    p, y = _check_probs_labels(probs, labels)
    if n_bins < 1:
        raise ValueError(f"n_bins >= 1 olmalı: {n_bins}")
    idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    n = p.size
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        total += (cnt / n) * abs(acc - conf)
    return total


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-indeksli, beraberlik-düzeltmeli (ortalama sıra) rank vektörü (artan sıraya göre)."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    n = x.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """ROC eğrisi altında kalan alan; rank-tabanlı (Mann-Whitney U), beraberlik-düzeltmeli.

    Mükemmel ayrımda 1.0; skorlar tamamen bağlıysa (ayırt edilemezse) 0.5.
    Girişte hem pozitif hem negatif etiket bulunmalı; tek sınıf varsa
    ValueError (0.5/NaN gibi örtük bir varsayım YOKTUR).
    """
    p, y = _check_probs_labels(probs, labels)
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == 0.0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc için girişte hem pozitif hem negatif etiket bulunmalı")
    ranks = _average_ranks(p)
    sum_pos_ranks = float(np.sum(ranks[y == 1.0]))
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _coverage_and_risk(probs: np.ndarray, labels: np.ndarray, tau: float) -> tuple[float, float]:
    """(coverage, risk): tau eşiğinde ``answered = probs >= tau`` alt kümesi üzerinden.

    answered boşsa (coverage=0) risk=0.0 yakınsaması uygulanır. Bu yakınsama
    YALNIZ ``risk_coverage`` taramasının coverage=0 ucunu tamamlamak için
    vardır (eğri sonsuza kadar sürsün, AUC hesaplanabilsin); tek bir eşik
    seçen ``selective_accuracy``/``abstain_precision_recall`` aynı durumda
    ValueError fırlatır — "hiç yanıt yok" sessizce "doğruluk 1.0" olamaz.
    """
    answered = probs >= tau
    coverage = float(np.mean(answered))
    if not np.any(answered):
        return coverage, 0.0
    risk = float(np.mean(labels[answered] == 0.0))
    return coverage, risk


def risk_coverage(probs: np.ndarray, labels: np.ndarray) -> list[tuple[float, float, float]]:
    """(tau, coverage, risk) üçlüleri; tau süpürmesi = benzersiz probs (artan) + 1 sentinel.

    coverage = mean(probs >= tau); risk = answered içindeki yanlışların oranı
    (coverage=0 ucunda risk=0.0 yakınsamasıyla, bkz. ``_coverage_and_risk``).
    tau arttıkça coverage KESİN AZALMAZ (mean(probs>=tau) tau'da artmayan bir
    fonksiyondur — yani "tau ↑ ⇒ coverage ↓" tekdüzeliği burada geçerlidir).
    İlk nokta tau=min(probs) → coverage=1.0 (tam yanıt/"all-answer"); son
    nokta tau=nextafter(max(probs), inf) → coverage=0.0, risk=0.0 (tam
    çekimser/"all-abstain"). Sentinel sonlu bir float'tır (``math.inf``
    DEĞİL) ki çıktı doğrudan JSON'a yazılabilsin.
    """
    p, y = _check_probs_labels(probs, labels)
    uniq = sorted(set(p.tolist()))
    sentinel = math.nextafter(uniq[-1], math.inf)
    taus = [*uniq, sentinel]
    return [(tau, *_coverage_and_risk(p, y, tau)) for tau in taus]


def risk_coverage_auc(points: list[tuple[float, float, float]]) -> float:
    """Risk-coverage eğrisinin altında kalan alan (AURC, El-Yaniv & Wiener 2010).

    DÜŞÜK DEĞER İYİDİR (ROC-AUC'nin tersine — burada risk bir maliyettir).
    ``points``: (tau, coverage, risk) üçlüleri (``risk_coverage`` çıktısı;
    herhangi bir sırada kabul edilir, coverage'a göre yeniden sıralanır).
    Trapez integrasyonu; en az 2 nokta gerekir.
    """
    if len(points) < 2:
        raise ValueError("risk_coverage_auc için en az 2 nokta gerekli")
    ordered = sorted(points, key=lambda t: t[1])
    coverage = np.array([c for _, c, _ in ordered], dtype=np.float64)
    risk = np.array([r for _, _, r in ordered], dtype=np.float64)
    widths = np.diff(coverage)
    avg_heights = (risk[1:] + risk[:-1]) / 2.0
    return float(np.sum(widths * avg_heights))


def selective_accuracy(probs: np.ndarray, labels: np.ndarray, tau: float) -> float:
    """tau eşiğinde answered (confidence >= tau) alt kümesinin doğruluğu (1 - risk).

    Bu eşikte hiç yanıt yoksa (coverage=0) ValueError — bkz. ``_coverage_and_risk``.
    """
    p, y = _check_probs_labels(probs, labels)
    coverage, risk = _coverage_and_risk(p, y, tau)
    if coverage == 0.0:
        raise ValueError(f"tau={tau!r} eşiğinde hiç yanıt yok; seçici doğruluk tanımsız")
    return 1.0 - risk


def abstain_precision_recall(
    probs: np.ndarray, labels: np.ndarray, tau: float
) -> tuple[float, float]:
    """tau eşiğinde 'abstain' (confidence < tau) kararının (precision, recall)'u.

    Pozitif sınıf = "abstain edilmeliydi" = label==0. precision = abstain
    edilenler içinde gerçekten label==0 olanların payı; recall = TÜM
    label==0 örnekleri içinde abstain edilenlerin payı. Abstain kümesi boşsa
    (tau <= min(probs), tam yanıt/"all-answer") ya da girişte hiç label==0
    yoksa ValueError (precision/recall sırasıyla tanımsızdır).
    """
    p, y = _check_probs_labels(probs, labels)
    abstained = p < tau
    n_abstained = int(np.sum(abstained))
    n_neg = int(np.sum(y == 0.0))
    if n_abstained == 0:
        raise ValueError(f"tau={tau!r} eşiğinde hiç abstain yok; precision tanımsız")
    if n_neg == 0:
        raise ValueError("girişte hiç label==0 yok; recall tanımsız")
    true_abstain = int(np.sum(abstained & (y == 0.0)))
    precision = true_abstain / n_abstained
    recall = true_abstain / n_neg
    return precision, recall


def conformal_threshold(probs: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> float:
    """Split-conformal eşik (Angelopoulos & Bates tarifi).

    label==0 (hatalı) örneklerin confidence'larının ``ceil((n+1)(1-alpha)) / n``
    sıra istatistiği. Yorum: answer-if-confidence>=tau politikasıyla, YENİ
    (değiştirilebilir/exchangeable) bir hatanın eşiği geçme olasılığı marjinal
    olarak <= alpha'dır — yani ``risk_coverage``'daki "risk"i dev kalibrasyon
    kümesinden alpha ile üstten sınırlayan koşullu bir deneydir (üretime
    yalnız dev'de cost-matrix seçiminden iyi kalırsa girer — bkz. plan T6/T7).
    n küçükse ``ceil((n+1)(1-alpha)) > n`` olabilir; bu durumda
    tau=max(hata confidence'ı) döner (mümkün en ihtiyatlı eşik). Kalibrasyon
    kümesinde hiç hata (label==0) yoksa ya da alpha (0,1) dışındaysa ValueError.
    """
    p, y = _check_probs_labels(probs, labels)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha (0,1) aralığında olmalı: {alpha!r}")
    errors = np.sort(p[y == 0.0])
    n = errors.size
    if n == 0:
        raise ValueError("kalibrasyon kümesinde hiç hata (label==0) yok; conformal eşik tanımsız")
    rank = min(math.ceil((n + 1) * (1 - alpha)), n)
    return float(errors[rank - 1])


def _norm_ppf(q: float) -> float:
    """Standart normal dağılımın ters CDF'i (bisection + ``math.erf``; scipy'siz)."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _check_binom(successes: int, n: int, confidence: float) -> None:
    if n <= 0:
        raise ValueError(f"n > 0 olmalı: {n}")
    if not (0 <= successes <= n):
        raise ValueError(f"successes [0, n] aralığında olmalı: {successes}/{n}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence (0,1) aralığında olmalı: {confidence!r}")


def wilson_upper_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """Wilson skor aralığının TEK YÖNLÜ üst sınırı (normal yaklaşıklık, ``z=Φ⁻¹(confidence)``).

    ``successes`` = gözlenen "olay" (ör. hata) sayısı, n = toplam deneme.
    Küçük n + successes=0 durumunda ``clopper_pearson_upper_bound``'a göre
    biraz daha dar (iyimser) olabilir — o, tam/muhafazakâr alternatiftir.
    """
    _check_binom(successes, n, confidence)
    z = _norm_ppf(confidence)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2.0 * n)
    margin = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return min(1.0, (center + margin) / denom)


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    """log P(X=k | n, p); log-uzayda (büyük n'de ``math.comb`` taşmasından kaçınır)."""
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_comb + k * math.log(p) + (n - k) * math.log(1.0 - p)


def _binom_cdf(x: int, n: int, p: float) -> float:
    """P(X <= x | n, p); 0 < p < 1 için log-uzayda toplama, uçlarda kapalı form."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if x >= n else 0.0
    return float(sum(math.exp(_log_binom_pmf(k, n, p)) for k in range(x + 1)))


def clopper_pearson_upper_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """Clopper-Pearson (tam/exact) TEK YÖNLÜ üst sınır.

    ``P(X <= successes | n, p) = alpha`` denklemini p için ikili aramayla
    çözer (alpha = 1 - confidence). n≈150, successes=0 için ~%1.98 verir —
    "kural-3" yaklaşıklığı ``3/n``'le tutarlıdır. ``wilson_upper_bound``'dan
    daha muhafazakârdır (küçük n'de tercih edilen).
    """
    _check_binom(successes, n, confidence)
    if successes == n:
        return 1.0
    alpha = 1.0 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _binom_cdf(successes, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def false_answer_rate_on_unanswerable(
    confidence: np.ndarray,
    answerable: np.ndarray,
    tau: float,
    method: Literal["wilson", "clopper_pearson"] = "clopper_pearson",
) -> tuple[float, int, int, float]:
    """G2.1: cevaplanamaz sorularda yanlışlıkla kesin yanıt oranı + %95 tek-yönlü üst sınır.

    Yalnız ``answerable=False`` alt kümesi: ``confidence >= tau`` olan (yani
    sistemin abstain etmesi gerekirken yanıt sunduğu) örneklerin payı.
    Dönüş: ``(rate, n, hata_sayisi, ust_sinir_95)`` — n≈150 ve hata_sayisi=0
    iken üst sınır Clopper-Pearson ile ~%2'dir (varsayılan ve önerilen
    ``method``; Wilson biraz daha iyimserdir, karşılaştırma için sunulur).
    Girişte hiç cevaplanamaz soru yoksa oran tanımsızdır → ValueError.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    ans = np.asarray(answerable, dtype=bool)
    if conf.shape != ans.shape:
        raise ValueError(f"confidence ({conf.shape}) ve answerable ({ans.shape}) boyu eşleşmiyor")
    if conf.size == 0:
        raise ValueError("boş girdi: en az bir kayıt gerekli")
    unanswerable_conf = conf[~ans]
    n = int(unanswerable_conf.size)
    if n == 0:
        raise ValueError("girişte hiç cevaplanamaz (answerable=False) soru yok")
    errors = int(np.sum(unanswerable_conf >= tau))
    rate = errors / n
    bound_fn = wilson_upper_bound if method == "wilson" else clopper_pearson_upper_bound
    upper = bound_fn(errors, n)
    return rate, n, errors, upper
