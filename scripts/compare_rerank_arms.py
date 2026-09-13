"""İki rerank koşumunu karar kuralına göre kıyaslar.

    uv run python scripts/compare_rerank_arms.py --base A.json --arm B.json

Toplu metrikler bir kolun daha iyi olduğunu söyler ama HANGİ sorunun kazanıp
kaybettiğini söylemez; sıralama deneylerinde karar tam olarak orada veriliyor
(n=47'de bir soru 0,0213). Bu yüzden hüküm, soru-düzeyi sıra hareketleriyle
BİRLİKTE basılır.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY = ("pinned", "recall_at", "5")
GUARDS = (
    ("pinned R@20", ("pinned", "recall_at", "20")),
    ("pinned R@50", ("pinned", "recall_at", "50")),
    ("nDCG@5", ("pinned", "ndcg5", None)),
)


def _value(report: dict, path: tuple) -> float:
    arm, field, key = path
    node = report[arm]["overall"][field]
    return float(node[key] if key is not None else node)


def _abstain(report: dict) -> int:
    return sum(bool(row["would_abstain"]) for row in report["unpinned"]["diagnostics"])


def _pinned_top5_hits(report: dict) -> set[str]:
    return {
        str(row["question_id"])
        for row in report["unpinned"]["diagnostics"]
        if (rank := row["gold_rank"]["pinned"]) is not None and int(rank) <= 5
    }


def compare(base: dict, arm: dict) -> str:
    n = int(base["pinned"]["overall"]["n"])
    arm_n = int(arm["pinned"]["overall"]["n"])
    if arm_n != n:
        raise ValueError(f"base n={n}, arm n={arm_n} ile eşleşmiyor")
    base_r5, arm_r5 = _value(base, PRIMARY), _value(arm, PRIMARY)
    base_rows = {row["question_id"]: row for row in base["unpinned"]["diagnostics"]}
    arm_rows = {row["question_id"]: row for row in arm["unpinned"]["diagnostics"]}
    if set(base_rows) != set(arm_rows):
        raise ValueError("base ve arm aynı question_id kümesini taşımalı")
    if len(base_rows) != n:
        raise ValueError(f"aggregate n={n}, diagnostics n={len(base_rows)} ile eşleşmiyor")
    base_hits = _pinned_top5_hits(base)
    arm_hits = _pinned_top5_hits(arm)
    delta_q = len(arm_hits) - len(base_hits)
    print(f"P R@5        {base_r5:.4f} -> {arm_r5:.4f}")
    print(f"ilk-5 soru  {len(base_hits)} -> {len(arm_hits)}  ({delta_q:+d}, n={n})")
    print(
        f"  %95 GA     {[round(x, 4) for x in base['pinned']['overall']['ci_recall5']]}"
        f" -> {[round(x, 4) for x in arm['pinned']['overall']['ci_recall5']]}"
    )

    regressed: list[str] = []
    for label, path in GUARDS:
        before, after = _value(base, path), _value(arm, path)
        flag = "  GERİLEME" if after < before - 1e-9 else ""
        print(f"{label:12s} {before:.4f} -> {after:.4f}{flag}")
        if flag:
            regressed.append(label)

    before_abstain, after_abstain = _abstain(base), _abstain(arm)
    print(
        f"çekimser     {before_abstain} -> {after_abstain}"
        + ("  GERİLEME" if after_abstain > before_abstain else "")
    )
    # çekimserlik ve gecikme RAPORLANIR, veto ETMEZ (plan, kural revizyonu):
    # çekimserlik yalnız U kolunda ölçülür, birincil metrik ise P kolundadır.
    print(
        f"rerank p50   {base['latency_ms']['rerank_p50']:.0f}"
        f" -> {arm['latency_ms']['rerank_p50']:.0f} ms"
    )

    print("\nsıra hareketleri (P kolu, yalnız değişenler):")
    for row in arm["unpinned"]["diagnostics"]:
        before_rank = base_rows[row["question_id"]]["gold_rank"]["pinned"]
        after_rank = row["gold_rank"]["pinned"]
        if before_rank == after_rank:
            continue
        in_before = before_rank is not None and before_rank <= 5
        in_after = after_rank is not None and after_rank <= 5
        if in_after and not in_before:
            mark = "  KAZANÇ"
        elif in_before and not in_after:
            mark = "  KAYIP"
        else:
            mark = ""  # ilk beş İÇİNDE yer değiştirme: metriği değiştirmez
        print(f"  {row['question_id']:6s} {row['slice']:24s} {before_rank} -> {after_rank}{mark}")

    if delta_q >= 2 and not regressed:
        verdict = "KEPT"
    elif delta_q == 1 and not regressed:
        verdict = "KEPT (tek soru — mekanizma soru düzeyinde açıklanmalı)"
    else:
        verdict = "DISCARDED"
    print(f"\nHÜKÜM: {verdict}" + (f" — gerileyen: {', '.join(regressed)}" if regressed else ""))
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    args = parser.parse_args()
    compare(
        json.loads(args.base.read_text(encoding="utf-8")),
        json.loads(args.arm.read_text(encoding="utf-8")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
