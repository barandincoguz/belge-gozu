"""Autoresearch sabit harness (DONUK — bkz. research/program.md).

Tek birincil metrik: retrieval_eval-answerable (43) üzerinde R@5.
Guardrail: R@1, R@20, MRR, requires_visual alt-küme R@5.
Vaka analizleri (chip1/chip2): gold sırası raporlanır, metriğe girmez.

Koşum: uv run python research/evaluate.py <deney-adi>
Çıktı: stdout özeti + research/results.jsonl'a künyeli bir satır.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import retrieve  # tek değiştirilebilir dosya (research/retrieve.py)

DATA = Path("data/research")
RESULTS = Path("research/results.jsonl")


@dataclass(frozen=True)
class QueryContext:
    query_text: str
    page_ids: list[str]
    visual_scores: np.ndarray  # float32[n_pages], page_ids hizalı
    page_texts: list[str]  # page_ids hizalı; taranmış sayfalarda ""


def gold_rank(ranking: list[str], gold: list[str]) -> int:
    pos = {pid: i + 1 for i, pid in enumerate(ranking)}
    ranks = [pos[g] for g in gold if g in pos]
    return min(ranks) if ranks else len(ranking) + 1


def main() -> None:
    exp = sys.argv[1] if len(sys.argv) > 1 else "unnamed"
    meta = json.loads((DATA / "queries.json").read_text())
    scores = np.load(DATA / "visual_scores.npz")["scores"]
    texts_df = pd.read_parquet(DATA / "page_texts.parquet")
    df_map = dict(zip(texts_df.page_id, texts_df.text, strict=True))
    # page_ids sırası: skor matrisi prepare'daki idx.page_ids sırasıyla üretildi;
    # parquet de aynı sırayla yazıldı — yine de kimliği parquet sırasından alıyoruz.
    page_ids = texts_df.page_id.tolist()
    page_texts = [df_map[p] for p in page_ids]
    assert scores.shape == (len(meta["queries"]), len(page_ids)), "hazırlık artefaktları uyumsuz"

    rows = []
    for j, q in enumerate(meta["queries"]):
        ctx = QueryContext(q["question"], page_ids, scores[j], page_texts)
        ranking = retrieve.rank_pages(ctx)
        assert len(set(ranking)) == len(ranking) <= len(page_ids), f"{q['qid']}: geçersiz sıralama"
        rows.append({**q, "rank": gold_rank(ranking, q["gold"])})

    retrieval_eval = [r for r in rows if r["role"] == "retrieval_eval"]
    cases = [r for r in rows if r["role"] == "case_study"]
    n = len(retrieval_eval)

    def r_at(k: int, subset=None) -> float:
        xs = subset if subset is not None else retrieval_eval
        return round(sum(r["rank"] <= k for r in xs) / max(1, len(xs)), 4)

    vis = [r for r in retrieval_eval if r["requires_visual"]]
    metrics = {
        "R@1": r_at(1),
        "R@5": r_at(5),
        "R@20": r_at(20),
        "MRR": round(float(np.mean([1.0 / r["rank"] for r in retrieval_eval])), 4),
        "visual_R@5": r_at(5, vis),
        "n": n,
        "n_visual": len(vis),
    }
    case_ranks = {c["qid"]: c["rank"] for c in cases}

    code = Path("research/retrieve.py").read_bytes()
    row = {
        "exp": exp,
        "retrieve_sha": hashlib.sha256(code).hexdigest()[:12],
        "git": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "index": meta["index_dir"],
        "quantization": meta["quantization"],
        **metrics,
        "case_ranks": case_ranks,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[{exp}] R@5={metrics['R@5']}  R@1={metrics['R@1']}  R@20={metrics['R@20']}  "
        f"MRR={metrics['MRR']}  visual_R@5={metrics['visual_R@5']} (n={n}, görsel {len(vis)})"
    )
    c1, c2 = case_ranks.get("chip1-uzun"), case_ranks.get("chip2-izin")
    print(f"  vaka: chip1-uzun rank={c1}  chip2-izin rank={c2}")
    print(f"  -> {RESULTS} (retrieve_sha {row['retrieve_sha']})")


if __name__ == "__main__":
    # `python research/evaluate.py` script dizinini sys.path'e kendisi ekler;
    # retrieve import'u oradan çözülür.
    main()
