"""B kolu — chunking'in getirim kazancını mevcut BM25 kanalında ölçer.

Tek değişken disiplini: yeni model YOK, yeni tokenizer YOK, BM25 parametreleri
ve yönlendirme SABİT. Değişen tek şey indeksleme birimi. Üç kol:

  A  sayfa          üretim tabanı (4.222 sayfa)
  B  madde chunk'ı  gövde metni (10.531 chunk)
  B+ madde chunk'ı  kenar başlığı ÖNEK olarak eklenmiş

A -> B chunking'i izole eder, B -> B+ başlık zenginleştirmesini izole eder.

Chunk sıralaması sayfa sıralamasına indirgenir (sırayı koruyarak, tekrarsız),
çünkü bench'in altın verisi sayfa bazlıdır ve ölçüm sürekliliği korunmalıdır —
"getirim için chunk, kanıt için sayfa" sözleşmesi.

    uv run python scripts/eval_chunking.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.metrics import recall_at_k  # noqa: E402
from belge_gozu.config import Settings  # noqa: E402
from belge_gozu.corpus.chunking import chunk_document  # noqa: E402
from belge_gozu.retrieval.hybrid import load_text_channel  # noqa: E402
from belge_gozu.retrieval.text import BM25Index, route_window  # noqa: E402
from belge_gozu.retrieval.text import routed_docs as routed  # noqa: E402

KS = (1, 5, 20, 50)


def page_ranking_from_chunks(
    order: list[str], chunk_pages: dict[str, tuple[str, ...]]
) -> list[str]:
    """Chunk sırası -> sayfa sırası; ilk görülme kazanır, tekrar atılır."""
    seen: set[str] = set()
    out: list[str] = []
    for cid in order:
        for pid in chunk_pages[cid]:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


def evaluate(name: str, ids: list[str], texts: list[str], rows: list[dict],
             doc_names, chunk_pages: dict[str, tuple[str, ...]] | None) -> dict:
    bm = BM25Index(ids, texts)
    per_slice: dict[str, list[dict]] = defaultdict(list)
    overall: list[dict] = []
    for r in rows:
        if not r["answerable"]:
            continue
        scores = bm.scores(r["question"])
        order = [ids[i] for i in scores.argsort(kind="stable")[::-1]]
        order = route_window(order, routed(r["question"], doc_names))
        pages = page_ranking_from_chunks(order, chunk_pages) if chunk_pages else order
        gold = set(r["gold_page_ids"])
        rec = {f"R@{k}": recall_at_k(gold, pages, k) for k in KS}
        overall.append(rec)
        per_slice[r["slice"]].append(rec)
    agg = {k: sum(x[k] for x in overall) / len(overall) for k in overall[0]}
    return {
        "name": name,
        "n": len(overall),
        "chunks": len(ids),
        "overall": agg,
        "per_slice": {
            s: {"n": len(v), **{k: sum(x[k] for x in v) / len(v) for k in v[0]}}
            for s, v in per_slice.items()
        },
    }


def main() -> int:
    s = Settings()
    df = pd.read_parquet(s.index_dir / "page_texts.parquet")
    df["doc"] = df.page_id.astype(str).str.split(":").str[0]
    df["pno"] = df.page_id.astype(str).str.split(":").str[1].astype(int)
    page_ids = list(df.page_id)
    _, doc_names = load_text_channel(s.index_dir, page_ids)
    bench = (REPO_ROOT / "data/bench/retrieval_eval_v2.jsonl").read_text().splitlines()
    rows = [json.loads(x) for x in bench if x.strip()]

    chunks = []
    for doc, g in df.groupby("doc"):
        pages = [(int(r.pno), r.text) for _, r in g.sort_values("pno").iterrows()]
        chunks += chunk_document(doc, pages)
    cmap = {c.chunk_id: c.page_ids for c in chunks}

    arms = [
        evaluate("A  sayfa (taban)", page_ids, list(df.text), rows, doc_names, None),
        evaluate("B  madde chunk'ı", [c.chunk_id for c in chunks],
                 [c.text for c in chunks], rows, doc_names, cmap),
        evaluate("B+ chunk + başlık", [c.chunk_id for c in chunks],
                 [(f"{c.heading}\n{c.text}" if c.heading else c.text) for c in chunks],
                 rows, doc_names, cmap),
    ]

    print(f"{'kol':22}{'birim':>8}" + "".join(f"{'R@'+str(k):>9}" for k in KS))
    for a in arms:
        print(f"{a['name']:22}{a['chunks']:>8}"
              + "".join(f"{a['overall'][f'R@{k}']:>9.4f}" for k in KS))
    base = arms[0]["overall"]
    for a in arms[1:]:
        print(f"{'  fark (A->' + a['name'][:2] + ')':22}{'':>8}"
              + "".join(f"{a['overall'][f'R@{k}'] - base[f'R@{k}']:>+9.4f}" for k in KS))

    print(f"\n{'dilim':28}" + "".join(f"{n:>12}" for n in ("A R@5", "B R@5", "B+ R@5", "n")))
    for sl in sorted(arms[0]["per_slice"]):
        vals = [a["per_slice"].get(sl, {}).get("R@5", float("nan")) for a in arms]
        print(f"{sl:28}" + "".join(f"{v:>12.4f}" for v in vals)
              + f"{arms[0]['per_slice'][sl]['n']:>12}")

    out = REPO_ROOT / "data/bench/results/chunking-arms.json"
    out.write_text(json.dumps(arms, ensure_ascii=False, indent=1))
    print(f"\nrapor -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
