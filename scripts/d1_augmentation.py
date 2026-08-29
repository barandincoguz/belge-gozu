"""P0 plan — D1: sorgu artırma (query augmentation) token sayısı ablasyonu.

Soru: doküman indeksi SABİT tutulurken, sorgu tarafındaki augmentation token
sayısını (`QueryFormat.n_suffix`, normalde 10) değiştirmek getirim kalitesini
değiştiriyor mu?

İki kol:
  - with-aug: indeksin manifest'indeki `query_format` aynen (n_suffix genelde 10)
  - no-aug  : aynı format ama `n_suffix=0` (prefix/suffix_token/trailing_newline
              korunur — bkz. `noaug_format`)

Her cevaplanabilir bench sorusu her iki kolda da encode edilir, aynı indekse
karşı `ExhaustiveRetriever.score_all` ile tam sıralanır; altın sayfa sıraları
`belge_gozu.bench.oracle.rank_of` ile kaydedilir. İndeks
`belge_gozu.index.loader.load_scorable_index` ile yüklenir: manifest'teki
`quantization`'a göre packed/int8/float16 — yani üretimin skorladığı temsil
neyse D1 de onu ölçer. Doküman prompt'u (`manifest.doc_prompt_sha256`) yalnız
raporlama amaçlıdır — sorgu encode'unda kullanılmaz (sorgular doküman
prompt'una dokunmaz).

Çalıştırma (model ağırlıkları HF önbelleğinden gelir):

    uv run python scripts/d1_augmentation.py --index data/index-traincompat-int8

CI'da koşmaz: model dokunan her şey runbook betiklerinde ya da `-m slow`'da.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.dataset import load_bench  # noqa: E402
from belge_gozu.bench.harness import git_commit  # noqa: E402
from belge_gozu.bench.metrics import mrr, recall_at_k  # noqa: E402
from belge_gozu.bench.oracle import rank_of  # noqa: E402
from belge_gozu.index.loader import load_scorable_index  # noqa: E402
from belge_gozu.index.manifest import CPE_0_3_18, QueryFormat, read_manifest  # noqa: E402

DEFAULT_MODEL = "vidore/colSmol-500M"
RECALL_KS: tuple[int, ...] = (1, 5, 10, 20, 50)


def noaug_format(fmt: QueryFormat) -> QueryFormat:
    """`fmt`'in n_suffix=0 hali (D1'in "no-aug" kolu).

    prefix/suffix_token/trailing_newline AYNEN korunur — yalnız augmentation
    token sayısı sıfırlanır ve ayırt edilebilsin diye format_id'ye "-noaug"
    eklenir."""
    return fmt.model_copy(update={"n_suffix": 0, "format_id": f"{fmt.format_id}-noaug"})


def summarize_arm(
    gold_sets: list[set[str]], rankings: list[list[str]], ks: tuple[int, ...] = RECALL_KS
) -> dict:
    """Saf toplulaştırma: (altın_küme, tam_sıralama) çiftlerinden özet metrik.

    Model/indekse dokunmaz — yalnız `belge_gozu.bench.metrics` üzerinden
    Recall@k ve MRR ortalaması alır; bu yüzden birim testte model gerekmez.
    `gold_sets[i]`/`rankings[i]` aynı soruya ait olmalı (aynı uzunlukta)."""
    if len(gold_sets) != len(rankings):
        raise ValueError(
            f"gold_sets ({len(gold_sets)}) ve rankings ({len(rankings)}) uzunlukları eşleşmiyor"
        )
    n = len(gold_sets)
    if n == 0:
        return {"recall_at": {k: 0.0 for k in ks}, "mrr": 0.0, "n": 0}
    pairs = list(zip(gold_sets, rankings, strict=True))
    return {
        "recall_at": {k: sum(recall_at_k(g, r, k) for g, r in pairs) / n for k in ks},
        "mrr": sum(mrr(g, r) for g, r in pairs) / n,
        "n": n,
    }


def _print_comparison(arms: dict[str, dict]) -> None:
    wa, na = arms["with-aug"]["summary"], arms["no-aug"]["summary"]
    print("\n== D1 karşılaştırma ==")
    print(f"{'kol':<12} {'recall@5':>9} {'recall@20':>10} {'mrr':>8}")
    for name, s in (("with-aug", wa), ("no-aug", na)):
        print(f"{name:<12} {s['recall_at'][5]:>9.3f} {s['recall_at'][20]:>10.3f} {s['mrr']:>8.3f}")
    d5 = wa["recall_at"][5] - na["recall_at"][5]
    d20 = wa["recall_at"][20] - na["recall_at"][20]
    dm = wa["mrr"] - na["mrr"]
    print(f"{'delta(w-n)':<12} {d5:>+9.3f} {d20:>+10.3f} {dm:>+8.3f}")
    if wa["recall_at"][5] > na["recall_at"][5]:
        winner = "with-aug"
    elif na["recall_at"][5] > wa["recall_at"][5]:
        winner = "no-aug"
    else:
        winner = "berabere"
    print(f"SONUÇ: recall@5'te kazanan = {winner} (fark={d5:+.3f})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--index",
        required=True,
        type=Path,
        help=(
            "skorlanacak indeks dizini — SABİT tutulur; temsil (packed/int8/float16) "
            "manifest'ten çözülür"
        ),
    )
    ap.add_argument(
        "--bench",
        type=Path,
        default=REPO_ROOT / "data" / "bench" / "canary_v1.jsonl",
        help="bench JSONL dosyası (varsayılan: data/bench/canary_v1.jsonl)",
    )
    verify_group = ap.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--only-verified",
        action="store_true",
        help="yalnız verification_status=='verified' sorular",
    )
    verify_group.add_argument(
        "--all",
        action="store_true",
        help="taslak dahil tüm sorular (VARSAYILAN — insan doğrulaması hâlâ sürüyor)",
    )
    ap.add_argument(
        "--device",
        default=os.environ.get("BG_DEVICE", "auto"),
        help="ColSmolEncoder cihazı (varsayılan: BG_DEVICE ortam değişkeni ya da 'auto')",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="rapor JSON çıktı yolu (varsayılan: data/bench/results/<tarih>-<sha>-d1.json)",
    )
    args = ap.parse_args()

    only_verified = args.only_verified
    mode = "yalnız doğrulanmış" if only_verified else "tümü (taslak dahil — --all varsayılan)"
    print(f"bench modu: {mode}")

    manifest = read_manifest(args.index)
    if manifest is None:
        print(
            f"UYARI: {args.index} içinde manifest.json yok; query_format=CPE_0_3_18 ve "
            f"model={DEFAULT_MODEL} varsayılıyor (indeksin gerçek yapılandırmasıyla "
            "eşleşmeyebilir)"
        )
        base_format = CPE_0_3_18
        model_name = DEFAULT_MODEL
        doc_prompt_sha256 = None
    else:
        base_format = manifest.query_format
        model_name = manifest.model_name
        # yalnız raporlama amaçlı — sorgu encode'u doküman prompt'unu kullanmaz.
        doc_prompt_sha256 = manifest.doc_prompt_sha256

    aug_format = base_format
    no_aug_format = noaug_format(base_format)
    print(f"with-aug format: {aug_format.format_id!r} (n_suffix={aug_format.n_suffix})")
    print(f"no-aug  format: {no_aug_format.format_id!r} (n_suffix={no_aug_format.n_suffix})")
    if doc_prompt_sha256:
        print(f"doc_prompt_sha256 (bilgi amaçlı, encode'a geçilmiyor): {doc_prompt_sha256[:12]}...")

    idx = load_scorable_index(args.index)
    meta = pd.read_parquet(args.index / "meta.parquet")
    known_page_ids = set(idx.page_ids)

    questions = load_bench(args.bench, only_verified=only_verified)
    answerable = [q for q in questions if q.answerable]
    print(f"bench: {args.bench} -> {len(questions)} soru, {len(answerable)} cevaplanabilir")

    missing_gold_pages = sorted(
        {p for q in answerable for p in q.gold_page_ids if p not in known_page_ids}
    )
    if missing_gold_pages:
        head = missing_gold_pages[:5]
        tail = "..." if len(missing_gold_pages) > 5 else ""
        print(f"UYARI: {len(missing_gold_pages)} altın sayfa indekste yok, atlanacak: {head}{tail}")

    # Model/torch dokunan importlar burada — `--help` bunlara hiç uğramaz.
    from belge_gozu.index.encode import ColSmolEncoder
    from belge_gozu.retrieval.core import ExhaustiveRetriever

    print(f"encoder yükleniyor: model={model_name} device(istenen)={args.device}")
    encoder = ColSmolEncoder(model_name, args.device, query_format=aug_format)
    print(f"  gerçek cihaz={encoder.device}")
    # score_all encoder'a dokunmaz (embedding burada elle hesaplanıp geçiliyor);
    # cli.py'deki `bench oracle` komutuyla aynı desen (encoder=None).
    retriever = ExhaustiveRetriever(idx, meta, None)

    arms: dict[str, dict] = {}
    for arm_name, fmt in (("with-aug", aug_format), ("no-aug", no_aug_format)):
        # Tek encoder örneği iki kol arasında query_format'ı takas ediyor
        # (görev talimatı: takas ucuz ve caiz — burada BİLİNÇLİ yapılıyor).
        encoder.query_format = fmt
        gold_sets: list[set[str]] = []
        rankings: list[list[str]] = []
        per_question: list[dict] = []
        for q in answerable:
            q_emb = encoder.encode_query(q.question)
            scores = retriever.score_all(q_emb)
            order = np.argsort(-scores, kind="stable")
            ranked = [idx.page_ids[i] for i in order]
            gold_ranks: dict[str, int] = {}
            for g in q.gold_page_ids:
                if g in known_page_ids:
                    gold_ranks[g] = rank_of(scores, idx.page_ids, g)
                # eksikse missing_gold_pages'e zaten yukarıda toplandı; atla.
            gold_sets.append(set(q.gold_page_ids) & known_page_ids)
            rankings.append(ranked)
            per_question.append({"question_id": q.question_id, "gold_ranks": gold_ranks})
        summary = summarize_arm(gold_sets, rankings, RECALL_KS)
        arms[arm_name] = {"summary": summary, "per_question": per_question}
        print(
            f"  [{arm_name}] n={summary['n']} recall@5={summary['recall_at'][5]:.3f} "
            f"mrr={summary['mrr']:.3f}"
        )

    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M}-{git_commit()}-d1"
    out_path = args.out or (REPO_ROOT / "data" / "bench" / "results" / f"{run_id}.json")

    report = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "index_dir": str(args.index),
        "index_manifest": manifest.model_dump() if manifest else None,
        "bench": str(args.bench),
        "arms": arms,
        "missing_gold_pages": missing_gold_pages,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    _print_comparison(arms)
    print(f"rapor -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
