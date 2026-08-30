"""`data/bench/unans_v1.jsonl` doğrulayıcısı — cevaplanamaz setin makine kapısı.

Bu setin 200 satırlık `korpus-disi` dilimi "insan onaylı" DEĞİLDİR; tek
doğrulaması MEKANİKTİR ve tam olarak şunu iddia eder: *sorunun dayandığı kanun
korpusta yoktur*. Bu betik o iddiayı repodaki veriden yeniden türetilen korpus
kümesine karşı sınar — ezberden kanun listesi taşımaz, `data/state.json` ve
`data/manifest/v0_manifest.csv`'yi okur. Bir kanun korpusa eklenirse (ya da
çıkarılırsa) ilgili satırlar burada patlar; künye sessizce yanlışlanmaz.

Kontroller:
  1. Şema — her satır `BenchQuestion` ile parse edilir (`load_bench` yolu).
  2. Kimlik — `u001..u300` biçimi, tekillik, beklenen sıra.
  3. Dilim sayıları — korpus-disi 200 / anlamsiz-ood 60 / eksik-kanit 40.
  4. Cevaplanamazlık değişmezleri — `answerable=False`, gold_* boş,
     `requires_visual/requires_multi_hop=False`, `source_type="ajan-taslak"`.
  5. korpus-dışı çapa — `_anchor_law` korpus kanun numaraları arasında OLMAMALI
     (numara kontrolü) ve `_anchor_name` hiçbir korpus belge adıyla ~aynı
     olmamalı (ad-token kontrolü; sadece numara bakmak yetmez, aynı kanun farklı
     numarayla yazılmış olabilir). Ayrıca soru metni çapayı ANMALI.
  6. eksik-kanıt konusu — `_subject_doc` korpusta OLMALI.
  7. Doğrulama künyesi — dilim başına beklenen `verification_status` /
     `verified_by` / `verification_kind` üçlüsü.
  8. Yakın-tekrar — set içinde ve canary'ye karşı, tr-duyarlı normalize edilmiş
     token kümesi örtüşmesi (Jaccard) >= 0.8 olan çiftler işaretlenir.
  9. Split — `data/bench/splits_v1.json` künyedeki kuralla yeniden türetilir ve
     dosyadaki listeyle karşılaştırılır; `assign_split` ile bileşim basılır.

Kullanım:
    uv run python scripts/validate_unans.py
    uv run python scripts/validate_unans.py --bench data/bench/unans_v1.jsonl

Herhangi bir ihlalde çıkış kodu 1'dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from belge_gozu.bench.dataset import (  # noqa: E402
    BenchQuestion,
    assign_split,
    load_splits,
)

SLICE_EXPECT = {"korpus-disi": 200, "anlamsiz-ood": 60, "eksik-kanit": 40}
# dilim -> (verification_status, verified_by, verification_kind)
# Dilim başına İZİNLİ doğrulama künyeleri (status, verified_by, kind).
# Çapraz-kontrol (2026-08-30, drafter≠checker) sonrası durumlar da geçerlidir;
# "rejected" satırlar her dilimde meşrudur (tüketiciler status ile dışlar) —
# yalnız künye bütünlüğü aranır. Bkz. data/bench/unans_v1.README.md §çapraz-kontrol.
_CHECKER = "model-cross-check:claude-fable-5-checker"
VERIF_EXPECT: dict[str, set[tuple[str, str, str]]] = {
    "korpus-disi": {
        ("verified", "script:validate_unans", "mechanical:manifest-absence"),
        ("verified", _CHECKER, "mechanical:manifest-absence"),
        # red kararı çapraz-kontrolden gelir; denetçi kind'i buna çevirir:
        ("rejected", _CHECKER, "model-cross-check"),
    },
    "anlamsiz-ood": {
        ("draft", "", "model-cross-check"),
        ("verified", _CHECKER, "model-cross-check"),
        ("rejected", _CHECKER, "model-cross-check"),
    },
    "eksik-kanit": {
        ("draft", "", "model-cross-check"),
        ("verified", _CHECKER, "model-cross-check"),
        ("rejected", _CHECKER, "model-cross-check"),
    },
}
DUP_THRESHOLD = 0.8
# ad karşılaştırmasında ayırt edici olmayan kelimeler
NAME_STOPWORDS = {
    "kanun",
    "kanunu",
    "kanunun",
    "hakkinda",
    "hakkindaki",
    "dair",
    "ve",
    "ile",
    "sayili",
    "iliskin",
    "genel",
    "bazi",
    "dolayisiyle",
    "arasindaki",
}


def tr_lower(s: str) -> str:
    """Yerel-ayardan bağımsız Türkçe küçültme (İ->i, I->ı ... sonra casefold)."""
    return (
        s.replace("İ", "i")
        .replace("I", "ı")
        .replace("Ş", "ş")
        .replace("Ğ", "ğ")
        .replace("Ü", "ü")
        .replace("Ö", "ö")
        .replace("Ç", "ç")
        .casefold()
    )


def fold(s: str) -> str:
    """Ad karşılaştırması için aksanları düzleştirir (şğüöçı -> sguoci)."""
    table = str.maketrans("şğüöçı", "sguoci")
    return tr_lower(s).translate(table)


def tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^0-9a-zçğıöşü]+", tr_lower(s)) if t}


def name_tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^0-9a-z]+", fold(s)) if t and t not in NAME_STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------- korpus
def load_corpus(repo: Path) -> tuple[set[str], dict[str, str]]:
    """Korpusun kesin kümesini repodaki veriden türetir.

    Dönüş: (doc_id kümesi, doc_id -> doc_name). İki kaynak (state.json anahtarları
    ve manifest satırları) BİRBİRİNE KARŞI sınanır; ayrışırlarsa hata verilir —
    çünkü çapaların "korpusta yok" iddiası tam olarak bu kümeye dayanır.
    """
    state = json.loads((repo / "data/state.json").read_text(encoding="utf-8"))
    state_ids = set(state.keys())
    names: dict[str, str] = {}
    with (repo / "data/manifest/v0_manifest.csv").open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            names[rec["doc_id"]] = rec["doc_name"]
    if state_ids != set(names):
        only_state = sorted(state_ids - set(names))
        only_man = sorted(set(names) - state_ids)
        raise SystemExit(
            f"korpus kaynakları ayrışıyor: yalnız state.json={only_state} "
            f"yalnız manifest={only_man}"
        )
    return state_ids, names


def corpus_law_numbers(doc_ids: set[str]) -> set[str]:
    """`k4857` -> `4857`. RG tarama belgelerinin kanun numarası yoktur."""
    return {d[1:] for d in doc_ids if re.fullmatch(r"k\d+", d)}


# ------------------------------------------------------------------ kontroller
def check_rows(rows: list[dict], corpus_ids: set[str], corpus_names: dict[str, str]) -> list[str]:
    errs: list[str] = []
    law_numbers = corpus_law_numbers(corpus_ids)
    corpus_name_tokens = {d: name_tokens(n) for d, n in corpus_names.items()}

    # 1) şema
    for i, rec in enumerate(rows, start=1):
        try:
            BenchQuestion(**rec)
        except Exception as e:  # pydantic ValidationError dahil
            errs.append(f"satır {i} ({rec.get('question_id')}): şema — {e}")

    # 2) kimlik
    seen: set[str] = set()
    for i, rec in enumerate(rows, start=1):
        qid = rec.get("question_id", "")
        if not re.fullmatch(r"u\d{3}", qid):
            errs.append(f"satır {i}: question_id biçimi bozuk: {qid!r}")
        if qid in seen:
            errs.append(f"satır {i}: question_id tekrar ediyor: {qid}")
        seen.add(qid)
        if qid != f"u{i:03d}":
            errs.append(f"satır {i}: beklenen id u{i:03d}, bulunan {qid}")

    # 3) dilim sayıları
    counts = Counter(r.get("slice") for r in rows)
    for sl, want in SLICE_EXPECT.items():
        if counts.get(sl, 0) != want:
            errs.append(f"dilim {sl}: {counts.get(sl, 0)} satır, beklenen {want}")
    extra = set(counts) - set(SLICE_EXPECT)
    if extra:
        errs.append(f"beklenmeyen dilim(ler): {sorted(extra)}")

    for rec in rows:
        qid, sl = rec.get("question_id"), rec.get("slice")

        # 4) cevaplanamazlık değişmezleri
        if rec.get("answerable") is not False:
            errs.append(f"{qid}: answerable False olmalı")
        for fld in ("gold_doc_ids", "gold_page_ids", "gold_article_ids", "minimal_evidence_spans"):
            if rec.get(fld) != []:
                errs.append(f"{qid}: {fld} boş olmalı")
        if rec.get("reference_answer") != "":
            errs.append(f"{qid}: reference_answer boş olmalı")
        if rec.get("requires_visual") is not False or rec.get("requires_multi_hop") is not False:
            errs.append(f"{qid}: requires_visual/requires_multi_hop False olmalı")
        if rec.get("source_type") != "ajan-taslak":
            errs.append(f"{qid}: source_type 'ajan-taslak' olmalı (insan onayı yok)")
        if not rec.get("verification_note"):
            errs.append(f"{qid}: verification_note boş olamaz")

        # 7) doğrulama künyesi
        allowed = VERIF_EXPECT.get(sl)
        if allowed:
            got = (
                rec.get("verification_status"),
                rec.get("verified_by"),
                rec.get("verification_kind"),
            )
            if got not in allowed:
                errs.append(f"{qid}: doğrulama künyesi {got} izinli kümede değil ({sl})")

        # 5) korpus-dışı çapa
        if sl == "korpus-disi":
            if rec.get("unanswerable_reason") != "korpus-disi":
                errs.append(f"{qid}: unanswerable_reason 'korpus-disi' olmalı")
            anchor = rec.get("_anchor_law")
            aname = rec.get("_anchor_name")
            if not anchor or not aname:
                errs.append(f"{qid}: _anchor_law/_anchor_name zorunlu")
                continue
            if anchor in law_numbers:
                errs.append(f"{qid}: ÇAPA KORPUSTA — {anchor} sayılı kanun k{anchor} olarak mevcut")
            at = name_tokens(aname)
            for doc, ct in corpus_name_tokens.items():
                if jaccard(at, ct) >= DUP_THRESHOLD:
                    errs.append(
                        f"{qid}: çapa adı korpus belgesiyle örtüşüyor — "
                        f"{aname!r} ~ {doc} {corpus_names[doc]!r} "
                        f"(jaccard {jaccard(at, ct):.2f})"
                    )
            # soru çapayı anmalı: ya numara ya da adın ayırt edici tokenlarının tümü
            qtok = tokens(rec.get("question", ""))
            qname = name_tokens(rec.get("question", ""))
            if anchor not in qtok and not at.issubset(qname):
                errs.append(f"{qid}: soru metni çapayı anmıyor ({anchor} / {aname!r} yok)")

        # 6) eksik-kanıt konusu
        elif sl == "eksik-kanit":
            if rec.get("unanswerable_reason") != "eksik-kanit":
                errs.append(f"{qid}: unanswerable_reason 'eksik-kanit' olmalı")
            subject = rec.get("_subject_doc")
            if not subject:
                errs.append(f"{qid}: _subject_doc zorunlu")
            elif subject not in corpus_ids:
                errs.append(f"{qid}: _subject_doc korpusta yok: {subject}")

        elif sl == "anlamsiz-ood":
            if rec.get("unanswerable_reason") != "anlamsiz":
                errs.append(f"{qid}: unanswerable_reason 'anlamsiz' olmalı")

    return errs


def find_near_dupes(
    rows: list[dict], canary: list[dict], threshold: float = DUP_THRESHOLD
) -> list[tuple[str, str, float]]:
    """Normalize token kümesi Jaccard'ı eşiği aşan çiftler (set içi + canary'ye karşı)."""
    hits: list[tuple[str, str, float]] = []
    items = [(r["question_id"], tokens(r["question"])) for r in rows]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = jaccard(items[i][1], items[j][1])
            if s >= threshold:
                hits.append((items[i][0], items[j][0], s))
    cit = [(c["question_id"], tokens(c["question"])) for c in canary]
    for qid, qt in items:
        for cid, ct in cit:
            s = jaccard(qt, ct)
            if s >= threshold:
                hits.append((qid, f"canary:{cid}", s))
    return hits


# ----------------------------------------------------------------------- split
def derive_test_docs(
    corpus_ids: set[str],
    doc_types: dict[str, str],
    seed: str,
    pinned: list[str],
    size: int,
    canary_docs: set[str],
) -> list[str]:
    """splits_v1.json'daki `derivation` kuralının makine karşılığı.

    `canary_docs`: canary'de cevaplanabilir sorusu OLAN belgeler. Sabitlenenler
    dışındakiler doldurmaya kapalıdır — aksi halde test'e giren her ek canary
    belgesi 26/17 hedefini kaydırırdı.
    """

    def h(doc: str) -> str:
        return hashlib.sha256(f"{seed}|{doc}".encode()).hexdigest()

    blocked = set(pinned) | (canary_docs - set(pinned))
    free = sorted(d for d in corpus_ids if d not in blocked)
    free_rg = sorted((d for d in free if doc_types[d] == "rg_tarihi"), key=h)
    rest = sorted((d for d in free if doc_types[d] != "rg_tarihi"), key=h)
    need = size - len(pinned)
    # kuralın garantisi: test'te en az 2 RG tarama belgesi
    rg_in_pinned = sum(1 for d in pinned if doc_types[d] == "rg_tarihi")
    take_rg = max(0, 2 - rg_in_pinned)
    chosen = free_rg[:take_rg] + rest[: need - take_rg]
    return sorted(list(pinned) + chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="data/bench/unans_v1.jsonl")
    ap.add_argument("--splits", default="data/bench/splits_v1.json")
    ap.add_argument("--canary", default="data/bench/canary_v1.jsonl")
    args = ap.parse_args()

    bench_path = (REPO / args.bench) if not Path(args.bench).is_absolute() else Path(args.bench)
    splits_path = (REPO / args.splits) if not Path(args.splits).is_absolute() else Path(args.splits)
    canary_path = (REPO / args.canary) if not Path(args.canary).is_absolute() else Path(args.canary)

    corpus_ids, corpus_names = load_corpus(REPO)

    def read_jsonl(p: Path) -> list[dict]:
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    rows = read_jsonl(bench_path)
    canary = read_jsonl(canary_path)

    errs = check_rows(rows, corpus_ids, corpus_names)
    dupes = find_near_dupes(rows, canary)
    errs += [f"yakın-tekrar: {a} ~ {b} (jaccard {s:.2f})" for a, b, s in dupes]

    try:
        shown_path = bench_path.relative_to(REPO)
    except ValueError:  # repo dışı --bench yolu: mutlak göster, çökme (inceleme L1)
        shown_path = bench_path
    print("=" * 72)
    print(f"unans doğrulama — {shown_path}")
    print("=" * 72)
    print(f"korpus: {len(corpus_ids)} belge, {len(corpus_law_numbers(corpus_ids))} kanun numarası")
    print(f"satır : {len(rows)}")
    print()

    hdr = f"{'dilim':<15}{'satır':>6}{'çapa/konu':>12}  {'doğrulama':<38}{'stil dağılımı':<26}"
    print(hdr)
    print("-" * len(hdr))
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)
    for sl in ("korpus-disi", "anlamsiz-ood", "eksik-kanit"):
        rs = by_slice.get(sl, [])
        if sl == "korpus-disi":
            uniq = len({r.get("_anchor_law") for r in rs})
            anchor_col = f"{uniq} kanun"
        elif sl == "eksik-kanit":
            uniq = len({r.get("_subject_doc") for r in rs})
            anchor_col = f"{uniq} belge"
        else:
            anchor_col = "-"
        vk = Counter(r.get("verification_kind") for r in rs)
        vs = Counter(r.get("verification_status") for r in rs)
        verif = f"{'/'.join(sorted(vs))}:{list(vk)[0] if len(vk) == 1 else 'karışık'}"
        st = Counter(r.get("query_style") for r in rs)
        styles = " ".join(f"{k[:4]}={v}" for k, v in sorted(st.items()))
        print(f"{sl:<15}{len(rs):>6}{anchor_col:>12}  {verif:<38}{styles:<26}")
    print("-" * len(hdr))
    diff = Counter(r.get("difficulty") for r in rows)
    print(f"zorluk: {dict(sorted(diff.items()))}")
    print(f"kaynak: {dict(Counter(r.get('source_type') for r in rows))}")
    print()

    # split bileşimi
    if splits_path.exists():
        meta = json.loads(splits_path.read_text(encoding="utf-8"))
        splits = load_splits(splits_path)
        doc_types: dict[str, str] = {}
        with (REPO / "data/manifest/v0_manifest.csv").open(encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                doc_types[rec["doc_id"]] = rec["doc_type"]
        rule = meta.get("derivation", {})
        if rule:
            canary_docs = {
                c["gold_doc_ids"][0] for c in canary if c["answerable"] and c["gold_doc_ids"]
            }
            want = derive_test_docs(
                corpus_ids,
                doc_types,
                seed=meta["seed"],
                pinned=rule["pinned_test_docs"],
                size=rule["test_doc_count"],
                canary_docs=canary_docs,
            )
            if want != sorted(meta["test_docs"]):
                errs.append(
                    "split: test_docs künyedeki kuraldan yeniden türetilemedi "
                    f"(türetilen {len(want)} belge, dosyadaki {len(meta['test_docs'])})"
                )
        n_rg = sum(1 for d in meta["test_docs"] if doc_types.get(d) == "rg_tarihi")
        if n_rg < 2:
            errs.append(f"split: test kümesinde yalnız {n_rg} RG tarama belgesi var (>=2 gerekli)")
        if set(meta["dev_docs"]) & set(meta["test_docs"]):
            errs.append("split: dev_docs ve test_docs kesişiyor")
        if set(meta["dev_docs"]) | set(meta["test_docs"]) != corpus_ids:
            errs.append("split: dev+test birleşimi korpusu tam kapsamıyor")

        comp: Counter[tuple[str, str]] = Counter()
        for r in rows:
            comp[(assign_split(r, splits), r["slice"])] += 1
        for c in canary:
            kind = "cevaplanabilir" if c["answerable"] else "cevaplanamaz"
            comp[(assign_split(c, splits), f"canary-{kind}")] += 1

        seed = meta.get("seed")
        print(f"split bileşimi (seed={seed!r}, test_docs={len(meta['test_docs'])}, RG={n_rg})")
        print(f"{'küme':<6}{'dilim':<24}{'adet':>6}")
        print("-" * 36)
        for (k, sl), n in sorted(comp.items()):
            print(f"{k:<6}{sl:<24}{n:>6}")
        print("-" * 36)
        for k in ("dev", "test"):
            una = sum(
                n for (kk, sl), n in comp.items() if kk == k and sl != "canary-cevaplanabilir"
            )
            ans = comp.get((k, "canary-cevaplanabilir"), 0)
            print(f"{k:<6}{'TOPLAM cevaplanamaz':<24}{una:>6}   cevaplanabilir: {ans}")
        print()

    if errs:
        print(f"İHLAL: {len(errs)}")
        for e in errs[:60]:
            print(f"  - {e}")
        if len(errs) > 60:
            print(f"  ... ve {len(errs) - 60} tane daha")
        return 1
    print("TEMİZ — tüm kontroller geçti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
