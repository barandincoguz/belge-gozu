"""Canary bench — insan doğrulama aracı (ön-kontrol + interaktif inceleme).

`data/bench/canary_v1.jsonl`'daki tüm sorular `verification_status: "draft"`
ile başlar; sayıların gate-kalitesinde sayılması için proje sahibinin her
birini elle onaylaması gerekir. Bu betik üç modda çalışır:

  --report   Makinenin kontrol edebildiği her şeyi kontrol eder (kanıt
             alıntısı sayfada geçiyor mu, altın sayfa üretim indeksinde var
             mı, gold_page_ids/gold_doc_ids öneki tutarlı mı, sayfa taranmış
             mı) ve soruları TEMİZ / ŞÜPHELİ / MANUEL gruplarına ayırır.
             Salt-okunur: hiçbir dosyayı DEĞİŞTİRMEZ. Terminale kompakt bir
             tablo + grup sayıları basar, ayrıca `data/bench/canary_precheck.md`
             yazar.

             uv run python scripts/verify_canary.py --report

  --review   Hâlâ "draft" olan soruları tek tek gösterir (soru, referans
             cevap, kanıt alıntıları, ön-kontrol sonucu), altın sayfa
             görüntüsünü açar, klavyeden karar okur (e=doğrula, h=reddet,
             a=atla, q=kaydet-çık; harften sonra isteğe bağlı not) ve HER
             karardan sonra dosyayı ATOMİK yazar (kesintiye dayanıklı).

             uv run python scripts/verify_canary.py --review --by baran
             uv run python scripts/verify_canary.py --review --by baran --only ŞÜPHELİ

  --status   verification_status / dilim (slice) / doğrulama TÜRÜ (insan vs.
             model çapraz-kontrolü) dağılımını ve plan hedefinin (>=25
             doğrulanmış cevaplanabilir + >=5 doğrulanmış cevaplanamaz)
             karşılanıp karşılanmadığını basar. Hedef satırı hem birleşik hem
             yalnız-insan sayısını gösterir; birleşik sayı insan doğrulaması
             DEĞİLDİR (bkz. data/bench/canary_v1.README.md).

             uv run python scripts/verify_canary.py --status

Normalizasyon notu (--report'taki alıntı eşleşmesi için): karşılaştırmadan
önce hem sayfa metni hem alıntı aynı iki adımdan geçer — (1) boşluk/satır
sonları tek boşluğa indirgenir, (2) Türkçe'ye duyarlı küçültme uygulanır
(İ->i, I->ı, ardından str.casefold()). Python'ın öntanımlı .lower()'ı
yerel-ayara duyarsız olduğundan (İ.lower() tek başına yanlış sonuç verir),
bu eşleme elle yapılıyor; iki taraf da AYNI fonksiyondan geçtiği için
tutarsızlık riski yok.

CI'da koşmaz (PDF/indeks I/O gerektirir); testler saf mantığı (normalizasyon,
alıntı eşleşmesi, atomik round-trip, durum sayımı) PDF/görüntü/stdin OLMADAN
kapsar — bkz. tests/test_verify_canary.py.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.dataset import BenchQuestion, bench_stats, load_bench  # noqa: E402
from belge_gozu.config import Settings  # noqa: E402
from belge_gozu.retrieval.text import tokenize as recipe_tokenize  # noqa: E402

DEFAULT_BENCH = REPO_ROOT / "data" / "bench" / "canary_v1.jsonl"
DEFAULT_PDF_DIR = REPO_ROOT / "data" / "pdf"
DEFAULT_IMAGES_DIR = REPO_ROOT / "data" / "images"
DEFAULT_REPORT_OUT = REPO_ROOT / "data" / "bench" / "canary_precheck.md"

# "sayfa boş/çok kısa" eşiği: bunun altı taranmış sayfa varsayılır (rg* belgeleri).
SCANNED_MIN_CHARS = 40

# `paraphrase` dilimi için AZAMİ sözlüksel örtüşme. Sorunun içerik
# token'larının bu orandan fazlası gold sayfada birebir geçiyorsa satır
# yeniden ifade edilmiş DEĞİLDİR ve dilim etiketi yanlıştır.
#
# 0.5 keyfi değil, çapraz-kontrol turunun bulduğu üç gerçek kusurdan türedi:
# c001 ("Yerleşim yeri nedir?"), c002 ve c108'in üçü de 1.0 örtüşmeyle
# `paraphrase` etiketliydi ve elle `dogrudan-madde`ye düzeltildi. Yarıdan
# fazlası birebir geçen bir soru, BM25'in kelime eşleşmesini ölçer; oysa bu
# dilim TAM OLARAK kelime eşleşmesinin olmadığı durumu ölçmek için var.
PARAPHRASE_MAX_OVERLAP = 0.5

Group = str  # "TEMİZ" | "ŞÜPHELİ" | "MANUEL"
GROUPS: tuple[Group, ...] = ("TEMİZ", "ŞÜPHELİ", "MANUEL")


# --------------------------------------------------------------------------
# Saf yardımcılar (PDF/ağ/stdin YOK — hepsi birim testli)
# --------------------------------------------------------------------------


def tr_lower(s: str) -> str:
    """Türkçe'ye duyarlı küçültme: İ->i, I->ı, ardından casefold().

    Python'ın öntanımlı str.lower()'ı yerel ayara duyarsızdır (İ.lower() ==
    'i̇' iki karakterli bileşik nokta üretir, I.lower() == 'i' kalır — Türkçe
    beklentisi I->ı'dır). Burada elle eşleyip casefold ile devam ediyoruz.
    """
    return s.replace("İ", "i").replace("I", "ı").casefold()


def normalize_ws(s: str) -> str:
    """Boşluk/satır sonlarını tek boşluğa indirger, baş/son boşluğu kırpar."""
    return re.sub(r"\s+", " ", s).strip()


def normalize_for_match(s: str) -> str:
    """Eşleşme için tam normalizasyon: boşluk indirgeme + Türkçe küçültme."""
    return tr_lower(normalize_ws(s))


def span_found(span: str, page_text: str) -> bool:
    """`span` normalize edilmiş `page_text` içinde birebir alt-dize olarak geçiyor mu."""
    if not span.strip():
        return True
    return normalize_for_match(span) in normalize_for_match(page_text)


def closest_line(span: str, page_text: str) -> str | None:
    """`span`e en çok benzeyen orijinal satırı döner (insan farkı hemen görsün diye).

    Salt bilgi amaçlıdır (eşleşme kararını etkilemez); sayfa metni yoksa None.
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return None
    target = normalize_for_match(span)

    def ratio(ln: str) -> float:
        return difflib.SequenceMatcher(None, target, normalize_for_match(ln)).ratio()

    return max(lines, key=ratio)


def content_overlap(question: str, page_text: str) -> float:
    """Sorunun içerik token'larının kaçta kaçı gold sayfada da geçiyor (0..1).

    ÜRETİM reçetesinin tokenizer'ını kullanır (`retrieval.text.tokenize`:
    tr_lower -> ASCII katlama -> stopword eleme -> F5 kırpma) — kendi
    tokenizasyonunu yazmaz. Sebep: kapı, getirimin GERÇEKTEN gördüğü örtüşmeyi
    ölçmeli. Ayrı bir tokenizer, ölçtüğünü sandığımız şeyden sapardı; üstelik
    reçete yazım-değişmez olduğu için aksanlı ve aksansız yazılmış aynı soru
    aynı oranı alır.

    Kapsama (containment) oranıdır, Jaccard DEĞİL: sayfa metni sorudan kat kat
    uzun olduğu için Jaccard her zaman sıfıra yakın çıkar ve hiçbir şey ayırt
    etmez. Sorulan şey "sorunun ne kadarı sayfadan aynen alınmış".
    """
    q_tokens = set(recipe_tokenize(question))
    if not q_tokens:
        return 0.0
    page_tokens = set(recipe_tokenize(page_text))
    return len(q_tokens & page_tokens) / len(q_tokens)


def doc_prefix_consistent(gold_doc_ids: list[str], gold_page_ids: list[str]) -> bool:
    """Her gold_page_ids öneki gold_doc_ids içinde mi (şema tutarlılığı)."""
    prefixes = {gp.split(":", 1)[0] for gp in gold_page_ids if ":" in gp}
    return prefixes.issubset(set(gold_doc_ids))


@dataclass
class SpanCheck:
    span: str
    found: bool
    page_id: str  # eşleşen (ya da denenen ilk) altın sayfa
    closest: str | None = None  # yalnız found=False iken doldurulur


@dataclass
class QuestionPrecheck:
    question_id: str
    group: Group
    notes: list[str] = field(default_factory=list)
    span_checks: list[SpanCheck] = field(default_factory=list)


def precheck_question(
    q: BenchQuestion, page_texts: dict[str, str | None], known_page_ids: set[str]
) -> QuestionPrecheck:
    """Tek bir soru için makine kontrolü: sonuç TEMİZ / ŞÜPHELİ / MANUEL.

    `page_texts`: "dok:sayfa" -> PyMuPDF'ten çıkarılan düz metin (bulunamadıysa
    None). `known_page_ids`: üretim indeksinin page_ids.json kümesi. İkisi de
    çağıran tarafından hazırlanır — bu fonksiyon hiçbir I/O yapmaz (test edilebilir).
    """
    if not q.answerable:
        return QuestionPrecheck(
            q.question_id, "TEMİZ", ["cevaplanamaz soru: kontrol edilecek kanıt yok"]
        )

    notes: list[str] = []
    if not doc_prefix_consistent(q.gold_doc_ids, q.gold_page_ids):
        notes.append("gold_page_ids öneki gold_doc_ids'te yok (şema tutarsızlığı)")

    missing_from_index = [gp for gp in q.gold_page_ids if gp not in known_page_ids]
    if missing_from_index:
        notes.append(f"indekste yok: {', '.join(missing_from_index)}")

    scanned_pages: list[str] = []
    for gp in q.gold_page_ids:
        text = page_texts.get(gp)
        if text is None:
            scanned_pages.append(gp)
            notes.append(f"{gp}: PDF/sayfa bulunamadı")
        elif len(text.strip()) < SCANNED_MIN_CHARS:
            scanned_pages.append(gp)
            notes.append(
                f"{gp}: sayfa metni çok kısa ({len(text.strip())} karakter) "
                "— taranmış sayfa olabilir"
            )

    if scanned_pages:
        return QuestionPrecheck(q.question_id, "MANUEL", notes)

    span_checks: list[SpanCheck] = []
    all_found = True
    for span in q.minimal_evidence_spans:
        matched_pages = [gp for gp in q.gold_page_ids if span_found(span, page_texts.get(gp) or "")]
        found = bool(matched_pages)
        cl = None
        display_page = matched_pages[0] if matched_pages else q.gold_page_ids[0]
        if not found:
            all_found = False
            cl = closest_line(span, page_texts.get(display_page) or "")
        span_checks.append(SpanCheck(span=span, found=found, page_id=display_page, closest=cl))

    if q.slice == "paraphrase":
        overlap = max(
            (content_overlap(q.question, page_texts.get(gp) or "") for gp in q.gold_page_ids),
            default=0.0,
        )
        if overlap > PARAPHRASE_MAX_OVERLAP:
            notes.append(
                f"paraphrase dilimi ama sözlüksel örtüşme %{overlap * 100:.0f} "
                f"(azami %{PARAPHRASE_MAX_OVERLAP * 100:.0f}) — soru sözcükleri gold "
                "sayfada birebir geçiyor, dilim etiketi yanlış olabilir"
            )

    group: Group = "ŞÜPHELİ" if (notes or not all_found) else "TEMİZ"
    return QuestionPrecheck(q.question_id, group, notes, span_checks)


def run_precheck(
    questions: list[BenchQuestion], page_texts: dict[str, str | None], known_page_ids: set[str]
) -> list[QuestionPrecheck]:
    return [precheck_question(q, page_texts, known_page_ids) for q in questions]


def parse_decision(raw: str) -> tuple[str, str] | None:
    """ "h yanlış sayfa" -> ("h", "yanlış sayfa"). Geçersiz harfte None döner."""
    raw = raw.strip()
    if not raw:
        return None
    parts = raw.split(maxsplit=1)
    letter = parts[0].lower()
    note = parts[1].strip() if len(parts) > 1 else ""
    if letter not in {"e", "h", "a", "q"}:
        return None
    return letter, note


def apply_decision(raw_row: dict, letter: str, note: str, by: str) -> dict:
    """Ham dict satırını (pydantic'ten GEÇMEDEN) günceller; bilinmeyen alanlar

    (ör. _hard_negatives) OLDUĞU GİBİ korunur çünkü kopya `raw_row`'un üstüne
    yalnız ilgili anahtarları yazıyoruz, tüm sözlüğü yeniden üretmiyoruz.
    letter: "e"=doğrula, "h"=reddet, "a"=atla (durum değişmez). "q" burada
    KABUL EDİLMEZ — çağıran taraf çıkışı decision döngüsünde ele alır.
    """
    if letter not in {"e", "h", "a"}:
        raise ValueError(f"apply_decision 'q' veya geçersiz harf almamalı: {letter!r}")
    updated = dict(raw_row)
    if letter == "e":
        updated["verification_status"] = "verified"
        updated["verified_by"] = by
        updated["verification_kind"] = "human"
    elif letter == "h":
        updated["verification_status"] = "rejected"
        updated["verified_by"] = by
        updated["verification_kind"] = "human"
    # "a": verification_status/verified_by/verification_kind'a dokunma.
    #
    # `verification_kind` NEDEN burada yazılıyor: bu betiğin her iki inceleme
    # yolu da (terminal --review ve HTML --serve) bir İNSANIN sayfayı okuyup
    # karar vermesidir. Alan yazılmazsa, model çapraz-kontrollü bir satır insan
    # onayından geçtikten sonra bile `model-cross-check` kalır ve `--status`
    # onu insan sayısına EKLEMEZ — yani emek harcanır ama kapı kapanmaz.
    if note:
        updated["verification_note"] = note
    return updated


def select_review_queue(
    raw_rows: list[dict], slices: set[str] | None = None
) -> list[dict]:
    """İnsan doğrulaması BEKLEYEN ham satırlar (giriş sırası korunur).

    Ölçüt tek: `verification_kind != "human"`. `verification_status` bilerek
    BAKILMAZ — v1'in 48 satırının hepsi `verified` ama 45'i model
    çapraz-kontrolü, ve terminal `--review` yalnız `draft` gösterdiği için o 45
    satır hiçbir insan kuyruğuna düşmüyordu. D1'in açtığı boşluk tam olarak bu.

    `slices` verilirse yalnız o dilimler döner (karar dilimlerini önceliklendirmek
    için). Alan yoksa `human` varsayılır — BenchQuestion'ın varsayılanıyla aynı,
    yani alan eklenmeden önce yazılmış satırlar kuyruğa düşmez.
    """
    out = []
    for row in raw_rows:
        if row.get("verification_kind", "human") == "human":
            continue
        if slices is not None and row.get("slice") not in slices:
            continue
        out.append(row)
    return out


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """`rows`'u JSONL olarak `path`'e ATOMİK yazar (aynı dizinde temp + os.replace).

    Kesinti (Ctrl-C, çökme) hiçbir zaman yarım yazılmış/bozuk dosya bırakmaz:
    ya eski içerik olduğu gibi kalır ya da yeni içerik BÜTÜN olarak yerine geçer.
    """
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if content:
        content += "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_raw_rows(path: Path) -> list[dict]:
    """Ham JSON satırlarını sırayla döner (şema dışı alanlar dahil).

    Çağıran taraf ÖNCE `load_bench(path, only_verified=False)` ile şemayı
    doğrulamış olmalı (aksi halde bozuk bir dosyayı sessizce ham okuruz) —
    main() bu sırayı garanti eder.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def gold_image_paths(q: BenchQuestion, images_dir: Path) -> list[Path]:
    """Sorunun altın sayfa görüntü yolları: data/images/<doc_id>/<sayfa:04d>.webp."""
    paths = []
    for gp in q.gold_page_ids:
        doc_id, page_str = gp.split(":", 1)
        paths.append(images_dir / doc_id / f"{int(page_str):04d}.webp")
    return paths


def compute_status(questions: list[BenchQuestion]) -> dict:
    """verification_status/dilim dağılımı + plan hedefinin (>=25+>=5) durumu.

    Hedef İKİ KEZ hesaplanır: `target_met` doğrulanmış tüm satırları (insan +
    model çapraz-kontrolü) sayar, `target_met_human_only` yalnız insan
    doğrulamalı satırları. İkisi ayrı raporlanır ki birleşik sayı yanlışlıkla
    "insan doğrulanmış" diye okunmasın — model çapraz-kontrolü insan onayının
    yerine geçmez (bkz. data/bench/canary_v1.README.md).
    """
    by_status = Counter(q.verification_status for q in questions)
    verified = [q for q in questions if q.verification_status == "verified"]
    verified_answerable = sum(1 for q in verified if q.answerable)
    verified_unanswerable = len(verified) - verified_answerable

    human = [q for q in verified if q.verification_kind == "human"]
    human_answerable = sum(1 for q in human if q.answerable)
    human_unanswerable = len(human) - human_answerable

    return {
        "by_status": dict(by_status),
        "by_slice_total": bench_stats(questions),
        "by_slice_verified": bench_stats(verified),
        "verified_total": len(verified),
        "verified_answerable": verified_answerable,
        "verified_unanswerable": verified_unanswerable,
        "verified_by_kind": dict(Counter(q.verification_kind for q in verified)),
        "human_total": len(human),
        "human_answerable": human_answerable,
        "human_unanswerable": human_unanswerable,
        "target_met": verified_answerable >= 25 and verified_unanswerable >= 5,
        "target_met_human_only": human_answerable >= 25 and human_unanswerable >= 5,
    }


# --------------------------------------------------------------------------
# I/O'ya dokunan yardımcılar (PDF/indeks okur — testlerde KULLANILMAZ)
# --------------------------------------------------------------------------


def load_known_page_ids(index_dir: Path) -> set[str]:
    p = index_dir / "page_ids.json"
    if not p.exists():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")))


def extract_page_texts(page_ids: set[str], pdf_dir: Path) -> dict[str, str | None]:
    """Gerekli "dok:sayfa" kimlikleri için PDF'ten düz metin çıkarır.

    PDF ya da sayfa bulunamazsa None döner (precheck bunu MANUEL'e düşürür).
    Sayfa numarası ":"den sonraki kısımdır ve 1-tabanlıdır (bkz.
    corpus/render.py: images/<doc_id>/<i:04d>.webp, i 1'den başlar).
    """
    import pymupdf as fitz  # lazy: --status'ün PDF'e hiç dokunmaması için

    by_doc: dict[str, list[tuple[str, int]]] = {}
    for pid in page_ids:
        doc_id, page_str = pid.split(":", 1)
        by_doc.setdefault(doc_id, []).append((pid, int(page_str)))

    result: dict[str, str | None] = {}
    for doc_id, items in by_doc.items():
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            for pid, _ in items:
                result[pid] = None
            continue
        try:
            with fitz.open(pdf_path) as doc:
                for pid, page_num in items:
                    page_idx = page_num - 1
                    if page_idx < 0 or page_idx >= doc.page_count:
                        result[pid] = None
                        continue
                    result[pid] = doc[page_idx].get_text("text")
        except (fitz.FileDataError, RuntimeError):
            for pid, _ in items:
                result[pid] = None
    return result


def build_prechecks(
    questions: list[BenchQuestion], pdf_dir: Path, index_dir: Path
) -> list[QuestionPrecheck]:
    known_page_ids = load_known_page_ids(index_dir)
    needed_pages = {gp for q in questions if q.answerable for gp in q.gold_page_ids}
    page_texts = extract_page_texts(needed_pages, pdf_dir)
    return run_precheck(questions, page_texts, known_page_ids)


def _open_image(path: Path, no_open: bool) -> None:
    if not path.exists():
        print(f"  [uyarı] görüntü yok: {path}")
        return
    if no_open:
        print(f"  görüntü: {path}")
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)
    else:
        print(f"  görüntü: {path}")


# --------------------------------------------------------------------------
# Rapor biçimlendirme
# --------------------------------------------------------------------------


def format_table(prechecks: list[QuestionPrecheck], by_id: dict[str, BenchQuestion]) -> str:
    header = f"{'id':<6} {'dilim':<26} {'zorluk':<7} {'grup':<9} not"
    lines = [header, "-" * len(header)]
    for p in prechecks:
        q = by_id[p.question_id]
        note = p.notes[0] if p.notes else ""
        lines.append(f"{p.question_id:<6} {q.slice:<26} {q.difficulty:<7} {p.group:<9} {note}")
    return "\n".join(lines)


def format_report_md(
    prechecks: list[QuestionPrecheck], by_id: dict[str, BenchQuestion], generated_at: str
) -> str:
    counts = Counter(p.group for p in prechecks)
    lines = [
        "# Canary Ön-Kontrol Raporu",
        "",
        f"Oluşturulma: {generated_at}  ·  toplam soru: {len(prechecks)}",
        "",
        "Normalizasyon: karşılaştırmadan önce boşluk/satır sonları tek boşluğa "
        "indirgenir, ardından Türkçe'ye duyarlı küçültme uygulanır (İ->i, I->ı, "
        "sonra str.casefold()); hem sayfa metni hem alıntı AYNI fonksiyondan geçer.",
        "",
        "## Özet",
        "",
    ]
    for g in GROUPS:
        lines.append(f"- **{g}**: {counts.get(g, 0)}")
    lines.append("")

    for g in GROUPS:
        group_items = [p for p in prechecks if p.group == g]
        lines.append(f"## {g} ({len(group_items)})")
        lines.append("")
        if g == "TEMİZ":
            for p in group_items:
                q = by_id[p.question_id]
                extra = f" — {p.notes[0]}" if p.notes else ""
                lines.append(f"- `{p.question_id}` ({q.slice}, {q.difficulty}){extra}")
        else:
            for p in group_items:
                q = by_id[p.question_id]
                lines.append(f"### `{p.question_id}` — {q.question}")
                lines.append(f"- dilim: {q.slice}, zorluk: {q.difficulty}")
                lines.append(f"- gold_page_ids: {q.gold_page_ids}")
                for note in p.notes:
                    lines.append(f"- not: {note}")
                for sc in p.span_checks:
                    if sc.found:
                        continue
                    lines.append(f'- eşleşmeyen alıntı ({sc.page_id}): "{sc.span}"')
                    if sc.closest:
                        lines.append(f'  en yakın satır: "{sc.closest}"')
                    else:
                        lines.append("  en yakın satır: (sayfa metni yok)")
                lines.append("")
        lines.append("")
    return "\n".join(lines)


def print_status(status: dict) -> None:
    print("== Doğrulama Durumu ==\n")
    print("verification_status dağılımı:")
    for key in ("draft", "verified", "rejected"):
        print(f"  {key:<10} {status['by_status'].get(key, 0)}")

    print("\ndilim (slice) dağılımı (doğrulanmış/toplam):")
    for s in sorted(status["by_slice_total"]):
        total = status["by_slice_total"][s]
        verified = status["by_slice_verified"].get(s, 0)
        if total:
            print(f"  {s:<28} {verified}/{total}")

    print("\ndoğrulama TÜRÜ dağılımı (yalnız verified satırlar):")
    kinds = status["verified_by_kind"]
    print(f"  {'human':<18} {kinds.get('human', 0):<4} (proje sahibi sayfayı kendisi okudu)")
    print(
        f"  {'model-cross-check':<18} {kinds.get('model-cross-check', 0):<4} "
        "(bağımsız model turu; insan onayı YERİNE GEÇMEZ)"
    )

    print(
        f"\ndoğrulanmış cevaplanabilir : {status['verified_answerable']} "
        f"(hedef >= 25) — yalnız insan: {status['human_answerable']}"
    )
    print(
        f"doğrulanmış cevaplanamaz   : {status['verified_unanswerable']} "
        f"(hedef >= 5) — yalnız insan: {status['human_unanswerable']}"
    )
    print(
        f"doğrulanmış toplam         : {status['verified_total']} "
        f"— yalnız insan: {status['human_total']}"
    )
    combined = "EVET" if status["target_met"] else "HAYIR"
    human_only = "EVET" if status["target_met_human_only"] else "HAYIR"
    print(
        "\nHEDEF (>=25 cevaplanabilir + >=5 cevaplanamaz): "
        f"{combined} (toplam) / {human_only} (yalnız insan: {status['human_total']})"
    )
    if status["target_met"] and not status["target_met_human_only"]:
        print(
            "  UYARI: 'toplam' rakamı model çapraz-kontrolünü içerir ve İNSAN DOĞRULAMASI\n"
            "  DEĞİLDİR. Bu set insan-doğrulanmış olarak alıntılanamaz; gerekçe ve bilinen\n"
            "  kısıtlar için bkz. data/bench/canary_v1.README.md"
        )


# --------------------------------------------------------------------------
# Komutlar
# --------------------------------------------------------------------------


def cmd_report(
    questions: list[BenchQuestion], pdf_dir: Path, index_dir: Path, report_out: Path
) -> int:
    prechecks = build_prechecks(questions, pdf_dir, index_dir)
    by_id = {q.question_id: q for q in questions}
    counts = Counter(p.group for p in prechecks)

    print(format_table(prechecks, by_id))
    print()
    for g in GROUPS:
        print(f"{g}: {counts.get(g, 0)}")

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        format_report_md(prechecks, by_id, datetime.now(UTC).isoformat()), encoding="utf-8"
    )
    print(f"\nrapor yazıldı -> {report_out}")
    return 0


def cmd_status(questions: list[BenchQuestion]) -> int:
    print_status(compute_status(questions))
    return 0


def cmd_review(
    raw_rows: list[dict],
    questions: list[BenchQuestion],
    pdf_dir: Path,
    index_dir: Path,
    images_dir: Path,
    bench_path: Path,
    by: str,
    only_group: Group | None,
    no_open: bool,
) -> int:
    prechecks = {p.question_id: p for p in build_prechecks(questions, pdf_dir, index_dir)}
    id_to_index = {q.question_id: i for i, q in enumerate(questions)}

    queue = [q for q in questions if q.verification_status == "draft"]
    if only_group:
        queue = [q for q in queue if prechecks[q.question_id].group == only_group]

    if not queue:
        print(f"gözden geçirilecek taslak soru yok (filtre: {only_group or 'yok'})")
        return 0

    total = len(queue)
    print(f"{total} taslak soru gözden geçirilecek (doğrulayıcı: {by})")
    for pos, q in enumerate(queue, start=1):
        pc = prechecks[q.question_id]
        print("\n" + "=" * 72)
        print(
            f"[{pos}/{total}] {q.question_id}  dilim={q.slice}  "
            f"zorluk={q.difficulty}  ön-kontrol={pc.group}"
        )
        print(f"soru           : {q.question}")
        print(f"referans cevap : {q.reference_answer}")
        print(f"kanıt alıntıları: {q.minimal_evidence_spans}")
        if pc.notes:
            print("ön-kontrol notları:")
            for n in pc.notes:
                print(f"  - {n}")
        for path in gold_image_paths(q, images_dir):
            _open_image(path, no_open)

        while True:
            raw = input("[e]doğrula / [h]reddet / [a]atla / [q]kaydet-çık (+not yazabilirsin): ")
            decision = parse_decision(raw)
            if decision is None:
                print("geçersiz giriş, tekrar dene (e/h/a/q)")
                continue
            break
        letter, note = decision

        if letter == "q":
            print("kaydedildi, çıkılıyor.")
            break

        idx = id_to_index[q.question_id]
        raw_rows[idx] = apply_decision(raw_rows[idx], letter, note, by)
        write_jsonl_atomic(bench_path, raw_rows)
        label = {"e": "doğrulandı", "h": "reddedildi", "a": "atlandı (taslak kalıyor)"}[letter]
        print(f"{q.question_id} {label}. ilerleme {pos}/{total}, kalan {total - pos}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", action="store_true", help="otomatik ön-kontrol (salt-okunur)")
    mode.add_argument("--review", action="store_true", help="taslak soruları tek tek incele")
    mode.add_argument("--status", action="store_true", help="doğrulama durumu + hedef kontrolü")

    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help="bench JSONL dosyası")
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="kaynak PDF dizini")
    ap.add_argument(
        "--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="sayfa görüntü dizini"
    )
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="üretim indeksi dizini (varsayılan: Settings().index_dir)",
    )
    ap.add_argument(
        "--report-out", type=Path, default=DEFAULT_REPORT_OUT, help="--report çıktı yolu"
    )
    ap.add_argument(
        "--by", default="", help="--review için zorunlu doğrulayıcı adı (ör. --by baran)"
    )
    ap.add_argument(
        "--only",
        choices=GROUPS,
        default=None,
        help="--review'ı yalnız bu ön-kontrol grubuyla sınırla",
    )
    ap.add_argument(
        "--no-open", action="store_true", help="--review sırasında görüntüyü otomatik açma"
    )
    args = ap.parse_args()

    if args.review and not args.by:
        ap.error("--review için --by zorunludur (ör. --by baran)")

    index_dir = args.index_dir if args.index_dir is not None else Settings().index_dir
    if not index_dir.is_absolute():
        index_dir = REPO_ROOT / index_dir

    try:
        # load_bench şemayı (BenchQuestion) tam doğrular; JSONL bozuksa ya da
        # bir satır şemaya uymuyorsa burada ValueError ile DURUR — hiçbir mod
        # bozuk bir dosya üzerinde çalışmaz.
        questions = load_bench(args.bench, only_verified=False)
    except ValueError as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1

    if args.report:
        return cmd_report(questions, args.pdf_dir, index_dir, args.report_out)
    if args.status:
        return cmd_status(questions)

    # --review: yalnız bu moda özel ham dict'leri oku (round-trip _hard_negatives
    # gibi şema dışı alanları korumak için pydantic modelinden DEĞİL, ham
    # dict'ten yazacağız). load_bench zaten doğruladı, bu yüzden burada güvenle
    # ham okunabilir.
    raw_rows = load_raw_rows(args.bench)
    return cmd_review(
        raw_rows,
        questions,
        args.pdf_dir,
        index_dir,
        args.images_dir,
        args.bench,
        args.by,
        args.only,
        args.no_open,
    )


if __name__ == "__main__":
    raise SystemExit(main())
