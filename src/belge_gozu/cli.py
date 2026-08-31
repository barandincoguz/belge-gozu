import json
import math
import shutil
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from PIL import Image
from pydantic import ValidationError

from belge_gozu.bench.dataset import BenchSelection, VerificationLevel
from belge_gozu.bench.oracle import FloatIndex, native_float_scores, rank_of
from belge_gozu.config import Settings
from belge_gozu.corpus.download import download_all
from belge_gozu.corpus.manifest import build_http_client, load_manifest, probe
from belge_gozu.corpus.render import render_all
from belge_gozu.index.encode import FakeEncoder
from belge_gozu.index.manifest import (
    CPE_0_3_18,
    DOC_PROMPTS,
    QUERY_FORMATS,
    DocPromptChoice,
    IndexManifest,
    Quantization,
    QueryFormatChoice,
    RenderConfig,
    corpus_checksum,
    read_manifest,
    write_manifest,
)
from belge_gozu.index.store import PackedIndex
from belge_gozu.provenance import git_commit

app = typer.Typer(help="Belge-Gözü: Türkçe mevzuat için görsel belge RAG")
corpus_app = typer.Typer()
index_app = typer.Typer()
metrics_app = typer.Typer()
bench_app = typer.Typer()
calibrate_app = typer.Typer(help="P2 güven kalibratörü: özellik çıkarımı, fit, değerlendirme")
verify_app = typer.Typer(help="P2 kanıt doğrulayıcı: iki kapılı ask hattının koşum harness'ı")
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")
app.add_typer(metrics_app, name="metrics")
app.add_typer(bench_app, name="bench")
app.add_typer(calibrate_app, name="calibrate")
app.add_typer(verify_app, name="verify")

DEFAULT_MANIFEST = Path("data/manifest/v0_manifest.csv")


class Pipeline(StrEnum):
    hybrid = "hybrid"
    exhaustive = "exhaustive"
    two_stage = "two-stage"


class Precision(StrEnum):
    packed = "packed"
    f16 = "f16"


# `index build --precision` -> manifest'e yazılacak quantization.
# `Quantization` T14'te belge_gozu.index.manifest'e taşındı (yükleyici de
# aynı sözlükten okur); burada yalnız import edilir.
PRECISION_QUANTIZATION: dict[Precision, Quantization] = {
    Precision.packed: Quantization.sign_1bit,
    Precision.f16: Quantization.float16,
}


# QueryFormatChoice/DocPromptChoice ve QUERY_FORMATS/DOC_PROMPTS sözlükleri
# belge_gozu.index.manifest'te tanımlı (T11/Step 6): serve config'i (Settings)
# ile CLI aynı tek sözlükten okur, iki kopya literal sürüklenmez.
#
# Final review CRITICAL-1: `index build`ın --query-format/--doc-prompt
# VARSAYILANLARI da config'ten (Settings) gelir. Daha önce burada sabit literal
# duruyordu (cpe-0.3.18 / processor-default) ve Settings train-compat'e geçince
# sessizce sürüklendi: `out_dir = out or s.index_dir` olduğu için belgelenmiş
# `uv run belge-gozu index build` çağrısı ÜRETİM indeksini (data/index-traincompat-1bit)
# A/B'yi KAYBEDEN formatla ezip serve'ü fail-fast'e düşürüyordu. Typer varsayılanları
# dekorasyon (import) anında bağlandığı için tek Settings örneği burada okunur;
# bayraklar hâlâ elle geçersiz kılınabilir.
#
# Bu satır IMPORT ANINDA koşar, yani `belge-gozu --help` bile ortam
# değişkenlerini doğrular. Bozuk bir BG_* değeri (audit C9) burada ham bir
# pydantic traceback'i olarak patlıyordu: kullanıcı yardım metni yerine 30
# satırlık bir yığın izi görüyordu. Hata GERÇEK ve ölümcül (yanlış config'le
# çalışmak sessiz sapma demektir) — ama okunur olmalı.
try:
    _CLI_DEFAULTS = Settings()
except ValidationError as e:
    typer.secho(
        "Yapılandırma hatası: ortam değişkenleri (BG_*) ya da .env dosyası geçersiz.",
        err=True,
        fg=typer.colors.RED,
    )
    for err in e.errors():
        field = ".".join(str(p) for p in err["loc"]) or "(bilinmeyen alan)"
        typer.secho(f"  - {field}: {err['msg']}", err=True, fg=typer.colors.RED)
    typer.secho("Düzeltip tekrar deneyin (örn. `unset BG_QUERY_FORMAT_ID`).", err=True, dim=True)
    raise SystemExit(2) from None

DEFAULT_QUERY_FORMAT = QueryFormatChoice(_CLI_DEFAULTS.query_format_id)
DEFAULT_DOC_PROMPT = DocPromptChoice(_CLI_DEFAULTS.doc_prompt_id)
# Aynı gerekçe `bench run --pipeline` için de geçerli (P1): sabit bir literal
# olsaydı üretim varsayılanı hibrite geçtiğinde bench sessizce ESKİ yolu
# ölçmeye devam ederdi — yani "bench" ile "servis edilen" iki farklı sistem
# olurdu. Bayrak hâlâ elle geçersiz kılınabilir (ablasyon koşumları).
DEFAULT_PIPELINE = Pipeline(_CLI_DEFAULTS.retrieval_pipeline)


def _settings() -> Settings:
    return Settings()


def _pkg_version(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "unknown"


def _engine_versions() -> dict[str, str]:
    return {
        "colpali-engine": _pkg_version("colpali-engine"),
        "transformers": _pkg_version("transformers"),
        "torch": _pkg_version("torch"),
    }


def _manifest_path(s: Settings, manifest: Path | None) -> Path:
    return manifest or (s.data_dir / "manifest" / "v0_manifest.csv")


@corpus_app.command("download")
def corpus_download(
    manifest: Path | None = typer.Option(None),  # noqa: B008
) -> None:
    s = _settings()
    rows = load_manifest(_manifest_path(s, manifest))
    with build_http_client() as client:
        report = download_all(rows, s.data_dir, client, delay_s=s.request_delay_s)
    typer.echo(f"ok={len(report.ok)} skipped={len(report.skipped)} failed={report.failed}")


@corpus_app.command("probe")
def corpus_probe(manifest: Path | None = typer.Option(None)) -> None:  # noqa: B008
    s = _settings()
    rows = load_manifest(_manifest_path(s, manifest))
    with build_http_client() as client:
        for doc_id, status in probe(rows, client):
            typer.echo(f"{doc_id}\t{status}")


@corpus_app.command("render")
def corpus_render(dpi: int = typer.Option(150)) -> None:  # noqa: B008
    s = _settings()
    rows = load_manifest(_manifest_path(s, None))
    df = render_all(rows, s.data_dir, dpi=dpi)
    typer.echo(f"{len(df)} sayfa render edildi")


@index_app.command("build")
def index_build(
    fake: bool = typer.Option(False, "--fake"),  # noqa: B008
    precision: Precision = typer.Option(Precision.packed, "--precision"),  # noqa: B008
    query_format: QueryFormatChoice = typer.Option(  # noqa: B008
        DEFAULT_QUERY_FORMAT, "--query-format"
    ),
    doc_prompt: DocPromptChoice = typer.Option(DEFAULT_DOC_PROMPT, "--doc-prompt"),  # noqa: B008
    out: Path | None = typer.Option(None, "--out"),  # noqa: B008
) -> None:
    s = _settings()
    if precision == Precision.f16 and out is None:
        raise typer.BadParameter("--precision f16 için --out zorunlu")
    # CRITICAL-1 emniyeti: --out verilmediğinde hedef ÜRETİM indeksidir
    # (s.index_dir). Serve'ün beklediği formattan sapan bir build o dizini
    # kullanılamaz hale getireceği için, sapma varsa açık bir --out istenir.
    if out is None and (
        query_format.value != s.query_format_id or doc_prompt.value != s.doc_prompt_id
    ):
        raise typer.BadParameter(
            "--query-format/--doc-prompt serve config'inden sapıyor "
            f"(build={query_format.value}/{doc_prompt.value} "
            f"config={s.query_format_id}/{s.doc_prompt_id}); üretim indeksini "
            f"({s.index_dir}) ezmemek için --out ile ayrı bir dizin verin"
        )
    quantization = PRECISION_QUANTIZATION[precision]
    # T14 emniyeti (aynı gerekçenin KUANTİZASYON ekseni): üretim indeksi artık
    # int8 ama `index build` yalnız packed/f16 üretir. --out'suz bir build
    # sessizce int8 indeksin üstüne 1-bit yazar, manifest'i de "sign-1bit"e
    # çevirdiği için yükleyici hiçbir şey fark etmeden onu servis eder —
    # ölçümde kaybeden temsile sessiz geri dönüş. Fail-fast:
    #
    # (review M3: --out'suz f16 yukarıda zaten reddedildiği için burada
    # `quantization` pratikte hep sign-1bit'tir. Karşılaştırma yine de
    # değişken üzerinden yazılıyor: yakalaması gereken durum "hedefteki
    # temsil bu build'in yazacağından farklı" ve o kural f16 kısıtı
    # gevşetilirse de doğru kalmalı.)
    if out is None:
        existing = read_manifest(s.index_dir)
        if existing is not None and existing.quantization != quantization.value:
            raise typer.BadParameter(
                f"{s.index_dir} indeksi quantization={existing.quantization} "
                f"taşıyor, bu build ise {quantization.value} yazacak — üretim "
                "indeksi sessizce başka bir temsile döndürülemez. Ayrı bir dizin "
                "için --out verin; int8/1-bit türetmek için f16 master'dan "
                "`belge-gozu index derive --from <f16> --quant <...> --out <...>` "
                "kullanın."
            )
    out_dir = out or s.index_dir
    qf = QUERY_FORMATS[query_format]
    doc_prompt_override = DOC_PROMPTS[doc_prompt]
    meta = pd.read_parquet(s.data_dir / "meta.parquet")
    if fake:
        encoder = FakeEncoder()
    else:
        from belge_gozu.index.encode import ColSmolEncoder

        encoder = ColSmolEncoder(
            s.retriever_model,
            s.device,
            query_format=qf,
            visual_prompt_override=doc_prompt_override,
        )
    embs, ids = [], []
    # batch_size=1 KASITLI: MPS'te batch içinde (padding'li) vs. tek başına encode
    # edilen sayfa bit-birebir aynı çıkmıyor (ölçüm 2026-08-26, sign uyuşması
    # 0.9990/0.9989 — bkz. tests/index/test_encode_mask.py::
    # test_batch_vs_single_sign_determinism). "Optimize" edip büyütmeyin.
    batch_size = 1
    total = len(meta)
    for start in range(0, total, batch_size):
        chunk = meta.iloc[start : start + batch_size]
        images = []
        for _, row in chunk.iterrows():
            with Image.open(s.data_dir / row["image_path"]) as raw:
                images.append(raw.convert("RGB"))
        embs.extend(encoder.encode_pages(images))
        ids.extend(chunk["page_id"])
        for img in images:
            img.close()
        chunk_no = start // batch_size + 1
        done = min(start + batch_size, total)
        if chunk_no % 10 == 0 or done == total:
            print(f"{done}/{total} sayfa", flush=True)

    # R3 (manifest ordering): dosyalar tamamen yazılana kadar manifest.json
    # yazılmaz — indeks kaydet -> meta.parquet kopyala -> checksum hesapla ->
    # manifest oluştur -> yaz. PackedIndex.build/FloatIndex.build'e manifest
    # verilmez (write_manifest ile ayrıca yazılır).
    if precision == Precision.f16:
        index = FloatIndex.build(ids, embs)
    else:
        index = PackedIndex.build(ids, embs)
    index.save(out_dir)
    shutil.copy(s.data_dir / "meta.parquet", out_dir / "meta.parquet")

    manifest = IndexManifest(
        model_name=s.retriever_model,
        model_revision=getattr(encoder, "model_revision", "unknown"),
        engine_versions=_engine_versions(),
        query_format=qf,
        doc_prompt_sha256=getattr(encoder, "doc_prompt_sha256", "unknown"),
        quantization=quantization.value,
        mask_policy="drop-padding",
        render=RenderConfig(),
        corpus_checksum=corpus_checksum(out_dir),
        n_pages=len(index.page_ids),
        n_tokens=int(index.offsets[-1]),
        built_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit(),
    )
    write_manifest(out_dir, manifest)
    typer.echo(f"{len(ids)} sayfa indekslendi -> {out_dir}")


@index_app.command("build-text")
def index_build_text(
    allow_missing: bool = typer.Option(False, "--allow-missing"),  # noqa: B008
) -> None:
    """PDF metin katmanını indeksin sayfa sırasıyla hizalı parquet'e yazar (hibrit kanal).

    Model GEREKTİRMEZ (saniyeler sürer): `data/pdf` altındaki PDF'lerin metin
    katmanı okunur ve `<index_dir>/page_texts.parquet` olarak yazılır. Artefakt
    indeks dizinine yazılır çünkü GÖRSEL İNDEKSİN SAYFA SIRASINA bağlıdır;
    `corpus_checksum` yalnız page_ids.json + meta.parquet'i okuduğu için
    manifest'i geçersiz KILMAZ (serve tarafı hizalamayı ayrıca doğrular).

    KISMİ KORPUS REDDEDİLİR (review M3): indekste geçen bir dokümanın PDF'i
    `data/pdf` altında yoksa komut listeyle birlikte durur. Aksi halde yarım
    kalmış bir `corpus download` (ağ hatası / Ctrl-C) satır-hizalı ve bu yüzden
    serve'ün hizalama kontrolünden GEÇEN, ama sayfalarının bir kısmı boş olan
    bir artefakt üretir: hibrit "çalışır", korpusun bir bölümü BM25 tarafından
    hiç görülmez ve bozulma tamamen sessizdir. Bilinçli kısmi koşumlar için
    `--allow-missing`.
    """
    from belge_gozu.corpus.text import extract_page_texts

    s = _settings()
    ids_path = s.index_dir / "page_ids.json"
    if not ids_path.exists():
        raise typer.BadParameter(
            f"indeks dizininde page_ids.json yok: {s.index_dir} — önce indeksi "
            "kurun/indirin (`belge-gozu index build` ya da `index pull`)"
        )
    if read_manifest(s.index_dir) is None:
        raise typer.BadParameter(
            f"{s.index_dir} indeksinde manifest.json yok; metin kanalı hangi korpusa "
            "hizalandığı bilinmeyen bir indeks için üretilmemeli"
        )
    pdf_dir = s.data_dir / "pdf"
    if not pdf_dir.is_dir():
        raise typer.BadParameter(
            f"PDF dizini yok: {pdf_dir} — metin katmanı kaynak PDF'lerden okunur "
            "(`belge-gozu corpus download`)"
        )
    page_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    doc_ids = sorted({pid.partition(":")[0] for pid in page_ids})
    missing = [d for d in doc_ids if not (pdf_dir / f"{d}.pdf").is_file()]
    if missing and not allow_missing:
        raise typer.BadParameter(
            f"{len(missing)}/{len(doc_ids)} dokümanın PDF'i {pdf_dir} altında yok: "
            f"{', '.join(missing)}. Bu dokümanların TÜM sayfaları boş metinle yazılır "
            "ve BM25 onları hiç göremez; kontrol satır sayısına baktığı için serve "
            "bunu fark etmez. Çözüm: `belge-gozu corpus download` (yarım kalmış indirme "
            "sürdürülebilir). Bilinçli kısmi koşum için: --allow-missing"
        )
    df = extract_page_texts(pdf_dir, page_ids)
    out = s.index_dir / "page_texts.parquet"
    df.to_parquet(out, index=False)

    blank_by_doc: Counter[str] = Counter()
    pages_by_doc: Counter[str] = Counter()
    for pid, text in zip(df["page_id"], df["text"], strict=True):
        doc = pid.partition(":")[0]
        pages_by_doc[doc] += 1
        if not text.strip():
            blank_by_doc[doc] += 1
    typer.echo(f"{len(df)} sayfa, {sum(blank_by_doc.values())} metin katmanı boş -> {out}")
    # Doküman kırılımı (review M3): 1/4222 (sağlıklı — tek taranmış RG sayfası) ile
    # 2500/4222 (yarım korpus) yalnız toplama bakınca ayırt edilemiyordu.
    for doc, n in sorted(blank_by_doc.items(), key=lambda kv: (-kv[1], kv[0])):
        typer.echo(f"  boş: {doc} {n}/{pages_by_doc[doc]} sayfa")
    if missing:
        typer.echo(f"  UYARI: --allow-missing ile PDF'i olmayan {len(missing)} doküman atlandı")


@index_app.command("derive")
def index_derive(
    from_dir: Path = typer.Option(..., "--from"),  # noqa: B008
    quant: Quantization = typer.Option(..., "--quant"),  # noqa: B008
    out: Path = typer.Option(..., "--out"),  # noqa: B008
) -> None:
    """f16 master'dan (T9 FloatIndex) sign-1bit veya int8 türetir (C1/C2 ablasyonu)."""
    from belge_gozu.index.quantize import Int8Index, derive_packed

    # `Quantization` T14'te üç üyeye çıktı (float16 dahil) — bu komut yalnız
    # ikisini türetebilir. Açıkça reddedilmezse aşağıdaki dallanma
    # `--quant float16` için sessizce int8 üretir ve manifest'e "float16"
    # yazardı: diskteki veriyle etiketi çelişen bir indeks.
    if quant == Quantization.float16:
        raise typer.BadParameter(
            "--quant float16 anlamsız: --from zaten float16 master. "
            "Türetilebilir temsiller: sign-1bit, int8"
        )
    if not (from_dir / "embs.npy").exists():
        raise typer.BadParameter(
            f"--from bir float16 (FloatIndex) dizini olmalı: {from_dir / 'embs.npy'} bulunamadı"
        )
    if not (from_dir / "meta.parquet").exists():
        raise typer.BadParameter(f"--from dizininde meta.parquet yok: {from_dir}")
    if out.resolve() == from_dir.resolve():
        raise typer.BadParameter("--out --from ile aynı olamaz (f16 master'ın üstüne yazılır)")
    findex = FloatIndex.load(from_dir, mmap=False)
    if findex.manifest is None:
        raise typer.BadParameter(f"--from indeksinde manifest.json yok: {from_dir}")
    if findex.manifest.quantization != Quantization.float16.value:
        raise typer.BadParameter(
            f"--from float16 indeks olmalı, bulunan quantization={findex.manifest.quantization}"
        )

    derived: PackedIndex | Int8Index
    derived = derive_packed(findex) if quant == Quantization.sign_1bit else Int8Index.derive(findex)

    # R3 (manifest ordering): dosyalar tamamen yazılana kadar manifest.json
    # yazılmaz. derive_packed/Int8Index.derive taşınan manifest'i quantization
    # dışında değiştirmeden döner; save() içindeki erken (bayat checksum'lu)
    # yazımı önlemek için burada geçici olarak None'lanır, tek doğru yazım en
    # sonda checksum yeniden hesaplanarak yapılır.
    source_manifest = derived.manifest
    if source_manifest is None:
        raise RuntimeError("beklenmedik: manifest türetilmiş indekse taşınmadı")
    derived.manifest = None
    derived.save(out)
    shutil.copy(from_dir / "meta.parquet", out / "meta.parquet")

    manifest = source_manifest.model_copy(
        update={
            "quantization": quant.value,
            "corpus_checksum": corpus_checksum(out),
            "n_pages": len(derived.page_ids),
            "n_tokens": int(derived.offsets[-1]),
            "built_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
        }
    )
    write_manifest(out, manifest)
    typer.echo(f"{len(derived.page_ids)} sayfa, {int(derived.offsets[-1])} token -> {out}")


@index_app.command("write-manifest")
def index_write_manifest(
    legacy: bool = typer.Option(False, "--legacy"),  # noqa: B008
) -> None:
    if not legacy:
        raise typer.BadParameter("şu an yalnız --legacy destekleniyor")
    s = _settings()
    index = PackedIndex.load(s.index_dir)
    n_pages = len(index.page_ids)
    n_tokens = int(index.offsets[-1])

    manifest = IndexManifest(
        model_name=s.retriever_model,
        model_revision="unknown",
        engine_versions=_engine_versions(),
        query_format=CPE_0_3_18,
        doc_prompt_sha256="unknown",
        quantization="sign-1bit",
        mask_policy="none",
        render=RenderConfig(),
        corpus_checksum=corpus_checksum(s.index_dir),
        n_pages=n_pages,
        n_tokens=n_tokens,
        built_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit(),
    )
    write_manifest(s.index_dir, manifest)
    typer.echo(f"manifest yazıldı -> {s.index_dir / 'manifest.json'}")


@index_app.command("push")
def index_push() -> None:
    from belge_gozu.index.hub import push_index

    s = _settings()
    images_dir = s.data_dir / "images"
    push_index(
        s.index_dir, s.hf_dataset_repo, images_dir=images_dir if images_dir.exists() else None
    )
    typer.echo(f"indeks {s.hf_dataset_repo} reposuna gönderildi")


@index_app.command("pull")
def index_pull() -> None:
    from belge_gozu.index.hub import pull_index

    s = _settings()
    pull_index(s.hf_dataset_repo, s.index_dir, data_dir=s.data_dir)
    typer.echo(f"indeks {s.hf_dataset_repo} reposundan indirildi")


@metrics_app.command("export")
def metrics_export(out: Path = typer.Option(Path("data/exports/events.parquet"))) -> None:  # noqa: B008
    from belge_gozu.telemetry.export import export_events

    s = _settings()
    try:
        n = export_events(s.data_dir / "requests.sqlite", out)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        typer.echo("0 olay — tablo yok")
        return
    typer.echo(f"{n} olay -> {out}")


@metrics_app.command("summary")
def metrics_summary() -> None:
    s = _settings()
    db: sqlite3.Connection | None = None
    try:
        db = sqlite3.connect(s.data_dir / "requests.sqlite")
        # `rejected` satırları gecikme toplamlarının DIŞINDA — `/stats` ile aynı
        # gerekçe ve aynı filtre (review M3); `n` sayımı tüm satırları kapsar.
        n, avg = db.execute(
            "SELECT (SELECT COUNT(*) FROM events), "
            "COALESCE((SELECT AVG(total_ms) FROM events WHERE status <> 'rejected'), 0)"
        ).fetchone()
        ab = db.execute(
            "SELECT COALESCE(AVG(abstained),0) FROM events "
            "WHERE endpoint='/ask' AND status <> 'degraded'"
        ).fetchone()[0]
        tok = db.execute(
            "SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), "
            "COALESCE(SUM(est_cost_usd),0) FROM events"
        ).fetchone()
        vals = sorted(
            r[0] for r in db.execute("SELECT total_ms FROM events WHERE status <> 'rejected'")
        )
    except sqlite3.OperationalError:
        typer.echo("henüz olay kaydı yok")
        return
    finally:
        if db is not None:
            db.close()
    p95 = vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)] if vals else 0.0
    typer.echo(f"istek={n} ort={avg:.0f}ms p95={p95:.0f}ms abstain={ab:.1%}")
    typer.echo(f"token in/out={tok[0]}/{tok[1]} maliyet≈${tok[2]:.4f}")


def _load_bench_mode(
    bench: Path,
    only_verified: bool,
    min_verification: VerificationLevel | str | None,
) -> BenchSelection:
    """Bench JSONL'i doğrulama düzeyine göre yükler ve seçimi açıklar.

    `select_bench` saf JSONL okur — model/indekse dokunmaz, bu yüzden birim
    testte gerçek encoder olmadan doğrudan test edilebilir. R15: canary_v1
    insan doğrulaması tamamlanana kadar taslak dahil TÜMÜ (--all) varsayılan
    olmalı, aksi halde `bench run`/`bench oracle` hiç koşamaz."""
    from belge_gozu.bench.dataset import select_bench

    selection = select_bench(
        bench,
        only_verified=only_verified,
        min_verification=min_verification,
    )
    status = "verified" if only_verified or selection.min_verification else "all"
    minimum = selection.min_verification.value if selection.min_verification else "none"
    typer.echo(
        "bench seçimi: "
        f"toplam={selection.total} seçilen={selection.selected} "
        f"elenen={selection.filtered_out}; verification_status={status}; min={minimum}"
    )
    return selection


@bench_app.command("run")
def bench_run(
    bench: Path = typer.Option(Path("data/bench/canary_v1.jsonl")),  # noqa: B008
    pipeline: Pipeline = typer.Option(DEFAULT_PIPELINE, "--pipeline"),  # noqa: B008
    only_verified: bool = typer.Option(False, "--only-verified/--all"),  # noqa: B008
    min_verification: VerificationLevel | None = typer.Option(  # noqa: B008
        None, "--min-verification"
    ),
    out: Path | None = typer.Option(None, "--out"),  # noqa: B008
) -> None:
    from belge_gozu.bench.harness import (
        ExhaustiveDiagnosticAdapter,
        HybridDiagnosticAdapter,
        TwoStageDiagnosticAdapter,
        run_retrieval_eval,
    )
    from belge_gozu.index.encode import ColSmolEncoder
    from belge_gozu.index.loader import load_scorable_index
    from belge_gozu.retrieval.core import ExhaustiveRetriever, TwoStageRetriever
    from belge_gozu.retrieval.hybrid import HybridRetriever, load_text_channel

    s = _settings()
    # T14: serve ile AYNI yükleyici — bench, üretimin skorladığı temsilin
    # dışında bir temsili ölçmesin (packed/int8/float manifest'ten çözülür).
    idx = load_scorable_index(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    query_format = idx.manifest.query_format if idx.manifest else CPE_0_3_18
    encoder = ColSmolEncoder(s.retriever_model, s.device, query_format=query_format)

    adapter: ExhaustiveDiagnosticAdapter | TwoStageDiagnosticAdapter | HybridDiagnosticAdapter
    if pipeline == Pipeline.hybrid:
        # serve ile AYNI metin kanalı kurulumu (retrieval.hybrid.load_text_channel —
        # serve de tam olarak bunu çağırır): artefakt yoksa bench sessizce
        # yalnız-görsel ölçmemeli.
        bm25, doc_names = load_text_channel(s.index_dir, list(idx.page_ids))
        adapter = HybridDiagnosticAdapter(HybridRetriever(idx, meta, encoder, bm25, doc_names))
    elif pipeline == Pipeline.two_stage:
        # app/main.py'deki aynı korkuluk: mean-sign eleme yalnız paketli
        # bit vektörleri üstünde tanımlı (int8/float16'da page_vecs yok).
        if not isinstance(idx, PackedIndex):
            quant = idx.manifest.quantization if idx.manifest else "bilinmiyor"
            raise typer.BadParameter(
                "--pipeline two-stage yalnız sign-1bit (PackedIndex) indeksle "
                f"çalışır; {s.index_dir} yüklü: {quant}"
            )
        adapter = TwoStageDiagnosticAdapter(
            TwoStageRetriever(idx, meta, encoder),
            candidates=s.stage1_candidates,
            record_top=max(200, s.stage1_candidates),
        )
    else:
        adapter = ExhaustiveDiagnosticAdapter(ExhaustiveRetriever(idx, meta, encoder))

    selection = _load_bench_mode(bench, only_verified, min_verification)
    questions = selection.questions
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M}-{git_commit()}-{pipeline.value}"
    out_path = out or Path("data/bench/results") / f"{run_id}.json"

    report = run_retrieval_eval(
        adapter,
        questions,
        known_page_ids=set(idx.page_ids),
        run_id=run_id,
        index_manifest=idx.manifest,
        config={
            "pipeline": pipeline.value,
            "bench": str(bench),
            "verification": selection.provenance(),
        },
    )
    report.to_json(out_path)
    o = report.overall
    typer.echo(
        f"recall@5={o.recall_at.get(5, 0.0):.3f} mrr={o.mrr:.3f} ndcg5={o.ndcg5:.3f} "
        f"n={o.n} ci_recall5={o.ci_recall5}"
    )
    typer.echo(f"rapor -> {out_path}")
    if report.missing_gold_pages:
        typer.echo(f"missing_gold_pages={len(report.missing_gold_pages)}")


@bench_app.command("oracle")
def bench_oracle(
    bench: Path = typer.Option(..., "--bench"),  # noqa: B008
    packed_index: Path = typer.Option(..., "--packed-index"),  # noqa: B008
    float_index: Path = typer.Option(..., "--float-index"),  # noqa: B008
    int8_index: Path | None = typer.Option(None, "--int8-index"),  # noqa: B008
    only_verified: bool = typer.Option(False, "--only-verified/--all"),  # noqa: B008
    min_verification: VerificationLevel | None = typer.Option(  # noqa: B008
        None, "--min-verification"
    ),
    out: Path = typer.Option(..., "--out"),  # noqa: B008
) -> None:
    from belge_gozu.bench.metrics import recall_at_k
    from belge_gozu.index.encode import ColSmolEncoder
    from belge_gozu.index.quantize import Int8Index
    from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever

    s = _settings()
    idx = PackedIndex.load(packed_index)
    findex = FloatIndex.load(float_index)
    meta = pd.read_parquet(packed_index / "meta.parquet")

    if idx.manifest is None or findex.manifest is None:
        raise typer.BadParameter(
            "her iki indeksin de manifest.json'ı olmalı (--packed-index/--float-index)"
        )

    # Final review IMPORTANT-4: kollar AYNI korpusu kapsamalı. Aksi halde
    # recall@k'lar farklı sayfa kümeleri üzerinde hesaplanır ve README'nin
    # kuantizasyon iddiaları (int8 == float16, 1-bit -7 puan) sessizce
    # elmayla armut karşılaştırmasına döner.
    def _require_same_corpus(name: str, other_ids: list[str]) -> None:
        if other_ids == idx.page_ids:
            return
        missing = sorted(set(idx.page_ids) - set(other_ids))
        extra = sorted(set(other_ids) - set(idx.page_ids))
        detail = (
            f"packed'de olup {name}'de olmayan={len(missing)} {missing[:3]}; "
            f"{name}'de olup packed'de olmayan={len(extra)} {extra[:3]}"
            if (missing or extra)
            else "aynı küme, farklı SIRA (page_ids listeleri birebir eşleşmeli)"
        )
        raise typer.BadParameter(
            f"page_ids uyuşmuyor: packed n={len(idx.page_ids)} {name} n={len(other_ids)} — {detail}"
        )

    _require_same_corpus("float", findex.page_ids)
    if idx.manifest.query_format.format_id != findex.manifest.query_format.format_id:
        raise typer.BadParameter(
            "query_format uyuşmuyor: packed="
            f"{idx.manifest.query_format.format_id} float={findex.manifest.query_format.format_id}"
        )
    # T11'den beri doküman prompt'u sorgu formatından bağımsız bir eksen (bkz.
    # cli --doc-prompt): yalnız doc prompt'u farklı iki indeks aynı format_id'yi
    # taşıyabiliyor ve bu guard olmadan sessizce karşılaştırılırdı.
    if idx.manifest.doc_prompt_sha256 != findex.manifest.doc_prompt_sha256:
        raise typer.BadParameter(
            "doc_prompt uyuşmuyor: packed="
            f"{idx.manifest.doc_prompt_sha256[:12]} float={findex.manifest.doc_prompt_sha256[:12]}"
        )

    # T12/review R1 IMPORTANT-3: üçüncü (isteğe bağlı) int8 kolu — C2 ablasyonu
    # bugüne kadar hiçbir şeyin Int8Index'i skorlayamaması yüzünden koşulamıyordu.
    # Bayrak verilmezse davranış birebir eskisiyle aynı kalır (i8 None -> tüm
    # int8_* dallar atlanır, çıktı şeması değişmez).
    i8: Int8Index | None = None
    i8_manifest: IndexManifest | None = None
    if int8_index is not None:
        if not (int8_index / "codes.npy").exists():
            raise typer.BadParameter(
                f"--int8-index bir Int8Index dizini olmalı: {int8_index / 'codes.npy'} bulunamadı"
            )
        i8 = Int8Index.load(int8_index)
        if i8.manifest is None:
            raise typer.BadParameter(f"--int8-index dizininde manifest.json yok: {int8_index}")
        i8_manifest = i8.manifest
        # diğer iki kolla aynı çapraz kontrol (query_format.format_id + doc_prompt_sha256).
        if (
            i8_manifest.query_format.format_id != idx.manifest.query_format.format_id
            or i8_manifest.doc_prompt_sha256 != idx.manifest.doc_prompt_sha256
        ):
            raise typer.BadParameter(
                "int8 indeksin query_format/doc_prompt'u packed/float ile uyuşmuyor: int8="
                f"{i8_manifest.query_format.format_id}/{i8_manifest.doc_prompt_sha256[:12]} "
                f"packed={idx.manifest.query_format.format_id}/"
                f"{idx.manifest.doc_prompt_sha256[:12]}"
            )
        _require_same_corpus("int8", i8.page_ids)

    encoder = ColSmolEncoder(s.retriever_model, s.device, query_format=idx.manifest.query_format)
    retriever = ExhaustiveBinaryRetriever(idx, meta, None)
    known_binary_ids = set(idx.page_ids)
    known_float_ids = set(findex.page_ids)
    known_int8_ids = set(i8.page_ids) if i8 is not None else set()

    selection = _load_bench_mode(bench, only_verified, min_verification)
    questions = selection.questions
    ks = (1, 5, 20, 50, 200)
    per_question: list[dict] = []
    binary_recalls: dict[int, list[float]] = {k: [] for k in ks}
    float_recalls: dict[int, list[float]] = {k: [] for k in ks}
    int8_recalls: dict[int, list[float]] = {k: [] for k in ks}
    missing_gold_pages: set[str] = set()

    for q in questions:
        if not q.answerable:
            continue
        q_emb = encoder.encode_query(q.question)
        binary_scores = retriever.score_all(q_emb)
        float_scores = native_float_scores(findex, q_emb)
        int8_scores = i8.score_all(q_emb) if i8 is not None else None

        binary_order = [idx.page_ids[i] for i in np.argsort(-binary_scores, kind="stable")]
        float_order = [findex.page_ids[i] for i in np.argsort(-float_scores, kind="stable")]
        int8_order = (
            [i8.page_ids[i] for i in np.argsort(-int8_scores, kind="stable")]
            if i8 is not None and int8_scores is not None
            else None
        )

        # Bir gold sayfa ilgili indekste yoksa rank_of ValueError fırlatır;
        # koşumu kaybetmemek için o girdiyi atla, sayfayı missing_gold_pages'e
        # ekle (bench run'daki known_page_ids desenine paralel).
        binary_rank = {}
        for g in q.gold_page_ids:
            if g in known_binary_ids:
                binary_rank[g] = rank_of(binary_scores, idx.page_ids, g)
            else:
                missing_gold_pages.add(g)
        float_rank = {}
        for g in q.gold_page_ids:
            if g in known_float_ids:
                float_rank[g] = rank_of(float_scores, findex.page_ids, g)
            else:
                missing_gold_pages.add(g)
        int8_rank = {}
        if i8 is not None and int8_scores is not None:
            for g in q.gold_page_ids:
                if g in known_int8_ids:
                    int8_rank[g] = rank_of(int8_scores, i8.page_ids, g)
                else:
                    missing_gold_pages.add(g)

        row = {
            "question_id": q.question_id,
            "binary_rank": binary_rank,
            "float_rank": float_rank,
        }
        if i8 is not None:
            row["int8_rank"] = int8_rank
        per_question.append(row)

        rel = set(q.gold_page_ids)
        for k in ks:
            binary_recalls[k].append(recall_at_k(rel, binary_order, k))
            float_recalls[k].append(recall_at_k(rel, float_order, k))
            if int8_order is not None:
                int8_recalls[k].append(recall_at_k(rel, int8_order, k))

    n = len(per_question)
    summary = {
        "n": n,
        "binary": {str(k): (sum(v) / n if n else 0.0) for k, v in binary_recalls.items()},
        "float": {str(k): (sum(v) / n if n else 0.0) for k, v in float_recalls.items()},
    }
    if i8 is not None:
        summary["int8"] = {str(k): (sum(v) / n if n else 0.0) for k, v in int8_recalls.items()}
    report = {
        "run_id": f"{datetime.now(UTC):%Y%m%d-%H%M}-{git_commit()}-oracle",
        "git_commit": git_commit(),
        "bench": str(bench),
        "only_verified": only_verified,
        "verification": selection.provenance(),
        "packed_index": str(packed_index),
        "float_index": str(float_index),
        "packed_manifest": idx.manifest.model_dump(),
        "float_manifest": findex.manifest.model_dump(),
        "missing_gold_pages": sorted(missing_gold_pages),
        "summary": summary,
        "per_question": per_question,
    }
    if i8 is not None and i8_manifest is not None:
        report["int8_index"] = str(int8_index)
        report["int8_manifest"] = i8_manifest.model_dump()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    echo_line = (
        f"n={n} recall@5 binary={summary['binary']['5']:.3f} float={summary['float']['5']:.3f}"
    )
    if i8 is not None:
        echo_line += f" int8={summary['int8']['5']:.3f}"
    typer.echo(echo_line)
    typer.echo(f"oracle raporu -> {out}")
    if missing_gold_pages:
        typer.echo(f"missing_gold_pages={len(missing_gold_pages)}")


# --------------------------------------------------------------------------
# calibrate — P2 T5+T6
# --------------------------------------------------------------------------

DEFAULT_CANARY = Path("data/bench/canary_v1.jsonl")
DEFAULT_UNANS = Path("data/bench/unans_v1.jsonl")
DEFAULT_SPLITS = Path("data/bench/splits_v1.json")
DEFAULT_CALIBRATION_REPORT = Path("data/bench/results/p2-calibration-dev-v1.json")


class Split(StrEnum):
    dev = "dev"
    test = "test"


def _gate_test_split(split: Split, yes_final_gate: bool) -> None:
    """`--split test` = FAZ SONU KAPI KOŞUMU. Kazayla koşulamaz (G2.4).

    Test bölmesi tek kullanımlıktır: üzerinde bir kez ölçüm yapıldıktan sonra
    her ek koşum onu sessizce bir dev kümesine çevirir (eşik seçimi test
    sayısına bakarak ayarlanmaya başlar) ve kapı sayısı geçerliliğini yitirir.
    Bu yüzden bariyer bir bayrak DEĞİL, açık bir onaydır.
    """
    if split != Split.test:
        return
    typer.secho("=" * 78, err=True, fg=typer.colors.RED, bold=True)
    typer.secho(
        "UYARI — TEST BÖLMESİ: G2.4 gereği test tek koşumdur ve FAZ SONUNDA,\n"
        "kapı koşumu olarak ölçülür. KAPI KOŞUMU DIŞINDA KULLANMAYIN: her ek\n"
        "koşum test kümesini fiilen bir dev kümesine çevirir ve kapı sayısını\n"
        "geçersiz kılar (eşik test'e bakarak seçilmiş olur).",
        err=True,
        fg=typer.colors.RED,
        bold=True,
    )
    typer.secho("=" * 78, err=True, fg=typer.colors.RED, bold=True)
    if not yes_final_gate:
        raise typer.BadParameter(
            "--split test için --yes-final-gate zorunludur (faz sonu kapı koşumu onayı)"
        )


def _calibration_setup(s: Settings, canary: Path, unans: Path, splits_path: Path):
    """Metin kanalı + etiketli satırlar + veri künyesi (MODEL/AĞ YOK, saf CPU).

    Görsel indeks YÜKLENMEZ: yalnız `page_ids.json` okunur ve BM25 metin
    kanalı kurulur. Ölçülen özelliklerin tamamı metin yanındandır (görsel
    özellikler AUC .34 ile ölçülmüş TERS yönde ve reddedildi), yani 481 MB'lık
    `codes.npy`'yi mmap'lemek için hiçbir neden yok.
    """
    from belge_gozu.answer.calibrate import build_rows, git_blob_sha, load_rows, sha256_file
    from belge_gozu.bench.dataset import load_splits
    from belge_gozu.index.manifest import index_revision
    from belge_gozu.retrieval.hybrid import load_text_channel

    ids_path = s.index_dir / "page_ids.json"
    if not ids_path.exists():
        raise typer.BadParameter(
            f"indeks dizininde page_ids.json yok: {s.index_dir} — "
            "önce indeksi kurun/indirin (`belge-gozu index build` ya da `index pull`)"
        )
    manifest = read_manifest(s.index_dir)
    if manifest is None:
        raise typer.BadParameter(
            f"{s.index_dir} indeksinde manifest.json yok; kalibrasyon artefaktı "
            "hangi indekse ait olduğu bilinmeyen bir kurulumdan üretilemez"
        )
    page_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    bm25, doc_names = load_text_channel(s.index_dir, list(page_ids))

    splits = load_splits(splits_path)
    rows = []
    data_files = []
    for label, path in (("canary", canary), ("unans", unans)):
        raw = load_rows(path, only_verified=True)
        rows.extend(build_rows(raw, splits, bm25, doc_names, source=label))
        data_files.append(
            {
                "name": label,
                "path": str(path),
                "sha256": sha256_file(path),
                # review M4: içeriği GERİ GETİREN referans (`git cat-file -p <blob>`),
                # yalnız kimliklendiren değil.
                "git_blob": git_blob_sha(path),
                "n_lines": len(path.read_text(encoding="utf-8").splitlines()),
                "n_verified": len(raw),
            }
        )
    splits_meta = json.loads(splits_path.read_text(encoding="utf-8"))
    kunye = {
        "data_files": data_files,
        "splits": {
            "path": str(splits_path),
            "sha256": sha256_file(splits_path),
            "git_blob": git_blob_sha(splits_path),
            "version": splits_meta.get("version"),
            "seed": splits_meta.get("seed"),
            "scheme": splits_meta.get("scheme"),
        },
        "index_dir": str(s.index_dir),
        "index_manifest": manifest.model_dump(),
        "only_verified": True,
        "label_definition": (
            "safe_to_answer=1 <=> answerable=True VE gold sayfa BM25+yönlendirme "
            "top-5'inde; cevaplanamaz TÜM sorular ve getirimi ıskalayan "
            "cevaplanabilir sorular 0 (LLM kullanılmadı)"
        ),
    }
    return bm25, doc_names, rows, index_revision(manifest), kunye


def _echo_rc_head(curve: list[dict], limit: int = 8) -> None:
    typer.echo("  tau      coverage  risk")
    for pt in sorted(curve, key=lambda p: -p["coverage"])[:limit]:
        typer.echo(f"  {pt['tau']:.6f}  {pt['coverage']:.4f}    {pt['risk']:.4f}")


@calibrate_app.command("fit")
def calibrate_fit(
    split: Split = typer.Option(Split.dev, "--split"),  # noqa: B008
    canary: Path = typer.Option(DEFAULT_CANARY, "--canary"),  # noqa: B008
    unans: Path = typer.Option(DEFAULT_UNANS, "--unans"),  # noqa: B008
    splits_path: Path = typer.Option(DEFAULT_SPLITS, "--splits"),  # noqa: B008
    max_risk: float = typer.Option(0.05, "--max-risk"),  # noqa: B008
    alpha: float = typer.Option(0.05, "--alpha"),  # noqa: B008
    calibration_dir_opt: Path | None = typer.Option(None, "--calibration-dir"),  # noqa: B008
    out: Path = typer.Option(DEFAULT_CALIBRATION_REPORT, "--out"),  # noqa: B008
    # Künyeye serbest metin not. Bench verisi AKTİF TASLAKTA olduğu için gerekli:
    # `--unans` bir dosya YOLU alır, ama koşumun kimliği o yolun O ANKİ İÇERİĞİDİR
    # (künyedeki sha256). Dosya sonradan değişince yol tek başına yanıltıcı olur;
    # not, hangi sürüme sabitlendiğini (ör. bir commit) insan diliyle yazar.
    note: str = typer.Option("", "--note"),  # noqa: B008
    yes_final_gate: bool = typer.Option(False, "--yes-final-gate"),  # noqa: B008
) -> None:
    """Güven kalibratörünü fit eder: sürüm-anahtarlı artefakt + künyeli koşum raporu.

    Model/ağ/kota KULLANMAZ — özelliklerin tamamı BM25 metin kanalından ve
    sorgudan hesaplanır, etiket ise gold sayfanın top-5'te olup olmadığından.
    """
    from belge_gozu.answer.calibrate import (
        CALIBRATOR_FILENAME,
        GUARANTEE_CP,
        calibration_dir,
        fit_calibration,
        per_question_rows,
    )

    _gate_test_split(split, yes_final_gate)
    s = _settings()
    if s.retrieval_pipeline != "hybrid":
        raise typer.BadParameter(
            f"calibrate fit yalnız hybrid boru hattı için tanımlı (özellikler BM25 metin "
            f"kanalından okunur); BG_RETRIEVAL_PIPELINE={s.retrieval_pipeline}"
        )

    _, _, rows, revision, kunye = _calibration_setup(s, canary, unans, splits_path)
    subset = [r for r in rows if r.split == split.value]
    if not subset:
        raise typer.BadParameter(f"{split.value} bölmesinde hiç soru yok")
    kunye["split"] = split.value
    if note:
        kunye["note"] = note

    artifact = fit_calibration(
        subset,
        index_revision=revision,
        pipeline=s.retrieval_pipeline,
        max_risk=max_risk,
        alpha=alpha,
        data_kunye=kunye,
    )
    base = calibration_dir_opt or (s.data_dir / "calibration")
    art_dir = calibration_dir(base, artifact.key)
    art_path = artifact.save(art_dir)

    counts = artifact.kunye["counts"]
    metrics = artifact.kunye["dev_metrics"]
    report = {
        "run_id": f"{datetime.now(UTC):%Y%m%d-%H%M}-{git_commit()}-calibrate-{split.value}",
        "git_commit": git_commit(),
        "created_at": artifact.kunye["created_at"],
        "split": split.value,
        "key": artifact.key,
        "index_revision": artifact.index_revision,
        "pipeline": artifact.pipeline,
        "recipe_fingerprint": artifact.recipe_fingerprint,
        "artifact_path": str(art_path),
        "artifact_committed": False,
        "kunye": {k: v for k, v in artifact.kunye.items() if k != "dev_metrics"},
        "calibrator": artifact.calibrator.to_dict(),
        "thresholds": artifact.thresholds,
        "metrics": metrics,
        # review M3: raporun HER sayısı yalnız bu dosyadan yeniden hesaplanabilsin.
        "per_question": per_question_rows(artifact, subset),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    typer.echo(
        f"bölme={split.value} n={counts['total']} "
        f"(pozitif={counts['positive_safe_to_answer']}, negatif={counts['negative']})"
    )
    typer.echo(
        f"  cevaplanabilir={counts['answerable']} "
        f"(gold@5={counts['answerable_gold_in_top5']}, ıska={counts['answerable_retrieval_miss']}) "
        f"cevaplanamaz={counts['unanswerable']}"
    )
    if counts["answerable"] < 20 or counts["unanswerable"] < 40:
        typer.secho(
            "  UYARI: dev n eşiğin altında (cevaplanabilir<20 ya da cevaplanamaz<40) — "
            "bu fit KIRILGANDIR, CI'lar geniştir",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    fi = artifact.calibrator.fit_info
    typer.echo(f"fit: iter={fi['n_iter']} converged={fi['converged']} nll={fi['final_nll']:.4f}")
    for name, w in zip(artifact.calibrator.feature_names, artifact.calibrator.weights, strict=True):
        typer.echo(f"  w[{name}] = {w:+.4f}")
    typer.echo(f"  bias = {artifact.calibrator.bias:+.4f}")

    ch = artifact.thresholds["chosen"]
    typer.echo(
        f"tau({ch['name']})={ch['value']:.6f} coverage={ch['coverage']:.3f} "
        f"risk={ch['risk_point']:.3f} (nokta tahmini)"
    )
    typer.echo(
        f"  belirsizlik: n_answered={ch['n_answered']} hata={ch['errors']} "
        f"%95 CP üst sınır={ch['risk_cp_upper_95']:.3f} "
        f"guarantee={ch['statistical_guarantee']}"
    )
    # review J1: seçilen eşiğin güvencesi yoksa bunu artefakt kadar CLI de
    # yüksek sesle söylemeli — conformal dalının "n yetersiz" kaydının aynısı.
    if ch["statistical_guarantee"] != GUARANTEE_CP:
        typer.secho(
            f"  UYARI: v1 eşiği NOKTA TAHMİNİDİR, n={ch['n_answered']}, "
            f"CP üst %{ch['risk_cp_upper_95'] * 100:.1f} — İSTATİSTİKSEL GÜVENCE YOK; "
            "kapı koşumu verifier sinyali olmadan yapılmayacak",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    typer.echo(f"  gerekçe: {ch['rationale']}")
    conf = artifact.thresholds["conformal"]
    typer.echo(f"  conformal: {conf['note']}")
    auroc_val = metrics.get("auroc")
    auroc_txt = f"{auroc_val:.4f}" if auroc_val is not None else "yok (tek sınıf)"
    typer.echo(
        f"dev: auroc={auroc_txt} brier={metrics['brier']:.4f} "
        f"ece={metrics['ece']:.4f} aurc={metrics['aurc']:.4f}"
    )
    far = metrics.get("false_answer_on_unanswerable")
    if far:
        typer.echo(
            f"     DEV yanlış-yanıt (cevaplanamaz): {far['rate']:.4f} "
            f"({far['errors']}/{far['n']}, %95 üst sınır {far['upper_bound_95']:.4f}) "
            "— G2.1 KAPI SAYISI DEĞİL"
        )
    typer.echo("risk-coverage (kapsama azalan, ilk 8):")
    _echo_rc_head(metrics["risk_coverage"])
    typer.echo(f"artefakt -> {art_path} ({CALIBRATOR_FILENAME} gitignore'da; yeniden üretilebilir)")
    typer.echo(f"rapor -> {out}")


@calibrate_app.command("eval")
def calibrate_eval(
    split: Split = typer.Option(Split.dev, "--split"),  # noqa: B008
    canary: Path = typer.Option(DEFAULT_CANARY, "--canary"),  # noqa: B008
    unans: Path = typer.Option(DEFAULT_UNANS, "--unans"),  # noqa: B008
    splits_path: Path = typer.Option(DEFAULT_SPLITS, "--splits"),  # noqa: B008
    calibration_dir_opt: Path | None = typer.Option(None, "--calibration-dir"),  # noqa: B008
    out: Path | None = typer.Option(None, "--out"),  # noqa: B008
    yes_final_gate: bool = typer.Option(False, "--yes-final-gate"),  # noqa: B008
) -> None:
    """Kayıtlı artefaktı yükleyip metrikleri YENİDEN hesaplar (fit ile aynı `evaluate` kodu)."""
    from belge_gozu.answer.calibrate import (
        calibration_dir,
        calibration_key,
        evaluate,
        load_calibrator,
    )

    _gate_test_split(split, yes_final_gate)
    s = _settings()
    _, _, rows, revision, _ = _calibration_setup(s, canary, unans, splits_path)
    subset = [r for r in rows if r.split == split.value]
    if not subset:
        raise typer.BadParameter(f"{split.value} bölmesinde hiç soru yok")

    key = calibration_key(revision, s.retrieval_pipeline)
    base = calibration_dir_opt or (s.data_dir / "calibration")
    artifact = load_calibrator(calibration_dir(base, key), key)
    metrics = evaluate(artifact, subset)

    typer.echo(f"anahtar={key}")
    typer.echo(
        f"bölme={split.value} n={metrics['n']} "
        f"(pozitif={metrics['n_positive']}, negatif={metrics['n_negative']})"
    )
    risk_at = metrics["risk_at_tau"]
    risk_txt = f"{risk_at:.3f}" if risk_at is not None else "tanımsız (kapsama 0)"
    typer.echo(
        f"tau={metrics['tau']:.6f} coverage={metrics['coverage_at_tau']:.3f} risk={risk_txt}"
    )
    # review J1: eşiğin güvence durumu artefaktın parçası — eval de göstermeli.
    ch = artifact.thresholds["chosen"]
    typer.echo(
        f"  artefakt eşiği: n_answered={ch['n_answered']} hata={ch['errors']} "
        f"%95 CP üst sınır={ch['risk_cp_upper_95']:.3f} "
        f"guarantee={ch['statistical_guarantee']}"
    )
    auroc_val = metrics.get("auroc")
    auroc_txt = f"{auroc_val:.4f}" if auroc_val is not None else "yok (tek sınıf)"
    typer.echo(
        f"auroc={auroc_txt} brier={metrics['brier']:.4f} "
        f"ece={metrics['ece']:.4f} aurc={metrics['aurc']:.4f}"
    )
    typer.echo("risk-coverage (kapsama azalan, ilk 8):")
    _echo_rc_head(metrics["risk_coverage"])
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now(UTC):%Y%m%d-%H%M}-{git_commit()}"
        payload = {
            "run_id": f"{stamp}-calibrate-eval-{split.value}",
            "git_commit": git_commit(),
            "key": key,
            "split": split.value,
            "metrics": metrics,
        }
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        typer.echo(f"rapor -> {out}")


# --------------------------------------------------------------------------
# verify — P2 T1+T2 kanıt kapısı koşum harness'ı
# --------------------------------------------------------------------------


def _verify_service(s: Settings, budget):
    """İki kapısı açık servis, kapı künyesi, manifest ve indeks sürümü.

    Serve ile AYNI parçalar (`load_text_channel` + `HybridRetriever` +
    `GeminiAnswerer` + `build_gates`): harness'ın ölçtüğü şey üretimin
    koştuğu şeyden sapamaz. Ağır kurulum (VLM ağırlıkları) burada TOPLANIR;
    testler bu tek fonksiyonu değiştirerek stub bir istemciyle koşabilir.
    """
    from belge_gozu.answer.base import AskService
    from belge_gozu.answer.gemini import GeminiAnswerer
    from belge_gozu.answer.verify import build_gates
    from belge_gozu.index.encode import ColSmolEncoder
    from belge_gozu.index.loader import load_scorable_index
    from belge_gozu.index.manifest import index_revision
    from belge_gozu.retrieval.hybrid import HybridRetriever, load_text_channel

    idx = load_scorable_index(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    query_format = idx.manifest.query_format if idx.manifest else CPE_0_3_18
    encoder = ColSmolEncoder(s.retriever_model, s.device, query_format=query_format)
    bm25, doc_names = load_text_channel(s.index_dir, list(idx.page_ids))
    retriever = HybridRetriever(idx, meta, encoder, bm25, doc_names)
    revision = index_revision(idx.manifest) if idx.manifest else None
    gates = build_gates(s, retriever, index_revision=revision, budget=budget)
    service = AskService(
        retriever,
        GeminiAnswerer(s.gemini_model, s.gemini_api_key, api_key_2=s.google_api_key_2),
        s.min_score_threshold,
        lambda p: (s.data_dir / p).read_bytes(),
        gate1=gates.retrieval,
        gate2=gates.evidence,
    )
    return service, gates, idx.manifest, revision


def _answer_eval_command(
    *,
    bench_paths: list[Path],
    split: Split,
    max_llm_attempts: int,
    limit: int,
    splits_path: Path,
    only_verified: bool,
    min_verification: VerificationLevel | None,
    out: Path | None,
    yes_final_gate: bool,
    command_name: str,
) -> None:
    """Run the two-gate answer evaluator used by both public CLI paths."""
    from belge_gozu.answer.base import is_honest_miss
    from belge_gozu.answer.calibrate import git_blob_sha, load_rows, sha256_file
    from belge_gozu.answer.verify import VerifierBudget
    from belge_gozu.bench.answer_eval import AnswerRecord, ClaimRecord, run_answer_eval
    from belge_gozu.bench.dataset import assign_split, load_splits
    from belge_gozu.retrieval.text import recipe_fingerprint
    from belge_gozu.telemetry.collect import collecting

    _gate_test_split(split, yes_final_gate)
    if max_llm_attempts < 0:
        raise typer.BadParameter("--max-llm-attempts negatif olamaz")
    if not bench_paths:
        raise typer.BadParameter("en az bir --bench dosyası zorunludur")

    base = _settings()
    if base.retrieval_pipeline != "hybrid":
        raise typer.BadParameter(
            f"{command_name} yalnız hibrit boru hattında tanımlı (kalibre kapı BM25 metin "
            f"kanalından okur); BG_RETRIEVAL_PIPELINE={base.retrieval_pipeline}"
        )
    s = base.model_copy(update={"gate_calibrated": True, "gate_verifier": True})
    splits = load_splits(splits_path)

    selections = [_load_bench_mode(path, only_verified, min_verification) for path in bench_paths]
    raw: list[dict] = []
    source_meta: list[dict] = []
    seen_ids: set[str] = set()
    for path, selection in zip(bench_paths, selections, strict=True):
        selected_ids = {q.question_id for q in selection.questions}
        source_rows = [
            rec
            for rec in load_rows(path, only_verified=False)
            if rec["question_id"] in selected_ids
        ]
        duplicates = seen_ids.intersection(str(rec["question_id"]) for rec in source_rows)
        if duplicates:
            raise typer.BadParameter(
                f"bench dosyalarında yinelenen question_id: {sorted(duplicates)[:5]}"
            )
        seen_ids.update(str(rec["question_id"]) for rec in source_rows)
        raw.extend(source_rows)
        source_meta.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "git_blob": git_blob_sha(path),
                **selection.provenance(),
            }
        )

    subset = [rec for rec in raw if assign_split(rec, splits) == split.value]
    if limit > 0:
        subset = subset[:limit]
    if not subset:
        paths = ", ".join(str(path) for path in bench_paths)
        raise typer.BadParameter(f"{split.value} bölmesinde koşulacak soru yok: {paths}")

    budget = VerifierBudget(max_llm_attempts)
    service, gates, manifest, revision = _verify_service(s, budget)
    if manifest is None:
        manifest_payload = None
    elif hasattr(manifest, "model_dump"):
        manifest_payload = manifest.model_dump()
    elif isinstance(manifest, dict):
        manifest_payload = dict(manifest)
    else:
        raise TypeError(f"desteklenmeyen manifest türü: {type(manifest).__name__}")

    records: list[AnswerRecord] = []
    legacy_rows: list[dict] = []
    stopped = None
    for rec in subset:
        if budget.remaining == 0 and records:
            stopped = (
                f"bütçe doldu ({budget.used}/{budget.max_attempts} API denemesi) "
                f"— {len(records)} soru koşuldu"
            )
            break
        with collecting() as col:
            answer, hits = service.ask(str(rec["question"]), k=s.top_k)
        status = (
            "degraded"
            if col.notes.get("degraded")
            else ("abstained" if answer.abstained else "answered")
        )
        gate1 = col.notes.get("gate1")
        if not isinstance(gate1, dict):
            gate1 = None
        gate2 = col.notes.get("gate2")
        if not isinstance(gate2, dict):
            gate2 = {}
        claims = tuple(
            ClaimRecord(
                claim_id=str(row["claim_id"]),
                verdict=row["verdict"],
                gerekce=str(row.get("gerekce", "")),
                cited_sources=tuple(row.get("cited_sources") or ()),
                inherited_sources=bool(row.get("inherited_sources", False)),
                cached=bool(row.get("cached", False)),
                attempts=int(row.get("attempts", 0) or 0),
            )
            for row in gate2.get("claims") or ()
        )
        record = AnswerRecord(
            question_id=str(rec["question_id"]),
            question=str(rec["question"]),
            answerable=bool(rec["answerable"]),
            unanswerable_reason=rec.get("unanswerable_reason"),
            slice=rec.get("slice"),
            status=status,
            honest_miss=is_honest_miss(answer),
            answer_text=answer.text,
            citations=tuple(answer.citations),
            top_score=hits[0].score if hits else None,
            gate1=gate1,
            n_claims=int(gate2.get("n_claims", len(claims)) or 0),
            claims=claims,
        )
        records.append(record)
        legacy_rows.append(
            {
                "qid": record.question_id,
                "answerable": record.answerable,
                "unanswerable_reason": record.unanswerable_reason,
                "slice": record.slice,
                "status": record.status,
                "citations": list(record.citations),
                "top_score": record.top_score,
                "gate1": gate1,
                "gate2": gate2,
            }
        )

    verdict_counts: Counter[str] = Counter()
    llm_calls = cache_hits = demoted = api_attempts = 0
    for row in legacy_rows:
        gate2 = row["gate2"] or {}
        llm_calls += int(gate2.get("llm_calls", 0) or 0)
        api_attempts += int(gate2.get("api_attempts", 0) or 0)
        cache_hits += int(gate2.get("cache_hits", 0) or 0)
        demoted += 1 if gate2.get("demoted") else 0
        for claim in gate2.get("claims") or ():
            verdict_counts[claim["verdict"]] += 1
    gate1_passed = sum(1 for row in legacy_rows if (row["gate1"] or {}).get("passed"))

    verification = {
        "only_verified": only_verified,
        "min_verification": min_verification.value if min_verification else None,
        "total": sum(selection.total for selection in selections),
        "selected": sum(selection.selected for selection in selections),
        "filtered_out": sum(selection.filtered_out for selection in selections),
    }
    split_meta = {"path": str(splits_path), "sha256": sha256_file(splits_path)}
    dataset = {
        "bench": source_meta[0],
        "sources": source_meta,
        "splits": split_meta,
        "verification": verification,
        "selected_after_split": len(subset),
        "run": len(records),
    }
    budget_meta = {
        "unit": "api_attempts",
        "max_attempts": budget.max_attempts,
        "used": budget.used,
        "stopped": stopped,
    }
    recipe = recipe_fingerprint()
    gate1_detail = gates.detail.get("gate1") or {}
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%d-%H%M}-{git_commit()}-answers-{split.value}"
    report = run_answer_eval(
        records,
        run_id=run_id,
        git_commit=git_commit(),
        created_at=now,
        split=split.value,
        index_manifest=manifest_payload,
        index_revision=revision,
        calibrator_key=gate1_detail.get("key"),
        config={
            "pipeline": s.retrieval_pipeline,
            "recipe_fingerprint": recipe,
            "gates": gates.detail,
            "gate_calibrated": s.gate_calibrated,
            "gate_verifier": s.gate_verifier,
            "gemini_model": s.gemini_model,
            "min_score_threshold": s.min_score_threshold,
            "top_k": s.top_k,
            "verifier_max_claims": s.verifier_max_claims,
            "verification": verification,
        },
        dataset=dataset,
        budget=budget_meta,
    )
    summary = {
        "n": len(records),
        "by_status": dict(Counter(record.status for record in records)),
        "gate1_passed": gate1_passed,
        "gate2_demoted": demoted,
        "verdicts": dict(verdict_counts),
        "verifier_llm_calls": llm_calls,
        "verifier_api_attempts": api_attempts,
        "verifier_cache_hits": cache_hits,
    }
    payload = report.model_dump(mode="json")
    # `verify run` was public before `AnswerEvalReport`; keep its diagnostic
    # aliases while both command paths now share the canonical records/metrics.
    payload.update(
        {
            "pipeline": s.retrieval_pipeline,
            "recipe_fingerprint": recipe,
            "gates": gates.detail,
            "bench": {**source_meta[0], "n_selected": len(subset), "n_run": len(records)},
            "splits": split_meta,
            "summary": summary,
            "per_question": legacy_rows,
        }
    )
    out_path = out or (Path("data/bench/results") / f"{run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    typer.echo(f"bölme={split.value} koşulan={len(records)}/{len(subset)}")
    typer.echo(f"durum: {summary['by_status']}")
    typer.echo(f"kapı1 geçen={gate1_passed}  kapı2 düşürülen={demoted}")
    typer.echo(f"kararlar: {dict(verdict_counts)}")
    typer.echo(
        f"LLM doğrulayıcı çağrısı={llm_calls} API denemesi={api_attempts} "
        f"önbellek isabeti={cache_hits} bütçe={budget.used}/{budget.max_attempts} deneme"
    )
    if stopped:
        typer.secho(f"  UYARI: {stopped}", fg=typer.colors.YELLOW, bold=True)
    typer.echo(f"rapor -> {out_path}")


@bench_app.command("answers")
def bench_answers(
    bench: list[Path] = typer.Option(  # noqa: B008
        [DEFAULT_CANARY, DEFAULT_UNANS],
        "--bench",
        help="Birden çok kez verilebilir; varsayılan canary + unanswerable kümesidir.",
    ),
    split: Split = typer.Option(Split.dev, "--split"),  # noqa: B008
    max_llm_attempts: int = typer.Option(..., "--max-llm-attempts", "--max-llm-calls"),  # noqa: B008
    limit: int = typer.Option(0, "--limit"),  # noqa: B008
    splits_path: Path = typer.Option(DEFAULT_SPLITS, "--splits"),  # noqa: B008
    only_verified: bool = typer.Option(True, "--only-verified/--all"),  # noqa: B008
    min_verification: VerificationLevel | None = typer.Option(  # noqa: B008
        None, "--min-verification"
    ),
    out: Path | None = typer.Option(None, "--out"),  # noqa: B008
    yes_final_gate: bool = typer.Option(  # noqa: B008
        False,
        "--yes-final-gate",
        help="Yalnız --split test için faz-sonu tek-koşum onayı.",
    ),
) -> None:
    """G2 answer metrics; test split requires --yes-final-gate."""
    _answer_eval_command(
        bench_paths=bench,
        split=split,
        max_llm_attempts=max_llm_attempts,
        limit=limit,
        splits_path=splits_path,
        only_verified=only_verified,
        min_verification=min_verification,
        out=out,
        yes_final_gate=yes_final_gate,
        command_name="bench answers",
    )


@verify_app.command("run")
def verify_run(
    bench: Path = typer.Option(DEFAULT_CANARY, "--bench"),  # noqa: B008
    split: Split = typer.Option(Split.dev, "--split"),  # noqa: B008
    max_llm_calls: int = typer.Option(..., "--max-llm-attempts", "--max-llm-calls"),  # noqa: B008
    limit: int = typer.Option(0, "--limit"),  # noqa: B008
    splits_path: Path = typer.Option(DEFAULT_SPLITS, "--splits"),  # noqa: B008
    only_verified: bool = typer.Option(True, "--only-verified/--all"),  # noqa: B008
    min_verification: VerificationLevel | None = typer.Option(  # noqa: B008
        None, "--min-verification"
    ),
    out: Path | None = typer.Option(None, "--out"),  # noqa: B008
    yes_final_gate: bool = typer.Option(False, "--yes-final-gate"),  # noqa: B008
) -> None:
    """Compatibility alias for the shared two-gate answer evaluator."""
    _answer_eval_command(
        bench_paths=[bench],
        split=split,
        max_llm_attempts=max_llm_calls,
        limit=limit,
        splits_path=splits_path,
        only_verified=only_verified,
        min_verification=min_verification,
        out=out,
        yes_final_gate=yes_final_gate,
        command_name="verify run",
    )


@app.command("serve")
def serve(
    pull: bool = typer.Option(False, "--pull"),  # noqa: B008
    port: int = typer.Option(7860),  # noqa: B008
) -> None:
    s = _settings()
    if pull and s.hf_dataset_repo:
        from belge_gozu.index.hub import pull_index

        pull_index(s.hf_dataset_repo, s.index_dir, data_dir=s.data_dir)
    import uvicorn

    uvicorn.run("belge_gozu.app.main:create_app", factory=True, host="0.0.0.0", port=port)
