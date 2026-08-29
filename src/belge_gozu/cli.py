import json
import math
import shutil
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from PIL import Image

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
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")
app.add_typer(metrics_app, name="metrics")
app.add_typer(bench_app, name="bench")

DEFAULT_MANIFEST = Path("data/manifest/v0_manifest.csv")


class Pipeline(StrEnum):
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
_CLI_DEFAULTS = Settings()
DEFAULT_QUERY_FORMAT = QueryFormatChoice(_CLI_DEFAULTS.query_format_id)
DEFAULT_DOC_PROMPT = DocPromptChoice(_CLI_DEFAULTS.doc_prompt_id)


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
        n, avg = db.execute("SELECT COUNT(*), COALESCE(AVG(total_ms),0) FROM events").fetchone()
        ab = db.execute(
            "SELECT COALESCE(AVG(abstained),0) FROM events "
            "WHERE endpoint='/ask' AND status <> 'degraded'"
        ).fetchone()[0]
        tok = db.execute(
            "SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), "
            "COALESCE(SUM(est_cost_usd),0) FROM events"
        ).fetchone()
        vals = sorted(r[0] for r in db.execute("SELECT total_ms FROM events"))
    except sqlite3.OperationalError:
        typer.echo("henüz olay kaydı yok")
        return
    finally:
        if db is not None:
            db.close()
    p95 = vals[min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)] if vals else 0.0
    typer.echo(f"istek={n} ort={avg:.0f}ms p95={p95:.0f}ms abstain={ab:.1%}")
    typer.echo(f"token in/out={tok[0]}/{tok[1]} maliyet≈${tok[2]:.4f}")


def _load_bench_mode(bench: Path, only_verified: bool) -> tuple[list, bool]:
    """Bench JSONL'i `only_verified` moduna göre yükler, aktif modu yazdırır.

    `load_bench` saf JSONL okur — model/indekse dokunmaz, bu yüzden birim
    testte gerçek encoder olmadan doğrudan test edilebilir. R15: canary_v1
    insan doğrulaması tamamlanana kadar taslak dahil TÜMÜ (--all) varsayılan
    olmalı, aksi halde `bench run`/`bench oracle` hiç koşamaz."""
    from belge_gozu.bench.dataset import load_bench

    questions = load_bench(bench, only_verified=only_verified)
    if only_verified:
        typer.echo(f"bench modu: yalnız doğrulanmış (n={len(questions)})")
    else:
        typer.echo(f"bench modu: TÜMÜ (taslak dahil, n={len(questions)})")
    return questions, only_verified


@bench_app.command("run")
def bench_run(
    bench: Path = typer.Option(Path("data/bench/canary_v1.jsonl")),  # noqa: B008
    pipeline: Pipeline = typer.Option(Pipeline.exhaustive, "--pipeline"),  # noqa: B008
    only_verified: bool = typer.Option(False, "--only-verified/--all"),  # noqa: B008
    out: Path | None = typer.Option(None, "--out"),  # noqa: B008
) -> None:
    from belge_gozu.bench.harness import (
        ExhaustiveDiagnosticAdapter,
        TwoStageDiagnosticAdapter,
        run_retrieval_eval,
    )
    from belge_gozu.index.encode import ColSmolEncoder
    from belge_gozu.index.loader import load_scorable_index
    from belge_gozu.retrieval.core import ExhaustiveRetriever, TwoStageRetriever

    s = _settings()
    # T14: serve ile AYNI yükleyici — bench, üretimin skorladığı temsilin
    # dışında bir temsili ölçmesin (packed/int8/float manifest'ten çözülür).
    idx = load_scorable_index(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    query_format = idx.manifest.query_format if idx.manifest else CPE_0_3_18
    encoder = ColSmolEncoder(s.retriever_model, s.device, query_format=query_format)

    adapter: ExhaustiveDiagnosticAdapter | TwoStageDiagnosticAdapter
    if pipeline == Pipeline.two_stage:
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

    questions, only_verified = _load_bench_mode(bench, only_verified)
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M}-{git_commit()}-{pipeline.value}"
    out_path = out or Path("data/bench/results") / f"{run_id}.json"

    report = run_retrieval_eval(
        adapter,
        questions,
        known_page_ids=set(idx.page_ids),
        run_id=run_id,
        index_manifest=idx.manifest,
        config={"pipeline": pipeline.value, "bench": str(bench), "only_verified": only_verified},
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

    questions, only_verified = _load_bench_mode(bench, only_verified)
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
