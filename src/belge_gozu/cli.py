import math
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import typer
from PIL import Image

from belge_gozu.config import Settings
from belge_gozu.corpus.download import download_all
from belge_gozu.corpus.manifest import build_http_client, load_manifest, probe
from belge_gozu.corpus.render import render_all
from belge_gozu.index.encode import FakeEncoder
from belge_gozu.index.manifest import (
    CPE_0_3_18,
    IndexManifest,
    RenderConfig,
    corpus_checksum,
    write_manifest,
)
from belge_gozu.index.store import PackedIndex

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


def _settings() -> Settings:
    return Settings()


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
def index_build(fake: bool = typer.Option(False, "--fake")) -> None:  # noqa: B008
    s = _settings()
    meta = pd.read_parquet(s.data_dir / "meta.parquet")
    if fake:
        encoder = FakeEncoder()
    else:
        from belge_gozu.index.encode import ColSmolEncoder

        encoder = ColSmolEncoder(s.retriever_model, s.device)
    embs, ids = [], []
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
    PackedIndex.build(ids, embs).save(s.index_dir)
    shutil.copy(s.data_dir / "meta.parquet", s.index_dir / "meta.parquet")
    typer.echo(f"{len(ids)} sayfa indekslendi -> {s.index_dir}")


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

    def _pkg_version(name: str) -> str:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as pkg_version

        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return "unknown"

    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        git_commit = "unknown"

    manifest = IndexManifest(
        model_name=s.retriever_model,
        model_revision="unknown",
        engine_versions={
            "colpali-engine": _pkg_version("colpali-engine"),
            "transformers": _pkg_version("transformers"),
            "torch": _pkg_version("torch"),
        },
        query_format=CPE_0_3_18,
        doc_prompt_sha256="unknown",
        quantization="sign-1bit",
        mask_policy="none",
        render=RenderConfig(),
        corpus_checksum=corpus_checksum(s.index_dir),
        n_pages=n_pages,
        n_tokens=n_tokens,
        built_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit,
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


@bench_app.command("run")
def bench_run(
    bench: Path = typer.Option(Path("data/bench/canary_v1.jsonl")),  # noqa: B008
    pipeline: str = typer.Option("exhaustive", "--pipeline"),  # noqa: B008
    out: Path | None = typer.Option(None, "--out"),  # noqa: B008
) -> None:
    from belge_gozu.bench.dataset import load_bench
    from belge_gozu.bench.harness import (
        ExhaustiveDiagnosticAdapter,
        TwoStageDiagnosticAdapter,
        _git_commit,
        run_retrieval_eval,
    )
    from belge_gozu.index.encode import ColSmolEncoder
    from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever, TwoStageRetriever

    s = _settings()
    idx = PackedIndex.load(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    encoder = ColSmolEncoder(s.retriever_model, s.device)

    adapter: ExhaustiveDiagnosticAdapter | TwoStageDiagnosticAdapter
    if pipeline == "two-stage":
        adapter = TwoStageDiagnosticAdapter(
            TwoStageRetriever(idx, meta, encoder), candidates=s.stage1_candidates
        )
    else:
        adapter = ExhaustiveDiagnosticAdapter(ExhaustiveBinaryRetriever(idx, meta, encoder))

    questions = load_bench(bench)
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M}-{_git_commit()}-{pipeline}"
    out_path = out or Path("data/bench/results") / f"{run_id}.json"

    report = run_retrieval_eval(
        adapter,
        questions,
        known_page_ids=set(idx.page_ids),
        run_id=run_id,
        index_manifest=idx.manifest,
        config={"pipeline": pipeline, "bench": str(bench)},
    )
    report.to_json(out_path)
    typer.echo(f"recall@5={report.overall.recall_at.get(5, 0.0):.3f} mrr={report.overall.mrr:.3f}")
    typer.echo(f"rapor -> {out_path}")
    if report.missing_gold_pages:
        typer.echo(f"missing_gold_pages={len(report.missing_gold_pages)}")


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
