import shutil
from pathlib import Path

import pandas as pd
import typer
from PIL import Image

from belge_gozu.config import Settings
from belge_gozu.corpus.download import download_all
from belge_gozu.corpus.manifest import build_http_client, load_manifest, probe
from belge_gozu.corpus.render import render_all
from belge_gozu.index.encode import FakeEncoder
from belge_gozu.index.store import PackedIndex

app = typer.Typer(help="Belge-Gözü: Türkçe mevzuat için görsel belge RAG")
corpus_app = typer.Typer()
index_app = typer.Typer()
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")

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
    batch_size = 8
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
