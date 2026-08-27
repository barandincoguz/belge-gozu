from pathlib import Path

import pytest
from tests.index.test_manifest import make_manifest
from typer.testing import CliRunner

from belge_gozu.cli import app
from belge_gozu.index.compat import IndexCompatibilityError, check_compatibility

runner = CliRunner()


def test_missing_manifest_is_mismatch(tmp_path: Path):
    problems = check_compatibility(
        None,
        model_name="m",
        model_revision="r",
        query_format_id="cpe-0.3.18",
        index_dir=tmp_path,
    )
    assert problems and "manifest" in problems[0]


def test_matching_manifest_ok(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    from belge_gozu.index.manifest import corpus_checksum

    m = make_manifest(corpus_checksum=corpus_checksum(tmp_path))
    assert (
        check_compatibility(
            m,
            model_name="vidore/colSmol-500M",
            model_revision="abc123",
            query_format_id="cpe-0.3.18",
            index_dir=tmp_path,
        )
        == []
    )


def test_format_mismatch_reported(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    from belge_gozu.index.manifest import corpus_checksum

    m = make_manifest(corpus_checksum=corpus_checksum(tmp_path))
    problems = check_compatibility(
        m,
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        query_format_id="train-compat-v1",
        index_dir=tmp_path,
    )
    assert any("query_format" in p for p in problems)


def test_matching_doc_prompt_ok(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    from belge_gozu.index.manifest import corpus_checksum

    m = make_manifest(corpus_checksum=corpus_checksum(tmp_path), doc_prompt_sha256="d" * 64)
    problems = check_compatibility(
        m,
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        query_format_id="cpe-0.3.18",
        doc_prompt_sha256="d" * 64,
        index_dir=tmp_path,
    )
    assert problems == []


def test_doc_prompt_mismatch_reported(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    from belge_gozu.index.manifest import corpus_checksum

    m = make_manifest(corpus_checksum=corpus_checksum(tmp_path), doc_prompt_sha256="d" * 64)
    problems = check_compatibility(
        m,
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        query_format_id="cpe-0.3.18",
        doc_prompt_sha256="e" * 64,
        index_dir=tmp_path,
    )
    assert any("doc_prompt" in p for p in problems)


def test_doc_prompt_none_or_unknown_skips_check(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    from belge_gozu.index.manifest import corpus_checksum

    m = make_manifest(corpus_checksum=corpus_checksum(tmp_path), doc_prompt_sha256="d" * 64)
    for value in (None, "unknown"):
        problems = check_compatibility(
            m,
            model_name="vidore/colSmol-500M",
            model_revision="abc123",
            query_format_id="cpe-0.3.18",
            doc_prompt_sha256=value,
            index_dir=tmp_path,
        )
        assert not any("doc_prompt" in p for p in problems)


def test_create_app_fails_fast_on_mismatch(tiny_corpus):
    """tiny_corpus indeksinin manifest'i kasten silinir -> create_app IndexCompatibilityError."""
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    (data_dir / "index" / "manifest.json").unlink()
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index")
    with pytest.raises(IndexCompatibilityError):
        create_app(settings=settings, encoder=enc, answerer=object())


def test_mismatch_override(tiny_corpus):
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    (data_dir / "index" / "manifest.json").unlink()
    settings = Settings(
        data_dir=data_dir,
        index_dir=data_dir / "index",
        allow_index_mismatch=True,
        min_score_threshold=-1e9,
    )
    app_ = create_app(settings=settings, encoder=enc, answerer=object())
    assert app_ is not None


def test_write_manifest_legacy_cli(tiny_corpus, monkeypatch):
    data_dir, _, _ = tiny_corpus
    (data_dir / "index" / "manifest.json").unlink()
    monkeypatch.setenv("BG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BG_INDEX_DIR", str(data_dir / "index"))

    result = runner.invoke(app, ["index", "write-manifest", "--legacy"])

    assert result.exit_code == 0, result.output
    from belge_gozu.index.manifest import read_manifest

    m = read_manifest(data_dir / "index")
    assert m is not None
    assert m.mask_policy == "none"
