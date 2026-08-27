import hashlib
import json
from pathlib import Path

from belge_gozu.index.manifest import (
    CPE_0_3_18,
    CPE_0_3_18_DOC_PROMPT,
    TRAIN_COMPAT_DOC_PROMPT,
    TRAIN_COMPAT_V1,
    IndexManifest,
    RenderConfig,
    corpus_checksum,
    read_manifest,
    write_manifest,
)

TRAIN_COMPAT_DOC_PROMPT_SHA256 = hashlib.sha256(TRAIN_COMPAT_DOC_PROMPT.encode()).hexdigest()


def make_manifest(**over) -> IndexManifest:
    """Fikstür manifest'i — varsayılanları ÜRETİM değerleridir.

    Final review IMPORTANT-2: eskiden burada `CPE_0_3_18` + uydurma bir
    doc_prompt sha'sı vardı. `create_app`'ın uyumluluk kontrolü artık
    config'ten çözülen değerlere düştüğü için (enjekte edilen encoder'ın
    `query_format`/`doc_prompt_sha256`'i yoksa), fikstürün de üretimin
    kullandığı formatı taşıması gerekiyor — aksi halde kontrol ya ölü kalır
    ya da her testte sahte bir uyumsuzluk üretir.
    """
    base = dict(
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        engine_versions={"colpali-engine": "0.3.18", "transformers": "5.15.1", "torch": "2.13.0"},
        query_format=TRAIN_COMPAT_V1,
        doc_prompt_sha256=TRAIN_COMPAT_DOC_PROMPT_SHA256,
        quantization="sign-1bit",
        mask_policy="drop-padding",
        render=RenderConfig(),
        corpus_checksum="c" * 64,
        n_pages=3,
        n_tokens=24,
        built_at="2026-08-26T00:00:00+00:00",
        git_commit="deadbeef",
    )
    base.update(over)
    return IndexManifest(**base)


def test_make_manifest_defaults_track_production_config():
    """Fikstür ile serve config'i birlikte sürüklenmeli (final review IMPORTANT-2).

    Bu kilit olmadan `Settings` varsayılanı değişince fikstür sessizce eski
    formatta kalır ve `create_app`'ın uyumluluk kontrolü ya ölür ya da tüm
    app testlerini kopyala-yapıştır bir literal'e mahkûm eder."""
    from belge_gozu.config import Settings

    s = Settings()
    m = make_manifest()
    assert m.query_format.format_id == s.query_format_id
    assert m.doc_prompt_sha256 == TRAIN_COMPAT_DOC_PROMPT_SHA256
    assert s.doc_prompt_id == "train-compat"


def test_query_format_render():
    assert CPE_0_3_18.render("soru") == "soru" + "<end_of_utterance>" * 10
    # T11/Step 1: sentence_transformers.jinja task=='query' dalıyla birebir
    # ('Query: ' + text + augmentation*10 + '\n' — newline EN SONDA).
    assert TRAIN_COMPAT_V1.render("soru") == "Query: soru" + "<end_of_utterance>" * 10 + "\n"


def test_doc_prompt_constants_are_verbatim():
    """T11/Step 1'de kilitlenen iki doküman prompt'u (kaynak: model reposunun
    sentence_transformers.jinja image dalı vs. colpali-engine 0.3.18 ClassVar)."""
    assert TRAIN_COMPAT_DOC_PROMPT == (
        "<|im_start|>User: Describe the image.<image><end_of_utterance>"
    )
    assert CPE_0_3_18_DOC_PROMPT == (
        "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )
    assert TRAIN_COMPAT_DOC_PROMPT != CPE_0_3_18_DOC_PROMPT


def test_roundtrip(tmp_path: Path):
    m = make_manifest()
    write_manifest(tmp_path, m)
    m2 = read_manifest(tmp_path)
    assert m2 == m
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 1


def test_read_missing_returns_none(tmp_path: Path):
    assert read_manifest(tmp_path) is None


def test_corpus_checksum_changes_with_content(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    c1 = corpus_checksum(tmp_path)
    (tmp_path / "meta.parquet").write_bytes(b"y")
    assert corpus_checksum(tmp_path) != c1
