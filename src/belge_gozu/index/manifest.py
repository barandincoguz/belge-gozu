import hashlib
from pathlib import Path

from pydantic import BaseModel


class QueryFormat(BaseModel):
    format_id: str
    prefix: str
    suffix_token: str
    n_suffix: int
    trailing_newline: bool

    def render(self, text: str) -> str:
        out = self.prefix + text + self.suffix_token * self.n_suffix
        return out + "\n" if self.trailing_newline else out


CPE_0_3_18 = QueryFormat(
    format_id="cpe-0.3.18",
    prefix="",
    suffix_token="<end_of_utterance>",
    n_suffix=10,
    trailing_newline=False,
)
# Model kartı: checkpoint "Query: " prefix + sondaki newline ile eğitildi
# (newline 0.3.11'de, prefix 0.3.13'te düştü). Kesin şablon T11'de
# config_sentence_transformers.json'a karşı doğrulanır; sapma varsa bu sabit
# orada düzeltilir ve test_query_format_render güncellenir.
TRAIN_COMPAT_V1 = QueryFormat(
    format_id="train-compat-v1",
    prefix="Query: ",
    suffix_token="<end_of_utterance>",
    n_suffix=10,
    trailing_newline=True,
)


class RenderConfig(BaseModel):
    dpi: int = 150
    format: str = "webp"
    quality: int = 80


class IndexManifest(BaseModel):
    schema_version: int = 1
    model_name: str
    model_revision: str
    engine_versions: dict[str, str]
    query_format: QueryFormat
    doc_prompt_sha256: str
    quantization: str
    mask_policy: str
    render: RenderConfig
    corpus_checksum: str
    n_pages: int
    n_tokens: int
    built_at: str
    git_commit: str


def corpus_checksum(index_dir: Path) -> str:
    h = hashlib.sha256()
    h.update((index_dir / "page_ids.json").read_bytes())
    h.update((index_dir / "meta.parquet").read_bytes())
    return h.hexdigest()


def write_manifest(dir: Path, m: IndexManifest) -> None:
    (dir / "manifest.json").write_text(m.model_dump_json(indent=1), encoding="utf-8")


def read_manifest(dir: Path) -> IndexManifest | None:
    p = dir / "manifest.json"
    if not p.exists():
        return None
    return IndexManifest.model_validate_json(p.read_text(encoding="utf-8"))
