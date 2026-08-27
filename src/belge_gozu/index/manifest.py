import hashlib
from enum import StrEnum
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
# T11/Step 1'de birincil kaynaktan doğrulandı (sapma çıkmadı):
# vidore/colSmol-500M @ 0aaa972 `additional_chat_templates/sentence_transformers.jinja`
# task=='query' dalı: 'Query: ' + text + '<end_of_utterance>'*10 + '\n'
# (model kartı: "The Sentence Transformers configuration in this repository
# reproduces the original training-time format"). Aynı dizi colpali-engine
# 0.3.8/0.3.9 `process_queries` gövdesinde de birebir var; newline 0.3.11'de,
# "Query: " prefix'i 0.3.13'te düştü.
TRAIN_COMPAT_V1 = QueryFormat(
    format_id="train-compat-v1",
    prefix="Query: ",
    suffix_token="<end_of_utterance>",
    n_suffix=10,
    trailing_newline=True,
)

# Doküman tarafı sorgudan farklı: burada sapma VAR.
# Eğitim zamanı (colpali-engine 0.3.8 `process_images` -> repo chat_template.jinja,
# messages=[text "Describe the image.", image], .strip()) ve ST jinja'sının image
# dalı aynı diziyi üretiyor: metin ÖNCE, <image> SONRA, "\nAssistant:" kuyruğu yok.
TRAIN_COMPAT_DOC_PROMPT = "<|im_start|>User: Describe the image.<image><end_of_utterance>"
# Yüklü colpali-engine 0.3.18'in ColIdefics3Processor.visual_prompt_prefix ClassVar'ı
# (0.3.11'de bu hale geldi): <image> metinden önce + "\nAssistant:" eklendi.
CPE_0_3_18_DOC_PROMPT = "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"


class QueryFormatChoice(StrEnum):
    cpe_0_3_18 = "cpe-0.3.18"
    train_compat_v1 = "train-compat-v1"


# CLI (--query-format) ve serve config'i (Settings.query_format_id) TEK bir
# sözlükten okur (T11/Step 6): iki kopya literal'in sürüklenmesini önler.
QUERY_FORMATS: dict[QueryFormatChoice, QueryFormat] = {
    QueryFormatChoice.cpe_0_3_18: CPE_0_3_18,
    QueryFormatChoice.train_compat_v1: TRAIN_COMPAT_V1,
}


class DocPromptChoice(StrEnum):
    """Doküman prompt'u sorgu formatından bağımsız seçilir: T11 A/B'sinde iki
    eksen ayrı ayrı denenebilsin diye. Varsayılan = processor'ın kendi ClassVar'ı
    (mevcut davranış); `train-compat` T11/Step 1'de kilitlenen eğitim zamanı dizisi."""

    processor_default = "processor-default"
    train_compat = "train-compat"


DOC_PROMPTS: dict[DocPromptChoice, str | None] = {
    DocPromptChoice.processor_default: None,
    DocPromptChoice.train_compat: TRAIN_COMPAT_DOC_PROMPT,
}


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
