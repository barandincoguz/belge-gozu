from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    retriever_model: str = "vidore/colSmol-500M"
    device: str = "auto"
    hf_dataset_repo: str = ""
    # gemini-2.0-flash Task 13'te canlı çağrıda 404 döndü (API: "no longer
    # available... use models/gemini-3.6-flash"); Task 13 canlı doğrulamasında güncellendi.
    gemini_model: str = "gemini-3.6-flash"
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("BG_GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    stage1_candidates: int = 200
    top_k: int = 5
    # Uncalibrated in v0; calibrated against the benchmark in Plan 2 (spec §6).
    min_score_threshold: float = 20.0
    request_delay_s: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
