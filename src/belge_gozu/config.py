from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BG_")

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    retriever_model: str = "vidore/colSmol-500M"
    device: str = "auto"
    hf_dataset_repo: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = ""
    stage1_candidates: int = 200
    top_k: int = 5
    # Uncalibrated in v0; calibrated against the benchmark in Plan 2 (spec §6).
    min_score_threshold: float = 20.0
    request_delay_s: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
