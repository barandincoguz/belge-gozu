from pydantic import BaseModel, Field

EVENTS_DDL = """CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  status TEXT NOT NULL,
  http_status INTEGER NOT NULL,
  total_ms REAL NOT NULL,
  encode_ms REAL, stage1_ms REAL, stage2_ms REAL, answer_ms REAL,
  top_score REAL, margin_1_2 REAL,
  abstained INTEGER, honest_miss INTEGER,
  k INTEGER, candidates INTEGER,
  query_len INTEGER NOT NULL DEFAULT 0,
  query_text TEXT,
  query_sha256 TEXT NOT NULL,
  answer_len INTEGER, citations_n INTEGER,
  tokens_in INTEGER, tokens_out INTEGER, tokens_per_s REAL, est_cost_usd REAL,
  error_type TEXT,
  pipeline TEXT,
  index_revision TEXT,
  detail TEXT NOT NULL DEFAULT '{}'
)"""

EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_endpoint_ts ON events(endpoint, ts)",
]


class RequestEvent(BaseModel):
    """Tek isteğin ham olay kaydı. Spec §5'in birebir karşılığı."""

    ts: str
    endpoint: str
    status: str
    http_status: int
    total_ms: float
    encode_ms: float | None = None
    stage1_ms: float | None = None
    stage2_ms: float | None = None
    answer_ms: float | None = None
    top_score: float | None = None
    margin_1_2: float | None = None
    abstained: bool | None = None
    honest_miss: bool | None = None
    k: int | None = None
    candidates: int | None = None
    query_len: int = 0
    query_text: str | None = None
    query_sha256: str = ""
    answer_len: int | None = None
    citations_n: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_per_s: float | None = None
    est_cost_usd: float | None = None
    error_type: str | None = None
    pipeline: str | None = None
    index_revision: str | None = None
    detail: dict = Field(default_factory=dict)
