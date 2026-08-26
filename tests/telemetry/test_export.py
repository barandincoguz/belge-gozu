import pandas as pd

from belge_gozu.telemetry.export import export_events
from belge_gozu.telemetry.recorder import EventRecorder
from belge_gozu.telemetry.schema import RequestEvent


def test_export_parquet_roundtrip(tmp_path):
    rec = EventRecorder(tmp_path / "r.sqlite")
    rec.record(
        RequestEvent(
            ts="t1",
            endpoint="/search",
            status="ok",
            http_status=200,
            total_ms=5.0,
            query_sha256="e" * 64,
        )
    )
    rec.close()
    out = tmp_path / "events.parquet"
    n = export_events(tmp_path / "r.sqlite", out)
    assert n == 1
    df = pd.read_parquet(out)
    assert list(df["endpoint"]) == ["/search"] and "total_ms" in df.columns


def test_export_csv(tmp_path):
    rec = EventRecorder(tmp_path / "r.sqlite")
    rec.record(
        RequestEvent(
            ts="t1",
            endpoint="/ask",
            status="answered",
            http_status=200,
            total_ms=5.0,
            query_sha256="e" * 64,
        )
    )
    rec.close()
    n = export_events(tmp_path / "r.sqlite", tmp_path / "events.csv")
    assert n == 1 and (tmp_path / "events.csv").read_text().startswith("id,")
