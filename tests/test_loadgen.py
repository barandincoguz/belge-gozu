import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from loadgen import summarize  # noqa: E402


def test_summarize_percentiles():
    lat = [float(i) for i in range(1, 101)]  # 1..100 ms
    s = summarize(lat, errors=2, duration_s=10.0)
    assert s["requests"] == 100 and s["errors"] == 2
    assert s["rps"] == 10.0
    assert s["p50_ms"] == 50.0 and s["p95_ms"] == 95.0 and s["p99_ms"] == 99.0


def test_summarize_empty():
    s = summarize([], errors=0, duration_s=1.0)
    assert s["requests"] == 0 and s["p95_ms"] == 0.0
