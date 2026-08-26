import threading

from belge_gozu.telemetry.collect import annotate, collecting, stage


def test_stage_noop_without_collector():
    with stage("query_encode"):
        pass  # kolektör yok; hata da yok, kayıt da yok
    annotate("tokens_in", 5)  # sessiz no-op


def test_collecting_captures_stages_and_notes():
    with collecting() as col:
        with stage("query_encode"):
            pass
        with stage("answerer"):
            annotate("tokens_out", 42)
    assert set(col.stages) == {"query_encode", "answerer"}
    assert col.stages["query_encode"] >= 0.0  # ms
    assert col.notes == {"tokens_out": 42}


def test_collectors_are_isolated_across_threads():
    seen: dict[str, set[str]] = {}

    def worker(name: str):
        with collecting() as col:
            with stage(name):
                pass
            seen[name] = set(col.stages)

    ts = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen == {f"s{i}": {f"s{i}"} for i in range(4)}
