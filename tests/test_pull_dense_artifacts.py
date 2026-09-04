from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "pull_dense_artifacts", REPO / "scripts" / "pull_dense_artifacts.py"
)
assert _spec and _spec.loader
puller = importlib.util.module_from_spec(_spec)
sys.modules["pull_dense_artifacts"] = puller
_spec.loader.exec_module(puller)

REVISION = "a" * 40


def test_pull_cli_passes_local_page_identity_and_prints_resolved_sha(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_pull(**kwargs: object) -> str:
        captured.update(kwargs)
        return REVISION

    monkeypatch.setattr(puller, "load_page_texts", lambda _: {"p1": "metin"})
    monkeypatch.setattr(puller, "sha256_file", lambda _: "b" * 64)
    monkeypatch.setattr(puller, "pull_dense_artifact", fake_pull)

    result = puller.main(
        [
            "--repo",
            "user/repo",
            "--revision",
            REVISION,
            "--model",
            "qwen3-embedding-4b",
            "--index-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert captured["revision"] == REVISION
    assert capsys.readouterr().out.strip().endswith(f"commit={REVISION}")
