"""Apple Silicon üzerinde dense artefakt üretimi — saf orkestrasyon testleri.

Model yüklenmez, GPU'ya dokunulmaz, ağ çağrısı yapılmaz: burada sınanan şey
bütçe kapısı, süreç yalıtımı ve "doğrulamadan yayımlama" yasağıdır.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_dense_artifacts_local", REPO / "scripts" / "build_dense_artifacts_local.py"
)
assert _spec and _spec.loader
local = importlib.util.module_from_spec(_spec)
sys.modules["build_dense_artifacts_local"] = local
_spec.loader.exec_module(local)

REVISION = "a" * 40
GIB = 1024**3


class _Runner:
    """`build_dense_artifacts.py` alt sürecinin yerine geçen kayıt tutucu."""

    def __init__(self, statuses: list[dict[str, object]]) -> None:
        self.statuses = statuses
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> dict[str, object]:
        self.commands.append(command)
        return self.statuses[len(self.commands) - 1]


def _argv(*extra: str, revision: str = REVISION) -> list[str]:
    return ["--index-dir", "index", "--source-revision", revision, *extra]


def _stub(
    monkeypatch: pytest.MonkeyPatch, runner: _Runner, *, order: list[str] | None = None
) -> list[str]:
    pushed: list[str] = []

    def _validate(artifact_dir: Path, expectation: object) -> dict[str, object]:
        if order is not None:
            order.append("validate")
        return {"schema_version": 1}

    def _push(
        artifact_dir: Path, repo_id: str, model_key: str, expectation: object, **_: object
    ) -> str:
        if order is not None:
            order.append("push")
        pushed.append(model_key)
        return "b" * 40

    monkeypatch.setattr(local, "run_model_build", runner)
    monkeypatch.setattr(local, "mps_budget_bytes", lambda torch_module=None: 72 * GIB)
    monkeypatch.setattr(local, "load_expectation", lambda spec, index_dir: object())
    monkeypatch.setattr(local, "validate_dense_artifact", _validate)
    monkeypatch.setattr(local, "push_dense_artifact", _push)
    return pushed


# --------------------------------------------------------------------------
# bütçe kapısı — HİÇBİR model kodlanmadan önce
# --------------------------------------------------------------------------


def test_refuses_the_unfit_model_before_any_encoding_starts() -> None:
    """24 GB'lık makinede 8B'yi saatlerce koşup sonda ölmek kabul edilemez."""
    with pytest.raises(local.InsufficientUnifiedMemory) as excinfo:
        local.require_models_fit(["qwen3-embedding-4b", "qwen3-embedding-8b"], 17 * GIB)

    message = str(excinfo.value)
    assert "qwen3-embedding-8b" in message
    assert "qwen3-embedding-4b" not in message


def test_mac_studio_budget_admits_both_models() -> None:
    local.require_models_fit(["qwen3-embedding-4b", "qwen3-embedding-8b"], 72 * GIB)


def test_budget_probe_rejects_a_machine_without_metal() -> None:
    class _NoMetal:
        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return False

    with pytest.raises(local.UnsupportedDevice):
        local.mps_budget_bytes(_NoMetal())


def test_budget_probe_reads_the_torch_working_set_not_total_ram() -> None:
    """Bu makinede toplam 24 GiB ama PyTorch'un bütçesi 17,8 GiB; karar bütçenin."""

    class _Metal:
        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return True

        class mps:
            @staticmethod
            def recommended_max_memory() -> int:
                return 72 * GIB

    assert local.mps_budget_bytes(_Metal()) == 72 * GIB


# --------------------------------------------------------------------------
# süreç yalıtımı
# --------------------------------------------------------------------------


def test_each_model_is_encoded_in_its_own_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """MPS 8B ağırlıklarını süreç içinde tam bırakmıyor; yalıtım şart."""
    runner = _Runner([{"status": "ok", "artifact": "a"}, {"status": "ok", "artifact": "b"}])
    _stub(monkeypatch, runner)

    assert local.main(_argv("--batch-size", "8")) == 0

    assert len(runner.commands) == 2
    first, second = runner.commands
    assert first[first.index("--model") + 1] == "qwen3-embedding-4b"
    assert second[second.index("--model") + 1] == "qwen3-embedding-8b"
    assert first[first.index("--batch-size") + 1] == "8"
    assert first[first.index("--source-revision") + 1] == REVISION


# --------------------------------------------------------------------------
# yayımlama disiplini
# --------------------------------------------------------------------------


def test_incomplete_model_is_never_published(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _Runner([{"status": "in_progress", "model": "qwen3-embedding-4b"}])
    pushed = _stub(monkeypatch, runner)

    assert local.main(_argv("--model", "qwen3-embedding-4b", "--push")) == 0
    assert pushed == []


def test_publication_validates_the_local_artifact_first(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _Runner([{"status": "ok", "artifact": "artifacts/qwen3-embedding-4b"}])
    order: list[str] = []
    pushed = _stub(monkeypatch, runner, order=order)

    assert local.main(_argv("--model", "qwen3-embedding-4b", "--push")) == 0

    assert order == ["validate", "push"]
    assert pushed == ["qwen3-embedding-4b"]


def test_completed_model_is_validated_even_without_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _Runner([{"status": "ok", "artifact": "artifacts/qwen3-embedding-4b"}])
    order: list[str] = []
    _stub(monkeypatch, runner, order=order)

    assert local.main(_argv("--model", "qwen3-embedding-4b")) == 0
    assert order == ["validate"]


def test_out_of_memory_is_reported_as_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _Runner([{"status": "skipped_oom", "model": "qwen3-embedding-4b"}])
    _stub(monkeypatch, runner)

    assert local.main(_argv("--model", "qwen3-embedding-4b")) == 2


# --------------------------------------------------------------------------
# köken kimliği
# --------------------------------------------------------------------------


def test_source_revision_must_be_an_immutable_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, _Runner([]))

    with pytest.raises(ValueError, match="40"):
        local.main(_argv(revision="main"))


def test_defaults_cover_both_models_and_the_project_datasets() -> None:
    args = local._parse_args(_argv())

    assert args.model == ["qwen3-embedding-4b", "qwen3-embedding-8b"]
    assert args.source_repo == "barandincoguz/belge-gozu-index"
    assert args.artifact_repo == "barandincoguz/belge-gozu-semantic-artifacts"
