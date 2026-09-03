from pathlib import Path

import pytest
import torch

from belge_gozu.retrieval.expand import (
    EXPANDER_REVISION,
    ExpansionRecord,
    LocalQueryExpander,
    load_expansion_cache,
    prompt_fingerprint,
    validate_expansion,
    write_expansion_cache,
)


def test_expansion_rejects_empty_and_identity_text() -> None:
    with pytest.raises(ValueError, match="boş"):
        validate_expansion("İzin süresi nedir?", "  ")
    with pytest.raises(ValueError, match="özgün"):
        validate_expansion("İzin süresi nedir?", "İzin süresi nedir?")


def test_cache_round_trip_requires_active_provenance(tmp_path: Path) -> None:
    path = tmp_path / "expansions.jsonl"
    record = ExpansionRecord(
        question_id="q1",
        question_sha256="a" * 64,
        prompt_fingerprint=prompt_fingerprint(),
        model_revision=EXPANDER_REVISION,
        expansion="yıllık izin hakkı süre koşulları",
    )

    write_expansion_cache(path, [record])

    assert load_expansion_cache(path) == {"q1": record}


def test_expander_uses_non_thinking_deterministic_qwen_call() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    expander = LocalQueryExpander(
        tokenizer=tokenizer, model=model, torch_module=torch, device="cpu"
    )

    assert expander.expand("Yıllık izin süresi nedir?") == "yıllık izin hakkı süre koşulları"
    assert tokenizer.template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert model.generate_kwargs == {"do_sample": False, "max_new_tokens": 64}


class _FakeTokenizer:
    def __init__(self) -> None:
        self.template_kwargs: dict[str, object] | None = None

    def apply_chat_template(self, _: list[dict[str, str]], **kwargs: object) -> str:
        self.template_kwargs = kwargs
        return "prompt"

    def __call__(self, _: str, **__: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.int64)}

    def decode(self, _: torch.Tensor, **__: object) -> str:
        return " yıllık izin hakkı süre koşulları "


class _FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, object] | None = None

    def to(self, _: str) -> "_FakeModel":
        return self

    def eval(self) -> None:
        return None

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_kwargs = {key: value for key, value in kwargs.items() if key != "input_ids"}
        return torch.tensor([[1, 2, 3]], dtype=torch.int64)
