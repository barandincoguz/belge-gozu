import numpy as np
import pytest
import torch

from belge_gozu.retrieval.dense import (
    DENSE_MODELS,
    DenseModelOutOfMemory,
    DensePageIndex,
    TransformerDenseEncoder,
    format_query,
    last_token_pool,
    model_load_kwargs,
)


def test_dense_index_returns_stable_descending_page_ids() -> None:
    index = DensePageIndex(["p1", "p2", "p3"], np.eye(3, dtype=np.float32))

    assert index.candidate_pages(np.array([0.1, 0.9, 0.9], dtype=np.float32), limit=2) == [
        "p2",
        "p3",
    ]


def test_dense_index_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="benzersiz"):
        DensePageIndex(["p1", "p1"], np.eye(2, dtype=np.float32))


def test_dense_index_rejects_bad_query_shape() -> None:
    index = DensePageIndex(["p1", "p2"], np.eye(2, dtype=np.float32))

    with pytest.raises(ValueError, match="boyutu"):
        index.candidate_pages(np.ones(3, dtype=np.float32))


def test_qwen_query_uses_fixed_instruction_without_changing_passage_text() -> None:
    spec = DENSE_MODELS["qwen3-embedding-8b"]

    assert format_query(spec, "İzin süresi nedir?") == (
        "Instruct: Given a Turkish legal search query, retrieve relevant passages that answer "
        "the query.\nQuery:İzin süresi nedir?"
    )


def test_last_token_pool_selects_each_sequences_final_non_padding_token() -> None:
    states = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    assert torch.equal(last_token_pool(states, mask), torch.tensor([[2.0, 0.0], [6.0, 0.0]]))


def test_encoder_instructs_queries_but_not_passages_and_normalizes_vectors() -> None:
    tokenizer = _FakeTokenizer()
    encoder = TransformerDenseEncoder(
        DENSE_MODELS["qwen3-embedding-4b"],
        tokenizer=tokenizer,
        model=_FakeModel(),
        torch_module=torch,
        device="cpu",
    )

    query = encoder.encode_queries(["İzin süresi nedir?"])
    passage = encoder.encode_passages(["Yıllık izin süresi..."])

    assert tokenizer.calls == [
        [
            "Instruct: Given a Turkish legal search query, retrieve relevant passages that "
            "answer the query.\nQuery:İzin süresi nedir?"
        ],
        ["Yıllık izin süresi..."],
    ]
    assert np.allclose(np.linalg.norm(query, axis=1), [1.0])
    assert np.allclose(np.linalg.norm(passage, axis=1), [1.0])


def test_encoder_preflight_exposes_device_oom() -> None:
    encoder = TransformerDenseEncoder(
        DENSE_MODELS["qwen3-embedding-4b"],
        tokenizer=_FakeTokenizer(),
        model=_OomModel(),
        torch_module=torch,
        device="cpu",
    )

    with pytest.raises(DenseModelOutOfMemory):
        encoder.preflight()


def test_mps_dense_loader_uses_float16_weights() -> None:
    assert model_load_kwargs("mps", torch)["torch_dtype"] is torch.float16
    assert model_load_kwargs("cpu", torch) == {}


def test_cuda_dense_loader_uses_float16_weights() -> None:
    assert model_load_kwargs("cuda", torch)["torch_dtype"] is torch.float16


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        self.calls.append(texts)
        values = torch.ones((len(texts), 2), dtype=torch.int64)
        return {"input_ids": values, "attention_mask": values}


class _FakeModel:
    def to(self, _: str) -> "_FakeModel":
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **inputs: torch.Tensor) -> object:
        batch = inputs["input_ids"].shape[0]
        states = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]] * batch)
        return type("Output", (), {"last_hidden_state": states})()


class _OomModel(_FakeModel):
    def __call__(self, **inputs: torch.Tensor) -> object:
        raise RuntimeError("MPS backend out of memory")
