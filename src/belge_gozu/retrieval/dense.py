"""Offline Qwen3 dense sayfa indeksinin saf, modelden bağımsız kısmı."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DenseModelSpec:
    repo: str
    revision: str
    instruction: str
    max_length: int


DENSE_MODELS = {
    "qwen3-embedding-8b": DenseModelSpec(
        repo="Qwen/Qwen3-Embedding-8B",
        revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        instruction=(
            "Given a Turkish legal search query, retrieve relevant passages that answer the query."
        ),
        max_length=8192,
    ),
    "qwen3-embedding-4b": DenseModelSpec(
        repo="Qwen/Qwen3-Embedding-4B",
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        instruction=(
            "Given a Turkish legal search query, retrieve relevant passages that answer the query."
        ),
        max_length=8192,
    ),
}


class DenseModelOutOfMemory(RuntimeError):
    """Dense checkpoint'in gerçek cihaz preflight'ında sığmadığını bildirir."""


def release_transformer_memory(torch_module: Any) -> None:
    """Sıralı offline kollar arasındaki PyTorch/MPS artıklarını bırakır."""
    gc.collect()
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        torch_module.mps.empty_cache()


def model_load_kwargs(device: str, torch_module: Any) -> dict[str, Any]:
    return {"torch_dtype": torch_module.float16} if device == "mps" else {}


def format_query(spec: DenseModelSpec, question: str) -> str:
    """Qwen3'ün instruction-aware retrieval girdi biçimi."""
    return f"Instruct: {spec.instruction}\nQuery:{question}"


def last_token_pool(last_hidden_states: object, attention_mask: object) -> object:
    """Qwen3 kartındaki sağ/sol dolguyla uyumlu son-token pooling."""
    import torch

    states = torch.as_tensor(last_hidden_states)
    mask = torch.as_tensor(attention_mask, device=states.device)
    if states.ndim != 3 or mask.shape != states.shape[:2]:
        raise ValueError("hidden state ve attention mask şekilleri uyuşmalı")
    if bool(torch.all(mask[:, -1])):
        return states[:, -1]
    lengths = mask.sum(dim=1) - 1
    return states[torch.arange(states.shape[0], device=states.device), lengths]


class TransformerDenseEncoder:
    """Sabit Qwen3 checkpoint'iyle sorgu ve sayfa vektörleri üretir."""

    def __init__(
        self,
        spec: DenseModelSpec,
        *,
        device: str | None = None,
        batch_size: int = 8,
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("dense batch_size en az 1 olmalı")
        if (tokenizer is None) != (model is None):
            raise ValueError("dense tokenizer ve model birlikte verilmelidir")

        if torch_module is None:
            import torch

            torch_module = torch
        self.spec = spec
        self.batch_size = batch_size
        self._torch = torch_module
        self._device = device or (
            "mps" if self._torch.backends.mps.is_available() else "cpu"
        )
        if tokenizer is None:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                spec.repo, revision=spec.revision, padding_side="left"
            )
            model = AutoModel.from_pretrained(
                spec.repo, revision=spec.revision, **model_load_kwargs(self._device, self._torch)
            )
        assert tokenizer is not None and model is not None
        self._tokenizer = tokenizer
        self._model = model.to(self._device)
        self._model.eval()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self._tokenizer(
                    list(texts[start : start + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                )
                inputs = {name: value.to(self._device) for name, value in encoded.items()}
                outputs = self._model(**inputs)
                pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(normalized.detach().cpu().numpy())
        return np.concatenate(vectors).astype(np.float32, copy=False)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([format_query(self.spec, text) for text in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def preflight(self) -> None:
        try:
            self.encode_queries(["Türk hukukunda yıllık izin süresi nedir?"])
        except RuntimeError as exc:
            if "out of memory" not in str(exc).casefold():
                raise
            release_transformer_memory(self._torch)
            raise DenseModelOutOfMemory(
                f"dense model preflight belleğe sığmadı: {self.spec.repo}@{self.spec.revision}"
            ) from exc


def _normalized_vector(vector: np.ndarray, dimension: int) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.shape != (dimension,):
        raise ValueError(f"sorgu vektörü boyutu {dimension} olmalı: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("sorgu vektörü sonlu olmalı")
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError("sorgu vektörü sıfır olamaz")
    return value / norm


class DensePageIndex:
    """L2-normalize sayfa vektörleri üzerinde kararlı kosinüs top-k araması."""

    def __init__(self, page_ids: Sequence[str], embeddings: np.ndarray) -> None:
        self.page_ids = tuple(page_ids)
        if not self.page_ids:
            raise ValueError("dense indeks boş olamaz")
        if len(set(self.page_ids)) != len(self.page_ids):
            raise ValueError("dense indeks page_ids benzersiz olmalı")
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(self.page_ids) or values.shape[1] == 0:
            raise ValueError("dense embeddings page_ids ile hizalı iki boyutlu olmalı")
        if not np.isfinite(values).all():
            raise ValueError("dense embeddings sonlu olmalı")
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("dense embeddings sıfır vektör içeremez")
        self.embeddings = values / norms[:, None]

    def candidate_pages(self, query_embedding: np.ndarray, limit: int = 50) -> list[str]:
        if limit < 1:
            raise ValueError("dense aday limiti en az 1 olmalı")
        query = _normalized_vector(query_embedding, self.embeddings.shape[1])
        order = np.argsort(-(self.embeddings @ query), kind="stable")
        return [self.page_ids[int(index)] for index in order[:limit]]
