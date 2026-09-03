"""Offline Qwen3 sorgu genişletmesi ve denetlenebilir JSONL önbelleği."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from belge_gozu.retrieval.dense import release_transformer_memory

EXPANDER_REPO = "Qwen/Qwen3-8B"
EXPANDER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPANSION_PROMPT = (
    "Tek satır Türkçe hukukî arama varyantı yaz. Anlamı koru; cevap, delil, "
    "madde numarası veya olmayan kanun adı uydurma. Yalnız varyantı yaz."
)


@dataclass(frozen=True)
class ExpansionRecord:
    question_id: str
    question_sha256: str
    prompt_fingerprint: str
    model_revision: str
    expansion: str


class ExpansionModelOutOfMemory(RuntimeError):
    """Qwen3 genişletme checkpoint'i cihaz preflight'ına sığmadı."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_fingerprint() -> str:
    return _sha256(EXPANSION_PROMPT)


def question_fingerprint(question: str) -> str:
    return _sha256(question)


def validate_expansion(question: str, expansion: str) -> str:
    value = " ".join(expansion.split())
    if not value:
        raise ValueError("genişletme boş")
    if value.casefold() == " ".join(question.split()).casefold():
        raise ValueError("genişletme özgün sorguyla aynı")
    return value


def _record(payload: Any) -> ExpansionRecord:
    if not isinstance(payload, dict):
        raise ValueError("genişletme önbellek kaydı nesne olmalı")
    try:
        record = ExpansionRecord(
            question_id=str(payload["question_id"]),
            question_sha256=str(payload["question_sha256"]),
            prompt_fingerprint=str(payload["prompt_fingerprint"]),
            model_revision=str(payload["model_revision"]),
            expansion=str(payload["expansion"]),
        )
    except KeyError as exc:
        raise ValueError(f"genişletme önbellek alanı eksik: {exc.args[0]}") from exc
    if record.prompt_fingerprint != prompt_fingerprint():
        raise ValueError("genişletme önbellek prompt parmak izi uyuşmuyor")
    if record.model_revision != EXPANDER_REVISION:
        raise ValueError("genişletme önbellek model revision uyuşmuyor")
    if len(record.question_sha256) != 64:
        raise ValueError("genişletme önbellek soru hash'i geçersiz")
    return record


def load_expansion_cache(path: Path) -> dict[str, ExpansionRecord]:
    if not path.exists():
        return {}
    records: dict[str, ExpansionRecord] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = _record(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"genişletme önbellek JSONL satırı geçersiz: {line_number}") from exc
        if record.question_id in records:
            raise ValueError(f"genişletme önbellek soru kimliği yinelenmiş: {record.question_id}")
        records[record.question_id] = record
    return records


def write_expansion_cache(path: Path, records: list[ExpansionRecord]) -> None:
    if len({record.question_id for record in records}) != len(records):
        raise ValueError("genişletme önbellek soru kimlikleri benzersiz olmalı")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


class LocalQueryExpander:
    """Sabit Qwen3 ile tek, deterministik Türkçe arama varyantı üretir."""

    def __init__(
        self,
        *,
        device: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if (tokenizer is None) != (model is None):
            raise ValueError("genişletme tokenizer ve model birlikte verilmelidir")
        if torch_module is None:
            import torch

            torch_module = torch
        self._torch = torch_module
        self._device = device or (
            "mps" if self._torch.backends.mps.is_available() else "cpu"
        )
        if tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                EXPANDER_REPO, revision=EXPANDER_REVISION
            )
            model = AutoModelForCausalLM.from_pretrained(
                EXPANDER_REPO, revision=EXPANDER_REVISION
            )
        assert tokenizer is not None and model is not None
        self._tokenizer = tokenizer
        self._model = model.to(self._device)
        self._model.eval()

    def expand(self, question: str) -> str:
        messages = [
            {"role": "system", "content": EXPANSION_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        inputs = {name: value.to(self._device) for name, value in encoded.items()}
        try:
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs, do_sample=False, max_new_tokens=64
                )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).casefold():
                raise
            release_transformer_memory(self._torch)
            raise ExpansionModelOutOfMemory(
                f"expander preflight belleğe sığmadı: {EXPANDER_REPO}@{EXPANDER_REVISION}"
            ) from exc
        completion = generated[:, inputs["input_ids"].shape[1] :]
        decoded = self._tokenizer.decode(completion[0], skip_special_tokens=True)
        return validate_expansion(question, decoded)

    def preflight(self) -> None:
        self.expand("Türk hukukunda yıllık izin süresi nedir?")
