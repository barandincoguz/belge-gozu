import hashlib
from typing import Protocol

import numpy as np
from PIL import Image


class Encoder(Protocol):
    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]: ...
    def encode_query(self, text: str) -> np.ndarray: ...


def _seeded(data: bytes, n_tokens: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_tokens, 128)).astype(np.float32)


class FakeEncoder:
    def __init__(self, tokens_per_item: int = 8):
        self.tokens_per_item = tokens_per_item

    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        return [_seeded(img.tobytes(), self.tokens_per_item) for img in images]

    def encode_query(self, text: str) -> np.ndarray:
        return _seeded(text.encode("utf-8"), self.tokens_per_item)


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    import torch  # type: ignore[import-not-found]

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ColSmolEncoder:
    """colpali-engine sarmalayıcı. Yalnız `ml` extra'sıyla çalışır; API yüzeyi
    Task 13'te gerçek modelle doğrulanır (spec §11 model-eskimesi riski)."""

    def __init__(self, model_name: str, device: str = "auto"):
        import torch  # type: ignore[import-not-found]
        from colpali_engine.models import (  # type: ignore[import-not-found]
            ColIdefics3,
            ColIdefics3Processor,
        )

        self.device = resolve_device(device)
        # device_map=self.device (ör. "mps") segfault veriyor: bu torch/transformers/
        # accelerate sürüm bileşiminde ağırlıkları doğrudan mps'e yükleyen yol kırık
        # (Task 13 canlı doğrulama, izole edildi). cpu'da yükleyip .to(device) ile
        # taşımak aynı sonucu güvenle veriyor.
        model = ColIdefics3.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
        # transformers'ın PreTrainedModel.to = @wraps(torch.nn.Module.to) sarmalayıcısı
        # pyright'ı .to()'nun "self"ini sıradan zorunlu parametre sanmaya itiyor
        # (typeshed/functools.wraps kısıtı); çalışma zamanında sorunsuz (üstte doğrulandı).
        self.model = model.to(self.device).eval()  # type: ignore[reportArgumentType]
        self.processor = ColIdefics3Processor.from_pretrained(model_name)

    def _run(self, batch) -> list[np.ndarray]:
        import torch  # type: ignore[import-not-found]

        with torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        return [e.cpu().float().numpy() for e in out]

    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        for i in range(0, len(images), 4):
            batch = self.processor.process_images(images[i : i + 4])
            results.extend(self._run(batch))
        return results

    def encode_query(self, text: str) -> np.ndarray:
        return self._run(self.processor.process_queries([text]))[0]
