import hashlib
import threading
from typing import Protocol

import numpy as np
from PIL import Image

from belge_gozu.index.manifest import CPE_0_3_18, QueryFormat

# SORGU KODLAMA EŞZAMANLILIK SINIRI — savunmacı, ölçüme dayalı DEĞİL bir tavan.
#
# FastAPI'nin senkron uç noktaları threadpool'da koşar, yani N eşzamanlı istek
# N eşzamanlı `encode_query` (VLM ileri geçişi) demektir. ÖLÇÜM: 40 istek @ c=8,
# 40/40 başarılı, p50 1.34 sn, çökme yok — yani BUGÜN bir sorun görülmedi.
# Sınır yine de konuyor çünkü MPS/CUDA bellek fırtınası bilinen bir risk
# SINIFIDIR (aynı korpusta daha önce bir eşzamanlılık çökmesi yaşandı, bkz.
# docs/research/) ve tavanın maliyeti ölçülen çalışma noktasında sıfırdır:
# c=8 kuyruğa girer, c<=4 hiç dokunulmaz. Bir "optimizasyon" değil, kuyruğa
# çevirme kararıdır — istekler reddedilmez, sıraya girer.
#
# Getirim katmanı (retrieval/core.py + retrieval/hybrid.py) bunu KODLAYICI
# ÇAĞRISININ ETRAFINDA kullanır; skorlama/BM25 sınırın dışındadır.
ENCODE_LIMIT = threading.Semaphore(4)


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


class _VisualPromptProcessor(Protocol):
    """`apply_visual_prompt_override`'ın dokunduğu tek yüzey.

    colpali-engine'de `visual_prompt_prefix` bir ClassVar; burada onu okunur-yazılır
    bir örnek niteliği olarak tanımlamak, aşağıdaki atamanın kasıtlı bir ClassVar
    gölgelemesi olduğunu tip düzeyinde belgeliyor (`type: ignore` gerekmeden)."""

    visual_prompt_prefix: str


def apply_visual_prompt_override(processor: _VisualPromptProcessor, override: str | None) -> str:
    """Doküman prompt'unu processor ÖRNEĞİNE yazar ve etkin prompt'u döner.

    Örnek üzerine atama ClassVar'ı yalnız bu örnek için gölgeler (sınıf sabiti
    bozulmaz) ve `process_images` `self.visual_prompt_prefix` okuduğu için override
    doğrudan text'e geçer. override=None ise mevcut davranış aynen korunur."""
    if override is not None:
        processor.visual_prompt_prefix = override
    return processor.visual_prompt_prefix


def trim_by_mask(emb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """(B, L, D) embedding + (B, L) attention mask -> padding'siz [(l_i, D)].

    colpali add_model_family sözleşmesi padding embedding'lerini sıfırlar; bu
    sıfırlar dot-product MaxSim'de zararsızdır ama sign-binarizasyonda geçerli
    bit desenine dönüşür (v0 bug'ı: indekste 3960 all-zero satır). Çözüm:
    binarize ETMEDEN önce padding satırlarını at."""
    return [e[m.astype(bool)] for e, m in zip(emb, mask, strict=True)]


class ColSmolEncoder:
    """colpali-engine sarmalayıcı. Yalnız `ml` extra'sıyla çalışır; API yüzeyi
    Task 13'te gerçek modelle doğrulanır (spec §11 model-eskimesi riski)."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        query_format: QueryFormat | None = None,
        visual_prompt_override: str | None = None,
    ):
        import torch  # type: ignore[import-not-found]
        from colpali_engine.models import (  # type: ignore[import-not-found]
            ColIdefics3,
            ColIdefics3Processor,
        )

        self.device = resolve_device(device)
        # gpu'da yarım hassasiyet: index build hızı için (Task 13 controller
        # optimizasyonu). cpu'da float32'de kalır (yarım hassasiyet cpu'da desteksiz/yavaş).
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        # device_map=self.device (ör. "mps") segfault veriyor: bu torch/transformers/
        # accelerate sürüm bileşiminde ağırlıkları doğrudan mps'e yükleyen yol kırık
        # (Task 13 canlı doğrulama, izole edildi). cpu'da yükleyip .to(device) ile
        # taşımak aynı sonucu güvenle veriyor.
        model = ColIdefics3.from_pretrained(model_name, torch_dtype=dtype, device_map="cpu")
        # transformers'ın PreTrainedModel.to = @wraps(torch.nn.Module.to) sarmalayıcısı
        # pyright'ı .to()'nun "self"ini sıradan zorunlu parametre sanmaya itiyor
        # (typeshed/functools.wraps kısıtı); çalışma zamanında sorunsuz (üstte doğrulandı).
        self.model = model.to(self.device).eval()  # type: ignore[reportArgumentType]
        self.processor = ColIdefics3Processor.from_pretrained(model_name)
        self.query_format = query_format or CPE_0_3_18
        self.model_revision = getattr(model.config, "_commit_hash", None) or "unknown"
        # pyright ClassVar ilan edilmiş bir üyeyi yazılabilir protokol üyesine denk
        # saymıyor; gölgeleme burada KASITLI (aşağıdaki protokolün docstring'i).
        self.doc_prompt = apply_visual_prompt_override(
            self.processor,  # type: ignore[reportArgumentType]
            visual_prompt_override,
        )
        self.doc_prompt_sha256 = hashlib.sha256(self.doc_prompt.encode()).hexdigest()

    def _run(self, batch) -> list[np.ndarray]:
        import torch  # type: ignore[import-not-found]

        with torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        emb = out.cpu().float().numpy()
        mask = batch["attention_mask"].cpu().numpy()
        return trim_by_mask(emb, mask)

    def encode_pages(self, images: list[Image.Image]) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        for i in range(0, len(images), 4):
            batch = self.processor.process_images(images[i : i + 4])
            results.extend(self._run(batch))
        return results

    def encode_query(self, text: str) -> np.ndarray:
        rendered = self.query_format.render(text)
        return self._run(self.processor.process_texts([rendered]))[0]
