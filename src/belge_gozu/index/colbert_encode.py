"""ColBERT geç-etkileşim kodlama sözleşmesi — `pylate` OLMADAN.

NEDEN pylate yok. Kurulumu `torch 2.13.0 -> 2.11.0` ve `transformers 5.15.1 ->
5.3.0` DÜŞÜRÜYOR: `pylate` -> `fast-plaid` zinciri `torch==2.11.0`'ı tam olarak
pinliyor ve hiçbir pylate sürümü bundan kaçamıyor. Bu bir sürüm ÇAKIŞMASI değil
(`uv lock` başarıyla çözüyor) — bedel revalidasyon: `colpali-engine==0.3.18`
pinlenmiş çünkü görsel prompt-format sözleşmesi MEVCUT torch/transformers
altında doğrulandı. Ölçülmüş ve dondurulmuş bir yığını, 10.531 chunk'ta hiç
ihtiyacımız olmayan PLAID indekslemesi için düşürmek kabul edilemez.

Karşılığında bu modül pylate'in `ColBERT.encode()` davranışını yeniden üretir.
Eşdeğerlik ölçüldü (max_abs_diff 4,5e-07…2,1e-06 — float32 gürültüsü, "bit
düzeyinde" DEĞİL) ve `tests/index/test_colbert_encode.py` sözleşmeyi kilitler.

SÖZLEŞME MODELDEN OKUNUR, TAHMİN EDİLMEZ. `config_sentence_transformers.json`
her şeyi bildirir: işaret token'ları, sorgu/belge uzunlukları, genişletme ve
noktalama skiplist'i. `moganai/Mogan-ColBERT-TR` için `[unused0]`/`[unused1]`,
32/512; `newmindai/ColmmBERT-*` için `"[Q] "`/`"[D] "` — **sondaki boşlukla**,
onsuz UNK'a düşer.

ÜÇ TUZAK, üçü de sessiz:

1. `sentence_bert_config.json` Mogan'da `max_seq_length: 31` gönderiyor. pylate
   onu tokenize zamanında eziyor; naif bir yükleyici HER BELGEYİ 31 token'a
   keser. Bu dosya BİLEREK okunmaz.
2. pylate'in `encode()` fonksiyonunun `is_query` varsayılanı **True**. Belgeleri
   bayrak vermeden kodlamak her chunk'a sorgu işareti verir, 32 token'a keser ve
   [MASK] ile doldurur — hata yok, uyarı yok. Bu modül `is_query` parametresi
   AÇMAZ; yalnız `encode_queries` / `encode_documents` vardır.
3. Noktalama skiplist'i çözülürken `unk_token_id` kümeden ÇIKARILMALI, yoksa
   gerçek UNK taşıyan her vektör de belgeden atılır.

Bu hata sınıfının bu projedeki ölçülmüş bedeli: görsel kodlayıcıda sorgu formatı
yanlış olduğunda R@5 0,233 yerine 0,093 (T11/Step 6 A/B).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ColBERT projeksiyon çıktısı; görsel kanalın `index/quantize.py:EMBED_DIM`
# değeriyle aynı olması tesadüf değil, Mogan'ın 128-d çıktısı mevcut depoya
# olduğu gibi oturuyor.
EMBED_DIM = 128


@dataclass(frozen=True)
class ColbertConfig:
    """Modelin kendi bildirdiği kodlama sözleşmesi."""

    query_prefix: str
    document_prefix: str
    query_length: int
    document_length: int
    do_query_expansion: bool
    attend_to_expansion_tokens: bool
    skiplist_ids: frozenset[int]


def load_colbert_config(
    raw: dict, skiplist_to_ids: dict[str, int] | None, unk_id: int | None = None
) -> ColbertConfig:
    """`config_sentence_transformers.json` -> `ColbertConfig`.

    `skiplist_to_ids` çağıran tarafından tokenizer ile çözülür (bu modül I/O
    yapmaz, test edilebilir kalır). `unk_id` verilirse skiplist'ten DÜŞÜLÜR:
    aksi halde gerçek UNK taşıyan her belge vektörü de atılırdı.
    """
    if skiplist_to_ids is None:
        raise ValueError("skiplist kimlikleri tokenizer ile çözülmeden config kurulamaz")
    ids = {skiplist_to_ids[w] for w in raw.get("skiplist_words", []) if w in skiplist_to_ids}
    if unk_id is not None:
        ids.discard(unk_id)
    return ColbertConfig(
        query_prefix=raw["query_prefix"],
        document_prefix=raw["document_prefix"],
        query_length=int(raw["query_length"]),
        document_length=int(raw["document_length"]),
        do_query_expansion=bool(raw.get("do_query_expansion", True)),
        attend_to_expansion_tokens=bool(raw.get("attend_to_expansion_tokens", False)),
        skiplist_ids=frozenset(ids),
    )


def _splice_marker(ids: list[int], marker_id: int) -> list[int]:
    """İşaret index 1'e — [CLS]'ten SONRA — token KİMLİĞİ olarak girer.

    Metin olarak öne eklemek (`tokenizer("[Q] " + text)`) yanlıştır: kesme
    bütçesini bir token kaydırır ve `[unused0]` sözleşmesinde tamamen kırılır.
    """
    return [ids[0], marker_id, *ids[1:]]


def build_query_ids(
    base_ids: list[int], marker_id: int, mask_id: int, cfg: ColbertConfig
) -> tuple[list[int], list[int]]:
    """Sorgu tarafı: kes -> işaretle -> [MASK] ile genişlet.

    Genişletme ColBERT'in kendi mekanizmasıdır (§3.2, Nq=32) ve makale onu
    "essential" diye niteler. `attend_to_expansion_tokens=False` olduğu için
    dikkat maskesi o pozisyonlarda 0'dır — ama vektörlerin TAMAMI korunur,
    MaxSim 32 vektörün hepsini görür.
    """
    ids = _splice_marker(base_ids[: cfg.query_length - 1], marker_id)
    attn = [1] * len(ids)
    if cfg.do_query_expansion:
        pad = cfg.query_length - len(ids)
        if pad > 0:
            ids = ids + [mask_id] * pad
            attn = attn + [1 if cfg.attend_to_expansion_tokens else 0] * pad
    return ids, attn


def build_document_ids(
    base_ids: list[int], marker_id: int, cfg: ColbertConfig
) -> tuple[list[int], list[int]]:
    """Belge tarafı: kes -> işaretle. DOLDURMA YOK.

    Belge sabit uzunluğa çekilmez; tam olarak `query_length` vektör üreten bir
    belge, sorgu gibi kodlanmış demektir (yukarıdaki 2. tuzak).
    """
    ids = _splice_marker(base_ids[: cfg.document_length - 1], marker_id)
    return ids, [1] * len(ids)


def document_keep_mask(
    ids: list[int], skiplist_ids: frozenset[int] | set[int], pad_id: int
) -> list[bool]:
    """Belge tarafında saklanacak pozisyonlar: dolgu ve noktalama düşer.

    Noktalama GİRDİ metninden silinmez, ÇIKTIDA elenir — metni kırpmak
    tokenizasyonu ve bağlamsallaştırmayı değiştirirdi.
    """
    return [not (t == pad_id or t in skiplist_ids) for t in ids]


def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
    """ColBERT MaxSim: sorgu token'ları üzerinde TOPLAM, belge üzerinde maksimum.

    Ortalama değil toplam (pylate: `einsum(...).max(-1).values.sum(-1)`).
    Genişletme açıkken |Eq| sabit 32 olduğu için ikisi monoton yeniden
    ölçeklemedir ve sorgu-içi sıralama aynıdır; fark yalnız sorgular ARASI
    karşılaştırmada ortaya çıkar.
    """
    if doc.shape[0] == 0 or query.shape[0] == 0:
        return 0.0
    return float((query @ doc.T).max(axis=1).sum())


# --------------------------------------------------------------------------
# Kodlayıcı — torch/transformers gerektirir (`ml` ekstrası)
# --------------------------------------------------------------------------

# Üretim kolu. Revizyon SABİT: sessiz bir upstream yeniden eğitimi indeksi
# geçersiz kılar ama hiçbir hata vermez.
MOGAN_REPO = "moganai/Mogan-ColBERT-TR"
MOGAN_REVISION = "ad90b4f64135e4db75a6453feee85fd7b44b33a1"


class ColBERTEncoder:
    """Geç-etkileşim kodlayıcı. `is_query` parametresi BİLEREK yoktur.

    Yalnız `encode_queries` ve `encode_documents` vardır; ikisi de bayrağı
    kendi içinde sabitler. pylate'in tek `encode(is_query=True)` yüzeyi bu
    projede ölçülmüş bir hata sınıfıdır (bkz. modül başlığı, tuzak 2).
    """

    def __init__(
        self,
        repo: str = MOGAN_REPO,
        revision: str = MOGAN_REVISION,
        device: str | None = None,
        document_length: int | None = None,
    ) -> None:
        import json as _json

        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        path = snapshot_download(repo, revision=revision)
        self.path = path
        self.repo, self.revision = repo, revision

        # SÖZLEŞME buradan okunur. `sentence_bert_config.json` BİLEREK
        # okunmuyor: Mogan'da max_seq_length=31 ve onu miras almak her belgeyi
        # 31 token'a keserdi.
        with open(f"{path}/config_sentence_transformers.json") as fh:
            raw = _json.load(fh)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        tok = self.tokenizer
        self.q_id = tok.convert_tokens_to_ids(raw["query_prefix"])
        self.d_id = tok.convert_tokens_to_ids(raw["document_prefix"])
        # İşaret UNK'a düşerse getirim SESSİZCE çöker. Tek assert tüm hata
        # sınıfını yakalar. `add_tokens` ASLA çağrılmaz — o, eğitilmemiş taze
        # satır basmaktır.
        if tok.unk_token_id in (self.q_id, self.d_id):
            raise ValueError(
                f"işaret token'ı UNK'a düştü: query={raw['query_prefix']!r} -> {self.q_id}, "
                f"document={raw['document_prefix']!r} -> {self.d_id}. "
                "Sözleşme checkpoint'e özgüdür; çapraz uygulamak sessizce bozar."
            )
        self.mask_id = tok.mask_token_id
        if self.mask_id is None:
            raise ValueError("sorgu genişletmesi için mask_token_id gerekli")

        skip = {w: tok.convert_tokens_to_ids(w) for w in raw.get("skiplist_words", [])}
        if document_length is not None:
            raw = {**raw, "document_length": document_length}
        self.cfg = load_colbert_config(raw, skip, unk_id=tok.unk_token_id)

        # Dense başlığın sözleşmesi de doğrulanır: uyumsuz bir checkpoint
        # çökmek yerine sessizce yanlış vektör üretmesin.
        with open(f"{path}/1_Dense/config.json") as fh:
            dense = _json.load(fh)
        if dense["bias"] or not dense["activation_function"].endswith("Identity"):
            raise ValueError(f"beklenmeyen dense başlığı: {dense}")
        if dense.get("use_residual") or dense["out_features"] != EMBED_DIM:
            raise ValueError(f"beklenmeyen dense çıkışı: {dense}")

        self.model = AutoModel.from_pretrained(path).eval()
        w = load_file(f"{path}/1_Dense/model.safetensors")["linear.weight"]
        self.proj = torch.nn.Linear(dense["in_features"], dense["out_features"], bias=False)
        self.proj.load_state_dict({"weight": w})
        self.proj.eval()

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.proj.to(self.device)
        self.torch = torch

    def _forward(self, ids, attn):
        t = self.torch
        with t.inference_mode():
            batch_ids = t.tensor(ids, device=self.device)
            batch_attn = t.tensor(attn, device=self.device)
            hidden = self.model(input_ids=batch_ids, attention_mask=batch_attn).last_hidden_state
            # SIRA ÖNEMLİ: önce projeksiyon, SONRA L2 normalize.
            out = t.nn.functional.normalize(self.proj(hidden), p=2, dim=-1)
        return out.float().cpu().numpy()

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        """Her sorgu tam olarak (query_length, 128) — sabit maliyet."""
        rows = [
            build_query_ids(
                self.tokenizer(t, add_special_tokens=True)["input_ids"],
                self.q_id, self.mask_id, self.cfg,
            )
            for t in texts
        ]
        ids = [r[0] for r in rows]
        attn = [r[1] for r in rows]
        out = self._forward(ids, attn)
        # Sorgu tarafında maske UYGULANMAZ: genişletme vektörleri de MaxSim'e girer.
        return [out[i] for i in range(len(texts))]

    def encode_documents(self, texts: list[str], batch_size: int = 16) -> list[np.ndarray]:
        """Değişken uzunlukta; dolgu ve noktalama ÇIKTIDA elenir."""
        out: list[np.ndarray] = []
        pad = self.tokenizer.pad_token_id
        for i in range(0, len(texts), batch_size):
            rows = [
                build_document_ids(
                    self.tokenizer(t, add_special_tokens=True)["input_ids"], self.d_id, self.cfg
                )
                for t in texts[i : i + batch_size]
            ]
            width = max(len(r[0]) for r in rows)
            ids = [r[0] + [pad] * (width - len(r[0])) for r in rows]
            attn = [r[1] + [0] * (width - len(r[1])) for r in rows]
            vecs = self._forward(ids, attn)
            for j, row in enumerate(rows):
                keep = document_keep_mask(ids[j], self.cfg.skiplist_ids, pad_id=pad)
                # dolgu her hâlükârda düşer; keep maskesi noktalamayı da eler
                keep = [k and (n < len(row[0])) for n, k in enumerate(keep)]
                out.append(vecs[j][np.array(keep, dtype=bool)].astype(np.float16))
        return out
