"""Hibrit getirim: BM25 metin kanalı + doküman-adı yönlendirmesi (P1 üretim yolu).

Sıralamayı METİN KANALI belirler (`retrieval/text.py` — autoresearch exp7
reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md, canary
answerable n=43 R@5 0.2326 -> 0.8140). Görsel MaxSim kanalı her sorguda
KOŞMAYA DEVAM EDER ama sıralamaya GİRMEZ:

  * ölçüm (bulgu 3): F5 kırpmasından sonra görselin top-5'e BENZERSİZ katkısı
    SIFIR soru — yani füzyon bugün ölçülebilir bir kazanç getirmiyor;
  * eşit-ağırlık RRF ÖLÇÜLDÜ ve REDDEDİLDİ (R@5 0.6744 -> 0.3953): zayıf
    kanalın kapak-sayfası gürültüsü güçlü kanalın kazanımlarını düşürüyor;
  * yine de koşuyor, çünkü iki kanalın skorlarının YAN YANA kaydı P2
    kalibrasyonunun girdisi (`detail.retrieval.visual_top1`) ve görselin
    "tablo/tarama fallback" rolü orada yeniden çerçevelenecek.

Bu bilinçli bir gecikme takasıdır: görsel kanal sorgu başına ~0.24 sn
(4222 sayfa, int8, CPU) ekler; BM25 milisaniyeler mertebesindedir.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from belge_gozu.index.chunking import CHUNK_TOKENS
from belge_gozu.index.encode import Encoder
from belge_gozu.retrieval.text import WINDOW, BM25Index, route_window, tokenize
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import stage

if TYPE_CHECKING:
    from belge_gozu.index.loader import ScorableIndex

# İstek başına getirim künyesi. Düz bir instance alanı olsaydı eşzamanlı
# isteklerde (FastAPI senkron uç noktaları threadpool'da koşar) bir isteğin
# künyesi diğerinin olayına yazılabilirdi. ContextVar `telemetry/collect.py`
# ile aynı desendir: uç nokta gövdesi ve `search()` aynı bağlamda çalışır.
_LAST_META: ContextVar[dict | None] = ContextVar("bg_hybrid_meta", default=None)


class HybridRetriever:
    """BM25 sıralaması + pencere-içi doküman yönlendirmesi; görsel kanal telemetride.

    `ExhaustiveRetriever` ile AYNI sözleşme: `search(query, k) -> list[PageHit]`,
    `index`/`encoder`/`meta` alanları ve `telemetry.collect.stage` ile aşama
    ölçümü. Aşama adları: query_encode, exhaustive_maxsim (görsel),
    text_bm25, route_fuse.

    `PageHit.score` BM25 ölçeğindedir (kalibre edilmemiş, üst sınırsız; ölçülen
    bant ~4-70) — görsel kolun normalize [-1,1] skoru DEĞİLDİR. `AskService`
    eşiği bu skorla karşılaştırdığı için `Settings.min_score_threshold` da bu
    ölçekte olmak zorundadır (`app/main.py` başlangıçta korkulukla doğrular).
    """

    # ExhaustiveRetriever ile aynı desen: indekse her çağrıda geçilir, test
    # override'ı için instance üstünde değiştirilebilir.
    CHUNK_TOKENS: ClassVar[int] = CHUNK_TOKENS

    def __init__(
        self,
        index: ScorableIndex,
        meta: pd.DataFrame,
        encoder: Encoder | None,
        text: BM25Index,
        doc_names: dict[str, frozenset[str]],
        window: int = WINDOW,
    ) -> None:
        if list(text.page_ids) != list(index.page_ids):
            raise ValueError(
                "metin indeksi ile görsel indeksin page_ids listeleri birebir "
                f"eşleşmeli (metin n={len(text.page_ids)}, görsel n={len(index.page_ids)}); "
                "aksi halde BM25 skorları yanlış sayfalara hizalanır"
            )
        self.index = index
        self.encoder = encoder
        self.meta = meta.set_index("page_id", drop=False)
        self.text = text
        self.doc_names = doc_names
        self.window = window

    @property
    def last_retrieval_meta(self) -> dict | None:
        """Son `search()` çağrısının künyesi (istek-yerel) ya da None.

        `app/main.py` bunu `detail.retrieval`'e karıştırır: iki kanalın top-1
        skoru yan yana kaydedilir (P2 kalibrasyon verisi) ve hangi
        dokümanların yönlendirmeyi tetiklediği görünür olur. Buradaki
        `bm25_top1` KANALIN en yüksek skorudur; SERVİS EDİLEN top-1 (yani
        eşikle karşılaştırılan skor) pencere-içi yönlendirmeden sonra farklı
        bir sayfa olabilir ve olayın `top_score` alanında durur.
        """
        return _LAST_META.get()

    def routed_docs(self, query: str) -> set[str]:
        """Adının jenerik-dışı TÜM token'ları sorguda geçen doküman(lar)."""
        q_toks = set(tokenize(query))
        return {doc for doc, toks in self.doc_names.items() if toks <= q_toks}

    def rank_all(self, query: str) -> list[str]:
        """Tam korpus sıralaması (reçetenin nihai sırası) — teşhis/bench/cırcır için.

        Görsel kanalı ÇALIŞTIRMAZ: sıralamaya girmediği için sonucu
        değiştirmez, ama model yüklemeden koşulabilmesini sağlar (deterministik).
        """
        return self._rank(query, self.text.scores(query))[0]

    def _rank(self, query: str, bm25: np.ndarray) -> tuple[list[str], set[str]]:
        order = np.argsort(-bm25, kind="stable")
        ranking = [self.index.page_ids[i] for i in order]
        routed = self.routed_docs(query)
        return route_window(ranking, routed, self.window), routed

    def search(self, query: str, k: int = 5) -> list[PageHit]:
        if self.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        with stage("query_encode"):
            q_emb = self.encoder.encode_query(query)
        # Görsel kanal: sıralamaya GİRMEZ (yukarıdaki modül açıklaması),
        # telemetri ve P2 kalibrasyon verisi için koşar.
        with stage("exhaustive_maxsim"):
            visual = self.index.score_all(q_emb, chunk_tokens=self.CHUNK_TOKENS)
        with stage("text_bm25"):
            bm25 = self.text.scores(query)
        with stage("route_fuse"):
            ranking, routed = self._rank(query, bm25)
        by_id = dict(zip(self.index.page_ids, bm25.tolist(), strict=True))
        _LAST_META.set(
            {
                "bm25_top1": float(bm25.max()) if bm25.size else 0.0,
                "visual_top1": float(visual.max()) if visual.size else 0.0,
                "routed_docs": sorted(routed),
            }
        )
        out: list[PageHit] = []
        for pid in ranking[:k]:
            row = self.meta.loc[pid]
            out.append(
                PageHit(
                    page_id=row["page_id"],
                    # BM25 ölçeği — hibritin SIRALAMA ölçeği. Görsel skor
                    # bilinçli olarak buraya karışmıyor: tek bir alanda iki
                    # farklı ölçek taşımak T14'ün ayıkladığı hatanın aynısı.
                    score=by_id[pid],
                    doc_name=row["doc_name"],
                    page_no=int(row["page_no"]),
                    image_path=row["image_path"],
                    source_url=row["source_url"],
                )
            )
        return out
