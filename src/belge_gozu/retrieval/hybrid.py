"""Hibrit getirim: BM25 metin kanalı + doküman-adı yönlendirmesi (P1 üretim yolu).

Sıralamayı METİN KANALI belirler (`retrieval/text.py` — autoresearch
exp7/exp8/exp12 reçetesi; ölçüm: findings 2026-08-29-autoresearch-text-channel.md
+ journal #8 ve #12, canary answerable n=43 R@5 0.2326 -> 0.8372 -> **0.8605**;
exp12'nin ASCII aksan katlaması sistemi YAZIM-DEĞİŞMEZ yapar — aksansız yazılan
aynı sorgu da 0.8605). Görsel MaxSim kanalı her sorguda KOŞMAYA DEVAM EDER ama
sıralamaya GİRMEZ:

  * ölçüm (bulgu 3): F5 kırpmasından sonra görselin top-5'e BENZERSİZ katkısı
    SIFIR soru — yani füzyon bugün ölçülebilir bir kazanç getirmiyor;
  * üç ayrı füzyon biçimi ÖLÇÜLDÜ ve REDDEDİLDİ: küresel eşit-ağırlık RRF
    (R@5 0.6744 -> 0.3953), mutlak doküman bölümlemesi (R@20 vetosu) ve
    pencere-içi RRF (0.8372 -> 0.5349, journal #10) — zayıf kanalın
    kapak-sayfası çekimi her granülaritede metin gold'larını eziyor;
  * yine de koşuyor, çünkü iki kanalın skorlarının YAN YANA kaydı P2
    kalibrasyonunun girdisi (`detail.retrieval.visual_top1`) ve görselin
    "tablo/tarama fallback" rolü orada yeniden çerçevelenecek.

Bu bilinçli bir gecikme takasıdır: görsel kanal sorgu başına ~0.24 sn
(4222 sayfa, int8, CPU) ekler; BM25 milisaniyeler mertebesindedir.

Metin kanalı ARTEFAKTININ yüklenmesi de burada (`require_text_artifact` /
`load_text_channel`): serve ve bench aynı kurulumu çağırsın diye getirim
katmanında durur — daha önce `app/main.py`'deydi ve `bench run` FastAPI
uygulama modülünü çekmek zorunda kalıyordu (review L8).
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from belge_gozu.index.chunking import CHUNK_TOKENS
from belge_gozu.index.compat import IndexCompatibilityError
from belge_gozu.index.encode import ENCODE_LIMIT, Encoder
from belge_gozu.retrieval.text import (
    WINDOW,
    BM25Index,
    extract_doc_name_tokens,
    route_window,
)
from belge_gozu.retrieval.text import routed_docs as _routed_docs
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import stage

if TYPE_CHECKING:
    from belge_gozu.index.loader import ScorableIndex

logger = logging.getLogger(__name__)

TEXT_ARTIFACT_NAME = "page_texts.parquet"

# İstek başına getirim künyesi. Düz bir instance alanı olsaydı eşzamanlı
# isteklerde (FastAPI senkron uç noktaları threadpool'da koşar) bir isteğin
# künyesi diğerinin olayına yazılabilirdi. ContextVar `telemetry/collect.py`
# ile aynı desendir: uç nokta gövdesi ve `search()` aynı bağlamda çalışır.
_LAST_META: ContextVar[dict | None] = ContextVar("bg_hybrid_meta", default=None)


def require_text_artifact(index_dir: Path) -> Path:
    """Metin kanalı artefaktının VARLIĞINI doğrular (saf dosya sistemi kontrolü).

    Ayrı bir fonksiyon çünkü `create_app` bunu VLM ağırlıklarını ve 474 MB'lık
    indeksi yüklemeden ÖNCE çağırır (review L6): tek satırlık "`index
    build-text` çalıştır" mesajı için dakikalarca model yüklemek anlamsız.
    Hizalama kontrolü indeksin `page_ids`'ini gerektirdiği için
    `load_text_channel`'da kalır."""
    path = index_dir / TEXT_ARTIFACT_NAME
    if not path.exists():
        raise IndexCompatibilityError(
            f"hibrit pipeline metin kanalı artefaktı gerektirir ama {path} yok. "
            "Çözüm: `uv run belge-gozu index build-text` (model gerekmez, saniyeler sürer). "
            "Alternatif: BG_RETRIEVAL_PIPELINE=exhaustive ile yalnız-görsel yola dönün "
            "(eşiği de o ölçeğe taşımayı unutmayın)."
        )
    return path


def load_text_channel(
    index_dir: Path, page_ids: list[str]
) -> tuple[BM25Index, dict[str, frozenset[str]]]:
    """`<index_dir>/page_texts.parquet` -> (BM25 indeksi, doküman-adı token'ları).

    Artefakt `belge-gozu index build-text` ile üretilir ve indeks dizininde
    durur: metin kanalı görsel indeksin SAYFA SIRASINA bağlıdır, ayrı bir
    dizinde tutulmak ikisinin sessizce ayrışmasına davetiye olurdu. Sıra
    burada ayrıca DOĞRULANIR (bir satır kayması yanlış sayfayı döndürür ve
    hiçbir yerde hata vermez)."""
    path = require_text_artifact(index_dir)
    df = pd.read_parquet(path)
    file_ids = df["page_id"].tolist()
    if file_ids != list(page_ids):
        missing = sorted(set(page_ids) - set(file_ids))
        extra = sorted(set(file_ids) - set(page_ids))
        detail = (
            f"indekste olup metinde olmayan={len(missing)} {missing[:3]}; "
            f"metinde olup indekste olmayan={len(extra)} {extra[:3]}"
            if (missing or extra)
            else "aynı küme, farklı SIRA (listeler birebir eşleşmeli)"
        )
        raise IndexCompatibilityError(
            f"{TEXT_ARTIFACT_NAME} indeksle hizalı değil: indeks n={len(page_ids)} "
            f"metin n={len(file_ids)} — {detail}. Çözüm: `uv run belge-gozu index build-text` "
            "ile yeniden üretin."
        )
    texts = df["text"].fillna("").tolist()
    t0 = time.perf_counter()
    bm25 = BM25Index(list(page_ids), texts)
    logger.info(
        "BM25 metin indeksi kuruldu: %d sayfa, %.2f sn", len(page_ids), time.perf_counter() - t0
    )
    return bm25, extract_doc_name_tokens(list(page_ids), texts)


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
        """Adının jenerik-dışı TÜM token'ları sorguda geçen doküman(lar).

        Yüklem `retrieval/text.py`'de durur ve buradan DELEGE edilir: P2
        kalibrasyonunun `routed` özelliği aynı fonksiyonu çağırır, yani
        eğitim-zamanı özellik ile servis-zamanı davranış birbirinden sapamaz.
        """
        return _routed_docs(query, self.doc_names)

    def rank_all(self, query: str) -> list[str]:
        """Tam korpus sıralaması (reçetenin nihai sırası) — teşhis/bench/cırcır için.

        Görsel kanalı ÇALIŞTIRMAZ: sıralamaya girmediği için sonucu
        değiştirmez, ama model yüklemeden koşulabilmesini sağlar (deterministik).
        """
        return self.rank(query, self.text.scores(query))[0]

    def rank(self, query: str, bm25: np.ndarray) -> tuple[list[str], set[str]]:
        """(nihai sıralama, yönlendirilen doküman kümesi) — reçetenin sıra kompozisyonu.

        PUBLIC çünkü bench adapter'ı (`bench/harness.py`) da bunu çağırır:
        kompozisyonu orada yeniden kurmak, üretim sırası değiştiğinde bench'in
        sessizce BAŞKA bir şey ölçmesi demekti (review L7)."""
        order = np.argsort(-bm25, kind="stable")
        ranking = [self.index.page_ids[i] for i in order]
        routed = self.routed_docs(query)
        return route_window(ranking, routed, self.window), routed

    def search(self, query: str, k: int = 5) -> list[PageHit]:
        if self.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        # savunmacı sınır, ölçüm: 40@c=8 sağlıklı (bkz. index/encode.py)
        # M2: ENCODE_LIMIT ÖNCE alınır — kuyruk beklemesi ölçüme karışmaz.
        with ENCODE_LIMIT, stage("query_encode"):
            q_emb = self.encoder.encode_query(query)
        # Görsel kanal: sıralamaya GİRMEZ (yukarıdaki modül açıklaması),
        # telemetri, gösterim ve P2 kalibrasyon verisi için koşar.
        with stage("exhaustive_maxsim"):
            visual = self.index.score_all(q_emb, chunk_tokens=self.CHUNK_TOKENS)
        with stage("text_bm25"):
            bm25 = self.text.scores(query)
        with stage("route_fuse"):
            ranking, routed = self.rank(query, bm25)
        by_id = dict(zip(self.index.page_ids, bm25.tolist(), strict=True))
        visual_by_id = dict(zip(self.index.page_ids, visual.tolist(), strict=True))
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
                    # ...ama AYRI bir alanda taşınıyor: aynı sayfaya görsel
                    # kanalın verdiği normalize ~[-1,1] skor. Sıralamaya
                    # girmez; arayüzde "iki kanal" iddiasını görünür kılar.
                    visual_score=visual_by_id[pid],
                    doc_name=row["doc_name"],
                    page_no=int(row["page_no"]),
                    image_path=row["image_path"],
                    source_url=row["source_url"],
                )
            )
        return out
