import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from belge_gozu.answer.base import HONEST_MISS_MARKER, Answer, AnswererError
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import annotate

logger = logging.getLogger(__name__)

# ZAMAN AŞIMI BÜTÇESİ (Y15). Çağrı SENKRON bir uç noktadan yapılıyor, yani
# Starlette'in iş parçacığı havuzundan bir iş parçacığı TUTUYOR. Zaman aşımı
# yokken Gemini tarafındaki asılı bir TCP bağlantısı o iş parçacığını süresiz
# tutuyordu; 40 istek sonrası havuz tükenip `/healthz` DAHİL her senkron uç
# nokta yanıt veremez hâle geliyordu — sistemin en büyük tek arıza noktası.
#
# `GEMINI_TIMEOUT_S` httpx'e SKALER olarak geçer ve httpx bunu
# `Timeout(connect=15, read=15, write=15, pool=15)` diye yorumlar — yani DÖRT
# BAĞIMSIZ FAZ SAYACI, toplam bir duvar-saati son tarihi DEĞİL. `/ask` her
# çağrıda beş WebP görüntüsü yüklüyor: `write` fazı birçok ayrı yazma işlemine
# bölünür ve her biri kendi 15 sn'sini alır, `read` de baytlar ARASINDAKİ süreyi
# ölçer. Ölçülmüş kanıt (olay 2899, `http_429` — retry EDİLMEYEN sınıf, yani
# kesinlikle tek HTTP çağrısı): `answer_ms = 16 225 ms`, tek deneme kendi 15
# sn'sini 1,2 sn AŞTI.
#
# Bu yüzden toplam tavan bir YORUM DEĞİL, `generate()` içinde ÖLÇÜLEN bir
# invariant: her denemeden önce geçen gerçek süre okunur ve kalan bütçe bir
# deneme daha kaldıramıyorsa retry YAPILMAZ. Tavanın kendisi (35 sn) istemci
# tarafındaki makul bekleme eşiğinin (~60 sn) yarısı: kullanıcı sekmesini
# kapatmadan önce sunucu kendi kararını vermiş olmalı.
#
# DÜRÜST SINIR: invariant retry KARARINI kapsar, tek bir denemenin İÇİNİ
# kapsamaz — faz sayaçları yüzünden tek bir deneme de tavanı aşabilir (yukarıda
# 16,2 sn ölçüldü). Yani garanti "toplam <= 35 sn" değil, "bütçe aşılmışken
# ÜSTÜNE bir deneme daha BİNMEZ"dir. Sert bir duvar-saati kesmesi ayrı bir iş
# parçacığı/iptal mekanizması gerektirir ve bu fazın kapsamında değildir.
GEMINI_TIMEOUT_S = 15.0
GEMINI_RETRY_BACKOFF_S = 0.5
GEMINI_MAX_ATTEMPTS = 2
GEMINI_TOTAL_BUDGET_S = 35.0

# Yalnız BUNLAR yeniden denenir. 429 (kota/hız) bilinçle DIŞARIDA: kota
# aşımında ikinci bir istek durumu yalnız kötüleştirir ve faturayı büyütür.
# auth/safety_block/parse deterministiktir — tekrar aynı sonucu verir.
RETRYABLE_ERROR_TYPES = frozenset({"timeout", "http_5xx"})

# Güvenlik/politika kaynaklı boş yanıtın finish_reason değerleri. Bu liste
# "cevaplamamalı" sınıfını "cevaplayamadı"dan ayırmak için var (P2 etiketi).
_BLOCK_REASONS = frozenset({"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"})

_API_KEY_MSG = re.compile(r"api[ _-]?key|credential", re.IGNORECASE)

SYSTEM = (
    "Sen Türk mevzuatı üzerine bir asistansın. YALNIZCA sana verilen sayfa "
    "görüntülerindeki bilgiye dayanarak Türkçe yanıt ver. Aşağıda her sayfa "
    "görüntüsünün HEMEN ÖNÜNDE o sayfanın kaynak etiketi ([S1], [S2], ...) "
    "duruyor: bir etiket, ondan SONRA gelen görüntüye aittir. Her iddianın "
    "sonuna dayandığı sayfanın etiketini [S1] gibi ekle. Sayfalarda yanıt "
    "yoksa hiçbir şey uydurma ve yanıtında TAM OLARAK şu ifadeyi kullan: "
    f'"{HONEST_MISS_MARKER}". Sayfa dışı bilgi ekleme.'
)
# İfadenin TAM OLARAK istenmesi kasıtlı (canlı sondaj 2026-08-30): "açıkça
# '...' de" biçimindeki yumuşak yönerge ile model ıskayı KENDİ sözcükleriyle
# yazıyordu ("...bilgi bulunmamaktadır") ve sunucunun mühür araması —
# eskisi de yenisi de — bunu KAÇIRIYORDU. Dürüst ıska P2'nin hedef
# değişkeni olduğu için etiketin modelden deterministik biçimde geri gelmesi
# gerekir; uyumun garanti OLMADIĞI (S35 borcunun kalan yarısı) hâlâ doğru.


@dataclass
class GenResult:
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None


def source_marker(n: int, page: PageHit) -> str:
    """`[Sk] <doküman adı> sayfa <no>` — görüntünün HEMEN ÖNÜNE giden metin part'ı."""
    return f"[S{n}] {page.doc_name} sayfa {page.page_no}"


def build_prompt(question: str, pages: list[PageHit]) -> str:
    """Sistem istemi + soru. Kaynak LİSTESİ artık burada DEĞİL (B14).

    Eskiden istem sonuna `[S1] doküman, sayfa N` biçiminde bir liste
    yazılıyor, görüntüler ise ETİKETSİZ bir dizi olarak ayrıca gönderiliyordu
    (`contents=[*parts, prompt]`). Modelin k'ıncı görüntüyü `[Sk]` ile
    eşlemesi tamamen KONUMSAL çıkarımdı ve hiçbir yerde doğrulanmıyordu —
    yani ölçülecek "citation precision" aslında konumsal şansı ölçerdi
    (G2.2 önkoşulu). Etiketler artık `build_contents` ile görüntülerin
    ARASINA serpiştiriliyor; istem yalnız kuralı ve sayıyı söyler.
    """
    return (
        f"{SYSTEM}\n\nSana {len(pages)} sayfa görüntüsü verildi; her biri kendi "
        f"[S1]-[S{len(pages)}] etiketinin hemen ardından geliyor.\n\nSoru: {question}"
    )


def classify_error(exc: BaseException) -> str:
    """SDK/ağ istisnasını `answer.base.ERROR_TYPES` taksonomisine indirger.

    Sınıflandırma İSİMLE değil DAVRANIŞLA yapılır (HTTP kodu, timeout sınıfı):
    google-genai'nin iç istisna adları sürümler arasında değişir, ama "5xx =
    sağlayıcı kesintisi, retry et" kararı değişmez.
    """
    from google.genai import errors as genai_errors

    # EN BAŞTA ve APIError dalının DIŞINDA: `UnknownApiResponseError`
    # `ValueError`dan türer, `APIError`dan DEĞİL (`google/genai/errors.py`).
    # Kontrol APIError dalının içindeyken hiçbir zaman çalışmıyordu, yani
    # taksonomideki `parse` değeri üretimde ASLA üretilemiyordu — SDK ham
    # `json.JSONDecodeError`ı `_api_client.py`'de yakalayıp bu sınıfa
    # sarmaladığı için çıplak JSONDecodeError pratikte hiç dışarı sızmaz.
    #
    # Sıra ayrıca ikincil bir yanlış sınıflandırmayı da kapatıyor: bu istisna
    # ham gövdeyi mesajına gömer, dolayısıyla "API key" geçen bir hata sayfası
    # aşağıdaki `_API_KEY_MSG` desenine takılıp `auth` raporlardı.
    if isinstance(exc, genai_errors.UnknownApiResponseError):
        return "parse"
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None) or 0
        if code == 429:
            return "http_429"
        if code in (401, 403):
            return "auth"
        if 500 <= code < 600:
            return "http_5xx"
        return "other"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "parse"
    # httpx zaman aşımı hiyerarşisi (TimeoutError'dan türemez).
    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.HTTPError):
            return "other"
    except ImportError:  # pragma: no cover - httpx google-genai ile birlikte gelir
        pass
    # Anahtarsız/geçersiz-anahtarlı kurulum SDK'da düz bir ValueError'dır
    # ("No API key was provided..."). Kendiliğinden geçmeyen bir DAĞITIM
    # hatasıdır, "other" değil "auth" olarak raporlanmalı.
    if isinstance(exc, ValueError) and _API_KEY_MSG.search(str(exc)):
        return "auth"
    return "other"


class GeminiClient:
    """google-genai ince sarmalayıcısı.

    Tembel kurulum: __init__ yalnızca model+api_key saklar, SDK'ya dokunmaz —
    böylece anahtarsız `serve` çökmez (keyless boot). Gerçek genai.Client, ve
    onunla birlikte boş-anahtar hatası, yalnızca ilk generate() çağrısında
    oluşur; AskService'in degradation guard'ı bunu SERVICE_ERROR_TEXT'e çevirir.

    Kurulum artık KİLİTLİ (Y15/K33): iki eşzamanlı ilk `/ask` yavaş
    `from google import genai` import'unu istek yolunda ÇAKIŞARAK yapıyordu.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout_s: float = GEMINI_TIMEOUT_S,
        max_attempts: int = GEMINI_MAX_ATTEMPTS,
        backoff_s: float = GEMINI_RETRY_BACKOFF_S,
        total_budget_s: float = GEMINI_TOTAL_BUDGET_S,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.total_budget_s = total_budget_s
        self._client = None
        self._client_lock = threading.Lock()

    def _ensure_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from google import genai
                    from google.genai import types

                    # HttpOptions.timeout MİLİSANİYE cinsindedir (SDK sözleşmesi;
                    # `_api_client` bunu 1000'e bölüp httpx'e verir). httpx skaler
                    # değeri FAZ BAŞINA yorumlar (connect/read/write/pool ayrı
                    # sayaçlar) — toplam duvar-saati sınırı `generate()`teki bütçe
                    # kontrolüdür, bu değer değil.
                    self._client = genai.Client(
                        api_key=self.api_key,
                        http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
                    )
        return self._client

    def build_contents(self, prompt: str, images: list[bytes], markers: list[str]) -> list:
        """`[Sk]` metni → k'ıncı görüntü → ... → istem (B14).

        Sıra SÖZLEŞMEDİR: her görüntünün ÖNÜNDE kendi etiketi durur, istem en
        SONA gelir. Böylece `[Sk]` ile görüntü arasındaki bağ konumsal bir
        varsayım olmaktan çıkıp modele açıkça söylenmiş bir olgu olur.
        """
        from google.genai import types

        if len(markers) != len(images):
            raise ValueError(f"markers ({len(markers)}) ve images ({len(images)}) eşleşmiyor")
        contents: list = []
        for marker, img in zip(markers, images, strict=True):
            contents.append(marker)
            contents.append(types.Part.from_bytes(data=img, mime_type="image/webp"))
        contents.append(prompt)
        return contents

    def generate(self, prompt: str, images: list[bytes], markers: list[str]) -> GenResult:
        contents = self.build_contents(prompt, images, markers)
        try:
            client = self._ensure_client()
        except Exception as exc:
            # Kurulum hatası da taksonomiye girer (anahtarsız dağıtım -> "auth");
            # aksi halde bu sınıf `AskService`te "other"a düşer ve telemetride
            # ağ kesintisinden ayırt edilemezdi.
            raise AnswererError(classify_error(exc), f"gemini istemcisi kurulamadı: {exc}") from exc
        last: AnswererError | None = None
        started = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = client.models.generate_content(model=self.model, contents=contents)
            except Exception as exc:
                error_type = classify_error(exc)
                last = AnswererError(error_type, f"gemini {error_type}: {exc}")
                last.__cause__ = exc
                if error_type not in RETRYABLE_ERROR_TYPES or attempt == self.max_attempts:
                    raise last from exc
                # BÜTÇE İNVARİANTI (ölçülen, varsayılan değil): httpx'in faz
                # başına sayaçları yüzünden tek bir deneme ilan edilen
                # `timeout_s`i aşabilir (canlı: 16,2 sn / 15 sn). Bir sonraki
                # denemenin EN KÖTÜ hâli kalan bütçeye sığmıyorsa retry
                # YAPILMAZ — aksi halde "<= 35 sn" bir tahmin olarak kalırdı.
                elapsed = time.monotonic() - started
                if elapsed + self.backoff_s + self.timeout_s > self.total_budget_s:
                    logger.warning(
                        "gemini çağrısı başarısız (%s); %.1f sn geçti, kalan bütçe "
                        "(%.1f sn) bir deneme daha kaldırmıyor — yeniden denenmiyor",
                        error_type,
                        elapsed,
                        max(0.0, self.total_budget_s - elapsed),
                    )
                    annotate("gemini_retry_skipped_budget", True)
                    raise last from exc
                logger.warning(
                    "gemini çağrısı başarısız (%s), %s sn sonra tek yeniden deneme",
                    error_type,
                    self.backoff_s,
                )
                annotate("gemini_retried", True)
                time.sleep(self.backoff_s)
                continue
            text = resp.text or ""
            if not text and _block_reason(resp):
                raise AnswererError("safety_block", f"gemini bloğu: {_block_reason(resp)}")
            usage = getattr(resp, "usage_metadata", None)
            return GenResult(
                text=text,
                tokens_in=getattr(usage, "prompt_token_count", None),
                tokens_out=getattr(usage, "candidates_token_count", None),
            )
        raise last or AnswererError("other", "gemini: deneme yapılamadı")  # pragma: no cover


def _block_reason(resp) -> str | None:
    """Boş yanıtın güvenlik/politika kaynaklı olup olmadığı (varsa sebep adı)."""
    feedback = getattr(resp, "prompt_feedback", None)
    reason = getattr(feedback, "block_reason", None)
    if reason is not None:
        return str(getattr(reason, "name", reason))
    for cand in getattr(resp, "candidates", None) or []:
        fr = getattr(cand, "finish_reason", None)
        name = str(getattr(fr, "name", fr)) if fr is not None else None
        if name in _BLOCK_REASONS:
            return name
    return None


class GeminiAnswerer:
    def __init__(self, model: str, api_key: str, client=None):
        self._client = client or GeminiClient(model, api_key)

    def answer(
        self, question: str, pages: list[PageHit], image_loader: Callable[[str], bytes]
    ) -> Answer:
        prompt = build_prompt(question, pages)
        images = [image_loader(p.image_path) for p in pages]
        markers = [source_marker(i + 1, p) for i, p in enumerate(pages)]
        gen = self._client.generate(prompt, images, markers)
        text = gen.text
        if gen.tokens_in is not None:
            annotate("tokens_in", gen.tokens_in)
        if gen.tokens_out is not None:
            annotate("tokens_out", gen.tokens_out)
        idxs = {int(m) for m in re.findall(r"\[S(\d+)\]", text)}
        # Atıf YOKSA atıf yok. Eskiden burada top-1 sayfayı otomatik atıf olarak
        # ekleyen bir fallback vardı; model doğru biçimde "verilen sayfalarda
        # bulamadım" dediğinde bile uydurma bir "dayanak" üretiyordu (P2 / G2.7).
        citations = [pages[i - 1].page_id for i in sorted(idxs) if 0 < i <= len(pages)]
        return Answer(text=text, citations=citations)
