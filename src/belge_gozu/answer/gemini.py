import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from belge_gozu.answer.base import HONEST_MISS_MARKER, Answer, AnswererError
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import annotate, merge_note, note

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
#
# DİKKAT: bu küme AYNI ANAHTAR üzerindeki yeniden denemeyi yönetir. ÖBÜR
# anahtara geçme kararı ayrı bir kümededir (`NON_ROTATABLE_ERROR_TYPES`) ve
# çok daha geniştir: 429 aynı anahtarda umutsuzdur ama BAŞKA bir anahtarda
# tam olarak umut vaat eden hatadır.
RETRYABLE_ERROR_TYPES = frozenset({"timeout", "http_5xx"})

# --- ANAHTAR ROTASYONU -------------------------------------------------------
#
# Anahtar SLOTLARININ etiketleri. Bu iki dize, anahtarlar hakkında koda,
# loglara, telemetriye ve testlere yazılması SERBEST olan TEK bilgidir —
# anahtar DEĞERİ (ya da parçası, uzunluğu, parmakizi) hiçbir yere yazılmaz.
# Sıra SÖZLEŞMEDİR: "key1" birincil anahtar (`Settings.gemini_api_key`),
# "key2" yedek (`Settings.google_api_key_2`). Etiket havuzdaki KONUMA değil
# SLOTA bağlıdır: birincil boş bırakılıp yalnız yedek doldurulursa havuz tek
# elemanlıdır ve o elemanın etiketi yine "key2"dir.
KEY_LABELS: tuple[str, ...] = ("key1", "key2")

# Anahtar değiştirmenin DÜZELTEMEYECEĞİ tek hata sınıfı. Kullanıcının kuralı
# "HERHANGİ bir API hatasında öbür anahtarı dene"dir ve bu katman onu taşıma
# (transport/API) düzeyinde birebir uygular; `parse` ise BAŞARILI bir yanıtın
# sınıflandırmasıdır — HTTP 200 geldi, gövdesi okunamadı. Aynı istek başka bir
# anahtarla gönderilirse aynı gövde geri gelir: rotasyon yalnız ikinci bir
# çağrının parasını ve süresini harcar.
#
# `safety_block` bilinçle İÇERİDE: sonucu değiştirmesi beklenmez (model
# politikası anahtardan bağımsızdır) ama zararsızdır, kullanıcının kuralına
# uyar ve TEK rotasyonla sınırlıdır — döngü yoktur.
NON_ROTATABLE_ERROR_TYPES = frozenset({"parse"})

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
        return self._generate(self.build_contents(prompt, images, markers), None)

    def generate_json(self, prompt: str, schema: dict | None = None) -> GenResult:
        """YALNIZ METİN + yapılandırılmış JSON çıktı (P2 doğrulayıcısı).

        `temperature=0`: doğrulayıcı bir YARGIÇTIR, üretici değil — aynı iddia
        + aynı kanıt aynı kararı vermeli, aksi halde sha256 önbelleği bir
        rastgele seçimi kalıcılaştırırdı.

        Görüntü GÖNDERİLMEZ (bilinçli sapma, gerekçesi `answer/verify.py`
        modül açıklamasında): kanıt, sıralamayı üreten metin kanalının ta
        kendisidir. `generate()` ile AYNI retry/zaman aşımı/bütçe
        invariantından geçer — ikinci bir HTTP politikası yoktur.
        """
        return self._generate([prompt], self._json_config(schema))

    def _json_config(self, schema: dict | None):
        """SDK'nın yapılandırılmış-çıktı yapılandırması, kurulamıyorsa None.

        None dönmek bir arıza DEĞİL, tanımlı bir geri düşüştür: istem zaten
        JSON istiyor ve `verify.parse_verdict` şema-dışı çıktıyı katı regex ile
        ayrıştırıp, ayrıştıramazsa `belirsiz` sayıyor.
        """
        try:
            from google.genai import types

            return types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            )
        except Exception:
            logger.warning(
                "google-genai yapılandırılmış çıktı yapılandırması kurulamadı; "
                "düz JSON istemi + katı regex ayrıştırmasına düşülüyor"
            )
            return None

    def budget_fits(self, started: float, extra_sleep: float = 0.0) -> bool:
        """Kalan toplam bütçe bir deneme DAHA kaldırıyor mu? (ölçülen invariant.)

        `started` isteğin duvar-saati başlangıcıdır ve İSTEK BOYUNCA
        paylaşılır — anahtar rotasyonunda ikinci anahtarın denemesi de aynı
        tavanın altındadır, kendi 35 sn'sini açmaz.
        """
        elapsed = time.monotonic() - started
        return elapsed + extra_sleep + self.timeout_s <= self.total_budget_s

    def _generate(
        self,
        contents: list,
        config,
        *,
        started: float | None = None,
        max_attempts: int | None = None,
    ) -> GenResult:
        """Tek anahtar üzerinde deneme serisi (retry + bütçe invariantı).

        İki kwarg YALNIZ rotasyon katmanı (`RotatingGeminiClient`) içindir:
        `started` iki anahtarın denemelerini TEK bir duvar-saati bütçesinde
        toplar, `max_attempts` toplam deneme tavanını (3) bölüştürür. Kwarg
        verilmezse davranış rotasyon katmanı yokken olduğu gibidir.
        """
        try:
            client = self._ensure_client()
        except Exception as exc:
            # Kurulum hatası da taksonomiye girer (anahtarsız dağıtım -> "auth");
            # aksi halde bu sınıf `AskService`te "other"a düşer ve telemetride
            # ağ kesintisinden ayırt edilemezdi.
            raise AnswererError(classify_error(exc), f"gemini istemcisi kurulamadı: {exc}") from exc
        last: AnswererError | None = None
        started = time.monotonic() if started is None else started
        attempts = self.max_attempts if max_attempts is None else max_attempts
        for attempt in range(1, attempts + 1):
            try:
                # `config` YOKSA kwarg hiç geçilmez: bayrak-kapalı yanıtlama
                # yolu SDK'ya birebir eskisi gibi görünsün.
                resp = (
                    client.models.generate_content(model=self.model, contents=contents)
                    if config is None
                    else client.models.generate_content(
                        model=self.model, contents=contents, config=config
                    )
                )
            except Exception as exc:
                error_type = classify_error(exc)
                last = AnswererError(error_type, f"gemini {error_type}: {exc}")
                last.__cause__ = exc
                if error_type not in RETRYABLE_ERROR_TYPES or attempt == attempts:
                    raise last from exc
                # BÜTÇE İNVARİANTI (ölçülen, varsayılan değil): httpx'in faz
                # başına sayaçları yüzünden tek bir deneme ilan edilen
                # `timeout_s`i aşabilir (canlı: 16,2 sn / 15 sn). Bir sonraki
                # denemenin EN KÖTÜ hâli kalan bütçeye sığmıyorsa retry
                # YAPILMAZ — aksi halde "<= 35 sn" bir tahmin olarak kalırdı.
                elapsed = time.monotonic() - started
                if not self.budget_fits(started, self.backoff_s):
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


class StickyKeyIndex:
    """SÜREÇ DÜZEYİNDE "şu an hangi anahtar" göstergesi (0-tabanlı slot no).

    YAPIŞKAN: bir rotasyondan sonra yeni anahtar GEÇERLİ anahtar olur ve
    sonraki istekler doğrudan onunla başlar. Alternatif ("her istek key1 ile
    başlasın, hata alırsa key2'ye geçsin") kotası tükenmiş bir anahtarda istek
    BAŞINA garantili bir başarısız çağrı demektir — gecikme, kota ve gürültü,
    hepsi bedavaya. Gösterge istek başına değil SÜREÇ başına tutulduğu için
    yanıtlayıcı ile doğrulayıcı da aynı kararı paylaşır (ikisi de aynı tek
    fabrikadan geçer).

    Kilit ZORUNLU: senkron uç noktalar Starlette'in iş parçacığı havuzunda
    koşar, yani bu gerçekten paylaşılan mutable durumdur. Eşzamanlı iki istek
    birbirinin yazdığını İYİ HUYLU biçimde ezebilir (ikisi de sıradaki aynı
    anahtara gider); garanti edilen, göstergenin her an GEÇERLİ bir slot
    numarası olmasıdır — yarım yazılmış bir durum yoktur.
    """

    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._lock = threading.Lock()

    def current(self, pool_size: int) -> int:
        """Geçerli slot; havuz boyutuna göre daima güvenli (`% pool_size`)."""
        with self._lock:
            return self._index % max(1, pool_size)

    def set(self, index: int) -> None:
        with self._lock:
            self._index = index


# Süreç düzeyindeki TEK gösterge. Testler kendi örneğini enjekte eder
# (`RotatingGeminiClient(..., sticky=StickyKeyIndex())`) — global durum
# testler arasında sızmasın diye.
_STICKY = StickyKeyIndex()


@dataclass(frozen=True)
class KeySlot:
    """Bir anahtar slotu: ETİKET + o anahtarla kurulmuş istemci.

    Anahtarın DEĞERİ yalnız `client` içinde yaşar; dışarıya (log, telemetri,
    istisna mesajı, test) çıkan tek şey `label`dır ("key1"/"key2").
    """

    label: str
    client: GeminiClient


class RotatingGeminiClient:
    """1-2 anahtarlık havuz: API katmanındaki HERHANGİ bir hatada öbür anahtar.

    DENEME MERDİVENİ (istek başına EN FAZLA 3 çağrı, hepsi TEK duvar-saati
    bütçesinin altında — `GEMINI_TOTAL_BUDGET_S`):

    | # | anahtar | hangi koşulda                                              |
    |---|---------|------------------------------------------------------------|
    | 1 | geçerli | HER ZAMAN                                                  |
    | 2 | öbürü   | 1 hata verdi VE sınıf `parse` DEĞİL VE ikinci anahtar VAR   |
    |   |         | VE kalan bütçe bir deneme daha kaldırıyor                  |
    | 3 | öbürü   | 2 hata verdi VE sınıf retry'lenebilir (timeout/http_5xx)    |
    |   |         | VE kalan bütçe yetiyor (`GeminiClient`in KENDİ retry'si)   |

    Yani aynı anahtardaki yeniden deneme hakkı SON anahtara bırakılır:
    rotasyondan sonra gidecek başka yer kalmadığı için mevcut retry
    semantiği orada anlamlıdır, ilk anahtarda ise "öbür anahtarı dene" her
    zaman daha iyi bir ikinci hamledir.

    HAVUZ TEK ANAHTARLIYSA rotasyon dalı HİÇ girilmez ve davranış bu katman
    eklenmeden önceki `GeminiClient.generate()` ile birebir aynıdır (deneme
    sayısı, backoff, taksonomi, notlar) — mevcut testler bunun kilididir.
    """

    def __init__(self, slots: list[KeySlot], sticky: StickyKeyIndex | None = None) -> None:
        if not slots:
            raise ValueError("anahtar havuzu boş olamaz")
        self._slots = slots
        self._sticky = sticky or _STICKY

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self._slots]

    def generate(self, prompt: str, images: list[bytes], markers: list[str]) -> GenResult:
        # Argüman kurulumu anahtardan BAĞIMSIZDIR ve BİR KEZ yapılır: rotasyon
        # aynı isteği gönderir, yeniden kurulmuş bir isteği değil (görüntü
        # part'ları da yeniden paketlenmez).
        return self._run(self._slots[0].client.build_contents(prompt, images, markers), None)

    def generate_json(self, prompt: str, schema: dict | None = None) -> GenResult:
        return self._run([prompt], self._slots[0].client._json_config(schema))

    def _run(self, contents: list, config) -> GenResult:
        started = time.monotonic()
        n = len(self._slots)
        i = self._sticky.current(n)
        first = self._slots[i]
        if n == 1:
            return self._served(first, first.client._generate(contents, config, started=started))
        try:
            out = first.client._generate(contents, config, started=started, max_attempts=1)
        except AnswererError as exc:
            if not self._may_rotate(first, exc, started):
                raise
            error_type = exc.error_type
        else:
            return self._served(first, out)

        j = (i + 1) % n
        second = self._slots[j]
        logger.warning(
            "gemini %s ile başarısız (%s) — istek %s ile bir kez yeniden deneniyor",
            first.label,
            error_type,
            second.label,
        )
        _note_rotation(first.label, error_type)
        # Gösterge ROTASYON ANINDA taşınır, ikinci anahtarın sonucundan
        # BAĞIMSIZ: birinci anahtar (kota/kimlik) bu süreçte artık şüphelidir,
        # bir sonraki isteğin ona yeniden çarpması için sebep yok.
        self._sticky.set(j)
        try:
            return self._served(second, second.client._generate(contents, config, started=started))
        except AnswererError as exc2:
            tried = [first.label, second.label]
            merge_note("llm", keys_tried=tried)
            # Taksonomi SON hatanınkidir (`AskService` bunu `events.error_type`e
            # yazar); mesaj hangi anahtarların denendiğini taşır — iki anahtarlı
            # bir kesinti ile tek anahtarlı bir kesinti log'da ayırt edilebilsin.
            raise AnswererError(
                exc2.error_type, f"{exc2} [denenen anahtarlar: {', '.join(tried)}]"
            ) from exc2

    def _may_rotate(self, slot: KeySlot, exc: AnswererError, started: float) -> bool:
        if exc.error_type in NON_ROTATABLE_ERROR_TYPES:
            logger.warning(
                "gemini %s: '%s' sınıfı anahtar değiştirmekle düzelmez (yanıt GELDİ, "
                "okunamadı) — rotasyon yapılmıyor",
                slot.label,
                exc.error_type,
            )
            return False
        if not slot.client.budget_fits(started):
            # Bütçe invariantı rotasyonu da KAPSAR: "öbür anahtarı da dene"
            # ikinci bir zaman aşımı penceresi açar ve toplam tavan aşılırdı.
            logger.warning(
                "gemini %s başarısız (%s); kalan bütçe bir deneme daha kaldırmıyor "
                "— rotasyon yapılmıyor",
                slot.label,
                exc.error_type,
            )
            annotate("gemini_rotation_skipped_budget", True)
            return False
        return True

    def _served(self, slot: KeySlot, out: GenResult) -> GenResult:
        merge_note("llm", key=slot.label)
        return out


def _note_rotation(from_label: str, error_type: str) -> None:
    """`detail.llm.rotations` — `{"from": ..., "error_type": ...}`, SIRAYLA.

    Liste (sayaç değil): bir istekte birden çok rotasyon olabilir (bayrak
    açıkken doğrulayıcı iddia başına çağrı yapar) ve sıra kendi başına bilgi
    taşır. `telemetry/prom.py` bu listeden `bg_llm_key_rotations_total`ı besler.

    HATA SINIFI da yazılır çünkü BAŞKA HİÇBİR YERDE GÖRÜNMEZ: rotasyon başarılı
    olduğunda istek `answered` biter ve `events.error_type` NULL kalır — yani
    "key1 kotası doldu" (`http_429`, kendiliğinden geçer) ile "key1 iptal
    edildi" (`auth`, insan müdahalesi ister) telemetride ayırt edilemezdi.
    Operatörün rotasyon karşısında sorduğu ilk soru tam olarak budur.
    """
    cur = note("llm")
    rotations = list(cur.get("rotations") or []) if isinstance(cur, dict) else []
    rotations.append({"from": from_label, "error_type": error_type})
    merge_note("llm", rotations=rotations)


def build_gemini_client(model: str, api_key: str, api_key_2: str = "") -> RotatingGeminiClient:
    """Gemini istemcisinin TEK kurulum noktası (+ anahtar havuzu).

    Bu projede Gemini'nin İKİ tüketicisi var: yanıtlayıcı (`GeminiAnswerer`,
    görüntü + metin) ve P2 kanıt doğrulayıcısı (`answer/verify.py`, yalnız
    metin + JSON şeması). İkisi de kendi `GeminiClient(...)` çağrısını yaparsa
    istemci düzeyindeki her politika (zaman aşımı, retry, ANAHTAR SEÇİMİ) iki
    yerde ayrı ayrı uygulanmak zorunda kalır ve biri sessizce geride kalır.
    Kurulum burada TOPLANIR: istemci davranışını değiştiren her katman bu tek
    fonksiyonu sarmalayarak her iki tüketiciyi birden kapsar — anahtar
    rotasyonu bu tekilliğin ilk gerçek müşterisidir.

    `genai.Client`in kendisi zaten tek yerde kuruluyor
    (`GeminiClient._ensure_client`, tembel + kilitli); bu fonksiyon SARMALAYICI
    düzeyinin aynı tekilliğidir. Her slot KENDİ tembel `GeminiClient`ini alır:
    ikinci anahtarın SDK istemcisi ilk rotasyona kadar hiç kurulmaz.
    """
    slots = [
        KeySlot(label, GeminiClient(model, key))
        for label, key in zip(KEY_LABELS, (api_key, api_key_2), strict=True)
        if key
    ]
    if not slots:
        # ANAHTARSIZ ÖNYÜKLEME (keyless boot) korunur: havuz tek slotlu kalır
        # ve boş anahtar hatası eskisi gibi ilk `generate()` çağrısında `auth`
        # olarak patlar. Kurulum zamanında patlamak `serve`i çökertirdi.
        slots = [KeySlot(KEY_LABELS[0], GeminiClient(model, api_key))]
    return RotatingGeminiClient(slots)


class GeminiAnswerer:
    def __init__(self, model: str, api_key: str, client=None, *, api_key_2: str = ""):
        self._client = client or build_gemini_client(model, api_key, api_key_2)

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
