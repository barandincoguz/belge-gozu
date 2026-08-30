"""İddia doğrulayıcı (P2 T1) + kanıt kapısı (T2'nin ikinci kapısı).

Modül ÜÇ YARIMDIR ve ayrım bilinçlidir:

1. **DETERMİNİSTİK** — `segment_claims`: kota yakmaz, ağ istemez, testlerde
   birebir sabitlenebilir. Yanıt metnini cümle düzeyinde iddialara böler ve her
   iddiaya KENDİ `[Sn]` atıflarını bağlar.
2. **LLM** — `verify_claim`: bir iddiayı YALNIZ atıf yaptığı sayfaların METNİNE
   karşı yargılar. sha256 önbellekli, bütçe korumalı, şema-ayrıştırmalı.
3. **KAPI** — `EvidenceGate`: yanıtın TÜM iddiaları desteklenmiyorsa yanıtı
   düşürür (ilke 20). `AskService` bunu duck-typed bir nesne olarak alır.

PLANDAN İKİ SAPMA (gerekçeleriyle):

* **Görüntü KULLANILMIYOR.** Plan (`p2-selective-answering.md:163`) doğrulayıcı
  isteminde "metin yoksa sayfa görüntüsü eklenir" diyordu. Bugün metin kanalı
  (`page_texts.parquet`) ZATEN sıralamayı üreten kanaldır ve doğrulayıcının
  aynı metne bakması iki şey kazandırır: (a) kota — beş WebP görüntüsü bir
  doğrulama çağrısını answerer kadar pahalı yapardı ve doğrulayıcı iddia başına
  çağrılır; (b) determinizm — önbellek anahtarı metnin sha256'sıdır, görüntü
  baytları üzerinde aynı garanti kurulamaz. Bedeli dürüstçe: metin katmanı BOŞ
  olan taranmış sayfalarda doğrulayıcı "belirsiz" der ve yanıt düşer — yani
  sistem o sayfalarda SESSİZCE değil, GÖRÜNÜR biçimde çekimser kalır.
* **Tek çağrı yerine iddia başına çağrı.** Plan `:163` "tek API çağrısı (claim
  başına değil)" diyordu. İddia başına çağrı, önbelleğin gerçekten işe
  yaramasının ÖN KOŞULUDUR: yanıt metninin tek bir cümlesi değişince toplu
  istemin anahtarı da değişir ve TÜM iddialar yeniden ödenir. İddia bazlı
  anahtar, tekrar koşumlarda değişmeyen iddiaları bedava yapar (aynı soru
  yeniden sorulduğunda tüm iddialar isabet eder). Üst sınır
  `Settings.verifier_max_claims` ile kapatılıdır.

`EvidenceUnit`/`EvidencePack` (plan `:64`) BURADA TANIMLANMAZ ve
`retrieval/evidence.py`'den İTHAL EDİLMEZ: o dosya hiç yazılmadı (R23 ile P1
daraltıldı, `p2-reality-audit.md:49`). Kanıt yüzeyi bugünkü gerçek arayüzdür:
`list[PageHit]` + `page_texts` eşlemesi.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel

from belge_gozu.answer.base import Answer, is_honest_miss
from belge_gozu.retrieval.text import tr_lower

if TYPE_CHECKING:  # pragma: no cover - yalnız tip denetimi
    from belge_gozu.config import Settings
    from belge_gozu.retrieval.types import PageHit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. TİPLER
# ---------------------------------------------------------------------------

Verdicts = Literal["supported", "unsupported", "belirsiz"]

# Kapalı küme. `telemetry/prom.py` etiket değerlerini BURADAN okur — ikinci bir
# kopya tutulmaz (duplike sözleşme denetimi, 2026-08-29).
VERDICTS: tuple[str, ...] = ("supported", "unsupported", "belirsiz")

# "supported DIŞINDAKİ HER ŞEY desteklenmemiştir" (G2.1 yönü / şüphede-reddet).
# `belirsiz` bilinçle bu tarafta: bir iddiayı doğrulayamamak, onu doğrulamış
# saymak için gerekçe değildir.
SUPPORTED: str = "supported"


class Claim(BaseModel):
    """Yanıt metninden çıkarılmış tek bir iddia + KENDİ atıfları.

    `cited_sources` 1-tabanlı `[Sn]` numaralarıdır — servis edilen sayfa
    listesindeki (`hits`) konuma karşılık gelir, sayfa kimliğine değil:
    dönüştürme kapının işidir (`EvidenceGate`), çünkü yalnız orada `hits` var.
    """

    claim_id: str
    text: str
    cited_sources: list[int] = []
    inherited_sources: bool = False
    """Atıflar cümlenin KENDİSİNDEN değil, paragrafından devralındıysa True."""


class Verdict(BaseModel):
    claim_id: str
    verdict: Verdicts
    gerekce: str = ""
    cached: bool = False
    """Bu karar önbellekten mi geldi?"""
    llm_called: bool = False
    """GERÇEKTEN bir API çağrısı yapıldı mı?

    `not cached` ile AYNI ŞEY DEĞİLDİR ve fark kota muhasebesinde ölçüldü
    (canlı sonda, 2026-08-30): atıfsız bir iddia ne önbellekten gelir ne de
    API'ye gider — `cached=False, llm_called=False`. `llm_calls`'ı
    `not cached` üzerinden saymak o iddiayı bir çağrı gibi raporluyordu.
    """

    @property
    def supported(self) -> bool:
        return self.verdict == SUPPORTED


# ---------------------------------------------------------------------------
# 2. CÜMLE/İDDİA BÖLÜMLEME — deterministik, Türkçe-farkında
# ---------------------------------------------------------------------------

_MARKER = re.compile(r"\[\s*S\s*(\d+)\s*\]")

# Sonrasında cümle BAŞLAMAYAN nokta: Türkçe hukuk metinlerinin kısaltmaları.
# `m.` (madde) listenin en önemli üyesi — "TMK m. 19" bir cümle sınırı DEĞİL.
_ABBREVS = frozenset(
    {
        "m", "md", "mad", "mdd", "f", "fk", "bkz", "krş", "vb", "vs", "vd",
        "örn", "ör", "s", "sf", "no", "nu", "age", "yy", "dr", "doç",
        "prof", "av", "sy", "c", "b", "bt", "yön", "rg",
    }
)  # fmt: skip

# Cümle sonu ADAYI: noktalama + (varsa) kapanış işaretleri + boşluk.
# Boşluk ZORUNLU — "2.806,50" ve "m.19" gibi noktalar bu yüzden hiç aday olmaz.
_CANDIDATE = re.compile(r"[.!?…]+[\)\]»\"'”’]*\s+")
# Sınırdan SONRA gelmesi gereken: (açılış işareti/atıf köşeli parantezi) +
# büyük harf ya da rakam. Türkçe büyük harfler AÇIKÇA yazılır — `str.isupper()`
# yerine sınıf kullanılır ki "İ"/"Ş"/"Ğ" atlanmasın.
_NEXT_STARTS = re.compile(r"[\"'“«(\[]*[A-ZÇĞİÖŞÜ0-9]")
_PREV_TOKEN = re.compile(r"([^\s.!?…]+)$")
# Madde/bent listeleri ("- ", "1) ", "• ") kendi başlarına bir iddia birimidir.
_LIST_START = re.compile(r"^\s*(?:[-*•·–—]|\(?\d+[.)]|[a-zçğıöşü]\))\s+")

# Bu uzunluğun altındaki parçalar ÖNCEKİ cümleye eklenir: "Ancak." tek başına
# bir iddia değildir ve doğrulayıcıya gönderilmesi bir çağrı israfıdır.
MIN_CLAIM_CHARS = 15


def _strip_markers(text: str) -> str:
    """`[Sn]` işaretlerini atar ve işaretin BIRAKTIĞI boşluğu toplar.

    İkinci adım kozmetik değil: iddia metni önbellek ANAHTARINA girer
    ("... yerdir ." ile "... yerdir." iki farklı anahtar olurdu) ve
    doğrulayıcının istemine birebir yazılır.
    """
    out = re.sub(r"\s+", " ", _MARKER.sub(" ", text)).strip()
    return re.sub(r"\s+([.,;:!?…»\)\]])", r"\1", out)


def _sources(text: str) -> list[int]:
    return sorted({int(m) for m in _MARKER.findall(text)})


def _is_boundary(block: str, match: re.Match[str]) -> bool:
    """Aday nokta gerçek bir cümle sınırı mı? (Türkçe kısaltma/sayı korumaları.)"""
    if not _NEXT_STARTS.match(block, match.end()):
        return False
    prev = _PREV_TOKEN.search(block[: match.start()])
    if prev is None:
        return True
    token = prev.group(1)
    # "320. maddesi" zaten yukarıdaki küçük-harf kontrolüne takılır; ama
    # "2. Madde" ve numaralı liste başlıkları takılmaz — sayı ile biten bir
    # token'dan sonra BÖLMEYİZ.
    if token.isdigit():
        return False
    if len(token) == 1 and token.isalpha():
        return False
    return tr_lower(token) not in _ABBREVS


def _split_sentences(block: str) -> list[str]:
    out: list[str] = []
    start = 0
    for m in _CANDIDATE.finditer(block):
        if not _is_boundary(block, m):
            continue
        out.append(block[start : m.start()] + m.group(0).rstrip())
        start = m.end()
    tail = block[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in (s.strip() for s in out) if s]


def _blocks(paragraph: str) -> list[str]:
    """Paragrafı iddia birimlerine böler: madde listeleri kendi satırındadır."""
    blocks: list[str] = []
    for line in paragraph.splitlines():
        if not line.strip():
            continue
        if blocks and not _LIST_START.match(line):
            blocks[-1] = f"{blocks[-1]} {line.strip()}"
        else:
            blocks.append(line.strip())
    return blocks


def _merge_fragments(sentences: Sequence[str]) -> list[str]:
    """`MIN_CLAIM_CHARS` altındaki parçaları ÖNCEKİ cümleye ekler."""
    out: list[str] = []
    for sent in sentences:
        if out and len(_strip_markers(sent)) < MIN_CLAIM_CHARS:
            out[-1] = f"{out[-1]} {sent}"
        else:
            out.append(sent)
    return out


def segment_claims(answer_text: str) -> list[Claim]:
    """Yanıt metni -> `c1, c2, ...` iddiaları (deterministik; LLM/ağ YOK).

    Türkçe-farkındalık ÖLÇÜLEBİLİR üç davranıştır:
      * `m.19` / `m. 19` madde kısaltması cümle BÖLMEZ (kısaltma listesi),
      * `2.806,50 TL` gibi sayılar cümle BÖLMEZ (nokta boşluk görmüyor),
      * `320. maddesinde` ordinali cümle BÖLMEZ (sonrası küçük harf; ayrıca
        rakamla biten token'dan sonra hiç bölünmez).

    ATIF BAĞI (`cited_sources`): bir iddianın atıfları KENDİ cümlesindeki
    `[Sn]` işaretleridir; cümlesinde hiç işaret yoksa PARAGRAFINDAN devralır
    (`inherited_sources=True`). Devralma, modelin atfı paragrafın sonunda tek
    seferde vermesi durumunu kurtarır; devralınacak işaret de yoksa iddia
    atıfsız kalır ve kanıt kapısı onu çağrı YAPMADAN `belirsiz` sayar.
    """
    claims: list[Claim] = []
    for paragraph in re.split(r"\n\s*\n", answer_text or ""):
        if not paragraph.strip():
            continue
        para_sources = _sources(paragraph)
        for block in _blocks(paragraph):
            for sent in _merge_fragments(_split_sentences(block)):
                text = _strip_markers(sent)
                if not text or not any(ch.isalnum() for ch in text):
                    continue
                own = _sources(sent)
                claims.append(
                    Claim(
                        claim_id=f"c{len(claims) + 1}",
                        text=text,
                        cited_sources=own or para_sources,
                        inherited_sources=not own and bool(para_sources),
                    )
                )
    return claims


# ---------------------------------------------------------------------------
# 3. İSTEM + AYRIŞTIRMA
# ---------------------------------------------------------------------------

# Önbellek anahtarına GİREN sürüm. İstem metni ya da şema değişirse BU DİZE DE
# değişmek zorundadır — aksi halde eski istemle üretilmiş kararlar yeni istemin
# kararıymış gibi sessizce yeniden kullanılır (p2-reality-audit §4.4, kusur 2).
PROMPT_VERSION = "verify-v1"

# Kanıt metni üst sınırı (karakter). Bir mevzuat sayfası ~2-4k karakter; iddia
# başına 1-2 sayfa atıf tipiktir. Tavan, atıfsız/aşırı atıflı bir yanıtın
# istemi (ve faturayı) sınırsız büyütmesini engeller.
EVIDENCE_CHAR_LIMIT = 12_000
_TRUNCATED = "\n[... kanıt metni kısaltıldı ...]"

VERIFIER_PROMPT = """Sen bir KANIT DENETÇİSİSİN. Tek işin, verilen İDDİA'nın \
aşağıdaki KANIT metinlerinde desteklenip desteklenmediğine karar vermek.

KURALLAR:
- YALNIZCA KANIT metnine bak. Genel hukuk bilgini, ezberini veya dış kaynakları KULLANMA.
- İddia kanıtta açıkça yazıyorsa ya da kanıttan tartışmasız çıkıyorsa: "supported".
- İddia kanıtla çelişiyorsa ya da kanıtta hiç geçmiyorsa: "unsupported".
- Kanıt eksik, kesik ya da ilgisiz olduğu için karar veremiyorsan: "belirsiz".
- ŞÜPHEDE REDDET: emin değilsen "supported" DEME.

Yanıtın YALNIZCA şu JSON nesnesi olsun, başka hiçbir şey yazma:
{{"verdict": "supported|unsupported|belirsiz", "gerekce": "<tek cümlelik Türkçe gerekçe>"}}

KANIT:
<<<
{evidence}
>>>

İDDİA: {claim}
"""

# `google-genai` yapılandırılmış çıktı şeması. SDK bunu desteklemiyorsa
# (ya da istemci bir stub'sa) `parse_verdict` katı regex'e düşer.
VERDICT_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": list(VERDICTS)},
        "gerekce": {"type": "STRING"},
    },
    "required": ["verdict", "gerekce"],
}

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_VERDICT_RE = re.compile(
    r"[\"']?verdict[\"']?\s*[:=]\s*[\"']?(supported|unsupported|belirsiz)\b", re.IGNORECASE
)
_GEREKCE_RE = re.compile(r"[\"']?gerekce[\"']?\s*[:=]\s*[\"']([^\"']*)[\"']", re.IGNORECASE)


def build_evidence(evidence_texts: Sequence[str]) -> str:
    """Kanıt metinlerini numaralı bloklara dizer ve tavana kırpar."""
    parts = []
    for i, text in enumerate(evidence_texts, start=1):
        body = (text or "").strip() or "(bu sayfanın metin katmanı boş)"
        parts.append(f"--- KANIT {i} ---\n{body}")
    joined = "\n\n".join(parts)
    if len(joined) > EVIDENCE_CHAR_LIMIT:
        return joined[:EVIDENCE_CHAR_LIMIT] + _TRUNCATED
    return joined


def build_verifier_prompt(claim_text: str, evidence_texts: Sequence[str]) -> str:
    return VERIFIER_PROMPT.format(evidence=build_evidence(evidence_texts), claim=claim_text)


def parse_verdict(raw: str) -> tuple[Verdicts, str]:
    """Ham model çıktısı -> `(verdict, gerekce)`. ASLA patlamaz, ASLA uydurmaz.

    Sıra: (1) JSON (kod çiti temizlenmiş, liste ise ilk nesne), (2) KATI regex
    (anahtarın kendisi aranır — çıplak bir "supported" sözcüğü YETMEZ), (3)
    `belirsiz`. Üçüncü adım sözleşmedir: ayrıştırılamayan bir yanıt asla
    `supported` sayılmaz (plan `:107`).
    """
    text = _FENCE.sub("", raw or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict in VERDICTS:
                return cast(Verdicts, verdict), str(data.get("gerekce", "") or "").strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    m = _VERDICT_RE.search(text)
    if m is None:
        return "belirsiz", "model çıktısı ayrıştırılamadı"
    g = _GEREKCE_RE.search(text)
    return cast(Verdicts, m.group(1).lower()), (g.group(1).strip() if g else "")


# ---------------------------------------------------------------------------
# 4. ÖNBELLEK + BÜTÇE
# ---------------------------------------------------------------------------


def evidence_sha256(evidence_texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for text in evidence_texts:
        h.update((text or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def cache_key(*, model: str, prompt_version: str, claim_text: str, evidence_sha: str) -> str:
    """sha256(model + istem sürümü + iddia + kanıt sha) — plan mandatı + iki düzeltme.

    Plan'ın anahtarı yalnız (soru + iddia + birim kimlikleri) idi. İki eksik,
    `p2-reality-audit.md:53`'te ölçülmüş biçimde tespit edildi ve burada
    kapatılıyor: **model kimliği** (2.0-flash -> 3.6-flash zaten bir kez
    değişti) ve **istem/şema sürümü**. Ayrıca kanıt "birim kimliği" değil
    İÇERİK SHA'sı ile anahtarlanır: aynı sayfa kimliği yeniden indekslendiğinde
    farklı metin taşıyabilir.
    """
    payload = "\x00".join([model, prompt_version, claim_text, evidence_sha])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VerifierBudgetExceeded(RuntimeError):
    """Koşum başına LLM çağrı tavanı aşılmak üzere (çağrı YAPILMADAN önce)."""


@dataclass
class VerifierBudget:
    """Koşum başına SERT çağrı tavanı — aşmadan ÖNCE fırlatır.

    Kota bu projenin kritik yoludur (2026-08-30: bir günde 2× http_429).
    "Sınırsız varsayılan" bir bütçe bayrağı bir bütçe değildir; bu yüzden CLI
    tarafında `--max-llm-calls` ZORUNLU bir seçenektir.
    """

    max_calls: int
    used: int = 0

    def consume(self) -> None:
        if self.used >= self.max_calls:
            raise VerifierBudgetExceeded(
                f"doğrulayıcı bütçesi doldu: {self.used}/{self.max_calls} çağrı yapıldı"
            )
        self.used += 1

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)


@dataclass(frozen=True)
class VerifierCache:
    """`<root>/<sha256>.json` dosya önbelleği — isabet API'ye GİTMEZ.

    İçerik KÜNYELİDİR (ts, model, istem sürümü, kanıt sha): bir önbellek
    dosyasına bakan insan, o kararın hangi koşulda üretildiğini dosyanın
    kendisinden okuyabilmeli. Dizin `data/cache/` altındadır ve `.gitignore`
    `data/*` kuralıyla zaten dışarıdadır (yeniden üretilebilir).
    """

    root: Path

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("doğrulayıcı önbelleği okunamadı, yok sayılıyor: %s", p)
            return None
        return data if isinstance(data, dict) and data.get("verdict") in VERDICTS else None

    def put(self, key: str, payload: Mapping[str, object]) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.path(key).write_text(
                json.dumps(dict(payload), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            # Önbellek bir HIZLANDIRMADIR; yazılamaması isteği düşürmez.
            logger.warning("doğrulayıcı önbelleği yazılamadı: %s", self.path(key))


# ---------------------------------------------------------------------------
# 5. DOĞRULAMA
# ---------------------------------------------------------------------------


class VerifierClient(Protocol):
    """Doğrulayıcının ihtiyaç duyduğu TEK yetenek: istemden ham metin üretmek.

    Yapılandırılmış çıktı (`response_schema`) istemcinin İÇİNDE kurulur;
    ayrıştırma her koşulda `parse_verdict`ten geçer, yani şema desteklenmese de
    (eski SDK, stub istemci) davranış tanımlıdır.
    """

    def generate_json(self, prompt: str, schema: dict | None = None) -> str: ...


def verify_claim(
    claim: Claim,
    evidence_texts: Sequence[str],
    client: VerifierClient,
    *,
    model: str,
    cache: VerifierCache | None = None,
    budget: VerifierBudget | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> Verdict:
    """Tek iddiayı KANIT METNİNE karşı yargılar. Önbellek isabeti = 0 API çağrısı.

    Kanıt yoksa (iddia atıfsız ya da atıf yaptığı sayfaların metni boş) çağrı
    YAPILMAZ: karar `belirsiz`dir ve gerekçesi bunu söyler. Bu hem kota
    dostudur hem de doğru olandır — kanıtsız bir iddia doğrulanmış sayılamaz.
    """
    usable = [t for t in evidence_texts if (t or "").strip()]
    if not usable:
        return Verdict(
            claim_id=claim.claim_id,
            verdict="belirsiz",
            gerekce=("iddianın atıf yaptığı sayfa metni yok (atıfsız ya da boş metin katmanı)"),
        )

    ev_sha = evidence_sha256(usable)
    key = cache_key(
        model=model,
        prompt_version=prompt_version,
        claim_text=claim.text,
        evidence_sha=ev_sha,
    )
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return Verdict(
                claim_id=claim.claim_id,
                verdict=hit["verdict"],
                gerekce=str(hit.get("gerekce", "")),
                cached=True,
            )

    # Bütçe kontrolü çağrıdan ÖNCE ve önbellek isabetinden SONRA: isabet bütçe
    # harcamaz (önbelleğin var olma sebebi tam olarak budur).
    if budget is not None:
        budget.consume()

    prompt = build_verifier_prompt(claim.text, usable)
    try:
        raw = client.generate_json(prompt, VERDICT_SCHEMA)
    except VerifierBudgetExceeded:  # pragma: no cover - bütçe burada tüketilmez
        raise
    except Exception as exc:
        # Kota/zaman aşımı/şema hatası: ŞÜPHEDE REDDET. İstisna yukarı
        # sızdırılmaz çünkü servis yolunda bir doğrulayıcı arızası isteği
        # düşürmemeli — yanıt yalnız KESİN diye sunulmamalı.
        logger.warning("doğrulayıcı çağrısı başarısız (%s): %s", type(exc).__name__, exc)
        return Verdict(
            claim_id=claim.claim_id,
            verdict="belirsiz",
            gerekce=f"doğrulayıcı çağrısı başarısız: {type(exc).__name__}",
            # Çağrı YAPILDI (ve başarısız oldu): kota harcandı, muhasebe bunu
            # görmeli — başarısız bir çağrı da bir çağrıdır.
            llm_called=True,
        )

    verdict, gerekce = parse_verdict(raw)
    if cache is not None:
        cache.put(
            key,
            {
                "key": key,
                "ts": datetime.now(UTC).isoformat(),
                "model": model,
                "prompt_version": prompt_version,
                "evidence_sha256": ev_sha,
                "claim": claim.text,
                "verdict": verdict,
                "gerekce": gerekce,
            },
        )
    return Verdict(claim_id=claim.claim_id, verdict=verdict, gerekce=gerekce, llm_called=True)


@dataclass
class ClaimVerifier:
    """`verify_claim`in bağlamı (istemci + model + önbellek + bütçe) taşıyan hâli."""

    client: VerifierClient
    model: str
    cache: VerifierCache | None = None
    budget: VerifierBudget | None = None
    prompt_version: str = PROMPT_VERSION

    def verify(self, claim: Claim, evidence_texts: Sequence[str]) -> Verdict:
        return verify_claim(
            claim,
            evidence_texts,
            self.client,
            model=self.model,
            cache=self.cache,
            budget=self.budget,
            prompt_version=self.prompt_version,
        )


class GeminiVerifierClient:
    """`GeminiClient` üstünde `VerifierClient` uyarlayıcısı (temperature=0).

    İstemci `answer/gemini.py::build_gemini_client` ile kurulur — yanıtlayıcı
    ile AYNI tek nokta. Doğrulayıcı ikinci bir `GeminiClient(...)` çağrı yeri
    AÇMAZ: zaman aşımı, retry, bütçe invariantı ve istemci düzeyindeki her
    politika (ör. anahtar seçimi) tek yerde uygulansın diye.
    """

    def __init__(self, model: str, api_key: str, client=None) -> None:
        from belge_gozu.answer.gemini import build_gemini_client

        self.model = model
        self._client = client or build_gemini_client(model, api_key)

    def generate_json(self, prompt: str, schema: dict | None = None) -> str:
        return self._client.generate_json(prompt, schema).text


# ---------------------------------------------------------------------------
# 6. KANIT KAPISI (T2'nin ikinci kapısı)
# ---------------------------------------------------------------------------


@dataclass
class EvidenceGate:
    """Yanıtın HER iddiası kendi atıf sayfalarında destekleniyor mu?

    Politika (ilke 20 / G2.6): tek bir desteklenmeyen iddia bile yanıtı
    DÜŞÜRÜR. `belirsiz` desteklenmemiş sayılır (şüphede-reddet).
    """

    verifier: ClaimVerifier
    page_texts: Mapping[str, str]
    max_claims: int = 8

    def evaluate(self, answer: Answer, hits: Sequence[PageHit]) -> dict:
        claims = segment_claims(answer.text)
        truncated = len(claims) > self.max_claims
        used = claims[: self.max_claims]
        verdicts = [self.verifier.verify(c, self._evidence(c, hits)) for c in used]

        rows = [
            {
                "claim_id": c.claim_id,
                "verdict": v.verdict,
                "gerekce": v.gerekce,
                "cited_sources": c.cited_sources,
                "inherited_sources": c.inherited_sources,
                "cached": v.cached,
            }
            for c, v in zip(used, verdicts, strict=True)
        ]
        n_supported = sum(1 for v in verdicts if v.supported)
        # Kırpılmış iddialar DOĞRULANMAMIŞTIR: kırpma da bir "kanıtlanamadı"
        # hâlidir ve yanıtı düşürür (sessizce geçirmek tam olarak ilke 20'nin
        # yasakladığı şey olurdu).
        demoted = truncated or any(not v.supported for v in verdicts) or not verdicts
        return {
            "demoted": demoted,
            "n_claims": len(claims),
            "n_verified": len(verdicts),
            "n_supported": n_supported,
            "truncated": truncated,
            "cache_hits": sum(1 for v in verdicts if v.cached),
            "llm_calls": sum(1 for v in verdicts if v.llm_called),
            "claims": rows,
        }

    def _evidence(self, claim: Claim, hits: Sequence[PageHit]) -> list[str]:
        """`[Sn]` -> servis edilen n'inci sayfanın METNİ (yeni çıkarım YOK).

        Metin, sıralamayı üreten kanalın TA KENDİSİNDEN gelir
        (`page_texts.parquet`); doğrulayıcı için ayrı bir çıkarım hattı
        kurulmaz — kurulsaydı "doğrulayıcının gördüğü metin" ile "getirimin
        skorladığı metin" sessizce ayrışabilirdi.
        """
        out: list[str] = []
        for n in claim.cited_sources:
            if 0 < n <= len(hits):
                out.append(self.page_texts.get(hits[n - 1].page_id, ""))
        return out


def gate2_skip_reason(answer: Answer) -> str | None:
    """Kanıt kapısının ATLANMA sebebi (yoksa None) — tek karar yolu.

    Dürüst ıska ve atıfsız yanıtlar atlanır: ikisi de zaten "kesin yanıt" diye
    sunulmuyor, doğrulamak yalnız kota yakardı.
    """
    if answer.abstained:
        return "abstained"
    if is_honest_miss(answer):
        return "honest_miss"
    if not answer.citations:
        return "no_citations"
    return None


# ---------------------------------------------------------------------------
# 7. KAPI KURULUMU (serve + CLI ortak yolu)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gates:
    """`AskService`e takılacak iki kapı (ikisi de None olabilir = bayrak kapalı)."""

    retrieval: object | None = None
    evidence: object | None = None
    budget: VerifierBudget | None = None
    detail: dict = field(default_factory=dict)
    """Künye: hangi kapı hangi artefaktla açıldı (healthz/rapor için)."""


def build_gates(
    s: Settings,
    retriever,
    *,
    index_revision: str | None,
    budget: VerifierBudget | None = None,
) -> Gates:
    """Bayraklara göre kapıları kurar. İKİSİ DE KAPALIYSA hiçbir şey yüklenmez.

    FAIL-FAST: `gate_calibrated` açık ama artefakt yok/anahtarı uyuşmuyorsa
    başlangıçta `IndexCompatibilityError` fırlar (`belge-gozu calibrate fit`
    ipucuyla). Sessizce "kalibrasyonsuz kalibre kapı" ile açılmak, tam olarak
    kalibrasyonun önlemek için var olduğu sessiz yanlışlıktır.
    """
    from belge_gozu.index.compat import IndexCompatibilityError

    if not (s.gate_calibrated or s.gate_verifier):
        return Gates()

    detail: dict = {}
    gate1 = None
    if s.gate_calibrated:
        from belge_gozu.answer.calibrate import (
            CalibratedRetrievalGate,
            CalibrationKeyMismatch,
            calibration_dir,
            calibration_key,
            load_calibrator,
        )

        if s.retrieval_pipeline != "hybrid":
            raise IndexCompatibilityError(
                "BG_GATE_CALIBRATED=true yalnız hibrit boru hattında tanımlıdır "
                f"(özellikler BM25 metin kanalından okunur); etkin: {s.retrieval_pipeline}"
            )
        if index_revision is None:
            raise IndexCompatibilityError(
                "BG_GATE_CALIBRATED=true indeks manifest'i gerektirir (kalibrasyon "
                "artefaktı sürüm anahtarına bağlıdır) ama manifest yok"
            )
        key = calibration_key(index_revision, s.retrieval_pipeline)
        try:
            artifact = load_calibrator(calibration_dir(s.data_dir / "calibration", key), key)
        except (FileNotFoundError, CalibrationKeyMismatch) as exc:
            raise IndexCompatibilityError(
                f"BG_GATE_CALIBRATED=true ama kalibrasyon artefaktı kullanılamıyor: {exc} "
                "Çözüm: `uv run belge-gozu calibrate fit` (bu anahtar için yeniden fit edin) "
                "ya da BG_GATE_CALIBRATED=false ile eski eşik davranışına dönün."
            ) from exc
        gate1 = CalibratedRetrievalGate(artifact, retriever.text, retriever.doc_names)
        detail["gate1"] = {
            "key": artifact.key,
            "tau": artifact.tau,
            "guarantee": artifact.thresholds["chosen"].get("statistical_guarantee"),
        }

    gate2 = None
    if s.gate_verifier:
        from belge_gozu.retrieval.hybrid import load_page_texts

        page_texts = load_page_texts(s.index_dir)
        verifier = ClaimVerifier(
            client=GeminiVerifierClient(s.gemini_model, s.gemini_api_key),
            model=s.gemini_model,
            cache=VerifierCache(s.data_dir / "cache" / "verifier"),
            budget=budget,
        )
        gate2 = EvidenceGate(verifier, page_texts, max_claims=s.verifier_max_claims)
        detail["gate2"] = {
            "model": s.gemini_model,
            "prompt_version": PROMPT_VERSION,
            "max_claims": s.verifier_max_claims,
            "cache_dir": str(s.data_dir / "cache" / "verifier"),
            "pages": len(page_texts),
        }
    return Gates(retrieval=gate1, evidence=gate2, budget=budget, detail=detail)
