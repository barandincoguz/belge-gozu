import csv
import io
import ssl
from pathlib import Path
from typing import Literal

import certifi
import httpx
from pydantic import BaseModel, ValidationError

# gov.tr WAF'ı varsayılan kütüphane User-Agent'larını (python-httpx/..., curl/...)
# sessizce (yanıtsız) düşürür; tarayıcı benzeri/özel bir UA ile anında 200 döner.
# download.py de aynı sabiti kullanır (tek kaynak burada, döngüsel import'tan kaçınmak için).
USER_AGENT = "belge-gozu/0.1 (acik kaynak arastirma projesi)"

# *.tccb.gov.tr sertifikasını kullanan gov.tr siteleri (mevzuat.gov.tr,
# resmigazete.gov.tr, ...) TLS el sıkışmasında ara sertifikayı göndermiyor.
# Tarayıcılar/curl bunu sistem önbelleği veya AIA getirmesiyle tamamlıyor;
# Python'un certifi tabanlı doğrulaması tamamlamıyor ("unable to get local
# issuer certificate"). Eksik ara sertifika (GeoTrust TLS RSA CA G1, DigiCert
# Global Root G2'ye bağlı — certifi'de zaten güvenilir) sunucunun AIA "CA
# Issuers" alanından (http://cacerts.geotrust.com/GeoTrustTLSRSACAG1.crt)
# alınıp burada gömülü; zincir openssl ile doğrulandı (Task 13 canlı kontrol).
_GEOTRUST_TLS_RSA_CA_G1 = """-----BEGIN CERTIFICATE-----
MIIEjTCCA3WgAwIBAgIQDQd4KhM/xvmlcpbhMf/ReTANBgkqhkiG9w0BAQsFADBh
MQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3
d3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBH
MjAeFw0xNzExMDIxMjIzMzdaFw0yNzExMDIxMjIzMzdaMGAxCzAJBgNVBAYTAlVT
MRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5j
b20xHzAdBgNVBAMTFkdlb1RydXN0IFRMUyBSU0EgQ0EgRzEwggEiMA0GCSqGSIb3
DQEBAQUAA4IBDwAwggEKAoIBAQC+F+jsvikKy/65LWEx/TMkCDIuWegh1Ngwvm4Q
yISgP7oU5d79eoySG3vOhC3w/3jEMuipoH1fBtp7m0tTpsYbAhch4XA7rfuD6whU
gajeErLVxoiWMPkC/DnUvbgi74BJmdBiuGHQSd7LwsuXpTEGG9fYXcbTVN5SATYq
DfbexbYxTMwVJWoVb6lrBEgM3gBBqiiAiy800xu1Nq07JdCIQkBsNpFtZbIZhsDS
fzlGWP4wEmBQ3O67c+ZXkFr2DcrXBEtHam80Gp2SNhou2U5U7UesDL/xgLK6/0d7
6TnEVMSUVJkZ8VeZr+IUIlvoLrtjLbqugb0T3OYXW+CQU0kBAgMBAAGjggFAMIIB
PDAdBgNVHQ4EFgQUlE/UXYvkpOKmgP792PkA76O+AlcwHwYDVR0jBBgwFoAUTiJU
IBiV5uNu5g/6+rkS7QYXjzkwDgYDVR0PAQH/BAQDAgGGMB0GA1UdJQQWMBQGCCsG
AQUFBwMBBggrBgEFBQcDAjASBgNVHRMBAf8ECDAGAQH/AgEAMDQGCCsGAQUFBwEB
BCgwJjAkBggrBgEFBQcwAYYYaHR0cDovL29jc3AuZGlnaWNlcnQuY29tMEIGA1Ud
HwQ7MDkwN6A1oDOGMWh0dHA6Ly9jcmwzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2VydEds
b2JhbFJvb3RHMi5jcmwwPQYDVR0gBDYwNDAyBgRVHSAAMCowKAYIKwYBBQUHAgEW
HGh0dHBzOi8vd3d3LmRpZ2ljZXJ0LmNvbS9DUFMwDQYJKoZIhvcNAQELBQADggEB
AIIcBDqC6cWpyGUSXAjjAcYwsK4iiGF7KweG97i1RJz1kwZhRoo6orU1JtBYnjzB
c4+/sXmnHJk3mlPyL1xuIAt9sMeC7+vreRIF5wFBC0MCN5sbHwhNN1JzKbifNeP5
ozpZdQFmkCo+neBiKR6HqIA+LMTMCMMuv2khGGuPHmtDze4GmEGZtYLyF8EQpa5Y
jPuV6k2Cr/N3XxFpT3hRpt/3usU/Zb9wfKPtWpoznZ4/44c1p9rzFcZYrWkj3A+7
TNBJE0GmP2fhXhP1D/XVfIW/h0yCJGEiV9Glm/uGOa3DXHlmbAcxSyCRraG+ZBkA
7h4SeM6Y8l/7MBRpPCz6l8Y=
-----END CERTIFICATE-----
"""


def build_ssl_context() -> ssl.SSLContext:
    """certifi CA paketi + eksik GeoTrust ara sertifikasıyla tam doğrulama
    yapan bir SSL context üretir (bkz. yukarıdaki not). TLS doğrulaması
    (verify_mode/check_hostname) varsayılan olarak açık kalır."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.load_verify_locations(cadata=_GEOTRUST_TLS_RSA_CA_G1)
    return ctx


def build_http_client(**kwargs) -> httpx.Client:
    """gov.tr indirmeleri için TLS zinciri tamamlanmış httpx.Client üretir
    (bkz. yukarıdaki not). Testler kendi MockTransport client'ını kurduğu için
    bu yalnız gerçek ağ (cli.py) çağrılarında kullanılır."""
    return httpx.Client(verify=build_ssl_context(), **kwargs)


class ManifestRow(BaseModel):
    doc_id: str
    doc_name: str
    doc_type: Literal["kanun", "rg_tarihi"]
    url: str


def load_manifest_from_text(text: str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for i, rec in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        try:
            rows.append(ManifestRow(**rec))  # type: ignore[arg-type]
        except ValidationError as e:
            raise ValueError(f"manifest satır {i}: {e}") from e
    if not rows:
        raise ValueError("manifest boş")
    return rows


def load_manifest(path: Path) -> list[ManifestRow]:
    return load_manifest_from_text(path.read_text(encoding="utf-8"))


def probe(rows: list[ManifestRow], client: httpx.Client) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for r in rows:
        try:
            resp = client.head(
                r.url,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=20,
            )
            out.append((r.doc_id, resp.status_code))
        except (httpx.HTTPError, httpx.InvalidURL):
            out.append((r.doc_id, 0))
    return out
