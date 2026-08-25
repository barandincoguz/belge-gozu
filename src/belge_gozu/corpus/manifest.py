import csv
import io
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError


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
            resp = client.head(r.url, follow_redirects=True, timeout=20)
            out.append((r.doc_id, resp.status_code))
        except (httpx.HTTPError, httpx.InvalidURL):
            out.append((r.doc_id, 0))
    return out
