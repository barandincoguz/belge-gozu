import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from pydantic import BaseModel

from belge_gozu.corpus.manifest import ManifestRow

USER_AGENT = "belge-gozu/0.1 (açık kaynak araştırma projesi)".encode()


class DownloadReport(BaseModel):
    ok: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def download_all(
    rows: list[ManifestRow],
    out_dir: Path,
    client: httpx.Client,
    delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadReport:
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = _load_state(state_path)
    report = DownloadReport()

    for row in rows:
        target = pdf_dir / f"{row.doc_id}.pdf"
        if state.get(row.doc_id, {}).get("status") == "ok" and target.exists():
            report.skipped.append(row.doc_id)
            continue
        if delay_s:
            sleep(delay_s)
        try:
            resp = client.get(
                row.url,
                headers={"User-Agent": USER_AGENT},  # type: ignore[arg-type]
                follow_redirects=True,
                timeout=60,
            )
            resp.raise_for_status()
            target.write_bytes(resp.content)
            sha = hashlib.sha256(resp.content).hexdigest()
            state[row.doc_id] = {"sha256": sha, "status": "ok"}
            report.ok.append(row.doc_id)
        except httpx.HTTPError:
            state[row.doc_id] = {"sha256": "", "status": "failed"}
            report.failed.append(row.doc_id)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    return report
