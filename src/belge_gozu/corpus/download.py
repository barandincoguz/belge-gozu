import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from pydantic import BaseModel

from belge_gozu.corpus.manifest import USER_AGENT, ManifestRow


class DownloadReport(BaseModel):
    ok: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


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
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=60,
            )
            resp.raise_for_status()
            target.write_bytes(resp.content)
            sha = hashlib.sha256(resp.content).hexdigest()
            state[row.doc_id] = {"sha256": sha, "status": "ok"}
            report.ok.append(row.doc_id)
        except (httpx.HTTPError, httpx.InvalidURL, OSError):
            state[row.doc_id] = {"sha256": "", "status": "failed"}
            report.failed.append(row.doc_id)
        tmp_path = state_path.parent / f"{state_path.name}.tmp"
        tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        os.replace(tmp_path, state_path)
    return report
