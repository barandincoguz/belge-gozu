# ruff: noqa: E501
"""Semantic coverage JSON'unu dış bağımlılığı olmayan karar raporuna çevirir."""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _coverage(arm: Mapping[str, Any], slice_name: str | None = None) -> str:
    source = arm.get("overall", {}) if slice_name is None else arm.get("per_slice", {}).get(slice_name, {})
    value = source.get("coverage")
    return "—" if value is None else f"{float(value):.4f}"


def _row(name: str, arm: Mapping[str, Any]) -> str:
    status = str(arm.get("status", "unknown"))
    if status != "ok":
        return (
            f"<tr><th>{html.escape(name)}</th><td colspan=\"3\" class=\"muted\">"
            f"{html.escape(status)} — {html.escape(str(arm.get('reason', '')))}</td></tr>"
        )
    overall = _coverage(arm)
    paraphrase = _coverage(arm, "paraphrase")
    bar_width = float(arm.get("overall", {}).get("coverage", 0.0)) * 100
    return (
        f"<tr><th>{html.escape(name)}</th><td>{overall}</td><td>{paraphrase}</td>"
        f"<td><span class=\"bar\" style=\"--value:{bar_width:.2f}%\"></span></td></tr>"
    )


def _diagnostics(arms: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for arm_name, arm in arms.items():
        if not isinstance(arm, Mapping) or arm.get("status") != "ok":
            continue
        for item in arm.get("diagnostics", []):
            if not isinstance(item, Mapping):
                continue
            pool = ", ".join(str(page) for page in item.get("candidate_pool", []))
            rows.append(
                "<tr>"
                f"<td>{html.escape(arm_name)}</td>"
                f"<td>{html.escape(str(item.get('question_id', '')))}</td>"
                f"<td>{html.escape(str(item.get('slice', '')))}</td>"
                f"<td><code>{html.escape(pool)}</code></td>"
                "</tr>"
            )
    return "".join(rows) or "<tr><td colspan=\"4\" class=\"muted\">Tanı satırı yok.</td></tr>"


def render_report(report: Mapping[str, Any]) -> str:
    """Kaçış uygulanmış, tek dosyalı development karar raporu üretir."""
    mode = str(report.get("mode", "unknown"))
    if mode != "development":
        raise ValueError("yalnız development semantic raporu render edilebilir")
    arms = report.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("semantic raporda arms nesnesi zorunlu")
    rows = "".join(_row(str(name), arm) for name, arm in arms.items() if isinstance(arm, Mapping))
    selected = html.escape(str(report.get("selected_dense") or "seçim yok"))
    diagnostics = _diagnostics(arms)
    return f"""<!doctype html>
<html lang=\"tr\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Semantic Coverage — development</title>
<style>
:root {{ color-scheme: dark; --ink:#edf3ff; --paper:#101827; --panel:#172238; --line:#2b3b59; --cyan:#56d9e8; --gold:#f2c56b; --muted:#9bacc6; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 ui-sans-serif,system-ui,sans-serif }}
main {{ max-width:1100px; margin:auto; padding:32px 20px 64px }} h1 {{ margin:0; font-size:clamp(30px,5vw,56px); letter-spacing:-.045em }}
.eyebrow {{ color:var(--cyan); font:700 12px/1.2 ui-monospace,monospace; letter-spacing:.12em; text-transform:uppercase }}
.warning {{ margin:28px 0; padding:14px 16px; border-left:4px solid var(--gold); background:#211e1a; color:#ffe5aa; font-weight:700 }}
.grid {{ display:grid; grid-template-columns:1fr 2fr; gap:18px }} .card {{ background:var(--panel); border:1px solid var(--line); padding:20px }}
.choice {{ font:700 22px/1.2 ui-monospace,monospace; color:var(--cyan); overflow-wrap:anywhere }} .muted {{ color:var(--muted) }}
table {{ width:100%; border-collapse:collapse; margin-top:12px }} th,td {{ padding:11px 8px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top }}
th {{ font-weight:700 }} code {{ color:#c7d6ee; white-space:normal; overflow-wrap:anywhere }}
.bar {{ display:block; width:var(--value); min-width:4px; height:8px; background:var(--cyan) }}
details {{ margin-top:20px }} summary {{ cursor:pointer; color:var(--cyan); font-weight:700 }}
@media (max-width:700px) {{ .grid {{ grid-template-columns:1fr }} main {{ padding:22px 12px }} table {{ font-size:13px }} }}
</style></head><body><main>
<p class=\"eyebrow\">candidate-pool evidence / r@50</p><h1>Anlamsal kapsama</h1>
<div class=\"warning\">DEVELOPMENT ONLY — HOLDOUT REQUIRED. Bu sonuç üretim seçimi değildir.</div>
<section class=\"grid\"><article class=\"card\"><p class=\"eyebrow\">seçilen dense kol</p><p class=\"choice\">{selected}</p></article>
<article class=\"card\"><p class=\"eyebrow\">okuma kuralı</p><p class=\"muted\">Paraphrase R@50 birincil karar metriğidir. Havuzlar ilk-görülme sırasıyla birleşir; skor füzyonu yoktur.</p></article></section>
<section class=\"card\" style=\"margin-top:18px\"><p class=\"eyebrow\">kol karşılaştırması</p><table><thead><tr><th>Kol</th><th>Havuz kapsaması</th><th>Paraphrase R@50</th><th>İz</th></tr></thead><tbody>{rows}</tbody></table></section>
<details class=\"card\"><summary>Soru bazlı aday izleri</summary><table><thead><tr><th>Kol</th><th>Soru</th><th>Dilim</th><th>Aday havuzu</th></tr></thead><tbody>{diagnostics}</tbody></table></details>
</main></body></html>"""


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("rapor JSON nesnesi olmalı")
    _write_atomic(args.out, render_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
