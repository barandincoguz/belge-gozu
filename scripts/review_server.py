"""RetrievalEval bench — insan doğrulama için yerel HTML arayüzü.

Terminal `verify_retrieval_eval.py --review` ile AYNI işi yapar, aynı saf fonksiyonları
çağırır (`select_review_queue`, `precheck_question`, `apply_decision`,
`write_jsonl_atomic`) — tek farkı kararı klavyeden değil tarayıcıdan almasıdır.
Kazanç somut: sayfa görüntüsü sorunun ve kanıt alıntılarının YANINDA durur,
ayrı bir görüntü penceresine geçip geri gelmek gerekmez. 46 satırlık bir
kuyrukta bu fark, incelemenin bitip bitmemesini belirler.

Neden ayrı bir betik: `verify_retrieval_eval.py` PDF/indeks I/O'suna dokunmayan saf
çekirdeğiyle CI'da test edilebiliyor. HTTP sunucusu o çekirdeğe HİÇBİR şey
eklemez, yalnız üstüne bir taşıyıcı koyar; ayrı dosyada durması çekirdeğin
test edilebilirliğini bozmaz.

Yazma yolu terminal sürümüyle birebir aynıdır: her karardan sonra dosya ATOMİK
yazılır (aynı dizinde temp + os.replace), yani tarayıcı sekmesi kapansa, sunucu
öldürülse ya da makine çökse bile JSONL ya eski ya yeni hâliyle BÜTÜN kalır.

    uv run python scripts/review_server.py --by baran
    uv run python scripts/review_server.py --by baran --slice paraphrase --slice tarihi-tarama

Sunucu YALNIZ 127.0.0.1'e bağlanır; dışarıdan erişilemez.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from verify_retrieval_eval import (  # noqa: E402
    DEFAULT_IMAGES_DIR,
    DEFAULT_PDF_DIR,
    apply_decision,
    extract_page_texts,
    gold_image_paths,
    load_known_page_ids,
    load_raw_rows,
    precheck_question,
    select_review_queue,
    write_jsonl_atomic,
)

from belge_gozu.bench.dataset import BenchQuestion  # noqa: E402
from belge_gozu.config import Settings  # noqa: E402

DEFAULT_BENCH_V2 = REPO_ROOT / "data" / "bench" / "retrieval_eval_v2.jsonl"

# Karar dilimleri: G1.2'nin üstünde hüküm verdiği dört dilim. Varsayılan
# kuyruk bunlarla sınırlıdır — emeği kapının gerçekten baktığı yere harcamak
# için (D1 tasarım kararı).
DECISION_SLICES = frozenset(
    {"paraphrase", "dogrudan-madde", "madde-numarali", "ayni-kanun-hard-negative"}
)


def display_path(path: Path, root: Path) -> str:
    """Repo köküne göre kısa yol; kök dışındaysa mutlak yolun kendisi.

    `--bench` repo dışını gösterebilir (duman testleri /tmp kullanır) ve
    `Path.relative_to` o durumda ValueError atar — arayüzün başlık satırı
    yüzünden tüm sunucunun 500 vermesi kabul edilemez.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


class ReviewState:
    """Sunucunun paylaşılan durumu; yazma tek kilit altında serileştirilir."""

    def __init__(self, bench: Path, by: str, slices: frozenset[str] | None) -> None:
        self.bench = bench
        self.by = by
        self.slices = slices
        self.lock = threading.Lock()
        self.rows = load_raw_rows(bench)
        self.prechecks: dict[str, dict] = {}

    def queue_ids(self) -> list[str]:
        selected = select_review_queue(
            self.rows, slices=set(self.slices) if self.slices else None
        )
        return [r["question_id"] for r in selected]

    def build_prechecks(self) -> None:
        """Kuyruktaki satırların gold sayfa metinlerini bir kez çıkar ve ön-kontrolü koş."""
        ids = set(self.queue_ids())
        questions: list[BenchQuestion] = []
        for row in self.rows:
            if row["question_id"] not in ids:
                continue
            try:
                questions.append(BenchQuestion(**row))
            except Exception as exc:  # şema hatası incelemeyi durdurmasın
                self.prechecks[row["question_id"]] = {
                    "group": "MANUEL",
                    "notes": [f"satır şemaya uymuyor: {exc}"],
                    "spans": [],
                }
        if not questions:
            return
        wanted = {gp for q in questions for gp in q.gold_page_ids}
        page_texts = extract_page_texts(wanted, DEFAULT_PDF_DIR)
        known = load_known_page_ids(Settings().index_dir)
        for q in questions:
            pc = precheck_question(q, page_texts, known)
            self.prechecks[q.question_id] = {
                "group": pc.group,
                "notes": pc.notes,
                "spans": [
                    {"span": s.span, "found": s.found, "page_id": s.page_id, "closest": s.closest}
                    for s in pc.span_checks
                ],
            }

    def payload(self) -> dict:
        ids = self.queue_ids()
        by_id = {r["question_id"]: r for r in self.rows}
        items = []
        for qid in ids:
            row = by_id[qid]
            pc = self.prechecks.get(qid, {"group": "?", "notes": [], "spans": []})
            items.append(
                {
                    "question_id": qid,
                    "question": row.get("question", ""),
                    "reference_answer": row.get("reference_answer", ""),
                    "slice": row.get("slice", ""),
                    "difficulty": row.get("difficulty", ""),
                    "query_style": row.get("query_style", ""),
                    "answerable": row.get("answerable", True),
                    "gold_page_ids": row.get("gold_page_ids", []),
                    "gold_article_ids": row.get("gold_article_ids", []),
                    "spans": row.get("minimal_evidence_spans", []),
                    "verification_kind": row.get("verification_kind", "human"),
                    "verification_note": row.get("verification_note", ""),
                    "precheck": pc,
                }
            )
        total = len(self.rows)
        human = sum(1 for r in self.rows if r.get("verification_kind", "human") == "human")
        return {
            "items": items,
            "by": self.by,
            "bench": display_path(self.bench, REPO_ROOT),
            "remaining": len(items),
            "human": human,
            "total": total,
        }

    def decide(self, question_id: str, letter: str, note: str) -> dict:
        with self.lock:
            for i, row in enumerate(self.rows):
                if row["question_id"] == question_id:
                    self.rows[i] = apply_decision(row, letter, note, by=self.by)
                    write_jsonl_atomic(self.bench, self.rows)
                    break
            else:
                raise KeyError(question_id)
        return self.payload()


def image_response_path(state: ReviewState, page_id: str) -> Path | None:
    """`dok:sayfa` -> WebP yolu; kuyruğa AİT olmayan sayfa servis edilmez."""
    for row in state.rows:
        if page_id in row.get("gold_page_ids", []):
            try:
                q = BenchQuestion(**row)
            except Exception:
                return None
            for path in gold_image_paths(q, DEFAULT_IMAGES_DIR):
                if path.stem == page_id.split(":", 1)[1].zfill(4):
                    return path if path.exists() else None
    return None


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, state: ReviewState, **kwargs) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:  # sessiz
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/queue":
            self._json(self.state.payload())
        elif path.startswith("/img/"):
            page_id = unquote(path[len("/img/") :])
            img = image_response_path(self.state, page_id)
            if img is None:
                self.send_error(404, "sayfa görüntüsü yok")
                return
            data = img.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/webp")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/decide":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            letter = payload["letter"]
            if letter not in {"e", "h", "a"}:
                raise ValueError(f"geçersiz karar: {letter!r}")
            result = self.state.decide(payload["question_id"], letter, payload.get("note", ""))
        except KeyError as exc:
            self._json({"error": f"bilinmeyen soru: {exc}"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)
        else:
            self._json(result)


PAGE = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RetrievalEval inceleme</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--panel:#fff;--ink:#12283d;--mut:#6e7f8f;
--line:#d7dfe8;--ok:#2e7d4f;--okbg:#e8f3ec;--bad:#a61c2c;--badbg:#fceceb;
--warn:#8a6d2c;--warnbg:#f7f4ec;--acc:#2c5b8a}
@media(prefers-color-scheme:dark){:root{--bg:#0f151b;--panel:#161e26;--ink:#e3eaf1;--mut:#8596a5;
--line:#27333e;--ok:#62cc92;--okbg:#142a1e;--bad:#ec8093;--badbg:#2e161b;
--warn:#d6ac5c;--warnbg:#2b2415;--acc:#74a9db}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);
padding:12px 20px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
header b{font-size:16px}
.prog{font:13px ui-monospace,Menlo,monospace;color:var(--mut)}
.bar{flex:1;min-width:120px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .25s}
main{max-width:1400px;margin:0 auto;padding:20px;display:grid;gap:20px;
align-items:start;grid-template-columns:minmax(340px,1fr) 1.4fr}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px 20px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.chip{font:11px ui-monospace,Menlo,monospace;padding:2px 7px;border-radius:3px;
border:1px solid currentColor}
.chip.s{color:var(--acc)} .chip.g-TEMİZ{color:var(--ok);background:var(--okbg)}
.chip\[data-g\]{}
.g-suspect{color:var(--warn);background:var(--warnbg)}
.g-clean{color:var(--ok);background:var(--okbg)}
h2{font-size:20px;line-height:1.35;margin:0 0 14px}
.lbl{font:11px ui-monospace,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;
color:var(--mut);margin:16px 0 6px}
.ref{color:var(--mut)}
ul.spans{list-style:none;padding:0;margin:0;display:grid;gap:6px}
ul.spans li{font-size:14px;padding:7px 10px;border-radius:5px;
background:var(--okbg);border-left:3px solid var(--ok)}
ul.spans li.miss{background:var(--badbg);border-left-color:var(--bad)}
.notes{margin:12px 0 0;padding:10px 12px;background:var(--warnbg);border-left:3px solid var(--warn);
border-radius:0 5px 5px 0;font-size:13.5px}
.imgs{display:grid;gap:12px}
.imgs img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.imgs .cap{font:11px ui-monospace,Menlo,monospace;color:var(--mut);margin-bottom:4px}
footer{position:sticky;bottom:0;background:var(--panel);border-top:1px solid var(--line);
padding:12px 20px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{font:inherit;font-weight:600;padding:9px 18px;border-radius:6px;border:1px solid var(--line);
background:var(--panel);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--acc)}
button.ok{color:var(--ok);border-color:var(--ok)}
button.no{color:var(--bad);border-color:var(--bad)}
button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
input[type=text]{flex:1;min-width:180px;font:inherit;padding:9px 12px;border-radius:6px;
border:1px solid var(--line);background:var(--bg);color:var(--ink)}
kbd{font:11px ui-monospace,Menlo,monospace;border:1px solid var(--line);
border-radius:3px;padding:1px 5px;color:var(--mut)}
.done{padding:60px 20px;text-align:center;color:var(--mut)}
.err{color:var(--bad);font-size:13px}
</style></head><body>
<header>
  <b>RetrievalEval inceleme</b>
  <span class="prog" id="prog">yükleniyor…</span>
  <span class="bar"><i id="bar"></i></span>
  <span class="prog" id="who"></span>
</header>
<main id="main"></main>
<footer id="foot" hidden>
  <button class="ok" onclick="decide('e')">Onayla <kbd>E</kbd></button>
  <button class="no" onclick="decide('h')">Reddet <kbd>H</kbd></button>
  <button onclick="decide('a')">Atla <kbd>A</kbd></button>
  <input type="text" id="note" placeholder="not (isteğe bağlı)">
  <span class="err" id="err"></span>
</footer>
<script>
let S=null, i=0;
const $=id=>document.getElementById(id);

async function load(){
  S=await (await fetch('/api/queue')).json();
  $('who').textContent='doğrulayan: '+S.by+' · '+S.bench;
  i=0; render();
}
function cur(){ return S.items[i]; }

function render(){
  const done=S.human, tot=S.total;
  $('prog').textContent=`insan ${done}/${tot} · kuyrukta ${S.items.length}`;
  $('bar').style.width=(100*done/tot)+'%';
  if(!S.items.length || i>=S.items.length){
    $('main').innerHTML='<div class="card done"><h2>Kuyruk bitti.</h2>'+
      '<p>Bu filtrede insan doğrulaması bekleyen satır kalmadı. '+
      'Sayımı görmek için: <code>python scripts/verify_retrieval_eval.py --status</code></p></div>';
    $('foot').hidden=true; return;
  }
  $('foot').hidden=false;
  const q=cur(), pc=q.precheck;
  const gcls = pc.group==='TEMİZ' ? 'g-clean' : 'g-suspect';
  const spans=q.precheck.spans.length? q.precheck.spans : q.spans.map(s=>({span:s,found:null}));
  $('main').innerHTML=`
    <div class="card">
      <div class="chips">
        <span class="chip s">${q.slice}</span>
        <span class="chip s">${q.difficulty}</span>
        <span class="chip s">${q.query_style}</span>
        <span class="chip ${gcls}">${pc.group}</span>
        <span class="chip s">${q.question_id}</span>
        ${q.answerable?'':'<span class="chip g-suspect">cevaplanamaz</span>'}
      </div>
      <h2>${esc(q.question)}</h2>
      <div class="lbl">Referans cevap</div>
      <div class="ref">${esc(q.reference_answer)||'<em>—</em>'}</div>
      <div class="lbl">Kanıt alıntıları · ✓ sayfada bulundu</div>
      <ul class="spans">${spans.map(s=>
        `<li class="${s.found===false?'miss':''}">${s.found===false?'✗':'✓'} ${esc(s.span)}</li>`
      ).join('')}</ul>
      ${pc.notes.length?`<div class="notes">${pc.notes.map(esc).join('<br>')}</div>`:''}
      ${q.verification_note?`<div class="lbl">Önceki not</div>
        <div class="ref">${esc(q.verification_note)}</div>`:''}
      <div class="lbl">Altın sayfa</div>
      <div class="ref">${q.gold_page_ids.join(', ')||'—'}
        ${q.gold_article_ids.length?'· '+q.gold_article_ids.join(', '):''}</div>
    </div>
    <div class="imgs">${q.gold_page_ids.map(p=>
      `<div><div class="cap">${p}</div>
       <img loading="lazy" src="/img/${encodeURIComponent(p)}" alt="${p}"></div>`
    ).join('')}</div>`;
  $('note').value=''; $('err').textContent='';
}

async function decide(letter){
  const q=cur(); if(!q) return;
  const r=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question_id:q.question_id,letter,note:$('note').value.trim()})});
  const j=await r.json();
  if(j.error){ $('err').textContent=j.error; return; }
  if(letter==='a'){ i++; S.human=j.human; render(); return; }
  S=j; i=0; render();          // onay/ret satırı kuyruktan düşer
  window.scrollTo({top:0});
}
const ESC={'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'};
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>ESC[c]);}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') { if(e.key==='Enter') decide('e'); return; }
  if(e.key==='e'||e.key==='E') decide('e');
  if(e.key==='h'||e.key==='H') decide('h');
  if(e.key==='a'||e.key==='A') decide('a');
});
load();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH_V2)
    ap.add_argument("--by", required=True, help="doğrulayan kişi (verified_by alanına yazılır)")
    ap.add_argument(
        "--slice",
        action="append",
        dest="slices",
        help="yalnız bu dilim(ler); verilmezse G1.2'nin dört karar dilimi",
    )
    ap.add_argument("--all-slices", action="store_true", help="dilim filtresi uygulama")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    slices = None if args.all_slices else frozenset(args.slices or DECISION_SLICES)
    state = ReviewState(args.bench, args.by, slices)
    pending = state.queue_ids()
    if not pending:
        print("insan doğrulaması bekleyen satır yok (bu filtrede).")
        return 0
    print(f"{len(pending)} satır için ön-kontrol koşuluyor (PDF metni çıkarılıyor)…")
    state.build_prechecks()

    url = f"http://127.0.0.1:{args.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, state=state))
    print(f"inceleme arayüzü: {url}   (Ctrl-C ile bitir)")
    print(f"kuyruk: {len(pending)} satır · doğrulayan: {args.by} · dosya: {args.bench}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbitti — her karar zaten diske yazıldı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
