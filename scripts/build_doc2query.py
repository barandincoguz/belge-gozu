"""Her madde chunk'ı için günlük-dil sorular üretir (doc2query).

    uv run python scripts/build_doc2query.py --limit 200 --out /tmp/pilot.jsonl
    uv run python scripts/build_doc2query.py

NEDEN. Ölçülen arıza kayıt uyuşmazlığı: korpus kanun dilinde ("veri
sorumlusu"), sorular günlük dilde ("şirket müşteri verisi"). Sonda kanıtladı —
sorgu elle kanun diline çevrilince gold rank 1'e geliyor (300/bulunamadı/88 ->
1/1/1). Sorgu-zamanı çeviri (genişletme) ÖLÇÜLDÜ: havuzu düzeltiyor, ilk beşi
düzeltmiyor ve sorgu başına 15 GB'lık bir checkpoint'ten generate istiyor —
üretim CPU'da koşarken bu yol kapalı.

doc2query bedeli İNDEKS zamanına taşır: maddeye, o maddenin günlük dilde
cevapladığı sorular eklenir ve BM25 onları da eşleştirir. Sorgu-zamanı maliyeti
yok (literatürde BM25 55 -> 58 ms). Üretilen metin YALNIZ eşleştirme içindir,
kullanıcıya gösterilen sayfa metni değişmez.

Üretim ANINDA diske yazılır: 10.531 chunk'lık koşum saatler sürer, yarıda
kesilen koşum tamamlanmış üretimleri kaybetmemelidir (dense artefakt
disiplininin aynısı).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.config import Settings  # noqa: E402

PROMPT = (
    "Aşağıdaki kanun maddesini okuyan sıradan bir vatandaşın, bu maddeyi bulmak için "
    "günlük Türkçeyle soracağı 3 farklı soru yaz. Hukuk terimi KULLANMA; maddeden "
    "kelime kopyalama. Her satıra bir soru yaz, başka hiçbir şey yazma."
)


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repo", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    index_dir = Settings().index_dir
    out = args.out or index_dir / "chunk_questions.jsonl"
    chunks = pd.read_parquet(index_dir / "chunks.parquet")

    done: set[str] = set()
    if out.exists():
        done = {
            json.loads(line)["chunk_id"]
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        print(f"{len(done)} chunk zaten üretilmiş; sürdürülüyor", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.repo, revision=args.revision, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.repo, revision=args.revision, dtype=torch.float16
    ).to("mps")
    model.eval()

    rows = chunks if args.limit is None else chunks.head(args.limit)
    pending = [
        (str(chunk_id), str(text))
        for chunk_id, text in zip(rows["chunk_id"], rows["text"], strict=True)
        if str(chunk_id) not in done
    ]
    print(f"{len(pending)} chunk üretilecek", flush=True)

    started = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for index, (chunk_id, text) in enumerate(pending, start=1):
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text[:4000]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            encoded = tokenizer(prompt, return_tensors="pt")
            batch = {name: value.to("mps") for name, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **batch, do_sample=False, max_new_tokens=args.max_new_tokens
                )
            decoded = tokenizer.decode(
                generated[0, batch["input_ids"].shape[1] :], skip_special_tokens=True
            )
            questions = [line.strip(" -•\t0123456789.") for line in decoded.splitlines()]
            questions = [question for question in questions if len(question) > 10][:3]
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk_id,
                        "questions": questions,
                        "model_repo": args.repo,
                        "model_revision": args.revision,
                        "prompt_sha256": prompt_sha256(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if index % 25 == 0:
                rate = index / (time.perf_counter() - started)
                print(f"  {index}/{len(pending)}  {rate:.2f} chunk/s", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
