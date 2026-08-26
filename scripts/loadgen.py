"""Belge-Gözü yük üreticisi.

Varsayılan /search'tür: /ask Gemini kotası yakar (≈20 çağrı/gün) ve ancak
--endpoint ask --yes-burn-quota ile açılır. Örnek:
    uv run python scripts/loadgen.py --concurrency 8 --duration 60 --out out.json
"""

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import httpx

QUERIES = Path(__file__).with_name("queries_sample.txt")


def summarize(latencies_ms: list[float], errors: int, duration_s: float) -> dict:
    lat = sorted(latencies_ms)

    def pct(p: float) -> float:
        if not lat:
            return 0.0
        return lat[min(len(lat) - 1, max(0, math.ceil(p * len(lat)) - 1))]

    return {
        "requests": len(lat),
        "errors": errors,
        "rps": round(len(lat) / duration_s, 2) if duration_s > 0 else 0.0,
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
    }


async def worker(
    client: httpx.AsyncClient,
    endpoint: str,
    questions: list[str],
    stop_at: float,
    lat: list[float],
    errs: list[int],
) -> None:
    while time.monotonic() < stop_at:
        q = random.choice(questions)
        body = {"question": q} if endpoint == "/ask" else {"query": q}
        t0 = time.perf_counter()
        try:
            r = await client.post(endpoint, json=body, timeout=120)
            if r.status_code == 200:
                lat.append((time.perf_counter() - t0) * 1000)
            else:
                errs[0] += 1
        except Exception:
            errs[0] += 1


async def run(args: argparse.Namespace) -> dict:
    questions = [q.strip() for q in QUERIES.read_text().splitlines() if q.strip()]
    endpoint = "/ask" if args.endpoint == "ask" else "/search"
    lat: list[float] = []
    errs = [0]
    stop_at = time.monotonic() + args.duration
    async with httpx.AsyncClient(base_url=args.base_url) as client:
        t0 = time.monotonic()
        await asyncio.gather(
            *(
                worker(client, endpoint, questions, stop_at, lat, errs)
                for _ in range(args.concurrency)
            )
        )
        dur = time.monotonic() - t0
    return {
        "config": vars(args),
        "endpoint": endpoint,
        "duration_s": round(dur, 2),
        **summarize(lat, errs[0], dur),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:7860")
    ap.add_argument("--endpoint", choices=["search", "ask"], default="search")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--yes-burn-quota", action="store_true")
    args = ap.parse_args()
    if args.endpoint == "ask" and not args.yes_burn_quota:
        ap.error("/ask Gemini kotası yakar; bilinçliysen --yes-burn-quota ekle")
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
