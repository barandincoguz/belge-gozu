"""Apple Silicon üzerinde doğrulanmış dense artefakt üretir.

Bu betik Colab notebook'unun yerini alır. Aynı işi yapar — sabit kaynak indeks
commit'inden başlar, `scripts/build_dense_artifacts.py` ile kodlar, sonucu
manifest sözleşmesine karşı doğrular, istenirse ayrı Hugging Face Dataset'ine
yükler — ama tek fark taşıyıcıdır: kod artık zaten yerel repodadır, checkpoint
Drive yerine diskte durur, GPU bütçesi Metal'den okunur.

Üç tasarım kararı ölçümle geldi:

1. **Bütçe kapısı ÖNCE koşar.** 24 GiB'lık bir makinede PyTorch'un gerçek Metal
   bütçesi 17,8 GiB'dir; 8B fp16 ağırlıkları oraya sığmaz. Bu kapı olmadan
   koşum, saatler sonra `resume` dosyası yarım kalmış hâlde ölüyor — yerelde
   tam olarak bu oldu (4.222 sayfanın 192'si).
2. **Her model AYRI süreçte kodlanır.** MPS, 8B ağırlıklarını süreç içinde
   `empty_cache()` sonrası bile tam bırakmıyor; ikinci model ilkinin artığıyla
   çarpışıyor. Süreç sınırı bunu işletim sistemine devrediyor.
3. **Doğrulanmamış artefakt yayımlanmaz.** `push_dense_artifact` zaten kendi
   içinde doğruluyor; burada yayım KAPALI olsa bile doğrulama koşuyor, çünkü
   yerelde kalan bir artefakt da ölçüme girecek.

    uv run python scripts/build_dense_artifacts_local.py \
      --index-dir "$BG_INDEX_DIR" \
      --source-revision 700ac324fffefb22de02c8e90347b31185547948

Yayım isteniyorsa `HF_TOKEN` ortam değişkeni write yetkili olmalıdır.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.dense_artifact_hub import push_dense_artifact  # noqa: E402
from belge_gozu.bench.dense_artifacts import (  # noqa: E402
    DenseArtifactExpectation,
    dense_model_key,
    sha256_file,
    validate_dense_artifact,
)
from belge_gozu.retrieval.dense import DENSE_MODELS, DenseModelSpec  # noqa: E402
from belge_gozu.retrieval.hybrid import load_page_texts  # noqa: E402

GIB = 1024**3
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

#: Küçük model önce: hızlı biter ve büyük model sığmazsa elde ölçülebilir bir kol kalır.
DEFAULT_MODELS = ["qwen3-embedding-4b", "qwen3-embedding-8b"]

#: fp16 ağırlık + 8192 token aktivasyonu için gereken Metal çalışma kümesi.
#: 4B ~8 GB, 8B ~15 GB ağırlık; kalan pay aktivasyon ve tokenizer içindir.
MODEL_MINIMUM_BUDGET_BYTES = {
    "qwen3-embedding-4b": 12 * GIB,
    "qwen3-embedding-8b": 24 * GIB,
}


class UnsupportedDevice(RuntimeError):
    """Metal/MPS bulunmayan bir makinede dense üretimi denendiğini bildirir."""


class InsufficientUnifiedMemory(RuntimeError):
    """İstenen modelin bu makinenin Metal bütçesine sığmadığını bildirir."""


def mps_budget_bytes(torch_module: Any | None = None) -> int:
    """PyTorch'un bu makinede kullanabileceği Metal çalışma kümesi (bayt).

    Toplam RAM DEĞİL: macOS, PyTorch'a fiziksel belleğin bir oranını verir ve
    karar veren sayı odur.
    """
    if torch_module is None:
        import torch

        torch_module = torch
    if not torch_module.backends.mps.is_available():
        raise UnsupportedDevice(
            "dense artefakt üretimi Apple Silicon GPU (Metal) ister; bu makinede MPS yok"
        )
    return int(torch_module.mps.recommended_max_memory())


def require_models_fit(model_keys: list[str], budget_bytes: int) -> None:
    """Sığmayan modelleri, TEK bir sayfa kodlanmadan önce reddeder."""
    unfit = [
        f"{key} ({MODEL_MINIMUM_BUDGET_BYTES[key] / GIB:.0f} GiB gerekir)"
        for key in model_keys
        if budget_bytes < MODEL_MINIMUM_BUDGET_BYTES[key]
    ]
    if unfit:
        raise InsufficientUnifiedMemory(
            f"Metal bütçesi {budget_bytes / GIB:.1f} GiB; sığmayan model(ler): {', '.join(unfit)}"
        )


def run_model_build(command: list[str]) -> dict[str, Any]:
    """Tek modeli ayrı süreçte kodlar ve durum satırını çözer."""
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stderr.write(result.stderr)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode not in (0, 2) or not lines:
        raise RuntimeError(f"dense üretimi başarısız (çıkış {result.returncode}): {result.stdout}")
    return json.loads(lines[-1])


def load_expectation(spec: DenseModelSpec, index_dir: Path) -> DenseArtifactExpectation:
    """Yerel indeksin sayfa kimliğini artefakt beklentisine dönüştürür."""
    return DenseArtifactExpectation(
        model=spec,
        page_ids=list(load_page_texts(index_dir)),
        page_texts_sha256=sha256_file(index_dir / "page_texts.parquet"),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--source-repo", default="barandincoguz/belge-gozu-index")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model", action="append", choices=sorted(DENSE_MODELS))
    parser.add_argument("--artifact-root", type=Path, default=Path("data/bench/dense-artifacts"))
    parser.add_argument("--artifact-repo", default="barandincoguz/belge-gozu-semantic-artifacts")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args(argv)
    args.model = args.model or list(DEFAULT_MODELS)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not _COMMIT_SHA.fullmatch(args.source_revision):
        raise ValueError("--source-revision 40 karakterli değişmez commit SHA olmalı")
    if args.batch_size < 1:
        raise ValueError("--batch-size en az 1 olmalı")

    require_models_fit(args.model, mps_budget_bytes())

    exit_code = 0
    for model_key in args.model:
        status = run_model_build(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_dense_artifacts.py"),
                "--index-dir",
                str(args.index_dir),
                "--source-repo",
                args.source_repo,
                "--source-revision",
                args.source_revision,
                "--model",
                model_key,
                "--artifact-root",
                str(args.artifact_root),
                "--batch-size",
                str(args.batch_size),
            ]
        )
        if status.get("status") == "in_progress":
            print(f"{model_key}: kesildi; aynı komut checkpoint'ten sürdürür", flush=True)
            continue
        if status.get("status") != "ok":
            print(f"{model_key}: {status}", flush=True)
            exit_code = 2
            continue

        spec = DENSE_MODELS[model_key]
        artifact_dir = Path(args.artifact_root) / dense_model_key(spec)
        expectation = load_expectation(spec, args.index_dir)
        validate_dense_artifact(artifact_dir, expectation)
        print(f"{model_key}: doğrulandı -> {artifact_dir}", flush=True)
        if args.push:
            commit = push_dense_artifact(
                artifact_dir,
                args.artifact_repo,
                model_key,
                expectation,
                token=os.environ.get("HF_TOKEN", ""),
            )
            print(f"{model_key}: yayımlandı; commit={commit}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
