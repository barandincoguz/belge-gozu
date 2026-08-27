"""Koşum künyesi yardımcıları (manifest/rapor damgaları).

`git_commit` daha önce `bench/harness.py`'de yaşıyordu ve `cli.py` üretim
manifest'lerini damgalamak için oradan import ediyordu — üretim yolunu bench
paketine bağlıyordu (final review IMPORTANT-5). Ortak nokta buraya taşındı;
`bench.harness.git_commit` geriye dönük uyumluluk için re-export kalır.
"""

import subprocess


def git_commit() -> str:
    """Kısa HEAD sha'sı; git yoksa/başarısızsa "unknown"."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
