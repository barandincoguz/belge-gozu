#!/usr/bin/env bash
# Ortak lab venv'inin (zsh'te `lab` alias'ı) ÜSTÜNE ince bir overlay kurar.
#
# NEDEN OVERLAY. Bu makinede /opt/llm-lab/.venv 5 GB'lık ORTAK bir ortamdır;
# torch, transformers, sentence-transformers orada, 170 GB'lık model önbelleği
# de /opt/llm-lab/hf-cache altındadır (HF_HOME .zshrc'de zaten oraya bakar).
# İkinci bir tam venv (`uv sync --all-extras`) aynı torch'u yeniden indirip
# diski birkaç GB şişirirdi. Ortak venv student2 tarafından YAZILAMAZ
# (hemekci:llmlab, drwxr-xr-x), yani eksik paket oraya kurulamaz.
#
# Çözüm: proje içinde ~55 MB'lık bir overlay venv. Ortak site-packages bir
# `.pth` ile sys.path'e EKLENİR; overlay kendi paketlerini sys.path'te ÖNCE
# koyduğu için sürüm farkı (numpy) overlay'den kazanır, gerisi ortaktan gelir.
#
# BİLEREK KABUL EDİLEN İKİ SAPMA (uv.lock ile fark; README'de de yazılı):
#   * Python 3.11.15 — ortak venv'in sürümü. pyproject `>=3.12` istiyor, bu
#     yüzden proje `pip install --ignore-requires-python` ile kurulur. Kaynak
#     3.11'de derleniyor ve testlerin tamamı geçiyor; 3.12'ye özgü çalışma
#     zamanı API'si kullanılmıyor. CI ve Docker 3.12'de KALIR.
#   * torch 2.11 + transformers 5.3 — ortak venv'in sürümleri. İkisi de
#     pyproject aralığını sağlar (torch>=2.4, transformers>=5.3,<6) ama
#     kilitli çözüm (torch 2.13) değildir. colpali-engine==0.3.18 pini kilitli
#     yığında doğrulanmıştı; görsel kodlama koşulacaksa bu revalidasyon borcu.
#
# Betik idempotenttir: tekrar çalıştırmak overlay'i tazeler, ortak venv'e
# DOKUNMAZ.
set -euo pipefail

cd "$(dirname "$0")/.."

LAB_VENV="${LAB_VENV:-/opt/llm-lab/.venv}"
OVERLAY="${OVERLAY:-.venv-lab}"
LAB_PY="$LAB_VENV/bin/python"

[ -x "$LAB_PY" ] || { echo "HATA: ortak lab venv'i yok: $LAB_PY" >&2; exit 1; }

LAB_SITE="$("$LAB_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PY_TAG="$("$LAB_PY" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
echo "ortak venv : $LAB_VENV ($PY_TAG)"

# 1) Overlay venv. `--seed` pip getirir: editable kurulum ve
#    --ignore-requires-python yalnız pip'te var (uv'de yok).
if [ ! -x "$OVERLAY/bin/python" ]; then
  if [ -x "$LAB_VENV/bin/uv" ]; then
    "$LAB_VENV/bin/uv" venv --python "$LAB_PY" --seed "$OVERLAY" >/dev/null
  else
    "$LAB_PY" -m venv "$OVERLAY"
  fi
fi
PY="$OVERLAY/bin/python"
PIP="$OVERLAY/bin/pip"

# 2) Ortak site-packages'ı zincirle. Yol MUTLAK olduğu için overlay'i
#    etkinleştirmek yeter — `lab` alias'ını ayrıca çağırmak gerekmez.
echo "$LAB_SITE" > "$OVERLAY/lib/$PY_TAG/site-packages/_lab_shared.pth"

# 3) Delta paketler. Kilitli sürüm bu Python'da yoksa tabana düşülür
#    (numpy 2.5.2 yalnız >=3.12 tekerleği yayımlıyor).
pin() {
  "$LAB_PY" - "$1" <<'PYEOF'
import pathlib, sys, tomllib
lock = tomllib.loads(pathlib.Path("uv.lock").read_text())
print({p["name"]: p["version"] for p in lock["package"]}.get(sys.argv[1], ""))
PYEOF
}
install_pinned() {
  local name="$1" fallback="$2" ver
  ver="$(pin "$name")"
  if [ -n "$ver" ] && "$PIP" install -q "$name==$ver" 2>/dev/null; then
    echo "  $name==$ver (kilitli)"
  else
    "$PIP" install -q "$fallback"
    echo "  $fallback (kilitli sürüm $PY_TAG'de yok)"
  fi
}
echo "delta paketler:"
# numpy: ortak ortamda 1.26.4 var, `np.bitwise_count` 2.0'da geldi ve ikili
# MaxSim'in sıcak yolunda — 2.0 altında 84 test AttributeError ile düşüyor.
install_pinned numpy "numpy>=2,<3"
# Gate araçları CI ile AYNI sürümde koşsun: ortak venv ruff 0.15/pyright 1.1.408
# taşıyor, kilit 0.16.4/1.1.411 diyor ve ruff'ın biçimlendiricisi sürümle değişir.
install_pinned ruff "ruff"
install_pinned pytest "pytest"
install_pinned pyright "pyright"

# 4) Projenin kendisi: editable, bağımlılıksız (hepsi zaten ortak venv'de),
#    requires-python kapısı atlanarak — sapma yukarıda gerekçeli.
"$PIP" install -q --no-deps --ignore-requires-python -e . && echo "  belge-gozu (editable, --ignore-requires-python)"

# 5) Doğrulama: sessiz kurulum değil, ÖLÇÜLMÜŞ ortam raporu.
"$PY" - <<'PYEOF'
import numpy, torch, transformers, belge_gozu  # noqa: F401
import importlib.util as u, sys
print(f"python     : {sys.version.split()[0]}")
print(f"numpy      : {numpy.__version__}  bitwise_count={hasattr(numpy, 'bitwise_count')}")
print(f"torch      : {torch.__version__}  mps={torch.backends.mps.is_available()}")
print(f"transformers: {transformers.__version__}")
if torch.backends.mps.is_available():
    print(f"Metal bütçesi: {torch.mps.recommended_max_memory() / 1024**3:.1f} GiB "
          "(4B 12 GiB, 8B 24 GiB ister)")
if u.find_spec("colpali_engine") is None:
    print("not: colpali-engine YOK — yalnız görsel kodlama (`index encode`) ister; "
          "dense/semantic kollarında kullanılmaz.")
PYEOF

echo
echo "hazır:  source $OVERLAY/bin/activate    (make lint / make test overlay'i kendi bulur)"
