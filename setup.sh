#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"

if [ ! -d "${VENV}" ]; then
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install -U pip wheel setuptools

python -m pip install -r "${ROOT}/jsonschemabench-epfl/requirements.txt"
python -m pip install -r "${ROOT}/schemabench/requirements.txt"
python -m pip install jsonlines gdown

python - <<'PY'
from datasets import load_dataset
load_dataset("epfl-dlab/JSONSchemaBench")
print("Cached JSONSchemaBench dataset.")
PY

DATA_ROOT="${ROOT}/schemabench/data"
if [ ! -d "${DATA_ROOT}" ] || [ -z "$(ls -A "${DATA_ROOT}" 2>/dev/null)" ]; then
  DOWNLOAD_DIR="${ROOT}/.downloads/schemabench"
  mkdir -p "${DOWNLOAD_DIR}"
  gdown --folder "https://drive.google.com/drive/folders/1NOx6xzS30HHRk5rikUdNOXvOT7UtwstR" -O "${DOWNLOAD_DIR}"

  if [ -d "${DOWNLOAD_DIR}/schemabench/data" ]; then
    cp -R "${DOWNLOAD_DIR}/schemabench/data" "${ROOT}/schemabench/"
  fi
  if [ -d "${DOWNLOAD_DIR}/data" ]; then
    cp -R "${DOWNLOAD_DIR}/data" "${ROOT}/schemabench/"
  fi
  if [ -d "${DOWNLOAD_DIR}/schemabench/train" ]; then
    cp -R "${DOWNLOAD_DIR}/schemabench/train" "${ROOT}/schemabench/"
  fi
  if [ -d "${DOWNLOAD_DIR}/train" ]; then
    cp -R "${DOWNLOAD_DIR}/train" "${ROOT}/schemabench/"
  fi
fi

echo "Setup complete."
