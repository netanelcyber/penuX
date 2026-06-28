#!/bin/bash
set -euo pipefail

OUT_DIR="${HOME}/Downloads"
OUT_FILE="${OUT_DIR}/mimic-iii-10k.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/bilal1907/mimic-iii-10k"

mkdir -p "${OUT_DIR}"
echo "[INFO] Downloading MIMIC-III-10k from Kaggle API..."
curl -L -o "${OUT_FILE}" "${URL}"
echo "[INFO] Saved: ${OUT_FILE}"
