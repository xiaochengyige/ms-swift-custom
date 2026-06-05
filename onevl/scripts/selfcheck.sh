#!/bin/bash
set -x
# ============================================================
# Self-check wrapper. Provides import resolution via PYTHONPATH only
# (the .py files never touch sys.path):
#   - ONEVL_DIR  -> so `register` / `onevl_plugin` are importable
#   - SWIFT_ROOT -> so `swift` (+ torch/transformers) are importable
# ============================================================
ONEVL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_ROOT="$(cd "${ONEVL_DIR}/../ms-swift-4.2.0" && pwd)"

PYTHONPATH="${ONEVL_DIR}:${SWIFT_ROOT}:${PYTHONPATH}" python "${ONEVL_DIR}/selfcheck.py"
