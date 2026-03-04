#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

# ---------------- Configuration ---------------- #

WORKERS=8
DATA_SRC="data"
DATA_DEST="../binary-jepa/data"

# ---------------- Utils ---------------- #

fail() {
  echo "[ERROR] $1" >&2
  exit 1
}

info() {
  echo "[INFO] $1"
}

# ---------------- Code ---------------- #

check_data() {
  local force="${1:-false}"

  if [[ ! -d "$DATA_SRC" ]] \
     || [[ -z "$(ls -A "$DATA_SRC" 2>/dev/null)" ]] \
     || [[ "$force" == "true" ]]; then

    info "Step 1 : Generation of ELF dataset"
    ( cd external/dataGen && ./generate_data.sh )
  fi
}

# ---------------- Main ---------------- #

main() {
  local force="${1:-false}"

  check_data "$force"

  info "Step 2 : Processing of ELF dataset"

  mkdir -p "$DATA_DEST"
  command -v python >/dev/null || fail "Python not found"

  python src/elf_processing_core.py \
    "$DATA_SRC" \
    "$WORKERS" \
    "$DATA_DEST"
}

main "$@"
