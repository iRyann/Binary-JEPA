#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

# ---------------- Configuration ---------------- #

VERSION="9.10"
DATA_SRC="https://github.com/coreutils/coreutils/releases/download/v${VERSION}/coreutils-${VERSION}.tar.xz"

SRC_DIR="coreutils-${VERSION}"
ARCHIVE="coreutils-${VERSION}.tar.xz"
DATASET_DIR="data"

COMPILERS=("gcc" "clang")
OPT_FLAGS=("-O0" "-O1" "-O2" "-O3" "-Os")

BINARIES=(
  arch b2sum base32 base64 basename basenc cat chcon chgrp chmod chown
  chroot cksum comm coreutils cp csplit cut date dd df dir dircolors dirname
  du echo env expand expr factor false fmt fold groups head hostid hostname
  id install join kill link ln logname ls md5sum mkdir mkfifo mknod mktemp
  mv nice nl nohup nproc numfmt od paste pathchk pinky pr printenv printf ptx
  pwd readlink realpath rm rmdir runcon seq sha1sum sha224sum sha256sum
  sha384sum sha512sum shred shuf sleep sort split stat stdbuf stty sum sync
  tac tail tee test timeout touch tr true truncate tsort tty uname unexpand
  uniq unlink uptime users vdir wc who whoami yes
)

# ---------------- Utils ---------------- #

fail() {
  echo "[ERROR] $1" >&2
  exit 1
}

info() {
  echo "[INFO] $1"
}

# ---------------- Data acquisition ---------------- #

get_sources() {
  if [[ ! -d "$SRC_DIR" ]]; then
    info "Downloading coreutils ${VERSION}"
    wget -q "$DATA_SRC" -O "$ARCHIVE" || fail "Download failed"
    tar -xf "$ARCHIVE" || fail "Extraction failed"
  fi

  cd "$SRC_DIR" || fail "Cannot enter source directory"

  if [[ ! -f configure ]]; then
    ./bootstrap || fail "Bootstrap failed"
  fi
}

# ---------------- Compilation ---------------- #

compile_variant() {
  local compiler="$1"
  local opt="$2"

  info "Building with ${compiler} ${opt}"

  make distclean >/dev/null 2>&1 || true

  mkdir -p build
  cd build

  ../configure \
    CC="$compiler" \
    CFLAGS="$opt" \
    --disable-nls \
    --quiet \
    >/dev/null || fail "Configure failed"

  make -j"$(nproc)" --quiet || fail "Make failed"

  for binary in "${BINARIES[@]}"; do
    if [[ -f "src/$binary" ]]; then
      out_name="${binary}_${compiler}_${opt#-}.elf"
      cp "src/$binary" "../../../../${DATASET_DIR}/${out_name}"
      strip --strip-all "../../../../${DATASET_DIR}/${out_name}" 2>/dev/null || true
    fi
  done

  cd ..
  rm -rf build
}

compilation_pipeline() {
  mkdir -p "../${DATASET_DIR}"

  for cc in "${COMPILERS[@]}"; do
    command -v "$cc" >/dev/null 2>&1 || continue
    for opt in "${OPT_FLAGS[@]}"; do
      compile_variant "$cc" "$opt"
    done
  done
}

# ---------------- Main ---------------- #

main() {
  get_sources
  compilation_pipeline
  info "Dataset generation complete"
}

main
