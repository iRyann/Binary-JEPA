#!/usr/bin/env bash
# ==============================================================================
# 1_compile_coreutils.sh
#
# Compile 5 utilitaires (ls, cat, cp, mv depuis coreutils + grep depuis GNU grep)
# sous 3 configurations de compilation : gcc -O0, gcc -O3, clang -Os
#
# Pré-requis :
#   - gcc, clang installés (apt install gcc clang)
#   - Source coreutils-9.10 clonée dans elf-processing/external/dataGen/coreutils-9.10
#   - Source GNU grep (optionnelle) dans elf-processing/external/dataGen/grep
#     Si absente, téléchargez-la :
#       wget https://ftp.gnu.org/gnu/grep/grep-3.11.tar.xz -P /tmp
#       tar -xf /tmp/grep-3.11.tar.xz -C elf-processing/external/dataGen/
#       mv elf-processing/external/dataGen/grep-3.11 elf-processing/external/dataGen/grep
#
# Usage : bash 1_compile_coreutils.sh  (depuis la racine du projet)
#
# Sorties : latentSpace/data/binaries/<util>_<cc>_<opt>.elf
#           Exemple : ls_gcc_O0.elf, cat_clang_Os.elf
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

# ─── Couleurs pour les logs ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
log_section() { echo -e "\n${BOLD}══════════════════════════════════════════════════════${RESET}"; \
                echo -e "${BOLD}  $*${RESET}"; \
                echo -e "${BOLD}══════════════════════════════════════════════════════${RESET}"; }

# ─── Chemins ──────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COREUTILS_SRC="$PROJECT_ROOT/elf-processing/external/dataGen/coreutils-9.10"
GREP_SRC="$PROJECT_ROOT/elf-processing/external/dataGen/grep"
OUTPUT_DIR="$PROJECT_ROOT/latentSpace/data/binaries"

# ─── Utilitaires à extraire ───────────────────────────────────────────────────
# ls, cat, cp, mv se trouvent dans coreutils/src/
# grep est un paquet GNU séparé — traité dans build_grep()
COREUTILS_UTILS=("ls" "cat" "cp" "mv")

# ─── Configurations compilateur ───────────────────────────────────────────────
# Format : "CC|CFLAGS|LABEL_COURT"
CONFIGS=(
    "gcc|-O0|gcc_O0"
    "gcc|-O3 -fno-omit-frame-pointer|gcc_O3"
    "clang|-Os|clang_Os"
)

# ─── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -d "$COREUTILS_SRC" ]]; then
    log_error "Sources coreutils introuvables : $COREUTILS_SRC"
    log_error "Lancez ce script depuis la racine du projet."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
log_info "Dossier de sortie : $OUTPUT_DIR"

# ==============================================================================
# Fonction : compiler coreutils et extraire les binaires demandés
# Arguments : $1=CC, $2=CFLAGS, $3=LABEL
# ==============================================================================
build_coreutils() {
    local cc="$1"
    local cflags="$2"
    local label="$3"

    log_section "Coreutils  CC=$cc  CFLAGS='$cflags'"

    # Vérifier que le compilateur est disponible
    if ! command -v "$cc" &>/dev/null; then
        log_warn "Compilateur '$cc' introuvable — configuration '$label' ignorée."
        return 0
    fi
    log_info "Compilateur : $(command -v "$cc")  version : $("$cc" --version | head -1)"

    pushd "$COREUTILS_SRC" > /dev/null

    # --- Étape 1 : nettoyage ---
    log_info "[1/3] Nettoyage des artefacts précédents..."
    # distclean supprime aussi le Makefile généré par configure
    make distclean &>/dev/null || make clean &>/dev/null || true

    # --- Étape 2 : configure ---
    log_info "[2/3] Configuration (CC=$cc CFLAGS='$cflags')..."
    if ! CC="$cc" CFLAGS="$cflags" ./configure \
            --disable-nls \
            --quiet 2>&1; then
        log_warn "configure a retourné une erreur — tentative de build quand même."
    fi

    # --- Étape 3 : compilation ---
    log_info "[3/3] Compilation ($(nproc) jobs parallèles)..."
    if ! make -j"$(nproc)" &>/dev/null; then
        log_error "make a échoué pour CC=$cc CFLAGS='$cflags'"
        popd > /dev/null
        return 1
    fi

    popd > /dev/null

    # --- Extraction des binaires ---
    local found=0
    local missing=0
    for util in "${COREUTILS_UTILS[@]}"; do
        local src_bin="$COREUTILS_SRC/src/$util"
        local dst_bin="$OUTPUT_DIR/${util}_${label}.elf"

        if [[ -f "$src_bin" ]]; then
            cp "$src_bin" "$dst_bin"
            local size
            size="$(du -h "$dst_bin" | cut -f1)"
            log_ok "${util}_${label}.elf  ($size)"
            (( found++ )) || true
        else
            log_warn "Binaire '$util' introuvable dans $COREUTILS_SRC/src/"
            (( missing++ )) || true
        fi
    done

    log_info "Résultat '$label' : $found extraits, $missing manquants."
}

# ==============================================================================
# Fonction : compiler GNU grep séparément (n'est PAS dans coreutils)
# Arguments : $1=CC, $2=CFLAGS, $3=LABEL
# ==============================================================================
build_grep() {
    local cc="$1"
    local cflags="$2"
    local label="$3"
    local dst_bin="$OUTPUT_DIR/grep_${label}.elf"

    log_section "GNU grep  CC=$cc  CFLAGS='$cflags'"

    if [[ ! -d "$GREP_SRC" ]]; then
        log_warn "Source GNU grep introuvable : $GREP_SRC"
        log_warn "Pour l'obtenir :"
        log_warn "  wget https://ftp.gnu.org/gnu/grep/grep-3.11.tar.xz -P /tmp"
        log_warn "  tar -xf /tmp/grep-3.11.tar.xz -C elf-processing/external/dataGen/"
        log_warn "  mv elf-processing/external/dataGen/grep-3.11 $GREP_SRC"
        log_warn "Configuration '$label' pour grep ignorée."
        return 0
    fi

    if ! command -v "$cc" &>/dev/null; then
        log_warn "Compilateur '$cc' introuvable — grep '$label' ignoré."
        return 0
    fi

    pushd "$GREP_SRC" > /dev/null

    log_info "[1/3] Nettoyage..."
    make distclean &>/dev/null || make clean &>/dev/null || true

    log_info "[2/3] Configuration (CC=$cc CFLAGS='$cflags')..."
    CC="$cc" CFLAGS="$cflags" ./configure \
        --disable-nls \
        --quiet 2>/dev/null || true

    log_info "[3/3] Compilation..."
    if ! make -j"$(nproc)" &>/dev/null; then
        log_error "make grep a échoué pour CC=$cc"
        popd > /dev/null
        return 1
    fi

    popd > /dev/null

    # grep se trouve dans src/grep après compilation
    local src_bin="$GREP_SRC/src/grep"
    if [[ -f "$src_bin" ]]; then
        cp "$src_bin" "$dst_bin"
        log_ok "grep_${label}.elf  ($(du -h "$dst_bin" | cut -f1))"
    else
        log_warn "Binaire grep introuvable dans $GREP_SRC/src/ après compilation."
    fi
}

# ==============================================================================
# Boucle principale sur les configurations
# ==============================================================================
TOTAL_START=$(date +%s)

for config in "${CONFIGS[@]}"; do
    IFS="|" read -r cc cflags label <<< "$config"

    build_coreutils "$cc" "$cflags" "$label"
    build_grep      "$cc" "$cflags" "$label"
done

# ==============================================================================
# Résumé final
# ==============================================================================
TOTAL_END=$(date +%s)
ELAPSED=$(( TOTAL_END - TOTAL_START ))

log_section "COMPILATION TERMINÉE  (${ELAPSED}s)"

echo ""
printf "  %-30s %s\n" "Fichier" "Taille"
printf "  %-30s %s\n" "──────────────────────────────" "──────"
for f in "$OUTPUT_DIR"/*.elf; do
    if [[ -f "$f" ]]; then
        printf "  %-30s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
    fi
done

ACTUAL=$(find "$OUTPUT_DIR" -name "*.elf" 2>/dev/null | wc -l)
EXPECTED=15  # 5 utils × 3 configs
echo ""
log_info "Attendus : $EXPECTED ELFs  |  Trouvés : $ACTUAL ELFs"

if (( ACTUAL < EXPECTED )); then
    log_warn "Il manque $(( EXPECTED - ACTUAL )) binaires."
    log_warn "Vérifiez que gcc, clang sont installés et que les sources grep sont présentes."
else
    log_ok "Tous les binaires ont été générés avec succès !"
fi
