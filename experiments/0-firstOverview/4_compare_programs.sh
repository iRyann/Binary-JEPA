#!/usr/bin/env bash
# ==============================================================================
# 4_compare_programs.sh
#
# Pour chaque couple de JSONs d'embeddings dans latentSpace/data/embeddings/,
# appelle `elf64ctx --compare <json1> <json2> --plot` et sauvegarde le heatmap
# généré dans latentSpace/data/comparisons/.
#
# Contournement du bug cli.py:95 (condition -O inversée) :
#   On lance le CLI depuis COMPARISONS_DIR ; le heatmap est créé dans
#   COMPARISONS_DIR/output/heatmap.png, puis déplacé vers
#   COMPARISONS_DIR/<label1>_vs_<label2>.png
#
# Pré-requis :
#   - elf64ctx installé : pip install -e latentSpace/elf64-context-hash/
#   - Avoir exécuté 2_generate_embeddings.sh au préalable
#
# Usage : bash 4_compare_programs.sh  (depuis la racine du projet)
#
# Sorties : latentSpace/data/comparisons/<label1>_vs_<label2>.png
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

# ─── Couleurs ─────────────────────────────────────────────────────────────────
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
EMBEDDINGS_DIR="$PROJECT_ROOT/latentSpace/data/embeddings"
COMPARISONS_DIR="$PROJECT_ROOT/latentSpace/data/comparisons"

# ─── Résolution de la commande CLI ────────────────────────────────────────────
resolve_cli() {
    if command -v elf64ctx &>/dev/null; then
        echo "elf64ctx"
    elif python -c "import elf64_context_hash" &>/dev/null 2>&1; then
        echo "python -m elf64_context_hash"
    else
        log_error "elf64ctx introuvable."
        log_error "Installez-le avec : pip install -e latentSpace/elf64-context-hash/"
        exit 1
    fi
}

# ─── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -d "$EMBEDDINGS_DIR" ]]; then
    log_error "Dossier d'embeddings introuvable : $EMBEDDINGS_DIR"
    log_error "Exécutez d'abord : bash 2_generate_embeddings.sh"
    exit 1
fi

mapfile -t JSON_FILES < <(find "$EMBEDDINGS_DIR" -maxdepth 1 -name "*.json" | sort)

if (( ${#JSON_FILES[@]} < 2 )); then
    log_error "Il faut au moins 2 fichiers JSON dans $EMBEDDINGS_DIR"
    log_error "Exécutez d'abord : bash 2_generate_embeddings.sh"
    exit 1
fi

mkdir -p "$COMPARISONS_DIR"
mkdir -p "$COMPARISONS_DIR/output"

CLI_CMD=$(resolve_cli)
N=${#JSON_FILES[@]}
N_PAIRS=$(( N * (N - 1) / 2 ))

log_info "CLI détecté   : $CLI_CMD"
log_info "JSONs trouvés : $N dans $EMBEDDINGS_DIR"
log_info "Couples       : $N_PAIRS à comparer"
log_info "Sorties       : $COMPARISONS_DIR"

# ─── Helper : label court depuis le nom de fichier ────────────────────────────
# "ls_gcc_O0.elf-embeddings.json" → "ls_gcc_O0"
make_label() {
    local fname
    fname="$(basename "$1")"
    fname="${fname%.json}"
    fname="${fname%.elf-embeddings}"
    echo "$fname"
}

# ==============================================================================
# Boucle sur tous les couples
# ==============================================================================
log_section "Comparaisons"

SUCCESS=0
FAILURES=0
SKIPPED=0
TOTAL_START=$(date +%s)

for (( i = 0; i < N; i++ )); do
    for (( j = i + 1; j < N; j++ )); do
        json1="${JSON_FILES[$i]}"
        json2="${JSON_FILES[$j]}"

        label1=$(make_label "$json1")
        label2=$(make_label "$json2")
        pair_label="${label1}_vs_${label2}"
        out_png="$COMPARISONS_DIR/${pair_label}.png"

        if [[ -f "$out_png" ]]; then
            log_warn "Déjà existant, ignoré : ${pair_label}.png"
            (( SKIPPED++ )) || true
            continue
        fi

        log_info "Comparaison : $label1  vs  $label2"
        START=$(date +%s)

        # Lancer depuis COMPARISONS_DIR pour que output/heatmap.png y soit créé
        pushd "$COMPARISONS_DIR" > /dev/null

        if $CLI_CMD --compare "$json1" "$json2" --plot -O dummy; then
            cli_status=0
        else
            cli_status=$?
        fi

        popd > /dev/null

        END=$(date +%s)
        ELAPSED=$(( END - START ))

        cli_output="$COMPARISONS_DIR/output/heatmap.png"

        if [[ $cli_status -ne 0 ]] || [[ ! -f "$cli_output" ]]; then
            log_error "Échec pour $pair_label (code=$cli_status)"
            (( FAILURES++ )) || true
            continue
        fi

        mv "$cli_output" "$out_png"
        log_ok "${pair_label}.png  (${ELAPSED}s)"
        (( SUCCESS++ )) || true
    done
done

# Nettoyage du dossier output/ temporaire s'il est vide
rmdir "$COMPARISONS_DIR/output" 2>/dev/null || true

# ==============================================================================
# Résumé
# ==============================================================================
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

log_section "COMPARAISONS TERMINÉES  (${TOTAL_ELAPSED}s)"

echo ""
printf "  %-6s %s\n" "$SUCCESS"  "heatmaps générés avec succès"
printf "  %-6s %s\n" "$FAILURES" "échecs"
printf "  %-6s %s\n" "$SKIPPED"  "déjà existants (ignorés)"
echo ""

if (( FAILURES > 0 )); then
    log_warn "$FAILURES couple(s) n'ont pas pu être comparés."
fi

if (( SUCCESS > 0 )); then
    log_ok "Heatmaps disponibles dans : $COMPARISONS_DIR"
fi
