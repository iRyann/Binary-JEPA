#!/usr/bin/env bash
# ==============================================================================
# 2_generate_embeddings.sh
#
# Itère sur tous les ELFs de latentSpace/data/binaries/ et génère les embeddings
# via l'outil CLI elf64ctx (python -m elf64_context_hash).
#
# ── Interface réelle du CLI ──────────────────────────────────────────────────
#   elf64ctx -E <elf_file1> [<elf_file2> ...] -O <dummy>
#
#   REMARQUE : il existe un bug dans cli.py:95 (condition inversée) qui fait que
#   la valeur passée à -O est ignorée. Le CLI crée toujours un dossier ./output/
#   dans le répertoire courant. On contourne ça en lançant le CLI depuis
#   latentSpace/data/binaries/ (les noms de fichiers restent courts et propres),
#   puis on déplace les JSONs produits vers latentSpace/data/embeddings/.
#
# Pré-requis :
#   - elf64ctx installé : pip install -e latentSpace/elf64-context-hash/
#     (ou disponible via `elf64ctx` dans le PATH)
#   - Avoir exécuté 1_compile_coreutils.sh au préalable
#
# Usage : bash 2_generate_embeddings.sh  (depuis la racine du projet)
#
# Sorties : latentSpace/data/embeddings/<util>_<cc>_<opt>.elf-embeddings.json
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
BINARIES_DIR="$PROJECT_ROOT/latentSpace/data/binaries"
EMBEDDINGS_DIR="$PROJECT_ROOT/latentSpace/data/embeddings"

# ─── Résolution de la commande CLI ────────────────────────────────────────────
# Priorité : e64ctx (installé) → python -m elf64_context_hash → erreur
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
if [[ ! -d "$BINARIES_DIR" ]]; then
    log_error "Dossier de binaires introuvable : $BINARIES_DIR"
    log_error "Exécutez d'abord : bash 1_compile_coreutils.sh"
    exit 1
fi

ELF_COUNT=$(find "$BINARIES_DIR" -maxdepth 1 -name "*.elf" | wc -l)
if (( ELF_COUNT == 0 )); then
    log_error "Aucun fichier .elf trouvé dans $BINARIES_DIR"
    log_error "Exécutez d'abord : bash 1_compile_coreutils.sh"
    exit 1
fi

mkdir -p "$EMBEDDINGS_DIR"

CLI_CMD=$(resolve_cli)
log_info "CLI détecté   : $CLI_CMD"
log_info "Binaires      : $ELF_COUNT ELFs dans $BINARIES_DIR"
log_info "Sorties       : $EMBEDDINGS_DIR"

# ==============================================================================
# Traitement de chaque ELF
# ==============================================================================
log_section "Génération des embeddings"

SUCCESS=0
FAILURES=0
SKIPPED=0
TOTAL_START=$(date +%s)

for elf_path in "$BINARIES_DIR"/*.elf; do
    [[ -f "$elf_path" ]] || continue
    elf_name="$(basename "$elf_path")"

    # Nom de sortie attendu par le CLI :
    # Le CLI génère <elf_name>-embeddings.json dans ./output/ (cf. cli.py:156)
    expected_json="$EMBEDDINGS_DIR/${elf_name}-embeddings.json"

    # Vérifier si déjà traité (mode reprise sur interruption)
    if [[ -f "$expected_json" ]]; then
        log_warn "Déjà existant, ignoré : $(basename "$expected_json")"
        (( SKIPPED++ )) || true
        continue
    fi

    log_info "Traitement : $elf_name"
    START=$(date +%s)

    # ── Contournement du bug output_path ──────────────────────────────────────
    # On se place dans le dossier des binaires pour que `file = Path("ls_gcc_O0.elf")`
    # soit un chemin court. Avec -O dummy, le CLI crée ./output/<name>-embeddings.json
    # dans le répertoire courant (i.e. BINARIES_DIR/output/).
    # On déplace ensuite vers EMBEDDINGS_DIR.

    pushd "$BINARIES_DIR" > /dev/null

    # Créer le dossier output/ si absent (le CLI le crée aussi, mais par sécurité)
    mkdir -p output/

    # Lancer le CLI
    if $CLI_CMD -E "$elf_name" -O dummy; then
        cli_status=0
    else
        cli_status=$?
    fi

    popd > /dev/null

    END=$(date +%s)
    ELAPSED=$(( END - START ))

    # Vérifier que le JSON de sortie existe dans BINARIES_DIR/output/
    cli_output="$BINARIES_DIR/output/${elf_name}-embeddings.json"

    if [[ $cli_status -ne 0 ]] || [[ ! -f "$cli_output" ]]; then
        log_error "Échec pour $elf_name (code=$cli_status, json_present=$([ -f "$cli_output" ] && echo oui || echo non))"
        (( FAILURES++ )) || true
        continue
    fi

    # Déplacer vers le dossier final des embeddings
    mv "$cli_output" "$expected_json"
    log_ok "$elf_name  → $(basename "$expected_json")  (${ELAPSED}s)"
    (( SUCCESS++ )) || true
done

# Nettoyage du dossier temporaire output/ dans binaries/ s'il est vide
rmdir "$BINARIES_DIR/output" 2>/dev/null || true

# ==============================================================================
# Résumé
# ==============================================================================
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

log_section "EXTRACTION TERMINÉE  (${TOTAL_ELAPSED}s)"

echo ""
printf "  %-6s %s\n" "$SUCCESS"  "JSON générés avec succès"
printf "  %-6s %s\n" "$FAILURES" "échecs"
printf "  %-6s %s\n" "$SKIPPED"  "déjà existants (ignorés)"
echo ""

if (( FAILURES > 0 )); then
    log_warn "$FAILURES fichier(s) n'ont pas pu être traités."
    log_warn "Vérifiez les logs ci-dessus et relancez le script (reprise automatique)."
fi

if (( SUCCESS > 0 )); then
    log_ok "Embeddings disponibles dans : $EMBEDDINGS_DIR"
    log_info "Prochaine étape : python 3_evaluate_similarity.py"
fi
