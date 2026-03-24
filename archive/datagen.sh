#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  datagen.sh  –  Génère un jeu de donné haché
#  Usage :
#    ./datagen.sh
#    ./datagen.sh --data_path Data/ 
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paramètres par défaut ────────────────────────────────────────────────────
DATA_PATH="data/"
binaries=($DATA_PATH*)
OUTPUT_PATH="encoded_data/"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--data_path)
		DATA_PATH="$2"
		shift 2
		;;
  --output_path)
		OUTPUT_PATH="$2"
		shift 2
		;;
	*)
		echo "[!] Argument inconnu : $1"
		exit 1
		;;
	esac
done

# ── Vérifications ────────────────────────────────────────────────────────────
if [[ ! -f "$DATA_PATH" ]]; then
	echo "[✗] Data introuvable : $DATA_PATH"
	exit 1
fi

mkdir -p encoded_data/

# ── Lancement de la génération ───────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  Démarrage de la génération"
echo "  Data : $DATA_PATH"
echo "═══════════════════════════════════════════════"
echo ""

e64ctx -E "${binaries[@]}" -O "$OUTPUT_PATH"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
	echo ""
	echo "[✗] La génération a échoué (code $EXIT_CODE)."
	exit $EXIT_CODE
fi
