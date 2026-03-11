#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  train.sh  –  Lance l'entraînement puis génère le rapport graphique
#  Usage :
#    ./train.sh                          # utilise train.yaml par défaut
#    ./train.sh --config_path my.yaml
#    ./train.sh --config_path my.yaml --smoothing 0.9
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paramètres par défaut ────────────────────────────────────────────────────
CONFIG_PATH="train.yaml"
SMOOTHING="0.85"
PLOT_SCRIPT="src/utils/plot_training.py"

# ── Parsing des arguments (pass-through vers Python + options locales) ───────
while [[ $# -gt 0 ]]; do
  case "$1" in
  --config_path)
    CONFIG_PATH="$2"
    shift 2
    ;;
  --smoothing)
    SMOOTHING="$2"
    shift 2
    ;;
  *)
    echo "[!] Argument inconnu : $1"
    exit 1
    ;;
  esac
done

# ── Vérifications ────────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[✗] Config introuvable : $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "[✗] Script de plot introuvable : $PLOT_SCRIPT"
  exit 1
fi

mkdir -p training/logs training/plots

# ── Lancement de l'entraînement ───────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  Démarrage de l'entraînement"
echo "  Config  : $CONFIG_PATH"
echo "═══════════════════════════════════════════════"
echo ""

python -m src.train.default --config_path "$CONFIG_PATH"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo ""
  echo "[✗] L'entraînement a échoué (code $EXIT_CODE). Rapport non généré."
  exit $EXIT_CODE
fi

# ── Récupération du log CSV le plus récent ────────────────────────────────────
LATEST_LOG=$(ls -t training/logs/*.csv 2>/dev/null | head -n 1)

if [[ -z "$LATEST_LOG" ]]; then
  echo "[!] Aucun log CSV trouvé dans training/logs/ — rapport non généré."
  exit 0
fi

# ── Génération du rapport ─────────────────────────────────────────────────────
# Même stem que le log, dans training/plots/
STEM=$(basename "$LATEST_LOG" .csv)
OUT_PLOT="training/plots/${STEM}.png"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Génération du rapport"
echo "  Log     : $LATEST_LOG"
echo "  Output  : $OUT_PLOT"
echo "═══════════════════════════════════════════════"
echo ""

python "$PLOT_SCRIPT" "$LATEST_LOG" --out "$OUT_PLOT" --smoothing "$SMOOTHING"

echo ""
echo "  Rapport disponible → $OUT_PLOT"
echo ""
