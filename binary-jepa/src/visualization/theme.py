"""
theme.py
========
Palette et configuration matplotlib partagées pour tous les panels de
visualisation de pipeline.

Étend la palette GitHub dark de plot_training.py avec :
  - BB_PALETTE      : 8 couleurs distinctes pour les basic blocks
  - TOKEN_CAT_COLORS: couleur par catégorie sémantique de token
  - apply_dark_theme(): configure rcParams matplotlib une seule fois
"""

import matplotlib
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# COULEURS DE BASE (GitHub dark)
# ══════════════════════════════════════════════════════════════════════════════

BG        = "#0d1117"   # fond figure
PANEL     = "#161b22"   # fond axes
BORDER    = "#30363d"   # bordures, séparateurs
TEXT_MAIN = "#e6edf3"   # texte principal
TEXT_DIM  = "#8b949e"   # texte secondaire (adresses, labels discrets)
GRID      = "#21262d"   # grille

# Couleurs d'accent
ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_RED    = "#f78166"
ACCENT_PURPLE = "#d2a8ff"
ACCENT_ORANGE = "#ffa657"
ACCENT_CYAN   = "#79c0ff"
ACCENT_YELLOW = "#e3b341"
ACCENT_PINK   = "#f0883e"

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE BASIC BLOCKS
# 8 couleurs distinctes, lisibles sur fond sombre, cyclables pour N > 8 BBs.
# ══════════════════════════════════════════════════════════════════════════════

BB_PALETTE: list[str] = [
    "#58a6ff",  # bleu
    "#3fb950",  # vert
    "#ffa657",  # orange
    "#d2a8ff",  # violet
    "#79c0ff",  # cyan clair
    "#e3b341",  # jaune doré
    "#f78166",  # rouge saumon
    "#56d364",  # vert clair
]


def bb_color(index: int) -> str:
    """Retourne la couleur d'un basic block par son index (cyclique)."""
    return BB_PALETTE[index % len(BB_PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
# COULEURS PAR CATÉGORIE DE TOKEN
# ══════════════════════════════════════════════════════════════════════════════
# Utilisées dans PathsPanel et IdsPanel pour colorer les cellules de tokens.
# Chaque catégorie sémantique reçoit une teinte distincte, cohérente entre
# panels pour permettre la corrélation visuelle.

TOKEN_CAT_COLORS: dict[str, str] = {
    "REG":     "#58a6ff",   # bleu      — registres (REG_READ, REG_WRITE)
    "ARITH":   "#3fb950",   # vert      — opérations arithmétiques/logiques
    "MEM":     "#79c0ff",   # cyan      — mémoire (LOAD, STORE)
    "CTRL":    "#ffa657",   # orange    — contrôle de flux (EXIT_COND)
    "FLOW":    "#e3b341",   # jaune     — jump kinds (JK_*)
    "TEMP":    "#8b949e",   # gris clair— temporaires VEX (WrTmp, CONST, 64T...)
    "API":     "#d2a8ff",   # violet    — appels API externes (<API_*>)
    "SYSCALL": "#f78166",   # rouge     — syscalls (<SYSCALL_*>)
    "SPECIAL": "#30363d",   # gris foncé— tokens spéciaux (PAD, UNK, MASK, UNLIFTABLE)
}

# Couleur de fallback si catégorie inconnue
TOKEN_CAT_DEFAULT = "#30363d"

# ══════════════════════════════════════════════════════════════════════════════
# COULEURS DES ARÊTES DU CFG
# Mappées sur le jump kind terminal du basic block source.
# ══════════════════════════════════════════════════════════════════════════════

EDGE_COLORS: dict[str, str] = {
    "JK_BORING":   "#8b949e",   # gris   — flux séquentiel
    "JK_CALL":     "#58a6ff",   # bleu   — appel fonction
    "JK_RET":      "#f78166",   # rouge  — retour
    "JK_NODECODE": "#30363d",   # foncé  — non décodable
    "JK_SIGTRAP":  "#e3b341",   # jaune  — trap
    "DEFAULT":     "#30363d",
}


def edge_color(jump_kind: str) -> str:
    return EDGE_COLORS.get(jump_kind, EDGE_COLORS["DEFAULT"])


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION MATPLOTLIB
# ══════════════════════════════════════════════════════════════════════════════

def apply_dark_theme() -> None:
    """
    Applique le thème sombre à matplotlib globalement.

    À appeler une seule fois avant de créer la figure (PipelineVisualizer.render).
    Idempotente — peut être rappelée sans effet de bord.
    """
    matplotlib.rcParams.update({
        "figure.facecolor":    BG,
        "axes.facecolor":      PANEL,
        "axes.edgecolor":      BORDER,
        "axes.labelcolor":     TEXT_DIM,
        "axes.titlecolor":     TEXT_MAIN,
        "axes.grid":           False,      # désactivé par défaut dans les panels
        "grid.color":          GRID,
        "grid.linewidth":      0.5,
        "xtick.color":         TEXT_DIM,
        "ytick.color":         TEXT_DIM,
        "text.color":          TEXT_MAIN,
        "legend.facecolor":    PANEL,
        "legend.edgecolor":    BORDER,
        "legend.labelcolor":   TEXT_DIM,
        "font.family":         "monospace",
        "lines.linewidth":     1.4,
        "patch.edgecolor":     BORDER,
        "savefig.facecolor":   BG,
        "savefig.edgecolor":   BG,
    })


def style_axes(ax: "matplotlib.axes.Axes", title: str = "", subtitle: str = "") -> None:
    """
    Applique le style sombre à un axes individuel.

    Args:
        ax:       axes matplotlib à styliser.
        title:    titre principal (blanc).
        subtitle: sous-titre (gris discret), affiché à droite du titre.
    """
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

    if title:
        header = title
        if subtitle:
            header += f"  {subtitle}"
        ax.set_title(header, fontsize=8, color=TEXT_DIM, pad=5, loc="left",
                     fontfamily="monospace")

    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
