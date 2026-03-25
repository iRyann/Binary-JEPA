"""
paths_panel.py
==============
Panel 3 : chemins d'exécution DFS sous forme de grille de cellules colorées.

Données attendues
-----------------
    paths    : list[list[str]]  — chemins (tokens string) issus des JSONL data/
    max_show : int              — nombre max de chemins à afficher (défaut 12)
    max_len  : int              — longueur max d'un chemin affiché (défaut 32)

Rendu
-----
Grille (path_idx × token_position) :

  path 0  ║ REG_R ║ SUB   ║ LOAD  ║ EXIT  ║ BORE  ║
  path 1  ║ REG_R ║ SUB   ║ LOAD  ║ EXIT  ║ RET   ║
  ...

  Chaque cellule :
    - fond : couleur de la TokenCategory du token
    - texte : display_name(token) — abréviation ≤ 7 chars
    - cellule vide (padding) si le chemin est plus court que max_len

  Légende des catégories affichée en bas de l'axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import numpy as np

from .base_panel import Panel
from .theme import BORDER, PANEL, TEXT_DIM, TEXT_MAIN, style_axes
from .token_colors import CATEGORY_LEGEND, categorize, display_name, hex_color

if TYPE_CHECKING:
    import matplotlib.axes

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

_LABEL_W  = 0.10     # largeur colonne labels "path N" (coords normalisées)
_LEGEND_H = 0.06     # hauteur zone légende en bas
_CELL_PAD = 0.004    # espace entre cellules
_FONT_TOK = 5.5      # taille police dans les cellules de tokens
_FONT_LBL = 6.5      # taille police labels de lignes
_ALPHA_CELL = 0.85   # transparence des cellules


class PathsPanel(Panel):
    """
    Panel grille de chemins DFS (tokens string).

    Args:
        paths:    liste de chemins — chaque chemin est une list[str] de tokens.
        max_show: nombre maximum de chemins à afficher.
        max_len:  longueur maximum d'un chemin (tronqué si dépassé).
    """

    def __init__(
        self,
        paths:    list[list[str]],
        max_show: int = 12,
        max_len:  int = 32,
    ) -> None:
        self._paths    = paths[:max_show]
        self._max_show = max_show
        self._max_len  = max_len

    # ── Interface Panel ────────────────────────────────────────────────────

    @property
    def title(self) -> str:
        return "CHEMINS DFS"

    @property
    def subtitle(self) -> str:
        total = len(self._paths)
        avg   = np.mean([len(p) for p in self._paths]) if self._paths else 0
        return f"{total} chemins / moy. {avg:.1f} tokens"

    def render(self, ax: "matplotlib.axes.Axes") -> None:
        style_axes(ax, title=self.title, subtitle=self.subtitle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        if not self._paths:
            ax.text(0.5, 0.5, "Aucun chemin chargé",
                    ha="center", va="center", color=TEXT_DIM, fontsize=9,
                    transform=ax.transAxes)
            return

        n_paths = len(self._paths)

        # Zone disponible pour la grille (hors légende)
        grid_top    = 0.97
        grid_bottom = _LEGEND_H + 0.02
        grid_left   = _LABEL_W
        grid_right  = 0.99

        cell_h = (grid_top - grid_bottom) / n_paths
        cell_w = (grid_right - grid_left) / self._max_len

        for path_idx, path in enumerate(self._paths):
            y_center = grid_top - (path_idx + 0.5) * cell_h

            # Label ligne
            ax.text(
                _LABEL_W - 0.01, y_center,
                f"p{path_idx}",
                ha="right", va="center",
                color=TEXT_DIM,
                fontsize=_FONT_LBL,
                fontfamily="monospace",
                transform=ax.transAxes,
            )

            displayed_toks = path[:self._max_len]

            for tok_idx, token in enumerate(displayed_toks):
                x_left   = grid_left + tok_idx * cell_w
                y_bottom = y_center - cell_h / 2

                color = hex_color(token)

                # Fond coloré
                rect = mpatches.FancyBboxPatch(
                    (x_left + _CELL_PAD, y_bottom + _CELL_PAD),
                    cell_w - 2 * _CELL_PAD,
                    cell_h - 2 * _CELL_PAD,
                    boxstyle="round,pad=0.002",
                    facecolor=color,
                    edgecolor="none",
                    alpha=_ALPHA_CELL,
                    transform=ax.transAxes,
                    clip_on=True,
                )
                ax.add_patch(rect)


            # Cellules vides pour les chemins plus courts que max_len
            for tok_idx in range(len(displayed_toks), self._max_len):
                x_left   = grid_left + tok_idx * cell_w
                y_bottom = y_center - cell_h / 2
                rect = mpatches.FancyBboxPatch(
                    (x_left + _CELL_PAD, y_bottom + _CELL_PAD),
                    cell_w - 2 * _CELL_PAD,
                    cell_h - 2 * _CELL_PAD,
                    boxstyle="round,pad=0.002",
                    facecolor=BORDER,
                    edgecolor="none",
                    alpha=0.3,
                    transform=ax.transAxes,
                    clip_on=True,
                )
                ax.add_patch(rect)

        # ── Légende catégories ────────────────────────────────────────────
        self._render_legend(ax)

    def _render_legend(self, ax: "matplotlib.axes.Axes") -> None:
        """Bande de légende des catégories en bas de l'axes."""
        n = len(CATEGORY_LEGEND)
        legend_y = _LEGEND_H / 2
        swatch_w = 0.009
        swatch_h = 0.022

        for i, (cat_name, cat_color) in enumerate(CATEGORY_LEGEND):
            x = _LABEL_W + i * (1 - _LABEL_W) / n

            # Carré de couleur
            rect = mpatches.FancyBboxPatch(
                (x, legend_y - swatch_h / 2),
                swatch_w, swatch_h,
                boxstyle="round,pad=0.001",
                facecolor=cat_color,
                edgecolor="none",
                alpha=0.9,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)

            # Label catégorie
            ax.text(
                x + swatch_w + 0.004, legend_y,
                cat_name,
                ha="left", va="center",
                color=TEXT_DIM,
                fontsize=5.5,
                fontfamily="monospace",
                transform=ax.transAxes,
            )
