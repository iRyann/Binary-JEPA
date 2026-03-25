"""
asm_panel.py
============
Panel 1 : dump assembleur x86/x64, groupé par basic block.

Données attendues
-----------------
    bb_asm_data  : dict[int, list[tuple[str, str, str]]]
                   {addr_bb: [(addr_hex, mnemonic, op_str), ...]}
    bb_order     : list[int]   — adresses des BBs dans l'ordre topologique
    bb_colors    : dict[int, str] — {addr_bb: hex_color}

Rendu
-----
Pour chaque basic block, dans l'ordre topologique :
  ┌── 0x402000 ──────────────────────────┐  ← header coloré (bandeau gauche)
  │  push   rbp                          │
  │  mov    rbp, rsp                     │
  │  sub    rsp, 0x28                    │
  └──────────────────────────────────────┘
  (espace inter-bloc)
  ┌── 0x402020 ───...

L'axe est en coordonnées de texte (ligne par ligne), axes.transData non utilisé.
"""

from __future__ import annotations

import matplotlib.patches as mpatches

from .base_panel import Panel
from .theme import (
    BORDER, PANEL, TEXT_DIM, TEXT_MAIN,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE,
    style_axes,
)


# Largeur du bandeau de couleur BB (en fraction de la largeur normalisée)
_STRIPE_W = 0.018
_CELL_H   = 0.048   # hauteur d'une ligne de texte (coords normalisées [0,1])
_BB_GAP   = 0.030   # espace entre basic blocks
_BB_HDR_H = 0.040   # hauteur du header de BB
_MARGIN_L = 0.08    # marge gauche (pour le bandeau + adresse)
_MARGIN_R = 0.02

# Couleurs syntaxiques du dump ASM
_COL_ADDR  = "#6e7681"   # adresse (très discret)
_COL_MNE   = "#79c0ff"   # mnémonique  (cyan)
_COL_OPS   = "#e6edf3"   # opérandes   (blanc)
_COL_SEP   = "#30363d"   # séparateur  (gris foncé)


class AsmPanel(Panel):
    """
    Panel assembleur.

    Args:
        bb_asm_data: {addr_bb: [(addr_hex, mnemonic, op_str), ...]}
        bb_order:    liste ordonnée des adresses de BBs (ordre topologique).
        bb_colors:   {addr_bb: hex_color}
    """

    def __init__(
        self,
        bb_asm_data: dict[int, list[tuple[str, str, str]]],
        bb_order:    list[int],
        bb_colors:   dict[int, str],
    ) -> None:
        self._data   = bb_asm_data
        self._order  = bb_order
        self._colors = bb_colors

    # ── Interface Panel ────────────────────────────────────────────────────

    @property
    def title(self) -> str:
        return "ASSEMBLY"

    @property
    def subtitle(self) -> str:
        total = sum(len(insns) for insns in self._data.values())
        return f"{total} instrs / {len(self._data)} BBs"

    def render(self, ax: "matplotlib.axes.Axes") -> None:
        style_axes(ax, title=self.title, subtitle=self.subtitle)
        ax.set_xlim(0, 1)
        ax.set_axis_off()

        # Calcul hauteur totale nécessaire
        total_lines = sum(len(insns) for insns in self._data.values())
        total_h = (
            len(self._order) * (_BB_HDR_H + _BB_GAP)
            + total_lines * _CELL_H
        )
        ax.set_ylim(0, max(total_h, 1.0))

        # Rendu de bas en haut (y croît vers le haut en matplotlib)
        # On calcule d'abord en "y descendant" puis on inverse en fin de boucle
        y_top = max(total_h, 1.0) - 0.01  # curseur, part du haut

        for addr in self._order:
            insns = self._data.get(addr, [])
            color = self._colors.get(addr, BORDER)
            block_h = _BB_HDR_H + len(insns) * _CELL_H

            # ── Bandeau de couleur BB (gauche) ──────────────────────────
            stripe = mpatches.FancyBboxPatch(
                (0.01, y_top - block_h),
                _STRIPE_W,
                block_h,
                boxstyle="round,pad=0",
                facecolor=color,
                edgecolor="none",
                transform=ax.transData,
                clip_on=False,
            )
            ax.add_patch(stripe)

            # ── Header BB : adresse ──────────────────────────────────────
            ax.text(
                _MARGIN_L, y_top - _BB_HDR_H / 2,
                f"0x{addr:x}",
                color=color,
                fontsize=7.5,
                fontfamily="monospace",
                fontweight="bold",
                va="center",
                transform=ax.transData,
            )

            # Ligne de séparation sous le header
            ax.axhline(
                y_top - _BB_HDR_H,
                xmin=_MARGIN_L, xmax=1 - _MARGIN_R,
                color=_COL_SEP,
                linewidth=0.5,
            )

            y_cursor = y_top - _BB_HDR_H

            # ── Instructions ─────────────────────────────────────────────
            for (iaddr, mne, ops) in insns:
                y_line = y_cursor - _CELL_H / 2

                # Adresse (très discrète)
                ax.text(
                    _MARGIN_L, y_line,
                    f"{iaddr}",
                    color=_COL_ADDR,
                    fontsize=6.5,
                    fontfamily="monospace",
                    va="center",
                    transform=ax.transData,
                )
                # Mnémonique
                ax.text(
                    _MARGIN_L + 0.20, y_line,
                    mne,
                    color=_COL_MNE,
                    fontsize=7,
                    fontfamily="monospace",
                    va="center",
                    transform=ax.transData,
                )
                # Opérandes
                ax.text(
                    _MARGIN_L + 0.38, y_line,
                    ops[:28],        # tronque les opérandes trop longues
                    color=_COL_OPS,
                    fontsize=7,
                    fontfamily="monospace",
                    va="center",
                    transform=ax.transData,
                )

                y_cursor -= _CELL_H

            y_top -= block_h + _BB_GAP
