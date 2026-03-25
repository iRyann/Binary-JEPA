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
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        # Toutes les positions sont en coordonnées normalisées [0,1] (transAxes),
        # taille fixe indépendamment du nombre d'instructions → pas d'entassement.
        CELL_H   = 0.032   # hauteur d'une ligne d'instruction
        BB_GAP   = 0.022   # espace inter-bloc
        BB_HDR_H = 0.034   # hauteur du header BB
        Y_MIN    = 0.015   # marge basse : on clip en dessous

        y_cursor = 0.975   # curseur descendant depuis le haut

        for bb_idx, addr in enumerate(self._order):
            insns = self._data.get(addr, [])
            color = self._colors.get(addr, BORDER)

            # Plus assez de place pour le header ?
            if y_cursor - BB_HDR_H < Y_MIN:
                remaining = len(self._order) - bb_idx
                ax.text(
                    0.5, Y_MIN,
                    f"… +{remaining} BB{'s' if remaining > 1 else ''} non affichés",
                    ha="center", va="bottom",
                    color=TEXT_DIM, fontsize=6,
                    fontfamily="monospace",
                    transform=ax.transAxes,
                )
                break

            # ── Bandeau de couleur BB (gauche) ──────────────────────────
            visible_h = min(
                BB_HDR_H + len(insns) * CELL_H,
                y_cursor - Y_MIN,
            )
            stripe = mpatches.FancyBboxPatch(
                (0.01, y_cursor - visible_h),
                _STRIPE_W,
                visible_h,
                boxstyle="round,pad=0",
                facecolor=color,
                edgecolor="none",
                transform=ax.transAxes,
                clip_on=True,
            )
            ax.add_patch(stripe)

            # ── Header BB : adresse ──────────────────────────────────────
            ax.text(
                _MARGIN_L, y_cursor - BB_HDR_H / 2,
                f"0x{addr:x}",
                color=color,
                fontsize=7.5,
                fontfamily="monospace",
                fontweight="bold",
                va="center",
                transform=ax.transAxes,
            )

            # Ligne de séparation sous le header
            ax.plot(
                [_MARGIN_L, 1 - _MARGIN_R],
                [y_cursor - BB_HDR_H, y_cursor - BB_HDR_H],
                color=_COL_SEP, linewidth=0.5,
                transform=ax.transAxes,
            )

            y_cursor -= BB_HDR_H

            # ── Instructions ─────────────────────────────────────────────
            for insn_idx, (iaddr, mne, ops) in enumerate(insns):
                if y_cursor - CELL_H < Y_MIN:
                    remaining_insns = len(insns) - insn_idx
                    ax.text(
                        _MARGIN_L + 0.05, y_cursor - CELL_H / 2,
                        f"… +{remaining_insns} instrs",
                        color=TEXT_DIM, fontsize=6,
                        fontfamily="monospace", va="center",
                        transform=ax.transAxes,
                    )
                    y_cursor -= CELL_H
                    break

                y_line = y_cursor - CELL_H / 2

                # Adresse (très discrète)
                ax.text(
                    _MARGIN_L, y_line,
                    f"{iaddr}",
                    color=_COL_ADDR,
                    fontsize=6.5,
                    fontfamily="monospace",
                    va="center",
                    transform=ax.transAxes,
                )
                # Mnémonique
                ax.text(
                    _MARGIN_L + 0.20, y_line,
                    mne,
                    color=_COL_MNE,
                    fontsize=7,
                    fontfamily="monospace",
                    va="center",
                    transform=ax.transAxes,
                )
                # Opérandes
                ax.text(
                    _MARGIN_L + 0.38, y_line,
                    ops[:28],        # tronque les opérandes trop longues
                    color=_COL_OPS,
                    fontsize=7,
                    fontfamily="monospace",
                    va="center",
                    transform=ax.transAxes,
                )

                y_cursor -= CELL_H

            y_cursor -= BB_GAP
