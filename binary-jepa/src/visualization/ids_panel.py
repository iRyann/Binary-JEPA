"""
ids_panel.py
============
Panel 4 : séquences d'IDs encodés visualisées en heatmap.

Données attendues
-----------------
    paths_encoded : list[list[int]]  — chemins encodés (IDs int) des JSONL encoded_dataset/
    vocab         : dict[str, int]   — vocabulaire {token: id}
    max_show      : int              — nombre max de chemins (défaut 12)
    max_len       : int              — longueur max d'un chemin (défaut 32)

Rendu
-----
Heatmap matplotlib.imshow :
  - axe X : position dans le chemin (0 … max_len-1)
  - axe Y : index du chemin (0 … max_show-1)
  - couleur : valeur de l'ID (colormap viridis, bornes [0, vocab_size])

La heatmap montre visuellement :
  - la répétition des patterns (bandes verticales = même token revient)
  - la longueur effective des chemins (zone gris foncé = padding ID=0)
  - la diversité sémantique (spectre de couleurs large → haute entropie)

Complété par :
  - une colorbar avec les labels des tokens les plus fréquents
  - des statistiques d'entropie de Shannon par ligne (marge droite)
  - les IDs numériques annotés dans chaque cellule si la grille est petite
    (≤ 8 chemins × 16 tokens)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import numpy as np

from .base_panel import Panel
from .theme import BORDER, PANEL, TEXT_DIM, TEXT_MAIN, style_axes

if TYPE_CHECKING:
    import matplotlib.axes

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

_ANNOTATE_THRESHOLD = (8, 16)   # (max_paths, max_len) pour annotation numérique
_CMAP = "viridis"
_ENTROPY_W = 0.06               # largeur de la barre d'entropie (normalisé)


def _shannon_entropy(ids: list[int]) -> float:
    """Entropie de Shannon d'une séquence d'IDs (bits)."""
    if not ids:
        return 0.0
    counts = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    n = len(ids)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


class IdsPanel(Panel):
    """
    Panel heatmap des IDs encodés.

    Args:
        paths_encoded: chemins encodés (int IDs).
        vocab:         {token_str: id_int}.
        max_show:      nombre max de chemins affichés.
        max_len:       longueur max d'un chemin (tronqué/paddé).
    """

    def __init__(
        self,
        paths_encoded: list[list[int]],
        vocab:         dict[str, int],
        max_show:      int = 12,
        max_len:       int = 32,
    ) -> None:
        self._paths   = paths_encoded[:max_show]
        self._vocab   = vocab
        self._max_len = max_len

        # Inversé vocab pour les labels colorbar
        self._id2tok: dict[int, str] = {v: k for k, v in vocab.items()}
        self._vocab_size = max(vocab.values()) + 1 if vocab else 1

    # ── Interface Panel ────────────────────────────────────────────────────

    @property
    def title(self) -> str:
        return "IDs ENCODÉS"

    @property
    def subtitle(self) -> str:
        total = len(self._paths)
        avg_h = np.mean([_shannon_entropy(p) for p in self._paths]) if self._paths else 0.0
        return f"{total} chemins / H̄={avg_h:.2f} bits"

    def render(self, ax: "matplotlib.axes.Axes") -> None:
        import matplotlib.colorbar as mcolorbar
        from matplotlib.colors import Normalize
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        style_axes(ax, title=self.title, subtitle=self.subtitle)

        if not self._paths:
            ax.text(0.5, 0.5, "Aucune donnée encodée",
                    ha="center", va="center", color=TEXT_DIM, fontsize=9,
                    transform=ax.transAxes)
            return

        n_paths = len(self._paths)
        pad_id  = self._vocab.get("<PAD>", 0)

        # ── Construction de la matrice ────────────────────────────────────
        matrix = np.full((n_paths, self._max_len), pad_id, dtype=np.float32)
        for i, path in enumerate(self._paths):
            row = path[:self._max_len]
            matrix[i, :len(row)] = row

        # ── Heatmap ───────────────────────────────────────────────────────
        norm = Normalize(vmin=0, vmax=self._vocab_size - 1)
        im   = ax.imshow(
            matrix,
            aspect="auto",
            cmap=_CMAP,
            norm=norm,
            interpolation="nearest",
            origin="upper",
        )

        # ── Axes ticks ────────────────────────────────────────────────────
        ax.set_xlabel("position dans le chemin", fontsize=7, color=TEXT_DIM)
        ax.set_ylabel("chemin", fontsize=7, color=TEXT_DIM)
        ax.tick_params(labelsize=6, colors=TEXT_DIM)

        # Ticks X : toutes les 4 positions
        x_ticks = list(range(0, self._max_len, max(1, self._max_len // 8)))
        ax.set_xticks(x_ticks)

        # Ticks Y : index des chemins
        ax.set_yticks(range(n_paths))
        ax.set_yticklabels([f"p{i}" for i in range(n_paths)], fontsize=6)

        # ── Annotation numérique (petites grilles seulement) ──────────────
        annotate = (
            n_paths <= _ANNOTATE_THRESHOLD[0]
            and self._max_len <= _ANNOTATE_THRESHOLD[1]
        )
        if annotate:
            for i in range(n_paths):
                for j in range(self._max_len):
                    val = int(matrix[i, j])
                    ax.text(
                        j, i, str(val),
                        ha="center", va="center",
                        fontsize=5.5,
                        fontfamily="monospace",
                        color="white" if val < self._vocab_size * 0.6 else "black",
                    )

        # ── Barre d'entropie par ligne (marge droite) ─────────────────────
        self._render_entropy_bar(ax, matrix, n_paths)

        # ── Colorbar ──────────────────────────────────────────────────────
        self._render_colorbar(ax, im, norm)

    def _render_entropy_bar(
        self,
        ax: "matplotlib.axes.Axes",
        matrix: "np.ndarray",
        n_paths: int,
    ) -> None:
        """Barres horizontales d'entropie par chemin (axes inset à droite)."""
        # Entropies par chemin (exclut le padding)
        pad_id = self._vocab.get("<PAD>", 0)
        entropies = []
        for i in range(n_paths):
            row = [int(v) for v in matrix[i] if int(v) != pad_id]
            entropies.append(_shannon_entropy(row))

        max_h = max(entropies) if entropies else 1.0

        ax_ins = ax.inset_axes([1.04, 0, 0.18, 1], transform=ax.transAxes)
        ax_ins.set_facecolor(PANEL)
        for spine in ax_ins.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(0.5)

        y_positions = range(n_paths)
        colors = [
            "#3fb950" if h >= max_h * 0.75 else
            "#ffa657" if h >= max_h * 0.40 else
            "#f78166"
            for h in entropies
        ]
        ax_ins.barh(y_positions, entropies, color=colors, height=0.7)
        ax_ins.set_xlim(0, max(max_h * 1.15, 0.1))
        ax_ins.set_ylim(-0.5, n_paths - 0.5)
        ax_ins.invert_yaxis()
        ax_ins.set_xlabel("H (bits)", fontsize=5.5, color=TEXT_DIM)
        ax_ins.tick_params(labelsize=5, colors=TEXT_DIM)
        ax_ins.set_yticks([])
        ax_ins.set_title("entropie", fontsize=5.5, color=TEXT_DIM, pad=2)

    def _render_colorbar(
        self,
        ax: "matplotlib.axes.Axes",
        im: "matplotlib.image.AxesImage",
        norm: "matplotlib.colors.Normalize",
    ) -> None:
        """Colorbar compacte avec quelques labels token importants."""
        import matplotlib.pyplot as plt

        cbar = ax.figure.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.18,
            fraction=0.04,
            aspect=40,
        )
        cbar.ax.tick_params(labelsize=6, colors=TEXT_DIM)
        cbar.set_label("ID token", fontsize=6, color=TEXT_DIM)

        # Marquer quelques tokens saillants sur la colorbar
        key_tokens = ["<PAD>", "<UNK>", "<MASK>", "VEX_REG_WRITE",
                      "VEX_REG_READ", "VEX_LOAD", "JK_BORING", "JK_RET"]
        tick_ids   = []
        tick_lbls  = []
        for tok in key_tokens:
            tid = self._vocab.get(tok)
            if tid is not None:
                tick_ids.append(tid)
                tick_lbls.append(tok.replace("VEX_", "").replace("JK_", "").replace("<", "").replace(">", ""))

        if tick_ids:
            cbar.set_ticks(tick_ids)
            cbar.set_ticklabels(tick_lbls, fontsize=5)
