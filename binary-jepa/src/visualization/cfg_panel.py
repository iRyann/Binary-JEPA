"""
cfg_panel.py
============
Panel 2 : Control Flow Graph avec tokens VEX IR à l'intérieur de chaque nœud.

Données attendues
-----------------
    cfg_graph     : nx.DiGraph  — nœuds = adresses (int) des BBs
    bb_vex_tokens : dict[int, list[str]]  — tokens VEX par BB
    bb_colors     : dict[int, str]        — couleur par BB
    jump_kinds    : dict[tuple[int,int], str]  — jump kind par arête (optionnel)

Layout
------
Algorithme hiérarchique BFS (sans dépendance graphviz) :
  - Couche y = -distance BFS depuis la racine (flux de haut en bas)
  - Positions x = réparties uniformément dans la couche
  - En cas de cycle (boucles), les back-edges sont gérées par BFS
    (elles apparaissent comme arêtes "remontantes")

Nœuds
-----
Chaque nœud est un FancyBboxPatch :
  ┌── 0x402000 ──────────┐
  │ REG_R  OP_SUB  REG_W │   ← tokens VEX (max MAX_TOKENS_PER_NODE)
  │ LOAD   EXIT_C  BORE  │
  └──────────────────────┘
Les tokens dépassant MAX_TOKENS_PER_NODE sont remplacés par "…+N".

Arêtes
------
Flèches colorées selon le jump kind terminal du nœud source :
  - JK_BORING   → gris
  - JK_CALL     → bleu
  - JK_RET      → rouge
  - VEX_EXIT_COND (branche conditionnelle) → orange
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from .base_panel import Panel
from .theme import (
    BORDER, PANEL, TEXT_DIM, TEXT_MAIN,
    edge_color, style_axes,
)
from .token_colors import display_name, hex_color

if TYPE_CHECKING:
    import matplotlib.axes

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE RENDU
# ══════════════════════════════════════════════════════════════════════════════

MAX_TOKENS_PER_NODE = 8   # tokens affichés par nœud (le reste → "…+N")
TOKENS_PER_ROW      = 3   # tokens par ligne dans le nœud
NODE_PAD            = 0.04  # padding interne du nœud (data coords normalisés)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT HIÉRARCHIQUE (BFS, sans graphviz)
# ══════════════════════════════════════════════════════════════════════════════

def _hierarchical_layout(
    g: nx.DiGraph,
    root: int,
) -> dict[int, tuple[float, float]]:
    """
    Calcule des positions (x, y) pour un DiGraph orienté (CFG).

    Algorithme :
      1. BFS depuis root pour assigner une couche à chaque nœud.
         Les nœuds non atteignables depuis root reçoivent une couche
         fictive (max_layer + 1) pour rester visibles.
      2. Tri topologique à l'intérieur de chaque couche pour réduire
         les croisements d'arêtes (heuristique simple).
      3. y = -layer (haut = racine), x = position centrée dans la couche.

    Args:
        g:    graphe orienté (peut contenir des cycles).
        root: nœud d'entrée de la fonction.

    Returns:
        dict {node: (x, y)} avec x,y dans [−0.5·W, 0.5·W] × [−layers, 0].
    """
    if not g.nodes:
        return {}

    # ── 1. Couches BFS ───────────────────────────────────────────────────────
    try:
        layers: dict[int, int] = nx.single_source_shortest_path_length(g, root)
    except nx.NodeNotFound:
        layers = {n: 0 for n in g.nodes}

    # Nœuds non atteignables depuis root
    max_layer = max(layers.values(), default=0)
    for node in g.nodes:
        if node not in layers:
            layers[node] = max_layer + 1

    # ── 2. Groupement par couche ─────────────────────────────────────────────
    by_layer: dict[int, list[int]] = {}
    for node, layer in layers.items():
        by_layer.setdefault(layer, []).append(node)

    # Tri dans chaque couche (adresse croissante → déterministe)
    for layer in by_layer:
        by_layer[layer].sort()

    # ── 3. Assignation des positions ─────────────────────────────────────────
    pos: dict[int, tuple[float, float]] = {}
    for layer, nodes in by_layer.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2.0)   # centré sur 0
            y = -float(layer)
            pos[node] = (x, y)

    return pos


# ══════════════════════════════════════════════════════════════════════════════
# PANEL
# ══════════════════════════════════════════════════════════════════════════════

class CfgPanel(Panel):
    """
    Panel CFG avec tokens VEX IR dans chaque nœud.

    Args:
        cfg_graph:     DiGraph dont les nœuds sont des adresses int (basic blocks).
        bb_vex_tokens: {addr: [token, ...]} — tokens VEX par BB.
        bb_colors:     {addr: hex_color}.
        root_addr:     adresse d'entrée de la fonction (racine du layout).
        jump_kinds:    {(src_addr, dst_addr): jump_kind_str} (optionnel).
    """

    def __init__(
        self,
        cfg_graph:     nx.DiGraph,
        bb_vex_tokens: dict[int, list[str]],
        bb_colors:     dict[int, str],
        root_addr:     int,
        jump_kinds:    dict[tuple[int, int], str] | None = None,
    ) -> None:
        self._graph      = cfg_graph
        self._tokens     = bb_vex_tokens
        self._colors     = bb_colors
        self._root       = root_addr
        self._jk         = jump_kinds or {}

    # ── Interface Panel ───────────────────────────────────────────────────

    @property
    def title(self) -> str:
        return "CFG + VEX IR"

    @property
    def subtitle(self) -> str:
        n_nodes = self._graph.number_of_nodes()
        n_edges = self._graph.number_of_edges()
        return f"{n_nodes} BBs / {n_edges} arêtes"

    def render(self, ax: "matplotlib.axes.Axes") -> None:
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch

        style_axes(ax, title=self.title, subtitle=self.subtitle)
        ax.set_aspect("equal")
        ax.set_axis_off()

        if self._graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "CFG vide", ha="center", va="center",
                    color=TEXT_DIM, fontsize=9, transform=ax.transAxes)
            return

        # ── Layout ───────────────────────────────────────────────────────
        raw_pos = _hierarchical_layout(self._graph, self._root)

        # Normalisation dans [0,1]² avec marges
        xs = [p[0] for p in raw_pos.values()]
        ys = [p[1] for p in raw_pos.values()]
        x_range = max(max(xs) - min(xs), 1e-6)
        y_range = max(max(ys) - min(ys), 1e-6)

        margin = 0.12
        pos: dict[int, tuple[float, float]] = {
            node: (
                margin + (x - min(xs)) / x_range * (1 - 2 * margin),
                margin + (y - min(ys)) / y_range * (1 - 2 * margin),
            )
            for node, (x, y) in raw_pos.items()
        }

        # ── Dimensions des nœuds (proportionnelles au nombre de tokens) ──
        node_w = 0.28
        node_h_base = 0.08    # header seul
        row_h = 0.038          # hauteur d'une ligne de tokens

        def _node_height(addr: int) -> float:
            toks = self._tokens.get(addr, [])
            displayed = min(len(toks), MAX_TOKENS_PER_NODE)
            n_rows = max(1, (displayed + TOKENS_PER_ROW - 1) // TOKENS_PER_ROW)
            return node_h_base + n_rows * row_h

        # ── Dessin des nœuds ─────────────────────────────────────────────
        for addr, (cx, cy) in pos.items():
            color = self._colors.get(addr, BORDER)
            nh    = _node_height(addr)
            nx_   = cx - node_w / 2
            ny_   = cy - nh / 2

            # Fond du nœud
            box = mpatches.FancyBboxPatch(
                (nx_, ny_), node_w, nh,
                boxstyle="round,pad=0.01",
                facecolor=PANEL,
                edgecolor=color,
                linewidth=1.2,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.add_patch(box)

            # Header : adresse
            ax.text(
                cx, cy + nh / 2 - node_h_base / 2,
                f"0x{addr:x}",
                ha="center", va="center",
                color=color,
                fontsize=6.5,
                fontfamily="monospace",
                fontweight="bold",
                transform=ax.transAxes,
            )

            # Séparateur header / tokens
            ax.plot(
                [nx_ + 0.01, nx_ + node_w - 0.01],
                [cy + nh / 2 - node_h_base, cy + nh / 2 - node_h_base],
                color=color, linewidth=0.5, alpha=0.5,
                transform=ax.transAxes,
            )

            # Tokens VEX (grille)
            toks = self._tokens.get(addr, [])
            shown = toks[:MAX_TOKENS_PER_NODE]
            overflow = len(toks) - len(shown)
            if overflow > 0:
                shown = shown[:-1] + [f"…+{overflow + 1}"]

            tok_area_top = cy + nh / 2 - node_h_base - row_h * 0.3
            for i, tok in enumerate(shown):
                row = i // TOKENS_PER_ROW
                col = i % TOKENS_PER_ROW
                tx = nx_ + 0.01 + col * (node_w - 0.02) / TOKENS_PER_ROW
                ty = tok_area_top - row * row_h

                tok_color = hex_color(tok) if not tok.startswith("…") else TEXT_DIM
                ax.text(
                    tx + (node_w - 0.02) / (2 * TOKENS_PER_ROW), ty,
                    display_name(tok) if not tok.startswith("…") else tok,
                    ha="center", va="top",
                    color=tok_color,
                    fontsize=5.5,
                    fontfamily="monospace",
                    transform=ax.transAxes,
                )

        # ── Dessin des arêtes ─────────────────────────────────────────────
        for src, dst in self._graph.edges():
            if src not in pos or dst not in pos:
                continue

            jk = self._jk.get((src, dst), "JK_BORING")
            ecolor = edge_color(jk)

            src_cx, src_cy = pos[src]
            dst_cx, dst_cy = pos[dst]
            src_nh = _node_height(src)
            dst_nh = _node_height(dst)

            # Points de départ/arrivée : bord bas du nœud source, bord haut du nœud dest
            x0, y0 = src_cx, src_cy - src_nh / 2
            x1, y1 = dst_cx, dst_cy + dst_nh / 2

            # Arête simple ou courbée (back-edge si y0 < y1)
            if y1 >= y0 - 0.01:
                # Back-edge (cycle) : arc courbé à droite
                connectionstyle = "arc3,rad=0.4"
            else:
                connectionstyle = "arc3,rad=0.0"

            ax.annotate(
                "",
                xy=(x1, y1), xycoords=ax.transAxes,
                xytext=(x0, y0), textcoords=ax.transAxes,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ecolor,
                    lw=0.9,
                    connectionstyle=connectionstyle,
                ),
            )
