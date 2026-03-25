"""
pipeline_viz.py
===============
Orchestrateur de la visualisation multi-stades de la pipeline binaire.

Flux de données
---------------
    ELF binaire ──(angr)──► ASM + CFG + VEX tokens
                                        │
    data/*.jsonl ───────────────────────┤──► PathsPanel   (tokens string)
    encoded_dataset/*.jsonl ────────────┘──► IdsPanel     (IDs int)
    vocab.json ──────────────────────────────► IdsPanel   (légende)

Tous les chargements sont centralisés ici — les panels reçoivent des données
Python pures et n'ont pas de dépendance directe sur angr.

Extensibilité backend
---------------------
Pour un backend HTML futur :
  - Remplacer _compose_matplotlib() par _compose_html()
  - Les panels gardent la même structure de données
  - Un paramètre backend="html" dans render() suffit

Usage CLI
---------
    python -m binary_jepa.visualization.pipeline_viz \\
        /path/to/binary.elf 0x402000 \\
        --jsonl-dir    binary-jepa/data/ \\
        --encoded-dir  binary-jepa/encoded_dataset/ \\
        --vocab        binary-jepa/vocab.json \\
        --out          output.png \\
        --max-paths    12 \\
        --max-len      32

    # Analyse offline (sans angr, panneaux ASM+CFG omis) :
    python -m ... binary.elf 0x402000 --no-angr --jsonl-dir data/ ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# Silence angr avant tout import (peut être importé comme sous-module)
logging.getLogger("angr").setLevel(logging.ERROR)
logging.getLogger("cle").setLevel(logging.ERROR)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsonl_for_func(
    jsonl_dir:      Path,
    binary_name:    str,
    func_addr_hex:  str,
    max_paths:      int,
    max_len:        int,
) -> list[list[str]]:
    """
    Charge les chemins (tokens string) d'une fonction depuis les shards JSONL.

    Args:
        jsonl_dir:     dossier contenant les *.jsonl.
        binary_name:   nom du binaire tel qu'indexé dans le JSONL (ex: "b2sum_gcc_O0.elf").
        func_addr_hex: adresse hex de la fonction (ex: "0x402000").
        max_paths:     nombre maximum de chemins à charger.
        max_len:       tronquer chaque chemin à max_len tokens.

    Returns:
        Liste de chemins, chaque chemin étant une list[str].
    """
    stem = Path(binary_name).stem
    candidates = list(jsonl_dir.glob(f"{stem}.jsonl"))
    if not candidates:
        logger.warning("Aucun JSONL trouvé pour '%s' dans %s", stem, jsonl_dir)
        return []

    paths: list[list[str]] = []
    with candidates[0].open("r", encoding="utf-8") as f:
        for line in f:
            if len(paths) >= max_paths:
                break
            try:
                rec = json.loads(line)
                if rec.get("func_addr") == func_addr_hex:
                    tokens = rec.get("tokens", [])[:max_len]
                    paths.append(tokens)
            except json.JSONDecodeError:
                continue

    return paths


def _load_encoded_for_func(
    encoded_dir:   Path,
    binary_name:   str,
    func_addr_hex: str,
    max_paths:     int,
    max_len:       int,
) -> list[list[int]]:
    """Même logique que _load_jsonl_for_func mais pour les IDs int."""
    stem = Path(binary_name).stem
    candidates = list(encoded_dir.glob(f"{stem}.jsonl"))
    if not candidates:
        logger.warning("Aucun JSONL encodé pour '%s' dans %s", stem, encoded_dir)
        return []

    paths: list[list[int]] = []
    with candidates[0].open("r", encoding="utf-8") as f:
        for line in f:
            if len(paths) >= max_paths:
                break
            try:
                rec = json.loads(line)
                if rec.get("func_addr") == func_addr_hex:
                    tokens = rec.get("tokens", [])[:max_len]
                    paths.append([int(t) for t in tokens])
            except json.JSONDecodeError:
                continue

    return paths


def _assign_bb_colors(bb_addrs: list[int]) -> dict[int, str]:
    """Assigne une couleur de la BB_PALETTE à chaque basic block."""
    from .theme import bb_color
    return {addr: bb_color(i) for i, addr in enumerate(bb_addrs)}


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION ANGR (ASM + CFG + VEX)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_angr_data(
    elf_path:  Path,
    func_addr: int,
) -> tuple[
    dict[int, list[tuple[str, str, str]]],   # bb_asm_data
    dict[int, list[str]],                     # bb_vex_tokens
    nx.DiGraph,                               # cfg_graph
    list[int],                                # bb_order (topologique)
    dict[tuple[int, int], str],              # jump_kinds
]:
    """
    Charge le binaire avec angr et extrait pour la fonction donnée :
      - bb_asm_data    : instructions capstone par BB
      - bb_vex_tokens  : tokens VEX IR par BB
      - cfg_graph      : DiGraph intra-fonction
      - bb_order       : ordre topologique (ou BFS depuis root si cycles)
      - jump_kinds     : jump kind par arête (src_addr, dst_addr)

    Raises:
        ImportError  si angr n'est pas installé.
        ValueError   si func_addr non trouvé dans le binaire.
    """
    # Import local pour ne pas bloquer les usages offline
    try:
        import angr
    except ImportError as e:
        raise ImportError(
            "angr requis pour les panels ASM et CFG. "
            "Installer avec : pip install angr\n"
            "Ou utiliser --no-angr pour les panels JSONL uniquement."
        ) from e

    # Import du BinaryAnalyzer (même repo, chemin relatif)
    _elf_src = Path(__file__).parents[3] / "elf-processing" / "src"
    if str(_elf_src) not in sys.path:
        sys.path.insert(0, str(_elf_src))

    from elf_processing_core import BinaryAnalyzer

    logger.info("Chargement angr : %s", elf_path.name)

    with BinaryAnalyzer(elf_path) as analyzer:
        proj = analyzer.proj
        cfg  = analyzer.cfg

        func = cfg.functions.get(func_addr)
        if func is None:
            raise ValueError(
                f"Fonction 0x{func_addr:x} non trouvée dans {elf_path.name}. "
                f"Fonctions disponibles : "
                + ", ".join(f"0x{a:x}" for a in list(cfg.functions)[:10])
            )

        blocks   = list(func.blocks)
        g, addr_to_node = analyzer._build_function_graph(func, blocks)

        # ── Ordre topologique (BFS depuis func_addr si cycles) ──────────
        entry = func_addr if func_addr in g else (blocks[0].addr if blocks else func_addr)
        try:
            bb_order = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            # Cycles détectés (boucles) → BFS depuis l'entrée
            bb_order = list(nx.bfs_tree(g, entry).nodes())
            for n in g.nodes():
                if n not in bb_order:
                    bb_order.append(n)

        # ── ASM par BB ───────────────────────────────────────────────────
        bb_asm_data: dict[int, list[tuple[str, str, str]]] = {}
        for addr in bb_order:
            try:
                cs_block = proj.factory.block(addr).capstone
                insns = [
                    (f"0x{i.address:x}", i.mnemonic, i.op_str)
                    for i in cs_block.insns
                ]
            except Exception:
                insns = [(f"0x{addr:x}", "<err>", "")]
            bb_asm_data[addr] = insns

        # ── VEX tokens par BB ────────────────────────────────────────────
        bb_vex_tokens: dict[int, list[str]] = {}
        block_cache: dict[int, list[str]] = {}
        for addr in bb_order:
            if addr not in block_cache:
                block_cache[addr] = analyzer._tokenize_block(
                    addr, addr_to_node.get(addr)
                )
            bb_vex_tokens[addr] = block_cache[addr]

        # ── Jump kinds par arête ─────────────────────────────────────────
        jump_kinds: dict[tuple[int, int], str] = {}
        for src_addr in bb_order:
            toks = bb_vex_tokens.get(src_addr, [])
            # Le dernier token est le jump kind ou l'API token
            jk = toks[-1] if toks else "JK_BORING"
            for dst_addr in g.successors(src_addr):
                jump_kinds[(src_addr, dst_addr)] = jk

        return bb_asm_data, bb_vex_tokens, g, bb_order, jump_kinds


# ══════════════════════════════════════════════════════════════════════════════
# STATISTIQUES (bandeau header de la figure)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_stats(
    bb_asm_data:    dict[int, list[tuple[str, str, str]]] | None,
    bb_vex_tokens:  dict[int, list[str]] | None,
    paths_raw:      list[list[str]],
    paths_encoded:  list[list[int]],
    vocab:          dict[str, int] | None,
) -> dict[str, str]:
    """Calcule les statistiques de compression inter-stades."""
    import math

    stats: dict[str, str] = {}

    # Stade 1 : assembleur
    if bb_asm_data:
        n_insns = sum(len(v) for v in bb_asm_data.values())
        n_bbs   = len(bb_asm_data)
        stats["ASM instrs"] = str(n_insns)
        stats["BBs"]        = str(n_bbs)

    # Stade 2 : VEX tokens par BB
    if bb_vex_tokens:
        total_vex = sum(len(v) for v in bb_vex_tokens.values())
        stats["VEX tokens/BBs"] = str(total_vex)
        if bb_asm_data:
            n_insns = sum(len(v) for v in bb_asm_data.values())
            ratio = total_vex / n_insns if n_insns else 0
            stats["ratio ASM→VEX"] = f"{ratio:.2f}x"

    # Stade 3 : chemins DFS
    if paths_raw:
        n_paths = len(paths_raw)
        avg_len = sum(len(p) for p in paths_raw) / n_paths
        stats["chemins DFS"]  = str(n_paths)
        stats["moy. len/chemin"] = f"{avg_len:.1f}"

    # Stade 4 : encodage + entropie
    if paths_encoded:
        from .ids_panel import _shannon_entropy
        entropies = [_shannon_entropy(p) for p in paths_encoded]
        avg_h = sum(entropies) / len(entropies)
        stats["H̄ Shannon"] = f"{avg_h:.2f} bits"

    if vocab:
        stats["vocab size"] = str(len(vocab))

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR
# ══════════════════════════════════════════════════════════════════════════════

class PipelineVisualizer:
    """
    Compose les 4 panels de visualisation en une figure matplotlib unique.

    Args:
        elf_path:     chemin vers le binaire ELF.
        func_addr:    adresse int de la fonction cible.
        jsonl_dir:    dossier data/ contenant les JSONL bruts.
        encoded_dir:  dossier encoded_dataset/.
        vocab_path:   chemin vers vocab.json.
        no_angr:      si True, saute les panels ASM et CFG.
    """

    def __init__(
        self,
        elf_path:    str | Path,
        func_addr:   int,
        jsonl_dir:   str | Path | None = None,
        encoded_dir: str | Path | None = None,
        vocab_path:  str | Path | None = None,
        no_angr:     bool = False,
    ) -> None:
        self.elf_path    = Path(elf_path)
        self.func_addr   = func_addr
        self.jsonl_dir   = Path(jsonl_dir)   if jsonl_dir   else None
        self.encoded_dir = Path(encoded_dir) if encoded_dir else None
        self.vocab_path  = Path(vocab_path)  if vocab_path  else None
        self.no_angr     = no_angr

    # ── Point d'entrée public ──────────────────────────────────────────────

    def render(
        self,
        out_path:  str | Path = "pipeline_stages.png",
        max_paths: int = 12,
        max_len:   int = 32,
        dpi:       int = 150,
    ) -> Path:
        """
        Génère la figure multi-panels et la sauvegarde.

        Args:
            out_path:  chemin de sortie (PNG ou PDF selon l'extension).
            max_paths: nombre max de chemins affichés dans les panels 3/4.
            max_len:   longueur max d'un chemin dans les panels 3/4.
            dpi:       résolution de sortie.

        Returns:
            Chemin absolu du fichier généré.
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        from .theme import (
            BG, BORDER, PANEL, TEXT_DIM, TEXT_MAIN,
            apply_dark_theme,
        )

        apply_dark_theme()
        out_path = Path(out_path)

        # ── Chargement des données ─────────────────────────────────────
        func_addr_hex = hex(self.func_addr)
        binary_name   = self.elf_path.name

        # Vocab
        vocab: dict[str, int] | None = None
        if self.vocab_path and self.vocab_path.exists():
            with self.vocab_path.open("r") as f:
                vocab = json.load(f)

        # JSONL bruts (Panel 3)
        paths_raw: list[list[str]] = []
        if self.jsonl_dir and self.jsonl_dir.exists():
            paths_raw = _load_jsonl_for_func(
                self.jsonl_dir, binary_name, func_addr_hex, max_paths, max_len
            )
            logger.info("%d chemins chargés depuis %s", len(paths_raw), self.jsonl_dir)

        # JSONL encodés (Panel 4)
        paths_encoded: list[list[int]] = []
        if self.encoded_dir and self.encoded_dir.exists():
            paths_encoded = _load_encoded_for_func(
                self.encoded_dir, binary_name, func_addr_hex, max_paths, max_len
            )

        # Angr (Panels 1 + 2)
        bb_asm_data:   dict[int, list[tuple[str, str, str]]] | None = None
        bb_vex_tokens: dict[int, list[str]] | None = None
        cfg_graph:     nx.DiGraph | None = None
        bb_order:      list[int] | None  = None
        jump_kinds:    dict | None        = None
        bb_colors:     dict[int, str]     = {}

        if not self.no_angr:
            try:
                bb_asm_data, bb_vex_tokens, cfg_graph, bb_order, jump_kinds = \
                    _extract_angr_data(self.elf_path, self.func_addr)
                bb_colors = _assign_bb_colors(bb_order)
                logger.info(
                    "angr : %d BBs, %d arêtes",
                    cfg_graph.number_of_nodes(),
                    cfg_graph.number_of_edges(),
                )
            except Exception as exc:
                logger.warning("Extraction angr échouée : %s — panels ASM/CFG omis", exc)

        # ── Construction des panels ────────────────────────────────────
        from .asm_panel   import AsmPanel
        from .cfg_panel   import CfgPanel
        from .paths_panel import PathsPanel
        from .ids_panel   import IdsPanel

        panels_to_render: list[Any] = []  # list[Panel]

        if bb_asm_data and bb_order:
            panels_to_render.append(
                AsmPanel(bb_asm_data, bb_order, bb_colors)
            )
        if cfg_graph and bb_vex_tokens and bb_order:
            panels_to_render.append(
                CfgPanel(cfg_graph, bb_vex_tokens, bb_colors,
                         root_addr=self.func_addr, jump_kinds=jump_kinds)
            )
        if paths_raw:
            panels_to_render.append(
                PathsPanel(paths_raw, max_show=max_paths, max_len=max_len)
            )
        if paths_encoded and vocab:
            panels_to_render.append(
                IdsPanel(paths_encoded, vocab, max_show=max_paths, max_len=max_len)
            )

        if not panels_to_render:
            logger.error("Aucun panel à rendre. Vérifier les chemins de données.")
            sys.exit(1)

        # ── Composition matplotlib ────────────────────────────────────
        fig = self._compose_figure(
            panels_to_render,
            binary_name=binary_name,
            func_addr_hex=func_addr_hex,
            stats=_compute_stats(bb_asm_data, bb_vex_tokens, paths_raw, paths_encoded, vocab),
            dpi=dpi,
        )

        # ── Sauvegarde ────────────────────────────────────────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        logger.info("Figure sauvegardée → %s", out_path.resolve())
        return out_path.resolve()

    # ── Construction de la figure ─────────────────────────────────────────

    def _compose_figure(
        self,
        panels:         list[Any],
        binary_name:    str,
        func_addr_hex:  str,
        stats:          dict[str, str],
        dpi:            int,
    ) -> "matplotlib.figure.Figure":
        """
        Crée la figure GridSpec et délègue le rendu à chaque panel.

        Layout :
          ┌──────────────────────────────── header stats ─────────────────────┐
          │  Panel 0  │  Panel 1  │  Panel 2  │  Panel 3                      │
          └───────────┴───────────┴───────────┴───────────────────────────────┘
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec as mgs

        from .theme import BG, BORDER, PANEL, TEXT_DIM, TEXT_MAIN

        n = len(panels)
        # Largeurs relatives : ASM étroit, CFG moyen, Chemins et IDs larges
        _widths = {
            "ASSEMBLY":    1.2,
            "CFG + VEX IR": 2.0,
            "CHEMINS DFS": 2.8,
            "IDs ENCODÉS": 2.8,
        }
        widths = [_widths.get(p.title, 2.0) for p in panels]

        fig_w = 6 * sum(widths) / n + 4
        fig_h = 10.0

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

        # GridSpec : 1 ligne header + 1 ligne contenu
        gs = mgs.GridSpec(
            2, n,
            figure=fig,
            height_ratios=[0.08, 0.92],
            hspace=0.04,
            wspace=0.08,
            width_ratios=widths,
        )

        # ── Header ────────────────────────────────────────────────────
        ax_hdr = fig.add_subplot(gs[0, :])
        ax_hdr.set_facecolor(PANEL)
        ax_hdr.set_axis_off()
        for spine in ax_hdr.spines.values():
            spine.set_edgecolor(BORDER)

        # Titre principal
        ax_hdr.text(
            0.01, 0.75,
            "PIPELINE STAGES",
            color=TEXT_MAIN, fontsize=12, fontweight="bold",
            fontfamily="monospace", va="top", transform=ax_hdr.transAxes,
        )
        ax_hdr.text(
            0.01, 0.35,
            f"{binary_name}  │  func {func_addr_hex}",
            color=TEXT_DIM, fontsize=8,
            fontfamily="monospace", va="top", transform=ax_hdr.transAxes,
        )

        # Stats en ligne (droite du header)
        stats_str = "   ".join(f"{k}: {v}" for k, v in stats.items())
        ax_hdr.text(
            0.99, 0.5,
            stats_str,
            color=TEXT_DIM, fontsize=7.5,
            fontfamily="monospace", va="center", ha="right",
            transform=ax_hdr.transAxes,
        )

        # ── Séparateurs verticaux entre panels ────────────────────────
        for i in range(1, n):
            x = sum(widths[:i]) / sum(widths)
            ax_hdr.plot(
                [x, x], [0, 1],
                color=BORDER, linewidth=0.8,
                transform=ax_hdr.transAxes, clip_on=False,
            )

        # ── Panels ────────────────────────────────────────────────────
        for col, panel in enumerate(panels):
            ax = fig.add_subplot(gs[1, col])
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
                spine.set_linewidth(0.8)
            panel.render(ax)

        return fig


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualisation multi-stades de la pipeline binaire JEPA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemples :
  # tous les panels (angr requis)
  python -m visualization.pipeline_viz binary.elf 0x402000 \\
      --jsonl-dir data/ --encoded-dir encoded_dataset/ --vocab vocab.json

  # offline uniquement (panels 3/4, sans angr)
  python -m visualization.pipeline_viz binary.elf 0x402000 \\
      --no-angr --jsonl-dir data/ --encoded-dir encoded_dataset/ --vocab vocab.json
        """,
    )
    p.add_argument("elf",       help="Chemin vers le binaire ELF")
    p.add_argument("func_addr", help="Adresse hex de la fonction (ex: 0x402000)")
    p.add_argument("--jsonl-dir",    default=None, help="Dossier data/ (tokens string)")
    p.add_argument("--encoded-dir",  default=None, help="Dossier encoded_dataset/ (IDs int)")
    p.add_argument("--vocab",        default=None, help="Chemin vocab.json")
    p.add_argument("--out",          default="pipeline_stages.png",
                   help="Fichier de sortie PNG/PDF (défaut: pipeline_stages.png)")
    p.add_argument("--max-paths",    type=int, default=12,
                   help="Chemins max affichés dans les panels 3/4 (défaut: 12)")
    p.add_argument("--max-len",      type=int, default=32,
                   help="Tokens max par chemin (défaut: 32)")
    p.add_argument("--dpi",          type=int, default=150,
                   help="Résolution de sortie en DPI (défaut: 150)")
    p.add_argument("--no-angr",      action="store_true",
                   help="Sauter les panels ASM et CFG (pas besoin d'angr)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Logs verbeux")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        func_addr = int(args.func_addr, 16)
    except ValueError:
        logger.error("Adresse invalide : '%s' (format attendu : 0x402000)", args.func_addr)
        sys.exit(1)

    viz = PipelineVisualizer(
        elf_path    = args.elf,
        func_addr   = func_addr,
        jsonl_dir   = args.jsonl_dir,
        encoded_dir = args.encoded_dir,
        vocab_path  = args.vocab,
        no_angr     = args.no_angr,
    )

    out = viz.render(
        out_path  = args.out,
        max_paths = args.max_paths,
        max_len   = args.max_len,
        dpi       = args.dpi,
    )
    print(f"[✓] Figure → {out}")


if __name__ == "__main__":
    main()
