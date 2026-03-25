"""
visualization
=============
Module de visualisation multi-stades de la pipeline binaire JEPA.

Expose les éléments publics :
  - PipelineVisualizer : orchestrateur principal (CLI + API Python)
  - Panel              : interface abstraite pour les backends futurs
  - TokenCategory      : enum de catégories sémantiques des tokens

Usage rapide :
    from visualization import PipelineVisualizer

    viz = PipelineVisualizer(
        elf_path    = "data/b2sum_gcc_O0.elf",
        func_addr   = 0x402000,
        jsonl_dir   = "data/",
        encoded_dir = "encoded_dataset/",
        vocab_path  = "vocab.json",
    )
    viz.render(out_path="stages.png")
"""

from .pipeline_viz  import PipelineVisualizer
from .base_panel    import Panel
from .token_colors  import TokenCategory, categorize, hex_color, display_name

__all__ = [
    "PipelineVisualizer",
    "Panel",
    "TokenCategory",
    "categorize",
    "hex_color",
    "display_name",
]
