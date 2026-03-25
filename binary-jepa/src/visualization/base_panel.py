"""
base_panel.py
=============
Interface abstraite commune à tous les panels de visualisation.

Architecture d'extensibilité
-----------------------------
Le design sépare *données* et *rendu* :
  - Les panels reçoivent des données pré-calculées (Python pur, pas d'objets angr).
  - La méthode render() est spécifique au backend (matplotlib pour la version
    statique, HTML/SVG pour les versions interactives futures).

Pour ajouter un backend interactif :
  1. Créer une classe abstraite HTMLPanel(ABC) analogue avec render(div_id) -> str.
  2. Faire hériter chaque panel concret des deux ABCs (mixin pattern) :
       class AsmPanel(MatplotlibPanel, HTMLPanel): ...
  3. Ou : factory pattern — PipelineVisualizer.render(backend="html") instancie
     les variantes HTML à la place des variantes matplotlib.

Aucun changement aux panels existants n'est requis pour ajouter un backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes


class Panel(ABC):
    """
    Interface d'un panel de visualisation ciblant matplotlib.

    Chaque panel encapsule :
      - des données immutables passées au constructeur
      - une logique de rendu dans render(ax)

    Les panels ne partagent pas d'état mutable et peuvent donc être
    rendus dans n'importe quel ordre ou en parallèle (future multi-thread).
    """

    @property
    @abstractmethod
    def title(self) -> str:
        """Titre du panel (affiché en haut à gauche de l'axes)."""
        ...

    @property
    def subtitle(self) -> str:
        """
        Sous-titre optionnel (stats concises, ex: "47 instrs / 8 BBs").
        Affiché en gris discret à côté du titre.
        """
        return ""

    @abstractmethod
    def render(self, ax: "matplotlib.axes.Axes") -> None:
        """
        Dessine le contenu de ce panel dans les axes matplotlib fournis.

        Le panel ne crée pas de figure, n'appelle pas plt.show() et ne
        modifie pas rcParams — ces responsabilités appartiennent à
        PipelineVisualizer.

        Args:
            ax: axes matplotlib alloués par l'orchestrateur.
        """
        ...

    def render_title(self, ax: "matplotlib.axes.Axes") -> None:
        """
        Convenience : affiche title + subtitle via style_axes().
        Appelé par render() si le panel ne gère pas lui-même son titre.
        """
        from .theme import style_axes
        style_axes(ax, title=self.title, subtitle=self.subtitle)
