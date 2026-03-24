#!/usr/bin/env python3
"""
3_evaluate_similarity.py
========================
Évalue la résilience du modèle binary-JEPA face aux variations de compilation
en calculant la similarité cosinus entre les embeddings de binaires compilés
sous différentes configurations.

Pipeline :
  1. Charge tous les JSONs d'embeddings depuis latentSpace/data/embeddings/
  2. Pour chaque binaire, isole la fonction `main` (ou la plus grosse en fallback)
  3. Agrège les embeddings de ses chemins d'exécution en un vecteur unique (256-dim)
  4. Calcule la similarité cosinus pour toutes les paires possibles
  5. Sépare Paires Positives (même utilitaire) et Paires Négatives (différents)
  6. Affiche les statistiques LaTeX-ready

Usage :
    python 3_evaluate_similarity.py
    python 3_evaluate_similarity.py --embeddings-dir latentSpace/data/embeddings
    python 3_evaluate_similarity.py --hidden-dim 256 --verbose
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ─── Constantes ──────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent
EMBEDDINGS_DIR  = PROJECT_ROOT / "latentSpace" / "data" / "embeddings"
BINARIES_DIR    = PROJECT_ROOT / "latentSpace" / "data" / "binaries"
HIDDEN_DIM      = 256  # Dimension de sortie du Conv1DEncoder par token

# Regex pour parser les noms de fichiers JSON générés par elf64ctx :
# Format : <util>_<compiler>_<opt>.elf-embeddings.json
# Exemple : ls_gcc_O0.elf-embeddings.json
FILENAME_PATTERN = re.compile(
    r"^(?P<util>[a-zA-Z0-9\-]+)_(?P<compiler>gcc|clang)_(?P<opt>O[0-9s])\.elf-embeddings\.json$"
)


# ─── Structures de données ───────────────────────────────────────────────────
@dataclass
class BinaryRecord:
    """Représente un binaire avec son embedding de fonction principale."""
    util: str           # Nom de l'utilitaire (ex: "ls")
    compiler: str       # Compilateur (ex: "gcc")
    opt: str            # Optimisation (ex: "O0")
    label: str          # Label court (ex: "ls_gcc_O0")
    embedding: torch.Tensor  # Vecteur normalisé 256-dim


@dataclass
class SimilarityResult:
    """Résultat d'une comparaison entre deux binaires."""
    label_a: str
    label_b: str
    util_a: str
    util_b: str
    similarity: float
    is_positive: bool  # True = même utilitaire (paire positive)


# ==============================================================================
# Chargement et décodage des embeddings
# ==============================================================================

def load_raw_embeddings(json_path: Path) -> dict[str, list[torch.Tensor]]:
    """
    Charge les embeddings bruts depuis un JSON produit par elf64ctx.

    Format JSON : { "func_address_decimal": [b64_pickled_tensor, ...], ... }
    Retourne    : { "func_address": [tensor_float32, ...] }
    """
    with open(json_path, "r") as f:
        raw: dict[str, list[str]] = json.load(f)

    decoded: dict[str, list[torch.Tensor]] = {}
    for func_addr, b64_list in raw.items():
        tensors = []
        for b64_str in b64_list:
            # Désérialisation : base64 → pickle → list[float] → tensor
            embedding_list = pickle.loads(base64.b64decode(b64_str))
            tensors.append(torch.tensor(embedding_list, dtype=torch.float32))
        decoded[func_addr] = tensors

    return decoded


def get_main_address(elf_path: Path) -> Optional[str]:
    """
    Utilise `nm` pour résoudre l'adresse de la fonction main().

    nm -n retourne des lignes du format : "<hex_addr> T main"
    On convertit l'adresse hexa → entier décimal pour matcher les clés JSON
    (json.dump() convertit les clés entières Python en chaînes décimales).

    Retourne l'adresse sous forme de string décimale, ou None si introuvable.
    """
    if not elf_path.exists():
        return None
    try:
        result = subprocess.run(
            ["nm", "-n", str(elf_path)],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            # Format attendu : "000000000040xxxx T main"
            if len(parts) == 3 and parts[1] in ("T", "t") and parts[2] == "main":
                decimal_addr = str(int(parts[0], 16))
                return decimal_addr
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def path_to_token_repr(embedding: torch.Tensor, hidden_dim: int = HIDDEN_DIM) -> torch.Tensor:
    """
    Convertit un embedding de chemin d'exécution (L × hidden_dim,) aplati
    en un vecteur représentatif fixe de dimension `hidden_dim` via mean-pooling.

    Le Conv1DEncoder produit (B, L, 256) → flatten → (L*256,) pour chaque chemin.
    On reshape en (L, 256) puis on moyenne sur la dimension temporelle → (256,).
    """
    flat_size = embedding.shape[0]
    n_tokens = flat_size // hidden_dim

    if n_tokens == 0:
        # Chemin trop court : on zero-pad à hidden_dim
        return F.pad(embedding, (0, hidden_dim - flat_size))

    # Couper les tokens en excès si la division n'est pas exacte
    usable = n_tokens * hidden_dim
    reshaped = embedding[:usable].reshape(n_tokens, hidden_dim)
    return reshaped.mean(dim=0)  # (hidden_dim,)


def aggregate_function_embedding(
    path_embeddings: list[torch.Tensor],
    hidden_dim: int = HIDDEN_DIM
) -> torch.Tensor:
    """
    Agrège tous les chemins d'exécution d'une fonction en un vecteur unique.

    1. Chaque chemin (L*256,) → mean-pool → (256,)
    2. Moyenne de tous les chemins → (256,)
    3. Re-normalisation L2 → vecteur unitaire

    Le dot-product entre deux vecteurs unitaires = similarité cosinus.
    """
    reprs = [path_to_token_repr(e, hidden_dim) for e in path_embeddings]
    stacked = torch.stack(reprs)          # (N_paths, hidden_dim)
    mean_repr = stacked.mean(dim=0)       # (hidden_dim,)
    norm = mean_repr.norm()

    if norm < 1e-8:
        return mean_repr  # Vecteur nul (pathologique, ne devrait pas arriver)

    return mean_repr / norm               # vecteur unitaire


def get_representative_embedding(
    raw_embeddings: dict[str, list[torch.Tensor]],
    main_addr: Optional[str],
    hidden_dim: int = HIDDEN_DIM,
    verbose: bool = False
) -> tuple[str, torch.Tensor]:
    """
    Sélectionne la fonction `main` (ou la plus grosse en fallback),
    puis agrège ses embeddings de chemins en un vecteur unique.

    Retourne : (func_addr_str, embedding_tensor_256d)
    """
    func_addr = None

    # Stratégie 1 : adresse de main() obtenue via nm
    if main_addr and main_addr in raw_embeddings:
        func_addr = main_addr
        if verbose:
            n_paths = len(raw_embeddings[func_addr])
            print(f"    → main() à l'adresse {func_addr}  ({n_paths} chemins)")

    # Stratégie 2 : fonction avec le plus de chemins d'exécution
    if func_addr is None:
        func_addr = max(raw_embeddings, key=lambda k: len(raw_embeddings[k]))
        n_paths = len(raw_embeddings[func_addr])
        if verbose:
            print(f"    → main() introuvable, fallback : addr={func_addr} ({n_paths} chemins)")

    embedding = aggregate_function_embedding(raw_embeddings[func_addr], hidden_dim)
    return func_addr, embedding


# ==============================================================================
# Calcul des similarités
# ==============================================================================

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Similarité cosinus entre deux vecteurs.
    Comme les embeddings sont déjà normalisés (||v||=1), cos(a,b) = a·b.
    On renormalise par sécurité au cas où l'agrégation aurait légèrement
    dénormalisé les vecteurs.
    """
    a_norm = a / a.norm().clamp(min=1e-8)
    b_norm = b / b.norm().clamp(min=1e-8)
    return torch.dot(a_norm, b_norm).item()


# ==============================================================================
# Parsing des noms de fichiers
# ==============================================================================

def parse_embedding_filename(filename: str) -> Optional[tuple[str, str, str]]:
    """
    Parse un nom de fichier JSON d'embeddings.

    Exemple :
      "ls_gcc_O0.elf-embeddings.json" → ("ls", "gcc", "O0")
      "cat_clang_Os.elf-embeddings.json" → ("cat", "clang", "Os")
      "fichier_inconnu.json" → None
    """
    m = FILENAME_PATTERN.match(filename)
    if m:
        return m.group("util"), m.group("compiler"), m.group("opt")
    return None


# ==============================================================================
# Affichage des résultats
# ==============================================================================

def print_stats(results: list[SimilarityResult]) -> None:
    """
    Affiche les statistiques des similarités pour les paires positives et négatives,
    formatées pour une intégration directe dans un tableau LaTeX.
    """
    positives = [r.similarity for r in results if r.is_positive]
    negatives = [r.similarity for r in results if not r.is_positive]

    def stats(values: list[float]) -> tuple[float, float, float, float]:
        t = torch.tensor(values)
        return t.mean().item(), t.std().item(), t.min().item(), t.max().item()

    # ─── En-tête ─────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  RÉSULTATS DE SIMILARITÉ — binary-JEPA (I-JEPA / BoP-VEX)")
    print("=" * 62)

    # ─── Paires Positives ─────────────────────────────────────────────────────
    if positives:
        μ_pos, σ_pos, min_pos, max_pos = stats(positives)
        print(f"\n  Paires POSITIVES  (même utilitaire, compilateurs/optimis. différents)")
        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  N        = {len(positives)}")
        print(f"  Moyenne  μ = {μ_pos:.4f}")
        print(f"  Écart-type σ = {σ_pos:.4f}")
        print(f"  Min      = {min_pos:.4f}  |  Max = {max_pos:.4f}")
        _print_pair_details(results, is_positive=True)
    else:
        print("\n  [WARN] Aucune paire positive trouvée.")
        μ_pos, σ_pos = float("nan"), float("nan")

    # ─── Paires Négatives ─────────────────────────────────────────────────────
    if negatives:
        μ_neg, σ_neg, min_neg, max_neg = stats(negatives)
        print(f"\n  Paires NÉGATIVES  (utilitaires différents)")
        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  N        = {len(negatives)}")
        print(f"  Moyenne  μ = {μ_neg:.4f}")
        print(f"  Écart-type σ = {σ_neg:.4f}")
        print(f"  Min      = {min_neg:.4f}  |  Max = {max_neg:.4f}")
        _print_pair_details(results, is_positive=False)
    else:
        print("\n  [WARN] Aucune paire négative trouvée.")
        μ_neg, σ_neg = float("nan"), float("nan")

    # ─── Tableau LaTeX ────────────────────────────────────────────────────────
    _print_latex_table(positives, negatives)


def _print_pair_details(results: list[SimilarityResult], is_positive: bool) -> None:
    """Affiche les paires individuelles triées par similarité décroissante."""
    subset = sorted(
        [r for r in results if r.is_positive == is_positive],
        key=lambda r: r.similarity,
        reverse=True
    )
    print()
    print(f"  {'Paire':<45} {'Cos. sim.':>10}")
    print(f"  {'─' * 45} {'─' * 10}")
    for r in subset:
        pair_label = f"{r.label_a}  vs  {r.label_b}"
        print(f"  {pair_label:<45} {r.similarity:>10.4f}")


def _print_latex_table(positives: list[float], negatives: list[float]) -> None:
    """Génère le code LaTeX prêt à coller dans un article scientifique."""

    def _fmt(values: list[float]) -> str:
        if not values:
            return "N/A"
        t = torch.tensor(values)
        μ = t.mean().item()
        σ = t.std().item()
        return f"{μ:.3f} ± {σ:.3f}"

    pos_str = _fmt(positives)
    neg_str = _fmt(negatives)
    sep = "─" * 62

    print()
    print(sep)
    print("  CODE LaTeX  (tableau à coller dans votre papier)")
    print(sep)
    print(r"""
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Type de paires} & \textbf{N} & \textbf{Sim. cosinus $\mu \pm \sigma$} \\
\midrule""")
    print(f"Positives (même binaire)      & {len(positives)} & ${pos_str}$ \\\\")
    print(f"Négatives (binaires différents) & {len(negatives)} & ${neg_str}$ \\\\")
    print(r"""\bottomrule
\end{tabular}
\caption{Similarité cosinus des embeddings de la fonction \texttt{main}
         produits par binary-JEPA (I-JEPA / BoP-VEX) sur les binaires
         Coreutils compilés avec \texttt{gcc~-O0}, \texttt{gcc~-O3},
         et \texttt{clang~-Os}.}
\label{tab:similarity}
\end{table}""")
    print(sep)


# ==============================================================================
# Point d'entrée
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation de la similarité cosinus des embeddings binary-JEPA"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
        help=f"Dossier contenant les JSONs d'embeddings (défaut: {EMBEDDINGS_DIR})"
    )
    parser.add_argument(
        "--binaries-dir",
        type=Path,
        default=BINARIES_DIR,
        help=f"Dossier contenant les ELFs (pour nm, défaut: {BINARIES_DIR})"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=HIDDEN_DIM,
        help=f"Dimension cachée du Conv1DEncoder (défaut: {HIDDEN_DIM})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Afficher les détails de chargement"
    )
    args = parser.parse_args()

    embeddings_dir: Path = args.embeddings_dir
    binaries_dir: Path   = args.binaries_dir
    hidden_dim: int      = args.hidden_dim
    verbose: bool        = args.verbose

    # ─── Vérifications préliminaires ─────────────────────────────────────────
    if not embeddings_dir.exists():
        print(f"[ERROR] Dossier embeddings introuvable : {embeddings_dir}")
        print("        Exécutez d'abord : bash 2_generate_embeddings.sh")
        sys.exit(1)

    json_files = sorted(embeddings_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] Aucun fichier JSON trouvé dans {embeddings_dir}")
        sys.exit(1)

    print(f"[INFO]  {len(json_files)} fichiers d'embeddings trouvés.")
    print(f"[INFO]  Dossier embeddings : {embeddings_dir}")
    print(f"[INFO]  Dossier binaires   : {binaries_dir}")
    print(f"[INFO]  hidden_dim         : {hidden_dim}")

    # ─── Chargement et agrégation ─────────────────────────────────────────────
    records: list[BinaryRecord] = []
    skipped: list[str] = []

    for json_path in json_files:
        parsed = parse_embedding_filename(json_path.name)
        if parsed is None:
            print(f"[WARN]  Nom de fichier non reconnu, ignoré : {json_path.name}")
            skipped.append(json_path.name)
            continue

        util, compiler, opt = parsed
        label = f"{util}_{compiler}_{opt}"

        if verbose:
            print(f"\n[LOAD]  {json_path.name}")

        # Charger les embeddings bruts
        try:
            raw_embeddings = load_raw_embeddings(json_path)
        except Exception as e:
            print(f"[ERROR] Impossible de charger {json_path.name} : {e}")
            skipped.append(json_path.name)
            continue

        if not raw_embeddings:
            print(f"[WARN]  Aucune fonction dans {json_path.name}, ignoré.")
            skipped.append(json_path.name)
            continue

        if verbose:
            print(f"    {len(raw_embeddings)} fonctions chargées.")

        # Résoudre l'adresse de main() via nm sur le binaire correspondant
        elf_path = binaries_dir / f"{label}.elf"
        main_addr = get_main_address(elf_path) if elf_path.exists() else None
        if verbose and not elf_path.exists():
            print(f"    [WARN] ELF introuvable pour nm : {elf_path.name}")

        # Obtenir l'embedding représentatif de la fonction principale
        try:
            _, func_embedding = get_representative_embedding(
                raw_embeddings, main_addr, hidden_dim, verbose
            )
        except Exception as e:
            print(f"[ERROR] Agrégation échouée pour {label} : {e}")
            skipped.append(json_path.name)
            continue

        records.append(BinaryRecord(
            util=util,
            compiler=compiler,
            opt=opt,
            label=label,
            embedding=func_embedding
        ))

    print(f"\n[INFO]  {len(records)} binaires chargés avec succès.")
    if skipped:
        print(f"[WARN]  {len(skipped)} fichiers ignorés : {skipped}")

    if len(records) < 2:
        print("[ERROR] Il faut au moins 2 binaires pour calculer les paires.")
        sys.exit(1)

    # ─── Calcul des paires de similarité ─────────────────────────────────────
    print(f"\n[INFO]  Calcul des similarités cosinus pour toutes les paires...")
    n = len(records)
    n_pairs = n * (n - 1) // 2
    print(f"[INFO]  {n} binaires → {n_pairs} paires à évaluer.")

    similarity_results: list[SimilarityResult] = []

    for rec_a, rec_b in itertools.combinations(records, 2):
        sim = cosine_similarity(rec_a.embedding, rec_b.embedding)
        is_positive = (rec_a.util == rec_b.util)

        similarity_results.append(SimilarityResult(
            label_a=rec_a.label,
            label_b=rec_b.label,
            util_a=rec_a.util,
            util_b=rec_b.util,
            similarity=sim,
            is_positive=is_positive
        ))

    n_pos = sum(1 for r in similarity_results if r.is_positive)
    n_neg = len(similarity_results) - n_pos
    print(f"[INFO]  {n_pos} paires positives  |  {n_neg} paires négatives")

    # ─── Affichage des résultats ──────────────────────────────────────────────
    print_stats(similarity_results)


if __name__ == "__main__":
    main()
