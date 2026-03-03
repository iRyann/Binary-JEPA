"""
build_vocab.py
==============
Script de preprocessing MLOps pour la construction du vocabulaire.
Il scanne les shards JSONL produits par elf_processing_core.py, compte les
occurrences de chaque token VEX/API, et genere un mapping {token: ID}
optimise pour PyTorch.

Usage :
    python build_vocab.py output/ --out vocab.json --min_freq 2
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SPECIAL TOKENS
# ══════════════════════════════════════════════════════════════════════════════
# L'ordre est critique — leurs IDs (0..4) sont hardcodes dans le modele PyTorch.

SPECIAL_TOKENS = [
    "<PAD>",  # ID 0 : remplissage pour forcer des tenseurs de taille fixe (batching)
    "<UNK>",  # ID 1 : token inconnu (instructions rares, tokens hors-vocab)
    "<MASK>",  # ID 2 : token cible JEPA (remplace les blocs a predire)
    "<BOS>",  # ID 3 : Beginning Of Sequence — repere spatial pour l'attention
    "<EOS>",  # ID 4 : End Of Sequence   — delimite la fin logique du chemin
]


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU VOCABULAIRE
# ══════════════════════════════════════════════════════════════════════════════


def build_vocabulary(
    input_dir: str | Path,
    output_vocab_path: str | Path,
    min_freq: int = 2,
) -> None:
    """
    Scanne tous les fichiers *.jsonl de input_dir en streaming (ligne par ligne)
    pour ne pas exploser la RAM, puis ecrit le vocabulaire final en JSON.

    Args:
        input_dir:          dossier contenant les shards *.jsonl.
        output_vocab_path:  chemin du fichier vocab.json de sortie.
        min_freq:           frequence minimale pour qu'un token soit conserve.
                            Les tokens plus rares sont exclus du vocabulaire et
                            seront mappes sur <UNK> a l'inference. Cela evite
                            d'apprendre des vecteurs latents pour des outliers
                            statistiques qui ne reapparaitront jamais.
    """
    input_dir = Path(input_dir)
    output_vocab_path = Path(output_vocab_path)

    if not input_dir.exists() or not input_dir.is_dir():
        logger.error("Le dossier source '%s' n'existe pas.", input_dir)
        return

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("Aucun fichier .jsonl trouve dans '%s'.", input_dir)
        return

    logger.info("Scan de %d fichiers JSONL...", len(jsonl_files))

    # ── 1. Comptage en streaming ─────────────────────────────────────────────
    token_counter: Counter = Counter()
    total_paths = 0
    skipped_lines = 0

    for file_path in tqdm(jsonl_files, desc="Scanning JSONL", unit="file"):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    tokens = json.loads(line).get("tokens", [])
                    token_counter.update(tokens)
                    total_paths += 1
                except json.JSONDecodeError:
                    skipped_lines += 1
                    continue

    logger.info(
        "Scan termine : %d chemins, %d tokens uniques bruts.",
        total_paths,
        len(token_counter),
    )
    if skipped_lines:
        logger.warning("%d lignes JSON invalides ignorees.", skipped_lines)

    # ── 2. Construction du mapping {token: ID} ───────────────────────────────
    vocab: dict[str, int] = {}

    # 2a. Special tokens en premier — IDs 0..4 fixes
    for idx, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = idx

    # 2b. Tokens du corpus, tries du plus frequent au plus rare (most_common)
    current_id = len(SPECIAL_TOKENS)
    kept_tokens = 0
    dropped_tokens = 0

    for token, count in token_counter.most_common():
        if count >= min_freq:
            vocab[token] = current_id
            current_id += 1
            kept_tokens += 1
        else:
            dropped_tokens += 1

    # ── 3. Sauvegarde ────────────────────────────────────────────────────────
    output_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with output_vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)

    logger.info(
        "Vocabulaire final : %d tokens (%d special + %d corpus, min_freq=%d).",
        len(vocab),
        len(SPECIAL_TOKENS),
        kept_tokens,
        min_freq,
    )
    if dropped_tokens:
        logger.info(
            "%d tokens rares exclus (freq < %d) -> mappes sur <UNK>.",
            dropped_tokens,
            min_freq,
        )
    logger.info("Sauvegarde -> %s", output_vocab_path)


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTREE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generateur de vocabulaire PyTorch a partir des shards JSONL."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Dossier contenant les fichiers *.jsonl produits par elf_processing_core.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="vocab.json",
        help="Chemin du fichier de sortie (defaut : vocab.json)",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="Frequence minimale pour conserver un token (defaut : 2)",
    )

    args = parser.parse_args()
    build_vocabulary(args.input_dir, args.out, args.min_freq)
