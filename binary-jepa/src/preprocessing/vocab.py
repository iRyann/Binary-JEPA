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
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SPECIAL TOKENS
# ══════════════════════════════════════════════════════════════════════════════
# L'ordre est critique — leurs IDs (0..4) sont hardcodes dans le modele PyTorch.
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
MASK_TOKEN = "<MASK>"

SPECIAL_TOKENS = [
    PAD_TOKEN,  # ID 0 : remplissage pour forcer des tenseurs de taille fixe (batching)
    UNKNOWN_TOKEN,  # ID 1 : token inconnu (instructions rares, tokens hors-vocab)
    MASK_TOKEN,  # ID 2 : token cible JEPA (remplace les blocs a predire)
    # "<BOS>",  # ID 3 : Beginning Of Sequence — repere spatial pour l'attention
    # "<EOS>",  # ID 4 : End Of Sequence   — delimite la fin logique du chemin
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
        sys.exit(1)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("Aucun fichier .jsonl trouve dans '%s'.", input_dir)
        sys.exit(1)

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


def encode_tokens(vocab_path: str | Path, tokens: list) -> list:
    """[TODO:summary]

    [TODO:description]

    Args:
        vocab_path: [TODO:description]
        tokens: [TODO:description]

    Returns:
        [TODO:description]
    """
    vocab_path = Path(vocab_path)
    vocab = None

    if not vocab_path.exists():
        logger.error("Le fichier %s n'existe pas", vocab_path)

    with vocab_path.open("r") as f:
        vocab = json.load(f)

    if not vocab:
        logger.error("Impossible de récupérer le vocabulaire")

    encoded_unkown_token = vocab.get(UNKNOWN_TOKEN)
    encoded_tokens = []
    for token in tokens:
        encoded_tokens.append(vocab.get(token, encoded_unkown_token))

    return encoded_tokens


def encode_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    vocab_path: str | Path,
) -> None:
    """
    Tokenise le dataset au gré du vocabulaire spécifié.

    Args:
        input_dir:  dossier contenant les shards *.jsonl.
        output_dir: chemin des fichiers contenant les shards *.jsonl dont les tokens sont encodés selon le vocab d'entrée.
        vocab_path: chemin d'accès vers le vocabulaire (vocab.json)
    Returns:
       Données `input_dir` tokenisées vers `output_dir` selon le vocabulaire `vocab_path`.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    vocab_path = Path(vocab_path)
    skipped_lines = 0

    if not input_dir.exists() or not input_dir.is_dir():
        logger.error("Le dossier source '%s' n'existe pas.", input_dir)
        sys.exit(1)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("Aucun fichier .jsonl trouve dans '%s'.", input_dir)
        sys.exit(1)

    logger.info("Scan de %d fichiers JSONL...", len(jsonl_files))
    for file_path in tqdm(jsonl_files, desc="Scanning JSONL", unit="file"):
        encoded_file_lines = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    line_as_json = json.loads(line)
                    tokens = line_as_json.get("tokens", [])
                    encoded_tokens = encode_tokens(vocab_path, tokens)
                    line_as_json["tokens"] = encoded_tokens
                    encoded_file_lines.append(json.dumps(line_as_json) + "\n")

                except json.JSONDecodeError:
                    skipped_lines += 1
                    continue

        if not output_dir.is_dir():
            Path.mkdir(output_dir)

        with open(output_dir / file_path.name, "w") as out:
            out.writelines(encoded_file_lines)

    if skipped_lines:
        logger.warning("%d lignes JSON invalides ignorees.", skipped_lines)


def save_raw_data(
    encoded_data_dir: str | Path,
    raw_data_output_path: str | Path,
    pad_token_id: int = 0,
    limit_sequence_length: int = 500,
) -> list:
    """
    Permet de sauvegarder les séqences de token brutes.

    Args:
        encoded_data_dir: Chemin vers les données encodées.
        raw_data_output_path: Chemin vers les données sérialisées désiré.
        pad_token_id: Id du token de padding selon l'encodage.
        limit_sequence_length: Limite de taille des séquences.

    Returns:
       Sérialise les données `encoded_data_dir` en `raw_data_output_path`.
    """

    encoded_data_dir = Path(encoded_data_dir)
    raw_data_output_path = Path(raw_data_output_path)

    if not encoded_data_dir.is_dir():
        logger.error("Le dossier %s n'existe pas", encoded_data_dir)
        sys.exit(1)

    raw_data = []
    jsonl_files = sorted(encoded_data_dir.glob("*.jsonl"))

    # obtention de la longueur de séquence max
    max_len = 0
    counter = 0
    for file_path in tqdm(
        jsonl_files,
        desc="Checking the maximum size of a token sequence from JSONL",
        unit="file",
    ):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    tokens = json.loads(line).get("tokens", [])
                    max_len = max(max_len, len(tokens))
                    counter += 1
                except:
                    continue

    logger.info(
        "The maximum size of a token sequence is found is %d", max_len, counter, max_len
    )
    logger.warning(
        "The maximumen sequence length found in the dataset (%d) exceeds the specified upper limit (%d), resulting in a truncation",
        max_len,
        limit_sequence_length,
    )
    max_len = min(max_len, limit_sequence_length)
    logger.info("A dataset of %dx%d will be created", counter, max_len)

    # extraction des simples séquences de token + padding
    for file_path in tqdm(
        jsonl_files, desc="Loading token sequences from JSONL", unit="file"
    ):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    tokens = json.loads(line).get("tokens", [])
                    # injection du padding
                    tokens = tokens[:max_len] + (max_len - len(tokens)) * [pad_token_id]
                    raw_data.append(json.dumps(tokens) + ",\n")
                except:
                    continue

    with open(raw_data_output_path, "w") as f:
        f.write(f'[\n{("".join(raw_data))[:-2]}\n]')


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
    parser.add_argument(
        "--with-vocab",
        type=str,
        help="Encoder les tokens du jeu de données. Prend en entrée le vocabulaire utlisé pour l'encodage",
    )
    parser.add_argument(
        "--save-raw",
        type=str,
        help="Encoder les tokens du jeu de données. Prend en entrée le vocabulaire utlisé pour l'encodage",
    )

    args = parser.parse_args()

    if args.with_vocab:
        encode_dataset(args.input_dir, "encoded_dataset", args.with_vocab)
    elif args.save_raw:
        save_raw_data(args.input_dir, args.save_raw)
    else:
        build_vocabulary(args.input_dir, args.out, args.min_freq)
