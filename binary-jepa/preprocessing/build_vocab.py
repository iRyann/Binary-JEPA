# build_vocab.py
import json
from collections import Counter
from pathlib import Path


def build_vocabulary(output_dir: str, min_freq: int = 1):
    """
    Lit tous les .jsonl générés et construit un dictionnaire Token -> ID.
    """
    dataset_dir = Path(output_dir)
    token_counts = Counter()

    print(f"[*] Analyse des fichiers dans {dataset_dir}...")

    # 1. Comptage de tous les tokens uniques
    for jsonl_file in dataset_dir.glob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                data = json.loads(line)
                token_counts.update(data["tokens"])

    print(f"[+] {len(token_counts)} tokens uniques trouvés au total.")

    # 2. Création du dictionnaire avec les Special Tokens (Ordre important)
    vocab = {
        "<PAD>": 0,  # Padding pour tenseurs de taille fixe
        "<UNK>": 1,  # Unknown token
        "<MASK>": 2,  # Le token de masquage pour l'entraînement JEPA !
    }

    # 3. Ajout des tokens fréquents
    current_id = 3
    for token, count in token_counts.most_common():
        if count >= min_freq:
            vocab[token] = current_id
            current_id += 1

    # 4. Sauvegarde
    vocab_path = dataset_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=4)

    print(f"[+] Vocabulaire sauvegardé dans {vocab_path} (Taille: {len(vocab)})")


if __name__ == "__main__":
    import sys

    d_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    build_vocabulary(d_dir)
