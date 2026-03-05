"""
dataset.py
==========
Moteur d'ingestion PyTorch pour l'architecture JEPA.
Transforme les fichiers JSONL (Bag-of-Paths) en tenseurs entiers paddés,
groupés par fonction (func_addr).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class BagOfPathsDataset(Dataset):
    def __init__(
        self,
        jsonl_dir: str,
        vocab_path: str,
        max_paths_per_func: int = 16,
        max_path_len: int = 64,
    ):
        """
        Args:
            jsonl_dir: Dossier contenant les shards JSONL.
            vocab_path: Chemin vers le vocab.json généré.
            max_paths_per_func: Nombre max de chemins à garder par fonction (Deep Sets).
            max_path_len: Troncature des chemins trop longs.
        """
        self.max_paths = max_paths_per_func
        self.max_len = max_path_len

        # 1. Chargement du Vocabulaire
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]
        self.bos_id = self.vocab["<BOS>"]
        self.eos_id = self.vocab["<EOS>"]

        # 2. Agrégation en mémoire : On groupe les chemins par 'func_addr'
        # Structure: { "file_name|func_addr": [ [path1], [path2] ] }
        self.functions_data = {}

        jsonl_files = list(Path(jsonl_dir).glob("*.jsonl"))
        print(f"[*] Chargement de {len(jsonl_files)} fichiers JSONL en mémoire...")

        for file_path in jsonl_files:
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Clé unique : Nom du binaire + Adresse de la fonction
                        func_key = f"{data['file']}|{data['func_addr']}"

                        if func_key not in self.functions_data:
                            self.functions_data[func_key] = []

                        # On ne garde que N chemins maximum par fonction
                        if len(self.functions_data[func_key]) < self.max_paths:
                            self.functions_data[func_key].append(data["tokens"])

                    except json.JSONDecodeError:
                        continue

        # Liste plate des clés pour l'indexation __getitem__
        self.function_keys = list(self.functions_data.keys())
        print(
            f"[+] Dataset chargé : {len(self.function_keys)} fonctions uniques trouvées."
        )

    def __len__(self):
        return len(self.function_keys)

    def _tokenize(self, path_tokens: list[str]) -> list[int]:
        """Convertit une liste de strings en liste d'IDs (avec BOS/EOS)."""
        # Troncature si le chemin est trop long (pour garder la place pour BOS/EOS)
        path_tokens = path_tokens[: self.max_len - 2]

        # Conversion avec fallback sur <UNK>
        token_ids = [self.vocab.get(tok, self.unk_id) for tok in path_tokens]

        # Ajout des marqueurs spatiaux
        return [self.bos_id] + token_ids + [self.eos_id]

    def __getitem__(self, idx):
        """
        Renvoie tous les chemins d'UNE fonction.
        Sortie : Un tenseur 2D de taille (num_paths, max_len).
        """
        func_key = self.function_keys[idx]
        raw_paths = self.functions_data[func_key]

        # Tensorisation
        tensor_paths = []
        for path in raw_paths:
            t_ids = self._tokenize(path)
            # Padding direct pour aligner à max_len
            pads_needed = self.max_len - len(t_ids)
            t_ids.extend([self.pad_id] * pads_needed)
            tensor_paths.append(t_ids)

        # Si la fonction n'a pas atteint max_paths, on ajoute des chemins "vides" (remplis de PAD)
        # C'est nécessaire pour que toutes les fonctions aient la même dimension de Batch
        paths_needed = self.max_paths - len(tensor_paths)
        if paths_needed > 0:
            empty_path = [self.pad_id] * self.max_len
            for _ in range(paths_needed):
                tensor_paths.append(empty_path)

        # On renvoie un tenseur PyTorch
        # Shape: (max_paths_per_func, max_path_len) -> Ex: (16, 64)
        return torch.tensor(tensor_paths, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# TEST RAPIDE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Test d'instanciation
    dataset = BagOfPathsDataset(jsonl_dir="output", vocab_path="vocab.json")

    if len(dataset) > 0:
        # On regarde la première fonction du dataset
        sample_tensor = dataset[0]
        print(f"\n[Test] Shape du tenseur pour 1 fonction : {sample_tensor.shape}")
        print(f"[Test] Aperçu du premier chemin (IDs) :\n{sample_tensor[0]}")
    else:
        print("\n❌ Dataset vide. Vérifiez vos chemins.")
