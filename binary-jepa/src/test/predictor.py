import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F

from src.models.encoder import Conv1DEncoder
from src.models.predictor import Predictor
import torch
import yaml
from src.utils._logging import CSVLogger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "testing"
DEFAULT_BATCH_SIZE = 1
DEFAULT_TEST_DATA_PATH = "raw_data.json"
DEFAULT_VOCAB_PATH = "vocab.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_run_id(config: dict) -> str:
    """
    Génère un identifiant de run lisible et triable.

    Format : YYYYMMDD_HHMMSS_ep{N}_lr{lr}_bs{bs}
    Exemple : 20250311_142301_ep50_lr1e-4_bs32

    Convention :
    - Préfixe horodaté  → tri chronologique naturel (ls, glob)
    - Hyperparamètres   → permet de comparer des runs sans ouvrir les logs
    - Séparateur '_'    → compatible avec noms de fichiers et tableaux
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bs = config.get("batch_size", DEFAULT_BATCH_SIZE)

    return f"{ts}__bs{bs}"


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error("Le fichier %s n'existe pas", config_path)
        sys.exit(1)
    with config_path.open("r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def load_data(file_path: str | Path) -> list:
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error("Le fichier %s n'existe pas", file_path)
        sys.exit(1)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(config: dict):

    # ── Run ID & chemins ──────────────────────────────────────────────────────
    run_id = make_run_id(config)

    base_dir = Path(config.get("output_path", DEFAULT_OUTPUT_DIR))
    logs_dir = base_dir / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"{run_id}.csv"

    logger.info("Run ID        : %s", run_id)
    logger.info("Log CSV       : %s", log_file)

    # ── Config ────────────────────────────────────────────────────────────────
    batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    checkpoint_path = config.get("checkpoint_path", "")

    # ── CSV Logger ────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(
        str(log_file),
        ("%d", "index1"),
        ("%d", "index2"),
        ("%.2f", "unlikeliness"),
        ("%.2f", "similarity"),
        ("%.2f", "distance"),
        ("%.2f", "dot product"),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = load_data(config.get("test_data_path", DEFAULT_TEST_DATA_PATH))
    vocab = None

    with open(config.get("vocab_path", DEFAULT_VOCAB_PATH), "r") as f:
        try:
            vocab = json.load(f)
        except Exception:
            logger.error("Impossible de charger le vocabulaire.")
            sys.exit(1)

    if not vocab:
        logger.error("Le vocabulaire est vide.")
        sys.exit(1)

    # ── Device & Model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Conv1DEncoder(vocab_size=len(vocab)).to(device)
    predictor = Predictor(256).to(device)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if os.path.isfile(checkpoint_path):
        logger.info("Chargement du checkpoint '%s'", checkpoint_path)
        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f)

        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        logger.info("Checkpoint chargé")

    # ── DataLoader ────────────────────────────────────────────────────────────

    data = torch.tensor(dataset, dtype=torch.int64).to(device).unsqueeze(1)

    # ── CUDA info ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        logger.info(
            "\x1b[0;32mCUDA disponible : %s\x1b[0;0m",
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )
    else:
        logger.info("\x1b[0;31mCUDA non disponible — CPU utilisé\x1b[0;0m")

    # ── Boucle de test  ─────────────────────────────────────────────────
    counter = 0
    metrics = []    
    for input_index1 in tqdm(
        range(len(data)),
        desc=f"Embedding only inference",
        unit="batch",
        leave=False,
    ):
        input_vector1 = data[input_index1]
        for input_index2 in range(input_index1,min(input_index1+10,len(data))):
            input_vector2 = data[input_index2]
            # calcul de non ressemblance naive (nombre de token inégaux / nombre de tokens)
            real_length1 = torch.sum(input_vector1 != 0)
            real_length2 = torch.sum(input_vector2 != 0)
            naive_unlikeliness = torch.sum(input_vector1 != input_vector2)/max(real_length1,real_length2)
            if (naive_unlikeliness < 0.40 and naive_unlikeliness != 0) or naive_unlikeliness >= 0.90:
                input_vector1[0][real_length1] = 2
                input_vector2[0][real_length2] = 2
                embedding1 = encoder(input_vector1)
                embedding2 = encoder(input_vector2)
                prediction1 = predictor(embedding1)[0][real_length1]
                prediction2 = predictor(embedding2)[0][real_length2]
                prediction1 /= torch.norm(prediction1)
                prediction2 /= torch.norm(prediction2)

                similarity = F.cosine_similarity(prediction1,prediction2, dim=-1)
                distance = torch.norm(prediction1 - prediction2, dim=-1)
                dot_product = torch.sum(prediction1 * prediction2, dim=-1)
                
                metrics.append((input_index1,
                                input_index2,
                                naive_unlikeliness.item(),
                                similarity.item(),
                                distance.item(),
                                dot_product.item()))
                counter += 1

                if counter > 5:
                    while counter != 0:
                        csv_logger.log(*metrics.pop())
                        counter -= 1

    while counter != 0:
        csv_logger.log(*metrics.pop())
        counter -= 1

    logger.info("Test terminé. Run ID : %s", run_id)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement IJEPA")
    parser.add_argument(
        "--config_path",
        type=str,
        default="test.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
