import argparse
import json
import logging
import sys
from pathlib import Path

import src.masks.noiseproof as mask
import torch
import yaml
from codecarbon import track_emissions
from src.models.jepa import IJEPA
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, str]:
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
        raw_data = json.load(f)

    return raw_data


@track_emissions()
def main(config: dict[str, str]):

    dataset = load_data(config["train_data_path"])

    model = IJEPA(vocab_size=5000, dim=256)  # obtenir la taille du vocab
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    data = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=2,
        persistent_workers=False,
    )

    for batch in tqdm(data):  # obtenir le jeu de données, sous quelle forme ?

        input_mask, pred_mask = mask.generate(batch.shape[0])

        loss = model(batch, input_mask, pred_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model._update_target_encoder()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline par défaut d'entrainement du modèle"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="train.yaml",
        help="Chemin d'accès vers le fichier de configuration",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
