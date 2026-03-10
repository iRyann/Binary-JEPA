import argparse
import json
import logging
import os
import sys
from pathlib import Path

import src.masks.noiseproof as mask
import torch
import yaml
from codecarbon import track_emissions
from networkx import connected_dominating_set
from setuptools.discovery import ConfigDiscovery
from src.models.jepa import IJEPA
from src.utils.logging import AverageMeter, CSVLogger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

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

    # -- Data
    batch_size = int(config["batch_size"])

    # -- Model logging
    log_path = config["logging_path"]
    load_path = (
        None
        if config.get("load_checkpoint") is None
        else config["load_checkpoint_path"]
    )  # TODO
    checkpoint_freq = config["chechpoint_freq"]

    # --- log/checkpoints path
    log_file = os.path.join(log_path, "log.csv")
    save_checkpoint_path = os.path.join(log_path, "ep{epoch}.pth.tar")
    latest_path = os.path.join(log_path, "latest.pth.tar")

    # --- CSV Logger
    csv_logger = (
        CSVLogger(log_file),
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "time (ms)"),
    )

    dataset = load_data(config["train_data_path"])
    vocab = None
    number_of_epoch = config.get("epoch", 1)

    with open(config["vocab_path"], "r") as f:
        try:
            vocab = json.load(f)
        except:
            sys.exit(1)

    if not vocab:
        sys.exit(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = IJEPA(vocab_size=len(vocab), dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    data = DataLoader(
        torch.Tensor(dataset).to(torch.int64).to(device),
        batch_size=batch_size,
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

    # CUDA logging
    if torch.cuda.is_available():
        logger.info("\x1b[0;32mCUDA is available\x1b[0;0m")
    else:
        logger.info("\x1b[0;31mCUDA is NOT available\x1b[0;0m")

    logger.info(
        "torch using device %s", torch.cuda.get_device_name(torch.cuda.current_device())
    )

    def save_checkpoint(epoch):
        save_dict = {
            "encoder": model.context_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_checkpoint_path.format(epoch=f"{epoch + 1}"))

    for epoch in tqdm(range(number_of_epoch), desc="training", unit="batch"):
        loss_meter = AverageMeter()
        for batch in tqdm(
            data, desc=f"training epoch {epoch+1}/{number_of_epoch}", unit="batch"
        ):

            input_mask, pred_mask = mask.generate(batch, batch.shape[0])

            loss = model(batch, input_mask, pred_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model._update_target_encoder()

        save_checkpoint(epoch + 1)  # TODO


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
