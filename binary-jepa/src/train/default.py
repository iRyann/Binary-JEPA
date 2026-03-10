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
from setuptools.discovery import ConfigDiscovery
from src.models.jepa import IJEPA
from src.utils.logging import AverageMeter, CSVLogger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = "."
DEFAULT_BATCH_SIZE = 10
DEFAULT_CHECKPOINT_FREQ = 1
DEFAULT_NUMBER_OF_EPOCH = 1
DEFAULT_TRAIN_DATA_PATH = "raw_data.json"
DEFAULT_VOCAB_PATH = "vocab.json"
DEFAULT_LEARNING_RATE = 1e-4

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


# @track_emissions()
def main(config: dict[str, str]):

    # -- Data
    batch_size = int(config.get("batch_size",DEFAULT_BATCH_SIZE))
    checkpoint_path = config.get("checkpoint_path","")
    use_checkpoint = config.get("use_checkpoint",False)

    # -- Model logging
    output_dir = config.get("output_path",DEFAULT_OUTPUT_DIR)
    if not os.path.isdir(output_dir):   os.mkdir(output_dir)        

    checkpoint_freq = config.get("checkpoint_freq",DEFAULT_CHECKPOINT_FREQ)

    # --- log/checkpoints path
    log_file = os.path.join(output_dir, "log.csv")
    save_checkpoint_path = os.path.join(output_dir, "ep{epoch}.pt")
    latest_path = os.path.join(output_dir, "latest.pt")

    # --- CSV Logger
    csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "loss"))

    dataset = load_data(config.get("train_data_path",DEFAULT_TRAIN_DATA_PATH))
    vocab = None
    number_of_epoch = config.get("epoch", DEFAULT_NUMBER_OF_EPOCH)

    with open(config.get("vocab_path",DEFAULT_VOCAB_PATH), "r") as f:
        try:
            vocab = json.load(f)
        except:
            sys.exit(1)

    if not vocab:
        sys.exit(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = IJEPA(vocab_size=len(vocab), dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate",DEFAULT_LEARNING_RATE)))

    # chargement du checkpoint
    if use_checkpoint and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        with open(checkpoint_path,"rb") as f:
            checkpoint = torch.load(f)
        
        model.context_encoder.load_state_dict(checkpoint["encoder"])
        model.target_encoder.load_state_dict(checkpoint["target_encoder"])
        model.predictor.load_state_dict(checkpoint["predictor"])
        optimizer.load_state_dict(checkpoint["opt"])
        print("Loading successful")
        

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


    for epoch in tqdm(range(number_of_epoch), desc="training", unit="epoch"):
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

        absolute_epoch = epoch + checkpoint["epoch"] if checkpoint else epoch
        csv_logger.log(absolute_epoch + 1 , loss)
        save_checkpoint(absolute_epoch)


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
