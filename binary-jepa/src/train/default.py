import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import src.masks.noiseproof as mask
import torch
import yaml
from src.models.jepa import IJEPA
from src.utils._logging import AverageMeter, CSVLogger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "training"
DEFAULT_BATCH_SIZE = 10
DEFAULT_CHECKPOINT_FREQ = 1
DEFAULT_NUMBER_OF_EPOCH = 1
DEFAULT_TRAIN_DATA_PATH = "raw_data.json"
DEFAULT_VOCAB_PATH = "vocab.json"
DEFAULT_LEARNING_RATE = 1e-4


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
    ep = config.get("epoch", DEFAULT_NUMBER_OF_EPOCH)
    lr = config.get("learning_rate", DEFAULT_LEARNING_RATE)
    bs = config.get("batch_size", DEFAULT_BATCH_SIZE)

    # Format lr compact : 0.0001 → "1e-4", 0.001 → "1e-3"
    lr_str = f"{float(lr):.0e}".replace("+0", "").replace("-0", "-")

    return f"{ts}_ep{ep}_lr{lr_str}_bs{bs}"


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

    base_dir      = Path(config.get("output_path", DEFAULT_OUTPUT_DIR))
    logs_dir      = base_dir / "logs"
    checkpts_dir  = base_dir / "checkpoints" / run_id

    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpts_dir.mkdir(parents=True, exist_ok=True)

    log_file            = logs_dir      / f"{run_id}.csv"
    save_checkpoint_path = checkpts_dir / "ep{epoch}.pt"
    latest_path         = checkpts_dir  / "latest.pt"

    logger.info("Run ID        : %s", run_id)
    logger.info("Log CSV       : %s", log_file)
    logger.info("Checkpoints   : %s", checkpts_dir)

    # ── Config ────────────────────────────────────────────────────────────────
    batch_size      = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    checkpoint_path = config.get("checkpoint_path", "")
    use_checkpoint  = config.get("use_checkpoint", False)
    checkpoint_freq = config.get("checkpoint_freq", DEFAULT_CHECKPOINT_FREQ)
    number_of_epoch = config.get("epoch", DEFAULT_NUMBER_OF_EPOCH)

    # ── CSV Logger ────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(
        str(log_file),
        ("%d",   "epoch"),
        ("%.5f", "loss"),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = load_data(config.get("train_data_path", DEFAULT_TRAIN_DATA_PATH))
    vocab   = None

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
    model  = IJEPA(vocab_size=len(vocab), dim=256).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", DEFAULT_LEARNING_RATE)),
    )

    # ── Checkpoint ────────────────────────────────────────────────────────────
    checkpoint = None
    if use_checkpoint and os.path.isfile(checkpoint_path):
        logger.info("Chargement du checkpoint '%s'", checkpoint_path)
        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f)

        model.context_encoder.load_state_dict(checkpoint["encoder"])
        model.target_encoder.load_state_dict(checkpoint["target_encoder"])
        model.predictor.load_state_dict(checkpoint["predictor"])
        optimizer.load_state_dict(checkpoint["opt"])
        logger.info("Checkpoint chargé (epoch %d)", checkpoint["epoch"])

    # ── DataLoader ────────────────────────────────────────────────────────────
    data = DataLoader(
        torch.tensor(dataset, dtype=torch.int64).to(device),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=False,
    )

    # ── CUDA info ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        logger.info("\x1b[0;32mCUDA disponible : %s\x1b[0;0m",
                    torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        logger.info("\x1b[0;31mCUDA non disponible — CPU utilisé\x1b[0;0m")

    # ── Checkpoint saver ──────────────────────────────────────────────────────
    loss_meter = AverageMeter()

    def save_checkpoint(epoch: int):
        save_dict = {
            "encoder":        model.context_encoder.state_dict(),
            "predictor":      model.predictor.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "opt":            optimizer.state_dict(),
            "epoch":          epoch,
            "loss":           loss_meter.avg,
            "batch_size":     batch_size,
            "run_id":         run_id, #  traçabilité dans le checkpoint
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, str(save_checkpoint_path).format(epoch=epoch + 1))

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    epoch_offset = checkpoint["epoch"] if use_checkpoint else 0

    for epoch in tqdm(range(number_of_epoch), desc="Entraînement", unit="epoch"):
        loss_meter.reset()

        for batch in tqdm(
            data,
            desc=f"Epoch {epoch + 1}/{number_of_epoch}",
            unit="batch",
            leave=False,
        ):
            input_mask, pred_mask = mask.generate(batch, batch.shape[0])

            loss = model(batch, input_mask, pred_mask)
            loss_meter.update(loss.item(), batch.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model._update_target_encoder()

        absolute_epoch = epoch_offset + epoch
        
        logger.info(
            "Époque %d terminée. Loss moyenne : %.4f",
            absolute_epoch + 1,
            loss_meter.avg,
        )
        csv_logger.log(absolute_epoch + 1, loss_meter.avg)
        save_checkpoint(absolute_epoch)

    logger.info("Entraînement terminé. Run ID : %s", run_id)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement IJEPA"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="train.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
