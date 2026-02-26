import json
import logging
from pathlib import Path
from typing import Iterator

from elf_processing_core import BinaryAnalyzer

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DATASET
# ══════════════════════════════════════════════════════════════════════════════


def process_dataset(
    elf_paths: Iterator[str | Path],
    output_path: str | Path = "bag_of_paths.jsonl",
    **analyzer_kwargs,
) -> None:
    """
    Itère sur un dataset de binaires et écrit le bag-of-paths en JSONL.

    Args:
        elf_paths:       itérable de chemins vers les binaires.
        output_path:     fichier de sortie JSONL.
        analyzer_kwargs: paramètres forwarded à BinaryAnalyzer
                         (max_paths, max_path_length, random_seed).
    """
    output_path = Path(output_path)
    total_paths = 0

    with output_path.open("w") as f:
        for elf in elf_paths:
            elf = Path(elf)
            logger.info("Processing %s", elf)
            try:
                analyzer = BinaryAnalyzer(elf, **analyzer_kwargs)
                bag = analyzer.extract_bag_of_paths()
                for path in bag:
                    f.write(json.dumps({"file": elf.name, "tokens": path}) + "\n")
                total_paths += len(bag)
                logger.info("  → %d paths extracted", len(bag))
            except Exception as e:
                logger.error("Failed on %s: %s", elf, e)

    logger.info("Done. %d total paths → %s", total_paths, output_path)


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    elf_paths = sorted(dataset_dir.glob("*.elf"))

    process_dataset(elf_paths, max_paths=500, max_path_length=50)
