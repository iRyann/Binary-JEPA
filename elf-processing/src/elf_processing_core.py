"""
elf_processing_core.py
======================
Pipeline d'extraction statique de bag-of-paths tokenisés depuis des binaires ELF.

Dépendances :
    pip install angr networkx pebble tqdm

Usage CLI :
    python elf_processing_core.py dataset/ 8

Sortie :
    output/<stem>-<hash>.jsonl   (hash dérivé du chemin complet)
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import random
import traceback
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any, Generator, Iterator

import angr
import networkx as nx
from pebble import ProcessPool
from tqdm import tqdm

# Silence angr/cle
logging.getLogger("angr").setLevel(logging.ERROR)
logging.getLogger("cle").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_PATH_LENGTH = 50
DEFAULT_MAX_PATHS = 500


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════


def _unique_output_name(elf_path: Path) -> str:
    """
    Génère un nom unique stable basé sur le chemin complet.
    Evite collisions si plusieurs fichiers ont le même stem.
    """
    digest = hashlib.sha1(str(elf_path.resolve()).encode()).hexdigest()[:8]
    return f"{elf_path.stem}-{digest}.jsonl"


# ══════════════════════════════════════════════════════════════════════════════
# COEUR DE TRAITEMENT
# ══════════════════════════════════════════════════════════════════════════════


class BinaryAnalyzer:
    """
    Analyse un binaire ELF et produit un bag-of-paths tokenisé.
    """

    def __init__(
        self,
        binary_path: str | Path,
        max_path_length: int = DEFAULT_MAX_PATH_LENGTH,
        max_paths: int = DEFAULT_MAX_PATHS,
        random_seed: int | None = None,
    ):
        self.binary_path = Path(binary_path)
        self.max_path_length = max_path_length
        self.max_paths = max_paths
        self.rng = random.Random(random_seed)

        self._proj: angr.Project | None = None
        self._cfg: Any | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Context manager
    # ──────────────────────────────────────────────────────────────────────────

    def __enter__(self) -> "BinaryAnalyzer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._cfg is not None:
                del self._cfg
            if self._proj is not None:
                del self._proj
        finally:
            self._cfg = None
            self._proj = None
            gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy load angr objects
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def proj(self) -> angr.Project:
        if self._proj is None:
            self._proj = angr.Project(
                str(self.binary_path),
                auto_load_libs=False,
            )
        return self._proj

    @property
    def cfg(self) -> Any:
        if self._cfg is None:
            self._cfg = self.proj.analyses.CFGFast(normalize=True)
        return self._cfg

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract_bag_of_paths(self) -> list[tuple[int, list[str]]]:
        bag: list[tuple[int, list[str]]] = []
        block_cache: dict[int, list[str]] = {}

        for func_addr, func in self.cfg.functions.items():

            if func is None or func.is_simprocedure or func.is_syscall or func.is_plt:
                continue

            blocks = list(func.blocks)
            if not blocks:
                continue

            graph, addr_to_cfgnode = self._build_function_graph(func, blocks)

            entry = func_addr if func_addr in graph else blocks[0].addr
            if entry not in graph:
                continue

            for path in self._enumerate_paths(graph, entry):

                tokens: list[str] = []

                for addr in path:

                    if addr not in block_cache:
                        block_cache[addr] = self._tokenize_block(
                            addr,
                            addr_to_cfgnode.get(addr),
                        )

                    tokens.extend(block_cache[addr])

                if tokens:
                    bag.append((func_addr, tokens))

        return bag

    # ──────────────────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_function_graph(
        self,
        func: Any,
        blocks: list[Any],
    ) -> tuple[nx.DiGraph, dict[int, Any]]:

        g = nx.DiGraph()
        addr_to_cfgnode: dict[int, Any] = {}

        func_addrs = {b.addr for b in blocks}

        for blk in blocks:

            try:
                node = self.cfg.model.get_any_node(blk.addr)
            except Exception:
                node = None

            addr_to_cfgnode[blk.addr] = node
            g.add_node(blk.addr)

        for blk in blocks:

            node = addr_to_cfgnode.get(blk.addr)
            if node is None:
                continue

            successors = getattr(node, "successors", [])

            for succ in successors:

                succ_addr = getattr(succ, "addr", None)

                if succ_addr in func_addrs:
                    g.add_edge(blk.addr, succ_addr)

        return g, addr_to_cfgnode

    # ──────────────────────────────────────────────────────────────────────────
    # Path enumeration
    # ──────────────────────────────────────────────────────────────────────────

    def _enumerate_paths(
        self,
        graph: nx.DiGraph,
        source: int,
    ) -> Generator[list[int], None, None]:

        count = 0
        stack = [(source, [source], frozenset([source]))]

        while stack:

            node, path, visited = stack.pop()

            successors = list(graph.successors(node))
            self.rng.shuffle(successors)

            if not successors or len(path) >= self.max_path_length:

                yield path

                count += 1
                if count >= self.max_paths:
                    return

                continue

            pushed = False

            for succ in successors:

                if succ not in visited:

                    stack.append(
                        (
                            succ,
                            path + [succ],
                            visited | {succ},
                        )
                    )

                    pushed = True

            if not pushed:

                yield path

                count += 1
                if count >= self.max_paths:
                    return

    # ──────────────────────────────────────────────────────────────────────────
    # Block tokenization
    # ──────────────────────────────────────────────────────────────────────────

    def _tokenize_block(
        self,
        block_addr: int,
        cfg_node: Any,
    ) -> list[str]:

        tokens: list[str] = []

        try:
            vex = self.proj.factory.block(block_addr).vex
        except Exception:
            return ["<UNLIFTABLE>"]

        for stmt in vex.statements:

            tag = getattr(stmt, "tag", None)

            if tag is None or tag == "Ist_IMark":
                continue

            if tag == "Ist_WrTmp":

                tokens.append(self._token_wrtmp(getattr(stmt, "data", None)))

            elif tag == "Ist_Store":
                tokens.append("VEX_STORE")

            elif tag == "Ist_Put":
                tokens.append("VEX_REG_WRITE")

            elif tag == "Ist_Exit":
                tokens.append("VEX_EXIT_COND")

            else:
                tokens.append(f"VEX_{tag}")

        api_token = self._get_terminal_api(cfg_node)

        if api_token:
            tokens.append(api_token)
        else:
            jumpkind = getattr(vex, "jumpkind", "Ijk_Unknown")
            tokens.append("JK_" + jumpkind.replace("Ijk_", "").upper())

        return tokens

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _token_wrtmp(data: Any) -> str:

        if data is None:
            return "VEX_WRTMP"

        tag = getattr(data, "tag", "")
        op = getattr(data, "op", None)

        if isinstance(op, str) and "_" in op:
            return "VEX_OP_" + op.split("_")[1][:3].upper()

        mapping = {
            "Iex_Load": "VEX_LOAD",
            "Iex_Const": "VEX_CONST",
            "Iex_Get": "VEX_REG_READ",
        }

        return mapping.get(tag, "VEX_WRTMP")

    def _get_terminal_api(
        self,
        cfg_node: Any,
    ) -> str | None:

        if cfg_node is None:
            return None

        for succ in getattr(cfg_node, "successors", []):

            succ_addr = getattr(succ, "addr", None)

            if succ_addr is None:
                continue

            try:
                func = self.cfg.functions.get_by_addr(succ_addr)
            except Exception:
                func = self.cfg.functions.get(succ_addr)

            tok = self._api_token(func)

            if tok:
                return tok

        return None

    @staticmethod
    def _api_token(func: Any) -> str | None:

        if func is None:
            return None

        if func.is_plt or func.is_simprocedure:
            return f"<API_{func.name.split('@')[0].upper()}>"

        if func.is_syscall:
            return f"<SYSCALL_{func.name.upper()}>"

        return None


# ══════════════════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════════════════


def _analyze_one(args: tuple[Path, Path, dict[str, Any]]) -> tuple[str, int]:

    elf_path, output_dir, kwargs = args

    out_name = _unique_output_name(elf_path)

    out_file = output_dir / out_name
    tmp_file = output_dir / (out_name + ".tmp")

    count = 0

    try:

        with tmp_file.open("w") as f:

            with BinaryAnalyzer(
                elf_path,
                **kwargs,
            ) as analyzer:

                for func_addr, tokens in analyzer.extract_bag_of_paths():

                    f.write(
                        json.dumps(
                            {
                                "file": elf_path.name,
                                "func_addr": hex(func_addr),
                                "tokens": tokens,
                            }
                        )
                        + "\n"
                    )

                    count += 1

        tmp_file.replace(out_file)

        return elf_path.name, count

    except Exception:

        logger.exception(
            "Worker failure on %s",
            elf_path,
        )

        tmp_file.unlink(missing_ok=True)

        raise


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def process_dataset(
    elf_paths: Iterator[str | Path],
    output_dir: str | Path = "output",
    max_workers: int = 4,
    timeout_sec: int = 120,
    **analyzer_kwargs: Any,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    elf_list = [Path(p) for p in elf_paths]

    if not elf_list:
        logger.error("No ELF files found.")
        return

    total_paths = 0

    with ProcessPool(max_workers=max_workers) as pool:

        futures = {
            pool.schedule(
                _analyze_one,
                args=((elf, output_dir, analyzer_kwargs),),
                timeout=timeout_sec,
            ): elf
            for elf in elf_list
        }

        for future in tqdm(
            as_completed(futures),
            total=len(elf_list),
            desc="Extraction",
            unit="elf",
        ):

            elf_path = futures[future]

            try:

                name, count = future.result()

                total_paths += count

                logger.info(
                    "OK      %-40s -> %d paths",
                    name,
                    count,
                )

            except FuturesTimeoutError:

                tmp = output_dir / (_unique_output_name(elf_path) + ".tmp")

                tmp.unlink(missing_ok=True)

                logger.warning(
                    "TIMEOUT %-40s (> %ds)",
                    elf_path.name,
                    timeout_sec,
                )

            except Exception:

                tmp = output_dir / (_unique_output_name(elf_path) + ".tmp")

                tmp.unlink(missing_ok=True)

                logger.exception(
                    "FAILED  %-40s",
                    elf_path.name,
                )

    logger.info(
        "Done. %d total paths in %s",
        total_paths,
        output_dir,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")

    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    if not dataset_dir.exists():

        logger.error(
            "Dataset directory does not exist: %s",
            dataset_dir,
        )

        sys.exit(1)

    elf_paths = sorted(dataset_dir.glob("*.elf"))

    if not elf_paths:

        logger.error(
            "No .elf files in %s",
            dataset_dir,
        )

        sys.exit(1)

    process_dataset(
        elf_paths,
        output_dir="output",
        max_workers=workers,
        timeout_sec=120,
        max_paths=DEFAULT_MAX_PATHS,
        max_path_length=DEFAULT_MAX_PATH_LENGTH,
    )
