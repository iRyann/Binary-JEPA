import angr
import networkx as nx
import random
import logging
import gc
from pathlib import Path
from typing import Generator, Any

logging.getLogger("angr").setLevel(logging.ERROR)
logging.getLogger("cle").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

DEFAULT_MAX_PATH_LENGTH = 50
DEFAULT_MAX_PATHS       = 500

class BinaryAnalyzer:
    """
    Analyse un unique binaire ELF et produit un bag-of-paths tokenisé.

    Utiliser impérativement comme context manager pour garantir la
    libération mémoire des objets angr (connus pour leur empreinte RAM) :

        with BinaryAnalyzer(elf_path) as analyzer:
            bag = analyzer.extract_bag_of_paths()
    """

    def __init__(
        self,
        binary_path: str | Path,
        max_path_length: int    = DEFAULT_MAX_PATH_LENGTH,
        max_paths: int          = DEFAULT_MAX_PATHS,
        random_seed: int | None = None,
    ):
        self.binary_path     = Path(binary_path)
        self.max_path_length = max_path_length
        self.max_paths       = max_paths

        # [Fix #3] Seed initialisée une seule fois à la construction,
        # pas à chaque appel de _enumerate_paths (évite la corrélation inter-fonctions).
        if random_seed is not None:
            random.seed(random_seed)

        self._proj: angr.Project | None = None
        self._cfg: Any                  = None

    # ── Context Manager (Fix #2 : libération mémoire angr) ─────────────────

    def __enter__(self) -> "BinaryAnalyzer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cfg  = None
        self._proj = None
        gc.collect()

    # ── Chargement paresseux ────────────────────────────────────────────────

    @property
    def proj(self) -> angr.Project:
        if self._proj is None:
            logger.debug("Lifting %s", self.binary_path)
            self._proj = angr.Project(str(self.binary_path), auto_load_libs=False)
        return self._proj

    @property
    def cfg(self) -> Any:
        if self._cfg is None:
            logger.debug("Building CFG for %s", self.binary_path)
            self._cfg = self.proj.analyses.CFGFast(normalize=True)
        return self._cfg

    # ── API publique ────────────────────────────────────────────────────────

    def extract_bag_of_paths(self) -> list[list[str]]:
        """Point d'entrée principal : retourne le bag-of-paths complet."""
        bag: list[list[str]]         = []
        block_cache: dict[int, list[str]] = {}  # [Fix #1] cache VEX par adresse de bloc

        for func_addr, func in self.cfg.functions.items():
            if func.is_simprocedure or func.is_syscall or func.is_plt:
                continue

            blocks = list(func.blocks)
            if not blocks:
                continue

            g, addr_to_cfgnode = self._build_function_graph(func, blocks)

            entry = func_addr if func_addr in g else blocks[0].addr
            if entry not in g:
                continue

            for path_addrs in self._enumerate_paths(g, entry):
                token_path: list[str] = []
                for blk_addr in path_addrs:
                    # [Fix #1] On ne lift le VEX qu'une seule fois par bloc,
                    # même si le bloc apparaît dans 500 chemins différents.
                    if blk_addr not in block_cache:
                        block_cache[blk_addr] = self._tokenize_block(
                            blk_addr, addr_to_cfgnode.get(blk_addr)
                        )
                    token_path.extend(block_cache[blk_addr])

                if token_path:
                    bag.append(token_path)

        return bag

    # ── Construction du graphe intra-fonction ───────────────────────────────

    def _build_function_graph(
        self, func: Any, blocks: list[Any]
    ) -> tuple[nx.DiGraph, dict[int, Any]]:
        g = nx.DiGraph()
        addr_to_cfgnode: dict[int, Any] = {}
        func_addrs = {b.addr for b in blocks}

        for blk in blocks:
            cfg_node = self.cfg.model.get_any_node(blk.addr)
            addr_to_cfgnode[blk.addr] = cfg_node
            g.add_node(blk.addr)

        for blk in blocks:
            cfg_node = addr_to_cfgnode[blk.addr]
            if cfg_node is None:
                continue
            for succ in cfg_node.successors:
                if succ.addr in func_addrs:
                    g.add_edge(blk.addr, succ.addr)

        return g, addr_to_cfgnode

    # ── Énumération des chemins (DFS aléatoire, sans break_cycles) ──────────

    def _enumerate_paths(
        self, graph: nx.DiGraph, source: int
    ) -> Generator[list[int], None, None]:
        count = 0
        stack = [(source, [source], frozenset([source]))]

        while stack:
            node, path, visited = stack.pop()
            successors = list(graph.successors(node))
            random.shuffle(successors)

            if not successors or len(path) >= self.max_path_length:
                yield path
                count += 1
                if count >= self.max_paths:
                    return
                continue

            pushed_any = False
            for succ in successors:
                if succ not in visited:
                    stack.append((succ, path + [succ], visited | {succ}))
                    pushed_any = True

            if not pushed_any:
                yield path
                count += 1
                if count >= self.max_paths:
                    return

    # ── Tokenisation d'un bloc ──────────────────────────────────────────────

    def _tokenize_block(self, block_addr: int, cfg_node: Any) -> list[str]:
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

        api_tok = self._get_terminal_api(cfg_node)
        if api_tok:
            tokens.append(api_tok)
        else:
            jk = getattr(vex, "jumpkind", "Ijk_Unknown")
            tokens.append(f"JK_{jk.replace('Ijk_', '').upper()}")

        return tokens

    # ── Helpers de tokenisation ─────────────────────────────────────────────

    @staticmethod
    def _token_wrtmp(data: Any) -> str:
        if data is None:
            return "VEX_WrTmp"
        tag    = getattr(data, "tag", "")
        op_val = getattr(data, "op", None)
        if op_val and isinstance(op_val, str) and "_" in op_val:
            famille = op_val.split("_")[1][:3].upper()
            return f"VEX_OP_{famille}"
        return {
            "Iex_Load":  "VEX_LOAD",
            "Iex_Const": "VEX_CONST",
            "Iex_Get":   "VEX_REG_READ",
        }.get(tag, "VEX_WrTmp")

    def _get_terminal_api(self, cfg_node: Any) -> str | None:
        if cfg_node is None:
            return None
        for succ in cfg_node.successors:
            tok = self._api_token(self.cfg.functions.get(succ.addr))
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

