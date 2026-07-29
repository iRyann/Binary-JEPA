"""
dedup.py
========
Stage B du pipeline data : déduplication sans fuite (data leakage) du
dataset désucré data_v2/.

Problème traité (report_v2.txt) :
    - 74.9% de doublons globaux : même fonction compilée sous 20 variantes
      (binaire × compilateur × -O*) + boilerplate CRT/glibc partagé par
      construction entre les 1000 binaires (cf. Allamanis 2018, "The
      Adverse Effects of Code Duplication in ML Models of Code")

Trois niveaux, dans l'ordre :
    1. EXACT   — hash blake2b-8B du chemin désucré : suppression globale
                 des occurrences identiques (1re occurrence conservée)
    2. NEAR    — MinHash (128 perms) sur shingles de 5-grammes de tokens,
                 LSH en 16 bandes × 8 lignes (seuil J ≈ 0.85), clustering
                 par union-find ; chaque cluster conserve ≤ K représentants
                 en privilégiant des variantes (compilateur, -O) DISTINCTES
                 — la variation cross-compilateur est le signal d'entraînement
                 pour la résilience à l'obfuscation, on plafonne l'influence
                 d'un cluster au lieu de le réduire à un singleton
    3. BOILER  — excision du boilerplate au niveau FONCTION (préserve la
                 structure Bag-of-Paths : on supprime des sacs entiers,
                 jamais des chemins intra-sac) : empreinte = hash de
                 l'ensemble des chemins uniques de la fonction ; une
                 empreinte présente dans ≥ N familles de binaires
                 (défaut 50) ⇒ stub CRT/glibc ⇒ sac entier supprimé

Split group-aware : le split train/val est haché sur l'ID de cluster
(jamais sur le chemin ou le fichier) ⇒ aucune famille de fonctions ne
peut apparaître à la fois en train et en val (leakage Allamanis).

Deux passages streaming sur data_v2/ ; seuls signatures MinHash,
ensembles de digests et métadonnées légères sont conservés en RAM.

Usage :
    python -m src.preprocessing.dedup \
        --data-dir data_v2/ --out-dir data_v3/
"""

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES MINHASH / LSH
# ══════════════════════════════════════════════════════════════════════════════

N_PERMS = 128          # permutations MinHash
N_BANDS = 16           # bandes LSH
ROWS_PER_BAND = N_PERMS // N_BANDS   # 8 lignes/bande → seuil J ≈ (1/16)^(1/8) ≈ 0.71
                                     # resserré par la validation Jaccard ci-dessous
JACCARD_THRESHOLD = 0.85             # validation exacte des paires candidates
SHINGLE_SIZE = 5       # n-grammes de tokens
MERSENNE_PRIME = np.uint64((1 << 61) - 1)

# Graine fixe → signatures reproductibles
_RNG = np.random.default_rng(42)
_HASH_A = _RNG.integers(1, 2**61 - 1, size=N_PERMS, dtype=np.uint64)
_HASH_B = _RNG.integers(0, 2**61 - 1, size=N_PERMS, dtype=np.uint64)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def path_digest(tokens: list[str]) -> bytes:
    """Hash blake2b 8 octets d'un chemin (dédoublonnage exact)."""
    h = hashlib.blake2b(digest_size=8)
    for t in tokens:
        h.update(t.encode())
        h.update(b"\x00")
    return h.digest()


def _shingles(tokens: list[str]) -> set[int]:
    """5-grammes du chemin, hashés en uint32. Chemins courts → 1 shingle global."""
    n = len(tokens)
    if n <= SHINGLE_SIZE:
        grams = [tuple(tokens)]
    else:
        grams = [tuple(tokens[i:i + SHINGLE_SIZE]) for i in range(n - SHINGLE_SIZE + 1)]
    out = set()
    for g in grams:
        h = hashlib.blake2b(digest_size=4)
        h.update("\x00".join(g).encode())
        out.add(int.from_bytes(h.digest(), "little"))
    return out


def minhash_signature(shingles: set[int]) -> np.ndarray:
    """Signature MinHash (N_PERMS uint32) d'un ensemble de shingles."""
    # (a*s + b) mod p, min sur les shingles — vectorisé sur les 128 permutations
    arr = np.fromiter(shingles, dtype=np.uint64)
    hashed = (_HASH_A[:, None] * arr[None, :] + _HASH_B[:, None]) % MERSENNE_PRIME
    return hashed.min(axis=1).astype(np.uint32)


def jaccard(shingles_a: set[int], shingles_b: set[int]) -> float:
    """Jaccard exact entre deux ensembles de shingles (validation des paires)."""
    if not shingles_a and not shingles_b:
        return 1.0
    return len(shingles_a & shingles_b) / len(shingles_a | shingles_b)


def parse_variant(stem: str) -> tuple[str, str]:
    """'b2sum_clang_O0' → (famille='b2sum', variante='clang_O0')."""
    parts = stem.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "?")


class UnionFind:
    """Union-find avec compression de chemin, sur des indices 0..n-1."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ══════════════════════════════════════════════════════════════════════════════
# PASSAGE 1 : collecte (digests exacts, signatures, sacs de fonctions)
# ══════════════════════════════════════════════════════════════════════════════

def collect(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"Aucun fichier JSONL dans {data_dir}")

    seen_exact: dict[bytes, int] = {}          # digest → index du chemin unique
    signatures: list[np.ndarray] = []          # signature MinHash par chemin unique
    shingles_list: list[np.ndarray] = []       # shingles conservés pour Jaccard exact
    path_meta: list[tuple] = []                # (file_idx, func_addr, variant, family)
    func_bags: dict[tuple, set] = defaultdict(set)  # (file_idx, func_addr) → digests

    n_paths_in = 0
    n_exact_dups = 0

    for file_idx, src in enumerate(jsonl_files):
        family, variant = parse_variant(src.stem)
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens = rec.get("tokens", [])
                if not tokens:
                    continue
                n_paths_in += 1

                dg = path_digest(tokens)
                func_bags[(file_idx, rec.get("func_addr", "?"))].add(dg)

                if dg in seen_exact:
                    n_exact_dups += 1
                    continue

                sh = _shingles(tokens)
                seen_exact[dg] = len(signatures)
                signatures.append(minhash_signature(sh))
                shingles_list.append(sh)
                path_meta.append((file_idx, rec.get("func_addr", "?"), variant, family))

        if (file_idx + 1) % 100 == 0 or file_idx == len(jsonl_files) - 1:
            logger.info(
                "  [passage 1] %4d/%d fichiers | chemins: %d | uniques exacts: %d",
                file_idx + 1, len(jsonl_files), n_paths_in, len(signatures),
            )

    return {
        "jsonl_files": jsonl_files,
        "seen_exact": seen_exact,
        "signatures": signatures,
        "shingles": shingles_list,
        "path_meta": path_meta,
        "func_bags": func_bags,
        "n_paths_in": n_paths_in,
        "n_exact_dups": n_exact_dups,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 2 : clustering LSH + sélection des représentants
# ══════════════════════════════════════════════════════════════════════════════

def cluster(data: dict) -> dict[int, list[int]]:
    """LSH banding → paires candidates → validation Jaccard → union-find."""
    sigs = np.stack(data["signatures"])        # (N, N_PERMS) uint32
    n = len(sigs)
    uf = UnionFind(n)
    bands = sigs.reshape(n, N_BANDS, ROWS_PER_BAND)

    n_candidates = 0
    n_validated = 0
    for b in range(N_BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        band_bytes = bands[:, b, :].tobytes()
        row_size = ROWS_PER_BAND * 4           # 8 uint32 = 32 octets
        for i in range(n):
            buckets[band_bytes[i * row_size:(i + 1) * row_size]].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # Validation Jaccard exacte des paires du bucket (les buckets LSH
            # peuvent contenir des faux positifs). Pour les gros buckets
            # (chemins triviaux quasi identiques), on borne la comparaison
            # aux 50 premiers pivots → évite l'explosion quadratique.
            for y in range(1, len(members)):
                for x in range(min(y, 50)):
                    i, j = members[x], members[y]
                    if uf.find(i) == uf.find(j):
                        continue
                    n_candidates += 1
                    if jaccard(data["shingles"][i], data["shingles"][j]) >= JACCARD_THRESHOLD:
                        uf.union(i, j)
                        n_validated += 1
        if (b + 1) % 4 == 0:
            logger.info("  [niveau 2] bande %2d/%d traitée", b + 1, N_BANDS)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    logger.info(
        "  [niveau 2] %d paires candidates, %d validées J≥%.2f → %d clusters",
        n_candidates, n_validated, JACCARD_THRESHOLD, len(clusters),
    )
    return dict(clusters)


def select_representatives(
    clusters: dict[int, list[int]],
    path_meta: list[tuple],
    k: int,
) -> dict[int, int]:
    """Par cluster : ≤ K représentants, variantes distinctes d'abord.

    Returns:
        {index_chemin_unique: cluster_id_compact}
    """
    keep: dict[int, int] = {}
    # Cluster IDs compacts et déterministes (tri par plus petit membre)
    ordered = sorted(clusters.values(), key=lambda m: min(m))
    for cid, members in enumerate(ordered):
        by_variant: dict[str, int] = {}
        for i in sorted(members):
            variant = path_meta[i][2]
            by_variant.setdefault(variant, i)   # 1er chemin de chaque variante
        chosen = list(by_variant.values())[:k]
        if not chosen:                          # ne devrait pas arriver
            chosen = [min(members)]
        for i in chosen:
            keep[i] = cid
    return keep


# ══════════════════════════════════════════════════════════════════════════════
# NIVEAU 3 : excision du boilerplate (fonctions entières)
# ══════════════════════════════════════════════════════════════════════════════

def find_boilerplate(
    func_bags: dict[tuple, set],
    jsonl_files: list[Path],
    min_families: int,
) -> set[tuple]:
    """Empreinte de sac = hash de l'ensemble des digests de la fonction.

    Une empreinte partagée par ≥ min_families familles de binaires distinctes
    ⇒ code runtime identique par construction (_start, stubs glibc).
    """
    fp_families: dict[bytes, set] = defaultdict(set)
    for (file_idx, func_addr), digests in func_bags.items():
        h = hashlib.blake2b(digest_size=8)
        for dg in sorted(digests):
            h.update(dg)
        fp = h.digest()
        family, _ = parse_variant(jsonl_files[file_idx].stem)
        fp_families[fp].add(family)

    boilerplate_fps = {fp for fp, fams in fp_families.items()
                       if len(fams) >= min_families}

    dropped: set[tuple] = set()
    for key, digests in func_bags.items():
        h = hashlib.blake2b(digest_size=8)
        for dg in sorted(digests):
            h.update(dg)
        if h.digest() in boilerplate_fps:
            dropped.add(key)

    logger.info(
        "  [niveau 3] %d empreintes de fonctions | %d boilerplate (≥%d familles) "
        "→ %d sacs de fonctions supprimés",
        len(fp_families), len(boilerplate_fps), min_families, len(dropped),
    )
    return dropped


# ══════════════════════════════════════════════════════════════════════════════
# PASSAGE 2 : écriture du dataset dédupliqué
# ══════════════════════════════════════════════════════════════════════════════

def split_of(cluster_id: int, val_pct: int) -> str:
    """Split déterministe haché sur l'ID de cluster (group-aware, anti-leakage)."""
    h = (cluster_id * 2654435761) % 2**32
    return "val" if h % 100 < val_pct else "train"


def write_output(args: argparse.Namespace, data: dict, keep: dict[int, int],
                 boilerplate: set[tuple]) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # digest → (cluster_id) pour les chemins conservés
    digest_to_cluster: dict[bytes, int] = {}
    for digest, idx in data["seen_exact"].items():
        if idx in keep:
            digest_to_cluster[digest] = keep[idx]

    stats = Counter()
    split_counts = Counter()
    cluster_sizes_kept = Counter()
    written: set[bytes] = set()   # digests déjà écrits (1re occurrence seulement)
    path_id = 0

    for file_idx, src in enumerate(data["jsonl_files"]):
        dst = out_dir / src.name
        with src.open("r", encoding="utf-8") as fin, \
             dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens = rec.get("tokens", [])
                if not tokens:
                    continue

                func_addr = rec.get("func_addr", "?")
                if (file_idx, func_addr) in boilerplate:
                    stats["dropped_boilerplate"] += 1
                    continue

                dg = path_digest(tokens)
                if dg not in digest_to_cluster:
                    # non-représentant de son cluster (niveau 2)
                    stats["dropped_dup_or_nonrep"] += 1
                    continue
                if dg in written:
                    # doublon exact d'un chemin déjà écrit (niveau 1)
                    stats["dropped_dup_or_nonrep"] += 1
                    continue
                written.add(dg)

                cid = digest_to_cluster[dg]
                split = split_of(cid, args.val_pct)
                rec_out = {
                    "file": rec.get("file", src.stem + ".elf"),
                    "func_addr": func_addr,
                    "path_id": path_id,
                    "cluster_id": cid,
                    "split": split,
                    "tokens": tokens,
                }
                fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                path_id += 1
                stats["kept"] += 1
                stats["tokens_kept"] += len(tokens)
                split_counts[split] += 1
                cluster_sizes_kept[cid] += 1

        if (file_idx + 1) % 200 == 0 or file_idx == len(data["jsonl_files"]) - 1:
            logger.info("  [passage 2] %4d/%d fichiers écrits | conservés: %d",
                        file_idx + 1, len(data["jsonl_files"]), stats["kept"])

    stats["n_clusters"] = len(cluster_sizes_kept)
    stats["paths_in"] = data["n_paths_in"]
    stats["unique_exact"] = len(data["seen_exact"])
    stats["exact_dups"] = data["n_exact_dups"]
    stats["split_train"] = split_counts["train"]
    stats["split_val"] = split_counts["val"]

    # Taux de duplication résiduel : part des chemins conservés appartenant
    # à un cluster de taille > 1 (redondance volontairement plafonnée à K)
    multi = sum(1 for c in cluster_sizes_kept.values() if c > 1)
    stats["clusters_multi_member"] = multi

    stats_path = out_dir / "stats_dedup.json"
    stats_path.write_text(json.dumps(dict(stats), indent=2), encoding="utf-8")
    logger.info("Statistiques → %s", stats_path)
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage B — déduplication anti-leakage : exact + MinHash LSH "
                    "+ excision boilerplate + split group-aware"
    )
    p.add_argument("--data-dir", default="data_v2/",
                   help="Répertoire des shards désucrés (défaut: data_v2/)")
    p.add_argument("--out-dir", default="data_v3/",
                   help="Répertoire de sortie dédupliqué (défaut: data_v3/)")
    p.add_argument("--keep-per-cluster", type=int, default=2,
                   help="Représentants max par cluster LSH, variantes distinctes "
                        "privilégiées (défaut: 2)")
    p.add_argument("--boilerplate-min-families", type=int, default=50,
                   help="Seuil de familles de binaires pour l'excision boilerplate "
                        "(défaut: 50)")
    p.add_argument("--val-pct", type=int, default=5,
                   help="Pourcentage de clusters en validation (défaut: 5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = collect(args)
    clusters = cluster(data)
    keep = select_representatives(clusters, data["path_meta"], args.keep_per_cluster)
    boilerplate = find_boilerplate(data["func_bags"], data["jsonl_files"],
                                   args.boilerplate_min_families)
    stats = write_output(args, data, keep, boilerplate)
    logger.info(
        "Terminé : %d chemins → %d conservés (%.1f%%) | %d clusters | "
        "boilerplate: %d | train/val: %d/%d",
        stats["paths_in"], stats["kept"],
        stats["kept"] / max(stats["paths_in"], 1) * 100,
        stats["n_clusters"], stats["dropped_boilerplate"],
        stats["split_train"], stats["split_val"],
    )
