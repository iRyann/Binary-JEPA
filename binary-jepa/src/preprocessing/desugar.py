"""
desugar.py
==========
Stage A du pipeline data-v2 : "desucrage" du VEX IR.

Problème traité (diagnostic experiments/0-dataset-evaluation/report.txt) :
    - Entropie de 3.5 bits / 8.5 théoriques (41% d'utilisation du vocabulaire)
    - Top-5 tokens = 67.8% du corpus, dominé par VEX_REG_WRITE (27.4%)
      et VEX_WrTmp (15.3%)
    - Les tokens de scaffolding VEX (temporaires, coercions de type) noient
      le signal sémantique (loi de Zipf, cf. Hindle et al. "On the
      Naturalness of Software")

Grammaire de réécriture (appliquée par chemin, gauche → droite) :
    1. DROP    — VEX_WrTmp + bookkeeping Ist_* (AbiHint optionnel) :
                 bruit syntaxique pur, aucune identité de data-flow
                 (le tokenizer a déjà effacé les indices de temporaires)
    2. FOLD    — un op sémantique suivi d'une ou plusieurs coercions
                 absorbe la largeur : VEX_OP_ADD + VEX_OP_64T → VEX_OP_ADD.64
                 (la largeur est une information sémantique : arithmétique
                 de pointeurs vs opérations byte). Désactivable via
                 --no-fold-width (les casts sont alors simplement supprimés)
    3. DROP    — les casts non précédés d'un op sémantique sont supprimés
    4. RUN-CAP — les runs de tokens identiques (REG_WRITE ×5 en préambule
                 de fonction) sont plafonnés à --run-cap occurrences

Le schéma JSONL est inchangé : {"file", "func_addr", "tokens"}.
Implémentation streaming : un seul passage sur les shards, écriture
incrémentale, seuls des compteurs agrégés sont conservés en RAM.

Usage :
    python -m src.preprocessing.desugar \
        --data-dir data/ --out-dir data_v2/ \
        --vocab-out vocab_v2.json --stats-out data_v2/stats.json
"""

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# GRAMMAIRE DE RÉÉCRITURE
# ══════════════════════════════════════════════════════════════════════════════

# Tokens spéciaux du vocabulaire (IDs 0-2 hardcodés dans le modèle PyTorch)
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<MASK>"]

# Bruit pur : déclaration de temporaire (sans identité → zéro data-flow)
DROP_ALWAYS = {"VEX_WrTmp"}

# Bookkeeping VEX Ist_* sans contenu sémantique pour le hachage.
# VEX_Ist_CAS est conservé : compare-and-swap = opération mémoire sémantique.
DROP_BOOKKEEPING = {"VEX_Ist_Dirty", "VEX_Ist_MBE", "VEX_Ist_PutI"}

# AbiHint est discutable (hint de convention d'appel) → flag dédié
ABIHINT_TOKEN = "VEX_Ist_AbiHint"

# Coercions de type VEX → tag de largeur compact.
# Nomenclature issue du tokenizer (noms Iop_* tronqués) :
#   64T/32T/16T   → Iop_64to*, Iop_32to*, Iop_16to*   (troncatures)
#   1UT/8UT/16U/32U/64U → Iop_*Uto*                   (zero-extend)
#   8ST/16S/32S   → Iop_*Sto*                          (sign-extend)
#   *H / 8HL      → accès high-half / vecteurs         (SIMD)
#   F32/F64/I32/I64 → conversions flottant/entier
CAST_WIDTH = {
    "VEX_OP_64T": "64",
    "VEX_OP_32T": "32",
    "VEX_OP_16T": "16",
    "VEX_OP_64U": "u64",
    "VEX_OP_32U": "u32",
    "VEX_OP_16U": "u16",
    "VEX_OP_8UT": "u8",
    "VEX_OP_1UT": "u1",
    "VEX_OP_32S": "s32",
    "VEX_OP_16S": "s16",
    "VEX_OP_8ST": "s8",
    "VEX_OP_64H": "h64",
    "VEX_OP_32H": "h32",
    "VEX_OP_16H": "h16",
    "VEX_OP_8HL": "h8",
    "VEX_OP_128": "128",
    "VEX_OP_64X": "x64",
    "VEX_OP_V12": "v128",
    "VEX_OP_V25": "v256",
    "VEX_OP_F32": "f32",
    "VEX_OP_F64": "f64",
    "VEX_OP_I32": "i32",
    "VEX_OP_I64": "i64",
}

# Tokens éligibles au plafonnement de runs (préambules de fonction,
# context-saves : information quasi nulle après la 1re occurrence)
RUN_CAP_TOKENS = {"VEX_REG_READ", "VEX_REG_WRITE", "VEX_CONST"}


def _is_semantic_op(token: str) -> bool:
    """Op arithmétique/logique (pas une coercion de type)."""
    return token.startswith("VEX_OP_") and token not in CAST_WIDTH


# ══════════════════════════════════════════════════════════════════════════════
# RÉÉCRITURE D'UN CHEMIN
# ══════════════════════════════════════════════════════════════════════════════

def desugar_path(
    tokens: list[str],
    drop_set: set[str],
    fold_width: bool = True,
    run_cap: int = 2,
) -> list[str]:
    """Applique la grammaire de réécriture à un chemin de tokens.

    Args:
        tokens:     séquence brute de tokens VEX/API
        drop_set:   tokens à supprimer inconditionnellement
        fold_width: si True, absorbe les casts dans l'op précédent
                    (VEX_OP_ADD + VEX_OP_64T → VEX_OP_ADD.64) ;
                    si False, supprime simplement les casts
        run_cap:    longueur max d'un run de tokens identiques
                    (0 = désactivé), appliqué à RUN_CAP_TOKENS

    Returns:
        Séquence désucrée (peut être vide si le chemin était 100% bruit).
    """
    out: list[str] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]

        # 1. Suppression du bruit pur et du bookkeeping
        if tok in drop_set:
            i += 1
            continue

        # 2. Op sémantique : absorption des casts qui le suivent
        if _is_semantic_op(tok):
            j = i + 1
            last_cast = None
            while j < n and tokens[j] in CAST_WIDTH:
                last_cast = tokens[j]
                j += 1
            if last_cast is not None:
                if fold_width:
                    # La dernière coercion de la chaîne donne le type effectif
                    out.append(f"{tok}.{CAST_WIDTH[last_cast]}")
                # fold_width=False → on garde l'op nu, les casts sont absorbés
                else:
                    out.append(tok)
                i = j
            else:
                out.append(tok)
                i += 1
            continue

        # 3. Cast orphelin (non précédé d'un op) → suppression
        if tok in CAST_WIDTH:
            i += 1
            continue

        # 4. Token sémantique conservé
        out.append(tok)
        i += 1

    # 5. Plafonnement des runs (REG_WRITE ×5 → REG_WRITE ×2)
    if run_cap > 0:
        capped: list[str] = []
        run_len = 0
        prev = None
        for tok in out:
            if tok == prev and tok in RUN_CAP_TOKENS:
                run_len += 1
                if run_len > run_cap:
                    continue
            else:
                run_len = 1
                prev = tok
            capped.append(tok)
        out = capped

    return out


# ══════════════════════════════════════════════════════════════════════════════
# PASSAGE STREAMING SUR LE CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(sorted_list: list[int], p: float) -> float:
    if not sorted_list:
        return 0.0
    idx = int(p / 100 * (len(sorted_list) - 1))
    return sorted_list[idx]


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drop_set = set(DROP_ALWAYS) | set(DROP_BOOKKEEPING)
    if not args.keep_abihint:
        drop_set.add(ABIHINT_TOKEN)

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"Aucun fichier JSONL dans {data_dir}")
    logger.info("%d fichiers à traiter → %s", len(jsonl_files), out_dir)

    # Compteurs agrégés (avant / après)
    stats = {
        "config": {
            "fold_width": args.fold_width,
            "run_cap": args.run_cap,
            "keep_abihint": args.keep_abihint,
            "drop_set": sorted(drop_set),
        },
        "n_files": 0,
        "n_paths_in": 0,
        "n_paths_out": 0,
        "n_paths_dropped_empty": 0,
        "n_tokens_in": 0,
        "n_tokens_out": 0,
    }
    counter_in: Counter = Counter()
    counter_out: Counter = Counter()
    lengths_in: list[int] = []
    lengths_out: list[int] = []

    for file_idx, src_path in enumerate(jsonl_files):
        dst_path = out_dir / src_path.name
        n_kept = 0
        with src_path.open("r", encoding="utf-8") as fin, \
             dst_path.open("w", encoding="utf-8") as fout:
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

                stats["n_paths_in"] += 1
                stats["n_tokens_in"] += len(tokens)
                counter_in.update(tokens)
                lengths_in.append(len(tokens))

                new_tokens = desugar_path(
                    tokens, drop_set,
                    fold_width=args.fold_width,
                    run_cap=args.run_cap,
                )
                if not new_tokens:
                    stats["n_paths_dropped_empty"] += 1
                    continue

                rec["tokens"] = new_tokens
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                stats["n_paths_out"] += 1
                stats["n_tokens_out"] += len(new_tokens)
                counter_out.update(new_tokens)
                lengths_out.append(len(new_tokens))
                n_kept += 1

        stats["n_files"] += 1
        if (file_idx + 1) % 50 == 0 or file_idx == len(jsonl_files) - 1:
            logger.info(
                "  [%4d/%4d] %-28s chemins conservés: %d",
                file_idx + 1, len(jsonl_files), src_path.name, n_kept,
            )

    # ── Vocabulaire v2 : spéciaux d'abord (IDs 0-2 figés), puis fréquence desc ──
    vocab_v2 = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tok, _ in sorted(counter_out.items(), key=lambda kv: (-kv[1], kv[0])):
        if tok not in vocab_v2:
            vocab_v2[tok] = len(vocab_v2)
    vocab_out = Path(args.vocab_out)
    vocab_out.write_text(
        json.dumps(vocab_v2, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Vocabulaire v2 : %d tokens → %s", len(vocab_v2), vocab_out)

    # ── Statistiques récapitulatives ──
    lengths_in.sort()
    lengths_out.sort()
    n_in = stats["n_tokens_in"]
    n_out = stats["n_tokens_out"]

    def _entropy(counter: Counter) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0
        return -sum(
            (c / total) * math.log2(c / total)
            for c in counter.values() if c > 0
        )

    top5_in = sum(c for _, c in counter_in.most_common(5)) / max(n_in, 1) * 100
    top5_out = sum(c for _, c in counter_out.most_common(5)) / max(n_out, 1) * 100
    stats["summary"] = {
        "entropy_in": round(_entropy(counter_in), 3),
        "entropy_out": round(_entropy(counter_out), 3),
        "top5_pct_in": round(top5_in, 1),
        "top5_pct_out": round(top5_out, 1),
        "n_distinct_tokens_out": len(counter_out),
        "median_len_in": _percentile(lengths_in, 50),
        "median_len_out": _percentile(lengths_out, 50),
        "p99_len_out": _percentile(lengths_out, 99),
        "max_len_out": lengths_out[-1] if lengths_out else 0,
        "token_reduction_pct": round((1 - n_out / max(n_in, 1)) * 100, 1),
    }
    stats["top_tokens_out"] = counter_out.most_common(30)

    stats_path = Path(args.stats_out)
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Statistiques → %s", stats_path)

    s = stats["summary"]
    logger.info(
        "Terminé : %d → %d tokens (-%.1f%%) | entropie %.2f → %.2f bits | "
        "top-5 %.1f%% → %.1f%% | médiane %d → %d tokens",
        n_in, n_out, s["token_reduction_pct"],
        s["entropy_in"], s["entropy_out"],
        s["top5_pct_in"], s["top5_pct_out"],
        s["median_len_in"], s["median_len_out"],
    )
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage A — désucrage du VEX IR (suppression du bruit "
                    "structurel, folding des largeurs, plafonnement des runs)"
    )
    p.add_argument("--data-dir", default="data/",
                   help="Répertoire des shards JSONL bruts (défaut: data/)")
    p.add_argument("--out-dir", default="data_v2/",
                   help="Répertoire de sortie des shards désucrés (défaut: data_v2/)")
    p.add_argument("--vocab-out", default="vocab_v2.json",
                   help="Fichier vocabulaire v2 généré (défaut: vocab_v2.json)")
    p.add_argument("--stats-out", default=None,
                   help="Fichier de statistiques JSON (défaut: <out-dir>/stats.json)")
    p.add_argument("--fold-width", dest="fold_width", action="store_true",
                   default=True,
                   help="Absorbe les casts dans l'op précédent (défaut: activé)")
    p.add_argument("--no-fold-width", dest="fold_width", action="store_false",
                   help="Supprime les casts sans folding")
    p.add_argument("--run-cap", type=int, default=2,
                   help="Longueur max d'un run de tokens identiques "
                        "(REG_READ/REG_WRITE/CONST, 0 = désactivé, défaut: 2)")
    p.add_argument("--keep-abihint", action="store_true",
                   help="Conserve VEX_Ist_AbiHint (hint de convention d'appel)")
    args = p.parse_args()
    if args.stats_out is None:
        args.stats_out = str(Path(args.out_dir) / "stats.json")
    return args


if __name__ == "__main__":
    run(parse_args())
