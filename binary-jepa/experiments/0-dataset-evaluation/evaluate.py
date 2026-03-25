"""
evaluate.py
===========
Diagnostic global du dataset d'entraînement binary-jepa/data/.

Métriques calculées
-------------------
1. Volume brut       — nombre de chemins, tokens, fichiers
2. Distribution des tokens — fréquence, cumul, entropie de Shannon
3. Tokens structurels vs sémantiques — ratio bruit / signal
4. Longueurs des chemins — min / percentiles / max
5. Doublons globaux  — % de chemins identiques toutes fonctions confondues
6. Diversité par fonction — ratio chemins uniques/total, entropie par fonction
7. Diversité de séquence — tokens distincts par chemin, nb de sauts JK_*

Implémentation streaming : les chemins ne sont jamais tous en mémoire
simultanément. Seuls les hash MD5 et les compteurs agrégés sont conservés.

Usage
-----
    python experiments/0-dataset-evaluation/evaluate.py
"""

import collections
import hashlib
import json
import math
import sys
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parents[2] / "data"

# Tokens structurels VEX IR : artefacts architecturaux sans valeur sémantique
# (coercions de type, déclarations de temporaires, hints ABI)
STRUCTURAL_TOKENS = {
    "VEX_WrTmp",
    "VEX_OP_64T", "VEX_OP_32U", "VEX_OP_16U", "VEX_OP_8UT", "VEX_OP_1UT",
    "VEX_OP_32S", "VEX_OP_16S", "VEX_OP_8ST", "VEX_OP_1ST",
    "VEX_OP_F64", "VEX_OP_F32", "VEX_OP_I64", "VEX_OP_I32",
    "VEX_Ist_AbiHint",
}

JUMP_TOKENS = {"JK_BORING", "JK_CALL", "JK_RET", "JK_NODECODE", "JK_SIGTRAP"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_path(tokens: list[str]) -> bytes:
    """Hash MD5 d'un chemin (16 octets). Utilisé pour détecter les doublons."""
    h = hashlib.md5()
    for t in tokens:
        h.update(t.encode())
        h.update(b"\x00")
    return h.digest()


def _percentile(sorted_list: list, p: float) -> float:
    if not sorted_list:
        return 0.0
    idx = int(p / 100 * (len(sorted_list) - 1))
    return sorted_list[idx]


def _entropy(counter: dict) -> float:
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c > 0)


def _progress(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    return f"\r[{bar}] {current}/{total}"


# ── Collecte (streaming) ──────────────────────────────────────────────────────

def collect(data_dir: Path) -> dict:
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    n_files = len(jsonl_files)
    if n_files == 0:
        sys.exit(f"Aucun fichier JSONL dans {data_dir}")

    # Compteurs globaux
    token_counter: collections.Counter = collections.Counter()
    global_hashes: set[bytes] = set()
    path_lengths: list[int] = []

    # Par fonction : {(stem, func_addr): {"n": int, "hashes": set}}
    func_data: dict[tuple, dict] = {}

    # Agrégats sémantiques
    n_total_paths        = 0
    n_structural_tokens  = 0
    n_total_tokens       = 0
    jump_counts: list[int] = []       # nb de sauts JK_* par chemin
    distinct_per_path: list[int] = [] # tokens distincts par chemin

    for file_idx, jsonl in enumerate(jsonl_files):
        print(_progress(file_idx + 1, n_files), end="", flush=True, file=sys.stderr)
        stem = jsonl.stem

        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                tokens    = rec.get("tokens", [])
                func_addr = rec.get("func_addr", "?")
                if not tokens:
                    continue

                ph = _hash_path(tokens)
                key = (stem, func_addr)

                # Volume
                n_total_paths += 1
                n = len(tokens)
                path_lengths.append(n)

                # Tokens
                token_counter.update(tokens)
                n_total_tokens += n
                n_structural_tokens += sum(1 for t in tokens if t in STRUCTURAL_TOKENS)

                # Doublons globaux
                global_hashes.add(ph)

                # Diversité par fonction
                if key not in func_data:
                    func_data[key] = {"n": 0, "hashes": set()}
                func_data[key]["n"]      += 1
                func_data[key]["hashes"].add(ph)

                # Sauts et diversité intra-chemin
                jump_counts.append(sum(1 for t in tokens if t in JUMP_TOKENS))
                distinct_per_path.append(len(set(tokens)))

    print(file=sys.stderr)  # fin de la barre de progression

    # ── Post-traitement ───────────────────────────────────────────────────────

    path_lengths.sort()
    jump_counts.sort()
    distinct_per_path.sort()

    # Ratio uniques par fonction
    unique_ratios = []
    func_entropies = []
    for fd in func_data.values():
        ratio = len(fd["hashes"]) / fd["n"] if fd["n"] > 0 else 0.0
        unique_ratios.append(ratio)

    # Entropie par fonction approximée via le compteur global de tokens
    # (trop coûteux de stocker tous les tokens par fonction)
    global_entropy = _entropy(token_counter)
    unique_ratios.sort()

    # Cumuls top-N
    total_tok = sum(token_counter.values())
    sorted_counts = [c for _, c in token_counter.most_common()]
    cumuls = {}
    s = 0
    for n_tok in [5, 10, 20, 50]:
        s = sum(sorted_counts[:n_tok])
        cumuls[n_tok] = s / total_tok * 100

    return {
        "n_files":             n_files,
        "n_total_paths":       n_total_paths,
        "n_unique_paths":      len(global_hashes),
        "n_total_tokens":      n_total_tokens,
        "n_distinct_tokens":   len(token_counter),
        "n_structural_tokens": n_structural_tokens,
        "token_counter":       token_counter,
        "global_entropy":      global_entropy,
        "cumuls":              cumuls,
        "path_lengths":        path_lengths,
        "jump_counts":         jump_counts,
        "distinct_per_path":   distinct_per_path,
        "n_functions":         len(func_data),
        "unique_ratios":       unique_ratios,
    }


# ── Rapport ───────────────────────────────────────────────────────────────────

def report(d: dict) -> None:
    sep = "─" * 60

    print(f"\n{'═'*60}")
    print(" DIAGNOSTIC DATASET BINARY-JEPA")
    print(f"{'═'*60}\n")

    # 1. Volume
    dup_pct = (1 - d["n_unique_paths"] / d["n_total_paths"]) * 100
    print(f"── 1. VOLUME ──────────────────────────────────────────────")
    print(f"  Fichiers JSONL      : {d['n_files']:>10,}")
    print(f"  Chemins totaux      : {d['n_total_paths']:>10,}")
    print(f"  Chemins uniques     : {d['n_unique_paths']:>10,}  ({100-dup_pct:.1f}%)")
    print(f"  Doublons            : {d['n_total_paths']-d['n_unique_paths']:>10,}  ({dup_pct:.1f}%)")
    print(f"  Tokens totaux       : {d['n_total_tokens']:>10,}")
    print(f"  Tokens distincts    : {d['n_distinct_tokens']:>10,}")
    print(f"  Fonctions uniques   : {d['n_functions']:>10,}")
    print()

    # 2. Distribution des tokens
    print(f"── 2. DISTRIBUTION DES TOKENS ─────────────────────────────")
    print(f"  Entropie de Shannon : {d['global_entropy']:.3f} bits")
    print(f"  Max théorique       : {math.log2(d['n_distinct_tokens']):.3f} bits")
    print(f"  Utilisation vocab   : {d['global_entropy']/math.log2(d['n_distinct_tokens'])*100:.1f}%")
    print()
    for n, pct in d["cumuls"].items():
        print(f"  Top {n:>2} tokens → {pct:5.1f}% du corpus")
    print()
    print(f"  {'Token':<35} {'count':>10}  {'%':>6}")
    print(f"  {'-'*35} {'-'*10}  {'-'*6}")
    total = d["n_total_tokens"]
    for tok, cnt in d["token_counter"].most_common(20):
        print(f"  {tok:<35} {cnt:>10,}  {cnt/total*100:>5.1f}%")
    print()

    # 3. Ratio structurel / sémantique
    struct_pct  = d["n_structural_tokens"] / d["n_total_tokens"] * 100
    sem_pct     = 100 - struct_pct
    print(f"── 3. SIGNAL / BRUIT (tokens structurels VEX) ─────────────")
    print(f"  Tokens structurels (WrTmp, type-casts...) : {struct_pct:5.1f}%")
    print(f"  Tokens sémantiques (ops, mem, flow...)    : {sem_pct:5.1f}%")
    print()

    # 4. Longueurs des chemins
    pl = d["path_lengths"]
    print(f"── 4. LONGUEUR DES CHEMINS ─────────────────────────────────")
    print(f"  min    : {pl[0]:>8}")
    print(f"  p10    : {_percentile(pl, 10):>8.0f}")
    print(f"  p25    : {_percentile(pl, 25):>8.0f}")
    print(f"  médiane: {_percentile(pl, 50):>8.0f}")
    print(f"  p75    : {_percentile(pl, 75):>8.0f}")
    print(f"  p90    : {_percentile(pl, 90):>8.0f}")
    print(f"  p99    : {_percentile(pl, 99):>8.0f}")
    print(f"  max    : {pl[-1]:>8}")
    print()

    # 5. Diversité par fonction
    ur = d["unique_ratios"]
    n  = len(ur)
    all_dup    = sum(1 for r in ur if r <= 0.01)
    all_unique = sum(1 for r in ur if r == 1.0)
    low_div    = sum(1 for r in ur if r < 0.5)
    print(f"── 5. DIVERSITÉ PAR FONCTION ───────────────────────────────")
    print(f"  Fonctions analysées              : {n:>8,}")
    print(f"  Ratio unique/total — p25 / p50 / p75 :")
    print(f"    {_percentile(ur,25):.3f}  /  {_percentile(ur,50):.3f}  /  {_percentile(ur,75):.3f}")
    print(f"  Fonctions quasi-100% doublons    : {all_dup:>8,}  ({all_dup/n*100:.1f}%)")
    print(f"  Fonctions < 50% diversité        : {low_div:>8,}  ({low_div/n*100:.1f}%)")
    print(f"  Fonctions 100% uniques           : {all_unique:>8,}  ({all_unique/n*100:.1f}%)")
    print()

    # 6. Diversité intra-chemin
    jc = d["jump_counts"]
    dp = d["distinct_per_path"]
    print(f"── 6. CARACTÉRISTIQUES DES SÉQUENCES ───────────────────────")
    print(f"  Tokens distincts par chemin (médiane)  : {_percentile(dp,50):.0f}")
    print(f"  Tokens distincts par chemin (p75)      : {_percentile(dp,75):.0f}")
    print(f"  Sauts JK_* par chemin (médiane)        : {_percentile(jc,50):.0f}")
    print(f"  Sauts JK_* par chemin (p75)            : {_percentile(jc,75):.0f}")
    print(f"  Chemins sans aucun saut                : "
          f"{sum(1 for j in jc if j==0):,}  ({sum(1 for j in jc if j==0)/len(jc)*100:.1f}%)")
    print()

    print(f"{'═'*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_file = Path(__file__).parent / "report.txt"
    print(f"Dataset : {DATA_DIR}", file=sys.stderr)
    print(f"Chargement en streaming...", file=sys.stderr)
    data = collect(DATA_DIR)

    import io
    buf = io.StringIO()
    # Redirige print vers buffer + stdout simultanément
    original_stdout = sys.stdout
    sys.stdout = buf
    report(data)
    sys.stdout = original_stdout

    output = buf.getvalue()
    print(output)
    out_file.write_text(output, encoding="utf-8")
    print(f"\nRapport sauvegardé → {out_file}", file=sys.stderr)
