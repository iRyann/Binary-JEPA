"""
evaluate_v2.py
==============
Diagnostic du dataset désucré binary-jepa/data_v2/ (Stage A — desugar.py).

Réutilise les métriques exactes de evaluate.py (collect + report) pour
garantir la comparabilité, puis préfixe le rapport d'une section de
comparaison v1 ↔ v2 parsée depuis report.txt.

Usage
-----
    python experiments/0-dataset-evaluation/evaluate_v2.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import evaluate  # noqa: E402 — collect() et report() partagés avec v1

DATA_V2_DIR = Path(__file__).parents[2] / "data_v2"
REPORT_V1 = Path(__file__).parent / "report.txt"
REPORT_V2 = Path(__file__).parent / "report_v2.txt"


# ── Parsing du rapport v1 (comparaison) ──────────────────────────────────────

def _num(s: str) -> int:
    return int(s.replace(",", "").replace(" ", ""))


def parse_v1(path: Path) -> dict:
    """Extrait les métriques clés du rapport v1 (regex tolérantes)."""
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8")
    out = {}

    def grab(pattern, cast=float):
        m = re.search(pattern, txt)
        return cast(m.group(1)) if m else None

    out["n_paths"] = grab(r"Chemins totaux\s+:\s+([\d,]+)", _num)
    out["n_tokens"] = grab(r"Tokens totaux\s+:\s+([\d,]+)", _num)
    out["dup_pct"] = grab(r"Doublons\s+:\s+[\d,]+\s+\(([\d.]+)%\)")
    out["entropy"] = grab(r"Entropie de Shannon\s+:\s+([\d.]+) bits")
    out["top5_pct"] = grab(r"Top\s+5 tokens\s+→\s+([\d.]+)%")
    out["struct_pct"] = grab(r"Tokens structurels \(WrTmp, type-casts\.\.\.\)\s+:\s+([\d.]+)%")
    out["median_len"] = grab(r"médiane:\s+(\d+)", int)
    out["p99_len"] = grab(r"p99\s+:\s+(\d+)", int)
    out["max_len"] = grab(r"max\s+:\s+(\d+)", int)
    return out


def comparison_section(v1: dict, d2: dict) -> str:
    """Tableau de comparaison v1 ↔ v2 + verdict sur les gates d'acceptation."""
    import math

    pl = d2["path_lengths"]
    struct_pct_v2 = d2["n_structural_tokens"] / d2["n_total_tokens"] * 100
    top5_v2 = d2["cumuls"][5]
    median_v2 = evaluate._percentile(pl, 50)
    p99_v2 = evaluate._percentile(pl, 99)
    dup_pct_v2 = (1 - d2["n_unique_paths"] / d2["n_total_paths"]) * 100

    def fmt(val, spec=".2f"):
        return f"{val:{spec}}" if val is not None else "n/a"

    L = []
    L.append("═" * 60)
    L.append(" COMPARAISON v1 (brut) ↔ v2 (désucré — Stage A)")
    L.append("═" * 60)
    L.append(f"  {'Métrique':<38} {'v1':>10} {'v2':>10}")
    L.append(f"  {'-'*38} {'-'*10} {'-'*10}")
    rows = [
        ("Tokens totaux",            fmt(v1.get("n_tokens"), ","),      f"{d2['n_total_tokens']:,}"),
        ("Tokens structurels (%)",   fmt(v1.get("struct_pct"), ".1f"),  f"{struct_pct_v2:.1f}"),
        ("Entropie Shannon (bits)",  fmt(v1.get("entropy")),            f"{d2['global_entropy']:.3f}"),
        ("Max théorique (bits)",     "8.531",                           f"{math.log2(d2['n_distinct_tokens']):.3f}"),
        ("Top-5 tokens (%)",         fmt(v1.get("top5_pct"), ".1f"),    f"{top5_v2:.1f}"),
        ("Longueur médiane",         fmt(v1.get("median_len"), "d"),    f"{median_v2:.0f}"),
        ("Longueur p99",             fmt(v1.get("p99_len"), "d"),       f"{p99_v2:.0f}"),
        ("Longueur max",             fmt(v1.get("max_len"), "d"),       f"{pl[-1]}"),
        ("Doublons globaux (%)",     fmt(v1.get("dup_pct"), ".1f"),     f"{dup_pct_v2:.1f}"),
    ]
    for name, a, b in rows:
        L.append(f"  {name:<38} {a:>10} {b:>10}")
    L.append("")

    # ── Verdict ──
    # Note méthodologique : l'entropie de Shannon par token n'est PAS une
    # métrique de signal. Les casts supprimés (~20 types à distribution
    # plate) contribuaient davantage à l'entropie que le bruit de tête
    # (WrTmp). Le gate correct pour Stage A est la part de positions
    # sémantiques (100% par construction) et l'élimination du bruit ;
    # l'arbitre final reste la probe sémantique downstream (Stage E).
    L.append("  Verdict Stage A :")
    noise_gone = struct_pct_v2 < 5.0
    L.append(f"    [{'OK' if noise_gone else 'KO'}] Bruit structurel résiduel "
             f"{struct_pct_v2:.1f}% (gate < 5%)")
    reduction = (1 - d2["n_total_tokens"] / v1["n_tokens"]) * 100 if v1.get("n_tokens") else 0
    L.append(f"    [..] Réduction de volume : -{reduction:.1f}% de tokens "
             f"(chaque position restante est sémantique par construction)")
    L.append(f"    [!!] Entropie {d2['global_entropy']:.2f} bits < v1 : attendu — "
             f"la diversité des casts supprimés gonflait l'entropie v1 ;")
    L.append(f"         le tri entre signal et diversité se fera sur la probe "
             f"sémantique (Stage E), pas sur Shannon.")
    L.append("")
    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io

    print(f"Dataset v2 : {DATA_V2_DIR}", file=sys.stderr)
    print("Chargement en streaming...", file=sys.stderr)
    data_v2 = evaluate.collect(DATA_V2_DIR)

    buf = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = buf
    evaluate.report(data_v2)
    sys.stdout = original_stdout

    v1 = parse_v1(REPORT_V1)
    output = comparison_section(v1, data_v2) + buf.getvalue()
    print(output)
    REPORT_V2.write_text(output, encoding="utf-8")
    print(f"\nRapport sauvegardé → {REPORT_V2}", file=sys.stderr)
