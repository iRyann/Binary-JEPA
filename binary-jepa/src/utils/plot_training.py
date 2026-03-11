"""
plot_training.py  –  Training report generator
Usage:
    python plot_training.py training_log.csv
    python plot_training.py training_log.csv --out report.png --smoothing 0.9
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ── Aesthetic config ──────────────────────────────────────────────────────────
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT_MAIN = "#e6edf3"
TEXT_DIM = "#8b949e"
ACCENT = "#58a6ff"
ACCENT2 = "#3fb950"
ACCENT3 = "#f78166"
ACCENT4 = "#d2a8ff"
GRID_COLOR = "#21262d"

PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#ffa657", "#79c0ff"]

matplotlib.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_DIM,
        "axes.titlecolor": TEXT_MAIN,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "text.color": TEXT_MAIN,
        "legend.facecolor": PANEL,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": TEXT_DIM,
        "font.family": "monospace",
        "lines.linewidth": 1.6,
    }
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average for smoothing."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def stat_box(ax, stats: dict[str, str]):
    """Render a key/value stats panel inside an axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(stats)
    for i, (k, v) in enumerate(stats.items()):
        y = 1 - (i + 0.5) / n
        ax.text(0.05, y, k, color=TEXT_DIM, fontsize=9, va="center", ha="left")
        ax.text(
            0.95,
            y,
            v,
            color=TEXT_MAIN,
            fontsize=9,
            va="center",
            ha="right",
            fontweight="bold",
        )
        if i < n - 1:
            ax.axhline(1 - (i + 1) / n, color=BORDER, linewidth=0.5)


def plot_metric(ax, epochs, raw, smooth, color, label, is_loss=False):
    """Plot a single metric with optional smoothed overlay."""
    ax.plot(epochs, raw, color=color, alpha=0.25, linewidth=1.0, label="_raw")
    ax.plot(epochs, smooth, color=color, linewidth=2.0, label=label)

    best_idx = int(np.argmin(smooth)) if is_loss else int(np.argmax(smooth))
    bx, by = epochs[best_idx], smooth[best_idx]
    ax.scatter([bx], [by], color=color, s=60, zorder=5)
    ax.annotate(
        f"{'min' if is_loss else 'max'} {by:.5f}",
        xy=(bx, by),
        xytext=(12, -12 if is_loss else 12),
        textcoords="offset points",
        color=color,
        fontsize=7.5,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )
    ax.set_ylabel(label, fontsize=9)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=8)


# ── Main ──────────────────────────────────────────────────────────────────────
def build_report(csv_path: str, out_path: str, smoothing: float):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    if "epoch" not in df.columns:
        sys.exit("CSV must contain an 'epoch' column.")

    epochs = df["epoch"].to_numpy()
    metric_cols = [c for c in df.columns if c != "epoch"]

    if not metric_cols:
        sys.exit("No metric columns found beyond 'epoch'.")

    # ── Classify columns ──────────────────────────────────────────────────────
    loss_cols = [c for c in metric_cols if "loss" in c]
    other_cols = [c for c in metric_cols if "loss" not in c]
    ordered = loss_cols + other_cols  # losses first

    n_metrics = len(ordered)
    n_cols = min(n_metrics, 2)
    n_rows = (n_metrics + 1) // 2  # metric plot rows

    # ── Figure layout ─────────────────────────────────────────────────────────
    total_rows = n_rows + 1  # +1 for stats row at bottom
    fig = plt.figure(figsize=(14, 4 * n_rows + 2.5))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        total_rows,
        n_cols + 1,
        figure=fig,
        hspace=0.45,
        wspace=0.15,
        width_ratios=[1] * n_cols + [0.32],
    )

    # ── Header ────────────────────────────────────────────────────────────────
    run_name = Path(csv_path).stem
    fig.text(
        0.03,
        0.985,
        "TRAINING REPORT",
        fontsize=13,
        fontweight="bold",
        color=TEXT_MAIN,
        va="top",
        family="monospace",
    )
    fig.text(
        0.03, 0.962, run_name, fontsize=9, color=TEXT_DIM, va="top", family="monospace"
    )
    fig.text(
        0.97,
        0.985,
        f"epochs  {int(epochs[0])} → {int(epochs[-1])}",
        fontsize=9,
        color=TEXT_DIM,
        va="top",
        ha="right",
        family="monospace",
    )

    # ── Metric plots ─────────────────────────────────────────────────────────
    for idx, col in enumerate(ordered):
        row = idx // n_cols
        col_pos = idx % n_cols
        ax = fig.add_subplot(gs[row, col_pos])

        raw = df[col].to_numpy(dtype=float)
        smooth = ema(raw, smoothing)
        color = PALETTE[idx % len(PALETTE)]
        is_loss = "loss" in col

        # shade area under curve
        ax.fill_between(epochs, smooth, alpha=0.07, color=color)
        plot_metric(ax, epochs, raw, smooth, color, col, is_loss=is_loss)

        ax.set_xlabel("epoch", fontsize=8)
        ax.set_title(
            col.replace("_", " ").upper(), fontsize=9, color=TEXT_DIM, pad=6, loc="left"
        )

        # convergence line (last 10% of training plateaus)
        plateau_start = int(len(epochs) * 0.9)
        if plateau_start > 0:
            ax.axvline(
                epochs[plateau_start],
                color=BORDER,
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
            )
            ax.text(
                epochs[plateau_start] + (epochs[-1] - epochs[0]) * 0.01,
                ax.get_ylim()[1] * 0.98,
                "plateau",
                color=TEXT_DIM,
                fontsize=6.5,
                va="top",
                alpha=0.7,
            )

    # ── Stats panel (right column, spans all metric rows) ────────────────────
    ax_stats = fig.add_subplot(gs[0:n_rows, n_cols])
    ax_stats.set_facecolor(PANEL)
    for spine in ax_stats.spines.values():
        spine.set_edgecolor(BORDER)

    stats: dict[str, str] = {"run": Path(csv_path).stem[:16]}
    stats["epochs"] = str(int(epochs[-1]))
    stats["samples"] = str(len(epochs))

    for col in ordered:
        raw = df[col].to_numpy(dtype=float)
        is_loss = "loss" in col
        best_val = raw.min() if is_loss else raw.max()
        best_ep = int(epochs[np.argmin(raw) if is_loss else np.argmax(raw)])
        stats[f"{'↓' if is_loss else '↑'} {col[:10]}"] = f"{best_val:.5f}"
        stats[f"  @ epoch"] = str(best_ep)
        stats[f"  final"] = f"{raw[-1]:.5f}"
        stats["──────────"] = ""

    stat_box(ax_stats, stats)
    ax_stats.set_title("STATS", fontsize=8, color=TEXT_DIM, pad=6, loc="left")

    # ── Δloss per-epoch bar (bottom row, spans full width) ────────────────────
    if "loss" in df.columns or loss_cols:
        loss_col = "loss" if "loss" in df.columns else loss_cols[0]
        raw_loss = df[loss_col].to_numpy(dtype=float)
        delta = np.diff(raw_loss, prepend=raw_loss[0])

        ax_d = fig.add_subplot(gs[n_rows, :])
        colors = [ACCENT3 if d > 0 else ACCENT2 for d in delta]
        ax_d.bar(
            epochs,
            delta,
            color=colors,
            width=max(1, (epochs[-1] - epochs[0]) / len(epochs) * 0.8),
            alpha=0.75,
        )
        ax_d.axhline(0, color=BORDER, linewidth=0.8)
        ax_d.set_xlabel("epoch", fontsize=8)
        ax_d.set_title(
            f"Δ {loss_col} per epoch  (red = regression)",
            fontsize=9,
            color=TEXT_DIM,
            pad=6,
            loc="left",
        )
        ax_d.tick_params(labelsize=8)
        ax_d.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.97,
        0.012,
        f"smoothing α={smoothing}  |  {csv_path}",
        fontsize=7,
        color=TEXT_DIM,
        ha="right",
        va="bottom",
        family="monospace",
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[✓] Report saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a training report from a CSV log."
    )
    parser.add_argument("csv", type=str, help="Path to the CSV log file")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: <csv_stem>_report.png)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.85,
        help="EMA smoothing factor 0–1 (default: 0.85)",
    )
    args = parser.parse_args()

    out = args.out or str(Path(args.csv).with_name(Path(args.csv).stem + "_report.png"))
    build_report(args.csv, out, args.smoothing)
