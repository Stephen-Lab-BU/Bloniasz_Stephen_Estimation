#!/usr/bin/env python3
"""
Generate the supplemental Figure 3 CVLL line plot.

The script reads cached Figure 3 CVLL metrics and plots trial-level
cross-validated likelihoods for specparam, multiplicative SL_specdecomp, and
additive SL_specdecomp.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns



PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# ---------------------- Style (match 03_paper_figure_3_known_ground_truth_decomposition_cvll.py) ----------------------
mpl.rcParams.update({
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 1.3,
})
sns.set_style("white")


# ---------------------- Methods / labels / palette ----------------------
METHOD_KEYS = ["specparam", "SL_specdecomp_multiplicative", "SL_specdecomp_additive"]

METHOD_LABELS = {
    "specparam": "specparam",
    "SL_specdecomp_additive": "SL_SD (Additive)",
    "SL_specdecomp_multiplicative": "SL_SD (Multiplicative)",
}

PLOT_METHOD_KEYS = ["specparam", "SL_specdecomp_multiplicative", "SL_specdecomp_additive"]
PLOT_ORDER_DISPLAY = [METHOD_LABELS[m] for m in PLOT_METHOD_KEYS]

deep = sns.color_palette("deep", n_colors=3)
PALETTE = {
    "specparam": deep[0],               # blue
    "SL_SD (Multiplicative)": deep[2],  # green
    "SL_SD (Additive)": deep[1],        # orange
}



DEFAULT_IN_DIR = str(PROJECT_ROOT / 'Output' / 'Results' / 'FiguresIntermediate' / 'Figure_3_CV')


# ---------------------- IO helpers ----------------------
def _pick_latest(prefix: str, directory: Path, suffix: str) -> Optional[Path]:
    """Pick the latest file (by mtime) that matches: <prefix>*<suffix> in directory."""
    cands = sorted(directory.glob(f"{prefix}*{suffix}"))
    if not cands:
        return None
    cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _load_cv_metrics(in_dir: Path, sim_mode: str) -> pd.DataFrame:
    """
    Load metrics CSV for a given sim_mode ("additive" or "multiplicative").

    Expected filename from 03_paper_figure_3_known_ground_truth_decomposition_cvll.py:
      Figure_3_CV_<mode>_v1.metrics.csv

    If not found, falls back to the latest matching version.
    """
    base_prefix = f"Figure_3_CV_{sim_mode}_"
    metrics = in_dir / f"{base_prefix}v1.metrics.csv"
    if not metrics.exists():
        metrics = _pick_latest(prefix=base_prefix, directory=in_dir, suffix=".metrics.csv")
    if metrics is None or (not metrics.exists()):
        raise FileNotFoundError(
            f"Could not find metrics CSV for mode='{sim_mode}' in {in_dir}. "
            f"Expected something like '{base_prefix}v1.metrics.csv'."
        )

    df = pd.read_csv(metrics)

    required = {"trial", "method", "cvll_gamma_mt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Metrics CSV '{metrics.name}' is missing required columns: {sorted(missing)}. "
            "Did you run figure_3_cv.py with CV enabled (cvll_gamma_mt)?"
        )

    # Clean and keep only relevant methods
    df = df.copy()
    df["method"] = df["method"].astype(str)
    df = df[df["method"].isin(METHOD_KEYS)].copy()

    # Remove non-finite CVLL
    df["cvll_gamma_mt"] = pd.to_numeric(df["cvll_gamma_mt"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _trial_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long df to trial x method table of CVLL, dropping incomplete trials."""
    tab = df.pivot_table(index="trial", columns="method", values="cvll_gamma_mt", aggfunc="mean")
    tab = tab.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    # enforce column order
    for m in PLOT_METHOD_KEYS:
        if m not in tab.columns:
            tab[m] = np.nan
    tab = tab[PLOT_METHOD_KEYS].dropna(axis=0, how="any")
    return tab


# ---------------------- Plotting ----------------------
def _spaghetti_panel(
    ax: plt.Axes,
    tab: pd.DataFrame,
    title: str,
    max_trials: Optional[int] = None,
    x_jitter: float = 0.018,
    line_alpha: float = 0.22,
    point_alpha: float = 0.80,
    point_size: float = 16.0,
    star_size: float = 70.0,
) -> None:
    """
    Draw a single panel: each row in tab is a trial, with 3 points connected.
    Stars mark the highest CVLL within each trial (row-wise max across the 3 models).
    """
    if tab.shape[0] == 0:
        ax.text(0.5, 0.5, "No finite CVLL trials found", ha="center", va="center")
        ax.axis("off")
        return

    # Optionally subsample trials (for readability on huge N)
    if max_trials is not None and tab.shape[0] > int(max_trials):
        tab = tab.sample(n=int(max_trials), random_state=0)

    # --- Tighter x spacing (columns closer together) ---
    x_step = 0.4  # <--- smaller than 1.0 compresses spacing
    xs = np.arange(len(PLOT_METHOD_KEYS), dtype=float) * x_step

    # Precompute jitter per trial for consistent x offsets across the 3 points
    rng = np.random.default_rng(0)
    jit = rng.uniform(-x_jitter, x_jitter, size=(tab.shape[0], 1))

    y = tab.to_numpy(dtype=float)              # (n_trials, 3)
    x = xs[np.newaxis, :] + jit                # (n_trials, 3)

    # Lines (gray)
    for i in range(y.shape[0]):
        ax.plot(x[i, :], y[i, :], color="0.4", alpha=line_alpha, zorder=1)

    # Points by method color (circles)
    for j, m in enumerate(PLOT_METHOD_KEYS):
        lab = METHOD_LABELS[m]
        ax.scatter(
            x[:, j],
            y[:, j],
            s=point_size,
            alpha=point_alpha,
            marker="o",
            label=lab,
            color=PALETTE[lab],
            edgecolor="none",
            zorder=3,
        )

    # --- Star overlay for the highest CVLL within each trial ---
    # If ties occur, we star all tied maxima (rare but safe).
    row_max = np.nanmax(y, axis=1, keepdims=True)
    is_max = np.isclose(y, row_max, rtol=0.0, atol=0.0)  # exact match
    for j, m in enumerate(PLOT_METHOD_KEYS):
        lab = METHOD_LABELS[m]
        mask = is_max[:, j]
        if np.any(mask):
            ax.scatter(
                x[mask, j],
                y[mask, j],
                s=star_size,
                marker="*",
                color=PALETTE[lab],
                edgecolor="k",
                linewidths=0.5,
                alpha=0.95,
                zorder=4,
                label=None,  # no extra legend entry
            )

    ax.set_title(f"{title} (N={tab.shape[0]})", pad=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS[m] for m in PLOT_METHOD_KEYS], rotation=20, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("CVLL (Gamma MT)")
    sns.despine(ax=ax, top=True, right=True)
    ax.minorticks_off()

    # Keep x-limits snug with compressed spacing
    pad = 0.28
    ax.set_xlim(xs[0] - pad, xs[-1] + pad)


def build_figure(
    in_dir: str,
    out_dir: Optional[str] = None,
    max_trials: Optional[int] = None,
    sharey: bool = False,
) -> Tuple[str, str]:
    in_dir_p = Path(os.path.expanduser(in_dir)).resolve()
    if out_dir is None:
        out_dir_p = in_dir_p
    else:
        out_dir_p = Path(os.path.expanduser(out_dir)).resolve()
        out_dir_p.mkdir(parents=True, exist_ok=True)

    # Load both regimes
    df_add = _load_cv_metrics(in_dir_p, "additive")
    df_mul = _load_cv_metrics(in_dir_p, "multiplicative")

    tab_add = _trial_table(df_add)
    tab_mul = _trial_table(df_mul)

    # --- Smaller overall figure ---
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.2, 5.4),
        sharey=bool(sharey),
        constrained_layout=True,
    )

    _spaghetti_panel(
        axes[0],
        tab_add,
        title="Additive ground truth",
        max_trials=max_trials,
    )
    _spaghetti_panel(
        axes[1],
        tab_mul,
        title="Multiplicative ground truth",
        max_trials=max_trials,
    )

    # Legend: one shared legend (right of figure)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles[:3],
            labels[:3],
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

    fig.suptitle("Figure 3 (aux): Trial-wise CVLL across models", y=1.02, fontsize=16)

    out_png = str(out_dir_p / "Figure_3_CVLL_Lines.png")
    out_svg = str(out_dir_p / "Figure_3_CVLL_Lines.svg")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_svg


# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, default=DEFAULT_IN_DIR, help="Directory containing Figure_3_CV_* caches.")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (defaults to --in-dir).")
    ap.add_argument("--max-trials", type=int, default=None, help="Optional: plot at most this many random trials (for readability).")
    ap.add_argument("--sharey", action="store_true", help="Share y-axis across the two subplots.")
    args = ap.parse_args()

    out_png, out_svg = build_figure(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        max_trials=args.max_trials,
        sharey=bool(args.sharey),
    )
    print(f"[saved] {out_png}\n[saved] {out_svg}")


if __name__ == "__main__":
    main()
