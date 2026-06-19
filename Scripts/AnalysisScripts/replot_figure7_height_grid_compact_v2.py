#!/usr/bin/env python3
"""
Render the compact supplemental height-grid benchmark from cached outputs.

The script reads the saved metrics table and component cache, then writes the
compact manuscript version of the height-grid figure. No simulation or model
fitting is performed.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
from typing import Dict, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# ---------------------- Style ----------------------
mpl.rcParams.update({
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
})
sns.set_style("white")


# ---------------------- Constants ----------------------
DEFAULT_OUT_DIR = os.path.expanduser(
    "CHANGE_THIS_ROOT_TO_PATH/Bloniasz_Stephen_Estimation/Output/Results/FiguresIntermediate/Figure_7_Fig3GroundTruth_HeightGrid"
)

ANALYSIS_FRANGE = (1.0, 200.0)
SLOPE_BAND = (40.0, 60.0)
DEFAULT_TRUE_CF_HZ = 8.0
DEFAULT_TRUE_KNEE_RAW = 60.0
DEFAULT_TRUE_EXPONENT = 2.0

# Cache arrays in Figure_7_fig3truth_row1_components.npz were written in this order
CACHE_METHOD_KEYS = ["specparam", "SL_SD_additive", "SL_SD_specparam"]

# Display order used here.
PLOT_METHOD_ORDER = ["specparam", "SL_SD_specparam", "SL_SD_additive"]

METHOD_LABELS = {
    "specparam": "specparam",
    "SL_SD_specparam": "SL_specdecomp\n(Multiplicative)",
    "SL_SD_additive": "SL_specdecomp\n(Additive)",
}

# Aliases support if the metrics CSV came from a related Figure 3 script.
METHOD_ALIASES = {
    "SL_specdecomp_additive": "SL_SD_additive",
    "SL_specdecomp_multiplicative": "SL_SD_specparam",
    "SL_specdecomp (Additive)": "SL_SD_additive",
    "SL_specdecomp (Multiplicative)": "SL_SD_specparam",
    "SL_SD (Additive)": "SL_SD_additive",
    "SL_SD (Multiplicative)": "SL_SD_specparam",
}

# Figure 3 seaborn-deep convention:
# specparam=blue, additive=orange, multiplicative=green.
# Since the specified display order is specparam, multiplicative, additive,
# the resulting displayed columns are blue, green, orange.
_fig3_palette = sns.color_palette("deep", n_colors=3)
METHOD_COLORS = {
    "specparam": _fig3_palette[0],
    "SL_SD_additive": _fig3_palette[1],
    "SL_SD_specparam": _fig3_palette[2],
}

TRUTH_HLINE_KW = dict(color="k", lw=1.8, alpha=0.82, zorder=90)
SLOPE_BAND_KW = dict(color="red", lw=1.0, alpha=0.90, zorder=8)


# ---------------------- Small helpers ----------------------
def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = map(float, band)
    f = np.asarray(freqs, float)
    return (f >= lo) & (f <= hi)


def _slope_loglog(freqs: np.ndarray, power_lin: np.ndarray, band: Tuple[float, float]) -> float:
    f = np.asarray(freqs, float).ravel()
    y = np.asarray(power_lin, float).ravel()
    m = _band_mask(f, band) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        return np.nan
    x_log = np.log10(f[m])
    y_log = np.log10(np.clip(y[m], 1e-20, np.inf))
    A = np.vstack([x_log, np.ones_like(x_log)]).T
    slope, _intercept = np.linalg.lstsq(A, y_log, rcond=None)[0]
    return float(slope)


def _knee_freq_hz_from_kappa(kappa: float, exponent: float) -> float:
    kappa = float(kappa)
    exponent = float(exponent)
    if not np.isfinite(kappa) or not np.isfinite(exponent) or kappa <= 0 or exponent <= 0:
        return np.nan
    return float(kappa ** (1.0 / exponent))


def _robust_limits(
    values: Sequence[float],
    pct: Tuple[float, float] = (0.5, 99.5),
    pad_frac: float = 0.08,
    include: Optional[Sequence[float]] = None,
) -> Optional[Tuple[float, float]]:
    arr = np.asarray(values, float).ravel()
    if include is not None:
        arr = np.concatenate([arr, np.asarray(include, float).ravel()])
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    lo, hi = np.percentile(arr, list(pct))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        pad = max(abs(float(lo)) * pad_frac, 1.0)
        return (float(lo - pad), float(hi + pad))
    pad = float(pad_frac * (hi - lo))
    return (float(lo - pad), float(hi + pad))


def _log_ylim_from_arrays(*arrays: np.ndarray) -> Tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, float).ravel()
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            vals.append(a)
    if not vals:
        return (1e-6, 1e3)
    all_vals = np.concatenate(vals)
    lo = np.percentile(all_vals, 0.8)
    hi = np.percentile(all_vals, 99.7)
    if not np.isfinite(lo) or lo <= 0:
        lo = np.min(all_vals[all_vals > 0])
    if not np.isfinite(hi) or hi <= lo:
        hi = np.max(all_vals)
    lo = max(1e-6, 10.0 ** np.floor(np.log10(lo)))
    hi = 10.0 ** np.ceil(np.log10(hi))
    if hi <= lo:
        hi = lo * 100.0
    return float(lo), float(hi)


def _load_inputs(metrics_csv: str, row1_npz: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Could not find metrics CSV: {metrics_csv}")
    if not os.path.exists(row1_npz):
        raise FileNotFoundError(f"Could not find row-1 NPZ: {row1_npz}")

    df = pd.read_csv(metrics_csv)
    if "method" not in df.columns:
        raise ValueError("Metrics CSV must contain a 'method' column.")
    df = df.copy()
    df["method"] = df["method"].replace(METHOD_ALIASES)

    raw = np.load(row1_npz, allow_pickle=False)
    required = ["freqs", "amp_vals", "bb", "rh", "bb_true", "rh_true"]
    missing = [k for k in required if k not in raw.files]
    if missing:
        raise ValueError(f"NPZ is missing required keys: {missing}; found keys: {raw.files}")

    payload = {k: np.asarray(raw[k]) for k in required}

    bb = np.asarray(payload["bb"], float)
    rh = np.asarray(payload["rh"], float)
    if bb.ndim != 3 or rh.ndim != 3:
        raise ValueError(f"Expected bb/rh arrays with shape (method, amp, freq); got {bb.shape} and {rh.shape}")
    if bb.shape[0] != len(CACHE_METHOD_KEYS):
        raise ValueError(
            f"Expected first bb/rh dimension to have {len(CACHE_METHOD_KEYS)} methods "
            f"in cache order {CACHE_METHOD_KEYS}; got shape {bb.shape}."
        )
    return df, payload


def _truth_from_cached_payload(
    payload: Dict[str, np.ndarray],
) -> float:
    """Return true broadband slope based on cached GT broadband curves."""
    freqs = np.asarray(payload["freqs"], float).ravel()
    bb_true = np.asarray(payload["bb_true"], float)
    slope_by_amp = [_slope_loglog(freqs, bb_true[i, :], SLOPE_BAND) for i in range(bb_true.shape[0])]
    return float(np.nanmean(slope_by_amp))


def _format_amp_label(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    if abs(v - round(v)) < 1e-12:
        return f"{int(round(v))}"
    return f"{v:g}"


# ---------------------- Main plotting ----------------------
def make_compact_figure(
    df: pd.DataFrame,
    payload: Dict[str, np.ndarray],
    out_dir: str,
    prefix: str,
    true_cf_hz: float = DEFAULT_TRUE_CF_HZ,
    true_knee_hz: Optional[float] = None,
    true_knee_raw: float = DEFAULT_TRUE_KNEE_RAW,
    true_exponent: float = DEFAULT_TRUE_EXPONENT,
    fig_width: float = 15.5,
    fig_height: float = 8.8,
    dpi: int = 300,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    if true_knee_hz is None:
        true_knee_hz = _knee_freq_hz_from_kappa(true_knee_raw, true_exponent)
    _ = true_knee_hz  # intentionally unused in this compact version
    _ = true_cf_hz    # intentionally unused in this compact version

    dfp = df.copy()
    for needed in ["amp_true", "slope_est"]:
        if needed not in dfp.columns:
            dfp[needed] = np.nan

    dfp["amp_val"] = pd.to_numeric(dfp["amp_true"], errors="coerce")
    dfp["amp_label"] = dfp["amp_val"].map(_format_amp_label)

    freqs = np.asarray(payload["freqs"], float).ravel()
    amp_vals = np.asarray(payload["amp_vals"], float).ravel()
    bb = np.asarray(payload["bb"], float)
    rh = np.asarray(payload["rh"], float)
    bb_true = np.asarray(payload["bb_true"], float)
    rh_true = np.asarray(payload["rh_true"], float)

    slope_true = _truth_from_cached_payload(payload)

    amp_order_labels = [_format_amp_label(a) for a in amp_vals]
    slope_values = pd.to_numeric(dfp["slope_est"], errors="coerce").to_numpy(float)
    slope_ylim = _robust_limits(slope_values, pct=(0.25, 99.75), pad_frac=0.14, include=[slope_true])

    spectra_ylim = _log_ylim_from_arrays(bb, rh, bb_true, rh_true)
    slope_band = _band_mask(freqs, SLOPE_BAND)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=[0.85, 1.15, 1.05],
        hspace=0.50,
        wspace=0.22,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.10, top=0.95)

    # Row 1: Ground truth in the middle column only.
    ax_blank_left = fig.add_subplot(gs[0, 0])
    ax_blank_left.axis("off")
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_blank_right = fig.add_subplot(gs[0, 2])
    ax_blank_right.axis("off")

    for li, _amp in enumerate(amp_vals):
        alpha = 0.40 + 0.55 * (li / max(len(amp_vals) - 1, 1))
        ax_gt.loglog(freqs, bb_true[li, :], ls="--", lw=1.15, color="0.25", alpha=alpha)
        ax_gt.loglog(freqs, rh_true[li, :], ls="-", lw=1.65, color="0.05", alpha=alpha)
        if slope_band.sum() >= 2:
            ax_gt.loglog(freqs[slope_band], bb_true[li, slope_band], ls="-", **SLOPE_BAND_KW)
    ax_gt.set_xlim(ANALYSIS_FRANGE)
    ax_gt.set_ylim(spectra_ylim)
    ax_gt.set_title("Ground truth", fontsize=18, pad=7)
    ax_gt.set_ylabel("Power")
    ax_gt.set_xlabel("Frequency (Hz, log10)")
    ax_gt.minorticks_off()
    sns.despine(ax=ax_gt, top=True, right=True)

    # Rows 2-3: three method columns.
    axs = np.empty((2, 3), dtype=object)
    for rr in range(2):
        for cc in range(3):
            axs[rr, cc] = fig.add_subplot(gs[rr + 1, cc])

    for ci, method in enumerate(PLOT_METHOD_ORDER):
        axs[0, ci].set_title(METHOD_LABELS[method], fontsize=18, pad=8)

    # Row 2: model spectra/components.
    cache_index = {m: i for i, m in enumerate(CACHE_METHOD_KEYS)}
    legend_lines, legend_labels = [], []
    for ci, method in enumerate(PLOT_METHOD_ORDER):
        ax = axs[0, ci]
        mi = cache_index[method]
        method_color = METHOD_COLORS[method]
        for li, amp in enumerate(amp_vals):
            alpha = 0.38 + 0.55 * (li / max(len(amp_vals) - 1, 1))
            lw_rh = 1.25 + 0.25 * li
            ax.loglog(freqs, bb[mi, li, :], ls="--", lw=1.05, color=method_color, alpha=alpha)
            line_rh, = ax.loglog(freqs, rh[mi, li, :], ls="-", lw=lw_rh, color=method_color, alpha=alpha)
            if ci == 0:
                legend_lines.append(line_rh)
                legend_labels.append(f"{float(amp):g}x")
            if slope_band.sum() >= 2:
                ax.loglog(freqs[slope_band], bb[mi, li, slope_band], ls="-", **SLOPE_BAND_KW)
        ax.set_xlim(ANALYSIS_FRANGE)
        ax.set_ylim(spectra_ylim)
        ax.set_xlabel("Frequency (Hz, log10)")
        ax.set_ylabel("Power" if ci == 0 else "")
        ax.minorticks_off()
        sns.despine(ax=ax, top=True, right=True)

    leg = axs[0, 0].legend(
        legend_lines,
        legend_labels,
        title="Peak amplitude",
        loc="upper left",
        fontsize=8,
        title_fontsize=8,
        frameon=True,
    )
    leg.get_frame().set_alpha(0.92)
    axs[0, 0].text(
        0.98,
        0.07,
        "solid: rhythm\ndashed: broadband\nred: 40–60 Hz BB",
        transform=axs[0, 0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
    )

    # Row 3: slope sampling distributions by peak amplitude.
    for ci, method in enumerate(PLOT_METHOD_ORDER):
        ax = axs[1, ci]
        sub = dfp[dfp["method"] == method].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        sns.violinplot(
            data=sub,
            x="amp_label",
            y="slope_est",
            order=amp_order_labels,
            inner="quartile",
            cut=3,
            bw="scott",
            linewidth=1.0,
            width=0.86,
            color=METHOD_COLORS[method],
            saturation=0.78,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="amp_label",
            y="slope_est",
            order=amp_order_labels,
            color="k",
            alpha=0.32,
            size=2.2,
            jitter=0.15,
            ax=ax,
        )
        if np.isfinite(slope_true):
            ax.axhline(slope_true, **TRUTH_HLINE_KW)
        ax.set_xlabel("Peak amplitude (linear units)")
        ax.set_ylabel("Broadband slope\n(40–60 Hz, log–log)" if ci == 0 else "")
        ax.tick_params(axis="x", rotation=25)
        if slope_ylim is not None:
            ax.set_ylim(slope_ylim)
        sns.despine(ax=ax, top=True, right=True)

    out_png = os.path.join(out_dir, f"{prefix}.png")
    out_svg = os.path.join(out_dir, f"{prefix}.svg")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_svg, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_png}")
    print(f"[saved] {out_svg}")
    return out_png, out_svg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot Figure 7 height-grid benchmark from cached CSV/NPZ without rerunning analyses."
    )
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--row1-npz", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="Figure_7_fig3truth_reordered_compact_v2")
    parser.add_argument("--true-cf-hz", type=float, default=DEFAULT_TRUE_CF_HZ)
    parser.add_argument("--true-knee-hz", type=float, default=None)
    parser.add_argument("--true-knee-raw", type=float, default=DEFAULT_TRUE_KNEE_RAW)
    parser.add_argument("--true-exponent", type=float, default=DEFAULT_TRUE_EXPONENT)
    parser.add_argument("--fig-width", type=float, default=15.5)
    parser.add_argument("--fig-height", type=float, default=8.8)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    metrics_csv = os.path.expanduser(args.metrics_csv) if args.metrics_csv else os.path.join(
        out_dir, "Figure_7_fig3truth_metrics.csv"
    )
    row1_npz = os.path.expanduser(args.row1_npz) if args.row1_npz else os.path.join(
        out_dir, "Figure_7_fig3truth_row1_components.npz"
    )

    print(f"[load metrics] {metrics_csv}")
    print(f"[load row1 npz] {row1_npz}")
    df, payload = _load_inputs(metrics_csv, row1_npz)

    make_compact_figure(
        df=df,
        payload=payload,
        out_dir=out_dir,
        prefix=args.prefix,
        true_cf_hz=args.true_cf_hz,
        true_knee_hz=args.true_knee_hz,
        true_knee_raw=args.true_knee_raw,
        true_exponent=args.true_exponent,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
