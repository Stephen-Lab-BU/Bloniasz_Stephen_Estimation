#!/usr/bin/env python3
"""
Generate the Figure 1 overview grid.

The script builds the manuscript overview figure using the canonical Figure 1
plotting utilities and, when available, cached empirical and simulation payloads
from Figures 4 and 5. It writes the figure and reproducibility payloads used by
the manuscript build.
"""

from __future__ import annotations

import os
from pathlib import Path
import argparse
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# Import the original figure builder to guarantee identical styling + behavior.
# TODO CLEAN UP: Refactor the original figure builder to separate style from data handling, so we can reuse the style without needing to import the whole thing.
import figure_1_final as F1

# ---------------------- Row 1 scale bar helper ----------------------
def _add_horizontal_scale_bar(ax, t_window, x_window, fs, scale_ms=100, pad_frac=0.06):
    """Draw a horizontal scale bar near bottom-right of a timeseries plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axis to draw on.
    t_window: 1D array
        Time vector (seconds) of the plotted window relative to window start.
    x_window: 1D array
        Data vector of the plotted window.
    fs: float
        Sampling frequency (Hz). Included for API symmetry.
    scale_ms: float, default 100
        Length of the scale bar in milliseconds.
    pad_frac: float, default 0.06
        Fraction of the window width reserved for right padding.
    """
    import numpy as _np
    t_window = _np.asarray(t_window, float)
    x_window = _np.asarray(x_window, float)
    duration = float(t_window[-1] - t_window[0])
    bar_len_s = min(scale_ms / 1000.0, max(1e-6, 0.3 * duration))  # cap at 30% of width
    x_pad = pad_frac * duration
    x1 = float(t_window[-1] - x_pad - bar_len_s)
    x2 = x1 + bar_len_s
    y_lo = float(_np.nanpercentile(x_window, 3.0))
    y_hi = float(_np.nanpercentile(x_window, 97.0))
    y_span = max(1e-12, y_hi - y_lo)
    y = y_lo + 0.12 * y_span
    ax.plot([x1, x2], [y, y], lw=2.0, color="k", solid_capstyle="butt")
    ax.text(0.5 * (x1 + x2), y - 0.06 * y_span, f"{int(scale_ms)} ms",
            ha="center", va="top", fontsize=11)


# ---------------------- Default Figure 4 output dir ----------------------
FIG4_DIR_DEFAULT = Path(
    "CHANGE_THIS_ROOT_TO_PATH/Bloniasz_Stephen_Estimation/Output/Results/FiguresIntermediate/Figure_4_CV/Figure_output"
)

# ---------------------- Default Figure 5 output dir ----------------------
FIG5_DIR_DEFAULT = Path(
    "CHANGE_THIS_ROOT_TO_PATH/Bloniasz_Stephen_Estimation/Output/Results/FiguresIntermediate/Figure_5_CV/Figure_output"
)


# ====================== Generic helpers ======================
def _pick_latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _load_npz_and_meta(npz_path: Path, meta_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    npz = np.load(npz_path, allow_pickle=False)
    arrays = {k: npz[k] for k in npz.files}
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return arrays, meta


def _get_meta(meta: dict, keys: tuple[str, ...], default=None):
    # Search top-level then meta["meta"] (the saver puts non-array stuff under "meta")
    for k in keys:
        if k in meta:
            return meta[k]
    m = meta.get("meta", None)
    if isinstance(m, dict):
        for k in keys:
            if k in m:
                return m[k]
    return default




def _fig4_target_rel_min_from_payload(fig4_dir: Path | str, state: str) -> float | None:
    fig4_dir = Path(fig4_dir).expanduser().resolve()
    npz_path = fig4_dir / f"Figure_4_CV_{state}.plotdata.npz"
    meta_path = fig4_dir / f"Figure_4_CV_{state}.plotmeta.json"
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        with open(meta_path, "r") as f:
            meta_file = json.load(f)
        meta = meta_file.get("meta", meta_file)
        idx = int(meta.get("idx_target", 0))
        with np.load(npz_path, allow_pickle=False) as z:
            if "T_rel_min" not in z.files:
                return None
            t_rel = np.asarray(z["T_rel_min"], float).ravel()
        if 0 <= idx < t_rel.size and np.isfinite(t_rel[idx]):
            return float(t_rel[idx])
    except Exception as exc:
        print(f"[WARN] Figure 1 could not read Figure 4 target time: {exc}")
    return None


def _idx_nearest_time(t_rel_min: np.ndarray, target_min: float, fallback_idx: int = 0) -> int:
    t = np.asarray(t_rel_min, float).ravel()
    if t.size == 0 or not np.isfinite(float(target_min)):
        return int(fallback_idx)
    return int(np.nanargmin(np.abs(t - float(target_min))))

def _get_arr(arrays: dict[str, np.ndarray], names: tuple[str, ...], default=None):
    # same match first
    for n in names:
        if n in arrays:
            return arrays[n]
    # Suffix match for flattened section keys like "row2__x"
    for n in names:
        suf = "__" + n
        for k in arrays.keys():
            if k.endswith(suf):
                return arrays[k]
    return default


def _align_to_x(y: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float).ravel()
    x_ref = np.asarray(x_ref, float).ravel()
    if y.size == x_ref.size:
        return y
    if y.size == 0:
        return np.full_like(x_ref, np.nan, dtype=float)
    x_src = np.linspace(float(x_ref[0]), float(x_ref[-1]), y.size)
    return np.interp(x_ref, x_src, y)


# ====================== Figure 4 loading helpers ======================
def _find_fig4_payload(fig4_dir: Path, state: str, mode: str) -> tuple[Path, Path]:
    """
    Find the latest matching Figure 4 payload. Supports either:
      Figure_4_CV_<state>.plotdata.npz
      Figure_4_CV_<state>.<mode>.plotdata.npz
    and their paired plotmeta JSONs.
    """
    fig4_dir = Path(fig4_dir).expanduser().resolve()

    patterns = [
        f"Figure_4_CV_{state}.{mode}.plotdata.npz",
        f"Figure_4_CV_{state}.plotdata.npz",
        f"Figure_4_CV_{state}*.{mode}.plotdata.npz",
        f"Figure_4_CV_{state}*.plotdata.npz",
    ]

    npz_path = None
    for pat in patterns:
        hits = [Path(p) for p in glob.glob(str(fig4_dir / pat))]
        if hits:
            npz_path = _pick_latest(hits)
            break

    if npz_path is None or (not npz_path.exists()):
        raise FileNotFoundError(
            f"[Row2/Row5] Could not find Figure 4 plotdata for state='{state}', mode='{mode}' in:\n"
            f"  {fig4_dir}\n"
            f"Expected something like:\n"
            f"  Figure_4_CV_{state}.plotdata.npz  OR  Figure_4_CV_{state}.{mode}.plotdata.npz"
        )

    meta_path = Path(str(npz_path).replace(".plotdata.npz", ".plotmeta.json"))
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[Row2/Row5] Found plotdata but missing paired plotmeta:\n"
            f"  plotdata: {npz_path}\n"
            f"  expected: {meta_path}"
        )

    return npz_path, meta_path


def _find_fig4_slsd_params(fig4_dir: Path, state: str, mode: str) -> tuple[dict, Path]:
    """
    Prefer fig4_slsd_params_<state>.json, else choose latest *slsd*params*<state>*.json.
    """
    fig4_dir = Path(fig4_dir).expanduser().resolve()

    preferred = fig4_dir / f"fig4_slsd_params_{state}.json"
    if preferred.exists():
        path = preferred
    else:
        patterns = [
            f"fig4_slsd_params_{state}*.json",
            f"*slsd*params*{state}*.json",
            f"*row5*sim*params*{state}*.json",
            f"*slsd*{state}*.json",
        ]
        hits: list[Path] = []
        for pat in patterns:
            hits.extend([Path(p) for p in glob.glob(str(fig4_dir / pat))])
        path = _pick_latest([p for p in hits if p.exists()]) if hits else None

    if path is None or (not path.exists()):
        raise FileNotFoundError(
            f"[Row5 compute-fallback] Could not find SL_SD params JSON exported by Figure 4 for state='{state}'.\n"
            f"Searched in:\n  {fig4_dir}\n"
            f"Preferred:\n  fig4_slsd_params_{state}.json\n"
            f"Or anything matching '*slsd*params*{state}*.json'"
        )

    with open(path, "r") as f:
        blob = json.load(f)

    params = blob.get("params", blob)
    if not isinstance(params, dict):
        raise RuntimeError(f"[Row5 compute-fallback] Params JSON did not contain a dict at {path}")

    return params, path


# ====================== Figure 5 loading helpers ======================
def _fig5_suffix(simlen: str, windows: str) -> str:
    # Figure_5_cv_final.py naming:
    # suffix = ("_2win" if windows_mode=="2" else "") + (f"_{simlen}" if simlen!="full" else "")
    suf = ""
    if windows == "2":
        suf += "_2win"
    if simlen != "full":
        suf += f"_{simlen}"
    return suf


def _find_fig5_payload(fig5_dir: Path, state: str, simlen: str, windows: str) -> tuple[Path, Path]:
    """
    Find latest matching Figure 5 payload:
      Figure_5_CV_<state><suffix>.plotdata.npz
      Figure_5_CV_<state><suffix>.plotmeta.json
    """
    fig5_dir = Path(fig5_dir).expanduser().resolve()
    suffix = _fig5_suffix(simlen=simlen, windows=windows)

    patterns = [
        f"Figure_5_CV_{state}{suffix}.plotdata.npz",
        f"Figure_5_CV_{state}*{suffix}.plotdata.npz",
        f"Figure_5_CV_{state}*.plotdata.npz",  # last resort
    ]

    npz_path = None
    for pat in patterns:
        hits = [Path(p) for p in glob.glob(str(fig5_dir / pat))]
        if hits:
            npz_path = _pick_latest([h for h in hits if h.exists()])
            break

    if npz_path is None or (not npz_path.exists()):
        raise FileNotFoundError(
            f"[Row5] Could not find Figure 5 plotdata for state='{state}', simlen='{simlen}', windows='{windows}' in:\n"
            f"  {fig5_dir}\n"
            f"Expected something like:\n"
            f"  Figure_5_CV_{state}{suffix}.plotdata.npz"
        )

    meta_path = Path(str(npz_path).replace(".plotdata.npz", ".plotmeta.json"))
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[Row5] Found plotdata but missing paired plotmeta:\n"
            f"  plotdata: {npz_path}\n"
            f"  expected: {meta_path}"
        )

    return npz_path, meta_path


def _find_fig5_slsd_fit_params(fig5_dir: Path, state: str) -> tuple[dict, Path] | tuple[None, None]:
    """
    Prefer fig5_slsd_fit_params_<state>.json.
    Returns (params_dict, path) or (None, None) if not found.
    """
    fig5_dir = Path(fig5_dir).expanduser().resolve()
    preferred = fig5_dir / f"fig5_slsd_fit_params_{state}.json"
    if not preferred.exists():
        hits = [Path(p) for p in glob.glob(str(fig5_dir / f"fig5_slsd_fit_params_{state}*.json"))]
        preferred = _pick_latest([h for h in hits if h.exists()]) if hits else None

    if preferred is None or (not preferred.exists()):
        return None, None

    with open(preferred, "r") as f:
        blob = json.load(f)

    params = blob.get("params", blob)
    if not isinstance(params, dict):
        return None, None

    return params, preferred


# ====================== Row 2 (from Figure 4) ======================
def build_row2_anesthesia_from_fig4(ax1, ax2, ax3, mode: str, fig4_dir: Path, state: str = "anesthesia"):
    """
    Plot Row 2 using Figure 4 exported arrays, but keep same plotting style of figure_1_final.py.
    Also returns Figure-4-exported SL_SD params for potential compute fallback simulation.
    """
    npz_path, meta_path = _find_fig4_payload(fig4_dir, state=state, mode=mode)
    arrays, meta = _load_npz_and_meta(npz_path, meta_path)

    x = _get_arr(arrays, ("x", "x_band", "F_fit", "freqs", "frequencies"))
    if x is None:
        raise KeyError(
            f"[Row2] Could not find frequency axis in Figure 4 payload {npz_path}.\n"
            f"Tried keys: x, x_band, F_fit, freqs, frequencies (and suffix-matches)."
        )
    x = np.asarray(x, float).ravel()

    P_tf = _get_arr(arrays, ("P_fit_tf", "P_tf", "mt_tf", "power_tf"))
    if P_tf is None:
        mt_this = _get_arr(arrays, ("mt_this", "mt", "P_this"))
        if mt_this is None:
            raise KeyError(
                f"[Row2] Could not find multitaper TF matrix in Figure 4 payload {npz_path}.\n"
                f"Tried keys: P_fit_tf, P_tf, mt_tf, power_tf (or mt_this/mt/P_this)."
            )
        P_tf = np.asarray(mt_this, float).reshape(1, -1)
    else:
        P_tf = np.asarray(P_tf, float)

    sp_full_all = _get_arr(arrays, ("specparam_full", "sp_full"))
    sp_ap_all   = _get_arr(arrays, ("specparam_aper", "specparam_ap", "sp_ap", "specparam_aperiodic"))
    sp_rh_all   = _get_arr(arrays, ("specparam_rh", "specparam_pk", "sp_pk", "specparam_rhythms"))

    sl_total_all = _get_arr(arrays, ("slsd_total", "sl_total", "slsd_full"))
    sl_bb_all    = _get_arr(arrays, ("slsd_bb", "sl_bb", "slsd_broadband"))
    sl_rh_all    = _get_arr(arrays, ("slsd_rh", "sl_rh", "slsd_rhythms"))

    if sp_full_all is None or sp_ap_all is None or sp_rh_all is None:
        raise KeyError(f"[Row2] Missing specparam components in {npz_path}. Need full + aperiodic + rhythms.")
    if sl_total_all is None or sl_bb_all is None or sl_rh_all is None:
        raise KeyError(f"[Row2] Missing SL_SD components in {npz_path}. Need total + broadband + rhythms.")

    slopes_sp = _get_arr(arrays, ("slopes_specparam", "slopes_sp", "slopes_specparam_aperiodic"))
    slopes_bb = _get_arr(arrays, ("slopes_slsd", "slopes_bb", "slopes_slsd_bb", "slopes_broadband"))
    if slopes_sp is None or slopes_bb is None:
        raise KeyError(f"[Row2] Missing slope arrays for violins in {npz_path}.")

    slopes_sp = np.asarray(slopes_sp, float).ravel()
    slopes_bb = np.asarray(slopes_bb, float).ravel()

    idx_target = _get_meta(meta, ("idx_target", "chosen_idx", "idx"), default=0)
    idx_target = int(np.clip(int(idx_target), 0, max(P_tf.shape[0] - 1, 0)))

    def _select(y_all):
        y_all = np.asarray(y_all, float)
        if y_all.ndim == 1:
            return y_all
        if y_all.ndim == 2:
            return y_all[idx_target, :]
        return np.asarray(y_all).reshape(-1)

    mt_this = _select(P_tf)
    sp_full = _select(sp_full_all)
    sp_ap   = _select(sp_ap_all)
    sp_rh   = _select(sp_rh_all)

    sl_total = _select(sl_total_all)
    sl_bb    = _select(sl_bb_all)
    sl_rh    = _select(sl_rh_all)

    mt_this  = _align_to_x(mt_this, x)
    sp_full  = _align_to_x(sp_full, x)
    sp_ap    = _align_to_x(sp_ap, x)
    sp_rh    = _align_to_x(sp_rh, x)
    sl_total = _align_to_x(sl_total, x)
    sl_bb    = _align_to_x(sl_bb, x)
    sl_rh    = _align_to_x(sl_rh, x)

    def _pos(y):
        return np.clip(np.asarray(y, float), 1e-20, np.inf)

    mt_this  = _pos(mt_this)
    sp_full  = _pos(sp_full)
    sp_ap    = _pos(sp_ap)
    sp_rh    = _pos(sp_rh)
    sl_total = _pos(sl_total)
    sl_bb    = _pos(sl_bb)
    sl_rh    = _pos(sl_rh)

    # --- Plot EXACTLY like figure_1_final.py Row 2 ---
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    F1.plot_ll(ax1, x, mt_this, kind="multitaper", label="Multitaper", lw=2, alpha=0.8)
    F1.plot_ll(ax1, x, sp_full, kind="full", label="Specparam full")
    F1.plot_ll(ax1, x, sp_ap,   kind="broadband", label="Specparam aperiodic")
    F1.plot_ll(ax1, x, sp_rh,   kind="rhythms", label="Specparam rhythms")
    ax1.set_title("Specparam")
    ax1.set_xlabel("Frequency (Hz, log10)")
    ax1.set_ylabel("Power (log10)")
    ax1.grid(False)
    ax1.legend(frameon=False)
    ax1.set_ylim(1e-1, 1e6)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    F1.plot_ll(ax2, x, mt_this,  kind="multitaper", label="Multitaper", lw=2, alpha=0.8)
    F1.plot_ll(ax2, x, sl_total, kind="full", label="SL_SD full")
    F1.plot_ll(ax2, x, sl_bb,    kind="broadband", label="SL_SD broadband")
    F1.plot_ll(ax2, x, sl_rh,    kind="rhythms", label="SL_SD rhythms")
    ax2.set_title("SL_SD")
    ax2.set_xlabel("Frequency (Hz, log10)")
    ax2.set_ylabel("Power (log10)")
    ax2.grid(False)
    ax2.legend(frameon=False)
    ax2.set_ylim(1e-1, 1e6)

    df = pd.DataFrame({
        "slope": np.concatenate([slopes_sp, slopes_bb]),
        "method": (["Specparam aperiodic"] * len(slopes_sp)) + (["SL_SD broadband"] * len(slopes_bb)),
    })
    palette = sns.color_palette("deep", 2)
    sns.violinplot(
        data=df,
        x="method",
        y="slope",
        hue="method",
        inner="quartile",
        cut=4,
        bw_method="scott",
        linewidth=1.0,
        width=0.9,
        palette=palette,
        legend=False,
        ax=ax3,
    )
    sns.stripplot(
        data=df,
        x="method",
        y="slope",
        hue="method",
        dodge=False,
        color="k",
        alpha=0.35,
        size=3,
        jitter=0.15,
        ax=ax3,
        legend=False,
    )
    if ax3.legend_ is not None:
        ax3.legend_.remove()
    ax3.set_title("40-60 Hz slope distributions")
    ax3.set_xlabel("")
    ax3.set_ylabel("40–60 Hz slope")
    ax3.grid(False)

    fs = float(_get_meta(meta, ("fs", "FS", "sampling_rate"), default=F1.FS))
    abs_t0 = _get_meta(meta, ("abs_t0", "t0", "anesthesia_t0"), default=np.nan)
    abs_t1 = _get_meta(meta, ("abs_t1", "t1", "anesthesia_t1"), default=np.nan)
    chosen_time = _get_meta(meta, ("chosen_time",), default=None)
    n_windows = int(_get_meta(meta, ("n_windows",), default=None) or max(len(slopes_sp), P_tf.shape[0]))

    sim_params, params_path = _find_fig4_slsd_params(fig4_dir, state=state, mode=mode)

    row2_cache = dict(
        x=np.asarray(x),
        mt_this=np.asarray(mt_this),
        sp_full=np.asarray(sp_full),
        sp_ap=np.asarray(sp_ap),
        sp_pk=np.asarray(sp_rh),
        sl_total=np.asarray(sl_total),
        sl_bb=np.asarray(sl_bb),
        sl_rh=np.asarray(sl_rh),
        slopes_sp=np.asarray(slopes_sp),
        slopes_bb=np.asarray(slopes_bb),
        chosen_time=chosen_time,
        chosen_idx=int(idx_target),
        abs_t0=abs_t0,
        abs_t1=abs_t1,
        n_windows=int(n_windows),
        fs=float(fs),
        F_fit=np.asarray(x),
        x_band=np.asarray(x),
    )

    print(f"[Row2] Using Figure 4 payload → {npz_path.name}")
    print(f"[Row2] Using Figure 4 params   → {params_path.name}")

    return dict(
        fs=float(fs),
        n_windows=int(n_windows),
        sim_params=sim_params,   # used only for compute fallback of Row5
        row2_cache=row2_cache,
    )


# ====================== Row 5 (from Figure 5 payload) ======================
def build_row5_from_fig5_payload(ax1, ax2, ax3, fig5_dir: Path, *, state: str, simlen: str, windows: str):
    """
    Plot Figure 1 Row 5 from Figure 5 payload arrays (NO recomputation).
    Panels:
      - left: Specparam decomposition on target window + GT PSD
      - mid: SL_SD decomposition on target window + GT PSD
      - right: 40–60 Hz slope distributions imported from Figure 5
    """
    npz_path, meta_path = _find_fig5_payload(fig5_dir, state=state, simlen=simlen, windows=windows)
    arrays, meta = _load_npz_and_meta(npz_path, meta_path)

    F_fit = _get_arr(arrays, ("F_fit", "x", "freqs", "frequencies"))
    P_fit_tf = _get_arr(arrays, ("P_fit_tf", "P_tf"))
    GT_on_fit = _get_arr(arrays, ("GT_on_fit",))

    sp_full_all = _get_arr(arrays, ("specparam_full",))
    sp_ap_all   = _get_arr(arrays, ("specparam_aper",))
    sp_rh_all   = _get_arr(arrays, ("specparam_rh",))

    sl_total_all = _get_arr(arrays, ("slsd_total",))
    sl_bb_all    = _get_arr(arrays, ("slsd_bb",))
    sl_rh_all    = _get_arr(arrays, ("slsd_rh",))

    cvll_sp = _get_arr(arrays, ("cvll_specparam",), default=None)
    cvll_sl = _get_arr(arrays, ("cvll_slsd",), default=None)

    if F_fit is None or P_fit_tf is None:
        raise KeyError(f"[Row5] Missing F_fit/P_fit_tf in Figure 5 payload: {npz_path}")

    x = np.asarray(F_fit, float).ravel()
    P_tf = np.asarray(P_fit_tf, float)

    idx_target = _get_meta(meta, ("idx_target",), default=0)
    idx_target = int(np.clip(int(idx_target), 0, max(P_tf.shape[0] - 1, 0)))

    def _select(y_all):
        if y_all is None:
            return None
        y_all = np.asarray(y_all, float)
        if y_all.ndim == 1:
            return y_all
        if y_all.ndim == 2:
            return y_all[idx_target, :]
        return y_all.reshape(-1)

    mt_this = _select(P_tf)
    GT = _select(GT_on_fit)

    sp_full = _select(sp_full_all)
    sp_ap   = _select(sp_ap_all)
    sp_rh   = _select(sp_rh_all)

    sl_total = _select(sl_total_all)
    sl_bb    = _select(sl_bb_all)
    sl_rh    = _select(sl_rh_all)

    mt_this = _align_to_x(mt_this, x)
    if GT is not None:
        GT = _align_to_x(GT, x)
    if sp_full is not None:
        sp_full = _align_to_x(sp_full, x)
        sp_ap   = _align_to_x(sp_ap, x)
        sp_rh   = _align_to_x(sp_rh, x)
    if sl_total is not None:
        sl_total = _align_to_x(sl_total, x)
        sl_bb    = _align_to_x(sl_bb, x)
        sl_rh    = _align_to_x(sl_rh, x)

    def _pos(y):
        if y is None:
            return None
        return np.clip(np.asarray(y, float), 1e-20, np.inf)

    mt_this = _pos(mt_this)
    GT      = _pos(GT)
    sp_full = _pos(sp_full); sp_ap = _pos(sp_ap); sp_rh = _pos(sp_rh)
    sl_total = _pos(sl_total); sl_bb = _pos(sl_bb); sl_rh = _pos(sl_rh)

    # --- Left panel: Specparam (Figure 5) ---
    ax1.set_xscale("log"); ax1.set_yscale("log")
    if GT is not None:
        ax1.plot(x, GT, color="k", lw=2.4, alpha=0.95, label="Ground Truth PSD")
    F1.plot_ll(ax1, x, mt_this, kind="multitaper", label="Multitaper", lw=2, alpha=0.8)
    if sp_full is not None:
        F1.plot_ll(ax1, x, sp_full, kind="full", label="Specparam full")
        F1.plot_ll(ax1, x, sp_ap,   kind="broadband", label="Specparam aperiodic")
        F1.plot_ll(ax1, x, sp_rh,   kind="rhythms", label="Specparam rhythms")
    ax1.set_title("Simulation (Figure 5) — Specparam")
    ax1.set_xlabel("Frequency (Hz, log10)")
    ax1.set_ylabel("Power (log10)")
    ax1.grid(False)
    ax1.legend(frameon=False)
    ax1.set_ylim(1e-1, 1e6)

    # --- Middle panel: SL_SD (Figure 5) ---
    ax2.set_xscale("log"); ax2.set_yscale("log")
    if GT is not None:
        ax2.plot(x, GT, color="k", lw=2.4, alpha=0.95, label="Ground Truth PSD")
    F1.plot_ll(ax2, x, mt_this, kind="multitaper", label="Multitaper", lw=2, alpha=0.8)
    if sl_total is not None:
        F1.plot_ll(ax2, x, sl_total, kind="full", label="SL_SD full")
        F1.plot_ll(ax2, x, sl_bb,    kind="broadband", label="SL_SD broadband")
        F1.plot_ll(ax2, x, sl_rh,    kind="rhythms", label="SL_SD rhythms")
    ax2.set_title("Simulation (Figure 5) — SL_SD")
    ax2.set_xlabel("Frequency (Hz, log10)")
    ax2.set_ylabel("Power (log10)")
    ax2.grid(False)
    ax2.legend(frameon=False)
    ax2.set_ylim(1e-1, 1e6)

    # --- Right panel: 40–60 Hz slope distributions (Figure 5 import) ---
    slopes_sp = _get_arr(arrays, ("slopes_specparam", "slopes_sp", "slopes_specparam_aperiodic"), default=None)
    slopes_bb = _get_arr(arrays, ("slopes_slsd", "slopes_bb", "slopes_slsd_bb", "slopes_broadband"), default=None)

    true_slope = _get_meta(meta, ("true_slope",), default=None)

    if slopes_sp is None or slopes_bb is None:
        ax3.text(0.5, 0.5, "No slope arrays in Figure 5 payload", ha="center", va="center")
        ax3.axis("off")
        slopes_sp = np.array([])
        slopes_bb = np.array([])
    else:
        slopes_sp = np.asarray(slopes_sp, float).ravel()
        slopes_bb = np.asarray(slopes_bb, float).ravel()

        df = pd.DataFrame({
            "slope": np.concatenate([slopes_sp, slopes_bb]),
            "method": (["Specparam aperiodic"] * len(slopes_sp)) + (["SL_SD broadband"] * len(slopes_bb)),
        })
        palette = sns.color_palette("deep", 2)
        sns.violinplot(
            data=df,
            x="method",
            y="slope",
            hue="method",
            inner="quartile",
            cut=4,
            bw_method="scott",
            linewidth=1.0,
            width=0.9,
            palette=palette,
            legend=False,
            ax=ax3,
        )
        sns.stripplot(
            data=df,
            x="method",
            y="slope",
            hue="method",
            dodge=False,
            color="k",
            alpha=0.35,
            size=3,
            jitter=0.15,
            ax=ax3,
            legend=False,
        )
        if ax3.legend_ is not None:
            ax3.legend_.remove()

        # If Figure 5 meta includes the true 40–60 Hz slope, draw it (matches Fig5 Row4 style)
        if true_slope is not None:
            try:
                ts = float(true_slope)
                if np.isfinite(ts):
                    ax3.axhline(ts, color="k", lw=2.2, alpha=0.95)
            except Exception:
                pass

        ax3.set_title("40-60 Hz slope distributions (Figure 5)")
        ax3.set_xlabel("")
        ax3.set_ylabel("40–60 Hz slope")
        ax3.grid(False)

    # Prefer sim params stored in meta (exported by figure_5_cv_final.py)
    sim_params = _get_meta(meta, ("slsd_fit_params",), default=None)
    params_path = _get_meta(meta, ("slsd_fit_params_path",), default=None)
    if not isinstance(sim_params, dict):
        sim_params, pth = _find_fig5_slsd_fit_params(fig5_dir, state=state)
        params_path = str(pth) if pth is not None else params_path
    if not isinstance(sim_params, dict):
        sim_params = {}

    # Keep ΔCVLL cached if present (even though we no longer plot it) for downstream 
    dcv = np.array([])
    if cvll_sp is not None and cvll_sl is not None:
        cvll_sp_arr = np.asarray(cvll_sp, float).ravel()
        cvll_sl_arr = np.asarray(cvll_sl, float).ravel()
        m = np.isfinite(cvll_sp_arr) & np.isfinite(cvll_sl_arr)
        dcv = (cvll_sl_arr[m] - cvll_sp_arr[m])
        dcv = dcv[np.isfinite(dcv)]

    row5_cache = dict(
        x=np.asarray(x),
        mt_this=np.asarray(mt_this),
        GT_on_fit=np.asarray(GT) if GT is not None else np.array([]),
        sp_full=np.asarray(sp_full) if sp_full is not None else np.array([]),
        sp_ap=np.asarray(sp_ap) if sp_ap is not None else np.array([]),
        sp_pk=np.asarray(sp_rh) if sp_rh is not None else np.array([]),
        sl_total=np.asarray(sl_total) if sl_total is not None else np.array([]),
        sl_bb=np.asarray(sl_bb) if sl_bb is not None else np.array([]),
        sl_rh=np.asarray(sl_rh) if sl_rh is not None else np.array([]),

        slopes_sp=np.asarray(slopes_sp) if slopes_sp is not None else np.array([]),
        slopes_bb=np.asarray(slopes_bb) if slopes_bb is not None else np.array([]),
        true_slope=(float(true_slope) if true_slope is not None else np.nan),

        cvll_specparam=np.asarray(cvll_sp) if cvll_sp is not None else np.array([]),
        cvll_slsd=np.asarray(cvll_sl) if cvll_sl is not None else np.array([]),
        dcv=np.asarray(dcv),

        chosen_idx=int(idx_target),
        fig5_payload=str(npz_path),
        fig5_params_path=str(params_path) if params_path is not None else "",
        fig5_state=str(state),
        fig5_simlen=str(simlen),
        fig5_windows=str(windows),
    )

    print(f"[Row5] Using Figure 5 payload → {npz_path.name}")
    if params_path:
        print(f"[Row5] Using Figure 5 SL_SD-fit params → {Path(params_path).name}")

    return dict(
        row5_cache=row5_cache,
        sim_params=sim_params,
    )


# ====================== Main ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "short"], default="full")
    ap.add_argument(
        "--out",
        type=str,
        default="Figure_Grid.png",
        help="Output filename (PNG or SVG). If relative, saved into --out-dir.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(F1.DEFAULT_OUT_DIR),
        help="Directory where outputs will be saved (used when --out is relative).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA256 hashing of input MAT files in the report.",
    )

    # Row 2 (Figure 4 import)
    ap.add_argument(
        "--row2-source",
        choices=["fig4", "compute"],
        default="fig4",
        help="Where Row 2 empirical curves/slopes come from. Default: fig4 (preferred).",
    )
    ap.add_argument(
        "--fig4-dir",
        type=str,
        default=str(FIG4_DIR_DEFAULT),
        help="Figure 4 output directory containing plotdata/plotmeta and exported SL_SD params.",
    )
    ap.add_argument(
        "--fig4-state",
        choices=["anesthesia", "awake"],
        default="anesthesia",
        help="Which Figure 4 state payload to use for Row 2 (and for Row5 compute fallback params).",
    )

    # Row 5 (Figure 5 import)
    ap.add_argument(
        "--row5-source",
        choices=["fig5", "compute"],
        default="fig5",
        help="Where Row 5 simulated panels come from. Default: fig5 (preferred).",
    )
    ap.add_argument(
        "--fig5-dir",
        type=str,
        default=str(FIG5_DIR_DEFAULT),
        help="Figure 5 output directory containing plotdata/plotmeta exports.",
    )
    ap.add_argument(
        "--fig5-state",
        choices=["anesthesia", "awake"],
        default="anesthesia",
        help="Which Figure 5 state payload to use for Row 5.",
    )
    ap.add_argument(
        "--fig5-simlen",
        choices=["full", "short"],
        default=None,
        help="Which Figure 5 simlen to import. Default: derived from --mode.",
    )
    ap.add_argument(
        "--fig5-windows",
        choices=["all", "2"],
        default=None,
        help="Which Figure 5 windows setting to import. Default: derived from --mode.",
    )

    args = ap.parse_args()

    # Derive Figure 5 selectors from Figure 1 mode unless explicitly provided
    if args.fig5_simlen is None:
        args.fig5_simlen = "full" if args.mode == "full" else "short"
    if args.fig5_windows is None:
        args.fig5_windows = "all" if args.mode == "full" else "2"

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = out_dir / out_path.name

    fig4_dir = Path(args.fig4_dir).expanduser().resolve()
    fig5_dir = Path(args.fig5_dir).expanduser().resolve()

    # Figure grid: identical to figure_1_final.py
    fig = plt.figure(figsize=(18, 23))
    outer = gridspec.GridSpec(
        nrows=5,
        ncols=3,
        height_ratios=[1.0, 1.5, 2.3, 1.6, 1.5],
        hspace=0.85,
        wspace=0.55,
    )

    # Row 1: unchanged (3 s ECoG)
    ax_r1 = fig.add_subplot(outer[0, :])
    row1_cache = F1.build_row1(ax_r1)

    # Row 1: draw ECoG trace in black and add 100 ms scale bar
    ax_r1 = fig.add_subplot(outer[0, :])
    row1_cache = F1.build_row1(ax_r1)
    # recolor the trace to black (original defaults to color cycle)
    for line in ax_r1.get_lines():
        line.set_color("k")
    # remove axes spines and ticks
    for side in ["top", "right", "left", "bottom"]:
        ax_r1.spines[side].set_visible(False)
    ax_r1.set_xticks([])
    ax_r1.set_yticks([])
    # add 100 ms scale bar using helper
    _add_horizontal_scale_bar(ax_r1, row1_cache.get("t", []),
                            row1_cache.get("x", []),
                            row1_cache.get("fs", 1.0),
                            scale_ms=100)


    # Row 2: import from Figure 4 (preferred) or compute
    ax21 = fig.add_subplot(outer[1, 0])
    ax22 = fig.add_subplot(outer[1, 1])
    ax23 = fig.add_subplot(outer[1, 2])

    sim_info = None
    if args.row2_source == "fig4":
        try:
            sim_info = build_row2_anesthesia_from_fig4(
                ax21, ax22, ax23, args.mode, fig4_dir, state=args.fig4_state
            )
        except Exception as e:
            print(f"[WARN] Failed to load Row 2 from Figure 4 ({e}). Falling back to compute.")
            sim_info = F1.build_row2_anesthesia(ax21, ax22, ax23, args.mode)
    else:
        sim_info = F1.build_row2_anesthesia(ax21, ax22, ax23, args.mode)

    # Row 3: unchanged
    row3_cache = F1.build_row3(fig, outer[2, :])

    # Row 4: unchanged (+ saves colorbars)
    ax41 = fig.add_subplot(outer[3, 0])
    ax42 = fig.add_subplot(outer[3, 1])
    ax43 = fig.add_subplot(outer[3, 2])
    row4_cache = F1.build_row4_exact(ax41, ax42, ax43, args.mode, out_base_for_cbars=out_path)

    # Row 5: import from Figure 5 payload (preferred) or compute fallback
    ax51 = fig.add_subplot(outer[4, 0])
    ax52 = fig.add_subplot(outer[4, 1])
    ax53 = fig.add_subplot(outer[4, 2])

    fig5_info = None
    if args.row5_source == "fig5":
        try:
            fig5_info = build_row5_from_fig5_payload(
                ax51, ax52, ax53,
                fig5_dir,
                state=args.fig5_state,
                simlen=args.fig5_simlen,
                windows=args.fig5_windows,
            )
            row5_cache = fig5_info.get("row5_cache", {})
        except Exception as e:
            print(f"[WARN] Failed to load Row 5 from Figure 5 ({e}). Falling back to compute.")
            row5_cache = F1.build_row5_sim(ax51, ax52, ax53, sim_info, args.mode)
    else:
        row5_cache = F1.build_row5_sim(ax51, ax52, ax53, sim_info, args.mode)

    # Choose which sim_params to persist in the Figure-1 payload meta:
    # - if Row 5 came from Figure 5, store its exported SL_SD-fit params
    # - else store the Figure 4 params used for compute fallback
    sim_params_for_meta = {}
    if fig5_info is not None and isinstance(fig5_info.get("sim_params", None), dict):
        sim_params_for_meta = fig5_info["sim_params"]
    elif isinstance(sim_info.get("sim_params", None), dict):
        sim_params_for_meta = sim_info["sim_params"]

    # ---- Save payload (same as figure_1_final.py) ----
    run_config = dict(
        mode=args.mode,
        FS=F1.FS,
        NW=F1.NW,
        K_TAPERS=F1.K_TAPERS,
        WIN_DUR=F1.WIN_DUR,
        ANALYSIS_FRANGE=F1.ANALYSIS_FRANGE,
        SLOPE_BAND=F1.SLOPE_BAND,
        FOUR_MIN_SEC=F1.FOUR_MIN_SEC,
        RNG_SEED=F1.RNG_SEED,
        SP_KW=F1.SP_KW,
        SL_KW=F1.SL_KW,
        MT_PARAMS=F1.MT_PARAMS,
        row2_source=args.row2_source,
        fig4_dir=str(fig4_dir),
        fig4_state=args.fig4_state,
        row5_source=args.row5_source,
        fig5_dir=str(fig5_dir),
        fig5_state=args.fig5_state,
        fig5_simlen=args.fig5_simlen,
        fig5_windows=args.fig5_windows,
    )

    arrays_sections = dict(
        row1=row1_cache,
        row2=sim_info.get("row2_cache", {}),
        row3=row3_cache,
        row4=row4_cache,
        row5=row5_cache,
    )
    meta_sections = dict(
        run_config=run_config,
        sim_params=sim_params_for_meta,
    )

    npz_path, meta_path = F1._save_plot_payload(
        out_path,
        args.mode,
        arrays_sections=arrays_sections,
        meta_sections=meta_sections,
    )

    # ---- Save figures ----
    fig.savefig(out_path, dpi=args.dpi)
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, dpi=args.dpi)

    # Save sim params JSON (for traceability; whichever source produced Row5)
    base = out_path.with_suffix("")
    params_path = Path(str(base) + f".{args.mode}.row5_sim_params.json")
    with open(params_path, "w") as f:
        json.dump(sim_params_for_meta, f, indent=2, default=F1._json_default)

    # ---- Methods/value report ----
    try:
        report_path, report_lines = F1._build_methods_report(
            out_path,
            args.mode,
            row1_cache=row1_cache,
            row2_cache=sim_info.get("row2_cache", {}),
            row3_cache=row3_cache,
            row4_cache=row4_cache,
            row5_cache=row5_cache,
            sim_params=sim_params_for_meta,
            payload_paths={"npz": str(npz_path), "meta_json": str(meta_path)},
            include_file_hashes=(not args.no_hash),
        )
        F1._write_report(report_path, report_lines)
    except Exception as e:
        report_path = out_path.with_suffix(".methods_report.txt")
        lines = [
            f"Figure 1 edit/import wrapper report (fallback) — {datetime.now().isoformat(timespec='seconds')}",
            "",
            "WARNING: figure_1_final._build_methods_report raised an exception.",
            f"Exception: {repr(e)}",
            "",
            "Run config:",
            json.dumps(run_config, indent=2, default=F1._json_default),
            "",
            "Payload paths:",
            json.dumps({"npz": str(npz_path), "meta_json": str(meta_path)}, indent=2),
            "",
        ]
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

    print(f"[INFO] Saved figure → {out_path}")
    print(f"[INFO] Saved figure → {svg_path}")
    print(f"[INFO] Saved payload → {npz_path}")
    print(f"[INFO] Saved payload meta → {meta_path}")
    print(f"[INFO] Saved sim params → {params_path}")
    print(f"[INFO] Saved report → {report_path}")
    print(f"[INFO] Saved → {out_path.with_suffix('.row4_cbar_mt.svg')}")
    print(f"[INFO] Saved → {out_path.with_suffix('.row4_cbar_th.svg')}")


if __name__ == "__main__":
    main()
