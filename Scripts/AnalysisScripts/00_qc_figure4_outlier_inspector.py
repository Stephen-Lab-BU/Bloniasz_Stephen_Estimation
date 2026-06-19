#!/usr/bin/env python3
"""
Run quality-control diagnostics for Figure 4 artifact-window exclusions.

The script loads Figure 4 plot payloads, reconstructs the corresponding raw
ECoG windows, fits specparam and SL_specdecomp on the matched frequency grid,
and writes diagnostic panels plus a CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat


PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# Multitaper PSD (same stack used in Figure 4)
from spectral_connectivity import Multitaper, Connectivity

# same SL_SD implementation used in Figure 4
from SL_specdecomp import Decompose

# same specparam class used in Figure 4
from specparam import SpectralModel


# ──────────────────────────── Plot style (match Figure 4) ────────────────────────────
def apply_style() -> None:
    mpl.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.grid": False,
        "legend.fontsize": 7,
    })
    sns.set(context="talk", style="white")


COLORS = {
    "emp": "0.35",
    "full": "#000000",
    "broad": "#ff7f0e",
    "rhythms": "#d62728",
    "overlay": "0.6",
    "specparam_pt": "#1f77b4",
    "slsd_pt": "#2ca02c",
    "fwhm": "#6a3d9a",
}
STYLES = {
    "emp": dict(lw=2.0, alpha=0.65, solid_capstyle="round"),
    "full": dict(lw=2.0),
    "component": dict(lw=1.8, ls="--"),
}
PSD_YLIM = (1e-1, 1e6)


# ──────────────────────────── Figure 4 condition config (copied) ────────────────────────────
ANALYSIS_FRANGE = (0.1, 200.0)
HG_BAND = (80.0, 180.0)
SLOPE_BAND = (40.0, 60.0)

COND_TIME_KEY = "ConditionTime"
COND_LABEL_KEY = "ConditionLabel"
TIME_KEY_DEFAULT = "ECoGTime"

CONDITION_CFG = {
    "awake": {
        "start_phrase": "AwakeEyesClosed-Start",
        "end_phrase": "AwakeEyesClosed-End",
        "slsd_kwargs": dict(
            mode="additive",
            n_aperiodics=1,
            n_rhythms=2,
            rhythm_bands=[(8.0, 20.0), (20.0, 30.0)],
            sample_kwargs=dict(draws=1000, tune=1000, chains=2, target_accept=0.90, cores=1),
            plot=False,
        ),
        "specparam_kwargs": dict(
            aperiodic_mode="knee",
            peak_width_limits=[1.0, 30.0],
            max_n_peaks=2,
            min_peak_height=0.0,
            peak_threshold=2.0,
            verbose=False,
        ),
    },
    "anesthesia": {
        "start_phrase": "Anesthetized Start",
        "end_phrase": "Anesthetized End",
        "slsd_kwargs": dict(
            mode="additive",
            n_aperiodics=1,
            n_rhythms=3,
            rhythm_bands=[(0.1, 4.0), (8.0, 20.0), (20.0, 30.0)],
            sample_kwargs=dict(draws=1000, tune=1000, chains=2, target_accept=0.90, cores=1),
            plot=False,
        ),
        "specparam_kwargs": dict(
            aperiodic_mode="knee",
            peak_width_limits=[1.0, 30.0],
            max_n_peaks=3,
            min_peak_height=0.0,
            peak_threshold=2.0,
            verbose=False,
        ),
    },
}


# ──────────────────────────── Payload selection + key helpers ────────────────────────────
def pick_payload(in_dir: Path, condition: str, payload_mode: str) -> Path:
    in_dir = Path(in_dir)
    base_all = in_dir / f"Figure_4_CV_{condition}.plotdata.npz"
    base_2w = in_dir / f"Figure_4_CV_{condition}_2win.plotdata.npz"

    if payload_mode == "all":
        if not base_all.exists():
            raise FileNotFoundError(base_all)
        return base_all
    if payload_mode == "2win":
        if not base_2w.exists():
            raise FileNotFoundError(base_2w)
        return base_2w

    # auto
    if base_all.exists():
        return base_all
    if base_2w.exists():
        return base_2w
    raise FileNotFoundError(f"Missing payloads for {condition} in {in_dir}")


def safe_1d(z: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in z.files:
        raise KeyError(f"Missing key '{key}' in payload. Available (first 40): {z.files[:40]}")
    return np.asarray(z[key], float).ravel()


def topk_idx(x: np.ndarray, k: int, mode: str) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    ok = np.isfinite(x)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    v = x[idx]
    if mode == "largest":
        sel = idx[np.argsort(v)[::-1]]
    elif mode == "smallest":
        sel = idx[np.argsort(v)]
    elif mode == "abs":
        sel = idx[np.argsort(np.abs(v))[::-1]]
    else:
        raise ValueError("mode must be one of: largest, smallest, abs")
    return sel[:k].astype(int)


def compute_metric_from_payload(z: np.lib.npyio.NpzFile, metric: str) -> np.ndarray:
    """
    Supported metrics (payload keys):
      hg_slsd, hg_specparam, hg_diff
      slope_slsd, slope_specparam, slope_diff
      delta_cvll, cvll_slsd, cvll_specparam
    """
    cv_sp = safe_1d(z, "cvll_specparam") if "cvll_specparam" in z.files else None
    cv_sl = safe_1d(z, "cvll_slsd") if "cvll_slsd" in z.files else None
    hg_sp = safe_1d(z, "hg_specparam") if "hg_specparam" in z.files else None
    hg_sl = safe_1d(z, "hg_slsd") if "hg_slsd" in z.files else None
    sl_sp = safe_1d(z, "slopes_specparam") if "slopes_specparam" in z.files else None
    sl_sl = safe_1d(z, "slopes_slsd") if "slopes_slsd" in z.files else None

    if metric == "hg_slsd":
        if hg_sl is None:
            raise KeyError("hg_slsd not available in payload.")
        return hg_sl
    if metric == "hg_specparam":
        if hg_sp is None:
            raise KeyError("hg_specparam not available in payload.")
        return hg_sp
    if metric == "hg_diff":
        if hg_sp is None or hg_sl is None:
            raise KeyError("hg_diff needs both hg_specparam and hg_slsd.")
        return hg_sl - hg_sp

    if metric == "slope_slsd":
        if sl_sl is None:
            raise KeyError("slopes_slsd not available in payload.")
        return sl_sl
    if metric == "slope_specparam":
        if sl_sp is None:
            raise KeyError("slopes_specparam not available in payload.")
        return sl_sp
    if metric == "slope_diff":
        if sl_sp is None or sl_sl is None:
            raise KeyError("slope_diff needs both slopes_specparam and slopes_slsd.")
        return sl_sl - sl_sp

    if metric == "cvll_slsd":
        if cv_sl is None:
            raise KeyError("cvll_slsd not available in payload.")
        return cv_sl
    if metric == "cvll_specparam":
        if cv_sp is None:
            raise KeyError("cvll_specparam not available in payload.")
        return cv_sp
    if metric == "delta_cvll":
        if cv_sp is None or cv_sl is None:
            raise KeyError("delta_cvll needs both cvll_specparam and cvll_slsd.")
        return cv_sl - cv_sp

    raise ValueError(f"Unsupported metric: {metric}")


# ──────────────────────────── Condition parsing (same logic as Figure 4) ────────────────────────────
def _normalize(s: str) -> str:
    s = str(s).lower().replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def _find_interval(cond_times: np.ndarray, cond_labels: List[str],
                   start_phrase: str, end_phrase: str) -> Optional[Tuple[float, float]]:
    labs = [_normalize(l) for l in cond_labels]
    t = np.asarray(cond_times, float).ravel()
    s_norm = _normalize(start_phrase)
    e_norm = _normalize(end_phrase)
    start_idx = next((i for i, lab in enumerate(labs) if s_norm in lab), None)
    end_idx   = next((i for i, lab in enumerate(labs) if e_norm in lab), None)
    if start_idx is None or end_idx is None:
        return None
    t0, t1 = float(t[start_idx]), float(t[end_idx])
    if t1 <= t0:
        return None
    return (t0, t1)


def _restrict_to_interval(x_time: np.ndarray, x_val: np.ndarray,
                          t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
    m = (x_time >= t0) & (x_time <= t1)
    return x_time[m], x_val[m]


def load_ecog_time_cond(
    raw_mat: str,
    sig_key: str,
    time_mat: str,
    time_key: str,
    cond_mat: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    ecog = loadmat(raw_mat, squeeze_me=True)
    if sig_key not in ecog:
        avail = [k for k in ecog.keys() if not k.startswith("__")]
        raise KeyError(f"sig_key='{sig_key}' not in {Path(raw_mat).name}. Available: {avail[:50]}")
    x = np.asarray(ecog[sig_key], float).squeeze()

    tm = loadmat(time_mat, squeeze_me=True)
    if time_key not in tm:
        avail = [k for k in tm.keys() if not k.startswith("__")]
        raise KeyError(f"time_key='{time_key}' not in {Path(time_mat).name}. Available: {avail[:50]}")
    t = np.asarray(tm[time_key], float).squeeze()

    mvalid = np.isfinite(x) & np.isfinite(t)
    x, t = x[mvalid], t[mvalid]

    cond = loadmat(cond_mat, simplify_cells=True)["Condition"]
    ct = np.asarray(cond[COND_TIME_KEY], float).ravel()
    raw_labels = np.ravel(cond[COND_LABEL_KEY])
    labels = [lab.decode("utf-8") if isinstance(lab, (bytes, bytearray)) else str(lab) for lab in raw_labels]
    order = np.argsort(ct)
    ct, labels = ct[order], [labels[i] for i in order]
    return x, t, ct, labels


def extract_condition_segment(
    x: np.ndarray,
    t: np.ndarray,
    ct: np.ndarray,
    labels: List[str],
    *,
    start_phrase: str,
    end_phrase: str,
) -> Tuple[np.ndarray, np.ndarray]:
    seg = _find_interval(ct, labels, start_phrase, end_phrase)
    if seg is None:
        raise RuntimeError(f"Could not find interval: '{start_phrase}' ... '{end_phrase}'")
    t0, t1 = seg
    t_seg, x_seg = _restrict_to_interval(t, x, t0, t1)
    if x_seg.size == 0:
        raise RuntimeError("Segment restriction produced empty signal.")
    return x_seg, t_seg


def extract_window_from_segment(x_seg: np.ndarray, fs: float, win_sec: float, win_index: int) -> Tuple[np.ndarray, np.ndarray]:
    n_win = int(round(win_sec * fs))
    i0 = int(round(win_index * win_sec * fs))
    i1 = i0 + n_win
    if i0 < 0 or i1 > x_seg.size:
        raise ValueError(f"Window {win_index} out of bounds on segment: i0={i0}, i1={i1}, segN={x_seg.size}")
    x_win = np.asarray(x_seg[i0:i1], float).ravel()
    t_win = np.arange(x_win.size, dtype=float) / float(fs)
    return t_win, x_win


# ──────────────────────────── Multitaper PSD (one window) ────────────────────────────
def mt_power_one_window(ts_1d: np.ndarray, fs: float, duration: float, nw: float, k_tapers: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(ts_1d, float).ravel()[:, np.newaxis, np.newaxis]
    mt = Multitaper(
        x,
        sampling_frequency=float(fs),
        n_tapers=int(k_tapers),
        time_halfbandwidth_product=float(nw),
        start_time=0.0,
        time_window_duration=float(duration),
        time_window_step=float(duration),
    )
    conn = Connectivity.from_multitaper(mt)
    f_emp = np.asarray(conn.frequencies, float).ravel()
    S_emp = np.asarray(conn.power().squeeze(), float).ravel()
    return f_emp, S_emp


def interp_psd_to_grid(f_emp: np.ndarray, S_emp: np.ndarray, f_grid: np.ndarray) -> np.ndarray:
    f_emp = np.asarray(f_emp, float).ravel()
    S_emp = np.asarray(S_emp, float).ravel()
    f_grid = np.asarray(f_grid, float).ravel()

    m = np.isfinite(f_emp) & np.isfinite(S_emp) & (f_emp > 0) & (S_emp > 0)
    if m.sum() < 5:
        raise RuntimeError("Too few finite positive PSD points to interpolate.")
    f_emp = f_emp[m]
    S_emp = S_emp[m]

    order = np.argsort(f_emp)
    f_emp = f_emp[order]
    S_emp = S_emp[order]

    y = np.interp(f_grid, f_emp, S_emp, left=np.nan, right=np.nan)

    # fill edge NaNs by nearest finite
    good = np.isfinite(y)
    if not np.all(good):
        idx = np.where(good)[0]
        if idx.size == 0:
            raise RuntimeError("Interpolation produced all NaNs.")
        y[:idx[0]] = y[idx[0]]
        y[idx[-1] + 1:] = y[idx[-1]]

    # final safety for specparam
    y = np.asarray(y, float)
    y[~np.isfinite(y)] = np.nan
    if np.any(np.isfinite(y) & (y > 0)):
        minpos = float(np.nanmin(y[(y > 0) & np.isfinite(y)]))
    else:
        minpos = 1e-20
    y = np.nan_to_num(y, nan=minpos, posinf=minpos, neginf=minpos)
    y = np.clip(y, 1e-20, np.inf)
    return y


# ──────────────────────────── specparam fit (robust, linear-power API) ────────────────────────────
def specparam_full_aper(freqs_fit: np.ndarray, power_lin: np.ndarray, freq_range: Tuple[float, float], **specparam_kwargs):
    fm = SpectralModel(**specparam_kwargs)
    freqs_fit = np.asarray(freqs_fit, float)
    power_lin = np.clip(np.asarray(power_lin, float), 1e-20, np.inf)

    # IMPORTANT: specparam expects linear power here; it will log internally.
    fm.fit(freqs_fit, power_lin, freq_range=freq_range)

    # Try robust extraction across specparam versions
    model_obj = getattr(getattr(fm, "results", None), "model", None)
    if model_obj is not None:
        full_attr = getattr(model_obj, "modeled_spectrum", None)
        if callable(full_attr):
            try:
                full_native = np.asarray(full_attr(space="linear"), float).ravel()
            except TypeError:
                full_native = 10.0 ** np.asarray(full_attr(), float).ravel()
        else:
            full_native = 10.0 ** np.asarray(full_attr, float).ravel()

        get_comp = getattr(model_obj, "get_component", None)
        try:
            ap_native = np.asarray(get_comp("aperiodic", space="linear"), float).ravel()
        except TypeError:
            ap_native = 10.0 ** np.asarray(get_comp("aperiodic"), float).ravel()

        freq_model = None
        for cand in ("freqs", "freqs_model", "_freqs", "_spectrum_freqs"):
            arr = getattr(model_obj, cand, None)
            if arr is not None and np.size(arr) == full_native.size:
                freq_model = np.asarray(arr, float).ravel()
                break
        if freq_model is None:
            freq_model = np.asarray(getattr(fm, "freqs", freqs_fit), float).ravel()

        full_lin = np.interp(freqs_fit, freq_model, full_native)
        ap_lin = np.interp(freqs_fit, freq_model, ap_native)
    else:
        try:
            full_lin_native = np.asarray(fm.get_model("full", space="linear"), float).ravel()
        except Exception:
            full_lin_native = 10.0 ** np.asarray(fm.get_model("full"), float).ravel()
        try:
            ap_lin_native = np.asarray(fm.get_model("aperiodic", space="linear"), float).ravel()
        except Exception:
            ap_lin_native = 10.0 ** np.asarray(fm.get_model("aperiodic"), float).ravel()

        freq_model = np.asarray(getattr(fm, "freqs", freqs_fit), float).ravel()
        full_lin = np.interp(freqs_fit, freq_model, full_lin_native)
        ap_lin = np.interp(freqs_fit, freq_model, ap_lin_native)

    rh_lin = np.clip(full_lin - ap_lin, 0.0, np.inf)
    return full_lin, ap_lin, rh_lin


# ──────────────────────────── SL_SD extraction (same as Fig4) ────────────────────────────
def extract_slsd(model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.asarray(getattr(model, "estimated_spectrum"), float).reshape(-1)
    F = total.size

    bb = getattr(model, "broadband", None)
    if bb is None:
        bb = getattr(model, "P_ap", None)
    if bb is None:
        comps = getattr(model, "broadband_components", None)
        if comps is not None:
            bb = np.sum(np.asarray(comps, float), axis=0)
    if bb is None:
        bb = np.zeros_like(total)
    else:
        bb = np.asarray(bb, float).reshape(-1)
        if bb.size != F:
            bb = bb[:F] if bb.size > F else np.pad(bb, (0, F - bb.size))

    rh = getattr(model, "rhythms", None)
    if rh is None:
        rh = getattr(model, "P_rh", None)
    if rh is None:
        rh = getattr(model, "rhythms_total", None)
    if rh is None:
        rh = np.clip(total - bb, 0.0, np.inf)
    else:
        rh = np.asarray(rh, float).reshape(-1)
        if rh.size != F:
            rh = rh[:F] if rh.size > F else np.pad(rh, (0, F - rh.size))

    return total, bb, rh


# ──────────────────────────── Plotting ────────────────────────────
def savefig(fig: plt.Figure, out_png: Path) -> None:
    out_svg = out_png.with_suffix(".svg")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_outlier_panel(
    *,
    condition: str,
    win_index: int,
    metric_name: str,
    metric_value: float,
    cvll_specparam: float,
    cvll_slsd: float,
    t_center_min: float,
    t_win: np.ndarray,
    x_win: np.ndarray,
    f_fit: np.ndarray,
    psd_fit: np.ndarray,
    psd_payload_all: Optional[np.ndarray],
    sp_full: Optional[np.ndarray],
    sp_ap: Optional[np.ndarray],
    sp_rh: Optional[np.ndarray],
    sl_full: Optional[np.ndarray],
    sl_bb: Optional[np.ndarray],
    sl_rh: Optional[np.ndarray],
    outpath: Path,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.5))

    # A) Time series
    ax = axs[0, 0]
    ax.plot(t_win, x_win, color="0.2", lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (a.u.)")
    ax.set_title(f"{condition} — window {win_index} (center {t_center_min:.2f} min)")
    sns.despine(ax=ax, top=True, right=True)

    # B) PSD (overlay all windows from payload if available)
    ax = axs[0, 1]
    if psd_payload_all is not None and psd_payload_all.ndim == 2 and psd_payload_all.shape[1] == f_fit.size:
        for i in range(psd_payload_all.shape[0]):
            ax.plot(f_fit, np.clip(psd_payload_all[i, :], 1e-20, np.inf), color="0.80", lw=1.0, alpha=0.18)
    ax.plot(f_fit, psd_fit, color=COLORS["emp"], **STYLES["emp"], label="Multitaper (window)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(*PSD_YLIM)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Power spectrum (multitaper; Figure 4 grid)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    sns.despine(ax=ax, top=True, right=True)

    # C) specparam decomposition
    ax = axs[1, 0]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(*PSD_YLIM)
    ax.plot(f_fit, psd_fit, color=COLORS["emp"], **STYLES["emp"], label="Multitaper (window)")
    if sp_full is not None:
        ax.plot(f_fit, sp_full, color=COLORS["full"], **STYLES["full"], label="specparam full")
    if sp_ap is not None:
        ax.plot(f_fit, sp_ap, color=COLORS["broad"], **STYLES["component"], label="aperiodic")
    if sp_rh is not None:
        ax.plot(f_fit, sp_rh, color=COLORS["rhythms"], **STYLES["component"], label="rhythms")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("specparam decomposition")
    ax.legend(frameon=False, fontsize=9, loc="best")
    sns.despine(ax=ax, top=True, right=True)

    # D) SL_SD decomposition
    ax = axs[1, 1]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(*PSD_YLIM)
    ax.plot(f_fit, psd_fit, color=COLORS["emp"], **STYLES["emp"], label="Multitaper (window)")
    if sl_full is not None:
        ax.plot(f_fit, sl_full, color=COLORS["full"], **STYLES["full"], label="SL_SD full")
    if sl_bb is not None:
        ax.plot(f_fit, sl_bb, color=COLORS["broad"], **STYLES["component"], label="broadband")
    if sl_rh is not None:
        ax.plot(f_fit, sl_rh, color=COLORS["rhythms"], **STYLES["component"], label="rhythms")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("SL_SD decomposition")
    ax.legend(frameon=False, fontsize=9, loc="best")
    sns.despine(ax=ax, top=True, right=True)

    dcv = cvll_slsd - cvll_specparam
    fig.suptitle(
        f"Outlier inspector — {condition} | window {win_index} | "
        f"{metric_name}={metric_value:.4g} | "
        f"CVLL(specparam)={cvll_specparam:.3g}  CVLL(SL_SD)={cvll_slsd:.3g}  ΔCVLL={dcv:.3g}",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, outpath)


# ──────────────────────────── CSV ────────────────────────────
def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ──────────────────────────── Main per-condition processing ────────────────────────────
def process_condition(
    *,
    condition: str,
    payload_path: Path,
    raw_mat: str,
    time_mat: str,
    cond_mat: str,
    sig_key: str,
    time_key: str,
    fs: float,
    win_sec: float,
    mt_nw: float,
    mt_k: int,
    metric: str,
    mode: str,
    k_outliers: int,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    cfg = CONDITION_CFG[condition]

    z = np.load(payload_path, allow_pickle=True)

    # Required payload keys for grid + CVLL
    f_fit = safe_1d(z, "F_fit")
    cv_sp = safe_1d(z, "cvll_specparam")
    cv_sl = safe_1d(z, "cvll_slsd")

    # all-window PSDs on Figure 4 grid 
    psd_payload_all = None
    if "P_fit_tf" in z.files:
        arr = np.asarray(z["P_fit_tf"], float)
        if arr.ndim == 2 and arr.shape[1] == f_fit.size:
            psd_payload_all = np.clip(arr, 1e-20, np.inf)

    # Metric array (same window axis)
    met = compute_metric_from_payload(z, metric=metric)
    n_win = int(min(cv_sp.size, cv_sl.size, met.size))
    cv_sp = cv_sp[:n_win]
    cv_sl = cv_sl[:n_win]
    met = met[:n_win]

    out_idx = topk_idx(met, k=k_outliers, mode=mode)

    # Load raw + time + condition markers; reconstruct segment
    x, t, ct, labels = load_ecog_time_cond(
        raw_mat=raw_mat,
        sig_key=sig_key,
        time_mat=time_mat,
        time_key=time_key,
        cond_mat=cond_mat,
    )
    x_seg, t_seg = extract_condition_segment(
        x=x, t=t, ct=ct, labels=labels,
        start_phrase=cfg["start_phrase"],
        end_phrase=cfg["end_phrase"],
    )

    # Window center times (minutes) relative to segment start
    # For 30 s non-overlapping windows, center is (i + 0.5)*win_sec
    centers_min = (np.arange(n_win, dtype=float) + 0.5) * (win_sec / 60.0)

    rows: List[Dict[str, Any]] = []
    print(f"\n[{condition}] payload={payload_path.name}")
    print(f"[{condition}] selecting outliers by '{metric}' ({mode}), k={k_outliers}: {list(map(int, out_idx))}")

    # Fit + plot each outlier
    for rank, wi in enumerate(out_idx, start=1):
        wi = int(wi)
        t_center_min = float(centers_min[wi])

        # Extract window time series (exactly as used for CVLL extraction in Fig4)
        t_win, x_win = extract_window_from_segment(x_seg, fs=fs, win_sec=win_sec, win_index=wi)

        # Multitaper PSD on native grid, then interpolate to Figure 4 grid (F_fit)
        f_emp, S_emp = mt_power_one_window(x_win, fs=fs, duration=win_sec, nw=mt_nw, k_tapers=mt_k)

        # Restrict/interp to Figure 4 grid
        psd_fit = interp_psd_to_grid(f_emp, S_emp, f_fit)

        # Fit specparam (linear power API; avoid log10 input)
        fr_k = (max(ANALYSIS_FRANGE[0], float(f_fit[0])), min(ANALYSIS_FRANGE[1], float(f_fit[-1])))
        sp_full = sp_ap = sp_rh = None
        try:
            sp_full, sp_ap, sp_rh = specparam_full_aper(f_fit, psd_fit, fr_k, **cfg["specparam_kwargs"])
        except Exception as e:
            print(f"[{condition}] window {wi}: specparam fit failed: {e}")

        # Fit SL_SD (same as Figure 4)
        sl_full = sl_bb = sl_rh = None
        try:
            sl = Decompose(f_fit, psd_fit, fs=fs, **cfg["slsd_kwargs"])
            sl_full, sl_bb, sl_rh = extract_slsd(sl)
        except Exception as e:
            print(f"[{condition}] window {wi}: SL_SD fit failed: {e}")

        # Plot panel
        out_plot = out_dir / f"outlier_{condition}_win{wi:04d}_{metric}_{mode}_rank{rank}.png"
        plot_outlier_panel(
            condition=condition,
            win_index=wi,
            metric_name=metric,
            metric_value=float(met[wi]),
            cvll_specparam=float(cv_sp[wi]),
            cvll_slsd=float(cv_sl[wi]),
            t_center_min=t_center_min,
            t_win=t_win,
            x_win=x_win,
            f_fit=f_fit,
            psd_fit=psd_fit,
            psd_payload_all=psd_payload_all,
            sp_full=sp_full,
            sp_ap=sp_ap,
            sp_rh=sp_rh,
            sl_full=sl_full,
            sl_bb=sl_bb,
            sl_rh=sl_rh,
            outpath=out_plot,
        )

        row = dict(
            condition=condition,
            outlier_rank=rank,
            window_index=wi,
            t_center_min=t_center_min,
            metric=metric,
            metric_value=float(met[wi]),
            cvll_specparam=float(cv_sp[wi]),
            cvll_slsd=float(cv_sl[wi]),
            delta_cvll=float(cv_sl[wi] - cv_sp[wi]),
            payload=str(payload_path.name),
            created_local=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        rows.append(row)

    return rows


# ──────────────────────────── CLI ────────────────────────────
def main() -> None:
    apply_style()

    ap = argparse.ArgumentParser(description="Figure 4 outlier inspector (TS + MT PSD + specparam + SL_SD).")
    ap.add_argument("--in-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--payload-mode", choices=["auto", "all", "2win"], default="auto")

    # Keep the existing interface (awake/anesthesia raw paths), but we reconstruct segments using shared time/cond files.
    ap.add_argument("--awake-raw", type=str, required=True)
    ap.add_argument("--anes-raw", type=str, required=True)
    ap.add_argument("--sig-key", type=str, required=True)
    ap.add_argument("--fs", type=float, required=True)
    ap.add_argument("--win-sec", type=float, default=30.0)

    # Default companion files (override if the analysis requires)
    ap.add_argument("--time-mat", type=str, default=os.path.expanduser("CHANGE_THIS_ROOT_TO_PATH/Bloniasz_Stephen_Estimation/Data/InputData/InputDataFiles/ECoGTime.mat"))
    ap.add_argument("--time-key", type=str, default=TIME_KEY_DEFAULT)
    ap.add_argument("--cond-mat", type=str, default=os.path.expanduser("CHANGE_THIS_ROOT_TO_PATH/Bloniasz_Stephen_Estimation/Data/InputData/InputDataFiles/Condition.mat"))

    # Outlier selection
    ap.add_argument("--metric", type=str, default="hg_slsd",
                    help="hg_slsd, hg_specparam, hg_diff, slope_slsd, slope_specparam, slope_diff, delta_cvll, cvll_slsd, cvll_specparam")
    ap.add_argument("--mode", choices=["largest", "smallest", "abs"], default="largest")
    ap.add_argument("--k", type=int, default=2)

    # Which condition(s)
    ap.add_argument("--which", choices=["both", "awake", "anesthesia"], default="both")

    # MT PSD params for the outlier window PSD
    ap.add_argument("--mt-nw", type=float, default=2.0)
    ap.add_argument("--mt-k", type=int, default=3)

    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser()
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (in_dir / "outlier_inspector")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    if args.which in ("both", "awake"):
        awake_npz = pick_payload(in_dir, "awake", args.payload_mode)
        rows += process_condition(
            condition="awake",
            payload_path=awake_npz,
            raw_mat=args.awake_raw,
            time_mat=args.time_mat,
            cond_mat=args.cond_mat,
            sig_key=args.sig_key,
            time_key=args.time_key,
            fs=float(args.fs),
            win_sec=float(args.win_sec),
            mt_nw=float(args.mt_nw),
            mt_k=int(args.mt_k),
            metric=str(args.metric),
            mode=str(args.mode),
            k_outliers=int(args.k),
            out_dir=out_dir,
        )

    if args.which in ("both", "anesthesia"):
        anes_npz = pick_payload(in_dir, "anesthesia", args.payload_mode)
        rows += process_condition(
            condition="anesthesia",
            payload_path=anes_npz,
            raw_mat=args.anes_raw,
            time_mat=args.time_mat,
            cond_mat=args.cond_mat,
            sig_key=args.sig_key,
            time_key=args.time_key,
            fs=float(args.fs),
            win_sec=float(args.win_sec),
            mt_nw=float(args.mt_nw),
            mt_k=int(args.mt_k),
            metric=str(args.metric),
            mode=str(args.mode),
            k_outliers=int(args.k),
            out_dir=out_dir,
        )

    out_csv = out_dir / "outliers_report.csv"
    write_csv(rows, out_csv)

    manifest = {
        "created_local": datetime.datetime.now().isoformat(timespec="seconds"),
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "payload_mode": args.payload_mode,
        "metric": args.metric,
        "mode": args.mode,
        "k": int(args.k),
        "which": args.which,
        "fs": float(args.fs),
        "win_sec": float(args.win_sec),
        "mt_nw": float(args.mt_nw),
        "mt_k": int(args.mt_k),
        "time_mat": str(Path(args.time_mat).expanduser()),
        "time_key": args.time_key,
        "cond_mat": str(Path(args.cond_mat).expanduser()),
        "awake_raw": str(Path(args.awake_raw).expanduser()),
        "anes_raw": str(Path(args.anes_raw).expanduser()),
        "sig_key": args.sig_key,
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[saved] {out_csv}")
    print(f"[saved] plots + manifest in {out_dir}")


if __name__ == "__main__":
    main()
