#!/usr/bin/env python3
"""
Generate the canonical Figure 1 overview grid and reproducibility payloads.

The script builds the overview figure, saves the arrays plotted in each panel,
and writes a methods-oriented report of the simulation parameters and summary
statistics needed to reproduce the manuscript figure.
"""

from __future__ import annotations

import os
from pathlib import Path
import argparse
import re
import json
import platform
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import gamma
from scipy.interpolate import interp1d

from SL_GPsim import spectrum
from spectral_connectivity import Multitaper, Connectivity
from SL_specdecomp import Decompose
from specparam import SpectralModel



PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# ---------------------- Style ----------------------
mpl.rcParams.update({
    "svg.fonttype": "none",          # keep text editable in SVGs
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 7,
    "lines.linewidth": 1.6,
})
sns.set_style("ticks")

# --- Consistent colors, linestyles, and zorder everywhere ---
COLORS  = dict(multitaper="0.45", full="C2", broadband="C1", rhythms="C3", truth="k")
LINES   = dict(multitaper="-",     full="-",  broadband="--", rhythms="--", truth="-")
ZORDER  = dict(truth=4, full=3, multitaper=-10, broadband=2, rhythms=1)


def plot_ll(ax, x, y, kind, label=None, log=True, **kw):
    f = ax.loglog if log else ax.plot
    # Allow overrides from **kw
    color  = kw.pop("color",  COLORS[kind])
    ls     = kw.pop("ls",     LINES[kind])
    zorder = kw.pop("zorder", ZORDER.get(kind, 5))
    return f(x, y, color=color, ls=ls, zorder=zorder, label=label, **kw)


# ---------------------- Paths ----------------------
DEFAULT_OUT_DIR = PROJECT_ROOT / 'Output' / 'Results' / 'FiguresIntermediate' / 'Figure_1' / 'Figure_output'
DATA_DIR = PROJECT_ROOT / 'Data'
ECOG_MAT = DATA_DIR / "ECoG_ch1.mat"
TIME_MAT = DATA_DIR / "ECoGTime.mat"
COND_MAT = DATA_DIR / "Condition.mat"


# ---------------------- Config ----------------------
FS               = 1000.0
NW               = 2
K_TAPERS         = 3
WIN_DUR          = 30.0
ANALYSIS_FRANGE  = (0.1, 200.0)
SLOPE_BAND       = (40.0, 60.0)
FOUR_MIN_SEC     = 4.0 * 60.0  # 4 minutes

# simulation for row 1 legacy + flowchart pieces
SIM_APER_EXP     = 1.5
SIM_APER_OFF     = 1.0
SIM_KNEE         = 10.0
SIM_PEAKS        = [dict(freq=12.0, amplitude=2.0, sigma=1.0)]
RNG_SEED         = 0

# Specparam config
SP_KW = dict(
    peak_width_limits=[1.0, 30.0],
    max_n_peaks=3,
    min_peak_height=0.0,
    peak_threshold=2.0,
    aperiodic_mode="knee",
    verbose=False,
)

# SL_specdecomp config (match anesthesia bands)
SL_KW = dict(
    mode="additive",
    n_aperiodics=1,
    n_rhythms=3,
    rhythm_bands=[(0.1, 4.0), (8.0, 20.0), (20.0, 30.0)],
    sample_kwargs=dict(draws=800, tune=800, chains=2, target_accept=0.90),
    plot=False,
)

MT_PARAMS = dict(
    time_halfbandwidth_product=NW,
    n_tapers=K_TAPERS,
    time_window_duration=WIN_DUR,
    time_window_step=WIN_DUR,
)


# ====================== Persistence helpers ======================
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        # avoid huge dumps; only allow small arrays
        if o.size <= 50:
            return o.tolist()
        return dict(__ndarray__=True, shape=list(o.shape), dtype=str(o.dtype))
    return str(o)


def _coerce_array(x):
    if x is None:
        return np.array([])
    if isinstance(x, (float, int, bool, np.number)):
        return np.array([x])
    try:
        arr = np.asarray(x)
        # keep object arrays out of npz; those belong in JSON metadata
        if arr.dtype == object:
            return np.array([])
        return arr
    except Exception:
        return np.array([])


def _save_plot_payload(out_path: Path, mode: str, *, arrays_sections: dict, meta_sections: dict):
    """
    Save everything needed to rebuild the figure without recomputation.

    Writes next to the Figure:
      - <out_base>.<mode>.plotdata.npz (all numeric arrays, flattened keys)
      - <out_base>.<mode>.plotmeta.json (what we saved, shapes/dtypes, + non-numeric metadata)
    """
    out_path = Path(out_path)
    base = out_path.with_suffix("")  # remove extension

    flat = {}
    meta = {
        "mode": mode,
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "arrays": {},
        "meta": meta_sections if meta_sections is not None else {},
    }

    for sec, d in (arrays_sections or {}).items():
        if d is None:
            d = {}
        for k, v in d.items():
            key = f"{sec}__{k}"
            arr = _coerce_array(v)
            flat[key] = arr
            meta["arrays"][key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "size": int(arr.size),
            }

    npz_path = Path(str(base) + f".{mode}.plotdata.npz")
    meta_path = Path(str(base) + f".{mode}.plotmeta.json")

    np.savez_compressed(npz_path, **flat)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    return npz_path, meta_path


def _sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ====================== Utilities ======================
def _first_numeric_vec(d):
    for k, v in d.items():
        if k.startswith("__"):
            continue
        a = np.asarray(v)
        if np.issubdtype(a.dtype, np.number) and a.size > 1:
            return a.squeeze()
    raise RuntimeError("No numeric array found in MAT file.")


def _load_ecog_time():
    ecog = _first_numeric_vec(loadmat(ECOG_MAT))
    tvec = _first_numeric_vec(loadmat(TIME_MAT))
    x = np.asarray(ecog, float).ravel()
    t = np.asarray(tvec, float).ravel()
    if t.size >= 2:
        dt = np.median(np.diff(t))
        fs = float(np.round(1.0 / dt)) if dt > 0 else 1000.0
    else:
        fs = 1000.0
        t = np.arange(x.size, dtype=float) / fs
    m = np.isfinite(x) & np.isfinite(t)
    return x[m], t[m], fs


def _find_anesthesia_interval():
    cond = loadmat(COND_MAT, simplify_cells=True)["Condition"]
    ct = np.asarray(cond["ConditionTime"], float).ravel()
    raw_labels = np.ravel(cond["ConditionLabel"])
    labels = [lab.decode("utf-8") if isinstance(lab, (bytes, bytearray)) else str(lab) for lab in raw_labels]
    order = np.argsort(ct)
    ct, labels = ct[order], [labels[i] for i in order]
    start_idx = None
    end_idx = None
    for i, s in enumerate(labels):
        s_low = s.lower().replace("-", " ")
        if ("anesthetized" in s_low) and ("start" in s_low) and start_idx is None:
            start_idx = i
        if ("anesthetized" in s_low) and ("end" in s_low) and end_idx is None:
            end_idx = i
    if start_idx is None or end_idx is None or ct[end_idx] <= ct[start_idx]:
        raise RuntimeError("Could not find valid 'Anesthetized Start/End' in Condition.mat")
    return float(ct[start_idx]), float(ct[end_idx])


def _restrict_interval(t, x, t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], x[m]


def _b0_for_spectrum(params: dict) -> float:
    """
    Return aperiodic_offset in the *log10* space expected by spectral_decomposition.spectrum.

    spectrum uses: 10**aperiodic_offset → so offset must be log10.
    If incoming b_0 is linear, convert via log10 defensively.
    """
    b0 = float(params.get("b_0", 0.0))
    space = str(params.get("b0_space", "log10")).lower()
    if space == "log10":
        return b0
    return float(np.log10(max(b0, 1e-300)))


def _multitaper(ts_1d, fs, start_time=0.0):
    ts3d = np.asarray(ts_1d).reshape(-1, 1, 1)
    mt = Multitaper(
        time_series=ts3d,
        sampling_frequency=fs,
        start_time=float(start_time),
        **MT_PARAMS,
    )
    conn = Connectivity.from_multitaper(mt)
    P = conn.power().squeeze()  # (T, F)
    F = np.asarray(conn.frequencies).ravel()
    T = np.asarray(conn.time).ravel()
    return P, F, T


def _pick_idx_at_time(T, rel_sec=FOUR_MIN_SEC):
    """Pick index of window whose time is nearest to (T.min + rel_sec)."""
    T = np.asarray(T, float).ravel()
    target = T.min() + float(rel_sec)
    return int(np.argmin(np.abs(T - target)))


def _independent_grid(freqs, twin, NW):
    f = np.asarray(freqs, float)
    df = float(np.median(np.diff(f))) if f.size > 1 else 1.0
    delta_f_indep = 2.0 * NW / twin
    step_bins = max(1, int(round(delta_f_indep / max(df, 1e-12))))
    return f[::step_bins], step_bins


def _compute_loglog_slope(freqs, power_lin, fmin=40.0, fmax=60.0):
    f = np.asarray(freqs, float)
    y = np.asarray(power_lin, float)
    m = (f >= fmin) & (f <= fmax) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        return np.nan
    xf = np.log10(f[m])
    yf = np.log10(y[m])
    a, _b = np.polyfit(xf, yf, 1)
    return float(a)


# ---------- Specparam helpers ----------
def _specparam_components(freqs, power_lin, freq_range):
    fm = SpectralModel(**SP_KW)
    fm.fit(freqs, np.clip(power_lin, 1e-20, np.inf), freq_range=freq_range)

    def _align(y, xref):
        y = np.asarray(y, float)
        if y.ndim == 0:
            return np.full_like(xref, float(y))
        if len(y) == len(xref):
            return y
        x_src = np.linspace(xref[0], xref[-1], len(y))
        return np.interp(xref, x_src, y)

    fit_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    x_band = freqs[fit_mask]

    model_obj = getattr(getattr(fm, "results", None), "model", None)
    if model_obj is None:
        raise RuntimeError("specparam results.model not available.")

    # Full
    full_attr = getattr(model_obj, "modeled_spectrum", None)
    if callable(full_attr):
        try:
            full_lin = np.asarray(full_attr(space="linear"), float)
        except TypeError:
            full_lin = 10.0 ** np.asarray(full_attr(), float)
    else:
        full_lin = 10.0 ** np.asarray(full_attr, float)
    full_lin = _align(full_lin, x_band)

    # Aperiodic
    get_comp = getattr(model_obj, "get_component", None)
    try:
        ap_lin = np.asarray(get_comp("aperiodic", space="linear"), float)
    except TypeError:
        ap_lin = 10.0 ** np.asarray(get_comp("aperiodic"), float)
    ap_lin = _align(ap_lin, x_band)

    # Combined peaks
    try:
        pk_lin = np.asarray(get_comp("peak", space="linear"), float)
    except TypeError:
        pk_lin = 10.0 ** np.asarray(get_comp("peak"), float)
    pk_lin = _align(pk_lin, x_band)

    return x_band, full_lin, ap_lin, pk_lin


# ---------- SL_specdecomp helpers ----------
def _slsd_components(freqs, power_lin, fs, **kw):
    model = Decompose(freqs, np.clip(power_lin, 1e-20, np.inf), fs=fs, **kw)
    total = np.asarray(getattr(model, "estimated_spectrum"), float).reshape(-1)

    bb = getattr(model, "broadband", None)
    if bb is None:
        bb = getattr(model, "P_ap", None)
    if bb is None:
        bb_comps = getattr(model, "broadband_components", None)
        if bb_comps is not None:
            bb = np.sum(np.asarray(bb_comps, float), axis=0)

    rh = getattr(model, "rhythms", None)
    if rh is None:
        rh = getattr(model, "P_rh", None)
    if rh is None:
        rh = getattr(model, "rhythms_total", None)

    F = total.size

    def _fix(x):
        if x is None:
            return None
        x = np.asarray(x, float).reshape(-1)
        if x.size != F:
            x = x[:F] if x.size > F else np.pad(x, (0, F - x.size))
        return x

    bb = _fix(bb)
    rh = _fix(rh)

    if bb is None and rh is None:
        bb = total.copy()
        rh = np.zeros_like(total)
    elif bb is None:
        bb = np.clip(total - rh, 0.0, np.inf)
    elif rh is None:
        rh = np.clip(total - bb, 0.0, np.inf)

    return model, total, bb, rh


def _guess_idata(model):
    for attr in ("idata", "trace", "posterior"):
        if hasattr(model, attr):
            cand = getattr(model, attr)
            if hasattr(cand, "posterior"):
                return cand
    for attr in ("m", "model"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            if hasattr(m, "idata"):
                return getattr(m, "idata")
    return None


def _posterior_mean(idata, name):
    if idata is None or not hasattr(idata, "posterior"):
        return None
    if hasattr(idata.posterior, "data_vars") and (name in idata.posterior.data_vars):
        vals = idata.posterior[name].values
        return float(np.nanmean(vals))
    return None


def _collect_slsd_params(model, freqs=None, bb_avg=None):
    """
    Collect posterior-mean parameters from SL_specdecomp fit (enough to re-run simulation).
    Also includes peaks_list (freq, sigma, amplitude).
    """
    idata = _guess_idata(model)
    out = {}

    # aperiodic
    for nm in ("knee_0", "knee", "k_0"):
        v = _posterior_mean(idata, nm)
        if v is not None:
            out["knee_0"] = v
            break

    alpha = _posterior_mean(idata, "alpha_eff")
    if alpha is None:
        alpha = _posterior_mean(idata, "alpha")
    slope0 = _posterior_mean(idata, "slope_0")
    if alpha is None and slope0 is not None:
        alpha = abs(float(slope0))
    out["slope_0"] = float(-alpha) if alpha is not None else (slope0 if slope0 is not None else -2.5)
    out["alpha"] = abs(out["slope_0"])

    b0 = _posterior_mean(idata, "b_0")
    if b0 is not None:
        out["b_0"] = float(b0)
    elif (freqs is not None) and (bb_avg is not None):
        f0 = 10.0
        ap_f0 = float(np.interp(f0, freqs, np.asarray(bb_avg, float)))
        knee = float(out.get("knee_0", 0.0))
        chi = float(out["alpha"])
        out["b_0"] = float(np.log10(max(ap_f0, 1e-20) * (knee + f0**chi)))
    else:
        out["b_0"] = 1.0

    out["b0_space"] = "log10"

    # rhythms
    names = set()
    if idata is not None and hasattr(idata, "posterior"):
        try:
            names = set(list(idata.posterior.data_vars))
        except Exception:
            names = set()

    def _grab(prefix):
        d = {}
        pat = re.compile(rf"{prefix}_(\d+)")
        for nm in names:
            m = pat.fullmatch(nm)
            if m:
                d[int(m.group(1))] = float(_posterior_mean(idata, nm))
        return d

    centers = _grab("center")
    sigmas = _grab("sigma")
    amps = _grab("A_lin")

    J = sorted(set(centers.keys()) | set(sigmas.keys()) | set(amps.keys()))
    peaks = []
    for j in J:
        c = centers.get(j)
        s = sigmas.get(j)
        a = amps.get(j)
        if (c is None) or (s is None) or (a is None):
            continue
        peaks.append(dict(freq=float(c), amplitude=float(a), sigma=float(s)))
    out["peaks_list"] = peaks
    return out


def _simulate_with_params(params, n_windows, fs=FS, seed0=12345):
    """
    Concatenate n_windows simulated 30 s chunks using the given params.
    Also return per-window ground-truth PSDs straight from each simulation object:
      gt_list[i] = {"f": frequencies, "full": combined_spectrum, "bb": broadband_spectrum}
    """
    chi = float(abs(params["slope_0"]))
    knee = float(params.get("knee_0", 0.0))
    b0 = _b0_for_spectrum(params)
    peaks = list(params.get("peaks_list", []))

    xs, gt_list = [], []
    for i in range(n_windows):
        res = spectrum(
            sampling_rate=fs,
            duration=WIN_DUR,
            aperiodic_exponent=chi,
            aperiodic_offset=b0,
            knee=knee,
            peaks=peaks,
            average_firing_rate=0.0,
            random_state=seed0 + i,
            direct_estimate=False,
            plot=False,
        )
        xs.append(np.asarray(res.time_domain.combined_signal, float).ravel())

        fd = res.frequency_domain
        f = np.asarray(fd.frequencies, float).ravel()
        full = np.asarray(fd.combined_spectrum, float).ravel()

        bb_attr = getattr(fd, "broadband_spectrum", None)
        if bb_attr is None:
            bb_attr = getattr(fd, "aperiodic_spectrum", None)
        if bb_attr is None:
            pk_attr = getattr(fd, "peaks_spectrum", None)
            if pk_attr is None:
                bb = full.copy()
            else:
                bb = np.asarray(full - np.asarray(pk_attr, float).ravel(), float)
        else:
            bb = np.asarray(bb_attr, float).ravel()

        gt_list.append({
            "f": f,
            "full": np.clip(full, 1e-20, np.inf),
            "bb": np.clip(bb, 1e-20, np.inf),
        })

    x = np.concatenate(xs) if len(xs) else np.zeros(int(WIN_DUR * fs), float)
    t = np.arange(x.size, dtype=float) / fs
    return x, t, gt_list


# ---------- Model packs ----------
def _avg_models_from_windows(P_tf, F, fs, sl_kw):
    """Independent grid → empirical average → Specparam & SL_SD (avg) + per-window slopes."""
    fmin, fmax = ANALYSIS_FRANGE
    mask = (F > 0) & (F >= fmin) & (F <= fmax)
    F_an = F[mask]
    P_an = P_tf[:, mask]

    F_fit, step_bins = _independent_grid(F_an, WIN_DUR, NW)
    P_fit = P_an[:, ::step_bins]
    emp_avg = np.nanmean(P_fit, axis=0)

    x_band, sp_full, sp_ap, sp_pk = _specparam_components(
        F_fit, emp_avg, [max(1.0, F_fit.min()), min(120.0, F_fit.max())]
    )

    k_eff = int(K_TAPERS * P_fit.shape[0])
    sl_kw_avg = dict(sl_kw)
    sl_kw_avg["k_tapers"] = k_eff
    sl_model, sl_total, sl_bb, sl_rh = _slsd_components(F_fit, emp_avg, fs, **sl_kw_avg)

    def _to_band(y):
        return np.interp(x_band, F_fit, np.asarray(y, float))

    emp_avg_band = _to_band(emp_avg)
    sl_total = _to_band(sl_total)
    sl_bb = _to_band(sl_bb)
    sl_rh = _to_band(sl_rh)

    slopes_sp, slopes_bb = [], []
    for row in P_fit:
        xb_i, _f_i, ap_i, _pk_i = _specparam_components(
            F_fit, row, [max(1.0, F_fit.min()), min(120.0, F_fit.max())]
        )
        _m_i, _tot_i, bb_i, _rh_i = _slsd_components(F_fit, row, fs, **sl_kw)
        bb_i = np.interp(xb_i, F_fit, bb_i)
        slopes_sp.append(_compute_loglog_slope(xb_i, ap_i, *SLOPE_BAND))
        slopes_bb.append(_compute_loglog_slope(xb_i, bb_i, *SLOPE_BAND))

    return dict(
        F_fit=F_fit,
        step_bins=step_bins,
        x=x_band,
        emp_avg=emp_avg_band,
        sp_full=sp_full,
        sp_ap=sp_ap,
        sp_pk=sp_pk,
        sl_model=sl_model,
        sl_total=sl_total,
        sl_bb=sl_bb,
        sl_rh=sl_rh,
        slopes_sp=np.asarray(slopes_sp),
        slopes_bb=np.asarray(slopes_bb),
        k_eff=k_eff,
        n_windows=P_fit.shape[0],
    )


def _single_window_models(P_tf, F, T, fs, sl_kw, rel_sec=FOUR_MIN_SEC):
    """Like above, but for the single window nearest the specified time."""
    fmin, fmax = ANALYSIS_FRANGE
    mask = (F > 0) & (F >= fmin) & (F <= fmax)
    F_an = F[mask]
    P_an = P_tf[:, mask]

    F_fit, step_bins = _independent_grid(F_an, WIN_DUR, NW)
    P_fit = P_an[:, ::step_bins]

    idx = _pick_idx_at_time(T, rel_sec=rel_sec)
    idx = max(0, min(idx, P_fit.shape[0] - 1))
    mt_this = P_fit[idx, :]

    x_band, sp_full, sp_ap, sp_pk = _specparam_components(
        F_fit, mt_this, [max(1.0, F_fit.min()), min(120.0, F_fit.max())]
    )
    _m, sl_total, sl_bb, sl_rh = _slsd_components(F_fit, mt_this, fs, **sl_kw)

    def _to_band(y):
        return np.interp(x_band, F_fit, np.asarray(y, float))

    return dict(
        x=x_band,
        mt_this=_to_band(mt_this),
        sp_full=sp_full,
        sp_ap=sp_ap,
        sp_pk=sp_pk,
        sl_total=_to_band(sl_total),
        sl_bb=_to_band(sl_bb),
        sl_rh=_to_band(sl_rh),
        chosen_time=float(np.asarray(T).ravel()[idx]) if len(np.atleast_1d(T)) else None,
        chosen_idx=int(idx),
        step_bins=int(step_bins),
    )


# ====================== Row builders ======================
def build_row1(ax):
    """Row 1: 3 s of *real* ECoG from the MIDDLE of the Anesthetized interval (time-domain)."""
    x, t, fs = _load_ecog_time()
    t0, t1 = _find_anesthesia_interval()
    t_an, x_an = _restrict_interval(t, x, t0, t1)

    mid = 0.5 * (t_an[0] + t_an[-1])
    half = 1.5
    start = max(t_an[0], mid - half)
    stop = min(t_an[-1], mid + half)

    m3 = (t_an >= start) & (t_an <= stop)
    tt = t_an[m3] - start
    xx = x_an[m3]

    ax.plot(tt, xx, lw=1.2)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("3 s ECoG (Anesthesia — middle)")

    return {
        "t": np.asarray(tt),
        "x": np.asarray(xx),
        "fs": float(fs),
        "anesthesia_t0": float(t0),
        "anesthesia_t1": float(t1),
        "window_abs_start": float(start),
        "window_abs_stop": float(stop),
    }


def _anesthesia_multitaper(mode):
    x, t, fs = _load_ecog_time()
    t0, t1 = _find_anesthesia_interval()
    t_an, x_an = _restrict_interval(t, x, t0, t1)

    if mode == "short":
        if (t_an[-1] - t_an[0]) >= 60.0:
            t_an, x_an = _restrict_interval(t_an, x_an, t_an[0], t_an[0] + 60.0)
            print("[Row2] Using first 60 s of anesthesia.")
        else:
            print("[Row2] Anesthesia <60 s; using full segment.")
    else:
        print("[Row2] Using full anesthesia segment.")

    P_tf, F, T = _multitaper(x_an, fs, float(t_an[0]))
    return P_tf, F, T, fs, float(t_an[0]), float(t_an[-1])


def build_row2_anesthesia(ax1, ax2, ax3, mode):
    """Empirical anesthesia pipeline. Cols 1-2 show single window near 4 min; col 3 shows slopes."""
    P_tf, F, T, fs, abs_t0, abs_t1 = _anesthesia_multitaper(mode)
    Ravg = _avg_models_from_windows(P_tf, F, fs, SL_KW)
    Rwin = _single_window_models(P_tf, F, T, fs, SL_KW, rel_sec=FOUR_MIN_SEC)

    # Left: Specparam (single window)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    plot_ll(ax1, Rwin["x"], Rwin["mt_this"], kind="multitaper", label="Multitaper", lw=2, alpha=0.8)
    plot_ll(ax1, Rwin["x"], Rwin["sp_full"], kind="full", label="Specparam full")
    plot_ll(ax1, Rwin["x"], Rwin["sp_ap"], kind="broadband", label="Specparam aperiodic")
    plot_ll(ax1, Rwin["x"], Rwin["sp_pk"], kind="rhythms", label="Specparam rhythms")
    ax1.set_title("Specparam")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power")
    ax1.grid(False)
    ax1.legend(frameon=False)
    ax1.set_ylim(1e-1, 1e6)

    # Middle: SL_SD (single window)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plot_ll(ax2, Rwin["x"], Rwin["mt_this"], kind="multitaper", label="Multitaper", lw=2, alpha=0.8)
    plot_ll(ax2, Rwin["x"], Rwin["sl_total"], kind="full", label="SL_SD full")
    plot_ll(ax2, Rwin["x"], Rwin["sl_bb"], kind="broadband", label="SL_SD broadband")
    plot_ll(ax2, Rwin["x"], Rwin["sl_rh"], kind="rhythms", label="SL_SD rhythms")
    ax2.set_title("SL_SD")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.grid(False)
    ax2.legend(frameon=False)
    ax2.set_ylim(1e-1, 1e6)

    # Right: Violins (from all windows)
    df = pd.DataFrame({
        "slope": np.concatenate([Ravg["slopes_sp"], Ravg["slopes_bb"]]),
        "method": (["Specparam aperiodic"] * len(Ravg["slopes_sp"])) + (["SL_SD broadband"] * len(Ravg["slopes_bb"])),
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
    ax3.set_ylabel("Slope")
    ax3.grid(False)

    # Params for Row 5 sims come from the averaged SL_SD fit
    sim_params = _collect_slsd_params(Ravg["sl_model"], freqs=Ravg["x"], bb_avg=Ravg["sl_bb"])

    # Everything needed to replot row 2 without recomputing
    row2_cache = dict(
        x=Rwin["x"],
        mt_this=Rwin["mt_this"],
        sp_full=Rwin["sp_full"],
        sp_ap=Rwin["sp_ap"],
        sp_pk=Rwin["sp_pk"],
        sl_total=Rwin["sl_total"],
        sl_bb=Rwin["sl_bb"],
        sl_rh=Rwin["sl_rh"],
        slopes_sp=Ravg["slopes_sp"],
        slopes_bb=Ravg["slopes_bb"],
        chosen_time=Rwin.get("chosen_time"),
        chosen_idx=Rwin.get("chosen_idx"),
        abs_t0=abs_t0,
        abs_t1=abs_t1,
        n_windows=Ravg["n_windows"],
        fs=float(fs),
        F_fit=Ravg["F_fit"],
        x_band=Ravg["x"],
        emp_avg=Ravg["emp_avg"],
        step_bins=Ravg["step_bins"],
        k_eff=Ravg["k_eff"],
    )

    return dict(
        fs=float(fs),
        n_windows=int(Ravg["n_windows"]),
        sim_params=sim_params,
        row2_cache=row2_cache,
    )


def build_row3(fig, outer_gs):
    """Flowchart row; extra vertical spacing for labels in the middle column."""
    gs = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=3, subplot_spec=outer_gs, width_ratios=[2.4, 1.1, 2.4], wspace=0.65
    )

    row3_sim = dict(
        sampling_rate=FS,
        duration=10.0,
        aperiodic_exponent=SIM_APER_EXP,
        aperiodic_offset=SIM_APER_OFF,
        knee=SIM_KNEE,
        peaks=[dict(freq=12.0, amplitude=10.0, sigma=3.0)],
        average_firing_rate=0.0,
        random_state=RNG_SEED,
    )

    res = spectrum(
        sampling_rate=row3_sim["sampling_rate"],
        duration=row3_sim["duration"],
        aperiodic_exponent=row3_sim["aperiodic_exponent"],
        aperiodic_offset=row3_sim["aperiodic_offset"],
        knee=row3_sim["knee"],
        peaks=row3_sim["peaks"],
        average_firing_rate=row3_sim["average_firing_rate"],
        random_state=row3_sim["random_state"],
        direct_estimate=False,
        plot=False,
    )
    td = res.time_domain
    fd = res.frequency_domain

    ts = np.ascontiguousarray(np.asarray(td.combined_signal, float).reshape(-1, 1, 1))
    mt = Multitaper(ts, sampling_frequency=FS, time_halfbandwidth_product=2, n_tapers=3, start_time=0.0)
    conn = Connectivity.from_multitaper(mt)
    f_emp = np.asarray(conn.frequencies)
    S_emp = conn.power().squeeze()

    fmin_plot = max(0.1, float(min(f_emp)))
    fmax_plot = float(min(max(f_emp), max(fd.frequencies)))
    mask_fd = (fd.frequencies >= fmin_plot) & (fd.frequencies <= fmax_plot)
    mask_emp = (f_emp >= fmin_plot) & (f_emp <= fmax_plot)

    f_theory = np.asarray(fd.frequencies[mask_fd])
    S_bb = np.asarray(fd.broadband_spectrum[mask_fd])
    S_rh = np.asarray(fd.rhythmic_spectrum[mask_fd])
    S_comb = np.asarray(fd.combined_spectrum[mask_fd])

    f_emp_c = np.asarray(f_emp[mask_emp])
    S_emp_c = np.asarray(S_emp[mask_emp])

    # Left PSD
    axL = fig.add_subplot(gs[0, 0])
    plot_ll(axL, f_theory, S_comb, "full", label="Ground Truth PSD", color="k")
    plot_ll(axL, f_theory, S_bb, "broadband", label="Broadband")
    plot_ll(axL, f_theory, S_rh, "rhythms", label="Rhythms")
    axL.set(
        xlabel="Frequency (Hz)",
        ylabel="Power",
        xscale="log",
        yscale="log",
        xlim=(fmin_plot, fmax_plot),
        ylim=(1e-3, 1e2),
    )
    sns.despine(ax=axL, top=True, right=True)
    axL.minorticks_off()
    axL.legend(loc="upper left")
    #axL.set_title("Theory PSD components", pad=8)
    axL.set_title("Ground Truth Components", pad=8)

    # Middle: 3 stacked tiny time panels
    gs_mid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], hspace=0.38)
    axM1 = fig.add_subplot(gs_mid[0, 0])
    axM1.plot(td.time, td.broadband_signal, color=COLORS["broadband"])
    axM1.set_title("Broadband", pad=7)
    axM1.tick_params(axis="x", labelbottom=False)
    sns.despine(ax=axM1, top=True, right=True)
    axM1.minorticks_off()

    axM2 = fig.add_subplot(gs_mid[1, 0])
    axM2.plot(td.time, td.rhythmic_signal, color="red")
    axM2.set_title("Rhythmic", pad=7)
    axM2.tick_params(axis="x", labelbottom=False)
    sns.despine(ax=axM2, top=True, right=True)
    axM2.minorticks_off()

    axM3 = fig.add_subplot(gs_mid[2, 0])
    axM3.plot(td.time, td.combined_signal, color="k")
    axM3.set_title("Combined", pad=7)
    axM3.set_xlabel("Time (s)")
    sns.despine(ax=axM3, top=True, right=True)
    axM3.minorticks_off()

    # Right: Theory + empirical
    axR = fig.add_subplot(gs[0, 2])
    plot_ll(axR, f_theory, S_comb, "full", label="Ground Truth PSD", color="k")
    plot_ll(axR, f_theory, S_bb, "broadband", label="Broadband")
    plot_ll(axR, f_theory, S_rh, "rhythms", label="Rhythms")
    plot_ll(axR, f_emp_c, S_emp_c, "multitaper", label="Multitaper", alpha=0.7)
    axR.set(
        xlabel="Frequency (Hz)",
        ylabel="Power",
        xscale="log",
        yscale="log",
        xlim=(fmin_plot, fmax_plot),
        ylim=(1e-3, 1e2),
    )
    sns.despine(ax=axR, top=True, right=True)
    axR.minorticks_off()
    axR.legend(loc="lower left")
    axR.set_title("Theory + empirical MT", pad=8)

    return dict(
        row3_sim=row3_sim,
        f_theory=f_theory,
        S_bb=S_bb,
        S_rh=S_rh,
        S_comb=S_comb,
        f_emp=f_emp_c,
        S_emp=S_emp_c,
        time=np.asarray(td.time),
        ts_broadband=np.asarray(td.broadband_signal),
        ts_rhythmic=np.asarray(td.rhythmic_signal),
        ts_combined=np.asarray(td.combined_signal),
    )


# ----- Row 4: QQ-row + save colorbars for later use -----
def _save_row4_colorbars(base_path, sm_mt, sm_th, qs):
    base = Path(base_path)
    # MT (Blues)
    fig_mt, ax_mt = plt.subplots(figsize=(3.6, 0.35))
    mpl.colorbar.ColorbarBase(ax_mt, cmap=sm_mt.cmap, norm=sm_mt.norm, orientation="horizontal", ticks=qs)
    ax_mt.set_title("MT percentiles", pad=3, fontsize=9)
    fig_mt.savefig(base.with_suffix(".row4_cbar_mt.svg"), bbox_inches="tight", dpi=300)
    plt.close(fig_mt)

    # Theory (Reds)
    fig_th, ax_th = plt.subplots(figsize=(3.6, 0.35))
    mpl.colorbar.ColorbarBase(ax_th, cmap=sm_th.cmap, norm=sm_th.norm, orientation="horizontal", ticks=qs)
    ax_th.set_title("Theory percentiles", pad=3, fontsize=9)
    fig_th.savefig(base.with_suffix(".row4_cbar_th.svg"), bbox_inches="tight", dpi=300)
    plt.close(fig_th)


def build_row4_exact(ax_top, ax_hist, ax_qq, mode, out_base_for_cbars=None):
    FS_loc = 1000.0
    DURATION = 60.0
    APERIODIC_EXP = 2.0
    APERIODIC_OFF = 0.5
    KNEE = 60.0 ** (APERIODIC_EXP / 2.0)
    PEAKS_LINEAR = [dict(freq=12.0, amplitude=0.5, sigma=2.0)]
    FMIN, FMAX = 1.0, 500.0

    K_TAP = 3
    TIME_BW = 2
    BANDWIDTH = 2.0 * TIME_BW / DURATION
    runs = (1000 if mode == "full" else 10)

    freq_grid = np.arange(FMIN, FMAX + 1e-9, BANDWIDTH)

    res0 = spectrum(
        sampling_rate=FS_loc,
        duration=DURATION,
        aperiodic_exponent=APERIODIC_EXP,
        aperiodic_offset=APERIODIC_OFF,
        knee=KNEE,
        peaks=PEAKS_LINEAR,
        average_firing_rate=0.0,
        random_state=0,
        direct_estimate=False,
        plot=False,
    )
    fd0 = res0.frequency_domain
    m0 = (fd0.frequencies >= FMIN) & (fd0.frequencies <= FMAX)
    interp_true = interp1d(
        fd0.frequencies[m0],
        fd0.combined_spectrum[m0],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    S_true = interp_true(freq_grid)

    S_mt = []
    for seed in range(runs):
        res = spectrum(
            sampling_rate=FS_loc,
            duration=DURATION,
            aperiodic_exponent=APERIODIC_EXP,
            aperiodic_offset=APERIODIC_OFF,
            knee=KNEE,
            peaks=PEAKS_LINEAR,
            average_firing_rate=0.0,
            random_state=seed,
            direct_estimate=False,
            plot=False,
        )
        ts = np.asarray(res.time_domain.combined_signal, float).reshape(-1, 1, 1)
        mt = Multitaper(
            time_series=ts,
            sampling_frequency=FS_loc,
            n_tapers=K_TAP,
            time_halfbandwidth_product=TIME_BW,
            start_time=0.0,
        )

        conn = Connectivity.from_multitaper(mt)
        f = conn.frequencies
        S = conn.power().squeeze()
        m = (f >= FMIN) & (f <= FMAX)
        S_mt.append(np.interp(freq_grid, f[m], S[m]))
    S_mt = np.vstack(S_mt)

    qs = [2.5, 25, 50, 75, 97.5]
    pct_mt = np.percentile(S_mt, qs, axis=0)
    pct_th = np.vstack([gamma.ppf(q / 100, a=K_TAP, scale=S_true / K_TAP) for q in qs])

    norm = mpl.colors.Normalize(vmin=qs[0], vmax=qs[-1])
    cmap_mt = mpl.colors.ListedColormap(plt.cm.Blues(np.linspace(0.4, 1, 256)))
    cmap_th = mpl.colors.ListedColormap(plt.cm.Reds(np.linspace(0.4, 1, 256)))
    sm_mt = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_mt)
    sm_th = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_th)

    if out_base_for_cbars is not None:
        _save_row4_colorbars(out_base_for_cbars, sm_mt, sm_th, qs)

    # Col 1: percentiles
    for i, q in enumerate(qs):
        ax_top.loglog(freq_grid, pct_mt[i], color=sm_mt.to_rgba(q), alpha=0.75, lw=1, ls="--")
        ax_top.loglog(freq_grid, pct_th[i], color=sm_th.to_rgba(q), alpha=1.0, lw=1.5)
    ax_top.set_xlabel("Frequency (Hz, log10)")
    ax_top.set_ylabel("Power (log10)")
    ax_top.minorticks_off()
    ax_top.set_xticks([1, 10, 100, 500])
    ax_top.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    sns.despine(ax=ax_top, top=True, right=True)
    ax_top.set_title(r"Theory vs Simulated Sampling Distribution $\mathcal{F}\{\hat S(f)\}$", pad=6)

    # Col 2: histogram @ 150 Hz
    idx = int(np.argmin(np.abs(freq_grid - 150.0)))
    vals = S_mt[:, idx]
    x = np.linspace(max(vals.min(), 1e-12), vals.max(), 200)
    ax_hist.hist(vals, bins="auto", density=True, color="gray", alpha=0.6)
    ax_hist.plot(x, gamma.pdf(x, a=K_TAP, scale=S_true[idx] / K_TAP), color="k", lw=2)
    ax_hist.set_xlabel("Power")
    ax_hist.set_ylabel("Density")
    ax_hist.minorticks_off()
    ax_hist.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax_hist, top=True, right=True)
    ax_hist.set_title(f"Histogram + PDF @ {freq_grid[idx]:.2f} Hz", pad=6)

    # Col 3: QQ plot @ same frequency
    emp = np.sort(vals)
    q_emp = (np.arange(1, len(emp) + 1) - 0.5) / len(emp)
    theor = gamma.ppf(q_emp, a=K_TAP, scale=S_true[idx] / K_TAP)
    mn = min(emp.min(), theor.min())
    mx = max(emp.max(), theor.max())
    ax_qq.scatter(theor, emp, color="gray", s=20, alpha=0.7)
    ax_qq.plot([mn, mx], [mn, mx], ls="--", color="k", lw=1)
    ax_qq.set_xlabel("Theoretical quantiles")
    ax_qq.set_ylabel("Empirical quantiles")
    ax_qq.minorticks_off()
    ax_qq.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax_qq, top=True, right=True)
    ax_qq.set_title(f"QQ-plot @ {freq_grid[idx]:.2f} Hz", pad=6)

    row4_sim = dict(
        sampling_rate=FS_loc,
        duration=DURATION,
        aperiodic_exponent=APERIODIC_EXP,
        aperiodic_offset=APERIODIC_OFF,
        knee=KNEE,
        peaks=PEAKS_LINEAR,
        FMIN=FMIN,
        FMAX=FMAX,
        K_TAP=K_TAP,
        TIME_BW=TIME_BW,
        BANDWIDTH=BANDWIDTH,
        runs=runs,
        hist_freq=float(freq_grid[idx]),
        seeds=list(range(runs)),
    )

    row4_cache = dict(
        row4_sim=row4_sim,
        freq_grid=freq_grid,
        S_true=S_true,
        S_mt=S_mt,
        pct_mt=pct_mt,
        pct_th=pct_th,
        qs=np.array(qs),
        hist_vals=vals,
        qq_emp=emp,
        qq_theor=theor,
    )
    return row4_cache


def build_row5_sim(ax1, ax2, ax3, sim_info, mode):
    """Simulated anesthesia-style row; cols 1-2 show single 4-min window + thick black ground truth."""
    fs = sim_info["fs"]
    n_wins = sim_info["n_windows"]
    params = sim_info["sim_params"]

    seed0 = 24680
    x_sim, t_sim, gt_list = _simulate_with_params(params, n_wins, fs=fs, seed0=seed0)
    P_tf, F, T = _multitaper(x_sim, fs, 0.0)

    Rwin = _single_window_models(P_tf, F, T, fs, SL_KW, rel_sec=FOUR_MIN_SEC)
    Ravg = _avg_models_from_windows(P_tf, F, fs, SL_KW)

    idx = _pick_idx_at_time(T, rel_sec=FOUR_MIN_SEC)
    idx = max(0, min(idx, len(gt_list) - 1))
    gt_fd = gt_list[idx]

    xlo, xhi = float(np.min(Rwin["x"])), float(np.max(Rwin["x"]))
    m = (gt_fd["f"] >= xlo) & (gt_fd["f"] <= xhi)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    plot_ll(ax1, gt_fd["f"][m], gt_fd["full"][m], "truth", label="Ground Truth", lw=3)
    plot_ll(ax1, Rwin["x"], Rwin["mt_this"], "multitaper", label="Multitaper", lw=2, alpha=0.8)
    plot_ll(ax1, Rwin["x"], Rwin["sp_full"], "full", label="Specparam full")
    plot_ll(ax1, Rwin["x"], Rwin["sp_ap"], "broadband", label="Specparam aperiodic")
    plot_ll(ax1, Rwin["x"], Rwin["sp_pk"], "rhythms", label="Specparam rhythms")
    ax1.set_title("Specparam (simulated anesthesia)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power")
    ax1.grid(False)
    ax1.legend(frameon=False)
    ax1.set_ylim(1e-1, 1e6)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plot_ll(ax2, gt_fd["f"][m], gt_fd["full"][m], "truth", label="Ground Truth", lw=3)
    plot_ll(ax2, Rwin["x"], Rwin["mt_this"], "multitaper", label="Multitaper", lw=2, alpha=0.8)
    plot_ll(ax2, Rwin["x"], Rwin["sl_total"], "full", label="SL_SD full")
    plot_ll(ax2, Rwin["x"], Rwin["sl_bb"], "broadband", label="SL_SD broadband")
    plot_ll(ax2, Rwin["x"], Rwin["sl_rh"], "rhythms", label="SL_SD rhythms")
    ax2.set_title("SL_SD (simulated anesthesia)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.grid(False)
    ax2.legend(frameon=False)
    ax2.set_ylim(1e-1, 1e6)

    # Right: Violins + true slope line (from ground-truth broadband)
    df = pd.DataFrame({
        "slope": np.concatenate([Ravg["slopes_sp"], Ravg["slopes_bb"]]),
        "method": (["Specparam aperiodic"] * len(Ravg["slopes_sp"])) + (["SL_SD broadband"] * len(Ravg["slopes_bb"])),
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

    true_slope = _compute_loglog_slope(gt_fd["f"], gt_fd["bb"], *SLOPE_BAND)
    ax3.axhline(true_slope, color="k", lw=2.4, alpha=0.95)
    ax3.set_title("40-60 Hz slope distributions")
    ax3.set_xlabel("")
    ax3.set_ylabel("Slope")
    ax3.grid(False)

    row5_cache = dict(
        sim_seed0=int(seed0),
        sim_n_windows=int(n_wins),
        chosen_idx=int(idx),
        chosen_time=float(np.asarray(T).ravel()[idx]) if len(np.atleast_1d(T)) else None,
        x=Rwin["x"],
        gt_f=gt_fd["f"][m],
        gt_full=gt_fd["full"][m],
        gt_bb=gt_fd["bb"][m],
        mt_this=Rwin["mt_this"],
        sp_full=Rwin["sp_full"],
        sp_ap=Rwin["sp_ap"],
        sp_pk=Rwin["sp_pk"],
        sl_total=Rwin["sl_total"],
        sl_bb=Rwin["sl_bb"],
        sl_rh=Rwin["sl_rh"],
        slopes_sp=Ravg["slopes_sp"],
        slopes_bb=Ravg["slopes_bb"],
        true_slope=float(true_slope),
    )
    return row5_cache


# ====================== Methods/report helpers ======================
def _summarize_vec(x):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        p25=float(q25),
        p50=float(q50),
        p75=float(q75),
        std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        min=float(np.min(x)),
        max=float(np.max(x)),
    )


def _fwhm_from_sigma(sigma_hz: float) -> float:
    # Gaussian FWHM = 2*sqrt(2*ln2)*sigma
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * float(sigma_hz))


def _interp_checkpoints(f, y, pts=(1.0, 10.0, 100.0)):
    f = np.asarray(f, float).ravel()
    y = np.asarray(y, float).ravel()
    out = {}
    if f.size < 2 or y.size != f.size:
        return out
    for p in pts:
        if p < f.min() or p > f.max():
            continue
        out[f"{p:g}Hz"] = float(np.interp(p, f, y))
    return out


def _write_report(report_path: Path, lines: list[str]):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _build_methods_report(
    out_path: Path,
    mode: str,
    *,
    row1_cache: dict,
    row2_cache: dict,
    row3_cache: dict,
    row4_cache: dict,
    row5_cache: dict,
    sim_params: dict,
    payload_paths: dict,
    include_file_hashes: bool = True,
):
    out_path = Path(out_path)
    base = out_path.with_suffix("")
    report_path = Path(str(base) + f".{mode}.report.txt")

    # Environment + provenance
    lines = []
    lines.append("FIGURE 1 METHODS/VALUES REPORT")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Mode: {mode}")
    lines.append("")
    lines.append("OUTPUT ARTIFACTS")
    lines.append(f"  Figure (PNG): {out_path}")
    lines.append(f"  Figure (SVG): {out_path.with_suffix('.svg')}")
    lines.append(f"  Plot payload (NPZ): {payload_paths.get('npz')}")
    lines.append(f"  Plot metadata (JSON): {payload_paths.get('meta_json')}")
    lines.append(f"  Row4 colorbar (MT): {out_path.with_suffix('.row4_cbar_mt.svg')}")
    lines.append(f"  Row4 colorbar (Theory): {out_path.with_suffix('.row4_cbar_th.svg')}")
    lines.append("")

    lines.append("RUNTIME ENVIRONMENT")
    lines.append(f"  Python: {platform.python_version()}")
    lines.append(f"  Platform: {platform.platform()}")
    lines.append(f"  numpy: {np.__version__}")
    lines.append(f"  pandas: {pd.__version__}")
    lines.append(f"  matplotlib: {mpl.__version__}")
    lines.append(f"  seaborn: {sns.__version__}")
    try:
        import specparam as _sp
        lines.append(f"  specparam: {_sp.__version__}")
    except Exception:
        lines.append("  specparam: (version unavailable)")
    lines.append("")

    # Input files
    lines.append("INPUT DATA FILES")
    for p in (ECOG_MAT, TIME_MAT, COND_MAT):
        if p.exists():
            sz = p.stat().st_size
            if include_file_hashes:
                try:
                    h = _sha256_file(p)
                    lines.append(f"  {p}  (bytes={sz}, sha256={h})")
                except Exception as e:
                    lines.append(f"  {p}  (bytes={sz}, sha256=ERROR: {e})")
            else:
                lines.append(f"  {p}  (bytes={sz})")
        else:
            lines.append(f"  {p}  (MISSING)")
    lines.append("")

    # Global config
    lines.append("GLOBAL ANALYSIS CONFIG")
    lines.append(f"  FS={FS}")
    lines.append(f"  Multitaper NW={NW}, K_TAPERS={K_TAPERS}, WIN_DUR={WIN_DUR} s")
    lines.append(f"  Analysis frange={ANALYSIS_FRANGE} Hz")
    lines.append(f"  Slope band={SLOPE_BAND} Hz (log10 power vs log10 freq)")
    lines.append(f"  FOUR_MIN_SEC={FOUR_MIN_SEC} s (window selection target)")
    lines.append("")

    # Row 1
    lines.append("ROW 1: 3 s ECoG (Anesthesia — middle)")
    lines.append(f"  anesthesia interval: t0={row1_cache.get('anesthesia_t0'):.6f}, t1={row1_cache.get('anesthesia_t1'):.6f} (abs time units from MAT)")
    lines.append(f"  window abs start={row1_cache.get('window_abs_start'):.6f}, abs stop={row1_cache.get('window_abs_stop'):.6f}")
    lines.append(f"  fs (estimated)={row1_cache.get('fs'):.6f} Hz")
    x1 = np.asarray(row1_cache.get("x", []), float)
    if x1.size:
        lines.append(f"  signal summary: mean={np.mean(x1):.6g}, std={np.std(x1, ddof=1) if x1.size>1 else 0.0:.6g}, min={np.min(x1):.6g}, max={np.max(x1):.6g}, n={x1.size}")
    lines.append("")

    # Row 2: slopes (violin stats)
    lines.append("ROW 2: Empirical anesthesia pipeline")
    lines.append(f"  anesthesia segment used (abs): t0={row2_cache.get('abs_t0'):.6f}, t1={row2_cache.get('abs_t1'):.6f}")
    lines.append(f"  n_windows={int(row2_cache.get('n_windows', -1))} (WIN_DUR={WIN_DUR}s)")
    lines.append(f"  chosen window for spectra: idx={row2_cache.get('chosen_idx')}, chosen_time={row2_cache.get('chosen_time')}")
    lines.append(f"  independent-grid step_bins={row2_cache.get('step_bins')}, k_eff (SL avg fit)={row2_cache.get('k_eff')}")
    lines.append("")
    sp_stats = _summarize_vec(row2_cache.get("slopes_sp", []))
    bb_stats = _summarize_vec(row2_cache.get("slopes_bb", []))
    lines.append("  40–60 Hz slope distributions (values plotted in violins)")
    lines.append(f"    Specparam aperiodic: {sp_stats}")
    lines.append(f"    SL_SD broadband:     {bb_stats}")
    lines.append("")

    # Row 2 → Row 5 sim params
    lines.append("ROW 2 → ROW 5 SIMULATION PARAMETERS (posterior-mean from Row 2 averaged SL_SD)")
    lines.append(f"  aperiodic: slope_0={sim_params.get('slope_0')}, alpha={sim_params.get('alpha')}, knee_0={sim_params.get('knee_0')}, b_0={sim_params.get('b_0')} ({sim_params.get('b0_space')})")
    peaks = list(sim_params.get("peaks_list", []))
    if len(peaks) == 0:
        lines.append("  rhythms: peaks_list is EMPTY (no peaks recovered).")
    else:
        lines.append("  rhythms (center/width/height):")
        for j, pk in enumerate(peaks):
            cf = float(pk["freq"])
            sig = float(pk["sigma"])
            amp = float(pk["amplitude"])
            lines.append(f"    peak[{j}]: cf={cf:.6g} Hz, sigma={sig:.6g} Hz (FWHM={_fwhm_from_sigma(sig):.6g} Hz), amplitude(A_lin)={amp:.6g}")
    lines.append("")

    # Row 3 ground truth checkpoints
    lines.append("ROW 3: Flowchart simulation (theory PSD + time series + empirical MT)")
    row3_sim = row3_cache.get("row3_sim", {})
    lines.append(f"  sim params: {row3_sim}")
    f3 = row3_cache.get("f_theory", np.array([]))
    s3 = row3_cache.get("S_comb", np.array([]))
    ck3 = _interp_checkpoints(f3, s3, pts=(1.0, 10.0, 12.0, 100.0))
    if ck3:
        lines.append(f"  ground-truth PSD checkpoints (combined, linear power): {ck3}")
    lines.append("")

    # Row 4 distribution summary
    lines.append("ROW 4: Theory vs simulated sampling distribution (Gamma model checks)")
    row4_sim = row4_cache.get("row4_sim", {})
    lines.append(f"  sim params: {row4_sim}")
    hist_freq = float(row4_sim.get("hist_freq", np.nan))
    vals = np.asarray(row4_cache.get("hist_vals", []), float)
    if vals.size:
        vstats = _summarize_vec(vals)
        lines.append(f"  histogram frequency: {hist_freq:.6g} Hz")
        lines.append(f"  MT draws at hist freq (values plotted): {vstats}")
        # Theoretical gamma at that frequency uses shape=K_TAP and scale=S_true/K_TAP
        K_TAP = int(row4_sim.get("K_TAP", 3))
        fg = np.asarray(row4_cache.get("freq_grid", []), float)
        S_true = np.asarray(row4_cache.get("S_true", []), float)
        if fg.size and S_true.size == fg.size:
            idx = int(np.argmin(np.abs(fg - hist_freq)))
            scale = float(S_true[idx] / max(K_TAP, 1))
            lines.append(f"  theory gamma at hist freq: shape(a)={K_TAP}, scale={scale:.6g}  (mean=a*scale={K_TAP*scale:.6g})")
    lines.append("")

    # Row 5: ground truth line + slope stats + PSD checkpoints
    lines.append("ROW 5: Simulated anesthesia (from Row 2 SL_SD posterior-mean params)")
    lines.append(f"  sim seed0={row5_cache.get('sim_seed0')}, n_windows={row5_cache.get('sim_n_windows')}")
    lines.append(f"  chosen simulated window: idx={row5_cache.get('chosen_idx')}, chosen_time={row5_cache.get('chosen_time')}")
    lines.append(f"  TRUE SLOPE LINE (black horizontal line on violin): true_slope={row5_cache.get('true_slope'):.9g}")
    sp5_stats = _summarize_vec(row5_cache.get("slopes_sp", []))
    bb5_stats = _summarize_vec(row5_cache.get("slopes_bb", []))
    lines.append("  40–60 Hz slope distributions (values plotted in violins)")
    lines.append(f"    Specparam aperiodic: {sp5_stats}")
    lines.append(f"    SL_SD broadband:     {bb5_stats}")
    f5 = row5_cache.get("gt_f", np.array([]))
    s5 = row5_cache.get("gt_full", np.array([]))
    ck5 = _interp_checkpoints(f5, s5, pts=(1.0, 10.0, 12.0, 100.0))
    if ck5:
        lines.append(f"  ground-truth PSD checkpoints (combined, linear power): {ck5}")
    lines.append("")

    # Note payload keys
    lines.append("PLOT PAYLOAD NOTES")
    lines.append("  The NPZ contains EVERY numeric x/y array plotted (flattened as <section>__<key>).")
    lines.append("  Use the JSON meta to see shapes/dtypes and the non-numeric metadata.")
    lines.append("")

    return report_path, lines


# ====================== Main Figure ======================
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
        default=str(DEFAULT_OUT_DIR),
        help="Directory where outputs will be saved (used when --out is relative).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA256 hashing of input MAT files in the report.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = out_dir / out_path.name

    # Figure grid: give Row 3 a bit more height to help labels
    fig = plt.figure(figsize=(18, 23))
    outer = gridspec.GridSpec(
        nrows=5,
        ncols=3,
        height_ratios=[1.0, 1.5, 2.3, 1.6, 1.5],
        hspace=0.85,
        wspace=0.55,
    )

    # Row 1
    ax_r1 = fig.add_subplot(outer[0, :])
    row1_cache = build_row1(ax_r1)

    # Row 2 (empirical)
    ax21 = fig.add_subplot(outer[1, 0])
    ax22 = fig.add_subplot(outer[1, 1])
    ax23 = fig.add_subplot(outer[1, 2])
    sim_info = build_row2_anesthesia(ax21, ax22, ax23, args.mode)

    # Row 3
    row3_cache = build_row3(fig, outer[2, :])

    # Row 4 (QQ row) + save colorbars as stand-alone SVGs
    ax41 = fig.add_subplot(outer[3, 0])
    ax42 = fig.add_subplot(outer[3, 1])
    ax43 = fig.add_subplot(outer[3, 2])
    row4_cache = build_row4_exact(ax41, ax42, ax43, args.mode, out_base_for_cbars=out_path)

    # Row 5 (simulated-from-Row2 params)
    ax51 = fig.add_subplot(outer[4, 0])
    ax52 = fig.add_subplot(outer[4, 1])
    ax53 = fig.add_subplot(outer[4, 2])
    row5_cache = build_row5_sim(ax51, ax52, ax53, sim_info, args.mode)

    # ---- Save a full payload of all ARRAYS needed to rebuild the figure (mode-specific) ----
    run_config = dict(
        mode=args.mode,
        FS=FS,
        NW=NW,
        K_TAPERS=K_TAPERS,
        WIN_DUR=WIN_DUR,
        ANALYSIS_FRANGE=ANALYSIS_FRANGE,
        SLOPE_BAND=SLOPE_BAND,
        FOUR_MIN_SEC=FOUR_MIN_SEC,
        RNG_SEED=RNG_SEED,
        SP_KW=SP_KW,
        SL_KW=SL_KW,
        MT_PARAMS=MT_PARAMS,
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
        sim_params=sim_info.get("sim_params", {}),
    )

    npz_path, meta_path = _save_plot_payload(
        out_path,
        args.mode,
        arrays_sections=arrays_sections,
        meta_sections=meta_sections,
    )

    # ---- Save figures ----
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, dpi=args.dpi, bbox_inches="tight")

    # Keep the Row5 sim params JSON (as before), but mode-aware filename
    base = out_path.with_suffix("")
    params_path = Path(str(base) + f".{args.mode}.row5_sim_params.json")
    with open(params_path, "w") as f:
        json.dump(sim_info["sim_params"], f, indent=2, default=_json_default)

    # ---- Methods/value report ----
    report_path, report_lines = _build_methods_report(
        out_path,
        args.mode,
        row1_cache=row1_cache,
        row2_cache=sim_info.get("row2_cache", {}),
        row3_cache=row3_cache,
        row4_cache=row4_cache,
        row5_cache=row5_cache,
        sim_params=sim_info.get("sim_params", {}),
        payload_paths={"npz": str(npz_path), "meta_json": str(meta_path)},
        include_file_hashes=(not args.no_hash),
    )
    _write_report(report_path, report_lines)

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
