#!/usr/bin/env python3
"""
Generate Figure 4 empirical ECoG state-decomposition panels.

The script analyzes awake and anesthetized macaque ECoG windows, fits specparam
and SL_specdecomp models, computes high-gamma power, broadband slope, rhythm
peak summaries, and CVLL diagnostics, and writes the manuscript figure plus
plot payloads.
"""

from __future__ import annotations

import os
import re
import json
import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat
from scipy.signal import find_peaks, peak_widths
from scipy.special import gammaln

from spectral_connectivity import Multitaper, Connectivity
from SL_specdecomp import Decompose
from specparam import SpectralModel



PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# ──────────────────────────── Global config ────────────────────────────
ROOT = str(PROJECT_ROOT)
DATA_DIR = os.path.join(ROOT, "Data", "InputData", "InputDataFiles")
SAVE_DIR = os.path.join(ROOT, "Output", "Results", "FiguresIntermediate", "Figure_4_CV", "Figure_output")
os.makedirs(SAVE_DIR, exist_ok=True)

# .mat keys
ECOG_MAT = os.path.join(DATA_DIR, "ECoG_ch1.mat")
TIME_MAT = os.path.join(DATA_DIR, "ECoGTime.mat")
COND_MAT = os.path.join(DATA_DIR, "Condition.mat")
ECOG_KEY = "ECoG_ch1"
TIME_KEY = "ECoGTime"
COND_TIME_KEY = "ConditionTime"
COND_LABEL_KEY = "ConditionLabel"

# Multitaper params: 30 s windows, 0 overlap (Rows 1–4)
MT_PARAMS = dict(
    time_halfbandwidth_product=2,  # NW
    n_tapers=3,
    time_window_duration=30.0,
    time_window_step=30.0,
)

# CV multitaper params (used for CVLL aux panel in Row 4, Col 3)
CV_FOLDS = 5
CV_NW = 1
CV_K_TAPERS = 1

# Analysis band & metric bands (Hz)
ANALYSIS_FRANGE = (0.1, 200.0)
HG_BAND = (80.0, 180.0)
SLOPE_BAND = (40.0, 60.0)

# Raster band (Hz)
RASTER_FRANGE = (1.0, 35.0)

# Aesthetics
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
    "fwhm": "#6a3d9a",  # kept for raster row (Row 3 uses widths as xerr)
}
STYLES = {
    "emp": dict(lw=2.0, alpha=0.65, solid_capstyle="round"),
    "full": dict(lw=2.0),
    "component": dict(lw=1.8, ls="--"),
}
PSD_YLIM = (1e-1, 1e6)

CONDITION_CFG = {
    "awake": {
        "start_phrase": "AwakeEyesClosed-Start",
        "end_phrase": "AwakeEyesClosed-End",
        "target_min_after_start": 17.29,
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
        "target_min_after_start": 4.0,
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


# ──────────────────────────── b-prior sensitivity helper ────────────────────────────
# Package default in build_additive_model is:
#     b_0 ~ Normal(log10(median(y_pos)), 2.0)
# This Figure 4 variant keeps the same data-dependent mean but widens the prior:
#     b_0 ~ Normal(log10(median(y_pos)), 5.0)
B_PRIOR_SIGMA = 5.0


def _b_prior_param_specs(y_lin: np.ndarray, sigma: float = B_PRIOR_SIGMA) -> Dict[str, Any]:
    """
    Build aperiodic_param_specs for the SL_specdecomp b parameter.

    The current additive model names the single aperiodic offset `b_0`.
    Some package variants use `b` when n_aperiodics == 1, so this returns both
    keys; only the key specified by the installed model is actually used.
    """
    y = np.asarray(y_lin, float)
    y_pos = y[np.isfinite(y) & (y > 0)]
    mu_b = float(np.log10(np.median(y_pos))) if y_pos.size else 0.0
    sigma = float(sigma)

    def _factory(name: str, mu: float = mu_b, sigma: float = sigma):
        return pm.Normal(name, mu=mu, sigma=sigma)

    return {
        "b_0": {"factory": _factory},
        "b": {"factory": _factory},
    }


A_HEIGHT_ANCHOR_Q = 50.0
A_HEIGHT_PRIOR_SIGMA = 1.25


def _robust_band_scale(
    freqs: np.ndarray,
    y_lin: np.ndarray,
    band: Tuple[float, float],
    q: float = A_HEIGHT_ANCHOR_Q,
) -> float:
    f = np.asarray(freqs, float).ravel()
    y = np.asarray(y_lin, float).ravel()

    lo, hi = map(float, band)
    m = (
        np.isfinite(f)
        & np.isfinite(y)
        & (y > 0)
        & (f >= lo)
        & (f <= hi)
    )

    # fallback if too few points in the rhythm band
    if m.sum() < 5:
        m = np.isfinite(y) & (y > 0)

    if m.sum() == 0:
        return 1e-12

    return float(np.percentile(y[m], float(q)))


def _height_prior_param_specs(
    freqs_fit: np.ndarray,
    y_lin: np.ndarray,
    rhythm_bands: List[Tuple[float, float]],
    *,
    q: float = A_HEIGHT_ANCHOR_Q,
    sigma: float = A_HEIGHT_PRIOR_SIGMA,
) -> Dict[str, Any]:
    """
    Median-anchor additive rhythm-height priors:

        A_lin_i ~ LogNormal(log(median power in rhythm band i), sigma=1.25)

    This is the Figure 4 analogue of the Figure 3 median A_lin_0 override.
    """
    specs: Dict[str, Any] = {}

    for i, band in enumerate(rhythm_bands):
        band_scale = _robust_band_scale(freqs_fit, y_lin, band, q=q)

        def _factory(
            name: str,
            band_scale: float = band_scale,
            sigma: float = sigma,
        ):
            return pm.LogNormal(
                name,
                mu=np.log(max(float(band_scale), 1e-12)),
                sigma=float(sigma),
            )

        specs[f"A_lin_{i}"] = {"factory": _factory}

    return specs


def _slsd_kwargs_with_b_prior(
    slsd_kwargs: Dict[str, Any],
    freqs_fit: np.ndarray,
    y_lin: np.ndarray,
    *,
    b_sigma: float = B_PRIOR_SIGMA,
    a_height_anchor_q: float = A_HEIGHT_ANCHOR_Q,
) -> Dict[str, Any]:
    """
    Return SL_specdecomp kwargs with:
      1. b prior widened to sigma=5
      2. additive A_lin_i height priors anchored to median rhythm-band power
    """
    out = dict(slsd_kwargs)

    # b prior
    aperiodic_specs = dict(out.get("aperiodic_param_specs", {}) or {})
    aperiodic_specs.update(_b_prior_param_specs(y_lin, sigma=b_sigma))
    out["aperiodic_param_specs"] = aperiodic_specs

    # additive rhythm-height prior
    if str(out.get("mode", "additive")) == "additive":
        rhythm_bands = list(out.get("rhythm_bands", []) or [])
        rhythm_specs = dict(out.get("rhythm_param_specs", {}) or {})
        rhythm_specs.update(
            _height_prior_param_specs(
                freqs_fit,
                y_lin,
                rhythm_bands,
                q=a_height_anchor_q,
                sigma=A_HEIGHT_PRIOR_SIGMA,
            )
        )
        out["rhythm_param_specs"] = rhythm_specs

    return out

def _slsd_kwargs_with_b_prior_legacy(
    slsd_kwargs: Dict[str, Any],
    y_lin: np.ndarray,
    *,
    sigma: float = B_PRIOR_SIGMA,
) -> Dict[str, Any]:
    """Return SL_specdecomp kwargs with only the b prior changed."""
    out = dict(slsd_kwargs)
    specs = dict(out.get("aperiodic_param_specs", {}) or {})
    specs.update(_b_prior_param_specs(y_lin, sigma=sigma))
    out["aperiodic_param_specs"] = specs
    return out


# ──────────────────────────── NEW: window-drop helpers (global outlier removal) ────────────────────────────
def _parse_int_list(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.replace(" ", "").split(","):
        if tok == "":
            continue
        out.append(int(tok))
    return sorted(set(out))


def _read_drop_from_outlier_csv(csv_path: str, condition: str) -> List[int]:
    """
    Reads figure_4_outlier_inspector.py output 'outliers_report.csv'
    and returns all window_index values for the given condition.
    """
    csv_path = os.path.expanduser(str(csv_path))
    df = pd.read_csv(csv_path)
    df = df[df["condition"].astype(str).str.lower() == str(condition).lower()]
    if df.empty:
        return []
    return sorted(set(int(x) for x in df["window_index"].values))


def _apply_window_drop(
    *,
    drop_orig: List[int],
    arrays: Dict[str, Any],
    sp_peaks_list: List[np.ndarray],
    sl_peaks_list: List[np.ndarray],
    idx_target: int,
    sel_orig: Optional[List[int]] = None,
    sl_models: Optional[List[Any]] = None,
) -> Tuple[Dict[str, Any], List[np.ndarray], List[np.ndarray], int, List[int], Optional[List[Any]]]:
    """
    drop_orig: indices in the *original* all-window space (i.e., 30s windows in segment)
    sel_orig: if windows_mode == '2', maps local -> original. If None, local==original.
    Returns:
      arrays (filtered), sp_peaks_list (filtered), sl_peaks_list (filtered),
      idx_target (remapped), keep_orig (original indices kept), sl_models (filtered if provided)
    """
    TW = int(np.asarray(arrays["P_fit_tf"]).shape[0])

    if sel_orig is None:
        sel_orig = list(range(TW))
    else:
        sel_orig = list(sel_orig)

    drop_set = set(int(i) for i in (drop_orig or []))
    keep_mask = np.array([orig_i not in drop_set for orig_i in sel_orig], dtype=bool)

    if keep_mask.sum() == 0:
        raise RuntimeError("All windows were dropped; check your drop indices / CSV.")

    # If nothing to drop, return as-is
    if keep_mask.all():
        keep_orig = list(sel_orig)
        return arrays, sp_peaks_list, sl_peaks_list, int(idx_target), keep_orig, sl_models

    keep_idx = np.where(keep_mask)[0].astype(int)
    keep_orig = [sel_orig[i] for i in keep_idx]

    # Remap idx_target: if it was dropped, move to nearest kept index
     # Remap idx_target into the filtered/post-drop row space.
    # If the specified target was dropped, move to the nearest kept pre-drop
    # local index, then convert that kept index to its post-filter row position.
    if int(idx_target) not in set(keep_idx.tolist()):
        nearest_local = int(keep_idx[np.argmin(np.abs(keep_idx - int(idx_target)))])
        idx_target_new = int(np.where(keep_idx == nearest_local)[0][0])
    else:
        idx_target_new = int(np.where(keep_idx == int(idx_target))[0][0])
    def _take(a):
        a = np.asarray(a)
        if a.ndim == 1:
            return a[keep_idx]
        if a.ndim == 2:
            return a[keep_idx, :]
        return a

    # Filter all per-window arrays used downstream
    for k in [
        "P_fit_tf", "T_rel_min",
        "specparam_full", "specparam_aper", "specparam_rh",
        "slsd_total", "slsd_bb", "slsd_rh",
        "slopes_specparam", "slopes_slsd",
        "hg_specparam", "hg_slsd",
        "cvll_specparam", "cvll_slsd",
    ]:
        if k in arrays:
            arrays[k] = _take(arrays[k])

    # Filter peak lists
    sp_peaks_list = [sp_peaks_list[i] for i in keep_idx] if sp_peaks_list else []
    sl_peaks_list = [sl_peaks_list[i] for i in keep_idx] if sl_peaks_list else []

    # Filter models (for export params)
    if sl_models is not None and len(sl_models) == TW:
        sl_models = [sl_models[i] for i in keep_idx]

    return arrays, sp_peaks_list, sl_peaks_list, idx_target_new, keep_orig, sl_models


# ──────────────────────────── JSON/NPZ payload helpers ────────────────────────────
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
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
        if arr.dtype == object:
            return np.array([])
        return arr
    except Exception:
        return np.array([])


def save_plot_payload_fig4(
    out_base: Path,
    *,
    arrays: Dict[str, Any],
    meta: Dict[str, Any],
) -> Tuple[Path, Path]:
    out_base = Path(out_base)
    npz_path = out_base.with_suffix(".plotdata.npz")
    meta_path = out_base.with_suffix(".plotmeta.json")

    flat = {}
    meta_out = {
        "created_local": datetime.datetime.now().isoformat(timespec="seconds"),
        "arrays": {},
        "meta": meta if meta is not None else {},
    }

    for k, v in (arrays or {}).items():
        arr = _coerce_array(v)
        flat[k] = arr
        meta_out["arrays"][k] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size),
        }

    np.savez_compressed(npz_path, **flat)
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2, default=_json_default)

    print(f"[INFO] Saved Figure4 payload → {npz_path}")
    print(f"[INFO] Saved Figure4 payload meta → {meta_path}")
    return npz_path, meta_path


def load_plot_payload_fig4(out_base: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    out_base = Path(out_base)
    npz_path = out_base.with_suffix(".plotdata.npz")
    meta_path = out_base.with_suffix(".plotmeta.json")

    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing payload files: {npz_path} / {meta_path}")

    arrays = dict(np.load(npz_path, allow_pickle=False))
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"[INFO] Loaded Figure4 payload ← {npz_path}")
    print(f"[INFO] Loaded Figure4 payload meta ← {meta_path}")
    return arrays, meta


def _fig4_payload_base(state_name: str, windows_mode: str) -> Path:
    suffix = "_2win" if windows_mode == "2" else ""
    return Path(SAVE_DIR) / f"Figure_4_CV_{state_name}{suffix}"


def _finite_metric_values(x: Any) -> np.ndarray:
    arr = np.asarray(x, float).ravel()
    return arr[np.isfinite(arr)]


def _metric_ylim_from_values(*values: Any, pad_frac: float = 0.05, min_span: Optional[float] = None) -> Optional[Tuple[float, float]]:
    chunks = [_finite_metric_values(v) for v in values if v is not None]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return None

    vals = np.concatenate(chunks)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return None

    if vmax < vmin:
        vmin, vmax = vmax, vmin

    span = float(vmax - vmin)
    if span == 0.0:
        base = abs(vmax) if vmax != 0.0 else 1.0
        half = max(base * 0.05, 1e-6)
        if min_span is not None:
            half = max(half, 0.5 * float(min_span))
        return (vmin - half, vmax + half)

    pad = float(span * pad_frac)
    lo = vmin - pad
    hi = vmax + pad

    if min_span is not None:
        cur_span = hi - lo
        need = float(min_span) - cur_span
        if need > 0.0:
            lo -= 0.5 * need
            hi += 0.5 * need

    return (lo, hi)


def _load_metric_arrays_from_payload_legacy(state_name: str, windows_mode: str) -> Dict[str, np.ndarray]:
    arrays, _meta = load_plot_payload_fig4(_fig4_payload_base(state_name, windows_mode))
    needed = {
        "hg_specparam": np.asarray(arrays["hg_specparam"], float),
        "hg_slsd": np.asarray(arrays["hg_slsd"], float),
        "slopes_specparam": np.asarray(arrays["slopes_specparam"], float),
        "slopes_slsd": np.asarray(arrays["slopes_slsd"], float),
    }
    return needed


def _load_metric_arrays_from_payload(
    state_name: str,
    windows_mode: str,
    drop_orig: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    arrays, meta = load_plot_payload_fig4(_fig4_payload_base(state_name, windows_mode))
    meta_inner = meta.get("meta", {})

    hg_specparam = np.asarray(arrays["hg_specparam"], float)
    hg_slsd = np.asarray(arrays["hg_slsd"], float)
    slopes_specparam = np.asarray(arrays["slopes_specparam"], float)
    slopes_slsd = np.asarray(arrays["slopes_slsd"], float)

    # Prefer the payload-recorded drop list unless an explicit one is passed in.
    drop_use = list(meta_inner.get("drop_orig", [])) if drop_orig is None else list(drop_orig)

    # Map local window index -> original window index.
    # If keep_orig is absent, assume local == original.
    keep_orig_saved = meta_inner.get("keep_orig", None)
    n = len(hg_specparam)

    if keep_orig_saved is None:
        orig_idx = list(range(n))
    else:
        orig_idx = [int(x) for x in keep_orig_saved]

    if len(orig_idx) != n:
        raise ValueError(
            f"{state_name}: keep_orig length ({len(orig_idx)}) does not match metric length ({n})."
        )

    drop_set = set(int(x) for x in drop_use)
    keep_mask = np.array([oi not in drop_set for oi in orig_idx], dtype=bool)

    return {
        "hg_specparam": hg_specparam[keep_mask],
        "hg_slsd": hg_slsd[keep_mask],
        "slopes_specparam": slopes_specparam[keep_mask],
        "slopes_slsd": slopes_slsd[keep_mask],
    }

def _resolve_shared_metric_ylims(
    state_name: str,
    windows_mode: str,
    *,
    hg_specparam: np.ndarray,
    hg_slsd: np.ndarray,
    slopes_specparam: np.ndarray,
    slopes_slsd: np.ndarray,
    drop_awake: Optional[List[int]] = None,
    drop_anes: Optional[List[int]] = None,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    other_state = "anesthesia" if state_name == "awake" else "awake"

    hg_values = [hg_specparam, hg_slsd]
    slope_values = [slopes_specparam, slopes_slsd]

    drop_map = {
        "awake": list(drop_awake or []),
        "anesthesia": list(drop_anes or []),
    }

    try:
        other = _load_metric_arrays_from_payload(
            other_state,
            windows_mode,
            drop_orig=drop_map[other_state],
        )
        hg_values.extend([other["hg_specparam"], other["hg_slsd"]])
        slope_values.extend([other["slopes_specparam"], other["slopes_slsd"]])
    except FileNotFoundError:
        pass

    hg_ylim = _metric_ylim_from_values(*hg_values, pad_frac=0.05)
    slope_ylim = _metric_ylim_from_values(*slope_values, pad_frac=0.05)
    return hg_ylim, slope_ylim

def _resolve_shared_metric_ylims_legacy(
    state_name: str,
    windows_mode: str,
    *,
    hg_specparam: np.ndarray,
    hg_slsd: np.ndarray,
    slopes_specparam: np.ndarray,
    slopes_slsd: np.ndarray,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    other_state = "anesthesia" if state_name == "awake" else "awake"

    hg_values = [hg_specparam, hg_slsd]
    slope_values = [slopes_specparam, slopes_slsd]

    try:
        other = _load_metric_arrays_from_payload(other_state, windows_mode)
        hg_values.extend([other["hg_specparam"], other["hg_slsd"]])
        slope_values.extend([other["slopes_specparam"], other["slopes_slsd"]])
    except FileNotFoundError:
        pass

    hg_ylim = _metric_ylim_from_values(*hg_values, pad_frac=0.05)
    slope_ylim = _metric_ylim_from_values(*slope_values, pad_frac=0.05)
    return hg_ylim, slope_ylim



# ──────────────────────────── Export helpers (Figure 4 → Figure 1) ────────────────────────────
def _posterior_mean_scalar(ds, names):
    for nm in names:
        if nm in ds:
            return float(ds[nm].mean().item())
    return None


def _knee_omega_to_hz(knee_omega: float, alpha: float) -> float:
    """
    Convert knee parameter from omega-domain convention ((2*pi*f)^alpha)
    to Hz-domain convention (f^alpha):

        knee_hz = knee_omega / (2*pi)^alpha
    """
    knee_omega = float(knee_omega)
    alpha = float(alpha)
    if knee_omega <= 0.0 or alpha <= 0.0:
        return float(max(knee_omega, 0.0))
    return float(knee_omega / (2.0 * np.pi) ** alpha)

def _aperiodic_from_model_or_fit(F_fit, bb_lin, sl_model):
    """
    Read the *actual decomposition parameters* from the SL_specdecomp posterior.

    For the PyMC API:
      - additive, n_aperiodics=1 -> b, chi, knee
      - additive, n_aperiodics>1 -> b_i, chi_i, knee_i

    We do NOT fall back to a broadband log-log fit here, because that quantity
    is not the same thing as the decomposition parameter used by spectrum.
    """
    out = {"aperiodic_offset": None, "aperiodic_exponent": None, "knee_0": None}

    idata = getattr(sl_model, "idata", None)
    ds = getattr(idata, "posterior", None) if idata is not None else None
    if ds is None:
        raise RuntimeError("SL_specdecomp model has no posterior; cannot export decomposition parameters.")

    # same API names first
    off = _posterior_mean_scalar(ds, [
        "b", "b_0", "aperiodic_offset", "offset", "b0"
    ])
    chi = _posterior_mean_scalar(ds, [
        "chi", "chi_0", "aperiodic_exponent", "alpha",
        "slope_0", "slope", "beta"
    ])
    knee = _posterior_mean_scalar(ds, [
        "knee", "knee_0", "kappa", "kappa_0"
    ])

    if off is None or chi is None:
        raise RuntimeError(
            "Could not find decomposition aperiodic parameters in posterior. "
            f"Available vars: {sorted(list(ds.data_vars))}"
        )

    out["aperiodic_offset"] = float(off)
    out["aperiodic_exponent"] = float(abs(chi))
    out["knee_0"] = 0.0 if knee is None else float(max(knee, 0.0))
    return out

def _aperiodic_from_model_or_fit_legacy(F_fit, bb_lin, sl_model):
    """
    Try to pull aperiodic params from SL_specdecomp posterior (offset/slope/knee-like).
    Fallback: log-log linear fit on broadband: log10(bb) ~ a + b*log10(f)
      - aperiodic_offset:= a (log10 intercept)
      - aperiodic_exponent:= abs(b)
    """
    out = {"aperiodic_offset": None, "aperiodic_exponent": None, "knee_0": None}
    idata = getattr(sl_model, "idata", None)
    ds = getattr(idata, "posterior", None) if idata is not None else None

    if ds is not None:
        off = _posterior_mean_scalar(ds, ["aperiodic_offset", "offset", "b0"])
        slope = _posterior_mean_scalar(ds, ["slope_0", "slope", "beta"])
        knee = _posterior_mean_scalar(ds, ["knee_0", "knee"])
        if off is not None:
            out["aperiodic_offset"] = float(off)
        if slope is not None:
            out["aperiodic_exponent"] = float(abs(slope))
        if knee is not None:
            out["knee_0"] = float(max(knee, 0.0))

    if (out["aperiodic_offset"] is None) or (out["aperiodic_exponent"] is None):
        m = (F_fit > 0) & np.isfinite(bb_lin) & (bb_lin > 0)
        if np.sum(m) >= 3:
            xf = np.log10(F_fit[m])
            yf = np.log10(bb_lin[m])
            b, a = np.polyfit(xf, yf, 1)  # y = a + b*x
            out["aperiodic_exponent"] = float(abs(b))
            out["aperiodic_offset"] = float(a)
        if out["knee_0"] is None:
            out["knee_0"] = 0.0

    return out


def _slsd_peak_params(model) -> np.ndarray:
    """
    Return posterior means for rhythm centers/widths/heights if available.
    Output shape (Npeaks,3): [center_Hz, A_lin, FWHM_Hz]
    """
    peaks = []
    idata = getattr(model, "idata", None)
    if idata is None:
        return np.empty((0, 3), float)
    ds = getattr(idata, "posterior", None)
    if ds is None:
        return np.empty((0, 3), float)

    # Vector variables
    if "center" in ds and "sigma" in ds:
        Aname = "A_lin" if "A_lin" in ds else ("A" if "A" in ds else None)
        if Aname is not None:
            c = ds["center"].mean(dim=[d for d in ds["center"].dims if d in ("chain", "draw")])
            s = ds["sigma"].mean(dim=[d for d in ds["sigma"].dims if d in ("chain", "draw")])
            a = ds[Aname].mean(dim=[d for d in ds[Aname].dims if d in ("chain", "draw")])
            arr_c = np.ravel(c.values)
            arr_s = np.ravel(s.values)
            arr_a = np.ravel(a.values)
            for ci, si, ai in zip(arr_c, arr_s, arr_a):
                peaks.append([float(ci), float(ai), float(2.3548 * si)])

    # Scalar-per-index variables
    if not peaks:
        varnames = set(ds.data_vars)
        idxs = sorted({int(m.group(1)) for v in varnames for m in [re.search(r"center[_\[](\d+)", v)] if m})
        for k in idxs:
            c = None
            s = None
            a = None
            for v in (f"center_{k}", f"center[{k}]"):
                if v in ds:
                    c = float(ds[v].mean().item())
                    break
            for v in (f"sigma_{k}", f"sigma[{k}]"):
                if v in ds:
                    s = float(ds[v].mean().item())
                    break
            for v in (f"A_lin_{k}", f"A_lin[{k}]", f"A_{k}", f"A[{k}]"):
                if v in ds:
                    a = float(ds[v].mean().item())
                    break
            if c is not None and s is not None and a is not None:
                peaks.append([c, a, 2.3548 * s])

    return np.asarray(peaks, float) if peaks else np.empty((0, 3), float)

def slsd_params_from_model(sl_model, F_fit, bb_lin, *, ref_freq_hz: float = 10.0) -> dict:
    """
    Export Figure 4 fit parameters for Figure 5 simulation.

    IMPORTANT:
    - aperiodic_exponent is the decomposition chi parameter
    - aperiodic_offset is the decomposition b parameter
    - knee_0 is already the Hz-domain kappa used by spectrum
    - broadband_slope_40_60 is saved separately for QA / plotting checks
    """
    peaks = _slsd_peak_params(sl_model)
    peaks_list = []
    rhythms = []
    for cf, A_lin, fwhm in peaks:
        cf = float(cf)
        A_lin = float(A_lin)
        fwhm = float(fwhm)
        sigma = float(fwhm / 2.3548)
        peaks_list.append({"freq": cf, "amplitude": A_lin, "sigma": sigma})
        rhythms.append({"center": cf, "A_lin": A_lin, "sigma": sigma, "FWHM": fwhm})

    ap = _aperiodic_from_model_or_fit(np.asarray(F_fit, float), np.asarray(bb_lin, float), sl_model)

    alpha = float(ap["aperiodic_exponent"])          # decomposition chi
    knee_0 = float(ap["knee_0"])                     # already Hz-domain kappa
    aperiodic_offset = float(ap["aperiodic_offset"]) # decomposition b

    # Keep this only as an auxiliary diagnostic field if the analysis requires it
    f0 = float(ref_freq_hz)
    bb_f0 = float(np.interp(f0, np.asarray(F_fit, float), np.asarray(bb_lin, float)))
    bb_f0 = max(bb_f0, 1e-20)
    b_0 = float(np.log10(bb_f0 * (knee_0 + (f0 ** alpha))))

    local_slope_40_60 = _compute_loglog_slope(
        np.asarray(F_fit, float),
        np.asarray(bb_lin, float),
        *SLOPE_BAND
    )

    return {
        "schema": "SL_specdecomp_params_for_spectrum.v2",
        "ref_freq_hz": f0,

        # Actual decomposition params to drive simulation
        "aperiodic_offset": aperiodic_offset,
        "aperiodic_exponent": alpha,
        "knee_0": knee_0,

        # Optional compatibility / diagnostics
        "slope_0": float(-alpha),
        "alpha": alpha,
        "b_0": b_0,
        "b0_space": "log10",

        # Rhythms
        "peaks_list": peaks_list,
        "rhythms": rhythms,

        # QA: this is the plotted broadband metric, not the simulation exponent
        "broadband_slope_40_60": float(local_slope_40_60),

        "notes": (
            "aperiodic_exponent is the decomposition chi parameter; "
            "knee_0 is already the Hz-domain kappa used by spectrum(); "
            "broadband_slope_40_60 is saved separately for QA."
        ),
    }

def slsd_params_from_model_legacy(sl_model, F_fit, bb_lin, *, ref_freq_hz: float = 10.0) -> dict:
    """
    Build a JSON-serializable dict with *simulation-ready* params for Figure 1.
    """
    peaks = _slsd_peak_params(sl_model)  # [center, A_lin, FWHM]
    peaks_list = []
    rhythms = []
    for cf, A_lin, fwhm in peaks:
        cf = float(cf)
        A_lin = float(A_lin)
        fwhm = float(fwhm)
        sigma = float(fwhm / 2.3548)
        peaks_list.append({"freq": cf, "amplitude": A_lin, "sigma": sigma})
        rhythms.append({"center": cf, "A_lin": A_lin, "sigma": sigma, "FWHM": fwhm})

    ap = _aperiodic_from_model_or_fit(F_fit, bb_lin, sl_model)

    alpha = float(ap["aperiodic_exponent"])
    knee_0_raw = float(ap["knee_0"]) if ap["knee_0"] is not None else 0.0
    knee_0 = _knee_omega_to_hz(knee_0_raw, alpha)
    slope_0 = float(-alpha)

    # Compute b_0 (log10) consistent with spectrum convention using a reference point
    f0 = float(ref_freq_hz)
    bb_f0 = float(np.interp(f0, np.asarray(F_fit, float), np.asarray(bb_lin, float)))
    bb_f0 = max(bb_f0, 1e-20)
    b_0 = float(np.log10(bb_f0 * (knee_0 + (f0 ** alpha))))

    return {
        "schema": "SL_specdecomp_params_for_spectrum.v1",
        "ref_freq_hz": f0,
        "slope_0": slope_0,
        "alpha": alpha,
        "knee_0": knee_0,
        "b_0": b_0,
        "b0_space": "log10",
        "peaks_list": peaks_list,
        "aperiodic_offset": ap["aperiodic_offset"],
        "aperiodic_exponent": ap["aperiodic_exponent"],
        "rhythms": rhythms,
        "notes": (
            "Posterior means where available. "
            "b_0 computed at ref_freq_hz to match spectral_decomposition.spectrum() convention."
        ),
    }


def save_fig4_params(state_name: str, params: dict, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state_name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "bands": {"hg": list(HG_BAND), "slope": list(SLOPE_BAND)},
    }
    path = Path(out_dir) / f"fig4_slsd_params_{state_name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[INFO] Exported SL_specdecomp params → {path}")
    return str(path)


# ──────────────────────────── Small helpers ────────────────────────────
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


def _ensure_tf(power: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(power)
    f = np.asarray(freqs).ravel()
    t = np.asarray(times).ravel()
    if p.ndim != 2:
        raise ValueError(f"spectrogram power must be 2D; got {p.shape}")
    T, F = p.shape
    if T == t.size and F == f.size:
        return p, f, t
    if T == f.size and F == t.size:
        return p.T, f, t
    if abs(T - t.size) + abs(F - f.size) <= abs(T - f.size) + abs(F - t.size):
        return p, f, t
    return p.T, f, t


def _independent_grid(freqs: np.ndarray, twin: float, NW: float) -> Tuple[np.ndarray, int]:
    f = np.asarray(freqs, float)
    df = float(np.median(np.diff(f))) if f.size > 1 else 1.0
    delta_f_indep = 2.0 * NW / twin
    step_bins = max(1, int(round(delta_f_indep / max(df, 1e-12))))
    return f[::step_bins], step_bins


def _compute_loglog_slope(freqs: np.ndarray, power_lin: np.ndarray,
                          fmin: float, fmax: float) -> float:
    f = np.asarray(freqs, float)
    y = np.asarray(power_lin, float)
    mask = (f >= fmin) & (f <= fmax) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return np.nan
    xf = np.log10(f[mask])
    yf = np.log10(y[mask])
    m, _b = np.polyfit(xf, yf, 1)
    return float(m)


def _hg_mean(freqs: np.ndarray, power_lin: np.ndarray,
             fmin: float, fmax: float) -> float:
    f = np.asarray(freqs, float)
    y = np.asarray(power_lin, float)
    mask = (f >= fmin) & (f <= fmax) & np.isfinite(y) & (y > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(y[mask]))


def _extract_slsd(model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ===== specparam: full + aperiodic on grid, plus peak params =====
def _specparam_full_aper_peaks(freqs_fit, power_lin, freq_range, **specparam_kwargs):
    fm = SpectralModel(**specparam_kwargs)
    freqs_fit = np.asarray(freqs_fit, float)
    power_lin = np.clip(np.asarray(power_lin, float), 1e-20, np.inf)
    fm.fit(freqs_fit, power_lin, freq_range=freq_range)

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
        ap_lin   = np.interp(freqs_fit, freq_model, ap_native)
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
        ap_lin   = np.interp(freqs_fit, freq_model, ap_lin_native)

    peaks = np.empty((0, 3), float)
    for key in ("peak", "peaks"):
        try:
            p = np.asarray(fm.get_params(key), float)
            if p.ndim == 1 and p.size == 3:
                p = p[None, :]
            if p.size:
                peaks = p
            break
        except Exception:
            continue

    return full_lin, ap_lin, peaks


def compute_multitaper(x: np.ndarray, fs: float, t0: float, params: Dict[str, Any]):
    x = np.asarray(x, float).ravel()
    x_3d = x[:, np.newaxis, np.newaxis]
    mt = Multitaper(
        x_3d,
        sampling_frequency=fs,
        start_time=float(t0),
        **params,
    )
    conn = Connectivity.from_multitaper(mt)
    P = conn.power().squeeze()
    F = np.asarray(conn.frequencies).ravel()
    T = np.asarray(conn.time).ravel()
    return P, F, T


# ──────────────────────────── Likelihood helpers ────────────────────────────
def _gamma_loglik_multitaper(y_lin: np.ndarray, mu_lin: np.ndarray, k_tapers: int) -> float:
    y = np.clip(np.asarray(y_lin, float).ravel(), 1e-30, np.inf)
    mu = np.clip(np.asarray(mu_lin, float).ravel(), 1e-30, np.inf)
    m = np.isfinite(y) & np.isfinite(mu)
    if m.sum() == 0:
        return np.nan
    y = y[m]
    mu = mu[m]
    K = float(int(k_tapers))
    theta = mu / K
    ll = (K - 1.0) * np.log(y) - (y / theta) - K * np.log(theta) - gammaln(K)
    return float(np.sum(ll))


def _mt_power_one_window(ts_1d: np.ndarray, fs: float, duration: float, nw: float, k_tapers: int) -> Tuple[np.ndarray, np.ndarray]:
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


def _compute_cvll_for_window(
    ts_30s: np.ndarray,
    fs: float,
    cfg: dict,
    cv_folds: int,
    cv_chunk_dur: float,
    cv_nw: float,
    cv_k_tapers: int,
    cv_sl_sample_overrides: Optional[dict] = None,
) -> Dict[str, float]:
    x = np.asarray(ts_30s, float).ravel()
    n_chunk = int(round(float(fs) * float(cv_chunk_dur)))
    n_expect = int(cv_folds) * n_chunk
    if x.size < n_expect:
        return {"specparam": np.nan, "slsd": np.nan}
    if x.size > n_expect:
        x = x[:n_expect]

    chunks = [x[i * n_chunk:(i + 1) * n_chunk] for i in range(int(cv_folds))]

    f0, _S0 = _mt_power_one_window(chunks[0], fs=fs, duration=cv_chunk_dur, nw=cv_nw, k_tapers=cv_k_tapers)
    f_ref_cv, _step = _independent_grid(f0, cv_chunk_dur, cv_nw)

    S_chunks = []
    for c in chunks:
        f_emp, S_emp = _mt_power_one_window(c, fs=fs, duration=cv_chunk_dur, nw=cv_nw, k_tapers=cv_k_tapers)
        S_fit = np.interp(f_ref_cv, f_emp, S_emp)
        S_chunks.append(np.clip(S_fit, 1e-20, np.inf))
    S_chunks = np.asarray(S_chunks, float)

    m_fr = (f_ref_cv >= ANALYSIS_FRANGE[0]) & (f_ref_cv <= ANALYSIS_FRANGE[1])
    f_cv = f_ref_cv[m_fr]
    if f_cv.size < 10:
        return {"specparam": np.nan, "slsd": np.nan}

    sl_kw_cv_base = dict(cfg["slsd_kwargs"])
    if cv_sl_sample_overrides:
        sk = dict(sl_kw_cv_base.get("sample_kwargs", {}))
        sk.update(cv_sl_sample_overrides)
        sk["cores"] = 1
        sl_kw_cv_base["sample_kwargs"] = sk

    cvll_spec = 0.0
    cvll_slsd = 0.0

    for i_test in range(int(cv_folds)):
        test = np.asarray(S_chunks[i_test, m_fr], float)
        train_idx = [j for j in range(int(cv_folds)) if j != i_test]
        train = np.mean(S_chunks[train_idx, :], axis=0)[m_fr]
        fr_k = (max(ANALYSIS_FRANGE[0], float(f_cv[0])), min(ANALYSIS_FRANGE[1], float(f_cv[-1])))

        # specparam
        if np.isfinite(cvll_spec):
            try:
                mu_sp, _ap, _pk = _specparam_full_aper_peaks(f_cv, train, fr_k, **cfg["specparam_kwargs"])
                ll = _gamma_loglik_multitaper(test, mu_sp, cv_k_tapers)
                cvll_spec = cvll_spec + ll if np.isfinite(ll) else np.nan
            except Exception:
                cvll_spec = np.nan

        # SL_specdecomp
        if np.isfinite(cvll_slsd):
            try:
                sl_kw_cv = _slsd_kwargs_with_b_prior(sl_kw_cv_base, f_cv, train)
                sl = Decompose(f_cv, np.clip(train, 1e-20, np.inf), fs=fs, **sl_kw_cv)
                mu_sl, _bb, _rh = _extract_slsd(sl)
                ll = _gamma_loglik_multitaper(test, mu_sl, cv_k_tapers)
                cvll_slsd = cvll_slsd + ll if np.isfinite(ll) else np.nan
            except Exception:
                cvll_slsd = np.nan

    return {"specparam": float(cvll_spec), "slsd": float(cvll_slsd)}


# ──────────────────────────── Peak helpers ────────────────────────────
def _quadratic_refine(x: np.ndarray, y: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(y) - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0:
        return float(x[i])
    delta = 0.5 * (y0 - y2) / denom
    return float(x[i] + delta * (x[i + 1] - x[i]))


def _collect_peaks_from_curve(freqs: np.ndarray, rh_matrix: np.ndarray,
                              rel_height: float = 0.5, min_rel_amp: float = 0.02
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(freqs, float).ravel()
    df = float(np.median(np.diff(f))) if len(f) > 1 else 1.0
    win_idx, centers, widths_hz = [], [], []
    TW = rh_matrix.shape[0]
    for ti in range(TW):
        y = np.asarray(rh_matrix[ti, :], float)
        if not np.any(np.isfinite(y)):
            continue
        ymax = float(np.nanmax(y))
        thr = max(1e-20, min_rel_amp * ymax)
        idx, _ = find_peaks(y, height=thr)
        if idx.size == 0:
            continue
        w_bins = peak_widths(y, idx, rel_height=rel_height)[0]
        for k, ii in enumerate(idx):
            centers.append(_quadratic_refine(f, y, int(ii)))
            widths_hz.append(float(w_bins[k] * df))
            win_idx.append(ti)
    return np.asarray(win_idx, int), np.asarray(centers, float), np.asarray(widths_hz, float)


# ──────────────────────────── Row 4, Col 3: CVLL “aux” panel ────────────────────────────
def _cvll_aux_panel(
    ax: plt.Axes,
    cvll_specparam: np.ndarray,
    cvll_slsd: np.ndarray,
    *,
    title: str,
    max_windows: Optional[int] = None,
    x_step: float = 0.40,
    x_jitter: float = 0.018,
    line_alpha: float = 0.22,
    point_alpha: float = 0.80,
    lw: float = 1.0,
    point_size: float = 16.0,
    star_size: float = 70.0,
    seed: int = 0,
) -> None:
    a = np.asarray(cvll_specparam, float).ravel()
    b = np.asarray(cvll_slsd, float).ravel()
    n = int(min(a.size, b.size))
    if n == 0:
        ax.text(0.5, 0.5, "No CVLL", ha="center", va="center")
        ax.axis("off")
        return

    if max_windows is not None:
        n = int(min(n, int(max_windows)))

    a = a[:n]
    b = b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) == 0:
        ax.text(0.5, 0.5, "No finite CVLL", ha="center", va="center")
        ax.axis("off")
        return

    a = a[m]
    b = b[m]
    n_valid = int(a.size)

    xs = np.arange(2, dtype=float) * float(x_step)

    rng = np.random.default_rng(int(seed))
    jit = rng.uniform(-float(x_jitter), float(x_jitter), size=(n_valid, 1))
    x0 = xs[0] + jit[:, 0]
    x1 = xs[1] + jit[:, 0]

    for i in range(n_valid):
        ax.plot([x0[i], x1[i]], [a[i], b[i]], color="0.4", alpha=float(line_alpha), lw=float(lw), zorder=1)

    ax.scatter(
        x0, a,
        s=float(point_size),
        alpha=float(point_alpha),
        marker="o",
        color=COLORS["specparam_pt"],
        edgecolor="none",
        zorder=3,
        label="specparam",
    )
    ax.scatter(
        x1, b,
        s=float(point_size),
        alpha=float(point_alpha),
        marker="o",
        color=COLORS["slsd_pt"],
        edgecolor="none",
        zorder=3,
        label="SL_specdecomp",
    )

    y = np.column_stack([a, b])
    row_max = np.nanmax(y, axis=1, keepdims=True)
    is_max = np.isclose(y, row_max, rtol=0.0, atol=0.0)

    if np.any(is_max[:, 0]):
        ax.scatter(
            x0[is_max[:, 0]], a[is_max[:, 0]],
            s=float(star_size),
            marker="*",
            color=COLORS["specparam_pt"],
            edgecolor="k",
            linewidths=0.5,
            alpha=0.95,
            zorder=4,
        )
    if np.any(is_max[:, 1]):
        ax.scatter(
            x1[is_max[:, 1]], b[is_max[:, 1]],
            s=float(star_size),
            marker="*",
            color=COLORS["slsd_pt"],
            edgecolor="k",
            linewidths=0.5,
            alpha=0.95,
            zorder=4,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(["specparam", "SL_specdecomp"], rotation=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("CVLL")
    ax.set_title(f"{title}\n(CVLL aux; N={n_valid} windows)")
    sns.despine(ax=ax, top=True, right=True)
    ax.minorticks_off()

    pad = 0.28
    ax.set_xlim(xs[0] - pad, xs[-1] + pad)

    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["specparam_pt"], markersize=6, label="specparam"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["slsd_pt"], markersize=6, label="SL_specdecomp"),
        ],
        frameon=False,
        loc="best",
    )


# ──────────────────────────── Core per-condition runner ────────────────────────────
def run_condition(
    name: str,
    windows_mode: str = "all",
    *,
    cv_folds: int = CV_FOLDS,
    cv_nw: float = CV_NW,
    cv_k_tapers: int = CV_K_TAPERS,
    cv_chunk_dur: Optional[float] = None,
    cv_sl_draws: int = -1,
    cv_sl_tune: int = -1,
    cv_sl_chains: int = -1,
    save_payload: bool = True,
    plot_only: bool = False,
    force_recompute: bool = False,
    drop_orig: Optional[List[int]] = None,
    drop_awake: Optional[List[int]] = None,
    drop_anes: Optional[List[int]] = None,
) -> None:
    cfg = CONDITION_CFG[name]
    drop_orig = list(drop_orig or [])
    drop_awake = list(drop_awake or [])
    drop_anes = list(drop_anes or [])

    if cv_chunk_dur is None:
        cv_chunk_dur = float(MT_PARAMS["time_window_duration"]) / float(cv_folds)

    base = _fig4_payload_base(name, windows_mode)

    # ── Optional cache load (plot-only mode) ─────────────────────────────
    skip_compute = False
    sel_orig = None

    if plot_only and not force_recompute:
        arrays, meta_file = load_plot_payload_fig4(base)

        # Pull arrays
        F_fit = arrays["F_fit"]
        P_fit_tf = arrays["P_fit_tf"]
        T_rel_min = arrays["T_rel_min"]

        specparam_full = arrays["specparam_full"]
        specparam_aper = arrays["specparam_aper"]
        specparam_rh   = arrays["specparam_rh"]

        slsd_total = arrays["slsd_total"]
        slsd_bb    = arrays["slsd_bb"]
        slsd_rh    = arrays["slsd_rh"]

        slopes_specparam = arrays["slopes_specparam"]
        slopes_slsd      = arrays["slopes_slsd"]
        hg_specparam     = arrays["hg_specparam"]
        hg_slsd          = arrays["hg_slsd"]

        cvll_specparam = arrays["cvll_specparam"]
        cvll_slsd      = arrays["cvll_slsd"]

        idx_target = int(meta_file["meta"]["idx_target"])
        label_min = meta_file["meta"]["label_min"]
        #raster_frange = tuple(meta_file["meta"]["raster_frange"])
        raster_frange = (0.0, float(meta_file["meta"]["raster_frange"][1]))

        sp_peaks_list = meta_file["meta"].get("specparam_peaks_list", [])
        sl_peaks_list = meta_file["meta"].get("slsd_peaks_list", [])
        sp_peaks_list = [np.asarray(p, float) for p in sp_peaks_list]
        sl_peaks_list = [np.asarray(p, float) for p in sl_peaks_list]

        if "CV" in meta_file["meta"]:
            cv_folds = int(meta_file["meta"]["CV"]["cv_folds"])
            cv_chunk_dur = float(meta_file["meta"]["CV"]["cv_chunk_dur"])
            cv_nw = float(meta_file["meta"]["CV"]["cv_nw"])
            cv_k_tapers = int(meta_file["meta"]["CV"]["cv_k_tapers"])

        # NEW: apply drop using meta if present; else use CLI drop list
        drop_use = list(meta_file["meta"].get("drop_orig", [])) or list(drop_orig)
        arrays_local = {
            "P_fit_tf": P_fit_tf,
            "T_rel_min": T_rel_min,
            "specparam_full": specparam_full,
            "specparam_aper": specparam_aper,
            "specparam_rh": specparam_rh,
            "slsd_total": slsd_total,
            "slsd_bb": slsd_bb,
            "slsd_rh": slsd_rh,
            "slopes_specparam": slopes_specparam,
            "slopes_slsd": slopes_slsd,
            "hg_specparam": hg_specparam,
            "hg_slsd": hg_slsd,
            "cvll_specparam": cvll_specparam,
            "cvll_slsd": cvll_slsd,
        }
        arrays_local, sp_peaks_list, sl_peaks_list, idx_target, keep_orig, _ = _apply_window_drop(
            drop_orig=drop_use,
            arrays=arrays_local,
            sp_peaks_list=sp_peaks_list,
            sl_peaks_list=sl_peaks_list,
            idx_target=idx_target,
            sel_orig=None,
            sl_models=None,
        )

        # unpack filtered
        P_fit_tf = arrays_local["P_fit_tf"]
        T_rel_min = arrays_local["T_rel_min"]
        specparam_full = arrays_local["specparam_full"]
        specparam_aper = arrays_local["specparam_aper"]
        specparam_rh   = arrays_local["specparam_rh"]
        slsd_total = arrays_local["slsd_total"]
        slsd_bb    = arrays_local["slsd_bb"]
        slsd_rh    = arrays_local["slsd_rh"]
        slopes_specparam = arrays_local["slopes_specparam"]
        slopes_slsd      = arrays_local["slopes_slsd"]
        hg_specparam     = arrays_local["hg_specparam"]
        hg_slsd          = arrays_local["hg_slsd"]
        cvll_specparam   = arrays_local["cvll_specparam"]
        cvll_slsd        = arrays_local["cvll_slsd"]

        label_min = f"t ≈ {T_rel_min[idx_target]:.2f} min"  # refresh label after drop

        skip_compute = True

    # ── Compute branch ───────────────────────────────────────────────────
    if not skip_compute:
        rbands = cfg["slsd_kwargs"].get("rhythm_bands", [])
        if rbands:
            #raster_frange = (min(b[0] for b in rbands), max(b[1] for b in rbands))
            raster_frange = (0.0, max(b[1] for b in rbands))
        else:
            raster_frange = RASTER_FRANGE

        ecog = loadmat(ECOG_MAT, squeeze_me=True)
        time = loadmat(TIME_MAT, squeeze_me=True)
        x = np.asarray(ecog[ECOG_KEY], float).squeeze()
        t = np.asarray(time[TIME_KEY], float).squeeze()
        mvalid = np.isfinite(x) & np.isfinite(t)
        x, t = x[mvalid], t[mvalid]
        fs = 1000.0

        cond = loadmat(COND_MAT, simplify_cells=True)["Condition"]
        ct = np.asarray(cond[COND_TIME_KEY], float).ravel()
        raw_labels = np.ravel(cond[COND_LABEL_KEY])
        labels = [lab.decode("utf-8") if isinstance(lab, (bytes, bytearray)) else str(lab) for lab in raw_labels]
        order = np.argsort(ct)
        ct, labels = ct[order], [labels[i] for i in order]

        seg = _find_interval(ct, labels, cfg["start_phrase"], cfg["end_phrase"])
        if seg is None:
            raise RuntimeError(f"Could not find interval for {name}: '{cfg['start_phrase']}' ... '{cfg['end_phrase']}'")
        t0_seg, t1_seg = seg

        t_seg, x_seg = _restrict_to_interval(t, x, t0_seg, t1_seg)

        P_tf, F_all, T_abs = compute_multitaper(x_seg, fs=fs, t0=float(t_seg[0]), params=MT_PARAMS)
        P_tf, F_all, T_abs = _ensure_tf(P_tf, F_all, T_abs)

        fmin, fmax = ANALYSIS_FRANGE
        mask = (F_all > 0) & (F_all >= fmin) & (F_all <= fmax)
        F_an = F_all[mask]
        P_an_tf = P_tf[:, mask]
        F_fit, step = _independent_grid(F_an, MT_PARAMS["time_window_duration"], MT_PARAMS["time_halfbandwidth_product"])
        P_fit_tf = P_an_tf[:, ::step]

        T_rel_min = (T_abs - float(t_seg[0])) / 60.0
        idx_target = int(np.argmin(np.abs(T_rel_min - float(cfg["target_min_after_start"]))))
        label_min = f"t ≈ {T_rel_min[idx_target]:.2f} min"

        sel_orig = None
        if windows_mode == "2":
            TW_all = P_fit_tf.shape[0]
            neighbor = min(idx_target + 1, TW_all - 1) if idx_target < TW_all - 1 else max(idx_target - 1, 0)
            sel = sorted(set([idx_target, neighbor]))
            sel_orig = list(sel)
            P_fit_tf = P_fit_tf[sel, :]
            T_rel_min = T_rel_min[sel]
            idx_target = 0
            label_min = f"t ≈ {T_rel_min[idx_target]:.2f} min  (2-window preview)"

        TW, FW = P_fit_tf.shape
        specparam_full = np.full((TW, FW), np.nan)
        specparam_aper = np.full((TW, FW), np.nan)
        specparam_rh   = np.full((TW, FW), np.nan)
        sp_peaks_list: List[np.ndarray] = []

        slsd_total = np.full((TW, FW), np.nan)
        slsd_bb    = np.full((TW, FW), np.nan)
        slsd_rh    = np.full((TW, FW), np.nan)
        sl_models: List[Any] = []
        sl_peaks_list: List[np.ndarray] = []

        slopes_specparam = np.full(TW, np.nan)
        slopes_slsd      = np.full(TW, np.nan)
        hg_specparam     = np.full(TW, np.nan)
        hg_slsd          = np.full(TW, np.nan)

        cvll_specparam = np.full(TW, np.nan)
        cvll_slsd      = np.full(TW, np.nan)

        cv_sl_overrides = {}
        if cv_sl_draws > 0:
            cv_sl_overrides["draws"] = int(cv_sl_draws)
        if cv_sl_tune > 0:
            cv_sl_overrides["tune"] = int(cv_sl_tune)
        if cv_sl_chains > 0:
            cv_sl_overrides["chains"] = int(cv_sl_chains)

        win_step = float(MT_PARAMS["time_window_step"])
        win_dur = float(MT_PARAMS["time_window_duration"])
        n_win = int(round(win_dur * fs))

        for ti in range(TW):
            y = np.clip(P_fit_tf[ti, :], 1e-20, np.inf)
            fr_k = (max(ANALYSIS_FRANGE[0], float(F_fit[0])), min(ANALYSIS_FRANGE[1], float(F_fit[-1])))

            sp_full, sp_ap, sp_peaks = _specparam_full_aper_peaks(F_fit, y, fr_k, **cfg["specparam_kwargs"])
            specparam_full[ti, :] = sp_full
            specparam_aper[ti, :] = sp_ap
            specparam_rh[ti, :]   = np.clip(sp_full - sp_ap, 0.0, np.inf)
            sp_peaks_list.append(np.asarray(sp_peaks, float))

            sl_kw = _slsd_kwargs_with_b_prior(cfg["slsd_kwargs"], F_fit, y)
            sl = Decompose(F_fit, y, fs=fs, **sl_kw)
            sl_tot, sl_bb, sl_rh = _extract_slsd(sl)
            slsd_total[ti, :] = sl_tot
            slsd_bb[ti, :]    = sl_bb
            slsd_rh[ti, :]    = sl_rh
            sl_models.append(sl)
            sl_peaks_list.append(_slsd_peak_params(sl))

            slopes_specparam[ti] = _compute_loglog_slope(F_fit, sp_ap, *SLOPE_BAND)
            slopes_slsd[ti]      = _compute_loglog_slope(F_fit, sl_bb, *SLOPE_BAND)
            hg_specparam[ti]     = _hg_mean(F_fit, sp_ap, *HG_BAND)
            hg_slsd[ti]          = _hg_mean(F_fit, sl_bb, *HG_BAND)

            # CVLL for this 30s window (used in Row 4 Col 3)
            orig_i = sel_orig[ti] if sel_orig is not None else ti
            i0 = int(round(orig_i * win_step * fs))
            i1 = i0 + n_win
            if i0 >= 0 and i1 <= x_seg.size:
                x_win = np.asarray(x_seg[i0:i1], float).ravel()
                cvll = _compute_cvll_for_window(
                    ts_30s=x_win,
                    fs=fs,
                    cfg=cfg,
                    cv_folds=int(cv_folds),
                    cv_chunk_dur=float(cv_chunk_dur),
                    cv_nw=float(cv_nw),
                    cv_k_tapers=int(cv_k_tapers),
                    cv_sl_sample_overrides=(cv_sl_overrides if len(cv_sl_overrides) else None),
                )
                cvll_specparam[ti] = cvll["specparam"]
                cvll_slsd[ti] = cvll["slsd"]

        # NEW: apply global window drop (affects all panels + payload)
        arrays_local = {
            "P_fit_tf": P_fit_tf,
            "T_rel_min": T_rel_min,
            "specparam_full": specparam_full,
            "specparam_aper": specparam_aper,
            "specparam_rh": specparam_rh,
            "slsd_total": slsd_total,
            "slsd_bb": slsd_bb,
            "slsd_rh": slsd_rh,
            "slopes_specparam": slopes_specparam,
            "slopes_slsd": slopes_slsd,
            "hg_specparam": hg_specparam,
            "hg_slsd": hg_slsd,
            "cvll_specparam": cvll_specparam,
            "cvll_slsd": cvll_slsd,
        }
        arrays_local, sp_peaks_list, sl_peaks_list, idx_target, keep_orig, sl_models = _apply_window_drop(
            drop_orig=list(drop_orig),
            arrays=arrays_local,
            sp_peaks_list=sp_peaks_list,
            sl_peaks_list=sl_peaks_list,
            idx_target=idx_target,
            sel_orig=sel_orig,
            sl_models=sl_models,
        )

        P_fit_tf = arrays_local["P_fit_tf"]
        T_rel_min = arrays_local["T_rel_min"]
        specparam_full = arrays_local["specparam_full"]
        specparam_aper = arrays_local["specparam_aper"]
        specparam_rh   = arrays_local["specparam_rh"]
        slsd_total = arrays_local["slsd_total"]
        slsd_bb    = arrays_local["slsd_bb"]
        slsd_rh    = arrays_local["slsd_rh"]
        slopes_specparam = arrays_local["slopes_specparam"]
        slopes_slsd      = arrays_local["slopes_slsd"]
        hg_specparam     = arrays_local["hg_specparam"]
        hg_slsd          = arrays_local["hg_slsd"]
        cvll_specparam   = arrays_local["cvll_specparam"]
        cvll_slsd        = arrays_local["cvll_slsd"]

                # Re-select the display/export target by actual time AFTER outlier dropping.
        # This guarantees anesthesia uses the nearest retained window to the specified
        # 4.00 min target, rather than inheriting a stale/remapped pre-drop index.
        target_min_requested = float(cfg["target_min_after_start"])
        idx_target = int(np.nanargmin(np.abs(np.asarray(T_rel_min, float) - target_min_requested)))
        label_min = f"t ≈ {T_rel_min[idx_target]:.2f} min"

        print(
            f"[INFO] {name}: requested target={target_min_requested:.2f} min; "
            f"post-drop selected idx={idx_target}, "
            f"T_rel_min={float(T_rel_min[idx_target]):.2f} min; "
            f"keep_orig={keep_orig}"
        )

        # Export SL_specdecomp params for Figure 1 (after drop; uses filtered idx_target)
        sl_target_model = sl_models[idx_target] if sl_models is not None else None

        if sl_target_model is not None:
            bb_target = slsd_bb[idx_target, :]
            export_params = slsd_params_from_model(sl_target_model, F_fit, bb_target, ref_freq_hz=10.0)
            _ = save_fig4_params(name, export_params, SAVE_DIR)
        else:
            export_params = {}

        # Payload saving
        if save_payload:
            arrays = {
                "F_fit": F_fit,
                "P_fit_tf": P_fit_tf,
                "T_rel_min": T_rel_min,

                "specparam_full": specparam_full,
                "specparam_aper": specparam_aper,
                "specparam_rh": specparam_rh,

                "slsd_total": slsd_total,
                "slsd_bb": slsd_bb,
                "slsd_rh": slsd_rh,

                "slopes_specparam": slopes_specparam,
                "slopes_slsd": slopes_slsd,
                "hg_specparam": hg_specparam,
                "hg_slsd": hg_slsd,

                "cvll_specparam": cvll_specparam,
                "cvll_slsd": cvll_slsd,
            }
            meta = {
                "state": name,
                "windows_mode": windows_mode,
                "idx_target": int(idx_target),
                "label_min": label_min,
                "target_min_after_start": float(cfg["target_min_after_start"]),
                "target_selection_policy": "nearest retained post-drop T_rel_min to target_min_after_start",
                "raster_frange": list(raster_frange),
                "analysis_frange": list(ANALYSIS_FRANGE),
                "HG_BAND": list(HG_BAND),
                "SLOPE_BAND": list(SLOPE_BAND),
                "MT_PARAMS": MT_PARAMS,
                "CV": dict(cv_folds=int(cv_folds), cv_chunk_dur=float(cv_chunk_dur), cv_nw=float(cv_nw), cv_k_tapers=int(cv_k_tapers)),
                "SL_specdecomp_b_prior": dict(
                    param_names=["b_0", "b"],
                    mu="log10(median(y_pos)) for each fit",
                    sigma=float(B_PRIOR_SIGMA),
                ),
                "specparam_peaks_list": [np.asarray(p, float).tolist() for p in sp_peaks_list],
                "slsd_peaks_list": [np.asarray(p, float).tolist() for p in sl_peaks_list],
                "export_params_path": f"fig4_slsd_params_{name}.json",
                "export_params": export_params,
                # NEW: record window drop
                "drop_orig": list(drop_orig),
                "keep_orig": list(keep_orig),
            }
            save_plot_payload_fig4(base, arrays=arrays, meta=meta)

    # ──────────────────────────── Figure (4 rows; row4 has 3 cols) ────────────────────────────
    fig = plt.figure(figsize=(16.0, 14.2))

    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.0, 0.9, 0.9, 1.1],
        wspace=0.28, hspace=0.62
    )

    # Row 0
    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])

    # Row 1
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])

    # Row 2
    ax31 = fig.add_subplot(gs[2, 0])
    ax32 = fig.add_subplot(gs[2, 1])

    # Row 3 (nested 1x3 spanning both columns)
    gs_row4 = gs[3, :].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.40)
    ax41 = fig.add_subplot(gs_row4[0, 0])
    ax42 = fig.add_subplot(gs_row4[0, 1])
    ax43 = fig.add_subplot(gs_row4[0, 2])

    # Row 1 - Single-window fits (NO FWHM/2 overlays)
    ax11.set_xscale("log"); ax11.set_yscale("log"); ax11.set_ylim(*PSD_YLIM)
    ax11.set_title(f"{name.capitalize()} — specparam ({label_min})")
    ax11.plot(F_fit, P_fit_tf[idx_target, :], color=COLORS["emp"], **STYLES["emp"], label="Multitaper (30 s)")
    ax11.plot(F_fit, specparam_full[idx_target, :], color=COLORS["full"], **STYLES["full"], label="specparam full")
    ax11.plot(F_fit, specparam_aper[idx_target, :], color=COLORS["broad"], **STYLES["component"], label="aperiodic")
    ax11.plot(F_fit, specparam_rh[idx_target, :],   color=COLORS["rhythms"], **STYLES["component"], label="rhythms")
    ax11.set_xlabel("Frequency (Hz, log10)")
    ax11.set_ylabel("Power (log10)")
    ax11.legend(frameon=False, fontsize=9, loc="best")

    ax12.set_xscale("log"); ax12.set_yscale("log"); ax12.set_ylim(*PSD_YLIM)
    ax12.set_title(f"{name.capitalize()} — SL_specdecomp ({label_min})")
    ax12.plot(F_fit, P_fit_tf[idx_target, :], color=COLORS["emp"], **STYLES["emp"], label="Multitaper (30 s)")
    ax12.plot(F_fit, slsd_total[idx_target, :], color=COLORS["full"], **STYLES["full"], label="SL_specdecomp full")
    ax12.plot(F_fit, slsd_bb[idx_target, :],    color=COLORS["broad"], **STYLES["component"], label="broadband")
    ax12.plot(F_fit, slsd_rh[idx_target, :],    color=COLORS["rhythms"], **STYLES["component"], label="rhythms")
    ax12.set_xlabel("Frequency (Hz, log10)")
    ax12.set_ylabel("Power (log10)")
    ax12.legend(frameon=False, fontsize=9, loc="best")

    # Row 2 - All-fits overlay
    ax21.set_xscale("log"); ax21.set_yscale("log"); ax21.set_ylim(*PSD_YLIM)
    for ti in range(P_fit_tf.shape[0]):
        ax21.plot(F_fit, specparam_full[ti, :], color=COLORS["overlay"], alpha=0.10, lw=1.2)
    ax21.set_title("specparam — full model across windows")
    ax21.set_xlabel("Frequency (Hz, log10)")
    ax21.set_ylabel("Power (log10)")

    ax22.set_xscale("log"); ax22.set_yscale("log"); ax22.set_ylim(*PSD_YLIM)
    for ti in range(P_fit_tf.shape[0]):
        ax22.plot(F_fit, slsd_total[ti, :], color=COLORS["overlay"], alpha=0.10, lw=1.2)
    ax22.set_title("SL_specdecomp — full model across windows")
    ax22.set_xlabel("Frequency (Hz, log10)")
    ax22.set_ylabel("Power (log10)")

    # Row 3 - Peak rasters
    rlo, rhi = raster_frange

    sp_wins, sp_cf, sp_w = [], [], []
    for wi, pk in enumerate(sp_peaks_list):
        if pk is None or np.size(pk) == 0:
            continue
        for cf, amp, fwhm in np.asarray(pk, float):
            if rlo <= cf <= rhi:
                sp_wins.append(wi); sp_cf.append(cf); sp_w.append(fwhm)
    if not sp_cf:
        w_i, cf_i, w_iw = _collect_peaks_from_curve(F_fit, specparam_rh)
        m = (cf_i >= rlo) & (cf_i <= rhi)
        sp_wins, sp_cf, sp_w = list(w_i[m]), list(cf_i[m]), list(w_iw[m])

    sl_wins, sl_cf, sl_w = [], [], []
    for wi, pk in enumerate(sl_peaks_list):
        if pk is None or np.size(pk) == 0:
            continue
        for cf, amp, fwhm in np.asarray(pk, float):
            if rlo <= cf <= rhi:
                sl_wins.append(wi); sl_cf.append(cf); sl_w.append(fwhm)
    if not sl_cf:
        w_i, cf_i, w_iw = _collect_peaks_from_curve(F_fit, slsd_rh)
        m = (cf_i >= rlo) & (cf_i <= rhi)
        sl_wins, sl_cf, sl_w = list(w_i[m]), list(cf_i[m]), list(w_iw[m])

    if sp_cf:
        ax31.errorbar(np.asarray(sp_cf), np.asarray(sp_wins),
                      xerr=np.asarray(sp_w) / 2.0,
                      fmt='o', ms=4.0, lw=1.0, mfc="none",
                      mec=COLORS["specparam_pt"], ecolor=COLORS["specparam_pt"],
                      alpha=0.9, label="specparam (center ± FWHM/2)")
    ax31.set_xlim(raster_frange)
    ax31.set_ylim(-0.5, max(P_fit_tf.shape[0] - 0.5, 0.5))
    ax31.set_xlabel("Frequency (Hz)")
    ax31.set_ylabel("Window #")
    ax31.set_title("Peak raster — specparam")
    ax31.legend(frameon=False, fontsize=9, loc="upper right")

    if sl_cf:
        ax32.errorbar(np.asarray(sl_cf), np.asarray(sl_wins),
                      xerr=np.asarray(sl_w) / 2.0,
                      fmt='s', ms=4.0, lw=1.0, mfc="none",
                      mec=COLORS["slsd_pt"], ecolor=COLORS["slsd_pt"],
                      alpha=0.9, label="SL_specdecomp (center ± FWHM/2)")
    ax32.set_xlim(raster_frange)
    ax32.set_ylim(-0.5, max(P_fit_tf.shape[0] - 0.5, 0.5))
    ax32.set_xlabel("Frequency (Hz)")
    ax32.set_ylabel("")
    ax32.set_title("Peak raster — SL_specdecomp")
    ax32.legend(frameon=False, fontsize=9, loc="upper right")

    # Row 4 - violin HG, violin slope, CVLL aux spaghetti
    shared_hg_ylim, shared_slope_ylim = _resolve_shared_metric_ylims(
        name,
        windows_mode,
        hg_specparam=hg_specparam,
        hg_slsd=hg_slsd,
        slopes_specparam=slopes_specparam,
        slopes_slsd=slopes_slsd,
        drop_awake=drop_awake,
        drop_anes=drop_anes,
    )

    df_hg = pd.DataFrame({
        "value": np.concatenate([hg_specparam, hg_slsd]),
        "method": (["specparam aperiodic"] * len(hg_specparam)) + (["SL_specdecomp broadband"] * len(hg_slsd)),
    })
    sns.violinplot(
        data=df_hg, x="method", y="value",
        inner="quartile", cut=4, bw="scott",
        linewidth=1.0, width=0.9, palette=sns.color_palette("deep", 2), ax=ax41
    )
    sns.stripplot(data=df_hg, x="method", y="value", color="k", alpha=0.55, size=4, jitter=0.10, ax=ax41)
    ax41.set_title("High-gamma mean power (80–180 Hz)")
    ax41.set_xlabel("")
    ax41.set_ylabel("Linear power")
    #ax41.tick_params(axis="x", rotation=15)
    ax41.tick_params(axis="x", rotation=15)
    if shared_hg_ylim is not None:
        ax41.set_ylim(*shared_hg_ylim)

    df_sl = pd.DataFrame({
        "value": np.concatenate([slopes_specparam, slopes_slsd]),
        "method": (["specparam aperiodic"] * len(slopes_specparam)) + (["SL_specdecomp broadband"] * len(slopes_slsd)),
    })
    sns.violinplot(
        data=df_sl, x="method", y="value",
        inner="quartile", cut=4, bw="scott",
        linewidth=1.0, width=0.9, palette=sns.color_palette("deep", 2), ax=ax42
    )
    sns.stripplot(data=df_sl, x="method", y="value", color="k", alpha=0.55, size=4, jitter=0.10, ax=ax42)
    ax42.set_title("Broadband slope (40–60 Hz)")
    ax42.set_xlabel("")
    ax42.set_ylabel("Slope")
    #ax42.tick_params(axis="x", rotation=15)
    ax42.tick_params(axis="x", rotation=15)
    if shared_slope_ylim is not None:
        ax42.set_ylim(*shared_slope_ylim)

    _cvll_aux_panel(
        ax43,
        cvll_specparam=cvll_specparam,
        cvll_slsd=cvll_slsd,
        title=f"{name.capitalize()}",
        max_windows=None,
        x_step=0.40,
        x_jitter=0.018,
        line_alpha=0.30,
        point_alpha=0.85,
        lw=1.0,
        point_size=16.0,
        star_size=70.0,
        seed=0,
    )

    fig.suptitle(
        f"Figure 4 (CV) — {name.capitalize()} "
        f"(rows1–3: 30 s MT K={MT_PARAMS['n_tapers']}, NW={MT_PARAMS['time_halfbandwidth_product']}; "
        f"row4: metrics + CVLL aux; CV={cv_folds}×{cv_chunk_dur:.0f}s, MT K={cv_k_tapers}, NW={cv_nw})",
        y=0.995,
        fontsize=14,
    )

    plt.tight_layout()

    suffix = "_2win" if windows_mode == "2" else ""
    png = os.path.join(SAVE_DIR, f"Figure_4_CV_{name}{suffix}.png")
    svg = os.path.join(SAVE_DIR, f"Figure_4_CV_{name}{suffix}.svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {name} figure → {png} / {svg}")


# ──────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Build Figure 4 (Awake & Anesthesia) with specparam + SL_specdecomp + CVLL aux panel (no Row 5)."
    )
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip all compute; load saved NPZ payload and only regenerate figure.")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Ignore existing payload cache and recompute everything.")

    ap.add_argument("--mode", choices=["both", "awake", "anesthesia"], default="both")
    ap.add_argument("--windows", choices=["all", "2"], default="all")
    ap.add_argument("--quick", action="store_true", help="Equivalent to --windows 2.")

    ap.set_defaults(save_payload=True)
    ap.add_argument("--no-payload", dest="save_payload", action="store_false",
                    help="Disable saving NPZ+JSON plot payload.")

    # CV params
    ap.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    ap.add_argument("--cv-chunk-dur", type=float, default=None, help="Seconds; default is 30/cv_folds.")
    ap.add_argument("--cv-nw", type=float, default=CV_NW)
    ap.add_argument("--cv-k-tapers", type=int, default=CV_K_TAPERS)

    # CV sampler overrides
    ap.add_argument("--cv-sl-draws", type=int, default=-1)
    ap.add_argument("--cv-sl-tune", type=int, default=-1)
    ap.add_argument("--cv-sl-chains", type=int, default=-1)

    # NEW: outlier removal
    ap.add_argument("--drop-awake", type=str, default="",
                    help="Comma-separated ORIGINAL window indices to drop for awake (e.g., '3,17').")
    ap.add_argument("--drop-anes", type=str, default="",
                    help="Comma-separated ORIGINAL window indices to drop for anesthesia.")
    ap.add_argument("--drop-from-outlier-csv", type=str, default="",
                    help="Path to outlier_inspector/outliers_report.csv; drops all reported windows per condition.")

    args = ap.parse_args()

    # Only enforce.mat presence when computing (plot-only doesn't need them)
    if not args.plot_only or args.force_recompute:
        for pth in [ECOG_MAT, TIME_MAT, COND_MAT]:
            if not os.path.exists(pth):
                raise FileNotFoundError(f"Missing required file: {pth}")

    windows_mode = "2" if args.quick else args.windows

    # Build drop lists (union manual + CSV if provided)
    drop_awake = _parse_int_list(args.drop_awake)
    drop_anes  = _parse_int_list(args.drop_anes)
    if str(args.drop_from_outlier_csv).strip():
        drop_awake = sorted(set(drop_awake).union(_read_drop_from_outlier_csv(args.drop_from_outlier_csv, "awake")))
        drop_anes  = sorted(set(drop_anes ).union(_read_drop_from_outlier_csv(args.drop_from_outlier_csv, "anesthesia")))

    if args.mode in ("both", "awake"):
        run_condition(
            "awake",
            windows_mode=windows_mode,
            cv_folds=int(args.cv_folds),
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            cv_chunk_dur=args.cv_chunk_dur,
            cv_sl_draws=int(args.cv_sl_draws),
            cv_sl_tune=int(args.cv_sl_tune),
            cv_sl_chains=int(args.cv_sl_chains),
            save_payload=bool(args.save_payload),
            plot_only=bool(args.plot_only),
            force_recompute=bool(args.force_recompute),
            drop_orig=drop_awake,
            drop_awake=drop_awake,
            drop_anes=drop_anes,
        )
    if args.mode in ("both", "anesthesia"):
        run_condition(
            "anesthesia",
            windows_mode=windows_mode,
            cv_folds=int(args.cv_folds),
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            cv_chunk_dur=args.cv_chunk_dur,
            cv_sl_draws=int(args.cv_sl_draws),
            cv_sl_tune=int(args.cv_sl_tune),
            cv_sl_chains=int(args.cv_sl_chains),
            save_payload=bool(args.save_payload),
            plot_only=bool(args.plot_only),
            force_recompute=bool(args.force_recompute),
            drop_orig=drop_anes,
            drop_awake=drop_awake,
            drop_anes=drop_anes,
        )

    if args.mode == "both" and (not args.plot_only) and bool(args.save_payload):
        print("[INFO] Re-rendering Figure 4 from cached payloads so awake/anesthesia share identical metric y-limits.")
        run_condition(
            "awake",
            windows_mode=windows_mode,
            cv_folds=int(args.cv_folds),
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            cv_chunk_dur=args.cv_chunk_dur,
            cv_sl_draws=int(args.cv_sl_draws),
            cv_sl_tune=int(args.cv_sl_tune),
            cv_sl_chains=int(args.cv_sl_chains),
            save_payload=False,
            plot_only=True,
            force_recompute=False,
            drop_orig=drop_awake,
        )
        run_condition(
            "anesthesia",
            windows_mode=windows_mode,
            cv_folds=int(args.cv_folds),
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            cv_chunk_dur=args.cv_chunk_dur,
            cv_sl_draws=int(args.cv_sl_draws),
            cv_sl_tune=int(args.cv_sl_tune),
            cv_sl_chains=int(args.cv_sl_chains),
            save_payload=False,
            plot_only=True,
            force_recompute=False,
            drop_orig=drop_anes,
        )


if __name__ == "__main__":
    main()
