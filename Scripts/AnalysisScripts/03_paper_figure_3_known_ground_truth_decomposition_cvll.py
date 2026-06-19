#!/usr/bin/env python3
"""
Generate Figure 3 known-ground-truth decomposition benchmarks.

The script simulates additive and multiplicative spectral ground truths, fits
specparam and SL_specdecomp candidate models, computes cross-validated
Gamma-observation log likelihoods, and writes the manuscript figure plus cached
metrics.
"""

from __future__ import annotations
from pathlib import Path
import os, argparse, re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List, Callable

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from scipy.special import gammaln

from SL_GPsim import spectrum
from spectral_connectivity import Multitaper, Connectivity


PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# Estimators
from specparam import SpectralModel
from SL_specdecomp import Decompose


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
    "lines.linewidth": 1.6,
})
sns.set_style("white")

COL_GT = "k"          # combined / ground truth
COL_BB = "#ff7f0e"    # orange broadband
COL_RH = "red"        # rhythmic
COL_MT = "0.4"        # multitaper grey

CACHE_VERSION = "v8_median_A_height_bsigma5_seedonce_continuous_functional_peak"

# consistent estimator naming & palette

METHOD_KEYS = ["specparam", "SL_specdecomp_additive", "SL_specdecomp_multiplicative"]

METHOD_LABELS = {
    "specparam": "specparam",
    "SL_specdecomp_additive": "SL_specdecomp (Additive)",
    "SL_specdecomp_multiplicative": "SL_specdecomp (Multiplicative)",
}

PALETTE = dict(
    zip(
        [METHOD_LABELS[m] for m in METHOD_KEYS],
        sns.color_palette("deep", n_colors=len(METHOD_KEYS)),
    )
)

# display / plotting order
PLOT_METHOD_KEYS = ["specparam", "SL_specdecomp_multiplicative", "SL_specdecomp_additive"]
PLOT_ORDER_DISPLAY = [METHOD_LABELS[m] for m in PLOT_METHOD_KEYS]


# ---------------------- Config (rows 1-3: match 03_paper_figure_3_known_ground_truth_decomposition_cvll.py) ----------------------
FS               = 1000.0
NW               = 2
K_TAPERS         = 3
WIN_DUR          = 30.0
#ANALYSIS_FRANGE = (1.0, 200.0)
ANALYSIS_FRANGE  = (1.0, 200.0)
SLOPE_BAND       = (40.0, 60.0)
HG_BAND          = (80.0, 180.0)
RHY_BAND         = (1.0, 20.0)

DEFAULT_OUT_DIR = os.path.expanduser(
    "CHANGE_THIS_ROOT_TO_PATH/Bloniasz_Stephen_Estimation/Output/Results/FiguresIntermediate/Figure_3_FinalContinuousFunctionalPeak"
)
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

# ---------------------- CV Config (row 4 only) ----------------------
CV_FOLDS     = 5
CV_CHUNK_DUR = WIN_DUR / CV_FOLDS  # 6 s
CV_NW        = 1
CV_K_TAPERS  = 1

# specified sensitivity settings.
# - Additive SL_specdecomp rhythm-height prior A_lin_0 is anchored to the
#   median (50th percentile) rhythm-band power instead of the default 99th percentile.
# - Aperiodic intercept b prior uses sigma=5 rather than the additive default sigma=2.
A_HEIGHT_ANCHOR_Q = 50.0
B_PRIOR_SIGMA     = 5.0

# Separate deterministic streams per simulation regime. The script creates one
# top-level RNG per regime, then advances that stream across windows/fits instead
# of using seed0 + k reseeding.
MODE_SEED_OFFSETS = {
    "additive": 0,
    "multiplicative": 1_000_000,
}
FIT_SEED_OFFSET = 123_457

# Specparam config
SP_KW = dict(
    peak_width_limits=[1.0, 30.0],
    max_n_peaks=1,
    min_peak_height=0.0,
    peak_threshold=2.0,
    aperiodic_mode="knee",
    verbose=False,
)

# SL_specdecomp config (cores=1 avoids nested multiprocessing issues)
SL_KW_BASE = dict(
    n_aperiodics=1,
    n_rhythms=1,
    rhythm_bands=[(RHY_BAND[0], RHY_BAND[1])],
    sample_kwargs=dict(
        draws=800,
        tune=800,
        chains=2,
        cores=1,
        target_accept=0.90,
        nuts_sampler="blackjax",
        nuts_sampler_kwargs={"chain_method": "vectorized"},
    ),
    plot=False,
)

# For CV refits (row 4): separate sampler budget (still overrideable via CLI)
SL_KW_CV_BASE = dict(SL_KW_BASE)
SL_KW_CV_BASE["sample_kwargs"] = dict(
    draws=300,
    tune=300,
    chains=2,
    cores=1,
    target_accept=0.90,
    nuts_sampler="blackjax",
    nuts_sampler_kwargs={"chain_method": "vectorized"},
)


# ---------------------- Utilities ----------------------

from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar, brentq

def _continuous_peak_from_curve(
    freqs: np.ndarray,
    power_lin: np.ndarray,
    band: Tuple[float, float],
) -> Tuple[float, float]:
    lo, hi = map(float, band)
    x = np.asarray(freqs, float).ravel()
    y = np.asarray(power_lin, float).ravel()

    m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if m.sum() < 3:
        return np.nan, np.nan

    x = x[m]
    y = np.clip(y[m], 1e-30, np.inf)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    if x_u.size < 3:
        i = int(np.nanargmax(y_u))
        return float(x_u[i]), float(y_u[i])

    spline = PchipInterpolator(x_u, y_u, extrapolate=False)

    def obj(f):
        return -float(spline(f))

    res = minimize_scalar(
        obj,
        bounds=(float(x_u[0]), float(x_u[-1])),
        method="bounded",
        options={"xatol": 1e-10},
    )
    f_star = float(res.x)
    y_star = float(spline(f_star))
    return f_star, y_star

def _independent_grid(freqs, twin, NW):
    f = np.asarray(freqs, float)
    df = float(np.median(np.diff(f))) if f.size > 1 else 1.0
    delta_f_indep = 2.0 * NW / twin
    step_bins = max(1, int(round(delta_f_indep / max(df, 1e-12))))
    return f[::step_bins], step_bins


def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freqs >= lo) & (freqs <= hi)


def _robust_band_scale(
    freqs: np.ndarray,
    y_lin: np.ndarray,
    band: Tuple[float, float],
    q: float,
) -> float:
    """Local copy of the package's band-scale logic, with q configurable."""
    lo, hi = map(float, band)
    f = np.asarray(freqs, float).ravel()
    y = np.asarray(y_lin, float).ravel()
    m = np.isfinite(f) & np.isfinite(y) & (y > 0) & (f >= lo) & (f <= hi)
    if m.sum() < 5:
        m = np.isfinite(y) & (y > 0)
    if m.sum() == 0:
        return 1e-12
    return float(np.percentile(y[m], float(q)))


def _slsd_prior_override_kwargs(
    freqs_fit: np.ndarray,
    power_lin: np.ndarray,
    *,
    mode: str,
    rhythm_band: Tuple[float, float] = RHY_BAND,
    a_height_anchor_q: float = A_HEIGHT_ANCHOR_Q,
    b_prior_sigma: float = B_PRIOR_SIGMA,
) -> Dict[str, Dict[str, Any]]:
    """
    Build per-fit SL_specdecomp prior overrides without editing SL_specdecomp.

    The uploaded API forwards aperiodic_param_specs and rhythm_param_specs into
    the PyMC model. We use that pathway to replace the additive model's default
    A_lin_0 anchor q=99 with q=50 and to replace the b prior scale with sigma=5.
    """
    y = np.asarray(power_lin, float).ravel()
    y_pos = y[np.isfinite(y) & (y > 0)]
    mu_b = float(np.log10(np.median(y_pos))) if y_pos.size else 0.0
    band_scale = _robust_band_scale(freqs_fit, power_lin, rhythm_band, q=a_height_anchor_q)

    def _b_factory(name, mu_b=mu_b, sigma=float(b_prior_sigma)):
        return pm.Normal(name, mu=mu_b, sigma=sigma)

    def _A_lin_factory(name, band_scale=band_scale):
        return pm.LogNormal(name, mu=np.log(max(float(band_scale), 1e-12)), sigma=1.25)

    # Additive model names the single aperiodic b parameter b_0; multiplicative
    # model names it b when n_aperiodics == 1. Include both keys so the override
    # is robust to either builder.
    aperiodic_param_specs = {
        "b_0": {"factory": _b_factory},
        "b": {"factory": _b_factory},
    }

    # The q=99 anchor applies to the additive linear height parameter A_lin_0.
    # Multiplicative uses a_log_0 with a HalfNormal scale, so there is no 99th
    # percentile A_lin anchor to replace in that model.
    rhythm_param_specs: Dict[str, Any] = {}
    if str(mode) == "additive":
        rhythm_param_specs["A_lin_0"] = {"factory": _A_lin_factory}

    return {
        "aperiodic_param_specs": aperiodic_param_specs,
        "rhythm_param_specs": rhythm_param_specs,
    }


def _merge_prior_specs(default_specs: Dict[str, Any], user_specs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """User specs win only if explicitly provided; otherwise use specified defaults."""
    merged = dict(default_specs or {})
    if user_specs:
        merged.update(user_specs)
    return merged


def _next_pm_seed(fit_seed_rng: Optional[np.random.Generator]) -> Optional[int]:
    """Advance one seeded stream for PyMC fits instead of reusing random_seed=42."""
    if fit_seed_rng is None:
        return None
    return int(fit_seed_rng.integers(1, np.iinfo(np.int32).max))


def _slope_loglog(
    freqs: np.ndarray, power_lin: np.ndarray, band: Tuple[float, float]
) -> float:
    m = _band_mask(freqs, band)
    xf = np.log10(freqs[m])
    yf = np.log10(np.clip(power_lin[m], 1e-20, np.inf))
    if xf.size < 2:
        return np.nan
    A = np.vstack([xf, np.ones_like(xf)]).T
    chi, _b = np.linalg.lstsq(A, yf, rcond=None)[0]
    return float(chi)


def _continuous_argmax_legacy(
    freqs: np.ndarray,
    y_lin: np.ndarray,
    band: Tuple[float, float],
    fine_step_hz: float = 0.001,
) -> Tuple[float, float]:
    """Continuous argmax inside `band` using dense interpolation + quadratic vertex."""
    lo, hi = band
    freqs = np.asarray(freqs, float)
    y_lin = np.asarray(y_lin, float)
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return np.nan, np.nan
    n_steps = max(2, int(round((hi - lo) / max(fine_step_hz, 1e-6))))
    f_fine = np.linspace(lo, hi, n_steps + 1)
    y_fine = np.interp(f_fine, freqs[m], y_lin[m])
    i0 = int(np.argmax(y_fine))
    i_lo = max(0, i0 - 2)
    i_hi = min(len(f_fine) - 1, i0 + 2)
    ff = f_fine[i_lo : i_hi + 1]
    yy = y_fine[i_lo : i_hi + 1]
    if ff.size < 3:
        return float(f_fine[i0]), float(y_fine[i0])
    a, b, c = np.polyfit(ff, yy, deg=2)
    f_star = -b / (2.0 * a) if a != 0 else f_fine[i0]
    f_star = float(np.clip(f_star, ff.min(), ff.max()))
    y_star = float(np.interp(f_star, ff, yy))
    return f_star, y_star


def _clip_ylim(ax, ymin=1e-6):
    lo, hi = ax.get_ylim()
    lo = max(ymin, lo)
    if hi < lo * 10:
        hi = lo * 10
    ax.set_ylim(lo, hi)

def _nearest_grid_value(freqs: np.ndarray, x: float) -> float:
    f = np.asarray(freqs, float).ravel()
    f = f[np.isfinite(f)]
    if f.size == 0 or not np.isfinite(x):
        return np.nan
    return float(f[np.argmin(np.abs(f - float(x)))])


def _root_debug_record(freqs: np.ndarray, root_x: float) -> Dict[str, float]:
    nearest = _nearest_grid_value(freqs, root_x)
    return {
        "nearest_grid_freq": float(nearest) if np.isfinite(nearest) else np.nan,
        "root_minus_nearest_hz": float(root_x - nearest) if np.isfinite(nearest) and np.isfinite(root_x) else np.nan,
    }


def _eval_1d_func(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray | float) -> np.ndarray:
    """Evaluate a spectrum/derivative callable and always return a flat float array."""
    xx = np.asarray(x, float)
    yy = np.asarray(func(xx), float)
    if yy.shape == ():
        yy = np.full_like(xx, float(yy), dtype=float)
    return yy.ravel()


def _continuous_zero_slope_peak_from_functions(
    power_func: Callable[[np.ndarray], np.ndarray],
    deriv_func: Callable[[np.ndarray], np.ndarray],
    band: Tuple[float, float],
    *,
    n_bracket: int = 12000,
    label: str = "",
) -> Tuple[float, float, Dict[str, float]]:
    """
    One peak-definition used everywhere in this script.

    The target is the full-spectrum local maximum inside `band`: the frequency f*
    where dS(f)/df = 0 with negative curvature, and the height S(f*) from the
    same continuous function. The bracket grid is used only to locate sign-change
    intervals for Brent root solving; the returned f* is not snapped to any grid.
    """
    lo, hi = map(float, band)
    eps = 1e-30

    def p_scalar(z: float) -> float:
        return float(np.clip(_eval_1d_func(power_func, np.array([z]))[0], eps, np.inf))

    def d_scalar(z: float) -> float:
        return float(_eval_1d_func(deriv_func, np.array([z]))[0])

    grid = np.linspace(lo, hi, int(n_bracket))
    dvals = _eval_1d_func(deriv_func, grid)

    candidates: List[Tuple[float, float, float]] = []
    for i in range(len(grid) - 1):
        x0, x1 = float(grid[i]), float(grid[i + 1])
        g0, g1 = float(dvals[i]), float(dvals[i + 1])
        if not (np.isfinite(g0) and np.isfinite(g1)):
            continue

        root = None
        if g0 == 0.0:
            root = x0
        elif g0 * g1 < 0.0:
            try:
                root = float(brentq(d_scalar, x0, x1, xtol=1e-13, rtol=1e-12, maxiter=200))
            except Exception:
                root = None

        if root is None or not (lo <= root <= hi):
            continue

        h = max(1e-5, 1e-5 * max(1.0, abs(root)))
        h = min(h, 0.25 * max(hi - lo, 1e-9))
        zm = max(lo, root - h)
        zp = min(hi, root + h)
        if zp == zm:
            continue
        second = (d_scalar(zp) - d_scalar(zm)) / (zp - zm)
        if np.isfinite(second) and second < 0.0:
            candidates.append((float(root), p_scalar(root), float(second)))

    if candidates:
        f_star, p_star, second = max(candidates, key=lambda t: t[1])
        return float(f_star), float(p_star), {
            "peak_solver": "brentq_dSdf_zero",
            "peak_label": label,
            "n_peak_candidates": int(len(candidates)),
            "peak_second_derivative": float(second),
        }

    res = minimize_scalar(lambda z: -p_scalar(float(z)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
    f_star = float(res.x)
    p_star = p_scalar(f_star)
    return f_star, p_star, {
        "peak_solver": "bounded_continuous_argmax_fallback",
        "peak_label": label,
        "n_peak_candidates": 0,
        "peak_second_derivative": np.nan,
    }


def _continuous_peak_from_sampled_curve_functions(
    freqs: np.ndarray,
    power_lin: np.ndarray,
    band: Tuple[float, float],
    *,
    label: str = "sampled_curve",
) -> Tuple[float, float, Dict[str, float]]:
    """Continuous PCHIP function + same derivative for sampled spectra, used for GT."""
    lo, hi = map(float, band)
    x = np.asarray(freqs, float).ravel()
    y = np.asarray(power_lin, float).ravel()
    m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    x = x[m]
    y = np.clip(y[m], 1e-30, np.inf)
    if x.size < 4:
        return np.nan, np.nan, {"peak_solver": "failed_too_few_points", "peak_label": label}

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    if x_u.size < 4:
        return np.nan, np.nan, {"peak_solver": "failed_too_few_unique_points", "peak_label": label}

    spline = PchipInterpolator(x_u, y_u, extrapolate=False)
    d1 = spline.derivative()

    def pfun(f):
        return np.clip(np.asarray(spline(f), float), 1e-30, np.inf)

    def dfun(f):
        return np.asarray(d1(f), float)

    return _continuous_zero_slope_peak_from_functions(pfun, dfun, band, n_bracket=max(2000, x_u.size * 4), label=label)


def _apply_peak_metrics(
    rec: Dict[str, Any],
    *,
    true_peak: Tuple[float, float, Dict[str, float]],
    est_peak: Tuple[float, float, Dict[str, float]],
    ref_freqs_for_debug: np.ndarray,
) -> Dict[str, Any]:
    """Overwrite only the two rhythm-peak panels from the continuous full-spectrum peak."""
    eps = 1e-20
    gt_f, gt_p, gt_dbg = true_peak
    est_f, est_p, est_dbg = est_peak

    rec["rh_cf_true"] = float(gt_f)
    rec["rh_cf_est"] = float(est_f)
    rec["rh_height_true_log10"] = float(np.log10(max(float(gt_p), eps)))
    rec["rh_height_est_log10"] = float(np.log10(max(float(est_p), eps)))

    rec.update({f"gt_{k}": v for k, v in _root_debug_record(ref_freqs_for_debug, gt_f).items()})
    rec.update({f"est_{k}": v for k, v in _root_debug_record(ref_freqs_for_debug, est_f).items()})
    for k, v in gt_dbg.items():
        rec[f"gt_{k}"] = v
    for k, v in est_dbg.items():
        rec[f"est_{k}"] = v
    return rec


# ---------------------- True params & simulation (match 03_paper_figure_3_known_ground_truth_decomposition_cvll.py) ----------------------
@dataclass
class TrueParams:
    exponent: float = 2.0
    offset: float = 0.5  # log10 offset
    knee: float = 60.0
    peak: Dict[str, float] = None  # dict(freq, amplitude (linear), sigma)


def _mt_power_one_window(
    ts_1d: np.ndarray,
    fs: float,
    duration: float,
    nw: float,
    k_tapers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multitaper power estimate for a single time window."""
    x = np.asarray(ts_1d, float).ravel()
    x = x[:, np.newaxis, np.newaxis]  # (n_time, n_trials=1, n_signals=1)

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


def simulate_once(
    mode: str,
    fs: float = FS,
    duration: float = WIN_DUR,
    params: Optional[TrueParams] = None,
    rng_seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    if params is None:
        params = TrueParams(peak=dict(freq=10.0, amplitude=1.0, sigma=1.0))
    elif params.peak is None:
        params.peak = dict(freq=10.0, amplitude=1.0, sigma=1.0)

    random_state = rng if rng is not None else rng_seed

    spectrum_kwargs = dict(
        sampling_rate=fs,
        duration=duration,
        aperiodic_exponent=params.exponent,
        aperiodic_offset=params.offset,
        knee=params.knee,
        peaks=[params.peak],
        average_firing_rate=0.0,
        random_state=random_state,
        direct_estimate=False,
        plot=False,
        mode=mode,
    )
    try:
        res = spectrum(**spectrum_kwargs)
    except TypeError:
        if rng is None:
            raise
        spectrum_kwargs["random_state"] = int(rng.integers(1, np.iinfo(np.int32).max))
        res = spectrum(**spectrum_kwargs)
    td = res.time_domain
    fd = res.frequency_domain

    # Multitaper on the whole 30 s window (rows 1-3; standard params)
    ts = np.asarray(td.combined_signal, float).ravel()
    f_emp, S_emp = _mt_power_one_window(ts, fs=fs, duration=duration, nw=NW, k_tapers=K_TAPERS)

    f_ref, step_bins = _independent_grid(f_emp, duration, NW)
    S_fit = S_emp[::step_bins]

    # Theoretical dense + components
    f_dense = np.asarray(fd.frequencies, dtype=float).ravel()

    def _maybe(nameA, nameB=None):
        v = getattr(fd, nameA, None)
        if v is None and nameB:
            v = getattr(fd, nameB, None)
        if callable(v):
            try:
                v = v()
            except TypeError:
                pass
        return np.asarray(v, float).ravel() if v is not None else None

    S_full = _maybe("combined_spectrum")
    if S_full is None:
        raise RuntimeError("frequency_domain.combined_spectrum unavailable")
    S_bb = _maybe("broadband_spectrum", "aperiodic_spectrum")
    S_rh = _maybe("rhythmic_spectrum", "peaks_spectrum")
    if S_bb is None and S_rh is None:
        S_bb = S_full.copy()
        S_rh = np.zeros_like(S_bb)
    elif S_bb is None:
        S_bb = np.clip(S_full - S_rh, 0.0, np.inf)
    elif S_rh is None:
        S_rh = np.clip(S_full - S_bb, 0.0, np.inf)

    # Project GT to the independent MT grid used everywhere (rows 1-3)
    GT_full_on_ref = np.interp(f_ref, f_dense, S_full)
    GT_bb_on_ref = np.interp(f_ref, f_dense, S_bb)
    GT_rh_on_ref = np.interp(f_ref, f_dense, S_rh)

    return dict(
        fs=fs,
        duration=duration,
        time=td.time,
        x_bb=getattr(td, "broadband_signal", None),
        x_rh=getattr(td, "rhythmic_signal", None),
        x_comb=td.combined_signal,  # raw 30 s time series (needed for CV row)
        f_dense=f_dense,
        S_bb=S_bb,
        S_rh=S_rh,
        S_full=S_full,
        f_ref=f_ref,
        S_fit=S_fit,
        GT_full_on_ref=GT_full_on_ref,
        GT_bb_on_ref=GT_bb_on_ref,
        GT_rh_on_ref=GT_rh_on_ref,
    )


# ---------------------- Estimator helpers (match 03_paper_figure_3_known_ground_truth_decomposition_cvll.py) ----------------------


def _fit_specparam_model(
    freqs_fit,
    power_lin,
    freq_range,
):
    fm = SpectralModel(**SP_KW)
    fm.fit(
        np.asarray(freqs_fit, float),
        np.clip(power_lin, 1e-20, np.inf),
        freq_range=freq_range,
    )
    return fm


def _safe_get_params(fm: SpectralModel, names: List[str]) -> Optional[np.ndarray]:
    for name in names:
        try:
            arr = np.asarray(fm.get_params(name), float)
            if arr.size > 0:
                return arr
        except Exception:
            continue
    return None


def _extract_aperiodic_params(fm: SpectralModel) -> Dict[str, float]:
    ap = _safe_get_params(fm, ["aperiodic_params", "aperiodic", "aperiodic_param"])
    if ap is None:
        raise RuntimeError("Could not extract specparam aperiodic parameters.")

    ap = np.asarray(ap, float).ravel()
    mode = getattr(fm, "aperiodic_mode", SP_KW.get("aperiodic_mode", "fixed"))

    if ap.size >= 3:
        return {
            "mode": "knee",
            "offset": float(ap[0]),
            "knee": float(ap[1]),
            "exponent": float(ap[2]),
        }
    if ap.size == 2:
        return {
            "mode": "fixed",
            "offset": float(ap[0]),
            "knee": 0.0,
            "exponent": float(ap[1]),
        }

    raise RuntimeError(f"Unexpected aperiodic parameter shape: {ap.shape} (mode={mode})")


def _extract_peak_params_for_callable(fm: SpectralModel) -> np.ndarray:
    """
    Returns array with columns [cf, amp, sigma] for the fitted Gaussians in
    log10-power space. Preference order:
      1) gaussian_params -> already [cf, amp, sigma]
      2) peak_params/peak -> [cf, amp, bw], convert sigma = bw / 2
    """
    gp = _safe_get_params(fm, ["gaussian_params", "gaussian"])
    if gp is not None:
        gp = np.asarray(gp, float)
        if gp.ndim == 1 and gp.size >= 3:
            gp = gp.reshape(1, -1)
        if gp.ndim == 2 and gp.shape[1] >= 3:
            return np.asarray(gp[:, :3], float)

    pp = _safe_get_params(fm, ["peak_params", "peak", "peaks"])
    if pp is not None:
        pp = np.asarray(pp, float)
        if pp.ndim == 1 and pp.size >= 3:
            pp = pp.reshape(1, -1)
        if pp.ndim == 2 and pp.shape[1] >= 3:
            cf = pp[:, 0]
            amp = pp[:, 1]
            sigma = pp[:, 2] / 2.0
            return np.column_stack([cf, amp, sigma]).astype(float)

    return np.empty((0, 3), float)


def _specparam_log10_callable(fm: SpectralModel):
    ap = _extract_aperiodic_params(fm)
    peaks = _extract_peak_params_for_callable(fm)

    def aperiodic_log10(f):
        f = np.asarray(f, float)
        f = np.clip(f, 1e-12, np.inf)
        offset = ap["offset"]
        knee = max(float(ap["knee"]), 0.0)
        exponent = ap["exponent"]
        return offset - np.log10(knee + np.power(f, exponent))

    def periodic_log10(f):
        f = np.asarray(f, float)
        out = np.zeros_like(f, dtype=float)
        for cf, amp, sigma in peaks:
            sigma = max(float(sigma), 1e-12)
            out = out + float(amp) * np.exp(-0.5 * ((f - float(cf)) / sigma) ** 2)
        return out

    def full_log10(f):
        return aperiodic_log10(f) + periodic_log10(f)

    def full_linear(f):
        return np.power(10.0, full_log10(f))

    return full_log10, full_linear, peaks, ap


def _specparam_dlog10_df(f, peaks: np.ndarray, ap: Dict[str, float]) -> np.ndarray:
    f = np.asarray(f, float)
    f = np.clip(f, 1e-12, np.inf)

    exponent = float(ap["exponent"])
    knee = max(float(ap["knee"]), 0.0)

    d_ap = -(exponent * np.power(f, exponent - 1.0)) / (
        (knee + np.power(f, exponent)) * np.log(10.0)
    )

    d_pk = np.zeros_like(f, dtype=float)
    for cf, amp, sigma in peaks:
        sigma = max(float(sigma), 1e-12)
        g = float(amp) * np.exp(-0.5 * ((f - float(cf)) / sigma) ** 2)
        d_pk = d_pk + g * (-(f - float(cf)) / (sigma ** 2))

    return d_ap + d_pk


def _specparam_linear_and_derivative_callable(fm: SpectralModel):
    """Continuous specparam full-spectrum callable and dS/df from fitted parameters."""
    _full_log10, full_linear, peaks, ap = _specparam_log10_callable(fm)

    def power_func(f):
        return np.clip(np.asarray(full_linear(f), float), 1e-30, np.inf)

    def deriv_func(f):
        f = np.asarray(f, float)
        S = power_func(f)
        dlog10 = _specparam_dlog10_df(f, peaks, ap)
        return np.log(10.0) * S * dlog10

    return power_func, deriv_func, peaks, ap


def _specparam_zero_slope_peak_from_params(
    fm: SpectralModel,
    band: Tuple[float, float],
    n_bracket: int = 12000,
) -> Tuple[float, float, Dict[str, float]]:
    """Continuous specparam peak: solve dS/df = 0 on the fitted functional form."""
    power_func, deriv_func, peaks, _ap = _specparam_linear_and_derivative_callable(fm)
    if peaks.size == 0:
        return np.nan, np.nan, {"peak_cf_param": np.nan, "peak_sigma_param": np.nan, "peak_solver": "no_peak"}

    f_star, p_star, dbg = _continuous_zero_slope_peak_from_functions(
        power_func, deriv_func, band, n_bracket=n_bracket, label="specparam_functional"
    )

    lo, hi = map(float, band)
    peak_mask = (peaks[:, 0] >= lo - 5.0) & (peaks[:, 0] <= hi + 5.0)
    peaks_band = peaks[peak_mask] if np.any(peak_mask) else peaks.copy()
    idx_dom = int(np.nanargmax(peaks_band[:, 1])) if peaks_band.size else 0
    dbg.update({
        "peak_cf_param": float(peaks_band[idx_dom, 0]) if peaks_band.size else np.nan,
        "peak_sigma_param": float(peaks_band[idx_dom, 2]) if peaks_band.size else np.nan,
    })
    return float(f_star), float(p_star), dbg

def _specparam_cf_from_model(
    fm: SpectralModel,
    true_cf: Optional[float] = None,
) -> float:
    try:
        peaks = np.asarray(fm.get_params("peak"), float)
    except Exception:
        try:
            peaks = np.asarray(fm.get_params("peaks"), float)
        except Exception:
            return np.nan

    if peaks.size == 0:
        return np.nan

    if peaks.ndim == 1 and peaks.size >= 3:
        cfs = peaks[0:1]
        heights = peaks[1:2]
    else:
        cfs = peaks[:, 0]
        heights = peaks[:, 1] if peaks.shape[1] > 1 else np.ones_like(peaks[:, 0])

    cfs = np.asarray(cfs, float).ravel()
    heights = np.asarray(heights, float).ravel()
    if cfs.size == 0:
        return np.nan

    if true_cf is not None and np.isfinite(true_cf):
        idx = int(np.argmin(np.abs(cfs - true_cf)))
    else:
        idx = int(np.nanargmax(heights))

    return float(cfs[idx])


def _specparam_full_on_grid(
    freqs_fit,
    power_lin,
    freq_range,
    true_cf: Optional[float] = None,
):
    fm = _fit_specparam_model(freqs_fit, power_lin, freq_range)

    cf_est = _specparam_cf_from_model(fm, true_cf=true_cf)

    model_obj = getattr(getattr(fm, "results", None), "model", None)
    if model_obj is None:
        raise RuntimeError("specparam results.model not available")

    cand_x = []
    for name in ("freqs", "freqs_model", "_freqs", "_spectrum_freqs"):
        x = getattr(model_obj, name, None)
        if x is not None:
            cand_x.append(np.asarray(x, float).ravel())
    x_fm = np.asarray(getattr(fm, "freqs", freqs_fit), float).ravel()
    cand_x.append(x_fm)

    def _to_lin(a):
        a = np.asarray(a, float).ravel()
        return 10.0 ** a if np.nanmin(a) < 0 and np.nanmax(a) < 15 else a

    def _interp_safe(x, xp, fp):
        xp = np.asarray(xp, float).ravel()
        fp = np.asarray(fp, float).ravel()
        m = np.isfinite(xp) & np.isfinite(fp)
        xp, fp = xp[m], fp[m]
        if xp.size == 0:
            return np.full_like(x, np.nan, dtype=float)
        order = np.argsort(xp)
        xp, fp = xp[order], fp[order]
        xp_u, idx = np.unique(xp, return_index=True)
        fp_u = fp[idx]
        return np.interp(x, xp_u, fp_u)

    def _pick_x_for_fp(fp, default_range):
        fp = np.asarray(fp).ravel()
        for cx in cand_x:
            if cx.size == fp.size:
                return cx
        lo, hi = map(float, default_range)
        n = max(2, fp.size)
        return np.linspace(lo, hi, n)

    full_attr = getattr(model_obj, "modeled_spectrum", None)
    if callable(full_attr):
        try:
            full_native = np.asarray(full_attr(space="linear"), float).ravel()
        except TypeError:
            full_native = _to_lin(full_attr())
    else:
        full_native = _to_lin(full_attr)
    xp_full = _pick_x_for_fp(full_native, freq_range)

    get_comp = getattr(model_obj, "get_component", None)
    try:
        ap_native = np.asarray(get_comp("aperiodic", space="linear"), float).ravel()
    except TypeError:
        ap_native = _to_lin(get_comp("aperiodic"))
    xp_ap = _pick_x_for_fp(ap_native, freq_range)

    full_lin = _interp_safe(freqs_fit, xp_full, full_native)
    ap_lin = _interp_safe(freqs_fit, xp_ap, ap_native)
    return fm, full_lin, ap_lin, cf_est, xp_full, np.clip(full_native, 1e-30, np.inf)


def _slsd_peak_params(model) -> np.ndarray:
    peaks: List[List[float]] = []
    idata = getattr(model, "idata", None)
    ds = getattr(idata, "posterior", None) if idata is not None else None
    if ds is None:
        return np.empty((0, 3), float)

    if "center" in ds and "sigma" in ds:
        Aname = "A_lin" if "A_lin" in ds else ("A" if "A" in ds else None)
        if Aname is not None:
            c = ds["center"].mean(dim=[d for d in ds["center"].dims if d in ("chain", "draw")]).values.ravel()
            s = ds["sigma"].mean(dim=[d for d in ds["sigma"].dims if d in ("chain", "draw")]).values.ravel()
            a = ds[Aname].mean(dim=[d for d in ds[Aname].dims if d in ("chain", "draw")]).values.ravel()
            for ci, si, ai in zip(c, s, a):
                peaks.append([float(ci), float(ai), float(2.3548 * si)])

    if not peaks:
        varnames = set(ds.data_vars)
        idxs = sorted(
            {
                int(m.group(1))
                for v in varnames
                for m in [re.search(r"center[_\[](\d+)", v)]
                if m
            }
        )
        for k in idxs:
            c = s = a = None
            for v in (f"center_{k}", f"center[{k}]"):
                if v in ds:
                    c = float(ds[v].mean().item())
                    break
            for v in (f"sigma_{k}", f"sigma[{k}]"):
                if v in ds:
                    s = float(ds[v].mean().item())
                    break

            for v in (
                f"A_lin_{k}", f"A_lin[{k}]",
                f"A_{k}",     f"A[{k}]",
                f"a_log_{k}", f"a_log[{k}]",
            ):
                if v in ds:
                    val = float(ds[v].mean().item())
                    a = 10.0 ** val if v.startswith("a_log") else val
                    break

            if c is not None and s is not None:
                if a is None:
                    a = 1.0
                peaks.append([c, a, 2.3548 * s])

    return np.asarray(peaks, float) if peaks else np.empty((0, 3), float)


def _slsd_center_from_model(
    model,
    true_cf: Optional[float] = None,
) -> float:
    peaks = _slsd_peak_params(model)
    if peaks.size == 0:
        return np.nan

    cfs = np.asarray(peaks[:, 0], float).ravel()
    amps = np.asarray(peaks[:, 1], float).ravel()
    if cfs.size == 0:
        return np.nan

    if true_cf is not None and np.isfinite(true_cf):
        idx = int(np.argmin(np.abs(cfs - true_cf)))
    else:
        idx = int(np.nanargmax(amps))

    return float(cfs[idx])




def _posterior_mean_from_candidates(ds, names: List[str]) -> Optional[float]:
    for name in names:
        if name in ds:
            try:
                return float(ds[name].mean().item())
            except Exception:
                pass
    return None


def _extract_slsd_aperiodic_params(model) -> Dict[str, float]:
    idata = getattr(model, "idata", None)
    ds = getattr(idata, "posterior", None) if idata is not None else None
    if ds is None:
        raise RuntimeError("SL_specdecomp posterior unavailable")

    offset = _posterior_mean_from_candidates(ds, [
        "offset", "offset_0", "offset[0]", "aperiodic_offset", "b", "b_0", "b[0]"
    ])
    knee = _posterior_mean_from_candidates(ds, [
        "knee", "knee_0", "knee[0]", "aperiodic_knee", "kappa", "kappa_0", "kappa[0]"
    ])
    exponent = _posterior_mean_from_candidates(ds, [
        "chi", "chi_0", "chi[0]", "exponent", "exponent_0", "exponent[0]",
        "aperiodic_exponent", "slope", "slope_0", "slope[0]"
    ])

    if offset is None:
        offset = 0.0
    if knee is None:
        knee = 0.0
    if exponent is None:
        exponent = 2.0

    return {"offset": float(offset), "knee": max(float(knee), 0.0), "exponent": float(exponent)}


def _extract_slsd_peak_params_for_callable(model, mode: str) -> np.ndarray:
    peaks = []
    idata = getattr(model, "idata", None)
    ds = getattr(idata, "posterior", None) if idata is not None else None
    if ds is None:
        return np.empty((0, 3), float)

    # Vectorized/common naming.
    if "center" in ds and "sigma" in ds:
        c = ds["center"].mean(dim=[d for d in ds["center"].dims if d in ("chain", "draw")]).values.ravel()
        s = ds["sigma"].mean(dim=[d for d in ds["sigma"].dims if d in ("chain", "draw")]).values.ravel()

        amp_name = None
        if str(mode) == "additive":
            for cand in ("A_lin", "A", "a_log"):
                if cand in ds:
                    amp_name = cand
                    break
        else:
            for cand in ("A", "a_log", "A_lin"):
                if cand in ds:
                    amp_name = cand
                    break

        if amp_name is not None:
            a = ds[amp_name].mean(dim=[d for d in ds[amp_name].dims if d in ("chain", "draw")]).values.ravel()
            for ci, si, ai in zip(c, s, a):
                peaks.append([float(ci), float(ai), float(si)])

    if not peaks:
        varnames = set(ds.data_vars)
        idxs = sorted({
            int(m.group(1))
            for v in varnames
            for m in [re.search(r"center[_\[]([0-9]+)", v)]
            if m
        })
        for k in idxs:
            c = _posterior_mean_from_candidates(ds, [f"center_{k}", f"center[{k}]"])
            s = _posterior_mean_from_candidates(ds, [f"sigma_{k}", f"sigma[{k}]"])
            if c is None or s is None:
                continue
            if str(mode) == "additive":
                a = _posterior_mean_from_candidates(ds, [f"A_lin_{k}", f"A_lin[{k}]", f"A_{k}", f"A[{k}]", f"a_log_{k}", f"a_log[{k}]"])
            else:
                a = _posterior_mean_from_candidates(ds, [f"A_{k}", f"A[{k}]", f"a_log_{k}", f"a_log[{k}]", f"A_lin_{k}", f"A_lin[{k}]"])
            if a is None:
                a = 0.0
            peaks.append([float(c), float(a), float(s)])

    return np.asarray(peaks, float) if peaks else np.empty((0, 3), float)


def _mirrored_gaussian_unit(f, center: float, sigma: float) -> np.ndarray:
    f = np.asarray(f, float)
    sigma = max(float(sigma), 1e-12)
    c = float(center)
    return np.exp(-0.5 * ((f - c) / sigma) ** 2) + np.exp(-0.5 * ((f + c) / sigma) ** 2)


def _posterior_samples_1d(ds, names: List[str]) -> Optional[np.ndarray]:
    """Return posterior samples for the first scalar/component found in `names`."""
    for name in names:
        if name not in ds:
            continue
        arr = np.asarray(ds[name].values, float)
        if arr.ndim >= 2:
            n = arr.shape[0] * arr.shape[1]
            arr = arr.reshape(n, *arr.shape[2:])
            if arr.ndim > 1:
                arr = arr[:, 0]
        arr = np.asarray(arr, float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return arr
    return None


def _broadcast_or_trim_samples(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(a, float).ravel() for a in arrays)
    lengths = [a.size for a in arrays if a.size > 1]
    n = min(lengths) if lengths else 1
    out = []
    for a in arrays:
        if a.size == n:
            out.append(a)
        elif a.size == 1:
            out.append(np.full(n, float(a[0])))
        else:
            out.append(a[:n])
    return tuple(out)


def _slsd_plugin_mean_callable(model, mode: str):
    """Fallback: continuous curve at posterior-mean parameters."""
    ap = _extract_slsd_aperiodic_params(model)
    peaks = _extract_slsd_peak_params_for_callable(model, mode=mode)

    def aperiodic_linear(f):
        f = np.asarray(f, float)
        f = np.clip(f, 1e-12, np.inf)
        return np.power(10.0, ap["offset"]) / (ap["knee"] + np.power(f, ap["exponent"]))

    def full_linear(f):
        f = np.asarray(f, float)
        base = aperiodic_linear(f)
        if peaks.size == 0:
            return base
        if str(mode) == "additive":
            out = base.copy()
            for cf, amp, sigma in peaks:
                if np.isfinite(amp):
                    out = out + float(amp) * _mirrored_gaussian_unit(f, cf, sigma)
            return out
        log10_bump = np.zeros_like(f, dtype=float)
        for cf, amp, sigma in peaks:
            if np.isfinite(amp):
                log10_bump = log10_bump + float(amp) * _mirrored_gaussian_unit(f, cf, sigma)
        return base * np.power(10.0, log10_bump)

    def deriv_func(f):
        f = np.asarray(f, float)
        h = np.maximum(1e-5, 1e-5 * np.maximum(1.0, np.abs(f)))
        fp = f + h
        fm = np.maximum(1e-12, f - h)
        return (full_linear(fp) - full_linear(fm)) / (fp - fm)

    return full_linear, deriv_func, peaks, {"slsd_callable_type": "plugin_posterior_mean_params"}


def _slsd_posterior_mean_functions(model, mode: str):
    """
    Continuous analogue of `estimated_spectrum` for SL_specdecomp.

    `api.py` stores `estimated_spectrum` as posterior mean E[mu(f)] on the original
    discrete input frequency grid. Here we evaluate the same PyMC functional form
    at arbitrary continuous frequencies for every posterior draw, then average:
    E[mu(f*)]. This avoids using `model.estimated_spectrum` for peak height or CF.
    """
    idata = getattr(model, "idata", None)
    ds = getattr(idata, "posterior", None) if idata is not None else None
    if ds is None:
        return _slsd_plugin_mean_callable(model, mode)

    b = _posterior_samples_1d(ds, ["b_0", "b", "offset_0", "offset", "aperiodic_offset"])
    chi = _posterior_samples_1d(ds, ["chi_0", "chi", "exponent_0", "exponent", "aperiodic_exponent"])
    knee = _posterior_samples_1d(ds, ["knee_0", "knee", "aperiodic_knee", "kappa_0", "kappa"])
    center = _posterior_samples_1d(ds, ["center_0", "center"])
    sigma = _posterior_samples_1d(ds, ["sigma_0", "sigma"])

    if str(mode) == "additive":
        amp = _posterior_samples_1d(ds, ["A_lin_0", "A_lin", "A_0", "A"])
    else:
        amp = _posterior_samples_1d(ds, ["a_log_0", "a_log", "A_0", "A"])

    needed = (b, chi, knee, center, sigma, amp)
    if any(a is None for a in needed):
        return _slsd_plugin_mean_callable(model, mode)

    b, chi, knee, center, sigma, amp = _broadcast_or_trim_samples(*needed)  # type: ignore[arg-type]
    sigma = np.clip(sigma, 1e-12, np.inf)
    knee = np.clip(knee, 0.0, np.inf)
    scale = np.power(10.0, b)
    n_samples = int(min(a.size for a in (b, chi, knee, center, sigma, amp)))

    def _eval(f, derivative: bool = False):
        f = np.asarray(f, float).ravel()
        out = np.empty_like(f, dtype=float)
        chunk = 2048
        for start in range(0, f.size, chunk):
            ff = np.clip(f[start:start + chunk], 1e-12, np.inf)[:, None]
            denom = knee[None, :] + np.power(ff, chi[None, :])
            base = scale[None, :] / np.clip(denom, 1e-30, np.inf)
            dbase = -scale[None, :] * chi[None, :] * np.power(ff, chi[None, :] - 1.0) / np.clip(denom ** 2, 1e-30, np.inf)

            z_pos = (ff - center[None, :]) / sigma[None, :]
            z_neg = (ff + center[None, :]) / sigma[None, :]
            g_pos = np.exp(-0.5 * z_pos ** 2)
            g_neg = np.exp(-0.5 * z_neg ** 2)
            G = g_pos + g_neg
            dG = g_pos * (-(ff - center[None, :]) / (sigma[None, :] ** 2)) + g_neg * (-(ff + center[None, :]) / (sigma[None, :] ** 2))

            if str(mode) == "additive":
                vals = dbase + amp[None, :] * dG if derivative else base + amp[None, :] * G
            else:
                L = amp[None, :] * G
                R = np.power(10.0, L)
                vals = R * dbase + base * R * np.log(10.0) * amp[None, :] * dG if derivative else base * R

            out[start:start + chunk] = np.nanmean(vals, axis=1)
        return np.clip(out, 1e-30, np.inf) if not derivative else out

    def power_func(f):
        return _eval(f, derivative=False)

    def deriv_func(f):
        return _eval(f, derivative=True)

    peaks = np.column_stack([
        [float(np.nanmean(center))],
        [float(np.nanmean(amp))],
        [float(np.nanmean(sigma))],
    ])
    debug = {"slsd_callable_type": "posterior_draw_mean_function", "n_posterior_samples": n_samples}
    return power_func, deriv_func, peaks, debug


def _slsd_zero_slope_peak_from_params(model, mode: str, band: Tuple[float, float], n_bracket: int = 12000) -> Tuple[float, float, Dict[str, float]]:
    power_func, deriv_func, peaks, dbg0 = _slsd_posterior_mean_functions(model, mode=mode)
    if peaks.size == 0:
        return np.nan, np.nan, {"peak_cf_param": np.nan, "peak_sigma_param": np.nan, "peak_solver": "no_peak"}

    f_star, p_star, dbg = _continuous_zero_slope_peak_from_functions(
        power_func,
        deriv_func,
        band,
        n_bracket=n_bracket,
        label=f"SL_specdecomp_{mode}_posterior_mean_function",
    )

    lo, hi = map(float, band)
    peak_mask = (peaks[:, 0] >= lo - 5.0) & (peaks[:, 0] <= hi + 5.0)
    peaks_band = peaks[peak_mask] if np.any(peak_mask) else peaks.copy()
    idx_dom = int(np.nanargmax(peaks_band[:, 1])) if peaks_band.size else 0
    dbg.update(dbg0)
    dbg.update({
        "peak_cf_param": float(peaks_band[idx_dom, 0]) if peaks_band.size else np.nan,
        "peak_sigma_param": float(peaks_band[idx_dom, 2]) if peaks_band.size else np.nan,
    })
    return float(f_star), float(p_star), dbg

def _slsd_components_on_grid(
    freqs_fit,
    power_lin,
    fs,
    mode="additive",
    fit_seed_rng: Optional[np.random.Generator] = None,
    **kw,
):
    kw_eff = dict(SL_KW_BASE)
    kw_eff.update(kw)
    kw_eff["mode"] = mode

    requested_prior_specs = _slsd_prior_override_kwargs(
        freqs_fit,
        power_lin,
        mode=mode,
        rhythm_band=RHY_BAND,
        a_height_anchor_q=A_HEIGHT_ANCHOR_Q,
        b_prior_sigma=B_PRIOR_SIGMA,
    )
    kw_eff["aperiodic_param_specs"] = _merge_prior_specs(
        requested_prior_specs["aperiodic_param_specs"],
        kw_eff.get("aperiodic_param_specs"),
    )
    kw_eff["rhythm_param_specs"] = _merge_prior_specs(
        requested_prior_specs["rhythm_param_specs"],
        kw_eff.get("rhythm_param_specs"),
    )

    # api.py defaults to random_seed=42 when omitted, which would reset the
    # sampler for every fit. Draw PyMC seeds from one stream per simulation
    # regime instead.
    sample_kwargs = dict(kw_eff.get("sample_kwargs", {}) or {})
    if "random_seed" not in sample_kwargs:
        pm_seed = _next_pm_seed(fit_seed_rng)
        if pm_seed is not None:
            sample_kwargs["random_seed"] = pm_seed
    kw_eff["sample_kwargs"] = sample_kwargs

    model = Decompose(
        freqs_fit, np.clip(power_lin, 1e-20, np.inf), fs=fs, **kw_eff
    )
    total = np.asarray(model.estimated_spectrum, float).ravel()
    bb = (
        np.asarray(model.broadband, float).ravel()
        if getattr(model, "broadband", None) is not None
        else None
    )
    rh = (
        np.asarray(model.rhythms, float).ravel()
        if getattr(model, "rhythms", None) is not None
        else None
    )
    if bb is None and rh is None:
        bb = total.copy()
        rh = np.zeros_like(bb)
    elif bb is None:
        bb = np.clip(total - rh, 0.0, np.inf)
    elif rh is None:
        rh = np.clip(total - bb, 0.0, np.inf)

    def _floor(a):
        return np.clip(np.asarray(a, float).ravel(), 1e-20, np.inf)

    return model, _floor(total), _floor(bb), _floor(rh)

# ---------------------- Metrics (GT + estimates from full spectra only) ----------------------
def _stationary_points_of_curve(
    freqs: np.ndarray,
    power_lin: np.ndarray,
    band: Tuple[float, float],
) -> List[Dict[str, float]]:
    """
    Find stationary points of a 1D spectrum by solving dS/df = 0 on the
    continuous PCHIP interpolant itself. This does NOT create a secondary
    frequency grid. It uses same roots of the piecewise-polynomial derivative.
    """
    lo, hi = map(float, band)
    x = np.asarray(freqs, float).ravel()
    y = np.asarray(power_lin, float).ravel()

    m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    x = x[m]
    y = np.clip(y[m], 1e-30, np.inf)
    if x.size < 4:
        return []

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    if x_u.size < 4:
        return []

    spline = PchipInterpolator(x_u, y_u, extrapolate=False)
    d1 = spline.derivative()
    d2 = d1.derivative()

    try:
        roots = np.asarray(d1.roots(extrapolate=False), float).ravel()
    except Exception:
        roots = np.array([], dtype=float)

    roots = roots[np.isfinite(roots)]
    roots = roots[(roots >= lo) & (roots <= hi)]
    if roots.size:
        roots = np.unique(np.round(roots, 12))

    out: List[Dict[str, float]] = []
    for r in roots:
        d1r = float(d1(r))
        d2r = float(d2(r))
        yr = float(spline(r))
        if d2r < 0.0:
            kind = "max"
        elif d2r > 0.0:
            kind = "min"
        else:
            kind = "flat"
        out.append(dict(f_root=float(r), power=yr, d1=d1r, d2=d2r, kind=kind))

    out.sort(key=lambda z: z["f_root"])
    return out

from scipy.interpolate import PchipInterpolator

def _continuous_zero_slope_peak_from_curve(
    freqs: np.ndarray,
    power_lin: np.ndarray,
    band: Tuple[float, float],
    n_dense: int = 200_001,
) -> Tuple[float, float]:
    """
    For the frequency violin only:
    treat the fitted/full spectrum as a continuous curve on an ultra-dense x-axis,
    then find the x-location where dS/df crosses 0 from + to -.

    This is intentionally NOT tied to the original sampled frequency grid.
    """
    lo, hi = map(float, band)

    x = np.asarray(freqs, float).ravel()
    y = np.asarray(power_lin, float).ravel()

    m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    x = x[m]
    y = np.clip(y[m], 1e-30, np.inf)

    if x.size < 4:
        return np.nan, np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    if x_u.size < 4:
        return np.nan, np.nan

    spline = PchipInterpolator(x_u, y_u, extrapolate=False)

    # ultra-dense continuous surrogate x-axis
    xd = np.linspace(lo, hi, int(n_dense))
    yd = np.asarray(spline(xd), float)
    yd = np.clip(yd, 1e-30, np.inf)

    dyd = np.gradient(yd, xd)

    cands = []
    for i in range(1, len(xd)):
        g0 = dyd[i - 1]
        g1 = dyd[i]
        if not (np.isfinite(g0) and np.isfinite(g1)):
            continue

        # positive -> nonpositive means a local max bracket
        if g0 > 0.0 and g1 <= 0.0:
            x0, x1 = xd[i - 1], xd[i]
            # linear interpolation of derivative zero crossing
            if g1 != g0:
                xr = x0 - g0 * (x1 - x0) / (g1 - g0)
            else:
                xr = x0
            yr = float(spline(xr))
            cands.append((float(xr), float(yr)))

    if cands:
        return max(cands, key=lambda t: t[1])

    # fallback: quadratic refinement around dense argmax
    i = int(np.nanargmax(yd))
    if 0 < i < len(xd) - 1:
        xx = xd[i - 1:i + 2]
        yy = yd[i - 1:i + 2]
        a, b, c = np.polyfit(xx, yy, 2)
        if a != 0:
            xr = -b / (2.0 * a)
            xr = float(np.clip(xr, xx.min(), xx.max()))
            return xr, float(spline(xr))

    return float(xd[i]), float(yd[i])

def _continuous_zero_slope_peak_from_curve_legacy(
    freqs: np.ndarray,
    power_lin: np.ndarray,
    band: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Return the continuous frequency where the slope of the full spectrum is zero
    and curvature is negative, using the same derivative roots of the PCHIP
    interpolant. No dense resampling grid is introduced here.
    """
    pts = _stationary_points_of_curve(freqs, power_lin, band=band)
    maxima = [p for p in pts if p["kind"] == "max"]
    if not maxima:
        return _continuous_peak_from_curve(freqs, power_lin, band)
    best = max(maxima, key=lambda p: p["power"])
    return float(best["f_root"]), float(best["power"])

def _metrics_from_spectra(
    freqs_true: np.ndarray,
    total_lin_true: np.ndarray,
    freqs_est: np.ndarray,
    total_lin_est: np.ndarray,
) -> Dict[str, float]:
    eps = 1e-20

    freqs_true = np.asarray(freqs_true, float)
    total_lin_true = np.asarray(total_lin_true, float)
    freqs_est = np.asarray(freqs_est, float)
    total_lin_est = np.asarray(total_lin_est, float)

    cf_true, pk_true = _continuous_zero_slope_peak_from_curve_legacy(
        freqs_true, total_lin_true, RHY_BAND
    )
    cf_est, pk_est = _continuous_zero_slope_peak_from_curve_legacy(
        freqs_est, total_lin_est, RHY_BAND
    )

    rh_height_true_log10 = float(np.log10(max(pk_true, eps)))
    rh_height_est_log10 = float(np.log10(max(pk_est, eps)))

    m_hg_true = _band_mask(freqs_true, HG_BAND)
    m_hg_est = _band_mask(freqs_est, HG_BAND)

    hg_true_log10 = float(np.log10(max(np.mean(total_lin_true[m_hg_true]), eps)))
    hg_est_log10 = float(np.log10(max(np.mean(total_lin_est[m_hg_est]), eps)))

    slope_true = _slope_loglog(freqs_true, np.clip(total_lin_true, eps, np.inf), SLOPE_BAND)
    slope_est = _slope_loglog(freqs_est, np.clip(total_lin_est, eps, np.inf), SLOPE_BAND)

    return dict(
        rh_cf_true=float(cf_true),
        rh_cf_est=float(cf_est),
        rh_height_true_log10=rh_height_true_log10,
        rh_height_est_log10=rh_height_est_log10,
        hg_true_log10=hg_true_log10,
        hg_est_log10=hg_est_log10,
        slope_true=slope_true,
        slope_est=slope_est,
    )

def _metrics_from_spectra_legacy(
    freqs_true: np.ndarray,
    total_lin_true: np.ndarray,
    freqs_est: np.ndarray,
    total_lin_est: np.ndarray,
) -> Dict[str, float]:
    eps = 1e-20
    freqs_true = np.asarray(freqs_true, float)
    total_lin_true = np.asarray(total_lin_true, float)
    freqs_est = np.asarray(freqs_est, float)
    total_lin_est = np.asarray(total_lin_est, float)

    cf_true, pk_true = _continuous_zero_slope_peak_from_curve(
        freqs_true, total_lin_true, RHY_BAND
    )
    cf_est, pk_est = _continuous_zero_slope_peak_from_curve(
        freqs_est, total_lin_est, RHY_BAND
    )

    rh_height_true_log10 = float(np.log10(max(pk_true, eps)))
    rh_height_est_log10 = float(np.log10(max(pk_est, eps)))

    m_hg_true = _band_mask(freqs_true, HG_BAND)
    m_hg_est = _band_mask(freqs_est, HG_BAND)
    hg_true_log10 = float(
        np.log10(
            max(
                np.mean(total_lin_true[m_hg_true]) if np.any(m_hg_true) else np.nan,
                eps,
            )
        )
    )
    hg_est_log10 = float(
        np.log10(
            max(
                np.mean(total_lin_est[m_hg_est]) if np.any(m_hg_est) else np.nan,
                eps,
            )
        )
    )

    slope_true = _slope_loglog(
        freqs_true, np.clip(total_lin_true, eps, np.inf), SLOPE_BAND
    )
    slope_est = _slope_loglog(
        freqs_est, np.clip(total_lin_est, eps, np.inf), SLOPE_BAND
    )

    rec = dict(
        rh_cf_true=float(cf_true),
        rh_cf_est=float(cf_est),
        rh_height_true_log10=rh_height_true_log10,
        rh_height_est_log10=rh_height_est_log10,
        hg_true_log10=hg_true_log10,
        hg_est_log10=hg_est_log10,
        slope_true=slope_true,
        slope_est=slope_est,
    )
    rec.update({f"gt_{k}": v for k, v in _root_debug_record(freqs_true, cf_true).items()})
    rec.update({f"est_{k}": v for k, v in _root_debug_record(freqs_est, cf_est).items()})
    return rec


# ---------------------- Likelihood (Gamma multitaper) ----------------------
def _gamma_loglik_multitaper(y_lin: np.ndarray, mu_lin: np.ndarray, k_tapers: int) -> float:
    """
    Sum log-likelihood under: Y_i | mu_i ~ Gamma(shape=K, scale=mu_i/K).
    """
    y = np.clip(np.asarray(y_lin, float).ravel(), 1e-30, np.inf)
    mu = np.clip(np.asarray(mu_lin, float).ravel(), 1e-30, np.inf)

    m = np.isfinite(y) & np.isfinite(mu)
    if m.sum() == 0:
        return np.nan

    y = y[m]
    mu = mu[m]
    K = float(int(k_tapers))

    theta = mu / K  # scale
    ll = (K - 1.0) * np.log(y) - (y / theta) - K * np.log(theta) - gammaln(K)
    return float(np.sum(ll))


# ---------------------- CVLL computation (row 4 only) ----------------------
def _compute_cvll_for_window(
    ts_1d: np.ndarray,
    fs: float,
    true_cf_param: Optional[float],
    cv_folds: int = CV_FOLDS,
    chunk_dur: float = CV_CHUNK_DUR,
    cv_nw: float = CV_NW,
    cv_k_tapers: int = CV_K_TAPERS,
    sl_kw_cv: Optional[dict] = None,
    fit_seed_rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Compute CVLL per model for a single 30s window time series, following the user's procedure.

    Returns dict: {method_key: cvll_sum} with one value per model.
    """
    x = np.asarray(ts_1d, float).ravel()
    n_total = int(round(float(fs) * WIN_DUR))
    if x.size < n_total:
        # pad with NaNs -> will propagate to failures; safer to just fail
        return {m: np.nan for m in METHOD_KEYS}
    if x.size > n_total:
        x = x[:n_total]

    n_chunk = int(round(float(fs) * float(chunk_dur)))
    n_expect = int(cv_folds) * n_chunk
    if n_expect > x.size:
        return {m: np.nan for m in METHOD_KEYS}
    if n_expect < x.size:
        # drop trailing samples to keep folds non-overlapping & equal length
        x = x[:n_expect]

    chunks = [x[i * n_chunk:(i + 1) * n_chunk] for i in range(int(cv_folds))]

    # Precompute MT spectra per chunk
    # Choose a common independent grid from the first chunk
    f0, S0 = _mt_power_one_window(chunks[0], fs=fs, duration=chunk_dur, nw=cv_nw, k_tapers=cv_k_tapers)
    f_ref_cv, _step_bins = _independent_grid(f0, chunk_dur, cv_nw)

    # Interpolate each chunk spectrum onto f_ref_cv for consistent averaging
    S_chunks = []
    for c in chunks:
        f_emp, S_emp = _mt_power_one_window(c, fs=fs, duration=chunk_dur, nw=cv_nw, k_tapers=cv_k_tapers)
        S_fit = np.interp(f_ref_cv, f_emp, S_emp)
        S_chunks.append(np.clip(S_fit, 1e-20, np.inf))
    S_chunks = np.asarray(S_chunks, float)  # (folds, n_freq)

    # Restrict CV fits + likelihood to ANALYSIS_FRANGE (keeps scale consistent with the figure)
    m_fr = (f_ref_cv >= ANALYSIS_FRANGE[0]) & (f_ref_cv <= ANALYSIS_FRANGE[1])
    f_cv = f_ref_cv[m_fr]
    if f_cv.size < 10:
        return {m: np.nan for m in METHOD_KEYS}

    sl_kw_cv = dict(SL_KW_CV_BASE) if sl_kw_cv is None else dict(sl_kw_cv)

    cvll = {m: 0.0 for m in METHOD_KEYS}

    for i_test in range(int(cv_folds)):
        test = np.asarray(S_chunks[i_test, m_fr], float)
        train_idx = [j for j in range(int(cv_folds)) if j != i_test]
        train = np.mean(S_chunks[train_idx, :], axis=0)[m_fr]

        fr_k = (
            max(ANALYSIS_FRANGE[0], float(f_cv[0])),
            min(ANALYSIS_FRANGE[1], float(f_cv[-1])),
        )

        # ---- specparam ----
        if np.isfinite(cvll["specparam"]):
            try:
                _fm_sp, mu_sp, _ap, _cf, _sp_x_native, _sp_y_native = _specparam_full_on_grid(
                    f_cv, train, fr_k, true_cf=true_cf_param
                )
                ll = _gamma_loglik_multitaper(test, mu_sp, cv_k_tapers)
                cvll["specparam"] = cvll["specparam"] + ll if np.isfinite(ll) else np.nan
            except Exception:
                cvll["specparam"] = np.nan

        # ---- SL_specdecomp additive ----
        if np.isfinite(cvll["SL_specdecomp_additive"]):
            try:
                _madd, mu_add, _bb, _rh = _slsd_components_on_grid(
                    f_cv, train, fs=fs, mode="additive", fit_seed_rng=fit_seed_rng, **sl_kw_cv
                )
                ll = _gamma_loglik_multitaper(test, mu_add, cv_k_tapers)
                cvll["SL_specdecomp_additive"] = cvll["SL_specdecomp_additive"] + ll if np.isfinite(ll) else np.nan
            except Exception:
                cvll["SL_specdecomp_additive"] = np.nan

        # ---- SL_specdecomp multiplicative ----
        if np.isfinite(cvll["SL_specdecomp_multiplicative"]):
            try:
                _mmul, mu_mul, _bbm, _rhm = _slsd_components_on_grid(
                    f_cv, train, fs=fs, mode="FOOOF_spectrum", fit_seed_rng=fit_seed_rng, **sl_kw_cv
                )
                ll = _gamma_loglik_multitaper(test, mu_mul, cv_k_tapers)
                cvll["SL_specdecomp_multiplicative"] = cvll["SL_specdecomp_multiplicative"] + ll if np.isfinite(ll) else np.nan
            except Exception:
                cvll["SL_specdecomp_multiplicative"] = np.nan

    return cvll


# ---------------------- Cache helpers ----------------------
def _save_payload(base, mode, arrays: dict, df_metrics: pd.DataFrame):
    base = os.path.join(base, f"Figure_3_FinalContinuousFunctionalPeak_{mode}_{CACHE_VERSION}")
    np.savez_compressed(
        base + ".plotdata.npz",
        **{k: np.asarray(v) for k, v in arrays.items()},
    )
    df_metrics.to_csv(base + ".metrics.csv", index=False)


def _maybe_load_payload(base, mode):
    base = os.path.join(base, f"Figure_3_FinalContinuousFunctionalPeak_{mode}_{CACHE_VERSION}")
    npz = base + ".plotdata.npz"
    csv = base + ".metrics.csv"
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return None, None
    arrays = np.load(npz, allow_pickle=True)
    df_metrics = pd.read_csv(csv)
    # Ensure this cache is actually CV-capable
    if "cvll_gamma_mt" not in df_metrics.columns:
        return None, None
    return arrays, df_metrics


# ---------------------- Figure builder ----------------------
def build_figure(
    sim_mode: str,  # "additive" | "multiplicative"
    out_dir: str,
    n_iter: int = 200,
    seed0: int = 10000,
    true_params: Optional[TrueParams] = None,
    recompute: bool = False,
    cv_folds: int = CV_FOLDS,
    cv_chunk_dur: float = CV_CHUNK_DUR,
    cv_nw: float = CV_NW,
    cv_k_tapers: int = CV_K_TAPERS,
    sl_cv_draws: int = 300,
    sl_cv_tune: int = 300,
    sl_cv_chains: int = 2,
) -> Tuple[str, str]:

    os.makedirs(out_dir, exist_ok=True)

    mode_seed = int(seed0) + int(MODE_SEED_OFFSETS.get(str(sim_mode), 0))
    sim_rng = np.random.default_rng(mode_seed)
    fit_seed_rng = np.random.default_rng(mode_seed + FIT_SEED_OFFSET)
    print(
        f"[rng] sim_mode={sim_mode}; one simulation RNG seed={mode_seed}; "
        f"one SL fit-seed stream seed={mode_seed + FIT_SEED_OFFSET}"
    )

    # Ground-truth rhythm center frequency from TrueParams (if available)
    true_cf_param: Optional[float] = None
    if (
        true_params is not None
        and isinstance(true_params.peak, dict)
        and "freq" in true_params.peak
    ):
        try:
            true_cf_param = float(true_params.peak["freq"])
        except Exception:
            true_cf_param = None

    cached, cached_df = _maybe_load_payload(out_dir, sim_mode)

    if cached is not None and cached_df is not None and not recompute:
        sim0 = {
            k: cached[k]
            for k in (
                "f_dense",
                "S_bb",
                "S_rh",
                "S_full",
                "f_ref",
                "S_fit",
                "GT_full_on_ref",
                "time",
                "x_bb",
                "x_rh",
                "x_comb",
            )
        }
        single_curves = {
            "specparam": cached["single_specparam"],
            "SL_specdecomp_additive": cached["single_SL_specdecomp_additive"],
            "SL_specdecomp_multiplicative": cached["single_SL_specdecomp_multiplicative"],
        }
        pct_bands = {
            "specparam": cached["pct_specparam"],
            "SL_specdecomp_additive": cached["pct_SL_specdecomp_additive"],
            "SL_specdecomp_multiplicative": cached["pct_SL_specdecomp_multiplicative"],
        }
        df_long = cached_df
    else:
        sim0 = simulate_once(sim_mode, params=true_params, rng=sim_rng)

        fr0 = (
            max(ANALYSIS_FRANGE[0], float(sim0["f_ref"][0])),
            min(ANALYSIS_FRANGE[1], float(sim0["f_ref"][-1])),
        )

        # single overlay fits (robust)
        def _nan_curve():
            return np.full_like(sim0["f_ref"], np.nan, dtype=float)

        try:
            _fm0, sp_full0, _ap0, _cf0, _sp_x0, _sp_y0 = _specparam_full_on_grid(
                sim0["f_ref"], sim0["S_fit"], fr0, true_cf=None
            )
        except Exception:
            sp_full0 = _nan_curve()

        try:
            _model_add0, add_full0, _add_bb0, _ = _slsd_components_on_grid(
                sim0["f_ref"], sim0["S_fit"], fs=sim0["fs"], mode="additive", fit_seed_rng=fit_seed_rng
            )
        except Exception:
            add_full0 = _nan_curve()

        try:
            _model_mul0, mul_full0, _mul_bb0, _ = _slsd_components_on_grid(
                sim0["f_ref"], sim0["S_fit"], fs=sim0["fs"], mode="FOOOF_spectrum", fit_seed_rng=fit_seed_rng
            )
        except Exception:
            mul_full0 = _nan_curve()

        single_curves = {
            "specparam": sp_full0,
            "SL_specdecomp_additive": add_full0,
            "SL_specdecomp_multiplicative": mul_full0,
        }

        # loop over trials
        f_ref = sim0["f_ref"]
        coll = {m: [] for m in METHOD_KEYS}
        records: List[Dict[str, Any]] = []

        sl_kw_cv = dict(SL_KW_CV_BASE)
        sl_kw_cv["sample_kwargs"] = dict(
            draws=int(sl_cv_draws),
            tune=int(sl_cv_tune),
            chains=int(sl_cv_chains),
            cores=1,
            target_accept=float(sl_kw_cv["sample_kwargs"].get("target_accept", 0.90)),
            nuts_sampler="blackjax",
            nuts_sampler_kwargs={"chain_method": "vectorized"},
        )

        for k in range(n_iter):
            sim = simulate_once(sim_mode, params=true_params, rng=sim_rng)

            # enforce common grid (rows 1-3)
            f_fit = sim["f_ref"]
            if not np.array_equal(f_fit, f_ref):
                def _to_ref(y):
                    return np.interp(f_ref, f_fit, y)
                S_fit = _to_ref(sim["S_fit"])
                true_total = _to_ref(sim["GT_full_on_ref"])
            else:
                S_fit = sim["S_fit"]
                true_total = sim["GT_full_on_ref"]

            fr_k = (
                max(ANALYSIS_FRANGE[0], float(f_ref[0])),
                min(ANALYSIS_FRANGE[1], float(f_ref[-1])),
            )

            # --- CVLL (row 4) computed from time series with separate MT params ---
            cvll_map = _compute_cvll_for_window(
                ts_1d=np.asarray(sim["x_comb"], float),
                fs=float(sim["fs"]),
                true_cf_param=true_cf_param,
                cv_folds=int(cv_folds),
                chunk_dur=float(cv_chunk_dur),
                cv_nw=float(cv_nw),
                cv_k_tapers=int(cv_k_tapers),
                sl_kw_cv=sl_kw_cv,
                fit_seed_rng=fit_seed_rng,
            )

            def _push(method_key: str, est_full: np.ndarray, rec: Dict[str, Any]):
                coll[method_key].append(est_full)
                rec["trial"] = int(k)
                rec["method"] = method_key
                rec["cvll_gamma_mt"] = float(cvll_map.get(method_key, np.nan))
                records.append(rec)

            # ---- specparam (rows 1-3) ----
            try:
                fm_sp, sp_full, _ap, _sp_cf, sp_x_native, sp_y_native = _specparam_full_on_grid(
                    f_ref, S_fit, fr_k, true_cf=true_cf_param
                )
                rec = _metrics_from_spectra(
                    sim["f_dense"], sim["S_full"],
                    np.asarray(sp_x_native, float), np.asarray(sp_y_native, float)
                )

                gt_peak = _continuous_peak_from_sampled_curve_functions(
                    sim["f_dense"], sim["S_full"], RHY_BAND, label="ground_truth_full_spectrum"
                )
                sp_peak = _specparam_zero_slope_peak_from_params(fm_sp, RHY_BAND)
                rec = _apply_peak_metrics(
                    rec, true_peak=gt_peak, est_peak=sp_peak, ref_freqs_for_debug=f_ref
                )

                _push("specparam", sp_full, rec)
            except Exception:
                sp_full = np.full_like(f_ref, np.nan, dtype=float)
                rec = {k_: np.nan for k_ in ("rh_cf_true","rh_cf_est","rh_height_true_log10","rh_height_est_log10","hg_true_log10","hg_est_log10","slope_true","slope_est")}
                _push("specparam", sp_full, rec)

                        # ---- SL_specdecomp additive (rows 1-3) ----
            try:
                model_add, add_full, _bb, _rh = _slsd_components_on_grid(
                    f_ref, S_fit, fs=sim["fs"], mode="additive", fit_seed_rng=fit_seed_rng
                )
                rec = _metrics_from_spectra(sim["f_dense"], sim["S_full"], f_ref, add_full)
                gt_peak = _continuous_peak_from_sampled_curve_functions(
                    sim["f_dense"], sim["S_full"], RHY_BAND, label="ground_truth_full_spectrum"
                )
                add_peak = _slsd_zero_slope_peak_from_params(
                    model_add, "additive", RHY_BAND
                )
                rec = _apply_peak_metrics(
                    rec, true_peak=gt_peak, est_peak=add_peak, ref_freqs_for_debug=f_ref
                )
                _push("SL_specdecomp_additive", add_full, rec)
            except Exception:
                add_full = np.full_like(f_ref, np.nan, dtype=float)
                rec = {k_: np.nan for k_ in ("rh_cf_true","rh_cf_est","rh_height_true_log10","rh_height_est_log10","hg_true_log10","hg_est_log10","slope_true","slope_est")}
                _push("SL_specdecomp_additive", add_full, rec)

            # ---- SL_specdecomp multiplicative (rows 1-3) ----
            try:
                model_mul, mul_full, _bbm, _rhm = _slsd_components_on_grid(
                    f_ref, S_fit, fs=sim["fs"], mode="FOOOF_spectrum", fit_seed_rng=fit_seed_rng
                )
                rec = _metrics_from_spectra(sim["f_dense"], sim["S_full"], f_ref, mul_full)
                gt_peak = _continuous_peak_from_sampled_curve_functions(
                    sim["f_dense"], sim["S_full"], RHY_BAND, label="ground_truth_full_spectrum"
                )
                mul_peak = _slsd_zero_slope_peak_from_params(
                    model_mul, "FOOOF_spectrum", RHY_BAND
                )
                rec = _apply_peak_metrics(
                    rec, true_peak=gt_peak, est_peak=mul_peak, ref_freqs_for_debug=f_ref
                )
                _push("SL_specdecomp_multiplicative", mul_full, rec)
            except Exception:
                mul_full = np.full_like(f_ref, np.nan, dtype=float)
                rec = {k_: np.nan for k_ in ("rh_cf_true","rh_cf_est","rh_height_true_log10","rh_height_est_log10","hg_true_log10","hg_est_log10","slope_true","slope_est")}
                _push("SL_specdecomp_multiplicative", mul_full, rec)

        def _pct(arr2d):
            a = np.asarray(arr2d, float)
            return np.nanpercentile(a, [2.5, 25, 50, 75, 97.5], axis=0)

        pct_bands = {m: _pct(coll[m]) for m in METHOD_KEYS}
        df_long = pd.DataFrame.from_records(records)

        arrays_to_save = dict(
            f_dense=sim0["f_dense"],
            S_bb=sim0["S_bb"],
            S_rh=sim0["S_rh"],
            S_full=sim0["S_full"],
            f_ref=sim0["f_ref"],
            S_fit=sim0["S_fit"],
            GT_full_on_ref=sim0["GT_full_on_ref"],
            time=np.asarray(sim0["time"]),
            x_bb=sim0["x_bb"] if sim0["x_bb"] is not None else np.array([]),
            x_rh=sim0["x_rh"] if sim0["x_rh"] is not None else np.array([]),
            x_comb=np.asarray(sim0["x_comb"], float),
            single_specparam=single_curves["specparam"],
            single_SL_specdecomp_additive=single_curves["SL_specdecomp_additive"],
            single_SL_specdecomp_multiplicative=single_curves["SL_specdecomp_multiplicative"],
            pct_specparam=pct_bands["specparam"],
            pct_SL_specdecomp_additive=pct_bands["SL_specdecomp_additive"],
            pct_SL_specdecomp_multiplicative=pct_bands["SL_specdecomp_multiplicative"],
            trials_specparam=np.asarray(coll["specparam"], float),
            trials_SL_specdecomp_additive=np.asarray(coll["SL_specdecomp_additive"], float),
            trials_SL_specdecomp_multiplicative=np.asarray(coll["SL_specdecomp_multiplicative"], float),
        )
        _save_payload(out_dir, sim_mode, arrays_to_save, df_long)
        cf_csv = os.path.join(out_dir, f"Figure_3_FinalContinuousFunctionalPeak_{sim_mode}_continuous_cf_values.csv")
        keep_cols = [c for c in [
            "trial", "method",
            "rh_cf_true", "rh_cf_est",
            "rh_height_true_log10", "rh_height_est_log10",
            "gt_nearest_grid_freq", "gt_root_minus_nearest_hz",
            "est_nearest_grid_freq", "est_root_minus_nearest_hz",
            "gt_peak_solver", "est_peak_solver",
            "gt_n_peak_candidates", "est_n_peak_candidates",
            "est_slsd_callable_type", "est_n_posterior_samples",
            "est_peak_cf_param", "est_peak_sigma_param",
        ] if c in df_long.columns]
        if keep_cols:
            df_long[keep_cols].to_csv(cf_csv, index=False)


    # ---------------- Figure layout: (same 3 rows) + (NEW 4th row: CVLL) ----------------
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(
        4, 4,
        height_ratios=[1.0, 1.2, 1.1, 0.95],
        wspace=0.38,
        hspace=0.62,
    )

    ROW_EXPL = 0
    ROW_PSD  = 1
    ROW_VIOL = 2
    ROW_CV   = 3

    # Helper for Row-2 percentile panels (everything on f_ref)
    def _pct_panel(ax, pct5x, label_key):
        p2, p25, p50, p75, p97 = pct5x

        m = (sim0["f_dense"] >= ANALYSIS_FRANGE[0]) & (sim0["f_dense"] <= ANALYSIS_FRANGE[1])
        ax.loglog(
            sim0["f_dense"][m],
            np.clip(sim0["S_full"][m], 1e-6, np.inf),
            color=COL_GT,
            lw=1.8,
            alpha=0.95,
            label="Ground truth",
            zorder=10,
        )

        f_ref_loc = sim0["f_ref"]
        col = PALETTE[METHOD_LABELS[label_key]]
        ax.fill_between(
            f_ref_loc,
            np.clip(p2, 1e-6, np.inf),
            np.clip(p97, 1e-6, np.inf),
            alpha=0.32,
            color=col,
            zorder=2,
        )
        ax.set_xlim(ANALYSIS_FRANGE)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        _clip_ylim(ax, ymin=1e-6)
        sns.despine(ax=ax, top=True, right=True)
        ax.minorticks_off()

    # ---------- Row 2: overlay + three percentile panels ----------
    ax_overlay = fig.add_subplot(gs[ROW_PSD, 0])

    m_dense = (sim0["f_dense"] >= ANALYSIS_FRANGE[0]) & (sim0["f_dense"] <= ANALYSIS_FRANGE[1])

    # Additive overlay: keep black below colored model lines.
    # Multiplicative overlay: keep black on top so it remains visible.
    overlay_gt_zorder = 20

    ax_overlay.loglog(
        sim0["f_dense"][m_dense],
        np.clip(sim0["S_full"][m_dense], 1e-6, np.inf),
        color=COL_GT,
        lw=1.8,
        label="Ground truth PSD",
        zorder=overlay_gt_zorder,
    )

    f_ref = sim0["f_ref"]
    ax_overlay.loglog(
        f_ref,
        np.clip(sim0["S_fit"], 1e-6, np.inf),
        color=COL_MT,
        lw=1.0,
        alpha=0.55,
        label="Multitaper (example)",
    )

    ax_overlay.loglog(
        f_ref,
        np.clip(single_curves["specparam"], 1e-6, np.inf),
        lw=0.8,
        label=METHOD_LABELS["specparam"],
        color=PALETTE[METHOD_LABELS["specparam"]],
        zorder=23,
    )
    ax_overlay.loglog(
        f_ref,
        np.clip(single_curves["SL_specdecomp_multiplicative"], 1e-6, np.inf),
        lw=0.8,
        label=METHOD_LABELS["SL_specdecomp_multiplicative"],
        color=PALETTE[METHOD_LABELS["SL_specdecomp_multiplicative"]],
        zorder=24,
    )
    ax_overlay.loglog(
        f_ref,
        np.clip(single_curves["SL_specdecomp_additive"], 1e-6, np.inf),
        lw=0.8,
        label=METHOD_LABELS["SL_specdecomp_additive"],
        color=PALETTE[METHOD_LABELS["SL_specdecomp_additive"]],
        zorder=21,
    )

    ax_overlay.set_xlim(ANALYSIS_FRANGE)
    ax_overlay.set_xlabel("Frequency (Hz)")
    ax_overlay.set_ylabel("Power")
    _clip_ylim(ax_overlay, ymin=1e-6)
    sns.despine(ax=ax_overlay, top=True, right=True)
    ax_overlay.minorticks_off()
    ax_overlay.legend(fontsize=9, ncol=1, frameon=True)
    ax_overlay.set_title("Overlay: single estimates", pad=6)

    ax_p_spec = fig.add_subplot(gs[ROW_PSD, 1])
    _pct_panel(ax_p_spec, pct_bands["specparam"], "specparam")


    ax_p_mul = fig.add_subplot(gs[ROW_PSD, 2])
    _pct_panel(ax_p_mul, pct_bands["SL_specdecomp_multiplicative"], "SL_specdecomp_multiplicative")


    ax_p_add = fig.add_subplot(gs[ROW_PSD, 3])
    _pct_panel(ax_p_add, pct_bands["SL_specdecomp_additive"], "SL_specdecomp_additive")

    # ---------- Row 1 (top): explanatory row ----------
    ax_th = fig.add_subplot(gs[ROW_EXPL, 2])
    m11 = (sim0["f_dense"] >= ANALYSIS_FRANGE[0]) & (sim0["f_dense"] <= ANALYSIS_FRANGE[1])
    f_d = sim0["f_dense"][m11]
    S_full = np.asarray(sim0["S_full"][m11], float)
    S_full[S_full <= 0] = np.nan

    if sim_mode == "multiplicative":
        ax_th.loglog(f_d, S_full, "-", color=COL_GT, label="Ground truth PSD")
    else:
        S_bb = np.asarray(sim0["S_bb"][m11], float)
        S_rh = np.asarray(sim0["S_rh"][m11], float)
        S_bb[S_bb <= 0] = np.nan
        S_rh[S_rh <= 0] = np.nan
        ax_th.loglog(f_d, S_bb, "--", color=COL_BB, label="Broadband")
        ax_th.loglog(f_d, S_rh, "--", color=COL_RH, label="Rhythmic")
        ax_th.loglog(f_d, S_full, "-", color=COL_GT, label="Ground truth PSD")

    ax_th.set(
        xlabel="Frequency (Hz)",
        ylabel="Power",
        xscale="log",
        yscale="log",
        xlim=ANALYSIS_FRANGE,
    )
    if sim_mode == "additive":
        ax_th.set_ylim(1e-6, 1e1)
    else:
        _clip_ylim(ax_th, ymin=1e-6)

    sns.despine(ax=ax_th, top=True, right=True)
    ax_th.minorticks_off()
    ax_th.legend(loc="best")
    ax_th.set_title("Theory PSD components", pad=6)

    # Col4: Time-domain snippets
    if sim_mode == "multiplicative":
        ax_ts = fig.add_subplot(gs[ROW_EXPL, 3])
        t = np.asarray(sim0["time"]).ravel()
        t0, t1 = t.min(), min(t.min() + 3.0, t.max())
        m_ts = (t >= t0) & (t <= t1)
        ax_ts.plot(
            t[m_ts],
            np.asarray(sim0["x_comb"])[m_ts],
            color=COL_GT,
        )
        ax_ts.set_title("Signal from specparam functional form", pad=6)
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel("Amplitude")
        sns.despine(ax=ax_ts, top=True, right=True)
        ax_ts.minorticks_off()
    else:
        gs_ts = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[ROW_EXPL, 3], hspace=0.25
        )
        ax_ts_bb = fig.add_subplot(gs_ts[0, 0])
        ax_ts_rh = fig.add_subplot(gs_ts[1, 0])

        t = np.asarray(sim0["time"]).ravel()
        t0, t1 = t.min(), min(t.min() + 3.0, t.max())
        m_ts = (t >= t0) & (t <= t1)

        if sim0["x_bb"] is not None and len(np.asarray(sim0["x_bb"]).ravel()) > 0:
            ax_ts_bb.plot(t[m_ts], np.asarray(sim0["x_bb"])[m_ts], color=COL_BB)
        ax_ts_bb.set_title("Broadband", pad=4)
        ax_ts_bb.tick_params(axis="x", labelbottom=False)
        sns.despine(ax=ax_ts_bb, top=True, right=True)
        ax_ts_bb.minorticks_off()

        if sim0["x_rh"] is not None and len(np.asarray(sim0["x_rh"]).ravel()) > 0:
            ax_ts_rh.plot(t[m_ts], np.asarray(sim0["x_rh"])[m_ts], color=COL_RH)
        ax_ts_rh.set_title("Rhythmic", pad=4)
        ax_ts_rh.set_xlabel("Time (s)")
        sns.despine(ax=ax_ts_rh, top=True, right=True)
        ax_ts_rh.minorticks_off()

    # ---------- Row 3: violins ----------
    order_display = PLOT_ORDER_DISPLAY

    metric_specs = [
        ("rh_height_est_log10", "Rhythm height @ exact continuous dS/df = 0 root", "rh_height_true_log10", "log10(power)"),
        ("rh_cf_est", "Rhythm peak frequency (exact continuous dS/df = 0 root; full precision)", "rh_cf_true", "Hz"),
        ("hg_est_log10", "High-γ mean power (80-180 Hz)", "hg_true_log10", "log10(power)"),
        ("slope_est", "Broadband slope (40-60 Hz, log-log)", "slope_true", "Slope"),
    ]

    df_plot = df_long.copy()
    df_plot["method_display"] = df_plot["method"].map(METHOD_LABELS)

    m_dense2 = (sim0["f_dense"] >= ANALYSIS_FRANGE[0]) & (sim0["f_dense"] <= ANALYSIS_FRANGE[1])
    gt_const = _metrics_from_spectra(
        sim0["f_dense"][m_dense2],
        sim0["S_full"][m_dense2],
        sim0["f_dense"][m_dense2],
        sim0["S_full"][m_dense2],
    )
    gt_peak_const = _continuous_peak_from_sampled_curve_functions(
        sim0["f_dense"][m_dense2],
        sim0["S_full"][m_dense2],
        RHY_BAND,
        label="ground_truth_full_spectrum_for_black_line",
    )
    gt_const["rh_cf_true"] = float(gt_peak_const[0])
    gt_const["rh_height_true_log10"] = float(np.log10(max(float(gt_peak_const[1]), 1e-20)))

    for j, (est_key, title_txt, true_key, ylab) in enumerate(metric_specs):
        ax = fig.add_subplot(gs[ROW_VIOL, j])
        sub = df_plot[["method_display", est_key]].rename(columns={est_key: "value"})
        sns.violinplot(
            data=sub,
            x="method_display",
            y="value",
            order=order_display,
            palette=[PALETTE[k] for k in order_display],
            inner="quartile",
            cut=4,
            bw="scott",
            linewidth=1.0,
            width=0.9,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="method_display",
            y="value",
            order=order_display,
            color="k",
            alpha=0.35,
            size=3,
            jitter=0.15,
            ax=ax,
        )
        ax.axhline(float(gt_const[true_key]), color="k", lw=2.2, alpha=0.95, zorder=10)
        ax.set_title(title_txt, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(ylab if j in (0, 2) else "")
        ax.tick_params(axis="x", rotation=50)
        if j == 1:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        sns.despine(ax=ax, top=True, right=True)

    # ---------- Row 4: CVLL preference + LR-style histograms ----------
    if "trial" in df_plot.columns and "cvll_gamma_mt" in df_plot.columns:
        cv_tab = df_plot.pivot_table(
            index="trial", columns="method", values="cvll_gamma_mt", aggfunc="mean"
        )
        # keep only trials with finite CVLL for all methods
        cv_tab = cv_tab.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    else:
        cv_tab = pd.DataFrame()

    # Col1: winner counts
    ax_win = fig.add_subplot(gs[ROW_CV, 0])
    if cv_tab.shape[0] > 0:
        winners = cv_tab.idxmax(axis=1)
        counts = winners.value_counts()
        xs = np.arange(len(PLOT_METHOD_KEYS))
        vals = [int(counts.get(m, 0)) for m in PLOT_METHOD_KEYS]
        cols = [PALETTE[METHOD_LABELS[m]] for m in PLOT_METHOD_KEYS]
        ax_win.bar(xs, vals, color=cols, alpha=0.9)
        ax_win.set_xticks(xs)
        ax_win.set_xticklabels([METHOD_LABELS[m] for m in PLOT_METHOD_KEYS], rotation=20, ha="right")
        ax_win.set_ylabel("Count (wins)")
        ax_win.set_title("CVLL wins (Gamma MT)", fontsize=12)
        ax_win.set_ylim(0, max(vals) + 1)
        sns.despine(ax=ax_win, top=True, right=True)
    else:
        ax_win.text(0.5, 0.5, "No CVLL data", ha="center", va="center")
        ax_win.axis("off")

    def _lr_hist(ax, a: str, b: str, title: str):
        if cv_tab.shape[0] == 0 or a not in cv_tab.columns or b not in cv_tab.columns:
            ax.text(0.5, 0.5, "No CVLL data", ha="center", va="center")
            ax.axis("off")
            return
        lr = (cv_tab[a] - cv_tab[b])   # ΔCVLL
        lr = lr.replace([np.inf, -np.inf], np.nan).dropna()
        if lr.size == 0:
            ax.text(0.5, 0.5, "No finite LR", ha="center", va="center")
            ax.axis("off")
            return
        ax.hist(lr.values, bins=25, alpha=0.85)
        ax.axvline(0.0, color="k", lw=2, alpha=1, zorder = 50)
        win_rate = float(np.mean(lr.values > 0.0)) * 100.0
        ax.set_title(f"{title}\n(ΔCVLL>0 in {win_rate:.1f}% trials)", fontsize=11)
        ax.set_xlabel("CVLL_A - CVLL_B")
        ax.set_ylabel("Count")
        sns.despine(ax=ax, top=True, right=True)

    ax_lr1 = fig.add_subplot(gs[ROW_CV, 1])
    _lr_hist(ax_lr1, "SL_specdecomp_additive", "specparam", "Pref: SL_specdecomp(Add) vs specparam")

    ax_lr2 = fig.add_subplot(gs[ROW_CV, 2])
    _lr_hist(ax_lr2, "SL_specdecomp_multiplicative", "specparam", "Pref: SL_specdecomp(Mult) vs specparam")

    ax_lr3 = fig.add_subplot(gs[ROW_CV, 3])
    _lr_hist(ax_lr3, "SL_specdecomp_multiplicative", "SL_specdecomp_additive", "Pref: SL_specdecomp(Mult) vs SL_specdecomp(Add)")

    title_mode = "Additive" if sim_mode == "additive" else "Multiplicative (specparam-like)"
    n_trials = int(df_plot["trial"].nunique()) if "trial" in df_plot.columns else int(len(df_plot)//len(METHOD_KEYS))
    fig.suptitle(
        f"Figure 3 (aux) — {title_mode} simulation "
        f"(N={n_trials}, rows1-3: 30 s MT K={K_TAPERS}, NW={NW}; "
        f"row4: {cv_folds}×{cv_chunk_dur:.0f}s CV, MT K={cv_k_tapers}, NW={cv_nw})",
        y=0.995,
        fontsize=16,
    )

    out_png = os.path.join(out_dir, f"Figure_3_FinalContinuousFunctionalPeak_{sim_mode}.png")
    out_svg = os.path.join(out_dir, f"Figure_3_FinalContinuousFunctionalPeak_{sim_mode}.svg")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}\n[saved] {out_svg}")
    return out_png, out_svg


# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["additive", "multiplicative", "both"], default="both")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--n-iter",
        type=int,
        default=200,
        help="Number of Monte-Carlo iterations per mode (ignored if --quick is set).",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="If set, override --n-iter and run with n_iter=2 (fast debug).",
    )
    ap.add_argument("--seed0", type=int, default=10000)

    # simulation params
    ap.add_argument("--peak", type=float, default=8.0)
    ap.add_argument("--amp", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--aper-exp", type=float, default=2.0)
    ap.add_argument("--aper-off", type=float, default=0.5)
    ap.add_argument("--knee", type=float, default=60.0)

    # CV params (row 4 only)
    ap.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    ap.add_argument("--cv-chunk-dur", type=float, default=CV_CHUNK_DUR)
    ap.add_argument("--cv-nw", type=float, default=CV_NW)
    ap.add_argument("--cv-k-tapers", type=int, default=CV_K_TAPERS)

    # CV sampler budget (row 4 only; rows 1-3 keep SL_KW_BASE)
    ap.add_argument("--cv-sl-draws", type=int, default=300)
    ap.add_argument("--cv-sl-tune", type=int, default=300)
    ap.add_argument("--cv-sl-chains", type=int, default=2)

    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    tp = TrueParams(
        exponent=float(args.aper_exp),
        offset=float(args.aper_off),
        knee=float(args.knee),
        peak=dict(
            freq=float(args.peak),
            amplitude=float(args.amp),
            sigma=float(args.sigma),
        ),
    )

    n_iter = 2 if args.quick else int(args.n_iter)
    modes = ("additive", "multiplicative") if args.mode == "both" else (args.mode,)
    print(f"[priors] A_lin_0 anchor percentile q={A_HEIGHT_ANCHOR_Q:g}; b prior sigma={B_PRIOR_SIGMA:g}")
    print(f"[run] n_iter={n_iter} per simulation regime; modes={modes}")

    for m in modes:
        build_figure(
            m,
            args.out_dir,
            n_iter=n_iter,
            seed0=int(args.seed0),
            true_params=tp,
            recompute=bool(args.recompute),
            cv_folds=int(args.cv_folds),
            cv_chunk_dur=float(args.cv_chunk_dur),
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            sl_cv_draws=int(args.cv_sl_draws),
            sl_cv_tune=int(args.cv_sl_tune),
            sl_cv_chains=int(args.cv_sl_chains),
        )


if __name__ == "__main__":
    main()
