#!/usr/bin/env python3
"""
Generate the supplemental height-grid benchmark. SEE replot_figure7_height_grid_compact_v2.py for compact plotting of the cached metrics which is plotted in the main text.

The script evaluates spectral decomposition performance across rhythm-height
levels using the Figure 3-aligned simulation ground truth and writes the compact
manuscript figure plus cached benchmark metrics.
"""

from __future__ import annotations

from pathlib import Path
import os
import argparse
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm

from SL_GPsim import spectrum
from spectral_connectivity import Multitaper, Connectivity
from specparam import SpectralModel
from SL_specdecomp import Decompose



PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# ---------------------- Style ----------------------
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

COL_GT = "k"
COL_BB = "#ff7f0e"
COL_RH = "red"
COL_MT = "0.4"


# ---------------------- Methods / palettes ----------------------
METHOD_KEYS = ["specparam", "SL_SD_additive", "SL_SD_specparam"]

METHOD_LABELS = {
    "specparam": "specparam",
    "SL_SD_additive": "SL_SD (Additive)",
    "SL_SD_specparam": "SL_SD (Multiplicative)",
}

PLOT_METHOD_ORDER = ["specparam", "SL_SD_additive", "SL_SD_specparam"]

METHOD_PALETTE = dict(
    zip(
        [METHOD_LABELS[m] for m in METHOD_KEYS],
        sns.color_palette("deep", n_colors=len(METHOD_KEYS)),
    )
)

AMP_GRID = np.array([0.5, 1.0, 2.0, 4.0], dtype=float)


# ---------------------- Config ----------------------
FS               = 1000.0
NW               = 2
K_TAPERS         = 3
WIN_DUR          = 30.0
ANALYSIS_FRANGE  = (1.0, 200.0)
SLOPE_BAND       = (40.0, 60.0)
HG_BAND          = (80.0, 180.0)
RHY_BAND         = (1.0, 20.0)

DEFAULT_TRUE_PEAK_HZ = 8.0

# Intentional Figure-7-final prior overrides
B_PRIOR_SIGMA = 5.0
ADDITIVE_AMP_ANCHOR_Q = 50.0
ADDITIVE_AMP_PRIOR_SIGMA = 1.25

# Figure 3-aligned prior settings for SL_specdecomp.
FIG3_PRIORS = dict(
    knee_hz_bounds=ANALYSIS_FRANGE,
    slope_mu=-2.0,
    slope_sigma=1.0,
    slope_bounds=(-5.0, -0.5),
    sigma_mu=3.0,
    sigma_sigma=2.0,
    sigma_bounds=(0.5, 12.0),
    a_log_sigma=1.0,
)

DEFAULT_OUT_DIR = os.path.expanduser(
    "CHANGE_THIS_ROOT_TO_PATH/Bloniasz_Stephen_Estimation/Output/Results/FiguresIntermediate/Figure_7_Fig3GroundTruth_HeightGrid"
)
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

SP_KW = dict(
    peak_width_limits=[1.0, 30.0],
    max_n_peaks=1,
    min_peak_height=0.0,
    peak_threshold=2.0,
    aperiodic_mode="knee",
    verbose=False,
)

SL_KW_BASE = dict(
    n_aperiodics=1,
    n_rhythms=1,
    rhythm_bands=[(RHY_BAND[0], RHY_BAND[1])],
    k_tapers=float(K_TAPERS),
    priors=FIG3_PRIORS,
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


# ---------------------- Utilities ----------------------
def _kappa_from_knee_hz(knee_hz: float, exponent: float) -> float:
    """Convert knee frequency (Hz) -> kappa (FOOOF/specparam knee parameter)."""
    knee_hz = float(knee_hz) if knee_hz is not None else np.nan
    exponent = float(exponent) if exponent is not None else np.nan
    if not np.isfinite(knee_hz) or not np.isfinite(exponent) or knee_hz <= 0 or exponent <= 0:
        return np.nan
    return float(knee_hz ** exponent)


def _knee_freq_hz_from_kappa(kappa: float, exponent: float) -> float:
    """Convert kappa -> knee frequency (Hz)."""
    kappa = float(kappa) if kappa is not None else np.nan
    exponent = float(exponent) if exponent is not None else np.nan
    if not np.isfinite(kappa) or not np.isfinite(exponent) or kappa <= 0 or exponent <= 0:
        return np.nan
    return float(kappa ** (1.0 / exponent))


def _knee_hz_auto(knee_raw: float, exponent: float, knee_true_hz: float) -> float:
    """
    Given a raw knee estimate that may be either:
      - already in Hz, OR
      - in kappa
    choose the interpretation that is closer to the known truth.
    """
    knee_raw = float(knee_raw) if knee_raw is not None else np.nan
    if not np.isfinite(knee_raw):
        return np.nan

    cand_hz_direct = knee_raw
    cand_hz_from_kappa = _knee_freq_hz_from_kappa(knee_raw, exponent)

    if not np.isfinite(knee_true_hz):
        return cand_hz_from_kappa if np.isfinite(cand_hz_from_kappa) else cand_hz_direct

    d0 = abs(cand_hz_direct - knee_true_hz) if np.isfinite(cand_hz_direct) else np.inf
    d1 = abs(cand_hz_from_kappa - knee_true_hz) if np.isfinite(cand_hz_from_kappa) else np.inf
    return cand_hz_from_kappa if d1 < d0 else cand_hz_direct


def _maybe(fd, nameA, nameB=None):
    v = getattr(fd, nameA, None)
    if v is None and nameB:
        v = getattr(fd, nameB, None)
    if v is None:
        return None
    if callable(v):
        try:
            v = v()
        except TypeError:
            pass
    return np.asarray(v, float).ravel()


def _spawn_rng_streams(seed0: int) -> Dict[str, np.random.Generator]:
    """
    Initialize all stochastic streams once from a single reproducible seed.

    We use separate spawned streams so simulation randomness, additive-MCMC
    randomness, and multiplicative-MCMC randomness are reproducible but not
    generated from ad hoc seed arithmetic.
    """
    ss = np.random.SeedSequence(int(seed0))
    sim_ss, add_ss, mul_ss, panel_ss = ss.spawn(4)
    return {
        "simulation": np.random.default_rng(sim_ss),
        "slsd_additive": np.random.default_rng(add_ss),
        "slsd_multiplicative": np.random.default_rng(mul_ss),
        "panel": np.random.default_rng(panel_ss),
    }


def _draw_u32_seed(rng: np.random.Generator) -> int:
    """Draw one PyMC/NumPy-compatible uint32 seed from a persistent RNG stream."""
    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def _independent_grid(freqs: np.ndarray, twin: float, NW: float) -> Tuple[np.ndarray, int]:
    """Downsample frequencies to an approximately independent multitaper grid."""
    f = np.asarray(freqs, float)
    if f.size <= 1:
        return f.copy(), 1
    df = float(np.median(np.diff(f)))
    delta_f_indep = 2.0 * NW / twin
    step_bins = max(1, int(round(delta_f_indep / max(df, 1e-12))))
    return f[::step_bins], step_bins


def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    f = np.asarray(freqs, float)
    return (f >= lo) & (f <= hi)


def _band_scale_quantile(
    freqs: np.ndarray,
    y_lin: np.ndarray,
    lo: float,
    hi: float,
    q: float,
) -> float:
    """same robust band scale as SL_specdecomp, but with explicit q."""
    f = np.asarray(freqs, float)
    y = np.asarray(y_lin, float)
    m = np.isfinite(y) & (y > 0) & (f >= lo) & (f <= hi)
    if m.sum() < 5:
        m = np.isfinite(y) & (y > 0)
    if not np.any(m):
        return 1e-12
    return float(np.percentile(y[m], q))


def _median_log10_positive_power(y_lin: np.ndarray) -> float:
    y = np.asarray(y_lin, float)
    y_pos = y[np.isfinite(y) & (y > 0)]
    return float(np.log10(np.median(y_pos))) if y_pos.size else 0.0


def _b_prior_specs(freqs_fit: np.ndarray, power_lin: np.ndarray, mode: str) -> Dict[str, Any]:
    """
    Override b prior through SL_specdecomp's aperiodic_param_specs API.

    build_additive_model uses the key b_0 when n_aperiodics=1.
    build_multiplicative_model uses the key b when n_aperiodics=1.
    """
    del freqs_fit
    mu_b = _median_log10_positive_power(power_lin)
    key = "b_0" if mode == "additive" else "b"

    return {
        key: {
            "factory": lambda name, mu_b=mu_b: pm.Normal(
                name,
                mu=mu_b,
                sigma=B_PRIOR_SIGMA,
            )
        }
    }


def _additive_rhythm_prior_specs(
    freqs_fit: np.ndarray,
    power_lin: np.ndarray,
    q: float = ADDITIVE_AMP_ANCHOR_Q,
) -> Dict[str, Any]:
    """
    Override additive A_lin_0 so its LogNormal location is anchored to the
    qth percentile in the rhythm band. This replaces the package default q=99
    while preserving the original LogNormal sigma=1.25.
    """
    band_scale = _band_scale_quantile(
        freqs_fit,
        power_lin,
        RHY_BAND[0],
        RHY_BAND[1],
        q=q,
    )
    return {
        "A_lin_0": {
            "factory": lambda name, band_scale=band_scale: pm.LogNormal(
                name,
                mu=np.log(max(float(band_scale), 1e-12)),
                sigma=ADDITIVE_AMP_PRIOR_SIGMA,
            )
        }
    }


def _slope_loglog(
    freqs: np.ndarray,
    power_lin: np.ndarray,
    band: Tuple[float, float],
) -> float:
    """Slope of log10(power) vs log10(freq) within a band."""
    freqs = np.asarray(freqs, float)
    power_lin = np.asarray(power_lin, float)
    m = _band_mask(freqs, band) & np.isfinite(power_lin) & (power_lin > 0)
    if m.sum() < 2:
        return np.nan
    xf = np.log10(freqs[m])
    yf = np.log10(np.clip(power_lin[m], 1e-20, np.inf))
    A = np.vstack([xf, np.ones_like(xf)]).T
    chi, _b = np.linalg.lstsq(A, yf, rcond=None)[0]
    return float(chi)
def _truth_values_from_cached_ground_truth(
    row1_payload: Dict[str, Any],
    true_cf: float,
) -> Dict[str, Any]:
    """
    Recover ground-truth plotting values directly from cached GT curves.

    Uses:
      - bb_true for the true broadband 40–60 Hz slope
      - bb_true + rh_true for true rhythm-height markers at true_cf

    This avoids trusting stale/incorrect truth columns in the metrics CSV.
    """
    freqs = np.asarray(row1_payload["freqs"], float).ravel()
    amp_vals = np.asarray(row1_payload["amp_vals"], float).ravel()
    bb_true = np.asarray(row1_payload["bb_true"], float)
    rh_true = np.asarray(row1_payload["rh_true"], float)

    if bb_true.ndim != 2:
        raise ValueError(f"Expected bb_true to be 2D, got shape {bb_true.shape}")
    if rh_true.shape != bb_true.shape:
        raise ValueError(f"Expected rh_true shape {bb_true.shape}, got {rh_true.shape}")

    # Correct horizontal ground-truth line:
    # broadband slope should come from the true broadband curve only.
    bb_slope_true_by_amp = np.asarray(
        [_slope_loglog(freqs, bb_true[i, :], SLOPE_BAND) for i in range(bb_true.shape[0])],
        float,
    )
    bb_slope_true = float(np.nanmean(bb_slope_true_by_amp))

    # Correct rhythm-height truth markers:
    # evaluate full true spectrum at the simulation's true center frequency.
    rh_height_true_log10_by_amp = {}
    for i, amp in enumerate(amp_vals):
        full_true = np.clip(bb_true[i, :] + rh_true[i, :], 1e-20, np.inf)
        height_lin = float(np.interp(float(true_cf), freqs, full_true))
        rh_height_true_log10_by_amp[float(amp)] = float(np.log10(max(height_lin, 1e-20)))

    return dict(
        bb_slope_true=bb_slope_true,
        bb_slope_true_by_amp=bb_slope_true_by_amp,
        rh_height_true_log10_by_amp=rh_height_true_log10_by_amp,
    )

def _continuous_argmax(
    freqs: np.ndarray,
    y_lin: np.ndarray,
    band: Tuple[float, float],
    fine_step_hz: float = 1e-5,
) -> Tuple[float, float]:
    """Continuous argmax in band via dense interpolation + quadratic vertex."""
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


# ---------------------- True params & simulation ----------------------
@dataclass
class TrueParams:
    exponent: float = 2.0
    offset: float = 0.5          # log10 offset for spectrum()
    knee_raw: float = 60.0       # passed directly to SL_GPsim.spectrum(knee=...)
    peak: Optional[Dict[str, float]] = None  # dict(freq, amplitude (linear), sigma)


def simulate_once_additive(
    params: TrueParams,
    fs: float = FS,
    duration: float = WIN_DUR,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Simulate one 30 s realization from the ADDITIVE generator and compute
    a multitaper PSD on an independent grid.

    Returns both the dense generator spectra AND the ground-truth spectra
    projected to the independent multitaper grid (f_ref).
    """
    peak = params.peak if params.peak is not None else dict(
        freq=float(DEFAULT_TRUE_PEAK_HZ), amplitude=2.0, sigma=3.0
    )

    knee_raw = float(params.knee_raw)
    implied_knee_hz = _knee_freq_hz_from_kappa(knee_raw, params.exponent)
    print(f"The raw knee passed to spectrum() is {knee_raw}; implied knee Hz is {implied_knee_hz:.6g}")

    res = spectrum(
        sampling_rate=fs,
        duration=duration,
        aperiodic_exponent=params.exponent,
        aperiodic_offset=params.offset,
        knee=knee_raw,        # Figure 3 convention: pass raw knee directly
        peaks=[peak],
        average_firing_rate=0.0,
        random_state=rng_seed,
        direct_estimate=False,
        plot=False,
        mode="additive",
    )

    td = res.time_domain
    fd = res.frequency_domain

    # True dense PSD components (from generator)
    f_dense = np.asarray(fd.frequencies, dtype=float).ravel()
    S_full = _maybe(fd, "combined_spectrum")
    if S_full is None:
        raise RuntimeError("frequency_domain.combined_spectrum unavailable")

    S_bb = _maybe(fd, "broadband_spectrum", "aperiodic_spectrum")
    S_rh = _maybe(fd, "rhythmic_spectrum", "peaks_spectrum")

    if S_bb is None and S_rh is None:
        S_bb = S_full.copy()
        S_rh = np.zeros_like(S_bb)
    elif S_bb is None:
        S_bb = np.clip(S_full - S_rh, 0.0, np.inf)
    elif S_rh is None:
        S_rh = np.clip(S_full - S_bb, 0.0, np.inf)

    # Multitaper on full 30 s window (single window)
    ts = np.asarray(td.combined_signal, float).ravel()[:, None, None]
    mt = Multitaper(
        ts,
        sampling_frequency=fs,
        n_tapers=K_TAPERS,
        time_halfbandwidth_product=NW,
        start_time=0.0,
        time_window_duration=duration,
        time_window_step=duration,
    )
    conn = Connectivity.from_multitaper(mt)
    f_emp = np.asarray(conn.frequencies, float).ravel()
    S_emp = np.asarray(conn.power().squeeze(), float).ravel()

    # Independent grid for fitting + metrics
    f_ref, step_bins = _independent_grid(f_emp, duration, NW)
    S_fit = S_emp[::step_bins]

    # Make np.interp well-defined (xp must be strictly increasing)
    m = np.isfinite(f_dense) & np.isfinite(S_full) & np.isfinite(S_bb) & np.isfinite(S_rh)
    f_dense = f_dense[m]; S_full = S_full[m]; S_bb = S_bb[m]; S_rh = S_rh[m]
    order = np.argsort(f_dense)
    f_dense = f_dense[order]; S_full = S_full[order]; S_bb = S_bb[order]; S_rh = S_rh[order]

    # Project GT spectra to the independent MT grid
    GT_full_on_ref = np.interp(f_ref, f_dense, S_full)
    GT_bb_on_ref   = np.interp(f_ref, f_dense, S_bb)
    GT_rh_on_ref   = np.interp(f_ref, f_dense, S_rh)

    return dict(
        fs=fs,
        duration=duration,
        time=np.asarray(td.time, float).ravel(),
        x_comb=np.asarray(td.combined_signal, float).ravel(),
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


# ---------------------- Estimator helpers ----------------------
def _specparam_cf_from_model(
    fm: SpectralModel,
    true_cf: Optional[float] = None,
) -> float:
    """Extract a center frequency from a fitted SpectralModel."""
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


def _specparam_knee_from_model(fm: SpectralModel) -> float:
    """
    Extract knee from a fitted SpectralModel (aperiodic_mode='knee').

    Typically aperiodic_params = [offset, knee, exponent].
    Knee is commonly returned as kappa (Hz^exponent), not knee frequency.
    """
    for key in ("aperiodic_params", "aperiodic"):
        try:
            ap = np.asarray(fm.get_params(key), float).ravel()
            if ap.size >= 3:
                return float(ap[1])
        except Exception:
            pass

    res = getattr(fm, "results", None)
    for attr in ("aperiodic_params", "aperiodic_params_"):
        ap = getattr(res, attr, None) if res is not None else None
        if ap is not None:
            ap = np.asarray(ap, float).ravel()
            if ap.size >= 3:
                return float(ap[1])

    return np.nan


def _specparam_full_on_grid(
    freqs_fit: np.ndarray,
    power_lin: np.ndarray,
    freq_range: Tuple[float, float],
    true_cf: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Fit specparam on (freqs_fit, power_lin) within freq_range and return:

        full_lin, ap_lin, rh_lin, cf_est, knee_est_raw

    all evaluated on freqs_fit.
    """
    eps = 1e-20
    freqs_fit = np.asarray(freqs_fit, float).ravel()
    power_lin = np.asarray(power_lin, float).ravel()

    fm = SpectralModel(**SP_KW)
    fm.fit(
        freqs_fit,
        np.clip(power_lin, eps, np.inf),
        freq_range=freq_range,
    )

    cf_est = _specparam_cf_from_model(fm, true_cf=true_cf)
    knee_est = _specparam_knee_from_model(fm)

    model_obj = getattr(getattr(fm, "results", None), "model", None)
    if model_obj is None:
        raise RuntimeError("specparam results.model not available")

    def _to_lin(a: Any) -> np.ndarray:
        a = np.asarray(a, float).ravel()
        return 10.0 ** a if (np.nanmin(a) < 0 and np.nanmax(a) < 15) else a

    def _interp_safe(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
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

    def _pick_x_for_fp(fp: np.ndarray, default_range: Tuple[float, float]) -> np.ndarray:
        fp = np.asarray(fp).ravel()
        cand_x: List[np.ndarray] = []
        for name in ("freqs", "freqs_model", "_freqs", "_spectrum_freqs"):
            x = getattr(model_obj, name, None)
            if x is not None:
                cand_x.append(np.asarray(x, float).ravel())
        x_fm = np.asarray(getattr(fm, "freqs", freqs_fit), float).ravel()
        cand_x.append(x_fm)

        for cx in cand_x:
            if cx.size == fp.size:
                return cx
        lo, hi = map(float, default_range)
        n = max(2, fp.size)
        return np.linspace(lo, hi, n)

    # FULL model
    full_attr = getattr(model_obj, "modeled_spectrum", None)
    if callable(full_attr):
        try:
            full_native = np.asarray(full_attr(space="linear"), float).ravel()
        except TypeError:
            full_native = _to_lin(full_attr())
    else:
        full_native = _to_lin(full_attr)
    xp_full = _pick_x_for_fp(full_native, freq_range)

    # Aperiodic component
    get_comp = getattr(model_obj, "get_component", None)
    try:
        ap_native = np.asarray(get_comp("aperiodic", space="linear"), float).ravel()
    except TypeError:
        ap_native = _to_lin(get_comp("aperiodic"))
    xp_ap = _pick_x_for_fp(ap_native, freq_range)

    full_lin = _interp_safe(freqs_fit, xp_full, full_native)
    ap_lin   = _interp_safe(freqs_fit, xp_ap, ap_native)
    full_lin = np.clip(full_lin, eps, np.inf)
    ap_lin   = np.clip(ap_lin, eps, np.inf)
    rh_lin   = np.clip(full_lin - ap_lin, eps, np.inf)

    return full_lin, ap_lin, rh_lin, cf_est, knee_est


def _slsd_peak_params(model) -> np.ndarray:
    """Extract (center, amplitude, FWHM) from an SL_SD model's posterior."""
    peaks: List[List[float]] = []
    idata = getattr(model, "idata", None)
    ds = getattr(idata, "posterior", None) if idata is not None else None
    if ds is None:
        return np.empty((0, 3), float)

    # Case 1: vectorized variables
    if "center" in ds and "sigma" in ds:
        Aname = "A_lin" if "A_lin" in ds else ("A" if "A" in ds else None)
        if Aname is not None:
            c = ds["center"].mean(dim=[d for d in ds["center"].dims if d in ("chain", "draw")]).values.ravel()
            s = ds["sigma"].mean(dim=[d for d in ds["sigma"].dims if d in ("chain", "draw")]).values.ravel()
            a = ds[Aname].mean(dim=[d for d in ds[Aname].dims if d in ("chain", "draw")]).values.ravel()
            for ci, si, ai in zip(c, s, a):
                peaks.append([float(ci), float(ai), float(2.3548 * si)])

    # Case 2: scalar/indexed variables
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


def _slsd_center_from_model(model, true_cf: Optional[float] = None) -> float:
    """Posterior-mean CF from SL_SD peak parameters, analogous to specparam."""
    peaks = _slsd_peak_params(model)
    if peaks.size == 0:
        return np.nan
    cfs  = np.asarray(peaks[:, 0], float).ravel()
    amps = np.asarray(peaks[:, 1], float).ravel()
    if cfs.size == 0:
        return np.nan
    if true_cf is not None and np.isfinite(true_cf):
        idx = int(np.argmin(np.abs(cfs - true_cf)))
    else:
        idx = int(np.nanargmax(amps))
    return float(cfs[idx])


def _slsd_knee_from_model(model, idx: int = 0) -> float:
    """
    Posterior-mean knee from an SL_SD model's posterior.

    May be kappa or knee-frequency depending on SL_SD implementation; we store
    it as knee_est_raw and convert to Hz with _knee_hz_auto.
    """
    idata = getattr(model, "idata", None)
    ds = getattr(idata, "posterior", None) if idata is not None else None
    if ds is None:
        return np.nan

    candidates = [
        f"knee_{idx}", f"knee[{idx}]",
        "knee", "knee_ap", "knee_aperiodic", "aperiodic_knee",
    ]
    for name in candidates:
        if name in ds:
            v = ds[name]
            dims = [d for d in ("chain", "draw") if d in v.dims]
            return float(v.mean(dim=dims).values.item())

    kneenames = [v for v in ds.data_vars if "knee" in v.lower()]
    if not kneenames:
        return np.nan

    kneenames.sort(
        key=lambda s: (
            0 if re.search(rf"(_|\[){idx}\]?$", s) else 1,
            len(s),
        )
    )
    name = kneenames[0]
    v = ds[name]
    dims = [d for d in ("chain", "draw") if d in v.dims]
    return float(v.mean(dim=dims).values.item())


def _slsd_components_on_grid(
    freqs_fit: np.ndarray,
    power_lin: np.ndarray,
    fs: float,
    mode: str = "additive",
    sample_seed: Optional[int] = None,
    **kw,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Fit SL_SD and return (model, total_lin, bb_lin, rh_lin) on freqs_fit."""
    eps = 1e-20
    freqs_fit = np.asarray(freqs_fit, float).ravel()
    power_lin = np.asarray(power_lin, float).ravel()

    kw_eff = dict(SL_KW_BASE)
    kw_eff.update(kw)

    # Critical: Decompose defaults to random_seed=42 inside the package API
    # unless sample_kwargs supplies a different seed. Draw this seed from a
    # persistent RNG stream in run_height_grid rather than resetting it by hand.
    if sample_seed is not None:
        skw = dict(kw_eff.get("sample_kwargs", {}) or {})
        skw["random_seed"] = int(sample_seed)
        kw_eff["sample_kwargs"] = skw

    kw_eff["mode"] = mode

    model = Decompose(
        freqs_fit,
        np.clip(power_lin, eps, np.inf),
        fs=fs,
        **kw_eff,
    )

    total = np.asarray(getattr(model, "estimated_spectrum"), float).ravel()
    bb = getattr(model, "broadband", None)
    if bb is None:
        bb = getattr(model, "P_ap", None)
    rh = getattr(model, "rhythms", None)
    if rh is None:
        rh = getattr(model, "P_rh", None)

    if bb is None and rh is None:
        bb = total.copy()
        rh = np.zeros_like(total)
    elif bb is None:
        bb = np.clip(total - rh, 0.0, np.inf)
    elif rh is None:
        rh = np.clip(total - bb, 0.0, np.inf)

    def _floor(a):
        return np.clip(np.asarray(a, float).ravel(), eps, np.inf)

    return model, _floor(total), _floor(bb), _floor(rh)


# ---------------------- Metrics ----------------------
def _metrics_from_full(
    freqs: np.ndarray,
    total_lin_true: np.ndarray,
    total_lin_est: np.ndarray,
    true_cf: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute metrics from FULL PSDs on a shared grid (ideally f_ref).

    Figure-3 additive truth convention:
      - rh_cf_true is the *simulation parameter* (true_cf)
      - rh_height_true is the PSD height *evaluated at true_cf* (not an argmax)
      - rh_height_est is the estimate evaluated at true_cf (not an argmax)
    """
    eps = 1e-20
    freqs = np.asarray(freqs, float).ravel()
    total_lin_true = np.asarray(total_lin_true, float).ravel()
    total_lin_est  = np.asarray(total_lin_est,  float).ravel()

    # Ensure monotone freqs for interp
    order = np.argsort(freqs)
    freqs = freqs[order]
    total_lin_true = total_lin_true[order]
    total_lin_est  = total_lin_est[order]

    # Always compute an argmax-based cf_est as a fallback / diagnostic
    cf_est_argmax, pk_est_argmax = _continuous_argmax(
        freqs, total_lin_est, RHY_BAND, fine_step_hz=1e-5
    )

    # --- Truth CF and heights ---
    if true_cf is not None and np.isfinite(true_cf) and (freqs.min() <= true_cf <= freqs.max()):
        cf_true = float(true_cf)
        pk_true = float(np.interp(cf_true, freqs, np.clip(total_lin_true, eps, np.inf)))
        pk_est  = float(np.interp(cf_true, freqs, np.clip(total_lin_est,  eps, np.inf)))
    else:
        # Fallback to continuous max if true_cf missing/out-of-range
        cf_true, pk_true = _continuous_argmax(
            freqs, total_lin_true, RHY_BAND, fine_step_hz=1e-5
        )
        pk_est = float(pk_est_argmax)

    rh_height_true_log10 = float(np.log10(max(pk_true, eps)))
    rh_height_est_log10  = float(np.log10(max(pk_est,  eps)))

    # HG mean power
    m_hg = _band_mask(freqs, HG_BAND)
    hg_true_log10 = float(np.log10(max(np.mean(total_lin_true[m_hg]) if np.any(m_hg) else np.nan, eps)))
    hg_est_log10  = float(np.log10(max(np.mean(total_lin_est[m_hg])  if np.any(m_hg) else np.nan, eps)))

    # Slope on the same grid
    slope_true = _slope_loglog(freqs, np.clip(total_lin_true, eps, np.inf), SLOPE_BAND)
    slope_est  = _slope_loglog(freqs, np.clip(total_lin_est,  eps, np.inf), SLOPE_BAND)

    return dict(
        rh_cf_true=float(cf_true),
        rh_cf_est=float(cf_est_argmax),  # will still be overwritten with model peak CF later
        rh_height_true_log10=rh_height_true_log10,
        rh_height_est_log10=rh_height_est_log10,
        hg_true_log10=hg_true_log10,
        hg_est_log10=hg_est_log10,
        slope_true=slope_true,
        slope_est=slope_est,
    )


# ---------------------- Monte-Carlo height grid ----------------------
def run_height_grid(
    params_base: TrueParams,
    n_levels: int,
    n_iter_per_level: int,
    amp_min: float,
    amp_max: float,
    seed0: int = 12345,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Vary peak amplitude over fixed AMP_GRID and run all decomposers.
    """
    amp_grid = AMP_GRID.copy()
    n_levels = amp_grid.size
    del amp_min, amp_max

    rng_streams = _spawn_rng_streams(seed0)
    sim_rng = rng_streams["simulation"]
    slsd_add_rng = rng_streams["slsd_additive"]
    slsd_mul_rng = rng_streams["slsd_multiplicative"]

    records: List[Dict[str, Any]] = []

    row1_freqs: Optional[np.ndarray] = None
    row1_bb: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in METHOD_KEYS}
    row1_rh: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in METHOD_KEYS}
    row1_bb_true: Dict[int, np.ndarray] = {}
    row1_rh_true: Dict[int, np.ndarray] = {}

    # CF used to pick the closest estimated peak for each model
    true_cf_param: float = float(DEFAULT_TRUE_PEAK_HZ)
    if params_base.peak is not None and "freq" in params_base.peak:
        try:
            true_cf_param = float(params_base.peak["freq"])
        except Exception:
            true_cf_param = float(DEFAULT_TRUE_PEAK_HZ)

    for level_idx in range(n_levels):
        amp_level = float(amp_grid[level_idx])

        base_peak_freq = float(DEFAULT_TRUE_PEAK_HZ)
        base_peak_sigma = 3.0
        if params_base.peak is not None:
            base_peak_freq = float(params_base.peak.get("freq", DEFAULT_TRUE_PEAK_HZ))
            base_peak_sigma = float(params_base.peak.get("sigma", 3.0))

        params_level = TrueParams(
            exponent=float(params_base.exponent),
            offset=float(params_base.offset),
            knee_raw=float(params_base.knee_raw),
            peak=dict(
                freq=base_peak_freq,
                amplitude=amp_level,
                sigma=base_peak_sigma,
            ),
        )

        # Ground-truth CF is the simulation parameter (NOT an argmax)
        rh_cf_true_param = float(params_level.peak["freq"])

        for rep in range(n_iter_per_level):
            # Draw all stochastic seeds sequentially from persistent streams.
            # This is reproducible from seed0, but avoids deterministic seed
            # arithmetic and avoids resetting the same PyMC seed for every fit.
            sim_seed = _draw_u32_seed(sim_rng)
            add_sample_seed = _draw_u32_seed(slsd_add_rng)
            mul_sample_seed = _draw_u32_seed(slsd_mul_rng)

            sim = simulate_once_additive(params_level, rng_seed=sim_seed)
            f_ref = sim["f_ref"]
            GT_full_on_ref = sim["GT_full_on_ref"]

            if row1_freqs is None:
                row1_freqs = f_ref.copy()
            else:
                if row1_freqs.shape != f_ref.shape or not np.allclose(row1_freqs, f_ref):
                    raise RuntimeError(
                        "Inconsistent f_ref across simulations; cannot construct a single Row-1 payload grid."
                    )

            fr_fit = (
                max(ANALYSIS_FRANGE[0], float(f_ref[0])),
                min(ANALYSIS_FRANGE[1], float(f_ref[-1])),
            )

            # specparam
            sp_full, sp_ap, sp_rh, sp_cf, sp_knee = _specparam_full_on_grid(
                f_ref, sim["S_fit"], fr_fit, true_cf=true_cf_param
            )

            # SL_SD additive: b sigma=5, A_lin_0 anchored to rhythm-band median (q=50)
            add_aperiodic_specs = _b_prior_specs(f_ref, sim["S_fit"], mode="additive")
            add_rhythm_specs = _additive_rhythm_prior_specs(
                f_ref,
                sim["S_fit"],
                q=ADDITIVE_AMP_ANCHOR_Q,
            )
            model_add, add_full, add_bb, add_rh = _slsd_components_on_grid(
                f_ref,
                sim["S_fit"],
                fs=sim["fs"],
                mode="additive",
                sample_seed=add_sample_seed,
                aperiodic_param_specs=add_aperiodic_specs,
                rhythm_param_specs=add_rhythm_specs,
            )
            add_cf = _slsd_center_from_model(model_add, true_cf=true_cf_param)
            add_knee = _slsd_knee_from_model(model_add, idx=0)

            # SL_SD multiplicative: b sigma=5. Rhythm-height prior is not q=99-based here.
            mul_aperiodic_specs = _b_prior_specs(f_ref, sim["S_fit"], mode="FOOOF_spectrum")
            model_mul, mul_full, mul_bb, mul_rh = _slsd_components_on_grid(
                f_ref,
                sim["S_fit"],
                fs=sim["fs"],
                mode="FOOOF_spectrum",
                sample_seed=mul_sample_seed,
                aperiodic_param_specs=mul_aperiodic_specs,
            )
            mul_cf = _slsd_center_from_model(model_mul, true_cf=true_cf_param)
            mul_knee = _slsd_knee_from_model(model_mul, idx=0)

            # Row-1 payload (first replicate per level)
            if rep == 0:
                row1_bb_true[level_idx] = sim["GT_bb_on_ref"].copy()
                row1_rh_true[level_idx] = sim["GT_rh_on_ref"].copy()

                row1_bb["specparam"][level_idx] = sp_ap.copy()
                row1_rh["specparam"][level_idx] = sp_rh.copy()

                row1_bb["SL_SD_additive"][level_idx] = add_bb.copy()
                row1_rh["SL_SD_additive"][level_idx] = add_rh.copy()

                row1_bb["SL_SD_specparam"][level_idx] = mul_bb.copy()
                row1_rh["SL_SD_specparam"][level_idx] = mul_rh.copy()

            # Metrics (same grid) + knees in Hz
            exp_true = float(params_level.exponent)
            knee_true_raw = float(params_level.knee_raw)
            knee_true_hz = _knee_freq_hz_from_kappa(knee_true_raw, exp_true)

            rec_sp = _metrics_from_full(f_ref, GT_full_on_ref, sp_full, true_cf=rh_cf_true_param)

            rec_sp["rh_cf_true"] = rh_cf_true_param
            rec_sp["rh_cf_est"]  = float(sp_cf)

            rec_sp["knee_true_raw"] = knee_true_raw
            rec_sp["knee_true_hz"] = knee_true_hz
            rec_sp["knee_est_raw"] = float(sp_knee)
            rec_sp["knee_hz_est"]  = _knee_freq_hz_from_kappa(sp_knee, exp_true)  # deterministic
            rec_sp.update(dict(
                grid_param="amp", grid_level=level_idx, amp_true=amp_level, method="specparam",
                seed=sim_seed, simulation_seed=sim_seed, sampler_seed=np.nan,
                slsd_additive_seed=np.nan, slsd_multiplicative_seed=np.nan,
            ))
            records.append(rec_sp)

            rec_add = _metrics_from_full(f_ref, GT_full_on_ref, add_full, true_cf=rh_cf_true_param)
            rec_add["rh_cf_true"] = rh_cf_true_param
            rec_add["rh_cf_est"]  = float(add_cf)

            rec_add["knee_true_raw"] = knee_true_raw
            rec_add["knee_true_hz"] = knee_true_hz
            rec_add["knee_est_raw"] = float(add_knee)
            rec_add["knee_hz_est"]  = _knee_hz_auto(add_knee, exp_true, knee_true_hz)
            rec_add.update(dict(
                grid_param="amp", grid_level=level_idx, amp_true=amp_level, method="SL_SD_additive",
                seed=sim_seed, simulation_seed=sim_seed, sampler_seed=add_sample_seed,
                slsd_additive_seed=add_sample_seed, slsd_multiplicative_seed=np.nan,
            ))
            records.append(rec_add)

            rec_mul = _metrics_from_full(f_ref, GT_full_on_ref, mul_full, true_cf=rh_cf_true_param)

            rec_mul["rh_cf_true"] = rh_cf_true_param
            rec_mul["rh_cf_est"]  = float(mul_cf)

            rec_mul["knee_true_raw"] = knee_true_raw
            rec_mul["knee_true_hz"] = knee_true_hz
            rec_mul["knee_est_raw"] = float(mul_knee)
            rec_mul["knee_hz_est"]  = _knee_hz_auto(mul_knee, exp_true, knee_true_hz)
            rec_mul.update(dict(
                grid_param="amp", grid_level=level_idx, amp_true=amp_level, method="SL_SD_specparam",
                seed=sim_seed, simulation_seed=sim_seed, sampler_seed=mul_sample_seed,
                slsd_additive_seed=np.nan, slsd_multiplicative_seed=mul_sample_seed,
            ))
            records.append(rec_mul)

    df = pd.DataFrame.from_records(records)

    if row1_freqs is None:
        raise RuntimeError("No row-1 payload frequencies found; something went wrong in simulation loop.")

    n_methods = len(METHOD_KEYS)
    n_freqs = row1_freqs.size
    bb_arr = np.empty((n_methods, n_levels, n_freqs), float)
    rh_arr = np.empty_like(bb_arr)
    bb_true_arr = np.empty((n_levels, n_freqs), float)
    rh_true_arr = np.empty_like(bb_true_arr)

    for mi, m in enumerate(METHOD_KEYS):
        for li in range(n_levels):
            if li not in row1_bb[m] or li not in row1_rh[m]:
                raise RuntimeError(f"Missing Row-1 payload for method '{m}', level index {li}.")
            bb_arr[mi, li, :] = row1_bb[m][li]
            rh_arr[mi, li, :] = row1_rh[m][li]

    for li in range(n_levels):
        if li not in row1_bb_true or li not in row1_rh_true:
            raise RuntimeError(f"Missing ground-truth Row-1 payload for level index {li}.")
        bb_true_arr[li, :] = row1_bb_true[li]
        rh_true_arr[li, :] = row1_rh_true[li]

    row1_payload = dict(
        freqs=row1_freqs,
        amp_vals=amp_grid,
        bb=bb_arr,
        rh=rh_arr,
        bb_true=bb_true_arr,
        rh_true=rh_true_arr,
    )

    return df, row1_payload


# ---------------------- Ground-truth panel helper ----------------------
def simulate_ground_truth_spectra_for_panel(
    params: Optional[TrueParams] = None,
    amp_grid: np.ndarray = AMP_GRID,
    rng_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ground-truth panel for Figure 7 (ADDITIVE): dense illustrative curves.
    """
    if params is None:
        params = TrueParams(peak=dict(freq=DEFAULT_TRUE_PEAK_HZ, amplitude=2.0, sigma=3.0))

    freqs_dense = None
    bb_list: List[np.ndarray] = []
    rh_list: List[np.ndarray] = []

    peak_freq = float(DEFAULT_TRUE_PEAK_HZ)
    peak_sigma = 3.0
    if params.peak is not None:
        peak_freq = float(params.peak.get("freq", DEFAULT_TRUE_PEAK_HZ))
        peak_sigma = float(params.peak.get("sigma", 3.0))

    for amp in amp_grid:
        peak_dict = dict(freq=peak_freq, amplitude=float(amp), sigma=peak_sigma)

        knee_raw = float(params.knee_raw)

        res = spectrum(
            sampling_rate=FS,
            duration=WIN_DUR,
            aperiodic_exponent=params.exponent,
            aperiodic_offset=params.offset,
            knee=knee_raw,
            peaks=[peak_dict],
            average_firing_rate=0.0,
            random_state=int(rng_seed),
            direct_estimate=False,
            plot=False,
            mode="additive",
        )

        fd = res.frequency_domain
        f = np.asarray(fd.frequencies, float).ravel()
        full = np.asarray(fd.combined_spectrum, float).ravel()

        bb_attr = getattr(fd, "broadband_spectrum", None)
        if bb_attr is None:
            bb_attr = getattr(fd, "aperiodic_spectrum", None)
        if bb_attr is None:
            pk_attr = getattr(fd, "peaks_spectrum", None)
            if pk_attr is not None:
                pk = np.asarray(pk_attr, float).ravel()
                if pk.size != full.size:
                    pk = np.resize(pk, full.shape)
                bb = np.clip(full - pk, 0.0, np.inf)
            else:
                bb = full.copy()
        else:
            bb = np.asarray(bb_attr, float).ravel()

        rh_attr = getattr(fd, "rhythmic_spectrum", None)
        if rh_attr is None:
            rh_attr = getattr(fd, "peaks_spectrum", None)
        if rh_attr is None:
            rh = np.clip(full - bb, 0.0, np.inf)
        else:
            rh = np.asarray(rh_attr, float).ravel()

        if freqs_dense is None:
            freqs_dense = f
        else:
            if freqs_dense.shape != f.shape or not np.allclose(freqs_dense, f):
                raise RuntimeError("Frequency grids are inconsistent across GT calls.")

        bb_list.append(bb)
        rh_list.append(rh)

    freqs_dense = np.asarray(freqs_dense, float)
    mask = (freqs_dense >= ANALYSIS_FRANGE[0]) & (freqs_dense <= ANALYSIS_FRANGE[1])
    freqs_panel = freqs_dense[mask]
    bb_arr = np.asarray(bb_list)[:, mask]
    rh_arr = np.asarray(rh_list)[:, mask]
    return freqs_panel, bb_arr, rh_arr


# ---------------------- Figure building ----------------------
def make_figure_7(
    df: pd.DataFrame,
    row1_payload: Dict[str, Any],
    slope_true_val: float,
    out_dir: str,
    prefix: str = "Figure_7_Fig3GroundTruth",
    base_params: Optional[TrueParams] = None,
    seed0: int = 0,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    if base_params is None:
        base_params = TrueParams(peak=dict(freq=DEFAULT_TRUE_PEAK_HZ, amplitude=2.0, sigma=3.0))

    true_cf = float(DEFAULT_TRUE_PEAK_HZ)
    if base_params.peak is not None:
        true_cf = float(base_params.peak.get("freq", DEFAULT_TRUE_PEAK_HZ))
    true_knee_raw = float(base_params.knee_raw)
    true_knee = _knee_freq_hz_from_kappa(true_knee_raw, float(base_params.exponent))

    dfp = df.copy()
    for col in ("knee_true_hz", "knee_hz_est", "knee_est_raw"):
        if col not in dfp.columns:
            dfp[col] = np.nan

    dfp["method_display"] = dfp["method"].map(METHOD_LABELS)
    dfp["amp_val"] = pd.to_numeric(dfp["amp_true"], errors="coerce")
    dfp["amp_label"] = dfp["amp_val"].map(lambda v: f"{float(v):.3g}" if np.isfinite(v) else "nan")
    cat_amp_order = sorted([a for a in dfp["amp_label"].unique() if a != "nan"], key=lambda s: float(s))

    freqs_ref = np.asarray(row1_payload["freqs"], float)
    amp_vals  = np.asarray(row1_payload["amp_vals"], float)
    bb_arr    = np.asarray(row1_payload["bb"], float)
    rh_arr    = np.asarray(row1_payload["rh"], float)

    n_levels = amp_vals.size
    amp_palette = plt.cm.cubehelix(np.linspace(0.2, 0.9, n_levels))
    amp_color_map = {float(a): amp_palette[i] for i, a in enumerate(amp_vals)}

    slope_band_mask_ref = _band_mask(freqs_ref, SLOPE_BAND)

    spectra_ylim = (1e-6, 1e3)
    TRUTH_VLINE_KW = dict(color="0.2", lw=2.6, alpha=0.95, zorder=100)

    slope_vals = pd.to_numeric(dfp["slope_est"], errors="coerce").to_numpy()
    slope_vals = slope_vals[np.isfinite(slope_vals)]
    if np.isfinite(slope_true_val):
        slope_vals = np.concatenate([slope_vals, [slope_true_val]])

    if slope_vals.size > 0:
        y_min = float(np.min(slope_vals))
        y_max = float(np.max(slope_vals))
        pad = 0.10 * (y_max - y_min if y_max > y_min else 1.0)
        slope_ylim = (y_min - pad, y_max + pad)
    else:
        slope_ylim = None

    def _robust_xlim(arr: np.ndarray, pct=(0.5, 99.5), pad_frac=0.08) -> Optional[Tuple[float, float]]:
        arr = np.asarray(arr, float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return None
        lo, hi = np.percentile(arr, list(pct))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        pad = pad_frac * (hi - lo)
        return (lo - pad, hi + pad)

    ht_xlim   = _robust_xlim(pd.to_numeric(dfp["rh_height_est_log10"], errors="coerce"), pct=(0.5, 99.5), pad_frac=0.10)
    cf_xlim   = (4.0, 12.0)
    knee_xlim = _robust_xlim(
        pd.to_numeric(dfp["knee_hz_est"], errors="coerce"),
        pct=(0.5, 99.5),
        pad_frac=0.10,
    )

    # The knee truth marker is a vertical line. Make sure the robust x-limits
    # do not clip out the true knee. This prevents the Figure 3 implied-knee-Hz truth line from being clipped.
    if np.isfinite(true_knee):
        if knee_xlim is None:
            knee_xlim = (0.0, true_knee * 1.10)
        else:
            lo, hi = map(float, knee_xlim)
            span = hi - lo if hi > lo else max(abs(true_knee), 1.0)
            pad = 0.05 * span
            lo = min(lo, true_knee - pad)
            hi = max(hi, true_knee + pad)
            knee_xlim = (lo, hi)

    VIOLIN_GRAY = "0.70"
    STRIP_GRAY = "0.30"

    fig = plt.figure(figsize=(18, 20.0))
    gs = fig.add_gridspec(
        nrows=6, ncols=3,
        height_ratios=[1.25, 1.25, 1.05, 1.0, 1.0, 1.0],
        hspace=0.65, wspace=0.28,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.06, top=0.95)

    ax_gt_L = fig.add_subplot(gs[0, 0]); ax_gt_L.axis("off")
    ax_gt   = fig.add_subplot(gs[0, 1])
    ax_gt_R = fig.add_subplot(gs[0, 2]); ax_gt_R.axis("off")

    axs = np.empty((5, 3), dtype=object)
    for r in range(5):
        for c in range(3):
            axs[r, c] = fig.add_subplot(gs[r + 1, c])

    # Force panel A to have the same box aspect as the B–D spectra panels
    ref_pos = axs[0, 0].get_position()
    ax_gt.set_box_aspect(ref_pos.height / ref_pos.width)
    ax_gt.set_anchor("C")

    for ci, m in enumerate(PLOT_METHOD_ORDER):
        axs[0, ci].set_title(METHOD_LABELS[m], fontsize=12)

    gt_freqs, gt_bb_arr, gt_rh_arr = simulate_ground_truth_spectra_for_panel(
        params=base_params, amp_grid=amp_vals, rng_seed=int(seed0)
    )
    slope_band_mask_gt = _band_mask(gt_freqs, SLOPE_BAND)

    for li, amp in enumerate(amp_vals):
        col = amp_color_map[float(amp)]
        bb = gt_bb_arr[li]
        rh = gt_rh_arr[li]
        ax_gt.loglog(gt_freqs, bb, ls="--", lw=1.1, color=col, alpha=0.95)
        ax_gt.loglog(gt_freqs, rh, ls="-",  lw=1.5, color=col, alpha=0.95)
        if slope_band_mask_gt.sum() >= 2:
            ax_gt.loglog(
                gt_freqs[slope_band_mask_gt],
                bb[slope_band_mask_gt],
                ls="-", lw=1.0, color="red", alpha=0.9, zorder=6,
            )

    ax_gt.set_xlim(ANALYSIS_FRANGE)
    ax_gt.set_ylim(spectra_ylim)
    ax_gt.set_title("Ground truth", fontsize=12)
    ax_gt.set_ylabel("Power")
    ax_gt.set_xlabel("")
    ax_gt.minorticks_off()
    sns.despine(ax=ax_gt, top=True, right=True)

    # Row 1: spectra per method (MT-independent grid)
    for ci, method in enumerate(PLOT_METHOD_ORDER):
        ax = axs[0, ci]
        mi = METHOD_KEYS.index(method)

        legend_lines, legend_labels = [], []
        for li, amp in enumerate(amp_vals):
            col = amp_color_map[float(amp)]
            y_bb = bb_arr[mi, li, :]
            y_rh = rh_arr[mi, li, :]

            ax.loglog(freqs_ref, y_bb, ls="--", lw=1.2, color=col, alpha=0.95)
            line_rh, = ax.loglog(freqs_ref, y_rh, ls="-", lw=1.6, color=col, alpha=0.95)

            if ci == 0:
                legend_lines.append(line_rh)
                legend_labels.append(f"{amp:g}x")

            if slope_band_mask_ref.sum() >= 2:
                ax.loglog(
                    freqs_ref[slope_band_mask_ref],
                    y_bb[slope_band_mask_ref],
                    ls="-", lw=1.0, color="red", alpha=0.9, zorder=6,
                )

        ax.set_xlim(ANALYSIS_FRANGE)
        ax.set_ylim(spectra_ylim)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power" if ci == 0 else "")
        ax.minorticks_off()
        sns.despine(ax=ax, top=True, right=True)

        if ci == 0:
            ax.legend(
                legend_lines, legend_labels,
                title="Rhythmic height\n(amplitude)",
                loc="upper left",
                fontsize=8, title_fontsize=8,
                frameon=True,
            )
            ax.text(
                0.98, 0.06,
                "solid: rhythm\n dashed: BB\n red: 40–60 Hz BB",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9,
            )

    # Truth markers: mean of trial-level truth per amplitude
    tmp = dfp.copy()
    tmp["amp_true_num"] = pd.to_numeric(tmp["amp_true"], errors="coerce")
    true_height_by_amp = (
        tmp.groupby("amp_true_num", dropna=True)["rh_height_true_log10"]
        .mean()
        .to_dict()
    )

    # Row 2: violins
    VIOLIN_CUT = 4
    for ci, method in enumerate(PLOT_METHOD_ORDER):
        ax = axs[1, ci]
        sub = dfp[dfp["method"] == method].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sns.violinplot(
            data=sub,
            x="amp_label",
            y="slope_est",
            order=cat_amp_order,
            inner="quartile",
            cut=VIOLIN_CUT,
            bw="scott",
            linewidth=1.0,
            color=VIOLIN_GRAY,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="amp_label",
            y="slope_est",
            order=cat_amp_order,
            color=STRIP_GRAY,
            alpha=0.35,
            size=2.5,
            jitter=0.15,
            ax=ax,
        )
        if np.isfinite(slope_true_val):
            ax.axhline(slope_true_val, color="k", lw=2.0, alpha=0.9, zorder=5)

        ax.set_xlabel("Peak amplitude (linear units)")
        ax.set_ylabel("Broadband slope\n(40–60 Hz, log–log)" if ci == 0 else "")
        ax.tick_params(axis="x", rotation=25)
        if slope_ylim is not None:
            ax.set_ylim(slope_ylim)
        sns.despine(ax=ax, top=True, right=True)

        if ci == 1:
            ax.set_title("Broadband slope vs rhythmic height\n(sampling distribution)", fontsize=12, pad=10)

    def _scatter_by_amp(ax, sub, xcol, add_amp_legend=False):
        handles, labels = [], []
        for amp in sorted(sub["amp_val"].dropna().unique()):
            col = amp_color_map.get(float(amp), "0.5")
            sm = sub[sub["amp_val"] == amp]
            sc = ax.scatter(sm[xcol], sm["slope_est"], s=22, alpha=0.65, color=col, edgecolor="none")
            if add_amp_legend:
                handles.append(sc)
                labels.append(f"{amp:g}x")
        if add_amp_legend and handles:
            leg = ax.legend(handles, labels, title="Peak amplitude", loc="upper right",
                            fontsize=8, title_fontsize=8, frameon=True)
            leg.set_zorder(1000)
            leg.get_frame().set_alpha(0.9)

    # Row 3: slope vs rhythm height
    for ci, method in enumerate(PLOT_METHOD_ORDER):
        ax = axs[2, ci]
        sub = dfp[dfp["method"] == method].copy()
        if sub.empty:
            ax.axis("off")
            continue

        _scatter_by_amp(ax, sub, "rh_height_est_log10", add_amp_legend=(ci == 0))

        if np.isfinite(slope_true_val):
            ax.axhline(slope_true_val, color="k", lw=1.8, alpha=0.7, zorder=4)

        for amp in amp_vals:
            xtrue = true_height_by_amp.get(float(amp), np.nan)
            if np.isfinite(xtrue):
                ax.axvline(xtrue, **TRUTH_VLINE_KW)

        ax.set_xlabel("Rhythm height estimate (log10 power)")
        ax.set_ylabel("Broadband slope\n(40–60 Hz, log–log)" if ci == 0 else "")
        if slope_ylim is not None:
            ax.set_ylim(slope_ylim)
        if ht_xlim is not None:
            ax.set_xlim(ht_xlim)
        sns.despine(ax=ax, top=True, right=True)

    # Row 4: slope vs center frequency
    for ci, method in enumerate(PLOT_METHOD_ORDER):
        ax = axs[3, ci]
        sub = dfp[dfp["method"] == method].copy()
        if sub.empty:
            ax.axis("off")
            continue

        _scatter_by_amp(ax, sub, "rh_cf_est", add_amp_legend=(ci == 0))

        if np.isfinite(slope_true_val):
            ax.axhline(slope_true_val, color="k", lw=1.8, alpha=0.7, zorder=4)
        if slope_ylim is not None:
            ax.set_ylim(slope_ylim)

        ax.set_xlim(cf_xlim)
        ax.axvline(true_cf, **TRUTH_VLINE_KW)
        ax.set_xlabel("Center frequency estimate (Hz)")
        ax.set_ylabel("Broadband slope\n(40–60 Hz, log–log)" if ci == 0 else "")
        sns.despine(ax=ax, top=True, right=True)

    # Row 5: slope vs knee estimate (Hz)
    for ci, method in enumerate(PLOT_METHOD_ORDER):
        ax = axs[4, ci]
        sub = dfp[dfp["method"] == method].copy()
        if sub.empty:
            ax.axis("off")
            continue

        _scatter_by_amp(ax, sub, "knee_hz_est", add_amp_legend=(ci == 0))

        ax.axvline(true_knee, **TRUTH_VLINE_KW)
        if np.isfinite(slope_true_val):
            ax.axhline(slope_true_val, color="k", lw=1.8, alpha=0.7, zorder=4)

        if slope_ylim is not None:
            ax.set_ylim(slope_ylim)
        if knee_xlim is not None:
            ax.set_xlim(knee_xlim)

        ax.set_xlabel("Knee estimate (Hz; raw kappa^(1/χ))")
        if ci == 0:
            ax.set_ylabel("Broadband slope\n(40–60 Hz, log–log)")
        sns.despine(ax=ax, top=True, right=True)

    out_png = os.path.join(out_dir, f"{prefix}.png")
    out_svg = os.path.join(out_dir, f"{prefix}.svg")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_png}")
    print(f"[saved] {out_svg}")
    return out_png, out_svg


# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-iter-per-level", type=int, default=30)
    ap.add_argument("--n-levels", type=int, default=4, help="Ignored; grid fixed to 4 levels.")
    ap.add_argument("--amp-min", type=float, default=0.5, help="Ignored; grid fixed.")
    ap.add_argument("--amp-max", type=float, default=4.0, help="Ignored; grid fixed.")
    ap.add_argument("--aper-exp", type=float, default=2.0)
    ap.add_argument("--aper-off", type=float, default=0.5)
    ap.add_argument("--knee", type=float, default=60.0, help="Raw knee passed directly to SL_GPsim.spectrum(knee=...), matching Figure 3.")
    ap.add_argument("--peak", type=float, default=DEFAULT_TRUE_PEAK_HZ, help="Peak center frequency (Hz).")
    ap.add_argument("--sigma", type=float, default=3.0, help="Peak sigma (Hz).")
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--metrics-csv", type=str, default=None)
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n_iter_per_level = int(args.n_iter_per_level)
    if args.quick:
        n_iter_per_level = min(n_iter_per_level, 2)

    base_params = TrueParams(
        exponent=float(args.aper_exp),
        offset=float(args.aper_off),
        knee_raw=float(args.knee),
        peak=dict(
            freq=float(args.peak),
            amplitude=2.0,
            sigma=float(args.sigma),
        ),
    )

    metrics_csv = os.path.expanduser(args.metrics_csv) if args.metrics_csv else os.path.join(out_dir, "Figure_7_fig3truth_metrics.csv")
    row1_npz = os.path.join(out_dir, "Figure_7_fig3truth_row1_components.npz")

    df: Optional[pd.DataFrame] = None
    row1_payload: Optional[Dict[str, Any]] = None

    use_cache = (not args.force_recompute) and os.path.exists(metrics_csv) and os.path.exists(row1_npz)

    if use_cache:
        try:
            df_cached = pd.read_csv(metrics_csv)
            raw = np.load(row1_npz)
            if "bb_true" in raw.files and "rh_true" in raw.files:
                df = df_cached
                row1_payload = dict(
                    freqs=np.asarray(raw["freqs"], float),
                    amp_vals=np.asarray(raw["amp_vals"], float),
                    bb=np.asarray(raw["bb"], float),
                    rh=np.asarray(raw["rh"], float),
                    bb_true=np.asarray(raw["bb_true"], float),
                    rh_true=np.asarray(raw["rh_true"], float),
                )
                print(f"[loaded metrics] {metrics_csv}")
                print(f"[loaded row-1 payload] {row1_npz}")
            else:
                print("[info] Row-1 cache is older; recomputing.")
        except Exception as exc:
            print(f"[warning] Failed to load cache ({exc}); recomputing.")

    if df is None or row1_payload is None:
        print(
            "[priors] SL_SD b prior: Normal(log10 median PSD, sigma=5.0); "
            "additive A_lin_0 prior anchored to rhythm-band q=50; Figure 3 prior dict and k_tapers=3."
        )
        print(
            f"[rng] Initializing one SeedSequence from seed0={int(args.seed0)}; "
            "using spawned persistent streams for simulation, SL_SD additive, and SL_SD multiplicative."
        )
        df, row1_payload = run_height_grid(
            params_base=base_params,
            n_levels=4,
            n_iter_per_level=n_iter_per_level,
            amp_min=float(args.amp_min),
            amp_max=float(args.amp_max),
            seed0=int(args.seed0),
        )
        df.to_csv(metrics_csv, index=False)
        print(f"[saved metrics] {metrics_csv}")

        np.savez(
            row1_npz,
            freqs=row1_payload["freqs"],
            amp_vals=row1_payload["amp_vals"],
            bb=row1_payload["bb"],
            rh=row1_payload["rh"],
            bb_true=row1_payload["bb_true"],
            rh_true=row1_payload["rh_true"],
        )
        print(f"[saved row-1 payload] {row1_npz}")

    if df is None or row1_payload is None:
        raise RuntimeError("Metrics and/or Row-1 payload not available.")

    # Ground-truth slope line: mean of trial-level slope_true (computed on f_ref)
    # Ground-truth plotting values recovered directly from cached GT curves.
    # This does NOT rerun MCMC or simulations.
    true_cf_for_cache = float(base_params.peak.get("freq", DEFAULT_TRUE_PEAK_HZ))
    truth_cache = _truth_values_from_cached_ground_truth(
        row1_payload=row1_payload,
        true_cf=true_cf_for_cache,
    )

    # Correct horizontal line: true broadband slope from cached bb_true.
    slope_true_val = float(truth_cache["bb_slope_true"])

    # Also overwrite the truth columns used for vertical rhythm-height markers,
    # so old/stale CSV truth values cannot leak into the plot.
    df = df.copy()
    df["slope_true"] = slope_true_val

    rh_height_map = truth_cache["rh_height_true_log10_by_amp"]
    df["amp_true_num_for_truth"] = pd.to_numeric(df["amp_true"], errors="coerce")
    df["rh_height_true_log10"] = df["amp_true_num_for_truth"].map(
        lambda a: rh_height_map.get(float(a), np.nan) if np.isfinite(a) else np.nan
    )

    print(f"[truth repair] Broadband slope truth from cached bb_true: {slope_true_val:.6f}")
    print("[truth repair] Rhythm-height truth markers recovered from cached bb_true + rh_true.")

    make_figure_7(
        df=df,
        row1_payload=row1_payload,
        slope_true_val=slope_true_val,
        out_dir=out_dir,
        base_params=base_params,
        seed0=int(args.seed0),
    )


if __name__ == "__main__":
    main()
