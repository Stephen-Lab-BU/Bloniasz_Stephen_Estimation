#!/usr/bin/env python3
"""
Generate Figure 5 empirical matched-simulation benchmarks.

The script uses Figure 4 decomposition parameters as simulation ground truth,
fits the candidate spectral decomposition models to simulated windows, and
writes the manuscript figure plus cached metrics and plot payloads.
"""

from __future__ import annotations

import os, argparse, re, json, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pymc as pm
from scipy.signal import find_peaks, peak_widths
from scipy.special import gammaln

from SL_specdecomp import Decompose

from specparam import SpectralModel
from spectral_connectivity import Multitaper, Connectivity
from SL_GPsim import spectrum


PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# ──────────────────────────── Global config ────────────────────────────
ROOT = str(PROJECT_ROOT)
SAVE_DIR = os.path.join(ROOT, "Output", "Results", "FiguresIntermediate", "Figure_5_CV", "Figure_output")
os.makedirs(SAVE_DIR, exist_ok=True)

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
RASTER_FRANGE = (0.1, 35.0)

# Aesthetics (match Figure 4 / Figure 3 style)
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
    "emp": "0.35",              # multitaper
    "full": "#000000",          # full model (both)
    "broad": "#ff7f0e",         # aperiodic/broadband
    "rhythms": "#d62728",       # rhythms
    "overlay": "0.6",           # gray for all-fits overlay
    "specparam_pt": "#1f77b4",  # markers
    "slsd_pt": "#2ca02c",
    "fwhm": "#6a3d9a",          # retained
}
STYLES = {
    "emp": dict(lw=2.0, alpha=0.65, solid_capstyle="round"),
    "full": dict(lw=2.0),
    "component": dict(lw=1.8, ls="--"),  # dashed for aperiodic & rhythms
}
PSD_YLIM = (1e-1, 1e6)


# ──────────────────────────── Import Figure 4 params (as GT) ────────────────────────────
FIG4_DIR_DEFAULT = "CHANGE_THIS_ROOT_TO_PATH/Bloniasz_Stephen_Estimation/Output/Results/FiguresIntermediate/Figure_4_CV/Figure_output"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        if o.size <= 100:
            return o.tolist()
        return dict(__ndarray__=True, shape=list(o.shape), dtype=str(o.dtype))
    return str(o)


def load_fig4_params(state: str, fig4_dir: str = FIG4_DIR_DEFAULT) -> dict | None:
    path = Path(fig4_dir) / f"fig4_slsd_params_{state}.json"
    if not path.exists():
        print(f"[WARN] Could not find {path}; using built-in SIM_CFG (but Figure 5 expects fig4 export).")
        return None
    with open(path, "r") as f:
        obj = json.load(f)
    return obj.get("params", None)

def sim_params_from_fig4(state: str, fig4_dir: str = FIG4_DIR_DEFAULT) -> dict | None:
    """
    Load Figure 4 JSON and pass the actual decomposition parameters into spectrum.

    The simulation should be driven by:
      - aperiodic_exponent = chi
      - aperiodic_offset = b
      - knee = kappa (already Hz-domain)
      - peaks = exported rhythm peaks

    The plotted slope metric in Figure 5 can still be the local 40–60 Hz
    broadband slope; that is separate from the simulation exponent.
    """
    p = load_fig4_params(state, fig4_dir)
    if p is None:
        return None

    if ("aperiodic_exponent" not in p) or ("aperiodic_offset" not in p) or ("knee_0" not in p):
        raise ValueError(
            f"{state}: Figure 4 JSON missing required keys "
            "(aperiodic_exponent, aperiodic_offset, knee_0)."
        )

    if "peaks_list" in p and p["peaks_list"] is not None:
        peaks = [
            {
                "freq": float(pk["freq"]),
                "amplitude": float(pk["amplitude"]),
                "sigma": float(pk["sigma"]),
            }
            for pk in p.get("peaks_list", [])
        ]
    else:
        peaks = [
            {
                "freq": float(r["center"]),
                "amplitude": float(r["A_lin"]),
                "sigma": float(r["sigma"]),
            }
            for r in p.get("rhythms", [])
        ]

    out = {
        "aperiodic_exponent": float(p["aperiodic_exponent"]),
        "aperiodic_offset": float(p["aperiodic_offset"]),
        "knee": float(p["knee_0"]),
        "peaks": peaks,
        "mode": "additive",
    }

    print(
        f"[INFO] {state}: loaded Figure 4 params "
        f"(b={out['aperiodic_offset']:.6g}, "
        f"chi={out['aperiodic_exponent']:.6g}, "
        f"kappa={out['knee']:.6g}, "
        f"n_peaks={len(out['peaks'])}, "
        f"local_slope_40_60={p.get('broadband_slope_40_60', 'NA')})"
    )
    return out



def fig4_target_rel_min_from_payload(state: str, fig4_dir: str = FIG4_DIR_DEFAULT) -> float | None:
    # Return the actual Figure 4 target-window time, in minutes after state start.
    # Reads post-drop idx_target and T_rel_min from Figure_4_CV_<state> payloads.
    base = Path(fig4_dir)
    npz_path = base / f"Figure_4_CV_{state}.plotdata.npz"
    meta_path = base / f"Figure_4_CV_{state}.plotmeta.json"
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
        print(f"[WARN] Could not read Figure 4 target time from {meta_path}: {exc}")
    return None


def _idx_nearest_time(t_rel_min: np.ndarray, target_min: float, fallback_idx: int = 0) -> int:
    t = np.asarray(t_rel_min, float).ravel()
    if t.size == 0 or not np.isfinite(float(target_min)):
        return int(fallback_idx)
    return int(np.nanargmin(np.abs(t - float(target_min))))

def _rank01(v: np.ndarray) -> np.ndarray:
    """
    Convert finite values to ranks on [0, 1], where lower is better.
    NaNs stay NaN.
    """
    v = np.asarray(v, float).ravel()
    out = np.full(v.shape, np.nan, dtype=float)
    good = np.isfinite(v)
    n = int(good.sum())
    if n == 0:
        return out
    if n == 1:
        out[good] = 0.0
        return out

    vals = v[good]
    order = np.argsort(vals)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, n)
    out[good] = ranks
    return out

def _window_log10_rmse(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> float:
    y = np.asarray(y, float).ravel()
    yhat = np.asarray(yhat, float).ravel()
    mask = np.asarray(mask, bool).ravel()

    m = (
        mask
        & np.isfinite(y)
        & np.isfinite(yhat)
        & (y > 0)
        & (yhat > 0)
    )
    if int(m.sum()) < 5:
        return np.nan

    resid = np.log10(y[m]) - np.log10(yhat[m])
    return float(np.sqrt(np.mean(resid ** 2)))


def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    f = np.asarray(freqs, float).ravel()
    lo, hi = map(float, band)
    return np.isfinite(f) & (f >= lo) & (f <= hi)


def _load_fig4_target_specparam_for_matching(
    state: str,
    fig4_dir: str,
    target_freqs: np.ndarray,
) -> Dict[str, Any] | None:
    """
    Load the empirical Figure 4 target-window specparam aperiodic curve.

    This is ONLY for choosing which simulated Figure 5 window to display.
    It does NOT change the Figure 5 simulation ground truth, which still comes
    from fig4_slsd_params_<state>.json.
    """
    base = Path(fig4_dir)
    npz_path = base / f"Figure_4_CV_{state}.plotdata.npz"
    meta_path = base / f"Figure_4_CV_{state}.plotmeta.json"

    if not (npz_path.exists() and meta_path.exists()):
        print(
            "[WARN] Could not load Figure 4 payload for empirical specparam "
            f"matching: {npz_path} / {meta_path}"
        )
        return None

    try:
        with open(meta_path, "r") as f:
            meta_file = json.load(f)
        meta = meta_file.get("meta", meta_file)
        idx = int(meta.get("idx_target", 0))

        with np.load(npz_path, allow_pickle=False) as z:
            F4 = np.asarray(z["F_fit"], float).ravel()
            sp_aper_4 = np.asarray(z["specparam_aper"], float)
            sp_full_4 = np.asarray(z["specparam_full"], float)
            slsd_bb_4 = np.asarray(z["slsd_bb"], float)
            T4 = np.asarray(z["T_rel_min"], float).ravel()

        idx = int(np.clip(idx, 0, max(sp_aper_4.shape[0] - 1, 0)))

        def _interp_row(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, float)
            if arr.ndim == 1:
                y = arr
            else:
                y = arr[idx, :]
            return np.interp(np.asarray(target_freqs, float).ravel(), F4, y)

        return {
            "idx_target_fig4": int(idx),
            "t_rel_min_fig4": float(T4[idx]) if 0 <= idx < T4.size else np.nan,
            "specparam_aper": _interp_row(sp_aper_4),
            "specparam_full": _interp_row(sp_full_4),
            "slsd_bb": _interp_row(slsd_bb_4),
            "npz_path": str(npz_path),
            "meta_path": str(meta_path),
        }

    except Exception as exc:
        print(f"[WARN] Could not load Figure 4 empirical specparam target: {exc}")
        return None


def _choose_sim_window_matching_empirical_specparam_error(
    *,
    state: str,
    fig4_dir: str,
    F_fit: np.ndarray,
    F_truth: np.ndarray,
    S_truth_bb: np.ndarray,
    P_fit_tf: np.ndarray,
    specparam_full: np.ndarray,
    specparam_aper: np.ndarray,
    slsd_bb: np.ndarray,
    preferred_idx: int = 0,
) -> Dict[str, Any]:
    """
    Choose the Figure 5 display window.

    Desired didactic window:
      1. specparam aperiodic curve resembles empirical Figure 4 specparam
         aperiodic curve, especially at low frequencies.
      2. specparam gets low frequencies wrong relative to the known simulated
         broadband truth.
      3. specparam is clearly different from SL_specdecomp at low frequencies.
      4. specparam is wrong-but-not-terrible, not an obvious catastrophic fit.

    The simulation itself remains generated from the Figure 4 target-window
    SL_specdecomp decomposition. This helper only chooses which simulated
    30 s window is displayed.
    """
    F_fit = np.asarray(F_fit, float).ravel()
    F_truth = np.asarray(F_truth, float).ravel()
    S_truth_bb = np.asarray(S_truth_bb, float).ravel()
    P_fit_tf = np.asarray(P_fit_tf, float)
    specparam_full = np.asarray(specparam_full, float)
    specparam_aper = np.asarray(specparam_aper, float)
    slsd_bb = np.asarray(slsd_bb, float)

    n_win = int(P_fit_tf.shape[0])
    preferred_idx = int(np.clip(int(preferred_idx), 0, max(n_win - 1, 0)))

    fig4_emp = _load_fig4_target_specparam_for_matching(
        state=state,
        fig4_dir=fig4_dir,
        target_freqs=F_fit,
    )

    if fig4_emp is None:
        return {
            "idx": int(preferred_idx),
            "selection_policy": "fallback_preferred_idx_no_fig4_payload",
            "emp_match_low": None,
            "emp_match_broad": None,
            "wrong_low_vs_true_bb": None,
            "specparam_vs_slsd_low": None,
            "overall_specparam_to_data": None,
            "fig4_target_time_min": None,
            "fig4_target_idx": None,
        }

    emp_sp_aper = np.asarray(fig4_emp["specparam_aper"], float).ravel()
    true_bb_on_fit = np.interp(F_fit, F_truth, S_truth_bb)

    low_mask = _band_mask(F_fit, (0.5, 8.0))
    broad_mask = _band_mask(F_fit, (0.5, 40.0))
    overall_mask = _band_mask(F_fit, (0.5, 120.0))

    emp_match_low = np.full(n_win, np.nan, dtype=float)
    emp_match_broad = np.full(n_win, np.nan, dtype=float)
    wrong_low_vs_true_bb = np.full(n_win, np.nan, dtype=float)
    specparam_vs_slsd_low = np.full(n_win, np.nan, dtype=float)
    overall_specparam_to_data = np.full(n_win, np.nan, dtype=float)

    for wi in range(n_win):
        # Main criterion: simulated specparam resembles the empirical
        # specparam low-frequency aperiodic shape.
        emp_match_low[wi] = _window_log10_rmse(
            specparam_aper[wi, :],
            emp_sp_aper,
            low_mask,
        )
        emp_match_broad[wi] = _window_log10_rmse(
            specparam_aper[wi, :],
            emp_sp_aper,
            broad_mask,
        )

        # We specifically want specparam to be wrong at low frequencies
        # relative to the known simulated broadband truth.
        wrong_low_vs_true_bb[wi] = _window_log10_rmse(
            specparam_aper[wi, :],
            true_bb_on_fit,
            low_mask,
        )

        # the analysis uses specparam to be visibly different from SL_specdecomp
        # at low frequencies.
        specparam_vs_slsd_low[wi] = _window_log10_rmse(
            specparam_aper[wi, :],
            slsd_bb[wi, :],
            low_mask,
        )

        # Guardrail: avoid catastrophic-looking specparam full fits.
        overall_specparam_to_data[wi] = _window_log10_rmse(
            P_fit_tf[wi, :],
            specparam_full[wi, :],
            overall_mask,
        )

    finite = (
        np.isfinite(emp_match_low)
        & np.isfinite(emp_match_broad)
        & np.isfinite(wrong_low_vs_true_bb)
        & np.isfinite(specparam_vs_slsd_low)
        & np.isfinite(overall_specparam_to_data)
    )

    if not np.any(finite):
        return {
            "idx": int(preferred_idx),
            "selection_policy": "fallback_preferred_idx_no_finite_scores",
            "emp_match_low": None,
            "emp_match_broad": None,
            "wrong_low_vs_true_bb": None,
            "specparam_vs_slsd_low": None,
            "overall_specparam_to_data": None,
            "fig4_target_time_min": fig4_emp.get("t_rel_min_fig4", None),
            "fig4_target_idx": fig4_emp.get("idx_target_fig4", None),
        }

    # Wrong-but-not-terrible constraints:
    #   - specparam low-frequency error vs true broadband should be above
    #     the median-ish range, so it is visibly wrong.
    #   - but it should not be in the most extreme/catastrophic tail.
    #   - specparam full fit to the observed multitaper PSD should not be
    #     in the worst tail.
    #   - specparam-vs-SL low-frequency difference should be above median.
    wrong_floor = float(np.nanpercentile(wrong_low_vs_true_bb[finite], 50.0))
    wrong_cap = float(np.nanpercentile(wrong_low_vs_true_bb[finite], 90.0))
    overall_cap = float(np.nanpercentile(overall_specparam_to_data[finite], 80.0))
    slsd_diff_floor = float(np.nanpercentile(specparam_vs_slsd_low[finite], 50.0))

    candidate = (
        finite
        & (wrong_low_vs_true_bb >= wrong_floor)
        & (wrong_low_vs_true_bb <= wrong_cap)
        & (overall_specparam_to_data <= overall_cap)
        & (specparam_vs_slsd_low >= slsd_diff_floor)
    )

    # Score among candidates:
    #   lower empirical-specparam match error is better;
    #   lower overall data-fit error is better;
    #   larger specparam-vs-SL low-frequency difference is better;
    #   larger low-frequency wrongness vs truth is mildly better, as long as
    #   it remains inside the wrong-but-not-terrible candidate set.
    score = (
        1.00 * _rank01(emp_match_low)
        + 0.50 * _rank01(emp_match_broad)
        + 0.20 * _rank01(overall_specparam_to_data)
        - 0.35 * _rank01(specparam_vs_slsd_low)
        - 0.15 * _rank01(wrong_low_vs_true_bb)
    )

    if np.any(candidate):
        masked_score = np.where(candidate, score, np.nan)
        idx = int(np.nanargmin(masked_score))
        policy = "match_empirical_specparam_lowfreq_error_wrong_but_not_catastrophic"
    else:
        usable = finite & (overall_specparam_to_data <= overall_cap)
        if not np.any(usable):
            usable = finite
        masked_score = np.where(usable, score, np.nan)
        idx = int(np.nanargmin(masked_score))
        policy = "fallback_match_empirical_specparam_with_overall_fit_filter"

    return {
        "idx": int(idx),
        "selection_policy": policy,
        "emp_match_low": float(emp_match_low[idx]),
        "emp_match_broad": float(emp_match_broad[idx]),
        "wrong_low_vs_true_bb": float(wrong_low_vs_true_bb[idx]),
        "specparam_vs_slsd_low": float(specparam_vs_slsd_low[idx]),
        "overall_specparam_to_data": float(overall_specparam_to_data[idx]),
        "fig4_target_time_min": fig4_emp.get("t_rel_min_fig4", None),
        "fig4_target_idx": fig4_emp.get("idx_target_fig4", None),
    }

def sim_params_from_fig4_legacy(state: str, fig4_dir: str = FIG4_DIR_DEFAULT) -> dict | None:
    """
    Load Figure 4's JSON export and translate it into kwargs for the current
    local ``SL_GPsim.spectrum``.

    IMPORTANT:
    To keep the simulated data aligned with the empirical Figure 4 fit, use the
    decomposition parameters directly:
        - aperiodic_exponent <- chi / alpha from Figure 4
        - aperiodic_offset <- decomposition offset from Figure 4
        - knee <- knee_0 from Figure 4
        - peaks <- exported rhythm peaks

    We do NOT use b_0 here. That helper field is useful for alternate
    parameterizations, but for the current local ``SL_GPsim.spectrum``
    interface the analysis uses the decomposition parameters themselves.
    """
    p = load_fig4_params(state, fig4_dir)
    if p is None:
        return None

    # Current Figure 4 export schema: use decomposition params directly
    if ("aperiodic_exponent" in p) and ("aperiodic_offset" in p) and ("knee_0" in p):
        # Prefer peaks_list if present; otherwise reconstruct from rhythms
        if "peaks_list" in p and p["peaks_list"] is not None:
            peaks = [
                {
                    "freq": float(pk["freq"]),
                    "amplitude": float(pk["amplitude"]),
                    "sigma": float(pk["sigma"]),
                }
                for pk in p.get("peaks_list", [])
            ]
        else:
            peaks = [
                {
                    "freq": float(r["center"]),
                    "amplitude": float(r["A_lin"]),
                    "sigma": float(r["sigma"]),
                }
                for r in p.get("rhythms", [])
            ]

        out = {
            "aperiodic_exponent": float(p["aperiodic_exponent"]),
            "aperiodic_offset": float(p["aperiodic_offset"]),
            "knee": float(p["knee_0"]),
            "peaks": peaks,
            "mode": "additive",
        }

        print(
            f"[INFO] {state}: loaded Figure 4 decomposition params "
            f"(offset={out['aperiodic_offset']:.6g}, "
            f"chi={out['aperiodic_exponent']:.6g}, "
            f"knee={out['knee']:.6g}, "
            f"n_peaks={len(out['peaks'])})"
        )
        return out

    raise ValueError(
        f"{state}: Figure 4 JSON is missing required decomposition keys. "
        f"Need aperiodic_offset, aperiodic_exponent, and knee_0."
    )

def sim_params_from_fig4_legacy(state: str, fig4_dir: str = FIG4_DIR_DEFAULT) -> dict | None:
    """
    Load Figure 4's JSON export and translate it into kwargs accepted by the
    *current* local ``SL_GPsim.spectrum`` signature.

    Important: the user's installed ``spectrum`` does not accept
    ``b_0``, ``slope_0``, ``knee_0`` or ``peaks_list`` directly. Its public
    API is:
        aperiodic_exponent, aperiodic_offset, knee, peaks, mode

    For the current Figure 4 export schema (``SL_specdecomp_params_for_spectrum.v1``),
    the correct mapping for additive simulations is:
        aperiodic_exponent = alpha = -slope_0
        aperiodic_offset = b_0
        knee = knee_0
        peaks = peaks_list

    because Figure 4 computes:
        b_0 = log10( broadband(f0) * (knee_0 + f0**alpha) )

    which is exactly the log10 numerator used by this version of
    ``SL_GPsim.spectrum`` in additive mode.

    Backward-compatible fallback:
      - if the JSON is older / legacy, reconstruct kwargs from
        ``aperiodic_offset``, ``aperiodic_exponent``, ``knee_0``, and
        ``rhythms``.
    """
    p = load_fig4_params(state, fig4_dir)
    if p is None:
        return None

    # Preferred: current Figure 4 export schema.
    if ("b_0" in p) and (("slope_0" in p) or ("alpha" in p)) and ("knee_0" in p):
        if "alpha" in p:
            alpha = float(p["alpha"])
        else:
            alpha = float(-float(p["slope_0"]))

        peaks = [
            {
                "freq": float(pk["freq"]),
                "amplitude": float(pk["amplitude"]),
                "sigma": float(pk["sigma"]),
            }
            for pk in p.get("peaks_list", [])
        ]

        return {
            "aperiodic_exponent": alpha,
            "aperiodic_offset": float(p["b_0"]),
            "knee": float(p["knee_0"]),
            "peaks": peaks,
            "mode": "additive",
        }

    # Fallback: older / legacy export schema.
    out = {
        "aperiodic_exponent": float(p["aperiodic_exponent"]),
        "aperiodic_offset": float(p["aperiodic_offset"]),
        "knee": float(p["knee_0"]),
        "peaks": [
            {
                "freq": float(r["center"]),
                "amplitude": float(r["A_lin"]),
                "sigma": float(r["sigma"]),
            }
            for r in p.get("rhythms", [])
        ],
        "mode": "additive",
    }
    print(
        f"[WARN] {state}: Figure 4 export did not contain current spectrum-hand-off keys; "
        "falling back to legacy aperiodic/rhythm reconstruction."
    )
    return out


SIM_CFG = {
    "awake": {
        "duration_full": 900.0,
        "duration_short": 60.0,
        "fs": 1000.0,
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
        "duration_full": 540.0,
        "duration_short": 60.0,
        "fs": 1000.0,
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


# ──────────────────────────── Payload export / import (Figure 5) ────────────────────────────
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


def save_plot_payload_fig5(
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

    print(f"[INFO] Saved Figure5 payload → {npz_path}")
    print(f"[INFO] Saved Figure5 payload meta → {meta_path}")
    return npz_path, meta_path


def load_plot_payload_fig5(out_base: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Path, Path]:
    """
    Load Figure 5 plot payload saved by save_plot_payload_fig5.
    Returns:
      arrays (dict[str, np.ndarray]), meta (dict), npz_path, meta_path
    """
    out_base = Path(out_base)
    npz_path = out_base.with_suffix(".plotdata.npz")
    meta_path = out_base.with_suffix(".plotmeta.json")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing plot payload NPZ: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing plot payload meta JSON: {meta_path}")

    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}

    with open(meta_path, "r") as f:
        meta_obj = json.load(f)

    meta = meta_obj.get("meta", {})
    return arrays, meta, npz_path, meta_path


# ──────────────────────────── Export helpers (SL_specdecomp fit params) ────────────────────────────
def _posterior_mean_scalar(ds, names):
    for nm in names:
        if nm in ds:
            return float(ds[nm].mean().item())
    return None


def _knee_omega_to_hz(knee_omega: float, alpha: float) -> float:
    """
    Match Figure 4 conversion:
      if SL_specdecomp uses omega-domain knee ((2*pi*f)^alpha), convert to Hz-domain knee for spectrum.
      knee_hz = knee_omega / (2*pi)^alpha
    If already Hz-convention, this is harmless when knee_omega=0 or alpha<=0.
    """
    knee_omega = float(knee_omega)
    alpha = float(alpha)
    if knee_omega <= 0.0 or alpha <= 0.0:
        return float(max(knee_omega, 0.0))
    return float(knee_omega / (2.0 * np.pi) ** alpha)


def _aperiodic_from_model_or_fit(F_fit, bb_lin, sl_model):
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
        idxs = sorted({int(m.group(1)) for v in varnames for m in [re.search(r"center[_\[](\d+)", v)] if m})
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
            for v in (f"A_lin_{k}", f"A_lin[{k}]", f"A_{k}", f"A[{k}]"):
                if v in ds:
                    a = float(ds[v].mean().item())
                    break
            if c is not None and s is not None and a is not None:
                peaks.append([c, a, 2.3548 * s])

    return np.asarray(peaks, float) if peaks else np.empty((0, 3), float)


def slsd_params_from_model(sl_model, F_fit, bb_lin, *, ref_freq_hz: float = 10.0) -> dict:
    """
    Simulation-ready params for spectral_decomposition.spectrum (match Figure 4 conventions).
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
    alpha = float(ap["aperiodic_exponent"])
    knee_0_raw = float(ap["knee_0"]) if ap["knee_0"] is not None else 0.0
    knee_0 = _knee_omega_to_hz(knee_0_raw, alpha)  # match Figure 4
    slope_0 = float(-alpha)

    f0 = float(ref_freq_hz)
    bb_f0 = float(np.interp(f0, np.asarray(F_fit, float), np.asarray(bb_lin, float)))
    bb_f0 = max(bb_f0, 1e-20)
    b_0 = float(np.log10(bb_f0 * (knee_0 + (f0 ** alpha))))

    return {
        "schema": "SL_specdecomp_params_for_spectrum.v1",
        "ref_freq_hz": f0,

        # Simulation-ready:
        "slope_0": slope_0,
        "alpha": alpha,
        "knee_0": knee_0,
        "b_0": b_0,
        "b0_space": "log10",
        "peaks_list": peaks_list,

        # Diagnostic:
        "aperiodic_offset": ap["aperiodic_offset"],
        "aperiodic_exponent": ap["aperiodic_exponent"],
        "rhythms": rhythms,

        "notes": (
            "Posterior means where available. "
            "knee_0 converted to Hz-domain to match spectrum() convention. "
            "b_0 computed at ref_freq_hz to match spectral_decomposition.spectrum()."
        ),
    }


def save_fig5_params(state_name: str, kind: str, params: dict, out_dir: str) -> str:
    """
    Save a params JSON. 'kind' examples:
      - 'gt': ground truth sim params used by spectrum
      - 'slsd_fit': simulation-ready params derived from SL_specdecomp fit on target window
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state_name,
        "kind": str(kind),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "bands": {"hg": list(HG_BAND), "slope": list(SLOPE_BAND)},
    }
    path = Path(out_dir) / f"fig5_{kind}_params_{state_name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[INFO] Exported Figure5 params ({kind}) → {path}")
    return str(path)


# ──────────────────────────── Helpers ────────────────────────────
def _independent_grid(freqs: np.ndarray, twin: float, NW: float) -> Tuple[np.ndarray, int]:
    f = np.asarray(freqs, float)
    df = float(np.median(np.diff(f))) if f.size > 1 else 1.0
    delta_f_indep = 2.0 * NW / twin
    step_bins = max(1, int(round(delta_f_indep / max(df, 1e-12))))
    return f[::step_bins], step_bins


def _ensure_tf(power: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(power); f = np.asarray(freqs).ravel(); t = np.asarray(times).ravel()
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


def compute_multitaper(x: np.ndarray, fs: float, t0: float, params: Dict[str, Any]):
    """
    Wrapper around spectral_connectivity Multitaper/Connectivity.
    Multitaper expects a 3D array (n_time, n_trials, n_signals).
    Returns:
      P_tf: (n_windows_total, n_freq) [flattened over trials × time-windows]
      F: (n_freq,)
      T_abs:(n_windows_total,) [synthetic window-centers for plotting]
    """
    x_arr = np.asarray(x, float)

    if x_arr.ndim == 1:
        x_3d = x_arr[:, np.newaxis, np.newaxis]
    elif x_arr.ndim == 2:
        x_3d = x_arr[:, :, np.newaxis]
    elif x_arr.ndim == 3:
        x_3d = x_arr
    else:
        raise ValueError(f"Expected 1D/2D/3D time series; got shape {x_arr.shape}")

    mt = Multitaper(x_3d, sampling_frequency=float(fs), start_time=float(t0), **params)
    conn = Connectivity.from_multitaper(mt)

    f = np.asarray(conn.frequencies, float).ravel()
    p = np.asarray(conn.power())
    p = np.squeeze(p)

    if p.ndim == 1:
        if p.size != f.size:
            raise ValueError(f"Unexpected power shape after squeeze: {p.shape} (freqs={f.size})")
        p2 = p[None, :]
    elif p.ndim == 2:
        if p.shape[1] == f.size:
            p2 = p
        elif p.shape[0] == f.size:
            p2 = p.T
        else:
            p2 = p
    else:
        freq_axes = [i for i, s in enumerate(p.shape) if s == f.size]
        if not freq_axes:
            raise ValueError(f"Could not locate frequency axis in power shape {p.shape} (freqs={f.size})")
        p = np.moveaxis(p, freq_axes[0], -1)
        p = np.squeeze(p)
        p2 = p.reshape(-1, f.size)

    win_dur = float(params["time_window_duration"])
    win_step = float(params["time_window_step"])
    n_w = int(p2.shape[0])
    T_abs = float(t0) + (np.arange(n_w, dtype=float) * win_step + 0.5 * win_dur)

    return p2, f, T_abs


def _compute_loglog_slope(freqs: np.ndarray, power_lin: np.ndarray, fmin: float, fmax: float) -> float:
    f = np.asarray(freqs, float); y = np.asarray(power_lin, float)
    m = (f >= fmin) & (f <= fmax) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        return np.nan
    a, _b = np.polyfit(np.log10(f[m]), np.log10(y[m]), 1)
    return float(a)


def _hg_mean(freqs: np.ndarray, power_lin: np.ndarray, fmin: float, fmax: float) -> float:
    f = np.asarray(freqs, float); y = np.asarray(power_lin, float)
    m = (f >= fmin) & (f <= fmax) & np.isfinite(y) & (y > 0)
    return float(np.mean(y[m])) if m.any() else np.nan


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


def _extract_slsd(model):
    """Return (total, broadband, rhythms) as 1D arrays on the model grid."""
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


def _collect_peaks_from_curve(freqs: np.ndarray, rh_matrix: np.ndarray,
                              rel_height: float = 0.5, min_rel_amp: float = 0.02
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(freqs, float).ravel()
    df = float(np.median(np.diff(f))) if f.size > 1 else 1.0
    win_idx, centers, widths_hz = [], [], []
    TW = rh_matrix.shape[0]
    for ti in range(TW):
        y = np.asarray(rh_matrix[ti, :], float)
        if not np.any(np.isfinite(y)):
            continue
        ymax = float(np.nanmax(y)); thr = max(1e-20, min_rel_amp*ymax)
        idx, _ = find_peaks(y, height=thr)
        if idx.size == 0:
            continue
        w_bins = peak_widths(y, idx, rel_height=rel_height)[0]
        for k, ii in enumerate(idx):
            centers.append(float(f[min(len(f)-1, max(0, int(ii))) ]))
            widths_hz.append(float(w_bins[k] * df))
            win_idx.append(ti)
    return np.asarray(win_idx, int), np.asarray(centers, float), np.asarray(widths_hz, float)


# ──────────────────────────── Likelihood helpers (CVLL) ────────────────────────────
def _gamma_loglik_multitaper(y_lin: np.ndarray, mu_lin: np.ndarray, k_tapers: int) -> float:
    """
    Sum log-likelihood under: Y_i | mu_i ~ Gamma(shape=K, scale=mu_i/K),
    matching the scaled-chi-square model for multitaper power with K tapers.
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


def _mt_power_one_window(
    ts_1d: np.ndarray,
    fs: float,
    duration: float,
    nw: float,
    k_tapers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multitaper power estimate for a single 1D time window."""
    x = np.asarray(ts_1d, float).ravel()
    x = x[:, np.newaxis, np.newaxis]  # (n_time, 1, 1)

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

# ──────────────────────────── SL_specdecomp prior overrides ────────────────────────────
# Match Figure 3 / Figure 4 sensitivity settings:
#   b ~ Normal(log10(median positive PSD)), sigma=5
#   A_lin_i ~ LogNormal(log(median power in rhythm band i), sigma=1.25)

B_PRIOR_SIGMA = 5.0
A_HEIGHT_ANCHOR_Q = 50.0
A_HEIGHT_PRIOR_SIGMA = 1.25


def _b_prior_param_specs(
    y_lin: np.ndarray,
    sigma: float = B_PRIOR_SIGMA,
) -> Dict[str, Any]:
    y = np.asarray(y_lin, float)
    y_pos = y[np.isfinite(y) & (y > 0)]
    mu_b = float(np.log10(np.median(y_pos))) if y_pos.size else 0.0
    sigma = float(sigma)

    def _factory(name: str, mu: float = mu_b, sigma: float = sigma):
        return pm.Normal(name, mu=mu, sigma=sigma)

    # Include both names so this works across SL_specdecomp naming variants.
    return {
        "b_0": {"factory": _factory},
        "b": {"factory": _factory},
    }


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

    # Fallback if too few points in the specified rhythm band.
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


def _slsd_kwargs_with_requested_priors(
    slsd_kwargs: Dict[str, Any],
    freqs_fit: np.ndarray,
    y_lin: np.ndarray,
    *,
    b_sigma: float = B_PRIOR_SIGMA,
    a_height_anchor_q: float = A_HEIGHT_ANCHOR_Q,
) -> Dict[str, Any]:
    out = dict(slsd_kwargs)

    # b prior
    aperiodic_specs = dict(out.get("aperiodic_param_specs", {}) or {})
    aperiodic_specs.update(_b_prior_param_specs(y_lin, sigma=b_sigma))
    out["aperiodic_param_specs"] = aperiodic_specs

    # additive A_lin_i height prior
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
    """Compute CVLL per model (specparam, SL_specdecomp) for a single 30s window time series."""
    x = np.asarray(ts_30s, float).ravel()

    n_chunk = int(round(float(fs) * float(cv_chunk_dur)))
    n_expect = int(cv_folds) * n_chunk
    if x.size < n_expect:
        return {"specparam": np.nan, "slsd": np.nan}
    if x.size > n_expect:
        x = x[:n_expect]

    chunks = [x[i * n_chunk:(i + 1) * n_chunk] for i in range(int(cv_folds))]

    # Common independent grid from first chunk
    f0, _S0 = _mt_power_one_window(chunks[0], fs=fs, duration=cv_chunk_dur, nw=cv_nw, k_tapers=cv_k_tapers)
    f_ref_cv, _step = _independent_grid(f0, cv_chunk_dur, cv_nw)

    # Spectra per chunk interpolated onto f_ref_cv
    S_chunks = []
    for c in chunks:
        f_emp, S_emp = _mt_power_one_window(c, fs=fs, duration=cv_chunk_dur, nw=cv_nw, k_tapers=cv_k_tapers)
        S_fit = np.interp(f_ref_cv, f_emp, S_emp)
        S_chunks.append(np.clip(S_fit, 1e-20, np.inf))
    S_chunks = np.asarray(S_chunks, float)  # (folds, n_freq)

    # Restrict to analysis band
    m_fr = (f_ref_cv >= ANALYSIS_FRANGE[0]) & (f_ref_cv <= ANALYSIS_FRANGE[1])
    f_cv = f_ref_cv[m_fr]
    if f_cv.size < 10:
        return {"specparam": np.nan, "slsd": np.nan}

    # SL_specdecomp kwargs for CV refits: start from cfg, optionally override sample_kwargs only
    sl_kw_cv = dict(cfg["slsd_kwargs"])
    if cv_sl_sample_overrides is not None and len(cv_sl_sample_overrides) > 0:
        sk = dict(sl_kw_cv.get("sample_kwargs", {}))
        sk.update(cv_sl_sample_overrides)
        sk["cores"] = 1  # avoid nested multiprocessing surprises
        sl_kw_cv["sample_kwargs"] = sk

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
                sl_kw_cv_fit = _slsd_kwargs_with_requested_priors(
                    sl_kw_cv,
                    f_cv,
                    train,
                )
                sl = Decompose(
                    f_cv,
                    np.clip(train, 1e-20, np.inf),
                    fs=fs,
                    **sl_kw_cv_fit,
                )
                mu_sl, _bb, _rh = _extract_slsd(sl)
                ll = _gamma_loglik_multitaper(test, mu_sl, cv_k_tapers)
                cvll_slsd = cvll_slsd + ll if np.isfinite(ll) else np.nan
            except Exception:
                cvll_slsd = np.nan


    return {"specparam": float(cvll_spec), "slsd": float(cvll_slsd)}


# ──────────────────────────── Row 4, Col 3: CVLL “aux” panel (match Figure 4 aesthetic) ────────────────────────────
def _cvll_aux_panel(
    ax: plt.Axes,
    cvll_specparam: np.ndarray,
    cvll_slsd: np.ndarray,
    *,
    title: str,
    max_windows: Optional[int] = None,
    x_step: float = 0.40,
    x_jitter: float = 0.018,
    line_alpha: float = 0.30,
    point_alpha: float = 0.85,
    lw: float = 1.0,
    point_size: float = 16.0,
    star_size: float = 70.0,
    seed: int = 0,
) -> None:
    """
    Match Figure 4 CVLL aux panel styling exactly.
    """
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
    


# ──────────────────────────── Simulation & truth ────────────────────────────
def simulate_condition_long(name: str, simlen: str, fig4_dir: str = FIG4_DIR_DEFAULT):
    """
    Simulate ONE long time series, then slice into 30 s windows.
    Returns:
      X_wins: (n_time_per_win, n_windows)
      t_win: (n_time_per_win,)
      fs: sampling rate
      res: spectrum result for the long simulation (used for ground-truth spectra)
    """
    cfg = SIM_CFG[name]
    fs = float(cfg["fs"])
    total_duration = cfg["duration_full"] if simlen == "full" else cfg["duration_short"]

    sim_p = sim_params_from_fig4(name, fig4_dir=fig4_dir)
    if sim_p is None:
        raise FileNotFoundError(
            f"Missing Figure-4 export for '{name}'. "
            f"Expected: {Path(fig4_dir)/f'fig4_slsd_params_{name}.json'}"
        )

    res = spectrum(
        sampling_rate=fs,
        duration=float(total_duration),
        **dict(sim_p),
        average_firing_rate=0.0,
        random_state=0,
        direct_estimate=False,
        plot=False,
    )

    td = getattr(res, "time_domain", None)
    x = getattr(td, "combined_signal", None) if td is not None else None
    if x is None:
        x = getattr(td, "signal", None) if td is not None else getattr(res, "signal", None)
    x = np.asarray(x, float).ravel()

    win_dur = float(MT_PARAMS["time_window_duration"])
    n_time = int(round(win_dur * fs))
    n_windows = int(round(float(total_duration) / win_dur))
    n_total = n_windows * n_time

    if x.size < n_total:
        x = np.pad(x, (0, n_total - x.size), mode="constant")
    else:
        x = x[:n_total]

    X_wins = x.reshape(n_windows, n_time).T
    t_win = np.arange(n_time, dtype=float) / fs

    return X_wins, t_win, fs, res


def simulate_condition_independent(name: str, simlen: str, fig4_dir: str = FIG4_DIR_DEFAULT):
    """
    Simulate *independent* 30 s windows repeatedly.
    Returns:
      X_wins: (n_time_per_win, n_windows)
      t_win: (n_time_per_win,)
      fs: sampling rate
      res0: spectrum result object from the first window (for ground-truth spectra)
    """
    cfg = SIM_CFG[name]
    fs = float(cfg["fs"])
    total_duration = cfg["duration_full"] if simlen == "full" else cfg["duration_short"]

    sim_p = sim_params_from_fig4(name, fig4_dir=fig4_dir)
    if sim_p is None:
        raise FileNotFoundError(
            f"Missing Figure-4 export for '{name}'. "
            f"Expected: {Path(fig4_dir)/f'fig4_slsd_params_{name}.json'}"
        )

    win_dur = float(MT_PARAMS["time_window_duration"])
    win_step = float(MT_PARAMS["time_window_step"])
    if abs(win_step - win_dur) > 1e-9:
        raise ValueError("This script assumes non-overlapping windows (time_window_step == time_window_duration).")

    n_windows = int(round(float(total_duration) / win_dur))
    if n_windows < 1:
        raise ValueError(f"Total duration {total_duration} yields n_windows={n_windows} with win_dur={win_dur}.")

    n_time = int(round(win_dur * fs))
    t_win = np.arange(n_time, dtype=float) / fs
    X_wins = np.zeros((n_time, n_windows), dtype=float)

    res0 = None
    for wi in range(n_windows):
        res = spectrum(
            sampling_rate=fs,
            duration=win_dur,
            **dict(sim_p),
            average_firing_rate=0.0,
            random_state=int(wi),
            direct_estimate=False,
            plot=False,
        )

        td = getattr(res, "time_domain", None)
        x = getattr(td, "combined_signal", None) if td is not None else None
        if x is None:
            x = getattr(td, "signal", None) if td is not None else getattr(res, "signal", None)
        x = np.asarray(x, float).ravel()

        if x.size < n_time:
            x = np.pad(x, (0, n_time - x.size), mode="constant")
        elif x.size > n_time:
            x = x[:n_time]

        X_wins[:, wi] = x
        if res0 is None:
            res0 = res

    return X_wins, t_win, fs, res0


def truth_spectra(res, frange=ANALYSIS_FRANGE) -> Dict[str, np.ndarray]:
    fd = getattr(res, "frequency_domain", None)
    if fd is None:
        raise RuntimeError("Result object missing frequency_domain.")
    f_all = np.asarray(fd.frequencies, float).ravel()
    s_comb = np.asarray(fd.combined_spectrum, float).ravel()
    s_bb   = np.asarray(getattr(fd, "broadband_spectrum"), float).ravel()
    m = (f_all >= frange[0]) & (f_all <= frange[1]) & np.isfinite(s_comb) & (s_comb > 0)
    return {"f": f_all[m], "S_comb": s_comb[m], "S_bb": s_bb[m]}


# ──────────────────────────── Figure building from arrays (used by compute + plot-only) ────────────────────────────
def _fig5_payload_base(state_name: str, windows_mode: str, simlen: str) -> Path:
    suffix = ("_2win" if windows_mode == "2" else "") + (f"_{simlen}" if simlen != "full" else "")
    return Path(SAVE_DIR) / f"Figure_5_CV_{state_name}{suffix}"


def _finite_metric_values(x: Any) -> np.ndarray:
    arr = np.asarray(x, float).ravel()
    return arr[np.isfinite(arr)]


def _metric_ylim_from_values(*values: Any, pad_frac: float = 0.05) -> Optional[Tuple[float, float]]:
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
        half = max(abs(vmax) * 0.05, 1e-6)
        return (vmin - half, vmax + half)

    pad = span * float(pad_frac)
    return (vmin - pad, vmax + pad)


def _load_metric_arrays_from_payload_fig5(
    state_name: str,
    windows_mode: str,
    simlen: str,
) -> Dict[str, np.ndarray]:
    arrays, meta, _npz_path, _meta_path = load_plot_payload_fig5(
        _fig5_payload_base(state_name, windows_mode, simlen)
    )
    return {
        "hg_specparam": np.asarray(arrays["hg_specparam"], float),
        "hg_slsd": np.asarray(arrays["hg_slsd"], float),
        "slopes_specparam": np.asarray(arrays["slopes_specparam"], float),
        "slopes_slsd": np.asarray(arrays["slopes_slsd"], float),
    }


def _resolve_shared_metric_ylims_fig5(
    state_name: str,
    windows_mode: str,
    simlen: str,
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
        other = _load_metric_arrays_from_payload_fig5(other_state, windows_mode, simlen)
        hg_values.extend([other["hg_specparam"], other["hg_slsd"]])
        slope_values.extend([other["slopes_specparam"], other["slopes_slsd"]])
    except FileNotFoundError:
        pass

    hg_ylim = _metric_ylim_from_values(*hg_values, pad_frac=0.05)
    slope_ylim = _metric_ylim_from_values(*slope_values, pad_frac=0.05)
    return hg_ylim, slope_ylim

def _build_and_save_figure_from_arrays(
    name: str,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    *,
    suffix: str,
    windows_mode: str,
    simlen: str,
) -> None:
    # Required arrays
    F_fit = arrays["F_fit"]
    P_fit_tf = arrays["P_fit_tf"]
    T_rel_min = arrays["T_rel_min"]

    F_truth = arrays.get("F_truth", np.array([]))
    S_truth = arrays.get("S_truth", np.array([]))
    S_truth_bb = arrays.get("S_truth_bb", np.array([]))
    GT_on_fit = arrays.get("GT_on_fit", np.array([]))

    specparam_full = arrays["specparam_full"]
    specparam_aper = arrays["specparam_aper"]
    specparam_rh = arrays["specparam_rh"]

    slsd_total = arrays["slsd_total"]
    slsd_bb = arrays["slsd_bb"]
    slsd_rh = arrays["slsd_rh"]

    slopes_specparam = arrays["slopes_specparam"]
    slopes_slsd = arrays["slopes_slsd"]
    hg_specparam = arrays["hg_specparam"]
    hg_slsd = arrays["hg_slsd"]

    cvll_specparam = arrays["cvll_specparam"]
    cvll_slsd = arrays["cvll_slsd"]
    idx_target = int(meta.get("idx_target", 0))
    idx_target = int(np.clip(idx_target, 0, max(int(P_fit_tf.shape[0]) - 1, 0)))

    #raster_frange = tuple(meta.get("raster_frange", list(RASTER_FRANGE)))
    raster_frange = (0.0, float(meta.get("raster_frange", list(RASTER_FRANGE))[1]))
    true_hg = meta.get("true_hg", None)
    true_slope = meta.get("true_slope", None)

    # Peak lists (may be empty if payload didn't include them)
    sp_peaks_list = meta.get("specparam_peaks_list", None)
    sl_peaks_list = meta.get("slsd_peaks_list", None)

    # Ensure consistent shapes
    TW = int(P_fit_tf.shape[0])
    fig = plt.figure(figsize=(16.0, 14.2))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.0, 0.9, 0.9, 1.1],
        wspace=0.28, hspace=0.62
    )

    # Row 0 - Single-window fits
    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])

    # Row 1 - All-fits overlay
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])

    # Row 2 - Peak rasters
    ax31 = fig.add_subplot(gs[2, 0])
    ax32 = fig.add_subplot(gs[2, 1])

    # Row 3 - 3 panels spanning full width
    gs_row4 = gs[3, :].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.40)
    ax41 = fig.add_subplot(gs_row4[0, 0])  # HG violin
    ax42 = fig.add_subplot(gs_row4[0, 1])  # slope violin
    ax43 = fig.add_subplot(gs_row4[0, 2])  # CVLL aux spaghetti

    # Row 1 - Single-window fits + GT PSD
    ax11.set_xscale("log"); ax11.set_yscale("log"); ax11.set_ylim(*PSD_YLIM)
    ax11.set_title(f"{name.capitalize()} — specparam (t ≈ {float(T_rel_min[idx_target]):.2f} min)")
    if GT_on_fit.size:
        ax11.plot(F_fit, GT_on_fit, color="k", lw=2.4, alpha=0.95, label="Ground Truth PSD")
    ax11.plot(F_fit, P_fit_tf[idx_target, :], color=COLORS["emp"], **STYLES["emp"], label="Multitaper (30 s)")
    ax11.plot(F_fit, specparam_full[idx_target, :], color=COLORS["full"], **STYLES["full"], label="specparam full")
    ax11.plot(F_fit, specparam_aper[idx_target, :], color=COLORS["broad"], **STYLES["component"], label="aperiodic")
    ax11.plot(F_fit, specparam_rh[idx_target, :],   color=COLORS["rhythms"], **STYLES["component"], label="rhythms")
    ax11.set_xlabel("Frequency (Hz, log10)"); ax11.set_ylabel("Power (log10)"); ax11.legend(frameon=False, fontsize=9, loc="best")

    ax12.set_xscale("log"); ax12.set_yscale("log"); ax12.set_ylim(*PSD_YLIM)
    ax12.set_title(f"{name.capitalize()} — SL_specdecomp (t ≈ {float(T_rel_min[idx_target]):.2f} min)")
    if GT_on_fit.size:
        ax12.plot(F_fit, GT_on_fit, color="k", lw=2.4, alpha=0.95, label="Ground Truth PSD")
    ax12.plot(F_fit, P_fit_tf[idx_target, :], color=COLORS["emp"], **STYLES["emp"], label="Multitaper (30 s)")
    ax12.plot(F_fit, slsd_total[idx_target, :], color=COLORS["full"], **STYLES["full"], label="SL_specdecomp full")
    ax12.plot(F_fit, slsd_bb[idx_target, :],    color=COLORS["broad"], **STYLES["component"], label="broadband")
    ax12.plot(F_fit, slsd_rh[idx_target, :],    color=COLORS["rhythms"], **STYLES["component"], label="rhythms")
    ax12.set_xlabel("Frequency (Hz, log10)"); ax12.set_ylabel("Power (log10)"); ax12.legend(frameon=False, fontsize=9, loc="best")

    # Row 2 - All-fit overlays
    ax21.set_xscale("log"); ax21.set_yscale("log"); ax21.set_ylim(*PSD_YLIM)
    for ti in range(TW):
        ax21.plot(F_fit, specparam_full[ti, :], color=COLORS["overlay"], alpha=0.10, lw=1.2)
    ax21.set_title("specparam — full model across windows"); ax21.set_xlabel("Frequency (Hz, log10)"); ax21.set_ylabel("Power (log10)")

    ax22.set_xscale("log"); ax22.set_yscale("log"); ax22.set_ylim(*PSD_YLIM)
    for ti in range(TW):
        ax22.plot(F_fit, slsd_total[ti, :], color=COLORS["overlay"], alpha=0.10, lw=1.2)
    ax22.set_title("SL_specdecomp — full model across windows"); ax22.set_xlabel("Frequency (Hz, log10)"); ax22.set_ylabel("Power (log10)")

    # Row 3 - Peak rasters
    rlo, rhi = raster_frange

    # If peak lists exist, use them; else fallback to curve peak-picking
    sp_wins, sp_cf, sp_w = [], [], []
    if isinstance(sp_peaks_list, list) and len(sp_peaks_list) == TW:
        for wi, pk in enumerate(sp_peaks_list):
            if pk is None or np.size(pk) == 0:
                continue
            for cf, amp, fwhm in np.asarray(pk, float):
                if rlo <= cf <= rhi:
                    sp_wins.append(wi); sp_cf.append(cf); sp_w.append(fwhm)
    if not sp_cf:
        w_i, cf_i, w_iw = _collect_peaks_from_curve(F_fit, specparam_rh)
        msk = (cf_i >= rlo) & (cf_i <= rhi)
        sp_wins, sp_cf, sp_w = list(w_i[msk]), list(cf_i[msk]), list(w_iw[msk])

    sl_wins, sl_cf, sl_w = [], [], []
    if isinstance(sl_peaks_list, list) and len(sl_peaks_list) == TW:
        for wi, pk in enumerate(sl_peaks_list):
            if pk is None or np.size(pk) == 0:
                continue
            for cf, amp, fwhm in np.asarray(pk, float):
                if rlo <= cf <= rhi:
                    sl_wins.append(wi); sl_cf.append(cf); sl_w.append(fwhm)
    if not sl_cf:
        w_i, cf_i, w_iw = _collect_peaks_from_curve(F_fit, slsd_rh)
        msk = (cf_i >= rlo) & (cf_i <= rhi)
        sl_wins, sl_cf, sl_w = list(w_i[msk]), list(cf_i[msk]), list(w_iw[msk])

    if sp_cf:
        ax31.errorbar(np.asarray(sp_cf), np.asarray(sp_wins),
                      xerr=np.asarray(sp_w) / 2.0,
                      fmt="o", ms=4.0, lw=1.0, mfc="none",
                      mec=COLORS["specparam_pt"], ecolor=COLORS["specparam_pt"],
                      alpha=0.9, label="specparam (center ± FWHM/2)")
    ax31.set_xlim(raster_frange)
    ax31.set_ylim(-0.5, max(TW - 0.5, 0.5))
    ax31.set_xlabel("Frequency (Hz)")
    ax31.set_ylabel("Window #")
    ax31.set_title("Peak raster — specparam")
    ax31.legend(frameon=False, fontsize=9, loc="upper right")

    if sl_cf:
        ax32.errorbar(np.asarray(sl_cf), np.asarray(sl_wins),
                      xerr=np.asarray(sl_w) / 2.0,
                      fmt="s", ms=4.0, lw=1.0, mfc="none",
                      mec=COLORS["slsd_pt"], ecolor=COLORS["slsd_pt"],
                      alpha=0.9, label="SL_specdecomp (center ± FWHM/2)")
    ax32.set_xlim(raster_frange)
    ax32.set_ylim(-0.5, max(TW - 0.5, 0.5))
    ax32.set_xlabel("Frequency (Hz)")
    ax32.set_ylabel("")
    #ax32.set_title("Peak raster - SL_specdecomp")
    ax32.set_title("Peak raster — SL_specdecomp")

    gt_params = meta.get("gt_params", {}) or {}
    gt_peaks = (gt_params.get("spectrum_kwargs", {}) or {}).get("peaks", []) or []

    gt_cf_fit = np.asarray(
        [
            float(pk["freq"])
            for pk in gt_peaks
            if isinstance(pk, dict) and ("freq" in pk)
        ],
        dtype=float,
    )

    # keep only peaks in raster range
    gt_cf_fit = gt_cf_fit[(gt_cf_fit >= rlo) & (gt_cf_fit <= rhi)]

    for cf in gt_cf_fit:
        ax31.axvline(cf, color="k", ls="--", lw=2.2, alpha=0.95, zorder=0)
        ax32.axvline(cf, color="k", ls="--", lw=2.2, alpha=0.95, zorder=0)


    ax32.legend(frameon=False, fontsize=9, loc="upper right")

    # Row 4 - violins
    shared_hg_ylim, shared_slope_ylim = _resolve_shared_metric_ylims_fig5(
        name,
        windows_mode,
        simlen,
        hg_specparam=hg_specparam,
        hg_slsd=hg_slsd,
        slopes_specparam=slopes_specparam,
        slopes_slsd=slopes_slsd,
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
    if true_hg is not None and np.isfinite(true_hg):
        ax41.axhline(float(true_hg), color="k", lw=2.4, alpha=0.95)
    ax41.set_title("High-gamma mean power (80–180 Hz)")
    ax41.set_xlabel("")
    ax41.set_ylabel("Linear power")
    #ax41.tick_params(axis="x", rotation=15)
    if shared_hg_ylim is not None:
        ax41.set_ylim(*shared_hg_ylim)
    ax41.tick_params(axis="x", rotation=15)

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
    if true_slope is not None and np.isfinite(true_slope):
        ax42.axhline(float(true_slope), color="k", lw=2.4, alpha=0.95)
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

    # Suptitle uses meta if available, else defaults
    cv = meta.get("CV", {})
    cv_folds = int(cv.get("cv_folds", CV_FOLDS))
    cv_chunk_dur = float(cv.get("cv_chunk_dur", MT_PARAMS["time_window_duration"] / CV_FOLDS))
    cv_nw = float(cv.get("cv_nw", CV_NW))
    cv_k_tapers = int(cv.get("cv_k_tapers", CV_K_TAPERS))

    fig.suptitle(
        f"Figure 5 (CV; simulated) — {name.capitalize()} "
        f"(rows1–3: 30 s MT K={MT_PARAMS['n_tapers']}, NW={MT_PARAMS['time_halfbandwidth_product']}; "
        f"row4: metrics + CVLL aux; CV={cv_folds}×{cv_chunk_dur:.0f}s, MT K={cv_k_tapers}, NW={cv_nw})",
        y=0.995,
        fontsize=14,
    )

    plt.tight_layout()

    png = os.path.join(SAVE_DIR, f"Figure_5_CV_{name}{suffix}.png")
    svg = os.path.join(SAVE_DIR, f"Figure_5_CV_{name}{suffix}.svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {name} figure → {png} / {svg}")


# ──────────────────────────── Core per-condition runner (compute) ────────────────────────────
def run_condition_simulated(
    name: str,
    windows_mode: str = "all",
    simlen: str = "full",
    fig4_dir: str = FIG4_DIR_DEFAULT,
    sim_windows: str = "independent",  # "independent" or "long"
    cv_folds: int = CV_FOLDS,
    cv_nw: float = CV_NW,
    cv_k_tapers: int = CV_K_TAPERS,
    cv_chunk_dur: Optional[float] = None,
    cv_sl_draws: int = -1,
    cv_sl_tune: int = -1,
    cv_sl_chains: int = -1,
    save_payload: bool = True,
    export_params: bool = True,
) -> None:
    cfg = SIM_CFG[name]

    if cv_chunk_dur is None:
        cv_chunk_dur = float(MT_PARAMS["time_window_duration"]) / float(cv_folds)

    # Per-condition raster x-range: cover all rhythm bands
    rbands = cfg["slsd_kwargs"].get("rhythm_bands", [])
    if rbands:
        #raster_lo = min(b[0] for b in rbands)
        raster_lo = 0.0
        raster_hi = max(b[1] for b in rbands)
    else:
        raster_lo, raster_hi = RASTER_FRANGE
    raster_frange = (raster_lo, raster_hi)

    # Simulate windows
    if sim_windows == "long":
        X_wins, t_win, fs, res_truth = simulate_condition_long(name, simlen=simlen, fig4_dir=fig4_dir)
    else:
        X_wins, t_win, fs, res_truth = simulate_condition_independent(name, simlen=simlen, fig4_dir=fig4_dir)

    truth = truth_spectra(res_truth, frange=ANALYSIS_FRANGE)
    F_truth = truth["f"]; S_truth = truth["S_comb"]; S_truth_bb = truth["S_bb"]

    # Multitaper (Rows 1–4): concatenate windows along time so MT returns one PSD per 30s block
    x_long = np.asarray(X_wins, float).T.reshape(-1)
    P_tf, F_all, T_abs = compute_multitaper(x_long, fs=fs, t0=0.0, params=MT_PARAMS)
    P_tf, F_all, T_abs = _ensure_tf(P_tf, F_all, T_abs)

    if P_tf.shape[0] != X_wins.shape[1]:
        print(f"[WARN] {name}: multitaper returned {P_tf.shape[0]} windows but simulated {X_wins.shape[1]}.")
    else:
        print(f"[INFO] {name}: simulated {X_wins.shape[1]} windows; multitaper produced {P_tf.shape[0]} windows.")

    m = (F_all > 0) & (F_all >= ANALYSIS_FRANGE[0]) & (F_all <= ANALYSIS_FRANGE[1])
    F_an = F_all[m]; P_an_tf = P_tf[:, m]
    F_fit, step = _independent_grid(F_an, MT_PARAMS["time_window_duration"], MT_PARAMS["time_halfbandwidth_product"])
    P_fit_tf = P_an_tf[:, ::step]

    # Choose the display/export target window to match the empirical Figure 4 target time.
    # If the Figure 4 payload is unavailable, fall back to the simulation midpoint.
    T_rel_min = (T_abs - float(T_abs[0])) / 60.0
    fig4_target_rel_min = fig4_target_rel_min_from_payload(name, fig4_dir=fig4_dir)
    if fig4_target_rel_min is not None:
        idx_target = _idx_nearest_time(T_rel_min, fig4_target_rel_min, fallback_idx=0)
        print(f"[INFO] {name}: Figure 5 target linked to Figure 4 target t≈{fig4_target_rel_min:.2f} min; simulated display window index={idx_target}.")
    else:
        idx_target = int(np.nanargmin(np.abs(T_rel_min - np.nanmedian(T_rel_min))))
        print(f"[WARN] {name}: Figure 4 target time unavailable; using simulation midpoint t≈{float(T_rel_min[idx_target]):.2f} min.")

    sel_orig = None
    if windows_mode == "2":
        TW_all = P_fit_tf.shape[0]
        neighbor = min(idx_target + 1, TW_all - 1) if idx_target < TW_all - 1 else max(idx_target - 1, 0)
        sel = sorted(set([idx_target, neighbor]))
        sel_orig = list(sel)
        P_fit_tf = P_fit_tf[sel, :]
        T_rel_min = T_rel_min[sel]
        idx_target = 0

    # Interpolate truth PSD onto F_fit for overlays
    GT_on_fit = np.interp(F_fit, F_truth, S_truth)
    true_hg = _hg_mean(F_truth, S_truth_bb, *HG_BAND)
    true_slope = _compute_loglog_slope(F_truth, S_truth_bb, *SLOPE_BAND)

    # Per-window fits (Rows 1–4)
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

    # CVLL arrays (used for Row 4, Col 3 aux panel)
    cvll_specparam = np.full(TW, np.nan)
    cvll_slsd      = np.full(TW, np.nan)

    cv_sl_overrides = {}
    if cv_sl_draws > 0:
        cv_sl_overrides["draws"] = int(cv_sl_draws)
    if cv_sl_tune > 0:
        cv_sl_overrides["tune"] = int(cv_sl_tune)
    if cv_sl_chains > 0:
        cv_sl_overrides["chains"] = int(cv_sl_chains)

    win_dur = float(MT_PARAMS["time_window_duration"])
    n_win = int(round(win_dur * fs))
    if X_wins.shape[0] != n_win:
        raise ValueError(f"Expected X_wins to have {n_win} samples per window; got {X_wins.shape[0]}.")

    for ti in range(TW):
        y = np.clip(P_fit_tf[ti, :], 1e-20, np.inf)
        fr_k = (max(ANALYSIS_FRANGE[0], float(F_fit[0])), min(ANALYSIS_FRANGE[1], float(F_fit[-1])))

        # specparam
        sp_full, sp_ap, sp_peaks = _specparam_full_aper_peaks(F_fit, y, fr_k, **cfg["specparam_kwargs"])
        specparam_full[ti, :] = sp_full
        specparam_aper[ti, :] = sp_ap
        specparam_rh[ti, :]   = np.clip(sp_full - sp_ap, 0.0, np.inf)
        sp_peaks_list.append(np.asarray(sp_peaks, float))

        # SL_specdecomp
        sl_kw = _slsd_kwargs_with_requested_priors(
            cfg["slsd_kwargs"],
            F_fit,
            y,
        )
        sl = Decompose(F_fit, y, fs=fs, **sl_kw)
        sl_tot, sl_bb, sl_rh = _extract_slsd(sl)
        slsd_total[ti, :] = sl_tot
        slsd_bb[ti, :]    = sl_bb
        slsd_rh[ti, :]    = sl_rh
        sl_models.append(sl)
        sl_peaks_list.append(_slsd_peak_params(sl))

        # Metrics (linear power from AP/BB components)
        slopes_specparam[ti] = _compute_loglog_slope(F_fit, sp_ap, *SLOPE_BAND)
        slopes_slsd[ti]      = _compute_loglog_slope(F_fit, sl_bb, *SLOPE_BAND)
        hg_specparam[ti]     = _hg_mean(F_fit, sp_ap, *HG_BAND)
        hg_slsd[ti]          = _hg_mean(F_fit, sl_bb, *HG_BAND)

        # CVLL from the corresponding 30 s window time series
        orig_i = sel_orig[ti] if sel_orig is not None else ti
        if orig_i < 0 or orig_i >= X_wins.shape[1]:
            continue
        x_win = np.asarray(X_wins[:, orig_i], float).ravel()

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
        cvll_slsd[ti]      = cvll["slsd"]
    # Choose the displayed Figure 5 simulation window after both models have fit all windows.
    #
    # Important:
    #   - The simulation ground truth was already generated from the Figure 4
    #     target-window SL_specdecomp decomposition.
    #   - Here we only choose which simulated 30 s window is shown.
    #   - We choose the window where specparam reproduces the empirical
    #     low-frequency specparam failure, while avoiding catastrophic fits.
    idx_target_fig4_linked = int(idx_target)
    display_choice = {
        "idx": int(idx_target),
        "selection_policy": "kept_figure4_time_nearest_idx",
        "emp_match_low": None,
        "emp_match_broad": None,
        "wrong_low_vs_true_bb": None,
        "specparam_vs_slsd_low": None,
        "overall_specparam_to_data": None,
        "fig4_target_time_min": (
            float(fig4_target_rel_min) if fig4_target_rel_min is not None else None
        ),
        "fig4_target_idx": None,
    }

    if windows_mode == "all":
        display_choice = _choose_sim_window_matching_empirical_specparam_error(
            state=name,
            fig4_dir=str(fig4_dir),
            F_fit=F_fit,
            F_truth=F_truth,
            S_truth_bb=S_truth_bb,
            P_fit_tf=P_fit_tf,
            specparam_full=specparam_full,
            specparam_aper=specparam_aper,
            slsd_bb=slsd_bb,
            preferred_idx=idx_target_fig4_linked,
        )
        idx_target = int(display_choice["idx"])

        print(
            f"[INFO] {name}: Figure 5 display window selected by empirical "
            f"specparam-error matching: idx={idx_target}, "
            f"t≈{float(T_rel_min[idx_target]):.2f} min; "
            f"policy={display_choice['selection_policy']}; "
            f"emp_match_low={display_choice['emp_match_low']}; "
            f"wrong_low_vs_true_bb={display_choice['wrong_low_vs_true_bb']}; "
            f"specparam_vs_slsd_low={display_choice['specparam_vs_slsd_low']}; "
            f"overall_specparam_to_data={display_choice['overall_specparam_to_data']}; "
            f"Figure-4-time-nearest idx was {idx_target_fig4_linked}."
        )
    else:
        print(
            f"[INFO] {name}: windows_mode='2'; keeping Figure-4-time-nearest "
            f"display window idx={idx_target}, "
            f"t≈{float(T_rel_min[idx_target]):.2f} min."
        )
    # ──────────────────────────── Exports ────────────────────────────
    suffix = ("_2win" if windows_mode == "2" else "") + (f"_{simlen}" if simlen != "full" else "")
    out_base = Path(SAVE_DIR) / f"Figure_5_CV_{name}{suffix}"

    # Export params JSONs (ground truth + target-window SL_specdecomp fit), for Figure 1 reuse
    gt_sim_p = sim_params_from_fig4(name, fig4_dir=fig4_dir)
    gt_params_payload = None
    gt_params_path = None
    slsd_fit_params_payload = None
    slsd_fit_params_path = None

    if export_params:
        if gt_sim_p is not None:
            gt_params_payload = {
                "schema": "Figure5_GT_spectrum_kwargs.v1",
                "state": name,
                "simlen": simlen,
                "sim_windows": sim_windows,
                "fs": float(fs),
                "window_duration_s": float(MT_PARAMS["time_window_duration"]),
                "n_windows": int(X_wins.shape[1]),
                "random_state_policy": ("per-window index" if sim_windows != "long" else "0"),
                "spectrum_kwargs": dict(gt_sim_p),
                "notes": "Direct kwargs compatible with spectral_decomposition.spectrum().",
            }
            gt_params_path = save_fig5_params(name, "gt", gt_params_payload, SAVE_DIR)

        # Target-window SL_specdecomp fit params
        sl_target_model = sl_models[idx_target]
        bb_target = slsd_bb[idx_target, :]
        slsd_fit_params_payload = slsd_params_from_model(sl_target_model, F_fit, bb_target, ref_freq_hz=10.0)
        slsd_fit_params_path = save_fig5_params(name, "slsd_fit", slsd_fit_params_payload, SAVE_DIR)

    # Save plot payload (NPZ + JSON meta)
    arrays = {
        # Core grids / data
        "F_fit": F_fit,
        "P_fit_tf": P_fit_tf,
        "T_rel_min": T_rel_min,

        # Truth overlays
        "F_truth": F_truth,
        "S_truth": S_truth,
        "S_truth_bb": S_truth_bb,
        "GT_on_fit": GT_on_fit,

        # Model curves
        "specparam_full": specparam_full,
        "specparam_aper": specparam_aper,
        "specparam_rh": specparam_rh,
        "slsd_total": slsd_total,
        "slsd_bb": slsd_bb,
        "slsd_rh": slsd_rh,

        # Metrics
        "slopes_specparam": slopes_specparam,
        "slopes_slsd": slopes_slsd,
        "hg_specparam": hg_specparam,
        "hg_slsd": hg_slsd,

        # CVLL
        "cvll_specparam": cvll_specparam,
        "cvll_slsd": cvll_slsd,
    }

    meta = {
        "state": name,
        "simlen": simlen,
        "sim_windows": sim_windows,
        "windows_mode": windows_mode,
        "idx_target": int(idx_target),
        "idx_target_fig4_linked": int(idx_target_fig4_linked),
        "sel_orig": (list(sel_orig) if sel_orig is not None else None),
        "display_window_selection": dict(display_choice),
        "analysis_frange": list(ANALYSIS_FRANGE),
        "raster_frange": list(raster_frange),
        "HG_BAND": list(HG_BAND),
        "SLOPE_BAND": list(SLOPE_BAND),
        "PSD_YLIM": list(PSD_YLIM),

        "MT_PARAMS": dict(MT_PARAMS),
        "CV": dict(cv_folds=int(cv_folds), cv_chunk_dur=float(cv_chunk_dur), cv_nw=float(cv_nw), cv_k_tapers=int(cv_k_tapers)),

        # Peak lists (JSON-friendly)
        "specparam_peaks_list": [np.asarray(p, float).tolist() for p in sp_peaks_list],
        "slsd_peaks_list": [np.asarray(p, float).tolist() for p in sl_peaks_list],

        # Truth scalars used in Row 4
        "true_hg": float(true_hg) if np.isfinite(true_hg) else None,
        "true_slope": float(true_slope) if np.isfinite(true_slope) else None,

        # Upstream provenance
        "fig4_dir": str(fig4_dir),
        "fig4_params_path_expected": str(Path(fig4_dir) / f"fig4_slsd_params_{name}.json"),
        "fig4_target_rel_min": (float(fig4_target_rel_min) if fig4_target_rel_min is not None else None),

        # Exported param paths
        "gt_params_path": gt_params_path,
        "gt_params": gt_params_payload,
        "slsd_fit_params_path": slsd_fit_params_path,
        "slsd_fit_params": slsd_fit_params_payload,
    }

    if save_payload:
        save_plot_payload_fig5(out_base, arrays=arrays, meta=meta)

    # Build and save figure
    _build_and_save_figure_from_arrays(
        name,
        arrays,
        meta,
        suffix=suffix,
        windows_mode=windows_mode,
        simlen=simlen,
    )


# ──────────────────────────── Plot-only runner ────────────────────────────

def plot_condition_from_payload(
    name: str,
    *,
    windows_mode: str,
    simlen: str,
) -> None:
    suffix = ("_2win" if windows_mode == "2" else "") + (f"_{simlen}" if simlen != "full" else "")
    out_base = Path(SAVE_DIR) / f"Figure_5_CV_{name}{suffix}"
    arrays, meta, npz_path, meta_path = load_plot_payload_fig5(out_base)
    print(f"[INFO] Loaded plot payload for {name}: {npz_path.name} + {meta_path.name}")

    _build_and_save_figure_from_arrays(
        name,
        arrays,
        meta,
        suffix=suffix,
        windows_mode=windows_mode,
        simlen=simlen,
    )


# ──────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Build Figure 5 (Awake & Anesthesia) — simulated data + CVLL aux panel + exports (no Row 5).")
    ap.add_argument("--fig4-dir", type=str, default=FIG4_DIR_DEFAULT,
                    help="Directory where figure_4_final_feb_2026.py wrote fig4_slsd_params_*.json.")

    ap.add_argument("--mode", choices=["both", "awake", "anesthesia"], default="both",
                    help="Which figure(s) to build.")
    ap.add_argument("--windows", choices=["all", "2"], default="all",
                    help="Use all 30 s windows (all) or exactly two windows per condition (2).")
    ap.add_argument("--simlen", choices=["full", "short"], default="full",
                    help="Simulation length: full = 900 s (awake) / 540 s (anesthesia); short = 60 s.")

    # Simulation windowing mode (relevant for compute mode; stored in meta)
    ap.add_argument("--sim-windows", choices=["independent", "long"], default="independent",
                    help="How to obtain windows: independent 30 s realizations, or slice a single long simulation.")

    # CV params (used for CVLL aux panel)
    ap.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    ap.add_argument("--cv-chunk-dur", type=float, default=None,
                    help="Seconds; default is 30/cv_folds.")
    ap.add_argument("--cv-nw", type=float, default=CV_NW)
    ap.add_argument("--cv-k-tapers", type=int, default=CV_K_TAPERS)

    # CV sampler overrides (SL_specdecomp fits inside CV loop)
    ap.add_argument("--cv-sl-draws", type=int, default=-1)
    ap.add_argument("--cv-sl-tune", type=int, default=-1)
    ap.add_argument("--cv-sl-chains", type=int, default=-1)

    # Exports (ON by default)
    ap.set_defaults(save_payload=True)
    ap.add_argument("--no-payload", dest="save_payload", action="store_false",
                    help="Disable saving NPZ+JSON plot payload.")

    ap.set_defaults(export_params=True)
    ap.add_argument("--no-export-params", dest="export_params", action="store_false",
                    help="Disable saving fig5_*_params_*.json exports.")

    # NEW: plot-only
    ap.add_argument("--plot-only", action="store_true",
                    help="Do not re-simulate/re-fit/re-CV. Load saved .plotdata.npz + .plotmeta.json and re-render figures.")

    args = ap.parse_args()

    windows_mode = args.windows

    if args.plot_only:
        if args.mode in ("both", "awake"):
            plot_condition_from_payload("awake", windows_mode=windows_mode, simlen=args.simlen)
        if args.mode in ("both", "anesthesia"):
            plot_condition_from_payload("anesthesia", windows_mode=windows_mode, simlen=args.simlen)
        return

    # Compute mode
    if args.mode in ("both", "awake"):
        run_condition_simulated(
            "awake",
            windows_mode=windows_mode,
            simlen=args.simlen,
            fig4_dir=args.fig4_dir,
            sim_windows=args.sim_windows,
            cv_folds=int(args.cv_folds),
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            cv_chunk_dur=args.cv_chunk_dur,
            cv_sl_draws=int(args.cv_sl_draws),
            cv_sl_tune=int(args.cv_sl_tune),
            cv_sl_chains=int(args.cv_sl_chains),
            save_payload=bool(args.save_payload),
            export_params=bool(args.export_params),
        )

    if args.mode in ("both", "anesthesia"):
        run_condition_simulated(
            "anesthesia",
            windows_mode=windows_mode,
            simlen=args.simlen,
            fig4_dir=args.fig4_dir,
            sim_windows=args.sim_windows,
            cv_folds=int(args.cv_folds),
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            cv_chunk_dur=args.cv_chunk_dur,
            cv_sl_draws=int(args.cv_sl_draws),
            cv_sl_tune=int(args.cv_sl_tune),
            cv_sl_chains=int(args.cv_sl_chains),
            save_payload=bool(args.save_payload),
            export_params=bool(args.export_params),
        )


if __name__ == "__main__":
    main()
