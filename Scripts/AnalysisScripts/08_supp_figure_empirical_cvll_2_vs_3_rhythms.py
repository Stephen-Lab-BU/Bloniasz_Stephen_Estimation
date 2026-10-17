#!/usr/bin/env python3
"""
Generate the supplemental empirical CVLL comparison for two- and three-rhythm models.

The script compares matched SL_specdecomp rhythm-model specifications in awake
and anesthetized ECoG windows and writes grouped slope and CVLL summary panels.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.special import gammaln
from spectral_connectivity import Connectivity, Multitaper
from specparam import SpectralModel
from SL_specdecomp import Decompose



PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# ──────────────────────────── Global config ────────────────────────────
ROOT = str(PROJECT_ROOT)
DATA_DIR = os.path.join(ROOT, "Data", "InputData", "InputDataFiles")
SAVE_DIR = os.path.join(ROOT, "Output", "Results", "FiguresIntermediate", "Aux_Figure_Broadband_Slope_Grouped_CVLL", "Figure_output")
os.makedirs(SAVE_DIR, exist_ok=True)

# Default Figure 4 inspector outlier report.
# This supplement should use the same state-window exclusions as the
# Figure 4-derived empirical summaries when the shared CSV is present.
DEFAULT_OUTLIER_CSV_CANDIDATES = [
    os.path.join(ROOT, "Output", "Results", "FiguresIntermediate", "Figure_4_CV", "outliers_report.csv"),
    os.path.join(ROOT, "Output", "Results", "FiguresIntermediate", "Figure_4_CV", "Figure_output", "outliers_report.csv"),
    os.path.join(ROOT, "Figures", "Figure_4_CV", "Figure_output", "outliers_report.csv"),
]
DEFAULT_OUTLIER_CSV = next((p for p in DEFAULT_OUTLIER_CSV_CANDIDATES if os.path.exists(os.path.expanduser(p))), "")


ECOG_MAT = os.path.join(DATA_DIR, "ECoG_ch1.mat")
TIME_MAT = os.path.join(DATA_DIR, "ECoGTime.mat")
COND_MAT = os.path.join(DATA_DIR, "Condition.mat")
ECOG_KEY = "ECoG_ch1"
TIME_KEY = "ECoGTime"
COND_TIME_KEY = "ConditionTime"
COND_LABEL_KEY = "ConditionLabel"

MT_PARAMS = dict(
    time_halfbandwidth_product=2,
    n_tapers=3,
    time_window_duration=30.0,
    time_window_step=30.0,
)

CV_FOLDS = 5
CV_NW = 1
CV_K_TAPERS = 1

ANALYSIS_FRANGE = (0.1, 200.0)
SLOPE_BAND = (40.0, 60.0)

mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.grid": False,
    "legend.fontsize": 8,
})
sns.set(context="talk", style="white")

PALETTE = sns.color_palette("deep")


COLORS = {
    "specparam": PALETTE[0],       # blue
    "slsd": PALETTE[1],            # orange
    "naive": "0.55",
    "cv_two": PALETTE[1],          # make this orange too
    "cv_three": PALETTE[1],        # already orange
    "pair_line": "0.45",
}

SEGMENT_CFG = {
    "awake": {
        "start_phrase": "AwakeEyesClosed-Start",
        "end_phrase": "AwakeEyesClosed-End",
    },
    "anesthesia": {
        "start_phrase": "Anesthetized Start",
        "end_phrase": "Anesthetized End",
    },
}

# These are the original state-specific fitting choices used for the slope panels.
SLOPE_MODEL_CFG = {
    "awake": {
        "specparam_kwargs": dict(
            aperiodic_mode="knee",
            peak_width_limits=[1.0, 30.0],
            max_n_peaks=2,
            min_peak_height=0.0,
            peak_threshold=2.0,
            verbose=False,
        ),
        "slsd_kwargs": dict(
            mode="additive",
            n_aperiodics=1,
            n_rhythms=2,
            rhythm_bands=[(8.0, 20.0), (20.0, 30.0)],
            sample_kwargs=dict(draws=1000, tune=1000, chains=2, target_accept=0.90, cores=1),
            plot=False,
        ),
    },
    "anesthesia": {
        "specparam_kwargs": dict(
            aperiodic_mode="knee",
            peak_width_limits=[1.0, 30.0],
            max_n_peaks=3,
            min_peak_height=0.0,
            peak_threshold=2.0,
            verbose=False,
        ),
        "slsd_kwargs": dict(
            mode="additive",
            n_aperiodics=1,
            n_rhythms=3,
            rhythm_bands=[(0.1, 4.0), (8.0, 20.0), (20.0, 30.0)],
            sample_kwargs=dict(draws=1000, tune=1000, chains=2, target_accept=0.90, cores=1),
            plot=False,
        ),
    },
}

# These are the matched model pair compared in the CVLL panels for BOTH states.
CV_COMPARE_CFG = {
    "two_rhythm_no_20_30": dict(
        mode="additive",
        n_aperiodics=1,
        n_rhythms=2,
        rhythm_bands=[(0.1, 4.0), (8.0, 20.0)],
        sample_kwargs=dict(draws=1000, tune=1000, chains=2, target_accept=0.90, cores=1),
        plot=False,
    ),
    "three_rhythm_with_20_30": dict(
        mode="additive",
        n_aperiodics=1,
        n_rhythms=3,
        rhythm_bands=[(0.1, 4.0), (8.0, 20.0), (20.0, 30.0)],
        sample_kwargs=dict(draws=1000, tune=1000, chains=2, target_accept=0.90, cores=1),
        plot=False,
    ),
}


# ──────────────────────────── Small helpers ────────────────────────────
def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        if o.size <= 50:
            return o.tolist()
        return {"__ndarray__": True, "shape": list(o.shape), "dtype": str(o.dtype)}
    return str(o)


def _coerce_array(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    if isinstance(x, (float, int, bool, np.number)):
        return np.array([x])
    arr = np.asarray(x)
    if arr.dtype == object:
        return np.array([])
    return arr


def save_plot_payload(out_base: Path, *, arrays: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[Path, Path]:
    out_base = Path(out_base)
    npz_path = out_base.with_suffix(".plotdata.npz")
    meta_path = out_base.with_suffix(".plotmeta.json")

    flat = {}
    meta_out = {
        "created_local": datetime.datetime.now().isoformat(timespec="seconds"),
        "arrays": {},
        "meta": meta if meta is not None else {},
    }

    for k, v in arrays.items():
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

    print(f"[INFO] Saved payload -> {npz_path}")
    print(f"[INFO] Saved payload meta -> {meta_path}")
    return npz_path, meta_path


def load_plot_payload(out_base: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    out_base = Path(out_base)
    npz_path = out_base.with_suffix(".plotdata.npz")
    meta_path = out_base.with_suffix(".plotmeta.json")
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing payload files: {npz_path} / {meta_path}")
    arrays = dict(np.load(npz_path, allow_pickle=False))
    with open(meta_path, "r") as f:
        meta = json.load(f)
    print(f"[INFO] Loaded payload <- {npz_path}")
    print(f"[INFO] Loaded payload meta <- {meta_path}")
    return arrays, meta


def _payload_base() -> Path:
    return Path(SAVE_DIR) / "Aux_Figure_Broadband_Slope_Grouped_CVLL"


def _normalize(s: str) -> str:
    s = str(s).lower().replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def _parse_int_list(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.replace(" ", "").split(","):
        if tok:
            out.append(int(tok))
    return sorted(set(out))


def _read_drop_from_outlier_csv(csv_path: str, condition: str) -> List[int]:
    csv_path = os.path.expanduser(str(csv_path))
    df = pd.read_csv(csv_path)
    df = df[df["condition"].astype(str).str.lower() == str(condition).lower()]
    if df.empty:
        return []
    return sorted(set(int(x) for x in df["window_index"].values))


def _find_interval(cond_times: np.ndarray, cond_labels: List[str], start_phrase: str, end_phrase: str) -> Optional[Tuple[float, float]]:
    labs = [_normalize(l) for l in cond_labels]
    t = np.asarray(cond_times, float).ravel()
    s_norm = _normalize(start_phrase)
    e_norm = _normalize(end_phrase)
    start_idx = next((i for i, lab in enumerate(labs) if s_norm in lab), None)
    end_idx = next((i for i, lab in enumerate(labs) if e_norm in lab), None)
    if start_idx is None or end_idx is None:
        return None
    t0, t1 = float(t[start_idx]), float(t[end_idx])
    if t1 <= t0:
        return None
    return (t0, t1)


def _restrict_to_interval(x_time: np.ndarray, x_val: np.ndarray, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
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


def _compute_loglog_slope(freqs: np.ndarray, power_lin: np.ndarray, fmin: float, fmax: float) -> float:
    f = np.asarray(freqs, float)
    y = np.asarray(power_lin, float)
    mask = (f >= fmin) & (f <= fmax) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return np.nan
    xf = np.log10(f[mask])
    yf = np.log10(y[mask])
    m, _b = np.polyfit(xf, yf, 1)
    return float(m)


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
    if vmax == vmin:
        half = max(abs(vmax) * 0.05, 1e-6)
        return (vmin - half, vmax + half)
    pad = pad_frac * (vmax - vmin)
    return (vmin - pad, vmax + pad)


def compute_multitaper(x: np.ndarray, fs: float, t0: float, params: Dict[str, Any]):
    x = np.asarray(x, float).ravel()
    x_3d = x[:, np.newaxis, np.newaxis]
    mt = Multitaper(x_3d, sampling_frequency=fs, start_time=float(t0), **params)
    conn = Connectivity.from_multitaper(mt)
    P = conn.power().squeeze()
    F = np.asarray(conn.frequencies).ravel()
    T = np.asarray(conn.time).ravel()
    return P, F, T


def _specparam_full_aper(freqs_fit: np.ndarray, power_lin: np.ndarray, freq_range: Tuple[float, float], **specparam_kwargs) -> Tuple[np.ndarray, np.ndarray]:
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

    return full_lin, ap_lin


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


def _with_sample_overrides(cfg: Dict[str, Any], draws: int, tune: int, chains: int) -> Dict[str, Any]:
    out = dict(cfg)
    sk = dict(out.get("sample_kwargs", {}))
    if draws > 0:
        sk["draws"] = int(draws)
    if tune > 0:
        sk["tune"] = int(tune)
    if chains > 0:
        sk["chains"] = int(chains)
    sk["cores"] = 1
    out["sample_kwargs"] = sk
    return out


def _compute_cvll_between_slsd_models(
    ts_30s: np.ndarray,
    fs: float,
    model_a_cfg: Dict[str, Any],
    model_b_cfg: Dict[str, Any],
    cv_folds: int,
    cv_chunk_dur: float,
    cv_nw: float,
    cv_k_tapers: int,
) -> Dict[str, float]:
    x = np.asarray(ts_30s, float).ravel()
    n_chunk = int(round(float(fs) * float(cv_chunk_dur)))
    n_expect = int(cv_folds) * n_chunk
    if x.size < n_expect:
        return {"model_a": np.nan, "model_b": np.nan}
    if x.size > n_expect:
        x = x[:n_expect]

    chunks = [x[i * n_chunk:(i + 1) * n_chunk] for i in range(int(cv_folds))]

    f0, _ = _mt_power_one_window(chunks[0], fs=fs, duration=cv_chunk_dur, nw=cv_nw, k_tapers=cv_k_tapers)
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
        return {"model_a": np.nan, "model_b": np.nan}

    cvll_a = 0.0
    cvll_b = 0.0

    for i_test in range(int(cv_folds)):
        test = np.asarray(S_chunks[i_test, m_fr], float)
        train_idx = [j for j in range(int(cv_folds)) if j != i_test]
        train = np.mean(S_chunks[train_idx, :], axis=0)[m_fr]

        try:
            sl_a = Decompose(f_cv, np.clip(train, 1e-20, np.inf), fs=fs, **model_a_cfg)
            mu_a, _bb_a, _rh_a = _extract_slsd(sl_a)
            ll_a = _gamma_loglik_multitaper(test, mu_a, cv_k_tapers)
            cvll_a = cvll_a + ll_a if np.isfinite(ll_a) else np.nan
        except Exception:
            cvll_a = np.nan

        try:
            sl_b = Decompose(f_cv, np.clip(train, 1e-20, np.inf), fs=fs, **model_b_cfg)
            mu_b, _bb_b, _rh_b = _extract_slsd(sl_b)
            ll_b = _gamma_loglik_multitaper(test, mu_b, cv_k_tapers)
            cvll_b = cvll_b + ll_b if np.isfinite(ll_b) else np.nan
        except Exception:
            cvll_b = np.nan

    return {"model_a": float(cvll_a), "model_b": float(cvll_b)}


def _load_raw_session() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    ecog = loadmat(ECOG_MAT, squeeze_me=True)
    time = loadmat(TIME_MAT, squeeze_me=True)
    x = np.asarray(ecog[ECOG_KEY], float).squeeze()
    t = np.asarray(time[TIME_KEY], float).squeeze()
    mvalid = np.isfinite(x) & np.isfinite(t)
    x, t = x[mvalid], t[mvalid]

    cond = loadmat(COND_MAT, simplify_cells=True)["Condition"]
    ct = np.asarray(cond[COND_TIME_KEY], float).ravel()
    raw_labels = np.ravel(cond[COND_LABEL_KEY])
    labels = [lab.decode("utf-8") if isinstance(lab, (bytes, bytearray)) else str(lab) for lab in raw_labels]
    order = np.argsort(ct)
    return x, t, ct[order], [labels[i] for i in order]


def _prepare_state_windows(name: str) -> Dict[str, Any]:
    x, t, ct, labels = _load_raw_session()
    fs = 1000.0
    seg = _find_interval(ct, labels, SEGMENT_CFG[name]["start_phrase"], SEGMENT_CFG[name]["end_phrase"])
    if seg is None:
        raise RuntimeError(f"Could not find interval for {name}.")
    t0_seg, t1_seg = seg
    t_seg, x_seg = _restrict_to_interval(t, x, t0_seg, t1_seg)

    P_tf, F_all, T_abs = compute_multitaper(x_seg, fs=fs, t0=float(t_seg[0]), params=MT_PARAMS)
    P_tf, F_all, T_abs = _ensure_tf(P_tf, F_all, T_abs)

    mask = (F_all > 0) & (F_all >= ANALYSIS_FRANGE[0]) & (F_all <= ANALYSIS_FRANGE[1])
    F_an = F_all[mask]
    P_an_tf = P_tf[:, mask]
    F_fit, step = _independent_grid(F_an, MT_PARAMS["time_window_duration"], MT_PARAMS["time_halfbandwidth_product"])
    P_fit_tf = P_an_tf[:, ::step]
    T_rel_min = (T_abs - float(t_seg[0])) / 60.0

    return {
        "name": name,
        "fs": fs,
        "x_seg": x_seg,
        "t_seg": t_seg,
        "F_fit": F_fit,
        "P_fit_tf": P_fit_tf,
        "T_rel_min": T_rel_min,
    }


def _apply_window_drop_simple(state: Dict[str, Any], drop_orig: List[int]) -> Dict[str, Any]:
    out = dict(state)
    n_wins = int(np.asarray(out["P_fit_tf"]).shape[0])
    orig = np.arange(n_wins, dtype=int)
    drop_set = set(int(i) for i in (drop_orig or []))
    keep_mask = np.array([i not in drop_set for i in orig], dtype=bool)
    if keep_mask.sum() == 0:
        raise RuntimeError(f"All windows were dropped for {state['name']}.")
    keep_idx = np.where(keep_mask)[0].astype(int)
    out["P_fit_tf"] = np.asarray(out["P_fit_tf"])[keep_idx, :]
    out["T_rel_min"] = np.asarray(out["T_rel_min"])[keep_idx]
    out["keep_orig"] = keep_idx.tolist()
    out["drop_orig"] = sorted(drop_set)
    return out


def compute_state_metrics(
    name: str,
    *,
    drop_orig: Optional[List[int]] = None,
    main_sl_draws: int = -1,
    main_sl_tune: int = -1,
    main_sl_chains: int = -1,
    cv_folds: int = CV_FOLDS,
    cv_chunk_dur: Optional[float] = None,
    cv_nw: float = CV_NW,
    cv_k_tapers: int = CV_K_TAPERS,
    cv_sl_draws: int = -1,
    cv_sl_tune: int = -1,
    cv_sl_chains: int = -1,
) -> Dict[str, Any]:
    if cv_chunk_dur is None:
        cv_chunk_dur = float(MT_PARAMS["time_window_duration"]) / float(cv_folds)

    state = _prepare_state_windows(name)
    state = _apply_window_drop_simple(state, list(drop_orig or []))

    F_fit = np.asarray(state["F_fit"], float)
    P_fit_tf = np.asarray(state["P_fit_tf"], float)
    T_rel_min = np.asarray(state["T_rel_min"], float)
    x_seg = np.asarray(state["x_seg"], float)
    fs = float(state["fs"])

    slope_cfg = SLOPE_MODEL_CFG[name]
    specparam_kwargs = dict(slope_cfg["specparam_kwargs"])
    slsd_kwargs = _with_sample_overrides(slope_cfg["slsd_kwargs"], main_sl_draws, main_sl_tune, main_sl_chains)

    cv_cfg_a = _with_sample_overrides(CV_COMPARE_CFG["two_rhythm_no_20_30"], cv_sl_draws, cv_sl_tune, cv_sl_chains)
    cv_cfg_b = _with_sample_overrides(CV_COMPARE_CFG["three_rhythm_with_20_30"], cv_sl_draws, cv_sl_tune, cv_sl_chains)

    n_wins = P_fit_tf.shape[0]
    slopes_specparam = np.full(n_wins, np.nan)
    slopes_slsd = np.full(n_wins, np.nan)
    slopes_naive = np.full(n_wins, np.nan)
    cvll_two = np.full(n_wins, np.nan)
    cvll_three = np.full(n_wins, np.nan)

    win_step = float(MT_PARAMS["time_window_step"])
    win_dur = float(MT_PARAMS["time_window_duration"])
    n_win = int(round(win_dur * fs))

    for ti in range(n_wins):
        y = np.clip(P_fit_tf[ti, :], 1e-20, np.inf)
        fr_k = (max(ANALYSIS_FRANGE[0], float(F_fit[0])), min(ANALYSIS_FRANGE[1], float(F_fit[-1])))

        slopes_naive[ti] = _compute_loglog_slope(F_fit, y, *SLOPE_BAND)

        _sp_full, sp_ap = _specparam_full_aper(F_fit, y, fr_k, **specparam_kwargs)
        slopes_specparam[ti] = _compute_loglog_slope(F_fit, sp_ap, *SLOPE_BAND)

        sl = Decompose(F_fit, y, fs=fs, **slsd_kwargs)
        _sl_tot, sl_bb, _sl_rh = _extract_slsd(sl)
        slopes_slsd[ti] = _compute_loglog_slope(F_fit, sl_bb, *SLOPE_BAND)

        orig_i = int(state["keep_orig"][ti])
        i0 = int(round(orig_i * win_step * fs))
        i1 = i0 + n_win
        if i0 >= 0 and i1 <= x_seg.size:
            x_win = np.asarray(x_seg[i0:i1], float).ravel()
            cvll = _compute_cvll_between_slsd_models(
                ts_30s=x_win,
                fs=fs,
                model_a_cfg=cv_cfg_a,
                model_b_cfg=cv_cfg_b,
                cv_folds=int(cv_folds),
                cv_chunk_dur=float(cv_chunk_dur),
                cv_nw=float(cv_nw),
                cv_k_tapers=int(cv_k_tapers),
            )
            cvll_two[ti] = cvll["model_a"]
            cvll_three[ti] = cvll["model_b"]

    m_both = np.isfinite(cvll_two) & np.isfinite(cvll_three)
    pct_three_higher = float(100.0 * np.mean(cvll_three[m_both] > cvll_two[m_both])) if np.any(m_both) else np.nan

    return {
        "name": name,
        "F_fit": F_fit,
        "T_rel_min": T_rel_min,
        "keep_orig": np.asarray(state["keep_orig"], int),
        "drop_orig": np.asarray(state["drop_orig"], int),
        "slopes_specparam": slopes_specparam,
        "slopes_slsd": slopes_slsd,
        "slopes_naive": slopes_naive,
        "cvll_two": cvll_two,
        "cvll_three": cvll_three,
        "pct_three_higher": pct_three_higher,
        "cv_folds": int(cv_folds),
        "cv_chunk_dur": float(cv_chunk_dur),
        "cv_nw": float(cv_nw),
        "cv_k_tapers": int(cv_k_tapers),
    }


# ──────────────────────────── Plot helpers ────────────────────────────
def _cvll_compare_panel(
    ax: plt.Axes,
    cvll_two: np.ndarray,
    cvll_three: np.ndarray,
    *,
    title: str,
    pct_three_higher: float,
    point_size: float = 18.0,
    star_size: float = 85.0,
    x_step: float = 0.55,
    x_jitter: float = 0.025,
    seed: int = 0,
) -> None:
    a = np.asarray(cvll_two, float).ravel()
    b = np.asarray(cvll_three, float).ravel()
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
        ax.plot([x0[i], x1[i]], [a[i], b[i]], color=COLORS["pair_line"], alpha=0.28, lw=1.0, zorder=1)

    ax.scatter(x0, a, s=point_size, alpha=0.90, marker="o", color=COLORS["cv_two"], edgecolor="none", zorder=3)
    ax.scatter(x1, b, s=point_size, alpha=0.90, marker="o", color=COLORS["cv_three"], edgecolor="none", zorder=3)

    y = np.column_stack([a, b])
    row_max = np.nanmax(y, axis=1, keepdims=True)
    is_max = np.isclose(y, row_max, rtol=0.0, atol=0.0)

    if np.any(is_max[:, 0]):
        ax.scatter(
            x0[is_max[:, 0]], a[is_max[:, 0]],
            s=star_size, marker="*", color=COLORS["cv_two"], edgecolor="k", linewidths=0.5, alpha=0.98, zorder=4,
        )
    if np.any(is_max[:, 1]):
        ax.scatter(
            x1[is_max[:, 1]], b[is_max[:, 1]],
            s=star_size, marker="*", color=COLORS["cv_three"], edgecolor="k", linewidths=0.5, alpha=0.98, zorder=4,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(["2 rhythms\n(no 20-30)", "3 rhythms\n(with 20-30)"], rotation=0)
    ax.set_xlabel("SL_specdecomp model")
    ax.set_ylabel("CVLL")
    ax.set_title(f"{title}\nWindowwise CVLL")
    sns.despine(ax=ax, top=True, right=True)
    ax.minorticks_off()
    ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.35)

    if np.isfinite(pct_three_higher):
        txt = f"3 rhythms higher in {pct_three_higher:.1f}% of windows"
    else:
        txt = "3 rhythms higher: n/a"
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
    )


def _single_model_violin(ax: plt.Axes, awake_vals: np.ndarray, anes_vals: np.ndarray, *, color: Any, title: str) -> None:
    df = pd.DataFrame({
        "value": np.concatenate([awake_vals, anes_vals]),
        "state": (["Awake"] * len(awake_vals)) + (["Anesthetized"] * len(anes_vals)),
    })
    sns.violinplot(
        data=df,
        x="state",
        y="value",
        inner="quartile",
        cut=4,
        bw="scott",
        linewidth=1.0,
        width=0.9,
        palette=[color, color],
        ax=ax,
    )
    sns.stripplot(data=df, x="state", y="value", color="k", alpha=0.55, size=4, jitter=0.10, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("40–60 Hz slope")
    sns.despine(ax=ax, top=True, right=True)


def build_figure(awake: Dict[str, Any], anesthesia: Dict[str, Any]) -> plt.Figure:
    slope_ylim = _metric_ylim_from_values(
        awake["slopes_specparam"],
        anesthesia["slopes_specparam"],
        awake["slopes_slsd"],
        anesthesia["slopes_slsd"],
        awake["slopes_naive"],
        anesthesia["slopes_naive"],
        pad_frac=0.05,
    )

    fig = plt.figure(figsize=(18.0, 10.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.45, wspace=0.9)

    ax11 = fig.add_subplot(gs[0, 0:2])
    ax12 = fig.add_subplot(gs[0, 2:4])
    ax13 = fig.add_subplot(gs[0, 4:6])
    ax21 = fig.add_subplot(gs[1, 0:3])
    ax22 = fig.add_subplot(gs[1, 3:6])

    _single_model_violin(
        ax11,
        awake["slopes_specparam"],
        anesthesia["slopes_specparam"],
        color=COLORS["specparam"],
        title="specparam aperiodic slope (40–60 Hz)",
    )
    _single_model_violin(
        ax12,
        awake["slopes_slsd"],
        anesthesia["slopes_slsd"],
        color=COLORS["slsd"],
        title="SL_specdecomp broadband slope (40–60 Hz)",
    )
    _single_model_violin(
        ax13,
        awake["slopes_naive"],
        anesthesia["slopes_naive"],
        color=COLORS["naive"],
        title="naive OLS slope (40–60 Hz)",
    )

    if slope_ylim is not None:
        for ax in (ax11, ax12, ax13):
            ax.set_ylim(*slope_ylim)

    _cvll_compare_panel(
        ax21,
        awake["cvll_two"], awake["cvll_three"],
        title="Awake",
        pct_three_higher=float(awake["pct_three_higher"]),
        seed=0,
    )
    _cvll_compare_panel(
        ax22,
        anesthesia["cvll_two"], anesthesia["cvll_three"],
        title="Anesthetized",
        pct_three_higher=float(anesthesia["pct_three_higher"]),
        seed=1,
    )

    fig.suptitle(
        "Aux broadband-slope summary + SL_specdecomp CVLL model comparison\n"
        f"30 s multitaper windows (K={MT_PARAMS['n_tapers']}, NW={MT_PARAMS['time_halfbandwidth_product']}); "
        f"CV = {awake['cv_folds']} x {awake['cv_chunk_dur']:.0f} s (K={awake['cv_k_tapers']}, NW={awake['cv_nw']})",
        y=0.99,
        fontsize=14,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    return fig


# ──────────────────────────── Main runner ────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build an auxiliary figure with broadband slope panels grouped by model and "
            "SL_specdecomp CVLL comparisons between 2-rhythm (no 20-30) and 3-rhythm (with 20-30) fits."
        )
    )
    ap.add_argument("--plot-only", action="store_true", help="Skip all compute; regenerate figure from saved payload.")
    ap.add_argument("--force-recompute", action="store_true", help="Ignore payload cache and recompute everything.")
    ap.set_defaults(save_payload=True)
    ap.add_argument("--no-payload", dest="save_payload", action="store_false", help="Do not save NPZ + JSON payload.")

    ap.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    ap.add_argument("--cv-chunk-dur", type=float, default=None, help="Seconds; default is 30 / cv_folds.")
    ap.add_argument("--cv-nw", type=float, default=CV_NW)
    ap.add_argument("--cv-k-tapers", type=int, default=CV_K_TAPERS)

    ap.add_argument("--main-sl-draws", type=int, default=-1)
    ap.add_argument("--main-sl-tune", type=int, default=-1)
    ap.add_argument("--main-sl-chains", type=int, default=-1)

    ap.add_argument("--cv-sl-draws", type=int, default=-1)
    ap.add_argument("--cv-sl-tune", type=int, default=-1)
    ap.add_argument("--cv-sl-chains", type=int, default=-1)

    ap.add_argument("--drop-awake", type=str, default="", help="Comma-separated ORIGINAL awake window indices to drop.")
    ap.add_argument("--drop-anes", type=str, default="", help="Comma-separated ORIGINAL anesthesia window indices to drop.")
    ap.add_argument("--drop-from-outlier-csv", type=str, default="", help="Path to outliers_report.csv from the outlier inspector.")

    args = ap.parse_args()


    # --- Figure 4 outlier exclusion begin ---
    drop_awake = _parse_int_list(getattr(args, "drop_awake", ""))
    drop_anes = _parse_int_list(getattr(args, "drop_anes", ""))

    outlier_csv = str(getattr(args, "drop_from_outlier_csv", "") or "").strip()
    if outlier_csv:
        if not os.path.exists(os.path.expanduser(outlier_csv)):
            raise FileNotFoundError(f"--drop-from-outlier-csv was set but does not exist: {outlier_csv}")
        drop_awake = sorted(set(drop_awake).union(_read_drop_from_outlier_csv(outlier_csv, "awake")))
        drop_anes = sorted(set(drop_anes).union(_read_drop_from_outlier_csv(outlier_csv, "anesthesia")))

    print(f"[INFO] Outlier CSV: {outlier_csv or 'none found / none used'}")
    print(f"[INFO] Dropping awake original window indices: {drop_awake}")
    print(f"[INFO] Dropping anesthesia original window indices: {drop_anes}")
    # --- Figure 4 outlier exclusion end ---

    if not args.plot_only or args.force_recompute:
        for pth in [ECOG_MAT, TIME_MAT, COND_MAT]:
            if not os.path.exists(pth):
                raise FileNotFoundError(f"Missing required file: {pth}")


    out_base = _payload_base()

    if args.plot_only and not args.force_recompute:
        arrays, meta = load_plot_payload(out_base)
        awake = {
            "name": "awake",
            "F_fit": arrays["awake_F_fit"],
            "T_rel_min": arrays["awake_T_rel_min"],
            "keep_orig": arrays["awake_keep_orig"],
            "drop_orig": arrays["awake_drop_orig"],
            "slopes_specparam": arrays["awake_slopes_specparam"],
            "slopes_slsd": arrays["awake_slopes_slsd"],
            "slopes_naive": arrays["awake_slopes_naive"],
            "cvll_two": arrays["awake_cvll_two"],
            "cvll_three": arrays["awake_cvll_three"],
            "pct_three_higher": float(meta["meta"]["awake_pct_three_higher"]),
            "cv_folds": int(meta["meta"]["cv_folds"]),
            "cv_chunk_dur": float(meta["meta"]["cv_chunk_dur"]),
            "cv_nw": float(meta["meta"]["cv_nw"]),
            "cv_k_tapers": int(meta["meta"]["cv_k_tapers"]),
        }
        anesthesia = {
            "name": "anesthesia",
            "F_fit": arrays["anesthesia_F_fit"],
            "T_rel_min": arrays["anesthesia_T_rel_min"],
            "keep_orig": arrays["anesthesia_keep_orig"],
            "drop_orig": arrays["anesthesia_drop_orig"],
            "slopes_specparam": arrays["anesthesia_slopes_specparam"],
            "slopes_slsd": arrays["anesthesia_slopes_slsd"],
            "slopes_naive": arrays["anesthesia_slopes_naive"],
            "cvll_two": arrays["anesthesia_cvll_two"],
            "cvll_three": arrays["anesthesia_cvll_three"],
            "pct_three_higher": float(meta["meta"]["anesthesia_pct_three_higher"]),
            "cv_folds": int(meta["meta"]["cv_folds"]),
            "cv_chunk_dur": float(meta["meta"]["cv_chunk_dur"]),
            "cv_nw": float(meta["meta"]["cv_nw"]),
            "cv_k_tapers": int(meta["meta"]["cv_k_tapers"]),
        }
    else:
        awake = compute_state_metrics(
            "awake",
            drop_orig=drop_awake,
            main_sl_draws=int(args.main_sl_draws),
            main_sl_tune=int(args.main_sl_tune),
            main_sl_chains=int(args.main_sl_chains),
            cv_folds=int(args.cv_folds),
            cv_chunk_dur=args.cv_chunk_dur,
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            cv_sl_draws=int(args.cv_sl_draws),
            cv_sl_tune=int(args.cv_sl_tune),
            cv_sl_chains=int(args.cv_sl_chains),
        )
        anesthesia = compute_state_metrics(
            "anesthesia",
            drop_orig=drop_anes,
            main_sl_draws=int(args.main_sl_draws),
            main_sl_tune=int(args.main_sl_tune),
            main_sl_chains=int(args.main_sl_chains),
            cv_folds=int(args.cv_folds),
            cv_chunk_dur=args.cv_chunk_dur,
            cv_nw=float(args.cv_nw),
            cv_k_tapers=int(args.cv_k_tapers),
            cv_sl_draws=int(args.cv_sl_draws),
            cv_sl_tune=int(args.cv_sl_tune),
            cv_sl_chains=int(args.cv_sl_chains),
        )

        if bool(args.save_payload):
            arrays = {
                "awake_F_fit": awake["F_fit"],
                "awake_T_rel_min": awake["T_rel_min"],
                "awake_keep_orig": awake["keep_orig"],
                "awake_drop_orig": awake["drop_orig"],
                "awake_slopes_specparam": awake["slopes_specparam"],
                "awake_slopes_slsd": awake["slopes_slsd"],
                "awake_slopes_naive": awake["slopes_naive"],
                "awake_cvll_two": awake["cvll_two"],
                "awake_cvll_three": awake["cvll_three"],
                "anesthesia_F_fit": anesthesia["F_fit"],
                "anesthesia_T_rel_min": anesthesia["T_rel_min"],
                "anesthesia_keep_orig": anesthesia["keep_orig"],
                "anesthesia_drop_orig": anesthesia["drop_orig"],
                "anesthesia_slopes_specparam": anesthesia["slopes_specparam"],
                "anesthesia_slopes_slsd": anesthesia["slopes_slsd"],
                "anesthesia_slopes_naive": anesthesia["slopes_naive"],
                "anesthesia_cvll_two": anesthesia["cvll_two"],
                "anesthesia_cvll_three": anesthesia["cvll_three"],
            }
            meta = {
                "analysis_frange": list(ANALYSIS_FRANGE),
                "slope_band": list(SLOPE_BAND),
                "MT_PARAMS": MT_PARAMS,
                "cv_folds": int(awake["cv_folds"]),
                "cv_chunk_dur": float(awake["cv_chunk_dur"]),
                "cv_nw": float(awake["cv_nw"]),
                "cv_k_tapers": int(awake["cv_k_tapers"]),
                "awake_pct_three_higher": float(awake["pct_three_higher"]),
                "anesthesia_pct_three_higher": float(anesthesia["pct_three_higher"]),
                "slope_model_cfg": SLOPE_MODEL_CFG,
                "cv_compare_cfg": CV_COMPARE_CFG,
                "drop_awake": drop_awake,
                "drop_anes": drop_anes,
            }
            save_plot_payload(out_base, arrays=arrays, meta=meta)

    fig = build_figure(awake, anesthesia)

    png = str(out_base.with_suffix(".png"))
    svg = str(out_base.with_suffix(".svg"))
    fig.savefig(png, dpi=300)
    fig.savefig(svg, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved figure -> {png}")
    print(f"[INFO] Saved figure -> {svg}")


if __name__ == "__main__":
    main()
