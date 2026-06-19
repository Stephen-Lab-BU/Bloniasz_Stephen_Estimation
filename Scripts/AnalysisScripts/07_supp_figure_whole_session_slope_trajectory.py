#!/usr/bin/env python3
"""
Generate the supplemental whole-session slope trajectory figure.

The script reads Figure 4 empirical payloads, computes or loads 40--60 Hz
broadband slope trajectories for retained windows, marks excluded artifact
windows, and writes whole-epoch trajectory diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat


PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
try:
    from spectral_connectivity import Connectivity, Multitaper
except Exception:  # pragma: no cover
    Connectivity = None
    Multitaper = None

try:
    from specparam import SpectralModel
except Exception:  # pragma: no cover
    SpectralModel = None

try:
    from SL_specdecomp import Decompose
except Exception:  # pragma: no cover
    Decompose = None

try:
    import pymc as pm
except Exception:  # pragma: no cover
    pm = None


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
ROOT = PROJECT_ROOT
DATA_DIR = ROOT / "Data" / "InputData" / "InputDataFiles"
FIG4_DIR_DEFAULT = ROOT / "Output" / "Results" / "FiguresIntermediate" / "Figure_4_CV" / "Figure_output"
SAVE_DIR_DEFAULT = ROOT / "Output" / "Results" / "FiguresIntermediate" / "Supp_Figure_WholeSessionSlopeTrajectory" / "Figure_output"

ECOG_MAT = DATA_DIR / "ECoG_ch1.mat"
TIME_MAT = DATA_DIR / "ECoGTime.mat"
COND_MAT = DATA_DIR / "Condition.mat"

ECOG_KEY = "ECoG_ch1"
TIME_KEY = "ECoGTime"
COND_STRUCT_KEY = "Condition"
COND_TIME_KEY = "ConditionTime"
COND_LABEL_KEY = "ConditionLabel"

FS = 1000.0
ANALYSIS_FRANGE = (0.1, 200.0)
SLOPE_BAND = (40.0, 60.0)
PSD_YLIM = (1e-1, 1e6)

MT_PARAMS = dict(
    time_halfbandwidth_product=2,
    n_tapers=3,
    time_window_duration=30.0,
    time_window_step=30.0,
)

STATE_CFG = {
    "awake": {
        "title": "Awake",
        "start_phrase": "AwakeEyesClosed-Start",
        "end_phrase": "AwakeEyesClosed-End",
        "target_min_after_start": 17.29,
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
        "title": "Anesthesia",
        "start_phrase": "Anesthetized Start",
        "end_phrase": "Anesthetized End",
        "target_min_after_start": 4.0,
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

# Match the current Figure 4 prior override.
B_PRIOR_SIGMA = 5.0
A_HEIGHT_ANCHOR_Q = 50.0
A_HEIGHT_PRIOR_SIGMA = 1.25

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
    "naive": "0.42",
    "specparam": PALETTE[0],
    "slsd": PALETTE[1],
    "drop_span": "#d62728",
    "target_box": "k",
    "multitaper": "0.35",
    "full": "#000000",
    "broadband": "#ff7f0e",
    "rhythms": "#d62728",
}
METHOD_ORDER = ["Naive OLS", "specparam", "SL_specdecomp"]
METHOD_TO_KEY = {
    "Naive OLS": "slope_naive",
    "specparam": "slope_specparam",
    "SL_specdecomp": "slope_slsd",
}
METHOD_TO_COLOR = {
    "Naive OLS": COLORS["naive"],
    "specparam": COLORS["specparam"],
    "SL_specdecomp": COLORS["slsd"],
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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


def _normalize_label(text: str) -> str:
    text = str(text).lower().replace("-", " ").replace("_", " ")
    return " ".join(text.split())


def _compute_loglog_slope(freqs: np.ndarray, power_lin: np.ndarray, fmin: float, fmax: float) -> float:
    f = np.asarray(freqs, float).ravel()
    y = np.asarray(power_lin, float).ravel()
    mask = (f >= float(fmin)) & (f <= float(fmax)) & np.isfinite(y) & (y > 0)
    if int(mask.sum()) < 2:
        return np.nan
    x = np.log10(f[mask])
    z = np.log10(y[mask])
    m, _b = np.polyfit(x, z, 1)
    return float(m)


def _safe_slope_array(freqs: np.ndarray, curves: Optional[np.ndarray]) -> np.ndarray:
    if curves is None:
        return np.array([], dtype=float)
    arr = np.asarray(curves, float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.array([_compute_loglog_slope(freqs, row, *SLOPE_BAND) for row in arr], dtype=float)


def _load_npz_and_meta(npz_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Path]:
    npz_path = Path(npz_path).expanduser().resolve()
    meta_path = Path(str(npz_path).replace(".plotdata.npz", ".plotmeta.json"))
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing Figure 6 plotdata file: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing paired Figure 6 plotmeta file: {meta_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    with open(meta_path, "r") as f:
        meta_file = json.load(f)
    meta = meta_file.get("meta", meta_file)
    return arrays, meta, meta_path


def _estimate_center_intercept(t_rel_min: np.ndarray, keep_orig: Sequence[int], step_min: float) -> float:
    t = np.asarray(t_rel_min, float).ravel()
    k = np.asarray(list(keep_orig), float).ravel()
    if t.size and k.size and t.size == k.size:
        vals = t - k * float(step_min)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            return float(np.median(vals))
    # Standard non-overlapping 30 s multitaper windows report the center time.
    return float(0.5 * MT_PARAMS["time_window_duration"] / 60.0)


def _window_span_from_orig_index(
    orig_idx: int,
    center0_min: float,
    step_min: float,
    duration_min: float,
) -> Tuple[float, float, float]:
    center = float(center0_min) + int(orig_idx) * float(step_min)
    return center - 0.5 * float(duration_min), center, center + 0.5 * float(duration_min)


# -----------------------------------------------------------------------------
# Figure 4 payload handling
# -----------------------------------------------------------------------------
@dataclass
class StatePayload:
    state: str
    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any]
    meta_path: Path
    fig4_npz_path: Path
    F_fit: np.ndarray
    P_fit_tf: np.ndarray
    T_rel_min: np.ndarray
    slope_naive: np.ndarray
    slope_specparam: np.ndarray
    slope_slsd: np.ndarray
    idx_target: int
    keep_orig: List[int]
    drop_orig: List[int]
    center0_min: float
    step_min: float
    duration_min: float


def _load_state_payload(
    state: str,
    fig4_dir: Path,
    manual_drop: Optional[Sequence[int]] = None,
    csv_drop: Optional[Sequence[int]] = None,
) -> StatePayload:
    fig4_dir = Path(fig4_dir).expanduser().resolve()
    npz_path = fig4_dir / f"Figure_4_CV_{state}.plotdata.npz"
    arrays, meta, meta_path = _load_npz_and_meta(npz_path)

    F_fit = np.asarray(arrays["F_fit"], float).ravel()
    P_fit_tf = np.asarray(arrays["P_fit_tf"], float)
    T_rel_min = np.asarray(arrays["T_rel_min"], float).ravel()
    if P_fit_tf.ndim != 2:
        raise ValueError(f"{state}: P_fit_tf must be 2D, got {P_fit_tf.shape}")

    n = int(P_fit_tf.shape[0])
    if T_rel_min.size != n:
        raise ValueError(f"{state}: T_rel_min length {T_rel_min.size} does not match P_fit_tf rows {n}")

    keep_orig_meta = meta.get("keep_orig", None)
    keep_orig = [int(x) for x in keep_orig_meta] if keep_orig_meta is not None else list(range(n))
    if len(keep_orig) != n:
        raise ValueError(f"{state}: keep_orig length {len(keep_orig)} does not match payload rows {n}")

    drop_orig = set(int(x) for x in meta.get("drop_orig", []) or [])
    drop_orig.update(int(x) for x in (manual_drop or []))
    drop_orig.update(int(x) for x in (csv_drop or []))
    drop_orig = sorted(drop_orig)

    mt_meta = dict(meta.get("MT_PARAMS", {}) or {})
    step_min = float(mt_meta.get("time_window_step", MT_PARAMS["time_window_step"])) / 60.0
    duration_min = float(mt_meta.get("time_window_duration", MT_PARAMS["time_window_duration"])) / 60.0
    center0_min = _estimate_center_intercept(T_rel_min, keep_orig, step_min)

    slope_naive = _safe_slope_array(F_fit, P_fit_tf)

    if "slopes_specparam" in arrays:
        slope_specparam = np.asarray(arrays["slopes_specparam"], float).ravel()
    else:
        slope_specparam = _safe_slope_array(F_fit, arrays.get("specparam_aper"))

    if "slopes_slsd" in arrays:
        slope_slsd = np.asarray(arrays["slopes_slsd"], float).ravel()
    else:
        slope_slsd = _safe_slope_array(F_fit, arrays.get("slsd_bb"))

    for name, arr in [("slope_naive", slope_naive), ("slope_specparam", slope_specparam), ("slope_slsd", slope_slsd)]:
        if arr.size != n:
            raise ValueError(f"{state}: {name} length {arr.size} does not match payload rows {n}")

    idx_target = int(meta.get("idx_target", int(np.nanargmin(np.abs(T_rel_min - STATE_CFG[state]["target_min_after_start"])))))
    idx_target = int(np.clip(idx_target, 0, max(n - 1, 0)))

    return StatePayload(
        state=state,
        arrays=arrays,
        meta=meta,
        meta_path=meta_path,
        fig4_npz_path=npz_path,
        F_fit=F_fit,
        P_fit_tf=P_fit_tf,
        T_rel_min=T_rel_min,
        slope_naive=slope_naive,
        slope_specparam=slope_specparam,
        slope_slsd=slope_slsd,
        idx_target=idx_target,
        keep_orig=keep_orig,
        drop_orig=drop_orig,
        center0_min=center0_min,
        step_min=step_min,
        duration_min=duration_min,
    )


def _build_long_df(payloads: Dict[str, StatePayload]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for state, p in payloads.items():
        method_arrays = {
            "Naive OLS": p.slope_naive,
            "specparam": p.slope_specparam,
            "SL_specdecomp": p.slope_slsd,
        }
        for method, values in method_arrays.items():
            for local_i, (t, value, orig_i) in enumerate(zip(p.T_rel_min, values, p.keep_orig)):
                rows.append({
                    "state": state,
                    "state_label": STATE_CFG[state]["title"],
                    "method": method,
                    "time_rel_min": float(t),
                    "slope": float(value),
                    "local_index": int(local_i),
                    "orig_index": int(orig_i),
                    "is_target": bool(local_i == p.idx_target),
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main figure plotting
# -----------------------------------------------------------------------------
def _finite_ylim(values: Iterable[np.ndarray], pad_frac: float = 0.10) -> Tuple[float, float]:
    chunks = []
    for v in values:
        arr = np.asarray(v, float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return (-4.0, 1.0)
    vals = np.concatenate(chunks)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    return lo - pad_frac * span, hi + pad_frac * span


def _draw_removed_window_spans(ax: plt.Axes, p: StatePayload, label_once: bool = True) -> None:
    labeled = False
    for orig_i in p.drop_orig:
        start, center, stop = _window_span_from_orig_index(orig_i, p.center0_min, p.step_min, p.duration_min)
        ax.axvspan(
            start,
            stop,
            facecolor=COLORS["drop_span"],
            edgecolor=COLORS["drop_span"],
            alpha=0.16,
            lw=1.0,
            zorder=0,
            label=("Dropped Figure 6 window" if label_once and not labeled else None),
        )
        ax.axvline(center, color=COLORS["drop_span"], lw=0.8, alpha=0.35, zorder=1)
        labeled = True


def _draw_target_box(ax: plt.Axes, p: StatePayload, ylims: Tuple[float, float], label_once: bool = True) -> None:
    t = float(p.T_rel_min[p.idx_target])
    start = t - 0.5 * p.duration_min
    stop = t + 0.5 * p.duration_min
    ax.axvspan(
        start,
        stop,
        facecolor="none",
        edgecolor=COLORS["target_box"],
        linewidth=1.7,
        zorder=5,
        label=("Figure 6 highlighted window" if label_once else None),
    )
    ax.set_ylim(*ylims)


def _plot_state_trajectory(ax: plt.Axes, p: StatePayload, ylims: Tuple[float, float]) -> None:
    _draw_removed_window_spans(ax, p)

    series = [
        ("Naive OLS", p.slope_naive),
        ("specparam", p.slope_specparam),
        ("SL_specdecomp", p.slope_slsd),
    ]
    for method, y in series:
        ax.plot(
            p.T_rel_min,
            y,
            marker="o",
            ms=4.2,
            lw=1.6,
            color=METHOD_TO_COLOR[method],
            alpha=0.88,
            label=method,
            zorder=3,
        )

    _draw_target_box(ax, p, ylims)
    ax.scatter(
        [p.T_rel_min[p.idx_target]] * 3,
        [p.slope_naive[p.idx_target], p.slope_specparam[p.idx_target], p.slope_slsd[p.idx_target]],
        marker="s",
        s=54,
        facecolor="white",
        edgecolor="k",
        linewidth=1.0,
        zorder=8,
    )

    ax.set_title(f"{STATE_CFG[p.state]['title']} epoch")
    ax.set_xlabel("Minutes after state start")
    ax.set_ylabel("40–60 Hz slope")
    ax.set_ylim(*ylims)

    x_candidates = [p.T_rel_min]
    for orig_i in p.drop_orig:
        start, _center, stop = _window_span_from_orig_index(orig_i, p.center0_min, p.step_min, p.duration_min)
        x_candidates.append(np.array([start, stop], dtype=float))
    x_all = np.concatenate([np.asarray(x, float).ravel() for x in x_candidates if np.asarray(x).size])
    if x_all.size:
        xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
        pad = max(0.25, 0.04 * (xmax - xmin))
        ax.set_xlim(xmin - pad, xmax + pad)
    sns.despine(ax=ax, top=True, right=True)


def plot_main_figure(payloads: Dict[str, StatePayload], save_dir: Path, stem: str) -> Tuple[Path, Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    long_df = _build_long_df(payloads)

    ylims = _finite_ylim([
        payloads["awake"].slope_naive,
        payloads["awake"].slope_specparam,
        payloads["awake"].slope_slsd,
        payloads["anesthesia"].slope_naive,
        payloads["anesthesia"].slope_specparam,
        payloads["anesthesia"].slope_slsd,
    ])

    fig = plt.figure(figsize=(15.5, 9.8))
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.35, 1.0], hspace=0.34)
    gs_top = outer[0].subgridspec(1, 2, wspace=0.24)
    gs_bot = outer[1].subgridspec(1, 3, wspace=0.34)

    ax_awake = fig.add_subplot(gs_top[0, 0])
    ax_anes = fig.add_subplot(gs_top[0, 1])
    _plot_state_trajectory(ax_awake, payloads["awake"], ylims)
    _plot_state_trajectory(ax_anes, payloads["anesthesia"], ylims)

    # Keep only one clean legend for the top row.
    handles, labels = ax_anes.get_legend_handles_labels()
    label_seen = set()
    uniq_h, uniq_l = [], []
    for h, lab in zip(handles, labels):
        if lab and lab not in label_seen:
            uniq_h.append(h)
            uniq_l.append(lab)
            label_seen.add(lab)
    fig.legend(uniq_h, uniq_l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.015), fontsize=9)

    common_vylim = _finite_ylim([long_df["slope"].to_numpy(float)], pad_frac=0.10)
    for j, method in enumerate(METHOD_ORDER):
        ax = fig.add_subplot(gs_bot[0, j])
        sub = long_df[(long_df["method"] == method) & np.isfinite(long_df["slope"].to_numpy(float))].copy()
        sns.violinplot(
            data=sub,
            x="state_label",
            y="slope",
            order=["Awake", "Anesthesia"],
            inner="quartile",
            cut=0,
            bw_method="scott",
            linewidth=1.0,
            width=0.85,
            color=METHOD_TO_COLOR[method],
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="state_label",
            y="slope",
            order=["Awake", "Anesthesia"],
            color="k",
            alpha=0.48,
            size=4,
            jitter=0.10,
            ax=ax,
        )
        targets = sub[sub["is_target"]]
        for _, row in targets.iterrows():
            xpos = 0 if row["state"] == "awake" else 1
            ax.scatter(
                [xpos],
                [float(row["slope"])],
                marker="s",
                s=86,
                facecolor="white",
                edgecolor="k",
                linewidth=1.0,
                zorder=8,
            )
        ax.set_title(method)
        ax.set_xlabel("")
        ax.set_ylabel("40–60 Hz slope")
        ax.set_ylim(*common_vylim)
        sns.despine(ax=ax, top=True, right=True)

    fig.suptitle(
        "Figure 6 state-specific slope trajectories: naive estimator, specparam, and SL_specdecomp",
        y=1.065,
        fontsize=15,
    )

    png = save_dir / f"{stem}.png"
    svg = save_dir / f"{stem}.svg"
    points_csv = save_dir / f"{stem}_points.csv"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, dpi=300, bbox_inches="tight")
    plt.close(fig)
    long_df.to_csv(points_csv, index=False)
    print(f"[INFO] Saved main figure -> {png}")
    print(f"[INFO] Saved main figure -> {svg}")
    print(f"[INFO] Saved point table  -> {points_csv}")
    return png, svg, points_csv


# -----------------------------------------------------------------------------
# Dropped-window refit support for aux decomposition PNG
# -----------------------------------------------------------------------------
def _check_refit_dependencies() -> None:
    missing = []
    if Multitaper is None or Connectivity is None:
        missing.append("spectral_connectivity")
    if SpectralModel is None:
        missing.append("specparam")
    if Decompose is None:
        missing.append("SL_specdecomp")
    if pm is None:
        missing.append("pymc")
    if missing:
        raise ImportError(
            "Missing required package(s) for dropped-window refits: " + ", ".join(missing) +
            ". Activate the same environment used for Figure 6."
        )


def _load_raw_session() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    ecog = loadmat(ECOG_MAT, squeeze_me=True)
    time = loadmat(TIME_MAT, squeeze_me=True)
    cond = loadmat(COND_MAT, simplify_cells=True)[COND_STRUCT_KEY]

    x = np.asarray(ecog[ECOG_KEY], float).squeeze()
    t = np.asarray(time[TIME_KEY], float).squeeze()
    mvalid = np.isfinite(x) & np.isfinite(t)
    x = x[mvalid]
    t = t[mvalid]

    ct = np.asarray(cond[COND_TIME_KEY], float).ravel()
    raw_labels = np.ravel(cond[COND_LABEL_KEY])
    labels = [lab.decode("utf-8") if isinstance(lab, (bytes, bytearray)) else str(lab) for lab in raw_labels]
    order = np.argsort(ct)
    ct = ct[order]
    labels = [labels[i] for i in order]
    return x, t, ct, labels


def _find_interval(cond_times: np.ndarray, cond_labels: Sequence[str], start_phrase: str, end_phrase: str) -> Tuple[float, float]:
    labs = [_normalize_label(x) for x in cond_labels]
    s_norm = _normalize_label(start_phrase)
    e_norm = _normalize_label(end_phrase)
    start_idx = next((i for i, lab in enumerate(labs) if s_norm in lab), None)
    end_idx = next((i for i, lab in enumerate(labs) if e_norm in lab), None)
    if start_idx is None or end_idx is None:
        raise RuntimeError(f"Could not find interval for {start_phrase!r} -> {end_phrase!r}")
    t0 = float(cond_times[start_idx])
    t1 = float(cond_times[end_idx])
    if t1 <= t0:
        raise RuntimeError(f"Invalid interval for {start_phrase!r} -> {end_phrase!r}: {t0}, {t1}")
    return t0, t1


def _ensure_tf(power: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(power)
    f = np.asarray(freqs).ravel()
    t = np.asarray(times).ravel()
    if p.ndim != 2:
        raise ValueError(f"Spectrogram power must be 2D, got {p.shape}")
    if p.shape == (t.size, f.size):
        return p, f, t
    if p.shape == (f.size, t.size):
        return p.T, f, t
    return p, f, t


def _independent_grid(freqs: np.ndarray, twin: float, nw: float) -> Tuple[np.ndarray, int]:
    f = np.asarray(freqs, float).ravel()
    df = float(np.median(np.diff(f))) if f.size > 1 else 1.0
    delta_f_indep = 2.0 * float(nw) / float(twin)
    step_bins = max(1, int(round(delta_f_indep / max(df, 1e-12))))
    return f[::step_bins], step_bins


def _prepare_unfiltered_state_windows(state: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, t, ct, labels = _load_raw_session()
    cfg = STATE_CFG[state]
    t0, t1 = _find_interval(ct, labels, cfg["start_phrase"], cfg["end_phrase"])
    mask = (t >= t0) & (t <= t1)
    x_seg = np.asarray(x[mask], float).ravel()
    t_seg = np.asarray(t[mask], float).ravel()
    if x_seg.size == 0:
        raise RuntimeError(f"No samples in {state} interval")

    x_3d = x_seg[:, None, None]
    mt = Multitaper(x_3d, sampling_frequency=FS, start_time=float(t_seg[0]), **MT_PARAMS)
    conn = Connectivity.from_multitaper(mt)
    power = conn.power().squeeze()
    freqs = np.asarray(conn.frequencies).ravel()
    times = np.asarray(conn.time).ravel()
    power, freqs, times = _ensure_tf(power, freqs, times)

    mask_f = (freqs > 0) & (freqs >= ANALYSIS_FRANGE[0]) & (freqs <= ANALYSIS_FRANGE[1])
    freqs_an = freqs[mask_f]
    power_an = power[:, mask_f]
    F_fit, step = _independent_grid(freqs_an, MT_PARAMS["time_window_duration"], MT_PARAMS["time_halfbandwidth_product"])
    P_fit_tf = np.asarray(power_an[:, ::step], float)
    return F_fit, P_fit_tf, times


def _b_prior_specs(y_lin: np.ndarray, sigma: float = B_PRIOR_SIGMA) -> Dict[str, Any]:
    y = np.asarray(y_lin, float)
    y_pos = y[np.isfinite(y) & (y > 0)]
    mu_b = float(np.log10(np.median(y_pos))) if y_pos.size else 0.0

    def _factory(name: str, mu: float = mu_b, sigma: float = float(sigma)):
        return pm.Normal(name, mu=mu, sigma=sigma)

    return {"b_0": {"factory": _factory}, "b": {"factory": _factory}}


def _robust_band_scale(freqs: np.ndarray, y_lin: np.ndarray, band: Tuple[float, float], q: float = A_HEIGHT_ANCHOR_Q) -> float:
    f = np.asarray(freqs, float).ravel()
    y = np.asarray(y_lin, float).ravel()
    lo, hi = map(float, band)
    m = np.isfinite(f) & np.isfinite(y) & (y > 0) & (f >= lo) & (f <= hi)
    if int(m.sum()) < 5:
        m = np.isfinite(y) & (y > 0)
    if int(m.sum()) == 0:
        return 1e-12
    return float(np.percentile(y[m], float(q)))


def _height_prior_specs(freqs: np.ndarray, y_lin: np.ndarray, rhythm_bands: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    specs: Dict[str, Any] = {}
    for i, band in enumerate(rhythm_bands):
        band_scale = _robust_band_scale(freqs, y_lin, band, q=A_HEIGHT_ANCHOR_Q)

        def _factory(name: str, band_scale: float = band_scale, sigma: float = A_HEIGHT_PRIOR_SIGMA):
            return pm.LogNormal(name, mu=np.log(max(float(band_scale), 1e-12)), sigma=float(sigma))

        specs[f"A_lin_{i}"] = {"factory": _factory}
    return specs


def _slsd_kwargs_with_figure4_priors(base_kwargs: Dict[str, Any], freqs: np.ndarray, y_lin: np.ndarray, *, draws: int, tune: int, chains: int, seed: int) -> Dict[str, Any]:
    out = dict(base_kwargs)
    sample_kwargs = dict(out.get("sample_kwargs") or {})
    sample_kwargs.update(draws=int(draws), tune=int(tune), chains=int(chains), cores=1, random_seed=int(seed))
    out["sample_kwargs"] = sample_kwargs

    ap_specs = dict(out.get("aperiodic_param_specs") or {})
    ap_specs.update(_b_prior_specs(y_lin, sigma=B_PRIOR_SIGMA))
    out["aperiodic_param_specs"] = ap_specs

    if str(out.get("mode", "additive")) == "additive":
        rhythm_bands = list(out.get("rhythm_bands", []) or [])
        rh_specs = dict(out.get("rhythm_param_specs") or {})
        rh_specs.update(_height_prior_specs(freqs, y_lin, rhythm_bands))
        out["rhythm_param_specs"] = rh_specs
    return out


def _extract_slsd(model: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.asarray(getattr(model, "estimated_spectrum"), float).reshape(-1)
    n_freqs = total.size

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
        if bb.size != n_freqs:
            bb = bb[:n_freqs] if bb.size > n_freqs else np.pad(bb, (0, n_freqs - bb.size))

    rh = getattr(model, "rhythms", None)
    if rh is None:
        rh = getattr(model, "P_rh", None)
    if rh is None:
        rh = getattr(model, "rhythms_total", None)
    if rh is None:
        rh = np.clip(total - bb, 0.0, np.inf)
    else:
        rh = np.asarray(rh, float).reshape(-1)
        if rh.size != n_freqs:
            rh = rh[:n_freqs] if rh.size > n_freqs else np.pad(rh, (0, n_freqs - rh.size))
    return total, bb, rh


def _fit_specparam_curves(freqs: np.ndarray, y: np.ndarray, kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fm = SpectralModel(**kwargs)
    freqs = np.asarray(freqs, float).ravel()
    y = np.clip(np.asarray(y, float).ravel(), 1e-20, np.inf)
    freq_range = (max(ANALYSIS_FRANGE[0], float(freqs[0])), min(ANALYSIS_FRANGE[1], float(freqs[-1])))
    fm.fit(freqs, y, freq_range=freq_range)

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
            freq_model = np.asarray(getattr(fm, "freqs", freqs), float).ravel()
        full_lin = np.interp(freqs, freq_model, full_native)
        ap_lin = np.interp(freqs, freq_model, ap_native)
    else:
        try:
            full_native = np.asarray(fm.get_model("full", space="linear"), float).ravel()
        except Exception:
            full_native = 10.0 ** np.asarray(fm.get_model("full"), float).ravel()
        try:
            ap_native = np.asarray(fm.get_model("aperiodic", space="linear"), float).ravel()
        except Exception:
            ap_native = 10.0 ** np.asarray(fm.get_model("aperiodic"), float).ravel()
        freq_model = np.asarray(getattr(fm, "freqs", freqs), float).ravel()
        full_lin = np.interp(freqs, freq_model, full_native)
        ap_lin = np.interp(freqs, freq_model, ap_native)
    rh_lin = np.clip(full_lin - ap_lin, 0.0, np.inf)
    return full_lin, ap_lin, rh_lin


def _plot_ll(ax: plt.Axes, x: np.ndarray, y: Optional[np.ndarray], *, color: str, label: str, lw: float = 1.8, ls: str = "-") -> None:
    if y is None:
        return
    xx = np.asarray(x, float).ravel()
    yy = np.asarray(y, float).ravel()
    m = np.isfinite(xx) & np.isfinite(yy) & (xx > 0) & (yy > 0)
    if int(m.sum()) == 0:
        return
    ax.plot(xx[m], yy[m], color=color, lw=lw, ls=ls, label=label)


def refit_dropped_windows_for_aux(
    payloads: Dict[str, StatePayload],
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> List[Dict[str, Any]]:
    _check_refit_dependencies()
    rng = np.random.default_rng(int(seed))
    results: List[Dict[str, Any]] = []

    for state, p in payloads.items():
        if not p.drop_orig:
            continue
        print(f"[INFO] Refitting dropped {state} windows for aux decomposition plot: {p.drop_orig}")
        F_fit, P_all, _times = _prepare_unfiltered_state_windows(state)
        for orig_i in p.drop_orig:
            if orig_i < 0 or orig_i >= P_all.shape[0]:
                print(f"[WARN] {state}: dropped window {orig_i} outside unfiltered P_fit_tf rows={P_all.shape[0]}; skipping")
                continue
            y = np.clip(P_all[orig_i, :], 1e-20, np.inf)
            child_seed = int(rng.integers(1, 2**30 - 1))

            sp_full = sp_ap = sp_rh = None
            try:
                sp_full, sp_ap, sp_rh = _fit_specparam_curves(F_fit, y, STATE_CFG[state]["specparam_kwargs"])
            except Exception as exc:
                print(f"[WARN] {state} dropped window {orig_i}: specparam failed: {exc}")

            sl_total = sl_bb = sl_rh = None
            try:
                sl_kwargs = _slsd_kwargs_with_figure4_priors(
                    STATE_CFG[state]["slsd_kwargs"],
                    F_fit,
                    y,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    seed=child_seed,
                )
                sl_model = Decompose(F_fit, y, fs=FS, **sl_kwargs)
                sl_total, sl_bb, sl_rh = _extract_slsd(sl_model)
            except Exception as exc:
                print(f"[WARN] {state} dropped window {orig_i}: SL_specdecomp failed: {exc}")

            start, center, stop = _window_span_from_orig_index(orig_i, p.center0_min, p.step_min, p.duration_min)
            results.append({
                "state": state,
                "orig_index": int(orig_i),
                "time_start_min": float(start),
                "time_center_min": float(center),
                "time_stop_min": float(stop),
                "F_fit": np.asarray(F_fit, float),
                "P": y,
                "sp_full": sp_full,
                "sp_ap": sp_ap,
                "sp_rh": sp_rh,
                "sl_total": sl_total,
                "sl_bb": sl_bb,
                "sl_rh": sl_rh,
                "seed": child_seed,
            })
    return results


def plot_aux_dropped_decompositions(results: List[Dict[str, Any]], save_dir: Path, stem: str) -> Tuple[Path, Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    png = save_dir / f"{stem}_removed_window_decompositions.png"
    svg = save_dir / f"{stem}_removed_window_decompositions.svg"
    csv = save_dir / f"{stem}_removed_window_decompositions.csv"

    summary_rows = []
    if not results:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.text(0.5, 0.5, "No dropped windows found/refit.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(svg, dpi=300, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(summary_rows).to_csv(csv, index=False)
        return png, svg, csv

    states = ["awake", "anesthesia"]
    max_cols = max(1, max(sum(1 for r in results if r["state"] == s) for s in states))
    fig, axes = plt.subplots(len(states), max_cols, figsize=(6.2 * max_cols, 4.7 * len(states)), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for row_i, state in enumerate(states):
        state_results = [r for r in results if r["state"] == state]
        for col_i, r in enumerate(state_results):
            ax = axes[row_i, col_i]
            ax.axis("on")
            F = r["F_fit"]
            P = r["P"]
            ax.set_xscale("log")
            ax.set_yscale("log")
            _plot_ll(ax, F, P, color=COLORS["multitaper"], label="Multitaper", lw=2.0)
            _plot_ll(ax, F, r["sp_full"], color=COLORS["specparam"], label="specparam full", lw=1.7)
            _plot_ll(ax, F, r["sp_ap"], color=COLORS["broadband"], label="specparam aperiodic", lw=1.6, ls="--")
            _plot_ll(ax, F, r["sp_rh"], color=COLORS["rhythms"], label="specparam peaks", lw=1.3, ls=":")
            _plot_ll(ax, F, r["sl_total"], color="k", label="SL_specdecomp full", lw=1.8)
            _plot_ll(ax, F, r["sl_bb"], color="#9467bd", label="SL_specdecomp broadband", lw=1.6, ls="--")
            _plot_ll(ax, F, r["sl_rh"], color="#8c564b", label="SL_specdecomp rhythms", lw=1.3, ls=":")
            ax.set_ylim(*PSD_YLIM)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.set_title(
                f"{STATE_CFG[state]['title']} dropped window {r['orig_index']}\n"
                f"center ≈ {r['time_center_min']:.2f} min after state start"
            )
            ax.legend(frameon=False, fontsize=7, loc="best")
            sns.despine(ax=ax, top=True, right=True)
            summary_rows.append({
                "state": state,
                "orig_index": int(r["orig_index"]),
                "time_start_min": float(r["time_start_min"]),
                "time_center_min": float(r["time_center_min"]),
                "time_stop_min": float(r["time_stop_min"]),
                "seed": int(r["seed"]),
            })

    fig.suptitle("Auxiliary check: decompositions for Figure 6 dropped windows", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(csv, index=False)
    print(f"[INFO] Saved aux dropped-window decomposition figure -> {png}")
    print(f"[INFO] Saved aux dropped-window decomposition figure -> {svg}")
    print(f"[INFO] Saved aux dropped-window summary CSV        -> {csv}")
    return png, svg, csv


# -----------------------------------------------------------------------------
# Payload/meta export
# -----------------------------------------------------------------------------
def save_meta(payloads: Dict[str, StatePayload], save_dir: Path, stem: str, args: argparse.Namespace) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_path = save_dir / f"{stem}.plotmeta.json"
    meta = {
        "created_local": _dt.datetime.now().isoformat(timespec="seconds"),
        "script": "07_supp_figure_whole_session_slope_trajectory.py",
        "root": str(ROOT),
        "fig4_dir": str(Path(args.fig4_dir).expanduser().resolve()),
        "save_dir": str(save_dir),
        "analysis_frange_hz": list(ANALYSIS_FRANGE),
        "slope_band_hz": list(SLOPE_BAND),
        "MT_PARAMS": dict(MT_PARAMS),
        "notes": [
            "specparam and SL_specdecomp slope trajectories are loaded from Figure 6 saved payloads.",
            "naive OLS slope is recomputed from Figure 6 saved P_fit_tf over 40-60 Hz.",
            "dropped windows are shaded by original segment window index rather than plotted as X markers.",
            "Figure 6 target windows are shown as outlined boxes and square markers.",
        ],
        "states": {},
        "args": vars(args),
    }
    for state, p in payloads.items():
        meta["states"][state] = {
            "fig4_npz_path": str(p.fig4_npz_path),
            "fig4_meta_path": str(p.meta_path),
            "idx_target_postdrop": int(p.idx_target),
            "target_center_min": float(p.T_rel_min[p.idx_target]),
            "target_orig_index": int(p.keep_orig[p.idx_target]),
            "keep_orig": [int(x) for x in p.keep_orig],
            "drop_orig": [int(x) for x in p.drop_orig],
            "center0_min_inferred": float(p.center0_min),
            "step_min": float(p.step_min),
            "duration_min": float(p.duration_min),
        }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    print(f"[INFO] Saved meta -> {meta_path}")
    return meta_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplemental whole-epoch slope trajectory figure using Figure 6 saved payloads."
    )
    parser.add_argument("--fig4-dir", default=str(FIG4_DIR_DEFAULT), help="Directory containing Figure_4_CV_awake/anesthesia plotdata + plotmeta files.")
    parser.add_argument("--save-dir", default=str(SAVE_DIR_DEFAULT), help="Output directory for this supplemental figure.")
    parser.add_argument("--stem", default="supp_whole_session_slope_trajectory_from_fig4_payloads", help="Base output filename stem.")
    parser.add_argument("--drop-awake", default=None, help="Comma-separated original awake window indices to shade as dropped.")
    parser.add_argument("--drop-anes", default=None, help="Comma-separated original anesthesia window indices to shade as dropped.")
    parser.add_argument("--drop-from-outlier-csv", default=None, help="Optional Figure 6 outlier_inspector CSV; unioned with payload/manual drops.")
    parser.add_argument("--skip-refit-dropped", action="store_true", help="Skip aux refitting of dropped windows and save only a placeholder aux PNG.")
    parser.add_argument("--dropped-sl-draws", type=int, default=1000, help="SL_specdecomp draws for dropped-window aux refits.")
    parser.add_argument("--dropped-sl-tune", type=int, default=1000, help="SL_specdecomp tuning samples for dropped-window aux refits.")
    parser.add_argument("--dropped-sl-chains", type=int, default=2, help="SL_specdecomp chains for dropped-window aux refits.")
    parser.add_argument("--dropped-refit-seed", type=int, default=42, help="Run-level seed for dropped-window aux refits.")
    args = parser.parse_args()

    fig4_dir = Path(args.fig4_dir).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_awake: List[int] = []
    csv_anes: List[int] = []
    if args.drop_from_outlier_csv:
        csv_awake = _read_drop_from_outlier_csv(args.drop_from_outlier_csv, "awake")
        csv_anes = _read_drop_from_outlier_csv(args.drop_from_outlier_csv, "anesthesia")

    payloads = {
        "awake": _load_state_payload(
            "awake",
            fig4_dir,
            manual_drop=_parse_int_list(args.drop_awake),
            csv_drop=csv_awake,
        ),
        "anesthesia": _load_state_payload(
            "anesthesia",
            fig4_dir,
            manual_drop=_parse_int_list(args.drop_anes),
            csv_drop=csv_anes,
        ),
    }

    print("[SUMMARY] Loaded Figure 6 payloads")
    for state in ["awake", "anesthesia"]:
        p = payloads[state]
        print(
            f"  - {state}: n_retained={len(p.T_rel_min)}, "
            f"drop_orig={p.drop_orig}, "
            f"idx_target={p.idx_target}, "
            f"target_center={float(p.T_rel_min[p.idx_target]):.3f} min, "
            f"target_orig={p.keep_orig[p.idx_target]}"
        )

    plot_main_figure(payloads, save_dir=save_dir, stem=args.stem)
    save_meta(payloads, save_dir=save_dir, stem=args.stem, args=args)

    if args.skip_refit_dropped:
        aux_results: List[Dict[str, Any]] = []
        print("[INFO] Skipping dropped-window refits; aux PNG will be a placeholder.")
    else:
        aux_results = refit_dropped_windows_for_aux(
            payloads,
            draws=int(args.dropped_sl_draws),
            tune=int(args.dropped_sl_tune),
            chains=int(args.dropped_sl_chains),
            seed=int(args.dropped_refit_seed),
        )
    plot_aux_dropped_decompositions(aux_results, save_dir=save_dir, stem=args.stem)


if __name__ == "__main__":
    main()
