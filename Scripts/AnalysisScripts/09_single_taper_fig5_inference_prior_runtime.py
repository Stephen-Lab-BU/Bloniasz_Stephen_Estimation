#!/usr/bin/env python3
"""Single-taper inference and prior-sensitivity extension of manuscript Figure 5.

Place this file in::

    Bloniasz_Stephen_Estimation/Scripts/AnalysisScripts/

This is the rhythm-height benchmark shown in manuscript Figure 5. It keeps the
live Figure 5 generating model and amplitude grid:

* additive SL_GPsim ground truth;
* exponent=2, offset=0.5, raw knee=60;
* one 8 Hz rhythm with sigma=3 Hz;
* peak amplitudes 0.5, 1, 2, and 4;
* 30-second, 1000-Hz realizations;
* broadband slope evaluated from 40--60 Hz.

The requested extension changes the spectral estimate to a single taper
(NW=1, K=1) and fits every simulated PSD with the same four estimators:

1. specparam;
2. additive SL_specdecomp MAP with the paper priors;
3. additive SL_specdecomp posterior sampling with the same priors;
4. bounded additive Gamma maximum likelihood without probability priors.

The fourth estimator uses the same additive Gamma mean family as SL_specdecomp.
Its positivity, frequency, slope, knee, and width bounds are identifiability
constraints rather than probability priors. It therefore isolates the effect
of the MAP/sampling priors without changing the likelihood family.

Short installation/smoke run (one replicate per amplitude, tiny MCMC)::

    python 09_single_taper_fig5_inference_prior_runtime.py --run-mode short

Full benchmark (30 replicates per amplitude, 800/800/2 sampling)::

    python 09_single_taper_fig5_inference_prior_runtime.py --run-mode full

Short-mode posterior draws are intentionally inadequate for inference. The
script labels them as a smoke test in the manifest and figure. Fits are
checkpointed by amplitude and replicate, so interrupted full runs resume.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_NAME = "09_single_taper_fig5_inference_prior_runtime"
SCRIPT_VERSION = "2026-08-28.2"

FS = 1000.0
WINDOW_SECONDS = 30.0
NW = 1.0
K_TAPERS = 1.0
ANALYSIS_FRANGE = (1.0, 200.0)
SLOPE_BAND = (40.0, 60.0)
RHYTHM_BAND = (1.0, 20.0)
AMP_GRID = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=float)

DEFAULT_EXPONENT = 2.0
DEFAULT_OFFSET = 0.5
DEFAULT_KNEE_RAW = 60.0
DEFAULT_PEAK_HZ = 8.0
DEFAULT_PEAK_SIGMA = 3.0

FIGURE5_PRIORS = {
    "knee_hz_bounds": ANALYSIS_FRANGE,
    "slope_mu": -2.0,
    "slope_sigma": 1.0,
    "slope_bounds": (-5.0, -0.5),
    "sigma_mu": 3.0,
    "sigma_sigma": 2.0,
    "sigma_bounds": (0.5, 12.0),
    "a_log_sigma": 1.0,
}
B_PRIOR_SIGMA = 5.0
AMP_PRIOR_ANCHOR_Q = 50.0
AMP_PRIOR_SIGMA = 1.25

METHOD_ORDER = ["specparam", "SL_MAP", "SL_sampling", "Gamma_ML_no_priors"]
METHOD_LABELS = {
    "specparam": "specparam",
    "SL_MAP": "SL_specdecomp MAP",
    "SL_sampling": "SL_specdecomp sampling",
    "Gamma_ML_no_priors": "Gamma ML (no priors)",
}
METHOD_COLORS = {
    "specparam": "#4C78A8",
    "SL_MAP": "#59A14F",
    "SL_sampling": "#9467BD",
    "Gamma_ML_no_priors": "#E15759",
}


pd = None
plt = None
sns = None
optimize = None
SpectralModel = None
Multitaper = None
Connectivity = None
Decompose = None
spectrum = None
pm = None
az = None
xr = None


def _project_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("SPECTRAL_DECOMP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    candidate = here.parents[2] if len(here.parents) >= 3 else Path.cwd()
    if (candidate / "Scripts" / "AnalysisScripts").exists():
        return candidate
    return Path.cwd().resolve()


def _load_dependencies() -> None:
    global pd, plt, sns, optimize, SpectralModel, Multitaper, Connectivity
    global Decompose, spectrum, pm, az, xr
    import pandas as _pd
    import matplotlib.pyplot as _plt
    import seaborn as _sns
    import pymc as _pm
    import arviz as _az
    import xarray as _xr
    from scipy import optimize as _optimize
    from specparam import SpectralModel as _SpectralModel
    from spectral_connectivity import Connectivity as _Connectivity
    from spectral_connectivity import Multitaper as _Multitaper
    from SL_GPsim import spectrum as _spectrum
    from SL_specdecomp import Decompose as _Decompose

    pd, plt, sns, optimize = _pd, _plt, _sns, _optimize
    SpectralModel, Multitaper, Connectivity = _SpectralModel, _Multitaper, _Connectivity
    Decompose, spectrum, pm, az, xr = _Decompose, _spectrum, _pm, _az, _xr
    _install_arviz_compatibility()


def _install_arviz_compatibility() -> None:
    """Adapt current SL_specdecomp to ArviZ >=1.3's removed posterior keyword."""
    if getattr(az.from_dict, "_slsd_compat", False):
        return
    try:
        az.from_dict(posterior={"_probe": np.zeros((1, 1), dtype=float)})
        return
    except TypeError as exc:
        if "posterior" not in str(exc):
            raise
    original = az.from_dict

    def compat_from_dict(*args: Any, posterior: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        if posterior is None:
            return original(*args, **kwargs)
        data_vars: Dict[str, Any] = {}
        for name, value in posterior.items():
            arr = np.asarray(value)
            if arr.ndim < 2:
                arr = arr.reshape((1, 1) + arr.shape)
            dims = ("chain", "draw") + tuple(
                f"{name}_dim_{index}" for index in range(arr.ndim - 2)
            )
            data_vars[name] = (dims, arr)
        return xr.DataTree.from_dict({"posterior": xr.Dataset(data_vars)})

    compat_from_dict._slsd_compat = True  # type: ignore[attr-defined]
    az.from_dict = compat_from_dict
    warnings.warn(
        "Installed an in-process ArviZ compatibility shim because the current "
        "SL_specdecomp release still calls az.from_dict(posterior=...).",
        RuntimeWarning,
    )


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _json_dump(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default)
    temporary.replace(path)


def _config_signature(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _slope(freqs: np.ndarray, power: np.ndarray) -> float:
    mask = (
        (freqs >= SLOPE_BAND[0])
        & (freqs <= SLOPE_BAND[1])
        & np.isfinite(power)
        & (power > 0)
    )
    if int(mask.sum()) < 2:
        return float("nan")
    value, _ = np.polyfit(np.log10(freqs[mask]), np.log10(power[mask]), 1)
    return float(value)


def _single_taper_psd(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    arr = np.asarray(x, dtype=float).ravel()[:, None, None]
    multitaper = Multitaper(
        arr,
        sampling_frequency=FS,
        n_tapers=1,
        time_halfbandwidth_product=NW,
        start_time=0.0,
        time_window_duration=WINDOW_SECONDS,
        time_window_step=WINDOW_SECONDS,
    )
    connectivity = Connectivity.from_multitaper(multitaper)
    freqs = np.asarray(connectivity.frequencies, dtype=float).ravel()
    power = np.asarray(connectivity.power(), dtype=float).squeeze().ravel()
    keep = (
        np.isfinite(freqs)
        & np.isfinite(power)
        & (power > 0)
        & (freqs >= ANALYSIS_FRANGE[0])
        & (freqs <= ANALYSIS_FRANGE[1])
    )
    freqs, power = freqs[keep], power[keep]
    if freqs.size < 20:
        raise RuntimeError("Single-taper PSD produced too few usable frequencies.")
    df = float(np.median(np.diff(freqs)))
    independent_spacing = 2.0 * NW / WINDOW_SECONDS
    stride = max(1, int(round(independent_spacing / max(df, 1e-12))))
    return freqs[::stride], np.clip(power[::stride], 1e-30, np.inf), stride


def _simulate(
    amplitude: float,
    seed: int,
    config: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    result = spectrum(
        sampling_rate=FS,
        duration=WINDOW_SECONDS,
        aperiodic_exponent=float(config["true_exponent"]),
        aperiodic_offset=float(config["true_offset"]),
        knee=float(config["true_knee_raw"]),
        peaks=[{
            "freq": float(config["true_peak_hz"]),
            "amplitude": float(amplitude),
            "sigma": float(config["true_peak_sigma"]),
        }],
        average_firing_rate=0.0,
        random_state=int(seed),
        direct_estimate=False,
        plot=False,
        mode="additive",
    )
    freqs, observed, stride = _single_taper_psd(result.time_domain.combined_signal)
    frequency_domain = result.frequency_domain
    truth_freqs = np.asarray(frequency_domain.frequencies, dtype=float).ravel()
    truth_total_native = np.asarray(frequency_domain.combined_spectrum, dtype=float).ravel()
    truth_bb_native = np.asarray(frequency_domain.broadband_spectrum, dtype=float).ravel()
    truth_rhythm_native = np.asarray(frequency_domain.rhythmic_spectrum, dtype=float).ravel()
    valid = np.isfinite(truth_freqs) & (truth_freqs >= 0)
    order = np.argsort(truth_freqs[valid])
    truth_freqs = truth_freqs[valid][order]
    truth_total_native = truth_total_native[valid][order]
    truth_bb_native = truth_bb_native[valid][order]
    truth_rhythm_native = truth_rhythm_native[valid][order]
    return {
        "freqs": freqs,
        "observed": observed,
        "truth_total": np.clip(np.interp(freqs, truth_freqs, truth_total_native), 1e-30, np.inf),
        "truth_broadband": np.clip(np.interp(freqs, truth_freqs, truth_bb_native), 1e-30, np.inf),
        "truth_rhythm": np.clip(np.interp(freqs, truth_freqs, truth_rhythm_native), 0.0, np.inf),
        "frequency_stride": np.asarray([stride], dtype=int),
    }


def _specparam_fit(freqs: np.ndarray, power: np.ndarray) -> Dict[str, Any]:
    model = SpectralModel(
        peak_width_limits=[1.0, 30.0],
        max_n_peaks=1,
        min_peak_height=0.0,
        peak_threshold=2.0,
        aperiodic_mode="knee",
        verbose=False,
    )
    model.fit(freqs, np.clip(power, 1e-30, np.inf), freq_range=ANALYSIS_FRANGE)
    components = getattr(getattr(model, "results", None), "model", None)
    if components is None:
        raise RuntimeError("specparam did not expose fitted model components.")
    full_attr = getattr(components, "modeled_spectrum")
    if callable(full_attr):
        try:
            full = np.asarray(full_attr(space="linear"), dtype=float).ravel()
        except TypeError:
            full = 10.0 ** np.asarray(full_attr(), dtype=float).ravel()
    else:
        full = 10.0 ** np.asarray(full_attr, dtype=float).ravel()
    getter = getattr(components, "get_component")
    try:
        broadband = np.asarray(getter("aperiodic", space="linear"), dtype=float).ravel()
    except TypeError:
        broadband = 10.0 ** np.asarray(getter("aperiodic"), dtype=float).ravel()
    model_freqs = freqs
    for candidate in (
        getattr(components, "freqs", None),
        getattr(getattr(model, "data", None), "freqs", None),
        getattr(model, "freqs", None),
    ):
        if candidate is not None and np.size(candidate) == full.size:
            model_freqs = np.asarray(candidate, dtype=float).ravel()
            break
    full = np.clip(np.interp(freqs, model_freqs, full), 1e-30, np.inf)
    broadband = np.clip(np.interp(freqs, model_freqs, broadband), 1e-30, np.inf)
    return {
        "total": full,
        "broadband": broadband,
        "rhythm": np.clip(full - broadband, 0.0, np.inf),
    }


def _prior_specs(freqs: np.ndarray, power: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    positive = power[np.isfinite(power) & (power > 0)]
    mu_b = float(np.log10(np.median(positive))) if positive.size else 0.0

    def b_factory(name: str, mu: float = mu_b):
        return pm.Normal(name, mu=mu, sigma=B_PRIOR_SIGMA)

    rhythm_mask = (
        (freqs >= RHYTHM_BAND[0])
        & (freqs <= RHYTHM_BAND[1])
        & np.isfinite(power)
        & (power > 0)
    )
    values = power[rhythm_mask] if int(rhythm_mask.sum()) >= 5 else positive
    amplitude_scale = float(np.percentile(values, AMP_PRIOR_ANCHOR_Q)) if values.size else 1e-12

    def amplitude_factory(name: str, scale: float = amplitude_scale):
        return pm.LogNormal(name, mu=np.log(max(scale, 1e-12)), sigma=AMP_PRIOR_SIGMA)

    return (
        {"b_0": {"factory": b_factory}, "b": {"factory": b_factory}},
        {"A_lin_0": {"factory": amplitude_factory}},
    )


def _slsd_fit(
    freqs: np.ndarray,
    power: np.ndarray,
    inference_method: str,
    seed: int,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    aperiodic_specs, rhythm_specs = _prior_specs(freqs, power)
    if inference_method == "map":
        sample_kwargs: Dict[str, Any] = {
            "inference_method": "map",
            "progressbar": False,
            "map_kwargs": {"method": "L-BFGS-B"},
        }
    elif inference_method == "sample":
        sample_kwargs = {
            "inference_method": "sample",
            "draws": int(config["draws"]),
            "tune": int(config["tune"]),
            "chains": int(config["chains"]),
            "cores": 1,
            "target_accept": 0.90,
            "random_seed": int(seed),
            "progressbar": False,
        }
        if config["nuts_sampler"] != "auto":
            sample_kwargs["nuts_sampler"] = config["nuts_sampler"]
            if config["nuts_sampler"] == "blackjax":
                sample_kwargs["nuts_sampler_kwargs"] = {"chain_method": "vectorized"}
    else:
        raise ValueError(inference_method)
    result = Decompose(
        freqs,
        np.clip(power, 1e-30, np.inf),
        fs=FS,
        mode="additive",
        n_aperiodics=1,
        n_rhythms=1,
        rhythm_bands=[RHYTHM_BAND],
        k_tapers=K_TAPERS,
        priors=dict(FIGURE5_PRIORS),
        aperiodic_param_specs=aperiodic_specs,
        rhythm_param_specs=rhythm_specs,
        sample_kwargs=sample_kwargs,
        plot=False,
    )
    total = np.clip(np.asarray(result.estimated_spectrum, dtype=float).ravel(), 1e-30, np.inf)
    broadband = np.clip(np.asarray(result.broadband, dtype=float).ravel(), 1e-30, np.inf)
    rhythm = getattr(result, "rhythms", None)
    rhythm = np.asarray(rhythm, dtype=float).ravel() if rhythm is not None else np.clip(total - broadband, 0.0, np.inf)
    return {"total": total, "broadband": broadband, "rhythm": np.clip(rhythm, 0.0, np.inf)}


def _mirrored_gaussian(freqs: np.ndarray, center: float, sigma_hz: float) -> np.ndarray:
    sigma_hz = max(float(sigma_hz), 1e-9)
    return np.exp(-0.5 * ((freqs - center) / sigma_hz) ** 2) + np.exp(
        -0.5 * ((freqs + center) / sigma_hz) ** 2
    )


def _gamma_ml_fit(
    freqs: np.ndarray,
    power: np.ndarray,
    seed: int,
    n_starts: int,
) -> Dict[str, Any]:
    """Bounded likelihood-only fit of the additive SL_specdecomp mean family."""
    f = np.asarray(freqs, dtype=float)
    y = np.clip(np.asarray(power, dtype=float), 1e-30, np.inf)
    slope_bounds = tuple(FIGURE5_PRIORS["slope_bounds"])
    sigma_bounds = tuple(FIGURE5_PRIORS["sigma_bounds"])
    knee_bounds = tuple(FIGURE5_PRIORS["knee_hz_bounds"])
    slope0, _ = np.polyfit(np.log10(f), np.log10(y), 1)
    slope0 = float(np.clip(slope0, *slope_bounds))
    knee0 = float(np.clip(np.median(f), *knee_bounds))
    chi0 = -slope0
    b0 = float(np.median(np.log10(y) + np.log10(knee0**chi0 + f**chi0)))
    bb0 = 10.0**b0 / (knee0**chi0 + f**chi0)
    rhythm_mask = (f >= RHYTHM_BAND[0]) & (f <= RHYTHM_BAND[1])
    residual = np.maximum(y - bb0, 0.0)
    center0 = float(f[np.where(rhythm_mask)[0][np.argmax(residual[rhythm_mask])]])
    sigma0 = 3.0
    scale = float(np.percentile(y[rhythm_mask], 99.0))
    amplitude0 = max(float(np.max(residual[rhythm_mask])), 0.05 * scale, 1e-30)
    theta0 = np.asarray([b0, slope0, math.log(knee0), center0, sigma0, math.log(amplitude0)])
    bounds = [
        (float(np.log10(np.min(y))) - 3.0, float(np.log10(np.max(y))) + 6.0),
        slope_bounds,
        (math.log(knee_bounds[0]), math.log(knee_bounds[1])),
        RHYTHM_BAND,
        sigma_bounds,
        (math.log(max(scale * 1e-8, 1e-30)), math.log(max(scale * 1e4, 1e-29))),
    ]

    def evaluate(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        b, slope_value, log_knee, center, sigma_hz, log_amplitude = map(float, theta)
        chi = -slope_value
        knee_hz = math.exp(log_knee)
        broadband = 10.0**b / (knee_hz**chi + f**chi)
        rhythm = math.exp(log_amplitude) * _mirrored_gaussian(f, center, sigma_hz)
        return np.clip(broadband + rhythm, 1e-30, np.inf), np.clip(broadband, 1e-30, np.inf), rhythm

    def objective(theta: np.ndarray) -> float:
        total, _, _ = evaluate(theta)
        value = float(np.sum(K_TAPERS * (np.log(total) + y / total)))
        return value if np.isfinite(value) else 1e300

    rng = np.random.default_rng(seed)
    starts = [theta0]
    for _ in range(max(0, int(n_starts) - 1)):
        candidate = theta0.copy()
        candidate[0] += rng.normal(0.0, 0.35)
        candidate[1] = np.clip(candidate[1] + rng.normal(0.0, 0.30), *slope_bounds)
        candidate[3] = rng.uniform(*RHYTHM_BAND)
        candidate[4] = rng.uniform(*sigma_bounds)
        candidate[5] += rng.normal(0.0, 0.75)
        for index, (lower, upper) in enumerate(bounds):
            candidate[index] = np.clip(candidate[index], lower + 1e-9, upper - 1e-9)
        starts.append(candidate)
    best = None
    for start in starts:
        fitted = optimize.minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-6},
        )
        if best is None or float(fitted.fun) < float(best.fun):
            best = fitted
    if best is None:
        raise RuntimeError("Gamma ML optimizer never ran.")
    total, broadband, rhythm = evaluate(best.x)
    return {
        "total": total,
        "broadband": broadband,
        "rhythm": rhythm,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "objective": float(best.fun),
    }


def _method_metrics(
    freqs: np.ndarray,
    result: Mapping[str, Any],
    truth_total: np.ndarray,
    truth_slope: float,
    true_peak_hz: float,
) -> Dict[str, Any]:
    broadband = np.asarray(result["broadband"], dtype=float)
    total = np.asarray(result["total"], dtype=float)
    rhythm = np.asarray(result["rhythm"], dtype=float)
    slope_est = _slope(freqs, broadband)
    return {
        "slope_est": slope_est,
        "slope_error": float(slope_est - truth_slope),
        "total_at_true_peak": float(np.interp(true_peak_hz, freqs, total)),
        "broadband_at_true_peak": float(np.interp(true_peak_hz, freqs, broadband)),
        "rhythm_at_true_peak": float(np.interp(true_peak_hz, freqs, rhythm)),
        "log10_spectrum_rmse": float(
            np.sqrt(np.mean((np.log10(total) - np.log10(np.clip(truth_total, 1e-30, np.inf))) ** 2))
        ),
    }


def _checkpoint_paths(directory: Path, amplitude: float, replicate: int) -> Tuple[Path, Path]:
    amp_token = str(amplitude).replace(".", "p")
    stem = directory / f"amp_{amp_token}_rep_{replicate:03d}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _run_one(
    amplitude: float,
    replicate: int,
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    resume: bool,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    json_path, npz_path = _checkpoint_paths(checkpoint_dir, amplitude, replicate)
    if resume and json_path.exists() and npz_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("config_signature") == config["config_signature"]:
            with np.load(npz_path, allow_pickle=False) as payload:
                arrays = {key: np.asarray(payload[key]) for key in payload.files}
            print(f"[resume] amplitude={amplitude:g}, replicate={replicate}", flush=True)
            return meta, arrays

    amp_index = int(np.where(np.isclose(AMP_GRID, amplitude))[0][0])
    sim_seed = int(config["seed"]) + amp_index * 10000 + replicate
    arrays = _simulate(amplitude, sim_seed, config)
    freqs = arrays["freqs"]
    observed = arrays["observed"]
    truth_slope = _slope(freqs, arrays["truth_broadband"])
    meta: Dict[str, Any] = {
        "amplitude": float(amplitude),
        "replicate": int(replicate),
        "simulation_seed": int(sim_seed),
        "config_signature": config["config_signature"],
        "truth_slope": float(truth_slope),
        "frequency_stride": int(arrays["frequency_stride"][0]),
        "methods": {},
    }
    fitters = {
        "specparam": lambda: _specparam_fit(freqs, observed),
        "SL_MAP": lambda: _slsd_fit(freqs, observed, "map", sim_seed + 100000, config),
        "SL_sampling": lambda: _slsd_fit(freqs, observed, "sample", sim_seed + 200000, config),
        "Gamma_ML_no_priors": lambda: _gamma_ml_fit(
            freqs, observed, sim_seed + 300000, int(config["ml_starts"])
        ),
    }
    for method in METHOD_ORDER:
        if method == "SL_sampling" and config["skip_sampling"]:
            continue
        print(f"[fit] amp={amplitude:g}, rep={replicate}: {METHOD_LABELS[method]}", flush=True)
        start = time.perf_counter()
        try:
            result = fitters[method]()
            runtime = time.perf_counter() - start
            arrays[f"{method}_total"] = np.asarray(result["total"], dtype=float)
            arrays[f"{method}_broadband"] = np.asarray(result["broadband"], dtype=float)
            arrays[f"{method}_rhythm"] = np.asarray(result["rhythm"], dtype=float)
            meta["methods"][method] = {
                "status": "ok",
                "runtime_seconds": float(runtime),
                **_method_metrics(
                    freqs,
                    result,
                    arrays["truth_total"],
                    truth_slope,
                    float(config["true_peak_hz"]),
                ),
            }
            for key in ("optimizer_success", "optimizer_message", "objective"):
                if key in result:
                    meta["methods"][method][key] = result[key]
        except Exception as exc:
            meta["methods"][method] = {
                "status": "error",
                "runtime_seconds": float(time.perf_counter() - start),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _json_dump(json_path, meta)
            raise RuntimeError(
                f"Amplitude {amplitude:g}, replicate {replicate}, {METHOD_LABELS[method]} failed. "
                f"Checkpoint: {json_path}"
            ) from exc
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    _json_dump(json_path, meta)
    return meta, arrays


def _tables(
    results: Sequence[Tuple[Dict[str, Any], Dict[str, np.ndarray]]],
) -> Tuple[Any, Any]:
    rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    for meta, _ in results:
        for method, values in meta["methods"].items():
            row = {
                "amplitude": meta["amplitude"],
                "replicate": meta["replicate"],
                "simulation_seed": meta["simulation_seed"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "truth_slope": meta["truth_slope"],
                **values,
            }
            row.pop("traceback", None)
            rows.append(row)
        methods = meta["methods"]
        comparisons = [
            ("SL_MAP", "Gamma_ML_no_priors", "MAP_minus_ML"),
            ("SL_sampling", "Gamma_ML_no_priors", "sampling_minus_ML"),
            ("SL_MAP", "specparam", "MAP_minus_specparam"),
            ("SL_sampling", "specparam", "sampling_minus_specparam"),
        ]
        for left, right, label in comparisons:
            if left not in methods or right not in methods:
                continue
            paired_rows.append({
                "amplitude": meta["amplitude"],
                "replicate": meta["replicate"],
                "comparison": label,
                "left_method": left,
                "right_method": right,
                "left_slope": methods[left]["slope_est"],
                "right_slope": methods[right]["slope_est"],
                "delta_slope": methods[left]["slope_est"] - methods[right]["slope_est"],
                "left_slope_error": methods[left]["slope_error"],
                "right_slope_error": methods[right]["slope_error"],
                "left_rhythm_at_true_peak": methods[left]["rhythm_at_true_peak"],
                "right_rhythm_at_true_peak": methods[right]["rhythm_at_true_peak"],
                "delta_rhythm_at_true_peak": methods[left]["rhythm_at_true_peak"] - methods[right]["rhythm_at_true_peak"],
            })
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(paired_rows)


def _method_component_panel(
    ax: Any,
    method: str,
    representative: Mapping[float, Tuple[Dict[str, Any], Dict[str, np.ndarray]]],
) -> None:
    color = METHOD_COLORS[method]
    for index, amplitude in enumerate(AMP_GRID):
        _, arrays = representative[float(amplitude)]
        alpha = 0.35 + 0.60 * index / max(len(AMP_GRID) - 1, 1)
        freqs = arrays["freqs"]
        if f"{method}_total" not in arrays:
            continue
        broadband = arrays[f"{method}_broadband"]
        rhythm = arrays[f"{method}_rhythm"]
        ax.loglog(freqs, rhythm, color=color, lw=1.5, alpha=alpha, label=f"{amplitude:g}×")
        ax.loglog(freqs, broadband, color=color, ls="--", lw=1.1, alpha=alpha)
        slope_mask = (freqs >= SLOPE_BAND[0]) & (freqs <= SLOPE_BAND[1])
        ax.loglog(freqs[slope_mask], broadband[slope_mask], color="red", lw=1.0, alpha=0.9)
    ax.set_xlim(*ANALYSIS_FRANGE)
    ax.set_title(METHOD_LABELS[method])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")


def _slope_distribution_panel(ax: Any, estimates: Any, method: str, truth_slope: float) -> None:
    subset = estimates[estimates["method"] == method].copy()
    subset["amplitude_label"] = subset["amplitude"].map(lambda value: f"{value:g}")
    order = [f"{value:g}" for value in AMP_GRID]
    counts = subset.groupby("amplitude_label").size()
    if not subset.empty and counts.max() >= 2:
        sns.violinplot(
            data=subset,
            x="amplitude_label",
            y="slope_est",
            order=order,
            inner="quartile",
            cut=2,
            linewidth=0.9,
            color=METHOD_COLORS[method],
            saturation=0.78,
            ax=ax,
        )
    sns.stripplot(
        data=subset,
        x="amplitude_label",
        y="slope_est",
        order=order,
        color="k",
        alpha=0.40,
        size=2.8,
        jitter=0.14,
        ax=ax,
    )
    ax.axhline(truth_slope, color="k", lw=1.5, alpha=0.80)
    ax.set_xlabel("Peak amplitude")
    ax.set_ylabel("40–60 Hz BB slope")
    ax.set_title(METHOD_LABELS[method])


def _plot(
    estimates: Any,
    paired: Any,
    results: Sequence[Tuple[Dict[str, Any], Dict[str, np.ndarray]]],
    output_base: Path,
    run_mode: str,
) -> None:
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    sns.set_style("white")
    representative = {
        float(amplitude): next(
            item for item in results
            if np.isclose(item[0]["amplitude"], amplitude) and int(item[0]["replicate"]) == 0
        )
        for amplitude in AMP_GRID
    }
    truth_slopes = [item[0]["truth_slope"] for item in results]
    truth_slope = float(np.mean(truth_slopes))
    all_positive = []
    for _, arrays in results:
        for key, value in arrays.items():
            if key in {"frequency_stride", "freqs"}:
                continue
            arr = np.asarray(value, dtype=float).ravel()
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size:
                all_positive.append(arr)
    merged = np.concatenate(all_positive)
    y_limits = (
        max(1e-6, 10.0 ** math.floor(math.log10(np.percentile(merged, 0.5)))),
        10.0 ** math.ceil(math.log10(np.percentile(merged, 99.8))),
    )

    fig = plt.figure(figsize=(16.5, 13.0), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=[0.85, 1.0, 1.0, 0.9])
    ax_truth = fig.add_subplot(grid[0, :2])
    ax_runtime = fig.add_subplot(grid[0, 2:])

    for index, amplitude in enumerate(AMP_GRID):
        _, arrays = representative[float(amplitude)]
        freqs = arrays["freqs"]
        alpha = 0.35 + 0.60 * index / max(len(AMP_GRID) - 1, 1)
        ax_truth.loglog(freqs, arrays["truth_rhythm"], color="0.05", lw=1.6, alpha=alpha, label=f"{amplitude:g}×")
        ax_truth.loglog(freqs, arrays["truth_broadband"], color="0.25", ls="--", lw=1.1, alpha=alpha)
        mask = (freqs >= SLOPE_BAND[0]) & (freqs <= SLOPE_BAND[1])
        ax_truth.loglog(freqs[mask], arrays["truth_broadband"][mask], color="red", lw=1.0)
    ax_truth.set_xlim(*ANALYSIS_FRANGE)
    ax_truth.set_ylim(*y_limits)
    ax_truth.set_title("Ground truth")
    ax_truth.set_xlabel("Frequency (Hz)")
    ax_truth.set_ylabel("Power")
    ax_truth.legend(title="Peak amplitude", frameon=False, ncol=2, fontsize=7)

    runtime_data = [
        estimates.loc[(estimates["method"] == method) & (estimates["status"] == "ok"), "runtime_seconds"].to_numpy(float)
        for method in METHOD_ORDER if method in set(estimates["method"])
    ]
    runtime_methods = [method for method in METHOD_ORDER if method in set(estimates["method"])]
    ax_runtime.boxplot(runtime_data, tick_labels=[METHOD_LABELS[method] for method in runtime_methods])
    ax_runtime.set_yscale("log")
    ax_runtime.set_ylabel("Wall time per fit (seconds, log scale)")
    ax_runtime.set_title("Runtime benchmark")
    ax_runtime.tick_params(axis="x", rotation=20)

    component_axes = [fig.add_subplot(grid[1, column]) for column in range(4)]
    slope_axes = [fig.add_subplot(grid[2, column]) for column in range(4)]
    for ax, method in zip(component_axes, METHOD_ORDER):
        _method_component_panel(ax, method, representative)
        ax.set_ylim(*y_limits)
    component_axes[0].legend(title="Peak amplitude", frameon=False, fontsize=7)
    for ax, method in zip(slope_axes, METHOD_ORDER):
        _slope_distribution_panel(ax, estimates, method, truth_slope)

    ax_prior = fig.add_subplot(grid[3, :2])
    prior = paired[paired["comparison"] == "MAP_minus_ML"].copy()
    for amplitude, group in prior.groupby("amplitude"):
        x = float(amplitude)
        ax_prior.scatter(
            np.full(group.shape[0], x),
            group["delta_slope"],
            color="#7A5195",
            alpha=0.45,
            s=16,
        )
        ax_prior.scatter([x], [group["delta_slope"].mean()], color="k", marker="D", s=28)
    ax_prior.axhline(0.0, color="0.4", lw=0.9)
    ax_prior.set_xticks(AMP_GRID)
    ax_prior.set_xlabel("Peak amplitude")
    ax_prior.set_ylabel("MAP slope − no-prior ML slope")
    ax_prior.set_title("Paired prior sensitivity")

    ax_bias = fig.add_subplot(grid[3, 2:])
    for method in METHOD_ORDER:
        subset = estimates[estimates["method"] == method]
        summary = subset.groupby("amplitude")["slope_error"].agg(["mean", "sem"]).reset_index()
        ax_bias.errorbar(
            summary["amplitude"],
            summary["mean"],
            yerr=summary["sem"].fillna(0.0),
            marker="o",
            lw=1.4,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax_bias.axhline(0.0, color="0.4", lw=0.9)
    ax_bias.set_xticks(AMP_GRID)
    ax_bias.set_xlabel("Peak amplitude")
    ax_bias.set_ylabel("Broadband slope error")
    ax_bias.set_title("Rhythm-height-dependent slope bias")
    ax_bias.legend(frameon=False, fontsize=7)

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    all_axes = [ax_truth, ax_runtime, *component_axes, *slope_axes, ax_prior, ax_bias]
    for label, ax in zip(labels, all_axes):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    note = "SHORT SMOKE TEST — tiny MCMC, not inferential" if run_mode == "short" else "FULL BENCHMARK"
    fig.suptitle(
        "Single-taper Figure 5 extension: inference, priors, and runtime\n" + note,
        fontsize=14,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def _preflight() -> Dict[str, Any]:
    _load_dependencies()
    test = spectrum(
        sampling_rate=100.0,
        duration=0.2,
        aperiodic_exponent=2.0,
        aperiodic_offset=0.5,
        knee=60.0,
        peaks=[{"freq": 8.0, "amplitude": 1.0, "sigma": 3.0}],
        random_state=1,
        mode="additive",
    )
    if not hasattr(test, "time_domain") or not callable(Decompose):
        raise RuntimeError("Public SL_GPsim/SL_specdecomp APIs were not available.")
    return {
        "status": "ok",
        "SL_specdecomp_import": str(Path(sys.modules["SL_specdecomp"].__file__).resolve()),
        "SL_GPsim_import": str(Path(sys.modules["SL_GPsim"].__file__).resolve()),
        "figure5_design": {
            "amplitudes": AMP_GRID.tolist(),
            "single_taper": {"NW": NW, "K": K_TAPERS},
            "truth": {
                "exponent": DEFAULT_EXPONENT,
                "offset": DEFAULT_OFFSET,
                "knee_raw": DEFAULT_KNEE_RAW,
                "peak_hz": DEFAULT_PEAK_HZ,
                "peak_sigma": DEFAULT_PEAK_SIGMA,
            },
        },
        "versions": {
            name: _package_version(name)
            for name in ("SL-specdecomp", "SL-GPsim", "specparam", "spectral-connectivity", "pymc", "arviz")
        },
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-mode", choices=("short", "full"), default="short")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-replicates", type=int, default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--nuts-sampler", choices=("auto", "pymc", "blackjax", "numpyro"), default="auto")
    parser.add_argument("--ml-starts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--true-exponent", type=float, default=DEFAULT_EXPONENT)
    parser.add_argument("--true-offset", type=float, default=DEFAULT_OFFSET)
    parser.add_argument("--true-knee-raw", type=float, default=DEFAULT_KNEE_RAW)
    parser.add_argument("--true-peak-hz", type=float, default=DEFAULT_PEAK_HZ)
    parser.add_argument("--true-peak-sigma", type=float, default=DEFAULT_PEAK_SIGMA)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.check_only:
        print(json.dumps(_preflight(), indent=2))
        return 0
    _load_dependencies()
    root = _project_root(args.project_root)
    if args.run_mode == "short":
        preset = {"n_replicates": 1, "draws": 20, "tune": 20, "chains": 1, "ml_starts": 2}
    else:
        preset = {"n_replicates": 30, "draws": 800, "tune": 800, "chains": 2, "ml_starts": 6}
    config: Dict[str, Any] = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "run_mode": args.run_mode,
        "amplitudes": AMP_GRID.tolist(),
        "fs": FS,
        "window_seconds": WINDOW_SECONDS,
        "analysis_frange": ANALYSIS_FRANGE,
        "slope_band": SLOPE_BAND,
        "rhythm_band": RHYTHM_BAND,
        "single_taper": {"NW": NW, "K": K_TAPERS},
        "true_exponent": float(args.true_exponent),
        "true_offset": float(args.true_offset),
        "true_knee_raw": float(args.true_knee_raw),
        "true_peak_hz": float(args.true_peak_hz),
        "true_peak_sigma": float(args.true_peak_sigma),
        "n_replicates": int(args.n_replicates if args.n_replicates is not None else preset["n_replicates"]),
        "draws": int(args.draws if args.draws is not None else preset["draws"]),
        "tune": int(args.tune if args.tune is not None else preset["tune"]),
        "chains": int(args.chains if args.chains is not None else preset["chains"]),
        "nuts_sampler": args.nuts_sampler,
        "ml_starts": int(args.ml_starts if args.ml_starts is not None else preset["ml_starts"]),
        "seed": int(args.seed),
        "skip_sampling": bool(args.skip_sampling),
        "priors": {
            "package_priors": FIGURE5_PRIORS,
            "b": f"Normal(log10 median positive PSD, sigma={B_PRIOR_SIGMA})",
            "A_lin_0": f"LogNormal(log(q{AMP_PRIOR_ANCHOR_Q:g} rhythm-band power), sigma={AMP_PRIOR_SIGMA})",
        },
        "ml_definition": "same additive Gamma mean; likelihood only; bounded identifiable parameter space",
    }
    if config["n_replicates"] < 1:
        raise ValueError("--n-replicates must be at least 1.")
    config["config_signature"] = _config_signature(config)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        root / "Output" / "Results" / "FiguresIntermediate" / "Figure_5_single_taper_inference_prior_runtime" / args.run_mode
    )
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **config,
        "project_root": str(root),
        "output_dir": str(output_dir),
        "source_design": "06_supp_figure_height_grid_benchmark.py / main-text compact replot",
        "warning": (
            "Short mode uses one replicate per amplitude and 20/20/1 MCMC only to test execution."
            if args.run_mode == "short" else None
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "paper_repo_sha": _git_sha(root),
            "versions": {
                name: _package_version(name)
                for name in ("SL-specdecomp", "SL-GPsim", "specparam", "spectral-connectivity", "pymc", "arviz")
            },
        },
    }
    _json_dump(output_dir / "run_manifest.json", manifest)
    print(
        f"[mode] {args.run_mode}; amplitudes={AMP_GRID.tolist()}; "
        f"replicates={config['n_replicates']}; output={output_dir}",
        flush=True,
    )
    results: List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]] = []
    for amplitude in AMP_GRID:
        for replicate in range(int(config["n_replicates"])):
            results.append(
                _run_one(
                    float(amplitude),
                    replicate,
                    config,
                    checkpoint_dir,
                    args.resume,
                )
            )
    estimates, paired = _tables(results)
    estimates.to_csv(output_dir / "figure5_single_taper_estimates_and_runtime.csv", index=False)
    paired.to_csv(output_dir / "figure5_single_taper_paired_prior_differences.csv", index=False)
    _plot(
        estimates,
        paired,
        results,
        output_dir / "Figure_5_single_taper_inference_prior_runtime",
        args.run_mode,
    )
    print(f"[done] Wrote checkpoints, CSVs, manifest, and figure to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
