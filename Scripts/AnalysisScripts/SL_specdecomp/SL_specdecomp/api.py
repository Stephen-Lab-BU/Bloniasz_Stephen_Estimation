# api.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, Any
import numpy as np
import arviz as az
import pymc as pm
import warnings


from .types import Decomposition
from .pymc_models import build_additive_model, build_multiplicative_model

def _posterior_mean(idata: az.InferenceData, var: str) -> np.ndarray:
    arr = idata.posterior[var].values  
    c, d = arr.shape[0], arr.shape[1]
    return arr.reshape(c * d, *arr.shape[2:]).mean(axis=0)

def Decompose(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    fs: Optional[float] = None,
    mode: str = "additive",  # or "FOOOF_spectrum"
    n_aperiodics: int = 1,  
    n_rhythms: int = 0,
    rhythm_bands: Optional[Iterable[Tuple[float, float]]] = None,
    k_tapers: float = 1.0,
    priors: Optional[Dict[str, Any]] = None,
    sample_kwargs: Optional[Dict[str, Any]] = None,
    aperiodic_predictor=None,
    rhythm_predictor=None,
    aperiodic_param_specs: Optional[Dict[str, Any]] = None,
    rhythm_param_specs: Optional[Dict[str, Any]] = None,
    plot: bool = False,
):
    f = np.asarray(freqs, dtype=float)
    y = np.asarray(psd, dtype=float)
    if f.ndim != 1:
        raise ValueError("freqs must be a 1D array.")
    if y.shape != f.shape:
        raise ValueError("psd must have the same shape as freqs.")
    if np.any(f <= 0):
        if not (f.min() == 0.0 and np.all(f[1:] > 0)):
            raise ValueError("freqs must be strictly non-negative and typically > 0 except possibly DC.")

    mode = str(mode)
    if mode not in {"additive", "FOOOF_spectrum"}:
        raise ValueError("mode must be 'additive' or 'FOOOF_spectrum'.")
        
    if int(n_aperiodics) > 1:
        warnings.warn(
            f"Multiple aperiodic components (n_aperiodics={n_aperiodics}) in '{mode}' mode are experimental. "
            "Component identifiability and sampler convergence have not been fully vetted; "
            "inspect traces and diagnostics carefully.",
            RuntimeWarning,
        )

    # Build model
    if mode == "additive":
        model = build_additive_model(
            f, y,
            k_tapers=k_tapers,
            n_aperiodics=int(n_aperiodics),
            n_rhythms=n_rhythms,
            rhythm_bands=rhythm_bands,
            priors=priors,
            aperiodic_predictor=aperiodic_predictor,
            rhythm_predictor=rhythm_predictor,
            aperiodic_param_specs=aperiodic_param_specs,
            rhythm_param_specs=rhythm_param_specs,
        )
    else:
        model = build_multiplicative_model(
            f, y,
            k_tapers=k_tapers,
            n_aperiodics=int(n_aperiodics),
            n_rhythms=n_rhythms,
            rhythm_bands=rhythm_bands,
            priors=priors
        )


    # Sampling
    skw = dict(draws=1000, tune=1000, chains=2, target_accept=0.9, progressbar=False, random_seed=42)
    if sample_kwargs:
        skw.update(sample_kwargs)
    with model:
        idata = pm.sample(**skw)

    # Extract posterior means
    mu_mean = _posterior_mean(idata, "mu")
    if mode == "additive":
        P_ap_mean = _posterior_mean(idata, "P_ap")
        P_rh_mean = _posterior_mean(idata, "P_rh")
        # Components are optional if M==1; we still produce a (M,F) array when present
        P_ap_comps = idata.posterior.get("P_ap_components", None)
        if P_ap_comps is not None:
            P_ap_components_mean = _posterior_mean(idata, "P_ap_components")  # (M, F)
        else:
            P_ap_components_mean = None

        result = Decomposition(
            freqs=f, observed=y,
            estimated_spectrum=mu_mean,
            rhythms=P_rh_mean,
            broadband=P_ap_mean,
            broadband_components=P_ap_components_mean,   # <-- NEW
            idata=idata, mode=mode,
            r_factor=None
        )
    else:
        # Multiplicative: expose baseline, optional components, rhythmic additive, and multiplier
        P_ap_mean = _posterior_mean(idata, "P_ap")
        P_rh_add_mean = _posterior_mean(idata, "P_rh_add")
        # Optional components
        P_ap_comps = idata.posterior.get("P_ap_components", None)
        P_ap_components_mean = _posterior_mean(idata, "P_ap_components") if P_ap_comps is not None else None
        # Multiplier (unitless)
        try:
            R_factor_mean = _posterior_mean(idata, "R_factor")
        except Exception:
            R_factor_mean = None

        result = Decomposition(
            freqs=f, observed=y,
            estimated_spectrum=mu_mean,
            rhythms=P_rh_add_mean,                     # <-- additive contribution above baseline
            broadband=P_ap_mean,
            broadband_components=P_ap_components_mean,
            r_factor=R_factor_mean,
            idata=idata, mode=mode
        )


    if plot:
        result.plot(plot_components=(mode=="additive"))

    return result



def from_multitaper(
    signal: np.ndarray,
    fs: float,
    *,
    mode: str = "additive",
    n_aperiodics: int = 1,
    n_rhythms: int = 0,
    rhythm_bands: Optional[Iterable[Tuple[float, float]]] = None,
    time_halfbandwidth_product: float = 2.0,
    k_tapers: int = 3,
    n_tapers: Optional[int] = None,            
    time_window_duration: Optional[float] = None,
    time_window_step: Optional[float] = None,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    downsample_independent: bool = True,
    priors: Optional[Dict[str, Any]] = None,
    sample_kwargs: Optional[Dict[str, Any]] = None,
    plot: bool = False,
    **decompose_kwargs,
):
    """
    Compute a multitaper PSD with `spectral_connectivity` and fit the model.

    """
    import numpy as _np
    from spectral_connectivity import Multitaper, Connectivity

    x = _np.asarray(signal, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("`signal` must have length > 1.")

    duration = x.size / float(fs)
    if time_window_duration is None:
        time_window_duration = duration
    if time_window_step is None:
        time_window_step = time_window_duration

    # Allow caller to provide either n_tapers or k_tapers; prefer explicit n_tapers
    k_tapers_internal = int(n_tapers if n_tapers is not None else k_tapers)

    # ---- Multitaper & connectivity -----------------------------------------
    mt = Multitaper(
        x,
        sampling_frequency=float(fs),
        start_time=0.0,
        time_halfbandwidth_product=float(time_halfbandwidth_product),
        time_window_duration=float(time_window_duration),
        time_window_step=float(time_window_step),
        n_tapers=k_tapers_internal,
    )

    conn = Connectivity.from_multitaper(mt)
    freqs_emp = _np.asarray(conn.frequencies, float)
    psd_raw = _np.asarray(conn.power(), float)   # linear power

    F = freqs_emp.size
    if psd_raw.ndim == 1:
        psd_emp = psd_raw
    else:
        if F not in psd_raw.shape:
            raise ValueError(f"No axis of PSD matches len(freqs)={F}. PSD shape = {psd_raw.shape}")
        faxis = list(psd_raw.shape).index(F)
        if faxis != 0:
            psd_raw = _np.moveaxis(psd_raw, faxis, 0)
        psd_emp = psd_raw.reshape(F, -1).mean(axis=1)

    # ---- Optional downsampling to ~independent bins (stride ≈ 2*TW/df) ----
    if downsample_independent and freqs_emp.size > 1:
        df = float(freqs_emp[1] - freqs_emp[0])
        bandwidth_hz = 2.0 * float(time_halfbandwidth_product) / float(time_window_duration)
        stride = max(1, int(round(bandwidth_hz / df)))
        freqs_emp = freqs_emp[::stride]
        psd_emp   = psd_emp[::stride]

    # ---- Optional frequency clipping ---------------------------------------
    if freq_min is not None or freq_max is not None:
        lo = -float("inf") if freq_min is None else float(freq_min)
        hi =  float("inf") if freq_max is None else float(freq_max)
        m = (freqs_emp >= lo) & (freqs_emp <= hi)
        freqs_emp = freqs_emp[m]
        psd_emp   = psd_emp[m]

    # ---- Keep Decompose kwargs clean and consistent ------------------------
    decompose_kwargs = dict(decompose_kwargs)
    # Ensure there is no conflicting tapers kwarg passed through
    decompose_kwargs.pop("k_tapers", None)
    decompose_kwargs.pop("n_tapers", None)

    assert freqs_emp.shape == psd_emp.shape, f"freq/PSD shape mismatch: {freqs_emp.shape} vs {psd_emp.shape}"

    # ---- Fit the model ------------------------------------------------------
    return Decompose(
        freqs_emp, psd_emp,
        fs=fs,
        mode=mode,
        n_aperiodics=n_aperiodics,
        n_rhythms=n_rhythms,
        rhythm_bands=rhythm_bands,
        k_tapers=float(k_tapers_internal),
        priors=priors,
        sample_kwargs=sample_kwargs,
        plot=plot,
        **decompose_kwargs,
    )
