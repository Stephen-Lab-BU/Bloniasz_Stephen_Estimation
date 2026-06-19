# pymc_models.py
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from functools import partial

from .predictors import Predictor, REGISTRY, APERIODIC_1OVERF, MIRRORED_GAUSSIAN

ParamSpec = Union[float, int, Dict[str, Any], None]  # constant or {"dist": pm.Normal(...)} or None

def _ensure_bands(n_rhythms: int, rhythm_bands: Optional[Iterable[Tuple[float, float]]]) -> List[Tuple[float,float]]:
    if rhythm_bands is None:
        rhythm_bands = [(1.0, 20.0)] * int(n_rhythms)
    rb = list(rhythm_bands)
    if n_rhythms is not None and len(rb) != int(n_rhythms):
        raise ValueError(f"n_rhythms={n_rhythms} but {len(rb)} bands provided.")
    return rb

def _band_scale(freqs, y_lin, lo, hi, q=99):
    f = np.asarray(freqs, float); y = np.asarray(y_lin, float)
    m = np.isfinite(y) & (y > 0) & (f >= lo) & (f <= hi)
    if m.sum() < 5:
        m = np.isfinite(y) & (y > 0)
    return float(np.percentile(y[m], q))

def _get_predictor(spec: Union[str, Predictor, None], default: Predictor) -> Predictor:
    if spec is None:
        return default
    if isinstance(spec, Predictor):
        return spec
    if isinstance(spec, str):
        try:
            return REGISTRY[spec]
        except KeyError:
            raise ValueError(f"Unknown predictor '{spec}'. Known: {list(REGISTRY)}")
    raise TypeError("predictor must be None, a Predictor, or a registry key string.")

def _as_data(name: str, value):
    """Create a fixed constant inside the model across PyMC versions."""
    # 1) PyMC ≥5: ConstantData exists
    if hasattr(pm, "ConstantData"):
        return pm.ConstantData(name, float(value))
    # 2) PyMC 4: use pm.Data
    if hasattr(pm, "Data"):
        return pm.Data(name, float(value))
    # 3) Last resort: deterministic tensor
    return pm.Deterministic(name, pt.as_tensor_variable(float(value)))

def _make_param(name: str, spec, fallback_dist):
    """
    - number -> fixed constant (data) inside the model
    - {"factory": callable(name)->RV} -> build RV inside the model
    - {"dist": RV or scalar} -> use RV if already built; scalar -> data
    - None -> fallback_dist
    """
    # 1) Fixed numeric constants
    if isinstance(spec, (float, int, np.floating, np.integer)):
        return _as_data(name, spec)

    # 2) Factory pattern (safe custom prior created in-context)
    if isinstance(spec, dict) and "factory" in spec and callable(spec["factory"]):
        return spec["factory"](name)

    # 3) Direct distribution (or scalar) passed in dict
    if isinstance(spec, dict) and "dist" in spec:
        dist = spec["dist"]
        # Already a PyMC RV?
        if hasattr(dist, "owner") or getattr(dist, "name", None):
            return dist
        # Plain scalar slipped in
        if isinstance(dist, (float, int, np.floating, np.integer)):
            return _as_data(name, dist)
        # Anything else: wrap as tensor deterministic
        return pm.Deterministic(name, pt.as_tensor_variable(dist))

    # 4) Default prior inside model
    return fallback_dist()


def build_additive_model(
    freqs: np.ndarray,
    y_lin: np.ndarray,
    *,
    k_tapers: float = 1.0,
    n_aperiodics: int = 1,
    n_rhythms: int = 0,
    rhythm_bands: Optional[Iterable[Tuple[float, float]]] = None,
    priors: Optional[Dict[str, Any]] = None,
    aperiodic_predictor: Union[str, Predictor, None] = None,     # default = 1/f Lorentzian
    rhythm_predictor: Union[str, Predictor, None] = None,        # default = mirrored Gaussian
    aperiodic_param_specs: Optional[Dict[str, ParamSpec]] = None,  # e.g. {"b_0": 2.1, "knee_0": {"dist": pm.LogNormal(...)}}
    rhythm_param_specs: Optional[Dict[str, ParamSpec]] = None,     # e.g. {"center_0": 10.,"sigma_0": {"dist": pm.HalfNormal(...) }}
):
    f = np.asarray(freqs, dtype=float)
    nyq = float(np.max(f))
    rb = _ensure_bands(n_rhythms, rhythm_bands)

    pri = dict(
        knee_mu=100.0, knee_sigma=100.0, knee_bounds=(1.0, nyq),
        slope_mu=-2.0,  slope_sigma=1.0,  slope_bounds=(-5.0, -0.5),
        sigma_mu=3.0,   sigma_sigma=2.0,  sigma_bounds=(0.5, 12.0),
    )
    if priors: pri.update(priors)

    # Choose predictors (defaults preserve current behavior)
    ap_pred = _get_predictor(aperiodic_predictor, APERIODIC_1OVERF)
    rh_pred = _get_predictor(rhythm_predictor,    MIRRORED_GAUSSIAN)

    y = np.asarray(y_lin, float)
    y_pos = y[np.isfinite(y) & (y > 0)]
    mu_b = float(np.log10(np.median(y_pos))) if y_pos.size else 0.0

    aperiodic_param_specs = aperiodic_param_specs or {}
    rhythm_param_specs    = rhythm_param_specs or {}

    with pm.Model() as m:
        f_t = pt.as_tensor_variable(f)

        nyq_t = m.named_vars.get("nyquist")
        if nyq_t is None:
            nyq_t = _as_data("nyquist", float(nyq))

        # ---- Aperiodic: sum of components of ap_pred ----
        ap_components = []
        for i in range(int(n_aperiodics)):
            # Provide sensible defaults for b, knee, chi (if those names are used)
            def knee_fallback():
                return pm.TruncatedNormal(f"knee_{i}", mu=pri['knee_mu'], sigma=pri['knee_sigma'],
                                          lower=pri['knee_bounds'][0], upper=pri['knee_bounds'][1])
            def slope_fallback():
                return pm.TruncatedNormal(f"slope_{i}", mu=pri['slope_mu'], sigma=pri['slope_sigma'],
                                          lower=pri['slope_bounds'][0], upper=pri['slope_bounds'][1])
            def chi_from_slope(slope_var):
                return pm.Deterministic(f"chi_{i}", -slope_var)
            def b_fallback():
                return pm.Normal(f"b_{i}", mu=mu_b, sigma=2.0)

            # Create variables according to predictor param names
            param_values: Dict[str, Any] = {}
            for pname in ap_pred.params:
                full = f"{pname}_{i}"
                if pname == "knee":
                    var = _make_param(full, aperiodic_param_specs.get(full), knee_fallback)
                elif pname == "chi":
                    # chi may depend on slope if user didn't specify chi directly
                    if full in aperiodic_param_specs:
                        var = _make_param(full, aperiodic_param_specs[full], lambda: pm.Normal(full, 0.0, 5.0))
                    else:
                        slope_var = slope_fallback()
                        var = chi_from_slope(slope_var)
                elif pname == "b":
                    var = _make_param(full, aperiodic_param_specs.get(full), b_fallback)
                else:
                    # generic default: Normal(0, 10)
                    var = _make_param(full, aperiodic_param_specs.get(full), lambda: pm.Normal(full, 0.0, 10.0))
                param_values[pname] = var

            ap_i = ap_pred.pt_func(f_t, *[param_values[p] for p in ap_pred.params])
            ap_components.append(ap_i)

        if ap_components:
            P_ap_components = pt.stack(ap_components, axis=0)
            P_ap = pm.Deterministic("P_ap", pt.sum(P_ap_components, axis=0))
            pm.Deterministic("P_ap_components", P_ap_components)
        else:
            P_ap = pt.zeros_like(f_t)

        # ---- Rhythmic terms using rh_pred (default mirrored Gaussian) ----
        rh_terms = []
        for j, (lo, hi) in enumerate(rb):
            center = _make_param(f"center_{j}", rhythm_param_specs.get(f"center_{j}"),
                                lambda: pm.Uniform(f"center_{j}", lower=float(lo), upper=float(hi)))
            sigma  = _make_param(f"sigma_{j}",  rhythm_param_specs.get(f"sigma_{j}"),
                                lambda: pm.TruncatedNormal(f"sigma_{j}", mu=pri['sigma_mu'], sigma=pri['sigma_sigma'],
                                                            lower=pri['sigma_bounds'][0], upper=pri['sigma_bounds'][1]))

            # USE the pre-created nyq_t; do NOT recreate inside the loop
            G = rh_pred.pt_func(f_t, center, sigma, nyq_t)

            band_scale = _band_scale(freqs, y_lin, lo, hi, q=99)
            A_j = _make_param(f"A_lin_{j}", rhythm_param_specs.get(f"A_lin_{j}"),
                            lambda: pm.LogNormal(f"A_lin_{j}", mu=np.log(max(band_scale, 1e-12)), sigma=1.25))
            rh_terms.append(A_j * G)

        P_rh = pm.Deterministic("P_rh", pt.sum(pt.stack(rh_terms, axis=0), axis=0) if rh_terms else pt.zeros_like(f_t))


        # for j, (lo, hi) in enumerate(rb):
        #     # Defaults that match the current behavior:
        #     center = _make_param(f"center_{j}", rhythm_param_specs.get(f"center_{j}"),
        #                          lambda: pm.Uniform(f"center_{j}", lower=float(lo), upper=float(hi)))
        #     sigma = _make_param(f"sigma_{j}", rhythm_param_specs.get(f"sigma_{j}"),
        #                          lambda: pm.TruncatedNormal(f"sigma_{j}", mu=pri['sigma_mu'], sigma=pri['sigma_sigma'],
        #                                                     lower=pri['sigma_bounds'][0], upper=pri['sigma_bounds'][1]))
        #     nyq_t = _as_data("nyquist", float(nyq)) # pass as tensor; harmless to reuse name
        #     G = rh_pred.pt_func(f_t, center, sigma, nyq_t)

        #     # amplitude follows the linear-space LogNormal heuristic
        #     band_scale = _band_scale(freqs, y_lin, lo, hi, q=99)
        #     A_j = _make_param(f"A_lin_{j}", rhythm_param_specs.get(f"A_lin_{j}"),
        #                       lambda: pm.LogNormal(f"A_lin_{j}", mu=np.log(max(band_scale, 1e-12)), sigma=1.25))
        #     rh_terms.append(A_j * G)

        # P_rh = pm.Deterministic("P_rh", pt.sum(pt.stack(rh_terms, axis=0), axis=0) if rh_terms else pt.zeros_like(f_t))

        mu = pm.Deterministic("mu", P_ap + P_rh)
        alpha = pt.as_tensor_variable(float(k_tapers))
        pm.Gamma("y", alpha=alpha, beta=alpha / mu, observed=y.astype(float))

    return m
def build_multiplicative_model(
        freqs: np.ndarray,
        y_lin: np.ndarray,
        *,
        k_tapers: float = 1.0,
        n_aperiodics: int = 1,                     # <-- NEW
        n_rhythms: int = 0,
        rhythm_bands: Optional[Iterable[Tuple[float, float]]] = None,
        priors: Optional[Dict[str, Any]] = None,
        aperiodic_predictor: Union[str, Predictor, None] = None,
        rhythm_predictor: Union[str, Predictor, None] = None,
        aperiodic_param_specs: Optional[Dict[str, ParamSpec]] = None,  # <-- ensure present
        rhythm_param_specs: Optional[Dict[str, ParamSpec]] = None,
    ):
    
    f = np.asarray(freqs, dtype=float)
    nyq = float(np.max(f))
    rb = _ensure_bands(n_rhythms, rhythm_bands)

    pri = dict(
        sigma_mu=3.0, sigma_sigma=2.0, sigma_bounds=(0.5, 12.0),
        a_log_sigma=1.0,
    )
    if priors:
        pri.update(priors)

    ap_pred = _get_predictor(aperiodic_predictor, APERIODIC_1OVERF)
    rh_pred = _get_predictor(rhythm_predictor,    MIRRORED_GAUSSIAN)

    aperiodic_param_specs = aperiodic_param_specs or {}
    rhythm_param_specs    = rhythm_param_specs or {}

    y = np.asarray(y_lin, dtype=float)
    with pm.Model() as m:
        f_t = pt.as_tensor_variable(f)

        nyq_t = m.named_vars.get("nyquist")
        if nyq_t is None:
            nyq_t = _as_data("nyquist", float(nyq))

        # ---------- Aperiodic in linear space; allow multiple components ----------
        def _knee_fallback(i=None):
            name = "knee" if i is None else f"knee_{i}"
            return pm.TruncatedNormal(name, mu=60.0, sigma=30.0, lower=1.0, upper=nyq)
        def _slope_fallback(i=None):
            name = "slope" if i is None else f"slope_{i}"
            return pm.TruncatedNormal(name, mu=-2.0, sigma=1.0, lower=-5.0, upper=-0.5)
        def _chi_from_slope(slope_var, i=None):
            name = "chi" if i is None else f"chi_{i}"
            return pm.Deterministic(name, -slope_var)
        def _b_fallback(i=None):
            name = "b" if i is None else f"b_{i}"
            y_pos = y[(y > 0) & np.isfinite(y)]
            mu_b = float(np.log10(np.median(y_pos))) if y_pos.size else 0.0
            return pm.Normal(name, mu=mu_b, sigma=10.0)

        ap_terms = []
        for i in range(int(n_aperiodics)):
            # Build parameter set matching predictor's param order
            pvals: Dict[str, Any] = {}
            for pname in ap_pred.params:
                key = pname if n_aperiodics == 1 else f"{pname}_{i}"
                if pname == "knee":
                    var = _make_param(key, aperiodic_param_specs.get(key), lambda: _knee_fallback(i if n_aperiodics>1 else None))
                elif pname == "chi":
                    if key in aperiodic_param_specs:
                        var = _make_param(key, aperiodic_param_specs[key], lambda: pm.Normal(key, 0.0, 5.0))
                    else:
                        var = _chi_from_slope(_slope_fallback(i if n_aperiodics>1 else None), i if n_aperiodics>1 else None)
                elif pname == "b":
                    var = _make_param(key, aperiodic_param_specs.get(key), lambda: _b_fallback(i if n_aperiodics>1 else None))
                else:
                    var = _make_param(key, aperiodic_param_specs.get(key), lambda: pm.Normal(key, 0.0, 10.0))
                pvals[pname] = var

            ap_i = ap_pred.pt_func(f_t, *[pvals[p] for p in ap_pred.params])  # linear
            ap_i = pm.math.clip(ap_i, 1e-40, np.inf)
            ap_terms.append(ap_i)

        if ap_terms:
            P_ap_components = pt.stack(ap_terms, axis=0)             # (M,F)
            pm.Deterministic("P_ap_components", P_ap_components)
            P_ap = pm.Deterministic("P_ap", pt.sum(P_ap_components, axis=0))
        else:
            P_ap = pm.Deterministic("P_ap", pt.zeros_like(f_t))

        L_ap = pm.Deterministic("L_ap", pm.math.log10(P_ap))

        # ---------- Rhythmic bumps in log10 space ----------
        L_bumps = pt.zeros_like(f_t)
        for j, (lo, hi) in enumerate(rb):
            center = _make_param(f"center_{j}", rhythm_param_specs.get(f"center_{j}"),
                                 lambda: pm.Uniform(f"center_{j}", lower=float(lo), upper=float(hi)))
            sigma  = _make_param(f"sigma_{j}",  rhythm_param_specs.get(f"sigma_{j}"),
                                 lambda: pm.TruncatedNormal(f"sigma_{j}", mu=pri['sigma_mu'], sigma=pri['sigma_sigma'],
                                                            lower=pri['sigma_bounds'][0], upper=pri['sigma_bounds'][1]))
            G = rh_pred.pt_func(f_t, center, sigma, nyq_t)
            a_log_j = _make_param(f"a_log_{j}", rhythm_param_specs.get(f"a_log_{j}"),
                                  lambda: pm.HalfNormal(f"a_log_{j}", sigma=pri['a_log_sigma']))
            L_bumps = L_bumps + a_log_j * G

        R_factor = pm.Deterministic("R_factor", pm.math.pow(10.0, L_bumps))   # unitless multiplier
        P_rh_add = pm.Deterministic("P_rh_add", P_ap * (R_factor - 1.0))      # additive above baseline
        mu       = pm.Deterministic("mu", P_ap * R_factor)                     # full spectrum

        alpha = pt.as_tensor_variable(float(k_tapers))
        pm.Gamma("y", alpha=alpha, beta=alpha / mu, observed=np.asarray(y_lin, dtype=float))

    return m





def build_multiplicative_model_legacy(
    freqs: np.ndarray,
    y_lin: np.ndarray,
    *,
    k_tapers: float = 1.0,
    n_rhythms: int = 0,
    rhythm_bands: Optional[Iterable[Tuple[float, float]]] = None,
    priors: Optional[Dict[str, Any]] = None,
    aperiodic_predictor: Union[str, Predictor, None] = None,       # default uses 1/f with offset (log10 via transform)
    rhythm_predictor: Union[str, Predictor, None] = None,          # default mirrored Gaussian (shape only)
    aperiodic_param_specs: Optional[Dict[str, ParamSpec]] = None,  # e.g. {"b": 2.1, "knee": {"dist": pm.LogNormal("knee", ...)}, "chi": ...}
    rhythm_param_specs: Optional[Dict[str, ParamSpec]] = None,     # e.g. {"center_0": 10., "sigma_0": {"dist": pm.HalfNormal("sigma_0", 2.0)}, ...}
):
    """
    Multiplicative spectrum (FOOOF-like) in log10 space:

        L_tot(f) = log10( Aperiodic(f) ) + sum_j a_log_j * G_j(f)
        mu(f) = 10 L_tot(f)

    Defaults reproduce the current model:
      - Aperiodic(f) = 10b / (knee + f**chi) (we take log10 inside)
      - Rhythms G_j are mirrored Gaussians; a_log_j ~ HalfNormal(pri['a_log_sigma'])
    the user can swap in any predictor with the Predictor(pt_func/np_func, params=...) spec and
    fix or fit each parameter via *_param_specs.
    """
    f = np.asarray(freqs, dtype=float)
    nyq = float(np.max(f))
    rb = _ensure_bands(n_rhythms, rhythm_bands)

    # Priors (kept minimal here; only rhythm amp sigma needed by default)
    pri = dict(
        sigma_mu=3.0, sigma_sigma=2.0, sigma_bounds=(0.5, 12.0),
        a_log_sigma=1.0,   # HalfNormal scale for bump heights in LOG space
    )
    if priors:
        pri.update(priors)

    # Choose predictors (defaults preserve existing behavior)
    ap_pred = _get_predictor(aperiodic_predictor, APERIODIC_1OVERF)   # returns *linear* power; we'll take log10
    rh_pred = _get_predictor(rhythm_predictor,    MIRRORED_GAUSSIAN)  # returns unitless shape (0..1-ish)

    aperiodic_param_specs = aperiodic_param_specs or {}
    rhythm_param_specs    = rhythm_param_specs or {}

    y = np.asarray(y_lin, dtype=float)
    with pm.Model() as m:
        f_t = pt.as_tensor_variable(f)

        nyq_t = m.named_vars.get("nyquist")
        if nyq_t is None:
            nyq_t = _as_data("nyquist", float(nyq))

        # ---------- Aperiodic (single component in multiplicative model) ----------
        # Provide sensible defaults for common param names if the predictor uses them.
        def _knee_fallback():
            # Wide, weakly-informative around typical EEG knees
            return pm.TruncatedNormal("knee", mu=60.0, sigma=30.0, lower=1.0, upper=nyq)
        def _slope_fallback():
            return pm.TruncatedNormal("slope", mu=-2.0, sigma=1.0, lower=-5.0, upper=-0.5)
        def _chi_from_slope(slope_var):
            return pm.Deterministic("chi", -slope_var)
        def _b_fallback():
            # Center b to rough log10 power level for stability if needed
            y_pos = y[(y > 0) & np.isfinite(y)]
            mu_b = float(np.log10(np.median(y_pos))) if y_pos.size else 0.0
            return pm.Normal("b", mu=mu_b, sigma=10.0)

        ap_param_values: Dict[str, Any] = {}
        for pname in ap_pred.params:
            if pname == "knee":
                var = _make_param("knee", aperiodic_param_specs.get("knee"), _knee_fallback)
            elif pname == "chi":
                if "chi" in aperiodic_param_specs:
                    var = _make_param("chi", aperiodic_param_specs["chi"], lambda: pm.Normal("chi", 0.0, 5.0))
                else:
                    slope_var = _slope_fallback()
                    var = _chi_from_slope(slope_var)
            elif pname == "b":
                var = _make_param("b", aperiodic_param_specs.get("b"), _b_fallback)
            else:
                # Generic default: Normal(0,10)
                var = _make_param(pname, aperiodic_param_specs.get(pname),
                                  lambda: pm.Normal(pname, 0.0, 10.0))
            ap_param_values[pname] = var

        # ap_pred gives *linear* power; convert to log10 safely
        ap_lin = ap_pred.pt_func(f_t, *[ap_param_values[p] for p in ap_pred.params])
        ap_lin = pm.math.clip(ap_lin, 1e-40, np.inf)
        L_ap   = pm.Deterministic("L_ap", pm.math.log10(ap_lin))

        # ---------- Rhythmic bumps in log space ----------
        L_bumps = pt.zeros_like(f_t)
        for j, (lo, hi) in enumerate(rb):
            center = _make_param(f"center_{j}", rhythm_param_specs.get(f"center_{j}"),
                                lambda: pm.Uniform(f"center_{j}", lower=float(lo), upper=float(hi)))
            sigma  = _make_param(f"sigma_{j}",  rhythm_param_specs.get(f"sigma_{j}"),
                                lambda: pm.TruncatedNormal(f"sigma_{j}", mu=pri['sigma_mu'], sigma=pri['sigma_sigma'],
                                                            lower=pri['sigma_bounds'][0], upper=pri['sigma_bounds'][1]))

            # USE the pre-created nyq_t; do NOT recreate inside the loop
            G = rh_pred.pt_func(f_t, center, sigma, nyq_t)

            a_log_j = _make_param(f"a_log_{j}", rhythm_param_specs.get(f"a_log_{j}"),
                                lambda: pm.HalfNormal(f"a_log_{j}", sigma=pri['a_log_sigma']))
            L_bumps = L_bumps + a_log_j * G

        L_tot = pm.Deterministic("L_tot", L_ap + L_bumps)
        mu    = pm.Deterministic("mu", pm.math.pow(10.0, L_tot))

        
        # L_bumps = pt.zeros_like(f_t)
        # for j, (lo, hi) in enumerate(rb):
        #     center = _make_param(f"center_{j}", rhythm_param_specs.get(f"center_{j}"),
        #                          lambda: pm.Uniform(f"center_{j}", lower=float(lo), upper=float(hi)))
        #     sigma = _make_param(f"sigma_{j}", rhythm_param_specs.get(f"sigma_{j}"),
        #                          lambda: pm.TruncatedNormal(f"sigma_{j}", mu=pri['sigma_mu'], sigma=pri['sigma_sigma'],
        #                                                     lower=pri['sigma_bounds'][0], upper=pri['sigma_bounds'][1]))
        #     nyq_t = _as_data("nyquist", float(nyq))
        #     G = rh_pred.pt_func(f_t, center, sigma, nyq_t)

        #     a_log_j = _make_param(f"a_log_{j}", rhythm_param_specs.get(f"a_log_{j}"),
        #                           lambda: pm.HalfNormal(f"a_log_{j}", sigma=pri['a_log_sigma']))
        #     L_bumps = L_bumps + a_log_j * G

        # L_tot = pm.Deterministic("L_tot", L_ap + L_bumps)
        # mu = pm.Deterministic("mu", pm.math.pow(10.0, L_tot))

        alpha = pt.as_tensor_variable(float(k_tapers))
        pm.Gamma("y", alpha=alpha, beta=alpha / mu, observed=np.asarray(y_lin, dtype=float))

    return m

