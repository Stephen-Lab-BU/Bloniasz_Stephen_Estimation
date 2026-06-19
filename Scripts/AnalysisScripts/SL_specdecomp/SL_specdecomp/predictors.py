# predictors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any
import numpy as np
import pytensor.tensor as pt
import pymc as pm

import numpy as _np
import pytensor.tensor as _pt
import pymc as _pm

# pow(x, y)
if not hasattr(_pm.math, "pow"):
    def _pm_pow(x, y):
        return _pt.power(x, y)
    _pm.math.pow = _pm_pow  # type: ignore[attr-defined]

# log10(x)
if not hasattr(_pm.math, "log10"):
    def _pm_log10(x):
        return _pm.math.log(x) / _np.log(10.0)
    _pm.math.log10 = _pm_log10  # type: ignore[attr-defined]

# clip(x, a_min, a_max)
if not hasattr(_pm.math, "clip"):
    def _pm_clip(x, a_min, a_max):
        return _pt.clip(x, a_min, a_max)
    _pm.math.clip = _pm_clip  # type: ignore[attr-defined]

# ---------- Small spec that names the ordered, non-freq params ----------
@dataclass(frozen=True)
class Predictor:
    name: str
    params: Tuple[str, ...]                      # ordered param names after "freqs"
    np_func: Callable[..., np.ndarray]           # (freqs, *params) -> np.ndarray
    pt_func: Callable[..., pt.TensorVariable]    # (f_t, *params) -> pt.TensorVariable

# ---------- Default aperiodic (Lorentzian / 1/f with offset) ----------
def _np_aperiodic_1overf(freqs: np.ndarray, b: float, knee: float, chi: float) -> np.ndarray:
    f = np.asarray(freqs, float)
    return (10.0 ** b) / (knee + np.power(f, chi))

def _pt_aperiodic_1overf(f_t, b, knee, chi):
    return pm.math.pow(10.0, b) / (knee + pm.math.pow(f_t, chi))

APERIODIC_1OVERF = Predictor(
    name="aperiodic_1overf",
    params=("b", "knee", "chi"),
    np_func=_np_aperiodic_1overf,
    pt_func=_pt_aperiodic_1overf,
)

# ---------- Default rhythmic (mirrored Gaussian) ----------
def _np_mirrored_gaussian(freqs, center, sigma, nyquist):
    f = np.asarray(freqs, float)
    g_pos = np.exp(-0.5 * ((f - center) / sigma) ** 2)
    g_neg = np.exp(-0.5 * ((f + center) / sigma) ** 2)
    g_ref = g_pos + g_neg
    g_ref[f == 0.0] = g_pos[f == 0.0]
    g_ref[f == nyquist] = g_pos[f == nyquist]
    return g_ref

def _pt_mirrored_gaussian(f_t, center, sigma, nyquist):
    z  = (f_t - center) / sigma
    z2 = (f_t + center) / sigma
    g_pos = pm.math.exp(-0.5 * pm.math.sqr(z))
    g_neg = pm.math.exp(-0.5 * pm.math.sqr(z2))
    is_dc = pm.math.eq(f_t, 0.0)
    is_nq = pm.math.eq(f_t, nyquist)
    return pm.math.switch(is_dc | is_nq, g_pos, g_pos + g_neg)

MIRRORED_GAUSSIAN = Predictor(
    name="mirrored_gaussian",
    params=("center", "sigma", "nyquist"),
    np_func=_np_mirrored_gaussian,
    pt_func=_pt_mirrored_gaussian,
)

# ---------- Lightweight registry for convenience (optional) ----------
REGISTRY: Dict[str, Predictor] = {
    APERIODIC_1OVERF.name: APERIODIC_1OVERF,
    MIRRORED_GAUSSIAN.name: MIRRORED_GAUSSIAN,
}
