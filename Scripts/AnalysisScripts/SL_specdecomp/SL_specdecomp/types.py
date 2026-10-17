# types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import arviz as az

@dataclass
class Decomposition:
    """Container returned by:func:`Decompose`.

    Attributes
    ----------
    freqs: np.ndarray
        Frequency grid (Hz) for the fit (must match observed PSD).
    observed: np.ndarray
        Observed multitaper (or other) power spectral density (linear power).
    estimated_spectrum: np.ndarray
        Posterior mean of the model-implied spectrum (linear power).
    rhythms: Optional[np.ndarray]
        Posterior mean of the rhythmic contribution in linear power.
        - additive mode: sum of rhythmic components (linear).
        - FOOOF_spectrum mode: additive contribution above baseline,
          P_rh_add = P_ap * (R_factor - 1).
    broadband: Optional[np.ndarray]
        Posterior mean of the aperiodic (baseline) spectrum (linear).
    broadband_components: Optional[np.ndarray]
        Posterior mean of individual aperiodic components (M, F) if present.
    r_factor: Optional[np.ndarray]
        (FOOOF_spectrum only) Posterior mean of the rhythmic multiplier,
        R_factor = 10**L_bumps (unitless).
    idata: az.InferenceData
        Full PyMC inference results.
    mode: str
        'additive' or 'FOOOF_spectrum'.
    """
    freqs: np.ndarray
    observed: np.ndarray
    estimated_spectrum: np.ndarray
    rhythms: Optional[np.ndarray]
    broadband: Optional[np.ndarray]
    broadband_components: Optional[np.ndarray]
    r_factor: Optional[np.ndarray]
    idata: az.InferenceData
    mode: str

    @property
    def total(self) -> np.ndarray:
        return self.estimated_spectrum

    def plot(self, plot_components: bool = True, ax=None, title: Optional[str]=None):
        from .plotting import plot_decomposition
        return plot_decomposition(self, plot_components=plot_components, ax=ax, title=title)
