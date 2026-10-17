# plotting.py
from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from .types import Decomposition

def plot_decomposition(model: Decomposition, plot_components: bool = True, ax=None, title: Optional[str]=None, ymin: float = 1e-14):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    else:
        fig = ax.figure

    f = model.freqs
    Y_FLOOR = 1e-40

    def _clip_floor(arr, floor=Y_FLOOR):
        return np.clip(arr, floor, np.inf)

    ax.loglog(f, _clip_floor(model.observed), lw=1.0, alpha=0.6, label="Observed (MT)")
    ax.loglog(f, _clip_floor(model.estimated_spectrum), lw=2.0, label="Posterior mean", zorder=5)

    if plot_components:
        if model.broadband is not None:
            ax.loglog(f, _clip_floor(model.broadband), lw=1.5, ls=":", label="Broadband (mean)")
        if model.rhythms is not None:
            # In FOOOF_spectrum this is P_rh_add; in additive it's the rhythmic sum
            ax.loglog(f, _clip_floor(model.rhythms), lw=1.5, ls="--", label="Rhythms (mean)")

        comps = getattr(model, "broadband_components", None)
        if comps is not None:
            for i, comp in enumerate(np.asarray(comps)):
                ax.loglog(f, _clip_floor(comp), lw=1.0, ls=":", alpha=0.4, label=f"Broadband #{i+1}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    if title:
        ax.set_title(title)

    ax.grid(which="both", ls=":", alpha=0.3)
    ax.set_ylim(bottom=ymin)
    # De-duplicate legends if many components exist
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys())
    fig.tight_layout()
    return ax
