#!/usr/bin/env python3
"""
Generate Figure 2 for the Gamma observation-model height-estimator comparison.

The script simulates spectra from the specified ground truth, evaluates the
single-frequency height estimators, and writes the manuscript figure plus
numeric payloads for reproducibility.
"""

from __future__ import annotations
import os
from pathlib import Path
from multiprocessing import freeze_support
import argparse, json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from SL_GPsim import spectrum
from SL_GPsim.simulation import make_broadband_predictor
from spectral_connectivity import Multitaper, Connectivity


PROJECT_ROOT = Path(os.environ.get('SPECTRAL_DECOMP_ROOT', os.getcwd())).expanduser().resolve()
# ── Config ───────────────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / 'Output' / 'Results' / 'FiguresIntermediate' / 'Figure_2' / 'Figure_output'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUT_DIR / "Figure_2.png"   # keep same names per request
SVG_PATH = OUT_DIR / "Figure_2.svg"
NPZ_PATH = OUT_DIR / "Figure_2_data.npz"
MAT_PATH = OUT_DIR / "Figure_2_data.mat"
CFG_PATH = OUT_DIR / "Figure_2_config.json"


FS = 1000
DURATION = 60.0
EXPONENT = 2.0
OFFSET = 0.5
KNEE = 90.0
PEAKS_REQUESTED = [{'freq': 12.0, 'amplitude': 0.0, 'sigma': 0.0}]  # → sanitized to []
AVG_RATE = 0.0
MODE = "additive"
RNG_SEED = 42
N_ITER = 50

K_TAPERS = 3
TIME_HALF_BANDWIDTH_PRODUCT = 2
FREQ_MIN = 1.0
CUTOFF = 200.0

FRAMEWORKS = ['Gamma-Id', 'Gamma-Log', 'OLS-Id', 'OLS-Log']

# ── Style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "svg.fonttype": "none", "axes.unicode_minus": False, "figure.facecolor": "white",
})
sns.set(style="ticks")
mpl.rcParams["axes.grid"] = False


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sanitize_peaks(peaks):
    if peaks is None:
        return []
    return [p for p in peaks if (p.get("amplitude", 0.0) > 0.0 and p.get("sigma", 0.0) > 0.0)]

def simulate_and_mt_psd(fs, duration, exponent, offset, knee, peaks, avg_rate, seed, mode):
    ts = spectrum(
        sampling_rate=fs, duration=duration,
        aperiodic_exponent=exponent, aperiodic_offset=offset,
        knee=knee, peaks=_sanitize_peaks(peaks), average_firing_rate=avg_rate,
        direct_estimate=False, plot=False, random_state=seed, mode=mode
    ).time_domain.combined_signal

    mt = Multitaper(
        ts[:, np.newaxis, np.newaxis], sampling_frequency=fs, n_tapers=K_TAPERS,
        start_time=0.0, time_halfbandwidth_product=TIME_HALF_BANDWIDTH_PRODUCT,
        time_window_duration=duration, time_window_step=duration
    )
    conn = Connectivity.from_multitaper(mt)
    freqs_emp = conn.frequencies
    psd_emp = conn.power().squeeze()

    step = int(2 * TIME_HALF_BANDWIDTH_PRODUCT)
    return freqs_emp[::step], psd_emp[::step]

def theoretical_broadband(fs, duration, exponent, offset, knee, peaks, seed, mode):
    fd = spectrum(
        sampling_rate=fs, duration=duration,
        aperiodic_exponent=exponent, aperiodic_offset=offset,
        knee=knee, peaks=_sanitize_peaks(peaks), average_firing_rate=0.0,
        direct_estimate=True, plot=False, random_state=seed, mode=mode
    ).frequency_domain
    return fd.frequencies, fd.combined_spectrum

def build_designs(freqs_fit, exponent, knee, fmin, fmax, eps=1e-300):
    bb = make_broadband_predictor(freqs_fit, fmin, fmax, exponent=exponent, knee=(knee or 0.0))
    bb = np.asarray(bb, dtype=float)
    bb[bb <= 0] = eps
    X_id  = bb[:, None]                     # identity-link: y ≈ α*bb
    lbb   = np.log(bb)                      # log-link offset
    X_const = np.ones((bb.size, 1))         # intercept-only for log-link models
    return dict(X_id=X_id, lbb=lbb, X_const=X_const)

def fit_four_estimators(y, X_id, lbb, X_const, eps=1e-300):
    out = {}
    gi = sm.GLM(y, X_id, family=sm.families.Gamma(link=sm.families.links.identity())).fit()
    out['Gamma-Id'] = (X_id @ gi.params).ravel()
    gl = sm.GLM(y, X_const, family=sm.families.Gamma(link=sm.families.links.log()), offset=lbb).fit()
    out['Gamma-Log'] = np.exp(X_const @ gl.params + lbb)
    oi = sm.OLS(y, X_id).fit()
    out['OLS-Id'] = (X_id @ oi.params).ravel()
    y_pos = np.maximum(y, eps)
    resid_target = np.log(y_pos) - lbb
    ol = sm.OLS(resid_target, X_const).fit()
    out['OLS-Log'] = np.exp((X_const @ ol.params).ravel() + lbb)
    return out

def prepare_or_load(force: bool = False):
    if NPZ_PATH.exists() and MAT_PATH.exists() and not force:
        npz = np.load(NPZ_PATH, allow_pickle=True)
        data = {k: npz[k].item() if k == "preds" else npz[k] for k in npz.files}
        if not isinstance(data["preds"], dict):
            data["preds"] = dict(data["preds"])
        return data

    f_dense, s_dense = theoretical_broadband(FS, DURATION, EXPONENT, OFFSET, KNEE, PEAKS_REQUESTED, RNG_SEED, MODE)
    step_bins = int(2 * TIME_HALF_BANDWIDTH_PRODUCT)
    f_indep = f_dense[::step_bins]
    s_indep = np.interp(f_indep, f_dense, s_dense)
    mask = (f_indep >= FREQ_MIN) & (f_indep <= CUTOFF)
    freqs_fit = f_indep[mask]
    true_spec = s_indep[mask]

    idx_80 = int(np.argmin(np.abs(freqs_fit - 80.0)))
    true_80 = float(np.interp(80.0, f_dense, s_dense))

    psd_matrix = np.zeros((N_ITER, freqs_fit.size))
    mt_example_freqs = None
    mt_example_psd = None
    for i in range(N_ITER):
        fe, pe = simulate_and_mt_psd(FS, DURATION, EXPONENT, OFFSET, KNEE, PEAKS_REQUESTED, AVG_RATE, RNG_SEED + i, MODE)
        pe_indep = np.interp(f_indep, fe, pe)
        psd_matrix[i, :] = pe_indep[mask]
        if i == 0:
            mt_example_freqs = freqs_fit.copy()
            mt_example_psd = psd_matrix[0, :].copy()

    designs = build_designs(freqs_fit, EXPONENT, KNEE, FREQ_MIN, CUTOFF)
    preds = {fw: np.zeros_like(psd_matrix) for fw in FRAMEWORKS}
    for i in range(N_ITER):
        y = psd_matrix[i, :]
        fit = fit_four_estimators(y, designs["X_id"], designs["lbb"], designs["X_const"])
        for fw in FRAMEWORKS:
            preds[fw][i, :] = fit[fw]

    pack = dict(
        freqs_fit=freqs_fit, true_spec=true_spec,
        mt_example_freqs=mt_example_freqs, mt_example_psd=mt_example_psd,
        preds=preds, true_80=true_80, idx_80=idx_80,
        frameworks=np.array(FRAMEWORKS, dtype=object),
    )

    np.savez_compressed(NPZ_PATH, **{k: (v if k != "preds" else v) for k, v in pack.items()})
    matsav = {"freqs_fit": freqs_fit, "true_spec": true_spec,
              "mt_example_freqs": mt_example_freqs, "mt_example_psd": mt_example_psd,
              "true_80": true_80, "idx_80": idx_80}
    for fw in FRAMEWORKS:
        matsav[f"preds_{fw.replace('-', '_')}"] = preds[fw]
    sio.savemat(MAT_PATH, matsav, do_compression=True)

    with CFG_PATH.open("w") as fh:
        json.dump({
            "FS": FS, "DURATION": DURATION, "EXPONENT": EXPONENT, "OFFSET": OFFSET, "KNEE": KNEE,
            "PEAKS_REQUESTED": PEAKS_REQUESTED, "AVG_RATE": AVG_RATE, "MODE": MODE,
            "RNG_SEED": RNG_SEED, "N_ITER": N_ITER, "K_TAPERS": K_TAPERS,
            "TIME_HALF_BANDWIDTH_PRODUCT": TIME_HALF_BANDWIDTH_PRODUCT,
            "FREQ_MIN": FREQ_MIN, "CUTOFF": CUTOFF
        }, fh, indent=2)

    return pack

# ── Plotting ──────────────────────────────────────────────────────────────────
def violin_80(ax, freqs_fit, preds_dict, idx_80, true_80, palette):
    """Log10-power violins (unchanged) with jittered points."""
    rows = []
    for fw in FRAMEWORKS:
        vals = preds_dict[fw][:, idx_80]
        rows.append(pd.DataFrame({"framework": fw, "value": np.log10(vals)}))
    df = pd.concat(rows, ignore_index=True)

    sns.violinplot(
        data=df, x="framework", y="value",
        order=FRAMEWORKS, palette=palette, cut=4, inner="quartile", width=0.85, ax=ax
    )
    max_pts = 1500
    if len(df) > max_pts:
        df_pts = (df.groupby("framework", group_keys=False)
                    .apply(lambda g: g.sample(n=min(max_pts//len(FRAMEWORKS), len(g)),
                                              random_state=0)))
    else:
        df_pts = df
    sns.stripplot(
        data=df_pts, x="framework", y="value",
        order=FRAMEWORKS, ax=ax, color="k", size=2.2, jitter=0.18, alpha=0.35
    )
    ax.collections[-1].set_rasterized(True)
    ax.axhline(np.log10(true_80), color="k", lw=1.2, alpha=0.9)
    ax.set_xlabel("")
    ax.set_ylabel("log10 Power at 80 Hz")
    ax.tick_params(axis="x", rotation=0)
    if ax.legend_:
        ax.legend_.remove()
    return df

def left_mt_with_four_preds(ax, freqs, mt_f, mt_p, preds_dict, palette, true_spec):
    """
    One log-log panel:
      • multitaper PSD (gray)
      • four predicted spectra (from the same example PSD: row 0 of preds)
    """
    #ax.loglog(mt_f, mt_p, color="0.55", lw=1.5, alpha=0.85, label="Multitaper (example)")
    ax.loglog(mt_f, mt_p, color="0.80", lw=1.2, alpha=0.9,
          label="Multitaper (example)", zorder=-10)
    for fw in FRAMEWORKS:
        z = 5 if fw == "Gamma-Id" else 3
        ax.loglog(freqs, preds_dict[fw][0, :],
                lw=1.2, alpha=0.95, color=palette[fw],
                label=fw, zorder=z)
    ax.legend(fontsize=10, frameon=False, ncol=2)
    #ax.loglog(freqs, true_spec, color="k", lw=1.2, alpha=0.98, zorder=10, label="Ground truth PSD")
    ax.loglog(freqs, true_spec, color="k", lw=1.2, alpha=0.98, zorder=10, label=r"$\mu$")




    
    #for fw in FRAMEWORKS:
        #ax.loglog(freqs, preds_dict[fw][0,:], lw=2.0, alpha=0.95, color=palette[fw], label=fw)
    #    ax.loglog(freqs, preds_dict[fw][0,:], lw=2.0, alpha=0.65, color=palette[fw], label=fw)


    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(False)
    ax.legend(fontsize=10, frameon=True, ncol=2)

def make_figure(data):
    freqs   = data["freqs_fit"]
    mt_f    = data["mt_example_freqs"]
    mt_p    = data["mt_example_psd"]
    preds   = data["preds"]

    DEEP = sns.color_palette("deep")
    COL_BLUE   = DEEP[0]   # #4C72B0
    COL_ORANGE = DEEP[1]   # #DD8452
    COL_GREEN  = DEEP[2]   # #55A868
    COL_RED    = DEEP[3]   # #C44E52

    palette = {
        "Gamma-Id": COL_ORANGE,
        "Gamma-Log": COL_GREEN,
        "OLS-Log": COL_BLUE,
        "OLS-Id": COL_RED,
    }

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.4], wspace=0.4)

    ax_left = fig.add_subplot(gs[0, 0])
    left_mt_with_four_preds(ax_left, freqs, mt_f, mt_p, preds, palette, data["true_spec"])

    ax_right = fig.add_subplot(gs[0, 1])
    violin_80(ax_right, freqs, preds, data["idx_80"], data["true_80"], palette)
    ax_right.grid(False)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300)
    fig.savefig(SVG_PATH)
    plt.close(fig)



# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="One-row layout: MT + four preds | log10 violin at 80 Hz.")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists.")
    args = parser.parse_args()

    data = prepare_or_load(force=args.force)
    make_figure(data)

    print("Saved:")
    print(f"  • {PNG_PATH}")
    print(f"  • {SVG_PATH}")
    print("Cache:")
    print(f"  • {NPZ_PATH}")
    print(f"  • {MAT_PATH}")
    print(f"  • {CFG_PATH}")

if __name__ == "__main__":
    freeze_support()
    main()
