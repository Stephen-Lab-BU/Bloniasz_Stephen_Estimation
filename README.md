# Bloniasz_Stephen_Estimation

Analysis and figure code for **Spectral decompositions of neural voltage recordings are susceptible to model misspecifications that cause meaningful estimation error**, by Patrick F. Bloniasz and Emily P. Stephen.

Paper: https://doi.org/10.64898/2026.06.12.718232

## Contents

- `Scripts/AnalysisScripts/`: simulations, spectral decomposition, cross-validated model comparisons, empirical analyses and figure plotting.
- `Data/`: prepared macaque ECoG inputs.
- `Output/Results/FinalFigures/Paper_markups/`: editable manuscript SVG figures, including Box 1.

The analysis scripts use the companion packages [SL_specdecomp](https://github.com/Stephen-Lab-BU/SL_specdecomp) and [SL_GPsim](https://github.com/Stephen-Lab-BU/SL_GPsim).

## Setup

Run these commands from the repository root:

```sh
mamba env create -f environment.yml
mamba activate SL_SpecDecomp_paper
python -m pip install git+https://github.com/Stephen-Lab-BU/SL_GPsim.git
git submodule update --init
```

Run analysis scripts from the repository root. Script options are available with `--help`, for example:

```sh
python Scripts/AnalysisScripts/04_paper_figure_empirical_ecog_state_decomposition.py --help
```

## Data

The empirical recordings are channel 1 from Sessions 1–3 of Chibi's 2012-07-30 propofol experiment, sampled at 1000 Hz. The original data are available from the [Neurotycho Anesthesia and Sleep Task](https://neurotycho.org/anesthesia-and-sleep-task). Select the ECoG128 MATLAB archive `20120730PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128.zip`.

The prepared MAT files concatenate those sessions in order. Session time and condition-time vectors are offset by the final times of preceding sessions, retaining duplicate boundary times. Empirical scripts read `ECoG_ch1.mat`, `ECoGTime.mat` and `Condition.mat` from `Data/InputData/InputDataFiles`; the Figure 1 scripts use the copies under `Data`.

[Neurotycho](https://neurotycho.org/) states that it has waived copyright and related rights in its downloadable datasets to the extent possible under law. Recording methods are described by Yanagawa et al. (2013), linked on the task page.

## Citation

If you use this code, please cite the manuscript and the relevant companion software repositories.

```bibtex
@misc{bloniasz_stephen_2026_spectral_decomposition,
  title  = {Spectral decompositions of neural voltage recordings are susceptible to model misspecifications that cause meaningful estimation error},
  author = {Bloniasz, Patrick F. and Stephen, Emily P.},
  year   = {2026},
  doi    = {10.64898/2026.06.12.718232},
  url    = {https://doi.org/10.64898/2026.06.12.718232},
  note   = {Preprint}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for details.

## Contact

For questions about the analyses or manuscript code, please open an issue on this repository or contact the corresponding author listed in the manuscript.
