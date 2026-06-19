# Bloniasz_Stephen_Estimation

Analysis and figure-generation code for:

**Spectral decompositions of neural voltage recordings are susceptible to model misspecifications that cause meaningful estimation error**  
Patrick F. Bloniasz and Emily P. Stephen  

DOI: https://doi.org/10.64898/2026.06.12.718232

This repository contains the code for the simulations, analyses, and manuscript figures associated with the paper. The central claim of the work is that common spectral decomposition methods can produce biased rhythmic and broadband estimates when they use model assumptions that are inconsistent with the physical and statistical structure of neural voltage recordings.

## Overview

Power spectra of neural field-potential recordings contain both narrowband rhythmic structure and broadband structure. Many analysis pipelines attempt to separate these components and then interpret the recovered broadband and rhythmic parameters as biomarkers of brain state.

This project evaluates spectral decomposition as an estimation problem. The analyses focus on two forms of model misspecification:

1. **Component structure:** neural field potentials are usually modeled as additive linear superpositions of biophysical processes, whereas several spectral decomposition approaches impose multiplicative structure in power or additive structure in log-power.
2. **Observation model:** averaged direct spectral estimates are Gamma distributed and heteroscedastic, whereas many methods assume Gaussian, homoscedastic residuals.

Using simulations with known ground truth and macaque ECoG during propofol anesthesia, the code compares decompositions produced by `specparam` and `SL_specdecomp`, evaluates recovery of broadband and rhythmic parameters, and computes cross-validated log likelihood for model comparison.

## Repository contents

```text
.
├── Data/
│   └── Data inputs, cached arrays used by scripts
├── Output/
│   └── Results/FinalFigures/Paper_markups/
│       └── Generated manuscript figures and marked-up outputs
├── Scripts/
│   └── AnalysisScripts/
│       └── Figure-generation, simulation, empirical-analysis
├── environment.yml
├── pyproject.toml
├── LICENSE
└── README.md
```

The spectral decomposition framework is implemented in the companion package `SL_specdecomp`; Gaussian-process simulations from analytically specified spectra are implemented in the companion package `SL_GPsim`.

## Main analyses

The scripts in `Scripts/AnalysisScripts/` are organized around the manuscript and supplemental figures. They include analyses for:

- empirical macaque ECoG examples during wakefulness and propofol anesthesia;
- Gamma sampling behavior of averaged direct spectral estimates;
- benchmark simulations with known ground truth;
- comparisons of additive versus multiplicative spectral structure;
- comparisons of Gamma versus Gaussian observation models;
- cross-validated log-likelihood model comparison;
- empirical state-dependent decompositions in awake and anesthetized windows;
- matched simulations generated from empirical spectra;

Scripts include (names may change as clode is cleaned up):

```text
Scripts/AnalysisScripts/00_qc_figure4_outlier_inspector.py
Scripts/AnalysisScripts/01_paper_figure_1_overview_grid.py
Scripts/AnalysisScripts/02_paper_figure_2_gamma_height_estimator.py
Scripts/AnalysisScripts/03_paper_figure_3_known_ground_truth_decomposition_cvll.py
Scripts/AnalysisScripts/03_supp_figure_3_cvll_lineplot.py
Scripts/AnalysisScripts/04_paper_figure_empirical_ecog_state_decomposition.py
Scripts/AnalysisScripts/05_paper_figure_empirical_matched_simulation.py
Scripts/AnalysisScripts/06_supp_figure_height_grid_benchmark.py
```

Run scripts from the repository root so that relative paths resolve consistently.

## Installation

Create the analysis environment from the supplied Conda file:

```bash
conda env create -f environment.yml
conda activate <environment-name>
```

Then install the repository in editable mode:

```bash
python -m pip install -e .
```

Be sure to include the companion packages:

```bash
python -m pip install git+https://github.com/Stephen-Lab-BU/SL_specdecomp.git
python -m pip install git+https://github.com/Stephen-Lab-BU/SL_GPsim.git
```

The empirical and simulation scripts also use standard scientific Python packages, including NumPy, SciPy, pandas, matplotlib, PyMC, ArviZ, and `specparam`. Exact versions should be taken from `environment.yml`.

Some scripts may take substantial time because they run Bayesian inference, repeated simulations, cross-validation, or empirical window-level decompositions. For expensive scripts, check the script-level configuration block before running large jobs.

## Data

The empirical ECoG analyses use previously collected, publicly available macaque electrocorticography recordings from the Neurotycho `Macaca fuscata` monkey dataset. The paper analyzes propofol anesthesia data and compares awake and anesthetized windows.

No new animal experiments were performed for this study. The original experimental data collection and implantation procedures are described in the Neurotycho publications cited in the manuscript.

## Relationship to companion repositories

This repository contains the analysis and figure code for the paper. The reusable methods are split into companion repositories:

- `SL_specdecomp`: Bayesian spectral decomposition with Gamma observation models and additive or multiplicative component structure.
- `SL_GPsim`: simulation of Gaussian processes from analytically specified power spectra.

Use this repository to reproduce the manuscript analyses. Use the companion packages for method development or application to new datasets.

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
