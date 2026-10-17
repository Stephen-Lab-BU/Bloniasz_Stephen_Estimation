SL_specdecomp
=============

A small, user-friendly **PyMC-based** package for neural **power spectral decomposition**.

Two Modes
----------

- **additive**: aperiodic *1/f^χ* **plus** rhythmic bumps (mirrored Gaussian-shaped)
- **FOOOF_spectrum**: FOOOF-like multiplicative model in log10-space (returns only total spectrum)

**Inputs**: any frequency grid (``freqs``) and PSD (``psd``) you provide (e.g., your multitaper estimate).

**Outputs**: posterior mean total spectrum, and for ``additive``, the decomposed **broadband** and **rhythms**.

**One-call API**::

   from SL_specdecomp import Decompose

Example
-------

.. code-block:: python

   import numpy as np
   from SL_specdecomp import Decompose

   # freqs, psd = your multitaper estimate (linear power), 1D arrays
   model = Decompose(
       freqs, psd,
       mode="additive",               # or "FOOOF_spectrum"
       n_rhythms=2,
       rhythm_bands=[(0,4), (8,12)],
       k_tapers=3,
       sample_kwargs=dict(draws=800, tune=800, chains=2, target_accept=0.9),
       plot=True
   )

   # Access pieces (additive only)
   total = model.estimated_spectrum
   rhy   = model.rhythms
   broad = model.broadband

   # You can also plot later:
   model.plot()


Install Required Packages
-------------------------

While the package is in development, we recommend installing in the following way.

Install Mamba through Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mamba is a drop-in replacement for Conda, but is faster and better at resolving dependency conflicts.

.. code-block:: bash

   conda install mamba -n base -c conda-forge


Create an Isolated Environment & Install SL_specdecomp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Stephen-Lab-BU/SL_specdecomp.git
   cd SL_specdecomp
   mamba env create -f environment.yml
   mamba activate SL_specdecomp
   python -m pip install git+https://github.com/Stephen-Lab-BU/SL_specdecomp.git


Requirements
------------

See ``requirements.txt`` or ``environment.yml`` for full dependency details.


Convenience: from_multitaper
----------------------------

If you have a time series and want the package to compute a multitaper PSD (via
`spectral_connectivity <https://pypi.org/project/spectral-connectivity/>`_) before fitting:

.. code-block:: python

   from SL_specdecomp import from_multitaper

   model = from_multitaper(
       signal, fs,
       mode="additive",
       n_rhythms=2,
       rhythm_bands=[(0,4), (8,12)],
       time_halfbandwidth_product=3,
       n_tapers=5,
       freq_min=1.0, freq_max=200.0,
       downsample_independent=True,       # stride ≈ 2*TW for ~independent bins
       sample_kwargs=dict(draws=800, tune=800, chains=2, target_accept=0.9),
       plot=True
   )
