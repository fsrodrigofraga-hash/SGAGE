# ============================================================
# ppEMG — parametrized post-Einsteinian Modified Gravity
#         (formerly CorrectionOmegaGW)
#
# Library of modified-gravity corrections applied to the astrophysical
# Omega_GW computed by popstock.
#
# Rename: CorrectionOmegaGW -> ppEMG. The PUBLIC API is IDENTICAL, so
#
#     import CorrectionOmegaGW as corr      # before
#     import ppEMG as corr                  # now
#
# works as a drop-in. The monolithic module was split into submodules:
#
#     ppEMG.weights      popstock-consistent weights (w = p/pdraw)
#     ppEMG.config       WaveformConfig / FreqGridConfig / CorrectionContext
#     ppEMG.waveforms    bilby generator, context and |h(f)|^2 cache
#     ppEMG.population   popstock samples -> dataset, probabilities
#     ppEMG.cutoff       inspiral end (ISCO / peak)
#     ppEMG.corrections  ppE correction, G correction, compute_correction
#
# No physics was changed in this refactor: the code was moved verbatim into
# the submodules.
# ============================================================
from __future__ import annotations

from .config import (
    CorrectionContext,
    FreqGridConfig,
    WaveformConfig,
    build_frequency_grid,
    build_models,
)
from .corrections import (
    G_correction,
    compute_Q_ppE,
    compute_correction,
    population_ratio_in_f,
    ppe_correction,
)
from .cutoff import f_isco_source_hz, robust_inspiral_end_frequency_obs
from .population import (
    calculate_probabilities,
    dataset_from_popstock_samples,
    draw_pop_and_dataset_from_popstock,
)
from .waveforms import (
    build_waveform_generator,
    compute_h2_cache_and_fpeak_parallel,
    compute_h2_cache_parallel,
    prepare_context,
)
from .weights import (
    USE_PDRAW_WEIGHTS,
    AlphaLike,
    EventWeightFn,
    resolve_alpha,
    weights_from_probabilities,
)

__version__ = "1.0.0"

__all__ = [
    # config
    "WaveformConfig", "FreqGridConfig", "CorrectionContext",
    "build_models", "build_frequency_grid",
    # waveforms
    "build_waveform_generator", "prepare_context",
    "compute_h2_cache_and_fpeak_parallel", "compute_h2_cache_parallel",
    # population
    "dataset_from_popstock_samples", "draw_pop_and_dataset_from_popstock",
    "calculate_probabilities",
    # cutoff
    "f_isco_source_hz", "robust_inspiral_end_frequency_obs",
    # weights
    "weights_from_probabilities", "resolve_alpha",
    "USE_PDRAW_WEIGHTS", "AlphaLike", "EventWeightFn",
    # corrections
    "compute_Q_ppE", "ppe_correction", "G_correction",
    "population_ratio_in_f", "compute_correction",
    "__version__",
]

# ── pipeline (optional: requires popstock installed) ───────────────────────
from .pipeline import (  # noqa: E402
    LAMBDA_FIDUCIAL,
    PopulationBundle,
    build_population_bundle,
)

__all__ += ["PopulationBundle", "build_population_bundle", "LAMBDA_FIDUCIAL"]
