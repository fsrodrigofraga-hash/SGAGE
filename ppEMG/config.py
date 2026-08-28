# ============================================================
# ppEMG.config — configuration dataclasses, population models
# and the reduced frequency grid
# ============================================================
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift


# ============================================================
# Configuration dataclasses
# ============================================================
@dataclass(frozen=True)
class WaveformConfig:
    """Waveform generator configuration (frequency-domain compact-binary waveform)."""
    duration: float = 4
    sampling_frequency: float = 4096
    binary_type: str = "BBH"  # "BBH", "BNS", "BHNS", "NSBH"
    waveform_approximant: str = "IMRPhenomD"
    reference_frequency: float = 25.0
    minimum_frequency: float = 10.0
    maximum_frequency: Optional[float] = None  # if set, passed to waveform_arguments (when supported)

    # NEW: modes (defaults keep backward behavior)
    inspiral_only: bool = False
    disable_inspiral_cutoff: bool = False


@dataclass(frozen=True)
class FreqGridConfig:
    """Reduced frequency grid configuration (log-spaced)."""
    fmin: float = 10.0
    fmax: float = 4096.0
    N_freq_eff: int = 2000


@dataclass(frozen=True)
class CorrectionContext:
    """
    Reusable context holding fixed objects/settings:
    - population models (mass + redshift)
    - waveform config
    - reduced frequency grid and mapping indices into the full waveform frequency array
    """
    models: Dict[str, Any]
    wf_cfg: WaveformConfig
    fgrid_cfg: FreqGridConfig
    frequencies: np.ndarray
    idx_freq: np.ndarray


# ============================================================
# Builders: waveform generator, frequency grid, population models
# ============================================================
def build_models(z_max: float = 10.0) -> Dict[str, Any]:
    """Build default population models (mass + redshift).

    Key names MUST be "mass" and "redshift" — these are the keys that
    popstock's PopulationOmegaGW and _ensure_popstock_args_and_fiducial
    expect. Using any other names (e.g. "mass_model") causes silent
    mismatch between injection and recovery probability weighting.
    """
    return {
        "mass": SinglePeakSmoothedMassDistribution(),
        "redshift": MadauDickinsonRedshift(z_max=z_max),
    }


def build_frequency_grid(full_frequencies: np.ndarray, cfg: FreqGridConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a reduced log-spaced frequency grid and indices that map each reduced frequency
    to the closest (or next) entry in the full frequency array.
    """
    fmax_eff = min(float(cfg.fmax), float(full_frequencies[-1]))
    if cfg.fmax > full_frequencies[-1]:
        warnings.warn(
            f"Requested fmax={cfg.fmax} Hz, but waveform Nyquist is {full_frequencies[-1]:.1f} Hz. "
            f"Clipping fmax to {fmax_eff:.1f} Hz. "
            "If you truly want 4096 Hz, set sampling_frequency >= 8192.",
            RuntimeWarning,
        )
    freqs = np.logspace(np.log10(cfg.fmin), np.log10(fmax_eff), int(cfg.N_freq_eff))
    idx = np.searchsorted(full_frequencies, freqs)
    idx = np.clip(idx, 0, len(full_frequencies) - 1)
    return freqs, idx
