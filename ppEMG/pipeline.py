# ============================================================
# ppEMG.pipeline — building the fiducial population in ONE place
#
# The plot_* scripts each repeated ~120 lines of popstock boilerplate
# (_ensure_popstock_args / _pop_draw_samples / _pop_calculate_omega /
#  build_population). All of it lives here now, and the result is packed into
# a PopulationBundle — the object the mgtheories theories consume.
# ============================================================
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .config import CorrectionContext, FreqGridConfig, WaveformConfig
from .population import calculate_probabilities, dataset_from_popstock_samples
from .waveforms import compute_h2_cache_and_fpeak_parallel, prepare_context

# gwpopulation model keys used by popstock
MASS_KEYS = ["alpha", "mmin", "mmax", "lam", "mpp", "sigpp", "beta", "delta_m"]
REDSHIFT_KEYS = ["gamma", "kappa", "z_peak"]

#: Fiducial population hyper-parameters (same convention as the plot_* scripts)
LAMBDA_FIDUCIAL: Dict[str, float] = {
    "alpha": 3.5,     # IMF slope
    "lam": 0.03,      # Gaussian peak fraction
    "rate": 15.0,     # local rate [Gpc^-3 yr^-1]
    "gamma": 2.7,     # z evolution (Madau-Dickinson)
    "beta": 1.0,
    "mpp": 33.0,
    "sigpp": 4.5,
    "delta_m": 5.0,
    "kappa": 5.7,
    "z_peak": 1.9,
    "mmin": 5.0,
    "mmax": 85.0,
}


@dataclass
class PopulationBundle:
    """Fiducial population plus everything the corrections need.

    Single interface consumed by mgtheories: any theory takes a
    PopulationBundle and returns an OmegaSet.
    """

    frequencies: np.ndarray
    dataset: Dict[str, np.ndarray]
    probabilities: np.ndarray
    omega_gr: np.ndarray
    h2_cache: Optional[np.ndarray] = None
    f_peak_obs: Optional[np.ndarray] = None
    ctx: Optional[CorrectionContext] = None
    Lambda: Dict[str, float] = field(default_factory=dict)
    disable_inspiral_cutoff: bool = True

    # ── waveform cache construction (only the ppE family needs it) ────────
    def ensure_cache(self, *, nproc: Optional[int] = None, chunksize: int = 50,
                     fast_binning: bool = True, max_bins: int = 100,
                     use_frequency_warp: bool = True) -> "PopulationBundle":
        """Compute h2_cache/f_peak_obs if not present yet (idempotent)."""
        if self.h2_cache is not None:
            return self
        if self.ctx is None:
            raise ValueError("PopulationBundle without ctx: cannot build h2_cache")
        self.h2_cache, self.f_peak_obs = compute_h2_cache_and_fpeak_parallel(
            dataset=self.dataset, idx=self.ctx.idx_freq, wf_cfg=self.ctx.wf_cfg,
            frequencies=self.frequencies, nproc=nproc, chunksize=chunksize,
            fast_binning=fast_binning, max_bins=max_bins,
            use_frequency_warp=use_frequency_warp,
        )
        return self

    @classmethod
    def from_simulated_signal(cls, sim, *, with_cache: bool = True, **cache_kw
                              ) -> "PopulationBundle":
        """Reuse an already precomputed SimulatedSignal instance."""
        omega = np.asarray(sim._omega_fid, float)
        omega = np.where(np.isfinite(omega) & (omega >= 0), omega, 0.0)
        bundle = cls(
            frequencies=np.asarray(sim._ctx.frequencies, float),
            dataset=sim._dataset,
            probabilities=sim._probabilities,
            omega_gr=omega,
            ctx=sim._ctx,
            Lambda=dict(getattr(sim, "Lambda", {}) or {}),
            disable_inspiral_cutoff=bool(
                getattr(sim._ctx.wf_cfg, "disable_inspiral_cutoff", True)),
        )
        return bundle.ensure_cache(**cache_kw) if with_cache else bundle


# ── popstock boilerplate (previously duplicated across 6 scripts) ─────────
def _ensure_popstock_args(pop, Lambda: dict) -> dict:
    """Ensure model_args/fiducial_parameters consistent with the given Lambda."""
    if not hasattr(pop, "model_args") or pop.model_args is None:
        pop.model_args = {}
    pop.model_args["mass"] = MASS_KEYS
    pop.model_args["redshift"] = REDSHIFT_KEYS
    fid = {k: Lambda[k] for k in MASS_KEYS + REDSHIFT_KEYS + ["rate"]
           if k in Lambda}
    for attr in ("fiducial_parameters", "fiducial_params",
                 "fiducial_parameter", "_fiducial_parameters"):
        if hasattr(pop, attr):
            try:
                setattr(pop, attr, fid)
            except Exception:
                pass
    return fid


def _draw_samples(pop, Lambda: dict, n_samples: int, seed: int):
    """draw_and_set_proposal_samples, tolerant to popstock versions."""
    np.random.seed(seed)
    fid = _ensure_popstock_args(pop, Lambda)
    sig = inspect.signature(pop.draw_and_set_proposal_samples)
    for method in ("direct", "grid"):
        try:
            kw = dict(N_proposal_samples=n_samples, mass=method, redshift=method)
            if "seed" in sig.parameters:
                kw["seed"] = seed
            return pop.draw_and_set_proposal_samples(fid, **kw)
        except (UnboundLocalError, TypeError):
            try:
                kw = dict(N_proposal_samples=n_samples)
                if "seed" in sig.parameters:
                    kw["seed"] = seed
                return pop.draw_and_set_proposal_samples(fid, **kw)
            except Exception:
                continue
    raise RuntimeError("Failed to draw proposal samples from popstock.")


def _calculate_omega(pop, Lambda: dict, *, wf_cfg: WaveformConfig, fmin: float):
    """calculate_omega_gw passing only the kwargs the local version supports."""
    sig = inspect.signature(pop.calculate_omega_gw)
    kw = dict(
        waveform_approximant=wf_cfg.waveform_approximant,
        sampling_frequency=wf_cfg.sampling_frequency,
        waveform_minimum_frequency=fmin,
        minimum_frequency=fmin,
        multiprocess=False,
    )
    supported = {k: v for k, v in kw.items() if k in sig.parameters}
    if "Lambda" in sig.parameters:
        return pop.calculate_omega_gw(Lambda=Lambda, **supported)
    return pop.calculate_omega_gw(Lambda, **supported)


def build_population_bundle(
    Lambda: Optional[Dict[str, float]] = None,
    *,
    fmin: float = 10.0,
    fmax: float = 2000.0,
    n_freq: int = 800,
    n_samples: int = 10_000,
    seed: int = 1042,
    waveform_approximant: str = "IMRPhenomD",
    sampling_frequency: float = 4096.0,
    duration: float = 4.0,
    binary_type: str = "BBH",
    inspiral_only: bool = True,
    disable_inspiral_cutoff: bool = True,
    z_max_models: float = 10.0,
    with_cache: bool = True,
    fast_binning: bool = True,
    max_bins: int = 100,
    use_frequency_warp: bool = True,
    chunksize: int = 50,
    nproc: Optional[int] = None,
) -> PopulationBundle:
    """Draw the popstock population and return a ready PopulationBundle.

    Replaces the duplicated `build_population` functions in the plot_* scripts.
    """
    from popstock.PopulationOmegaGW import PopulationOmegaGW

    Lambda = dict(Lambda or LAMBDA_FIDUCIAL)

    wf_cfg = WaveformConfig(
        duration=duration,
        sampling_frequency=sampling_frequency,
        binary_type=binary_type,
        waveform_approximant=waveform_approximant,
        minimum_frequency=fmin,
        inspiral_only=inspiral_only,
        disable_inspiral_cutoff=disable_inspiral_cutoff,
    )
    ctx = prepare_context(
        wf_cfg=wf_cfg,
        fgrid_cfg=FreqGridConfig(fmin=fmin, fmax=fmax, N_freq_eff=n_freq),
        z_max_models=z_max_models,
    )

    pop = PopulationOmegaGW(models=ctx.models, frequency_array=ctx.frequencies)
    _draw_samples(pop, Lambda, n_samples, seed)
    _calculate_omega(pop, Lambda, wf_cfg=wf_cfg, fmin=fmin)

    dataset = dataset_from_popstock_samples(pop, set_extrinsics=True)
    omega_gr = np.asarray(getattr(pop, "omega_gw"), float)
    omega_gr = np.where(np.isfinite(omega_gr) & (omega_gr >= 0), omega_gr, 0.0)
    probabilities = np.asarray(
        calculate_probabilities(dataset, Lambda, ctx.models), float)

    bundle = PopulationBundle(
        frequencies=np.asarray(ctx.frequencies, float),
        dataset=dataset,
        probabilities=probabilities,
        omega_gr=omega_gr,
        ctx=ctx,
        Lambda=Lambda,
        disable_inspiral_cutoff=disable_inspiral_cutoff,
    )
    if with_cache:
        bundle.ensure_cache(nproc=nproc, chunksize=chunksize,
                            fast_binning=fast_binning, max_bins=max_bins,
                            use_frequency_warp=use_frequency_warp)
    return bundle
