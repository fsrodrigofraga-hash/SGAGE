from __future__ import annotations

# ============================================================
# ParametersEstimator.estimator — main class
# ============================================================
import gc
import hashlib
import inspect
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import bilby
from joblib import Parallel, delayed

import ppEMG as corr

from .config import EstimatorConfig, EventVectorFn, LambdaParamSpec, ParamGridSpec
from .covariance import _ledoit_wolf_shrinkage, estimate_C_pop
from .cpu import _effective_jobs, _resolve_cpu_jobs
from .direct import (
    DirectModelEvaluator,
    OmegaGaussianLikelihoodDirectDiag,
    OmegaGaussianLikelihoodDirectFullCov,
)
from .grid import build_model_grid_nd_parallel
from .interpolation import InterpModelNDWithCache, _nd_multilinear_interp
from .io_utils import (
    _axis_coord_to_theta,
    _axis_values_for_spec,
    _stable_hash_of_dict,
    _theta_to_axis_coord,
    load_npz_flexible,
)
from .likelihoods import (
    OmegaGaussianLikelihoodDiag,
    OmegaGaussianLikelihoodFullCov,
    OmegaOptimalFilterLikelihood,
)
from .models import omega_full_from_corrections
from .plotting import CornerPlotMixin
from .population import precompute_population_objects
from .popstock_utils import (
    _ensure_popstock_args_and_fiducial,
    _infer_required_kwargs,
    _pop_calculate_omega,
)


# Main estimator class
# ============================================================
class ParametersEstimator(CornerPlotMixin):
    r"""
    Usage patterns:

    (1) Scalar alpha_ppE inference (diagonal likelihood):
        est = ParametersEstimator()
        est.load("outputs/simulated_data.npz")
        est.prepare_model()
        result = est.run_inference()

    (2) Full-covariance with population uncertainty (recommended):
        est = ParametersEstimator()
        est.load("outputs/simulated_data.npz")
        est.prepare_model()
        est.estimate_population_covariance(
            a_ref=2.0,
            alpha_ref_scalar=4e-3,
            freq_settings=freq, det_settings=det,
            td_settings=td, welch_settings=welch,
            pop_settings=pop,
        )
        result = est.run_inference()   # uses full_cov automatically

    (3) Noncommutative-like alpha per event:
        def alpha_nc(dataset, Lambda2):
            ...
            return alpha_event_array

        alpha_specs = [ParamGridSpec("Lambda2", -12, 0, 61, transform="log10", latex_label=r"$\Lambda^2$")]
        est = ParametersEstimator(alpha_event_model=alpha_nc, alpha_param_specs=alpha_specs)
        est.load(...)
        est.prepare_model()
        est.estimate_population_covariance(
            a_ref=2.0,
            alpha_ref_callable=lambda d: alpha_nc(d, Lambda2=1.5**4),
            freq_settings=..., ...
        )
        result = est.run_inference()
    """

    def __init__(
        self,
        config: EstimatorConfig = EstimatorConfig(),
        *,
        # alpha per-event model (optional). If None, infer scalar alpha_ppE.
        alpha_event_model: Optional[EventVectorFn] = None,
        alpha_id: str = "alpha_default",
        alpha_param_specs: Optional[List[ParamGridSpec]] = None,

        # G event weights (optional)
        G_event_weight: Optional[EventVectorFn] = None,
        G_id: str = "G_default_1",
        G_param_specs: Optional[List[ParamGridSpec]] = None,

        # Free Lambda hyperparameters (optional).
        # Each LambdaParamSpec defines the prior/grid for one Lambda key.
        # Keys NOT listed here are held fixed at the values passed to prepare_model().
        lambda_param_specs: Optional[List["LambdaParamSpec"]] = None,

        # compute_correction knobs
        chunksize: int = 20,
        fast_binning: bool = True,
        max_bins: int = 400,
        use_frequency_warp: bool = True,
        nproc: Optional[int] = 1,
        disable_inspiral_cutoff: Optional[bool] = None,
    ):
        self.cfg = config
        os.makedirs(self.cfg.outdir, exist_ok=True)
        os.makedirs(self.cfg.grid_cache_dir, exist_ok=True)

        self.alpha_event_model = alpha_event_model
        self.alpha_id = str(alpha_id)
        self.alpha_param_specs = list(alpha_param_specs) if alpha_param_specs is not None else []

        self.G_event_weight = G_event_weight
        self.G_id = str(G_id)
        self.G_param_specs = list(G_param_specs) if G_param_specs is not None else []

        # Free Lambda hyperparameters — these become additional grid/prior dimensions
        self.lambda_param_specs: List[LambdaParamSpec] = (
            list(lambda_param_specs) if lambda_param_specs is not None else []
        )
        # True values for corner-plot truth markers; set via set_lambda_true_values()
        self.lambda_true_values: Dict[str, float] = {}

        self.compute_knobs = dict(
            chunksize=int(chunksize),
            fast_binning=bool(fast_binning),
            max_bins=int(max_bins),
            use_frequency_warp=bool(use_frequency_warp),
            nproc=nproc,
            disable_inspiral_cutoff=disable_inspiral_cutoff,
        )

        # Resolve CPU counts once [MEM-2]
        n_cores = cpu_count() or 1
        _auto_grid, _auto_pool, _auto_cpop = _resolve_cpu_jobs(n_cores)
        self._grid_jobs    = _effective_jobs(config.grid_jobs,    _auto_grid)
        self._sampler_pool = _effective_jobs(config.sampler_pool, _auto_pool)
        self._cpop_jobs    = _effective_jobs(config.cpop_jobs,    _auto_cpop)

        # Garante que o cache do grid fique sempre dentro do outdir escolhido
        if not self.cfg.grid_cache_dir:
            object.__setattr__(self.cfg, 'grid_cache_dir',
                               os.path.join(self.cfg.outdir, "grid_cache"))

        # Loaded data
        self.saved: Optional[Dict[str, Any]] = None
        self.f_bin: Optional[np.ndarray] = None
        self.Om_bin: Optional[np.ndarray] = None
        self.sOm_bin: Optional[np.ndarray] = None
        # Noise info for the optimal-filter likelihood
        self.P1_bin:    Optional[np.ndarray] = None
        self.P2_bin:    Optional[np.ndarray] = None
        self.gamma_bin: Optional[np.ndarray] = None

        # Model grid + interpolator
        self.specs: Optional[List[ParamGridSpec]] = None
        self.axes: Optional[List[np.ndarray]] = None
        self.model_grid: Optional[np.ndarray] = None
        self.interp: Optional[InterpModelNDWithCache] = None

        self.cache_hash: Optional[str] = None

        # Population covariance  [FIX-1,2,3]
        self.C_pop: Optional[np.ndarray] = None          # (n_bins, n_bins)
        self.C_inst: Optional[np.ndarray] = None         # diag(sOm_bin²)
        self._cpop_omega_samples: Optional[np.ndarray] = None  # (K_ok, n_bins)

    # -------------------------
    # Load
    # -------------------------
    def load(self, infile_npz: Optional[str] = None) -> Dict[str, Any]:
        path = self.cfg.infile_npz if infile_npz is None else infile_npz
        saved = load_npz_flexible(path)

        self.saved = saved
        self.f_bin  = np.asarray(saved["f_bin"],           dtype=np.float64)
        self.Om_bin = np.asarray(saved["Omega_hat_bin"],   dtype=np.float64)
        self.sOm_bin = np.asarray(saved["sigma_Omega_bin"], dtype=np.float64)

        # Noise info for the optimal-filter likelihood (optional)
        self.P1_bin    = np.asarray(saved["P1_bin"],    dtype=np.float64) if "P1_bin"    in saved else None
        self.P2_bin    = np.asarray(saved["P2_bin"],    dtype=np.float64) if "P2_bin"    in saved else None
        self.gamma_bin = np.asarray(saved["gamma_bin"], dtype=np.float64) if "gamma_bin" in saved else None

        # Build C_inst from loaded sigma_Omega_bin  [FIX-4]
        self.C_inst = np.diag(self.sOm_bin ** 2)
        return saved

    # -------------------------
    # Population covariance  [FIX-1,2,3,6,7]
    # -------------------------
    def estimate_population_covariance(
        self,
        a_ref: float,
        *,
        # Alpha at injection truth: provide callable OR scalar
        alpha_ref_callable: Optional[Callable] = None,
        alpha_ref_scalar: Optional[float] = None,
        # SimulatedSignal settings objects
        freq_settings: Any,
        det_settings: Any,
        td_settings: Any,
        welch_settings: Any,
        pop_settings: Any,
        # Override MC parameters (fall back to EstimatorConfig defaults)
        k_pop: Optional[int] = None,
        seed_base: Optional[int] = None,
        jitter: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ) -> np.ndarray:
        """
        Estimate C_pop via K Monte Carlo realisations of Omega_MODEL at the
        injection truth parameters, then store it internally so that
        run_inference() automatically uses the full-covariance likelihood.

        Parameters
        ----------
        a_ref : float
            True value of the ppE exponent used in the injection.
        alpha_ref_callable : callable, optional
            Function (dataset) -> array(N_sys,) returning per-event alpha at
            the injection truth.  Takes precedence over alpha_ref_scalar.
        alpha_ref_scalar : float, optional
            Scalar alpha at the injection truth (used when callable is None).
        freq_settings, det_settings, td_settings, welch_settings, pop_settings :
            Settings dataclasses from SimulatedSignal used to build each
            Monte Carlo realisation of Omega_MODEL.
        k_pop : int, optional
            Number of MC realisations (defaults to cfg.k_pop = 50).
        seed_base : int, optional
            Base seed (defaults to cfg.pop_seed_base = 90000).
        jitter : float, optional
            Diagonal jitter added to C_pop (defaults to cfg.cpop_jitter).
        n_jobs : int, optional
            Parallel workers (defaults to auto-detected self._cpop_jobs).

        Returns
        -------
        C_pop : np.ndarray, shape (n_bins, n_bins)
            Ledoit-Wolf–shrunk population covariance matrix.
        """
        if self.f_bin is None:
            raise RuntimeError("Call load() before estimate_population_covariance().")

        k_pop_eff    = int(k_pop    if k_pop    is not None else self.cfg.k_pop)
        seed_eff     = int(seed_base if seed_base is not None else self.cfg.pop_seed_base)
        jitter_eff   = float(jitter  if jitter   is not None else self.cfg.cpop_jitter)
        n_jobs_eff   = int(n_jobs   if n_jobs   is not None else self._cpop_jobs)

        C_pop, omega_samples = estimate_C_pop(
            f_bin=self.f_bin,
            a_ref=float(a_ref),
            alpha_ref_callable=alpha_ref_callable,
            alpha_ref_scalar=alpha_ref_scalar,
            freq_settings=freq_settings,
            det_settings=det_settings,
            td_settings=td_settings,
            welch_settings=welch_settings,
            pop_settings=pop_settings,
            k_pop=k_pop_eff,
            seed_base=seed_eff,
            jitter=jitter_eff,
            n_jobs=n_jobs_eff,
        )

        self.C_pop = C_pop
        self._cpop_omega_samples = omega_samples

        # Release raw samples to recover RAM  [MEM-3]
        gc.collect()

        return C_pop

    # -------------------------
    # Internal helpers
    # -------------------------
    @staticmethod
    def _meta_get(meta: Any, *path: str, default: Any = None) -> Any:
        """Navigate a nested dict from saved metadata, returning default if any key is missing."""
        cur = meta
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def set_lambda_true_values(self, true_values: Dict[str, float]) -> None:
        """
        Register the injection truth values for free Lambda parameters so they
        appear as orange lines/squares on the corner plot (via plot_corner_with_truths).

        Only keys that match a name in lambda_param_specs are stored; the rest
        are silently ignored so you can safely pass the full Lambda dict.

        Example
        -------
        est.set_lambda_true_values({"alpha": 2.5, "beta": 1.0, "rate": 15.0})
        """
        valid = {s.name for s in self.lambda_param_specs}
        self.lambda_true_values = {k: float(v) for k, v in true_values.items() if k in valid}

    def _build_specs(self) -> List[ParamGridSpec]:
        specs: List[ParamGridSpec] = [
            ParamGridSpec(
                name="a",
                min=self.cfg.a_min,
                max=self.cfg.a_max,
                n=self.cfg.na,
                transform="linear",
                latex_label=r"$a$",
            )
        ]

        # alpha axis: either scalar alpha_ppE OR parameters of alpha_event_model
        if self.alpha_event_model is None:

            scale = str(getattr(self.cfg, "alpha_scale", "log10")).lower().strip()
            if scale not in ("log10", "linear"):
                raise ValueError(f"alpha_scale must be 'log10' or 'linear', got {scale!r}")

            if scale == "log10":
                specs.append(
                    ParamGridSpec(
                        name="alpha_ppE",
                        min=self.cfg.logalpha_min,
                        max=self.cfg.logalpha_max,
                        n=self.cfg.nlogalpha,
                        transform="log10",
                        latex_label=r"$\alpha_{\rm ppE}$",
                    )
                )
            else:
                specs.append(
                    ParamGridSpec(
                        name="alpha_ppE",
                        min=self.cfg.alpha_min,
                        max=self.cfg.alpha_max,
                        n=self.cfg.nalpha,
                        transform="linear",
                        latex_label=r"$\alpha_{\rm ppE}$",
                    )
                )
        else:
            specs.extend(self.alpha_param_specs)

        specs.extend(self.G_param_specs)

        # Free Lambda hyperparameters appended last (after ppE and G params)
        for lspec in self.lambda_param_specs:
            specs.append(
                ParamGridSpec(
                    name=lspec.name,
                    min=lspec.min,
                    max=lspec.max,
                    n=lspec.n,
                    transform=lspec.transform,
                    latex_label=lspec.latex_label,
                )
            )

        return specs

    def _cache_config(self) -> Dict[str, Any]:
        if self.saved is None or self.f_bin is None:
            raise RuntimeError("Call load() first.")

        specs = self._build_specs()
        axes = [_axis_values_for_spec(s) for s in specs]

        # Try to infer simulation settings from saved meta if legacy keys are missing.
        meta = self.saved.get("meta", None)

        FMIN  = float(self.saved["FMIN"])       if "FMIN"       in self.saved else float(self._meta_get(meta, "freq", "fmin",              default=10.0))
        FMAX  = float(self.saved["FMAX"])       if "FMAX"       in self.saved else float(self._meta_get(meta, "freq", "fmax",              default=2048.0))
        N_FREQ= int(  self.saved["N_FREQ"])     if "N_FREQ"     in self.saved else int(  self._meta_get(meta, "freq", "n_freq_eff",        default=800))
        N_SYS = int(  self.saved["N_SYS"])      if "N_SYS"      in self.saved else int(  self._meta_get(meta, "freq", "n_proposal_samples",default=10000))
        REBIN = float(self.saved["REBIN_DF_HZ"])if "REBIN_DF_HZ"in self.saved else float(self._meta_get(meta, "freq", "rebin_df_hz",       default=np.nan))

        cfg = {
            "data": {
                "FMIN": FMIN,
                "FMAX": FMAX,
                "N_FREQ": N_FREQ,
                "N_SYS": N_SYS,
                "REBIN_DF_HZ": REBIN,
                "f_bin_len": int(len(self.f_bin)),
                "f_bin_sha16": hashlib.sha256(np.asarray(self.f_bin, dtype=np.float64).tobytes()).hexdigest()[:16],
            },
            "specs": [asdict(s) for s in specs],
            "axes_sha16": hashlib.sha256(b"".join([ax.tobytes() for ax in axes])).hexdigest()[:16],
            "alpha_id": self.alpha_id,
            "G_id": self.G_id,
            "corr_module": "ppEMG",
            "compute_knobs": dict(self.compute_knobs),
        }
        return cfg

    def _cache_paths(self, cfg: Dict[str, Any]) -> Tuple[str, str, str]:
        h = _stable_hash_of_dict(cfg)
        npz_path = os.path.join(self.cfg.grid_cache_dir, f"model_grid_{h}.npz")
        meta_path = os.path.join(self.cfg.grid_cache_dir, f"model_grid_{h}.json")
        return h, npz_path, meta_path

    def _cache_is_compatible(self, npz_path: str, f_bin: np.ndarray, axes: List[np.ndarray]) -> bool:
        if not os.path.exists(npz_path):
            return False
        try:
            d = np.load(npz_path, allow_pickle=False)
            if "model_grid" not in d.files or "f_bin" not in d.files or "axes_sha16" not in d.files:
                return False
            if not np.array_equal(d["f_bin"], f_bin):
                return False
            sha = str(d["axes_sha16"])
            sha_now = hashlib.sha256(b"".join([ax.tobytes() for ax in axes])).hexdigest()[:16]
            return sha == sha_now
        except Exception:
            return False

    def _save_grid_cache(
        self,
        npz_path: str,
        meta_path: str,
        cfg: Dict[str, Any],
        model_grid: np.ndarray,
        f_bin: np.ndarray,
        axes: List[np.ndarray],
    ) -> None:
        axes_sha16 = hashlib.sha256(b"".join([ax.tobytes() for ax in axes])).hexdigest()[:16]
        np.savez_compressed(
            npz_path,
            model_grid=np.asarray(model_grid, dtype=np.float64),
            f_bin=np.asarray(f_bin, dtype=np.float64),
            axes_sha16=np.array(axes_sha16),
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

    def _load_grid_cache(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        d = np.load(npz_path, allow_pickle=False)
        return d["model_grid"], d["f_bin"]

    # -------------------------
    # Prepare model
    # -------------------------
    def prepare_model(
        self,
        *,
        Lambda: Optional[Dict[str, float]] = None,
        binary_type: str = "BBH",
        waveform_approximant: str = "IMRPhenomD",
        inspiral_only: bool = True,
        disable_inspiral_cutoff: bool = True,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
    ) -> Tuple[InterpModelNDWithCache, str]:
        if self.saved is None or self.f_bin is None:
            self.load()

        self.specs = self._build_specs()
        self.axes = [_axis_values_for_spec(s) for s in self.specs]

        cfg = self._cache_config()
        h, cache_npz, cache_meta = self._cache_paths(cfg)
        self.cache_hash = h

        if self._cache_is_compatible(cache_npz, self.f_bin, self.axes):
            model_grid, _ = self._load_grid_cache(cache_npz)
        else:
            meta = self.saved.get("meta", None)

            FMIN  = float(self.saved["FMIN"])       if "FMIN"       in self.saved else float(self._meta_get(meta, "freq", "fmin",              default=10.0))
            FMAX  = float(self.saved["FMAX"])       if "FMAX"       in self.saved else float(self._meta_get(meta, "freq", "fmax",              default=2048.0))
            N_FREQ= int(  self.saved["N_FREQ"])     if "N_FREQ"     in self.saved else int(  self._meta_get(meta, "freq", "n_freq_eff",        default=800))
            N_SYS = int(  self.saved["N_SYS"])      if "N_SYS"      in self.saved else int(  self._meta_get(meta, "freq", "n_proposal_samples",default=10000))

            freqs_pop, omega_fid, dataset, Lambda_eff, ctx, probabilities = precompute_population_objects(
                FMIN, FMAX, N_FREQ, N_SYS,
                binary_type=binary_type,
                waveform_approximant=waveform_approximant,
                inspiral_only=inspiral_only,
                disable_inspiral_cutoff=disable_inspiral_cutoff,
                sampling_frequency=sampling_frequency,
                duration=duration,
                Lambda=Lambda,
            )

            alpha_param_names = [s.name for s in self.alpha_param_specs] if self.alpha_event_model is not None else []
            G_param_names = [s.name for s in self.G_param_specs] if self.G_param_specs else []
            lambda_param_names_eff = [s.name for s in self.lambda_param_specs]
            n_jobs = int(self.cfg.n_jobs) if self.cfg.n_jobs is not None else -1

            # pop_cfg carries the fixed settings needed by _grid_point_eval_with_pop
            # so each Lambda grid point can re-draw its own population independently.
            pop_cfg_eff = dict(
                FMIN=FMIN, FMAX=FMAX, N_FREQ=N_FREQ, N_SYS=N_SYS,
                binary_type=binary_type,
                waveform_approximant=waveform_approximant,
                inspiral_only=inspiral_only,
                disable_inspiral_cutoff=disable_inspiral_cutoff,
                sampling_frequency=sampling_frequency,
                duration=duration,
            )

            model_grid = build_model_grid_nd_parallel(
                self.f_bin,
                freqs_pop,
                omega_fid,
                dataset,
                Lambda_eff,
                ctx,
                probabilities,
                specs=self.specs,
                axes=self.axes,
                alpha_event_model=self.alpha_event_model,
                alpha_param_names=alpha_param_names,
                G_event_weight=self.G_event_weight,
                G_param_names=G_param_names,
                n_jobs=n_jobs,
                compute_knobs=dict(self.compute_knobs),
                lambda_param_names=lambda_param_names_eff,
                base_Lambda=Lambda_eff,
                pop_cfg=pop_cfg_eff,
            )
            self._save_grid_cache(cache_npz, cache_meta, cfg, model_grid, self.f_bin, self.axes)

        self.model_grid = np.asarray(model_grid, dtype=np.float64)
        self.interp = InterpModelNDWithCache(self.specs, self.axes, self.model_grid, subdiv=int(self.cfg.cache_subdiv))
        return self.interp, h

    # -------------------------
    # Inference
    # -------------------------
    def run_inference(
        self,
        *,
        label: Optional[str] = None,
        sampler: str = "dynesty",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        include_params: Optional[List[str]] = None,
        likelihood_mode: Optional[str] = None,
        npool: Optional[int] = None,
        checkpoint: bool = False,
        resume: bool = False,
    ) -> bilby.core.result.Result:
        """
        Run Bilby nested sampling inference.

        Parameters
        ----------
        label : str, optional
            Output label for Bilby (default: auto from cache hash).
        sampler : str
            Sampler name (default "dynesty").
        sampler_kwargs : dict, optional
            Extra kwargs forwarded to bilby.run_sampler (default: rwalk/multi).
        clean : bool
            Remove sampler checkpoint files after run.
        include_params : list of str, optional
            Subset of parameters to infer (must include 'a').
        likelihood_mode : str, optional
            Override cfg.likelihood_mode.
            "diagonal"  — Gaussian diagonal on sigma_inst (legacy).
            "full_cov"  — C_eff = k²·C_inst + C_pop  [FIX-4].
            If "full_cov" is requested but C_pop has not been estimated,
            falls back to "diagonal" with a warning.
        npool : int, optional
            Override auto-detected sampler_pool for bilby npool argument.
        """
        if self.interp is None or self.cache_hash is None or self.specs is None:
            self.prepare_model()
        if self.f_bin is None or self.Om_bin is None or self.sOm_bin is None:
            raise RuntimeError("Data not loaded; call load() first.")

        param_names = [s.name for s in self.specs]
        if include_params is not None:
            param_names = [p for p in param_names if p in include_params]
            if "a" not in param_names:
                raise ValueError("include_params must include at least ['a', ...].")
            if (self.alpha_event_model is None) and ("alpha_ppE" not in param_names):
                raise ValueError("Scalar-alpha mode requires including 'alpha_ppE'.")

        # Resolve likelihood mode  [FIX-4]
        mode = str(likelihood_mode if likelihood_mode is not None else self.cfg.likelihood_mode).lower()
        if mode == "full_cov" and self.C_pop is None:
            import warnings
            warnings.warn(
                "likelihood_mode='full_cov' requested but C_pop has not been estimated. "
                "Call estimate_population_covariance() first, or set likelihood_mode='diagonal'. "
                "Falling back to diagonal likelihood.",
                RuntimeWarning,
            )
            mode = "diagonal"

        # Ensure C_inst is available
        if self.C_inst is None:
            self.C_inst = np.diag(np.asarray(self.sOm_bin, dtype=np.float64) ** 2)

        # Build likelihood
        if mode == "optimal":
            likelihood = OmegaOptimalFilterLikelihood(
                f_bin=self.f_bin,
                omega_hat=self.Om_bin,
                sigma_inst=self.sOm_bin,
                P1_bin=self.P1_bin,
                P2_bin=self.P2_bin,
                gamma_bin=self.gamma_bin,
                evaluator=self.interp,
                param_names=param_names,
            )
        elif mode == "full_cov":
            likelihood = OmegaGaussianLikelihoodFullCov(
                f_bin=self.f_bin,
                omega_hat=self.Om_bin,
                C_inst=self.C_inst,
                C_pop=self.C_pop,
                model_interp=self.interp,
                param_names=param_names,
                jitter=float(self.cfg.cpop_jitter),
            )
        else:
            # diagonal (legacy)
            likelihood = OmegaGaussianLikelihoodDiag(
                f_bin=self.f_bin,
                omega_hat=self.Om_bin,
                sigma_inst=self.sOm_bin,
                model_interp=self.interp,
                param_names=param_names,
            )

        # Build priors
        priors = bilby.core.prior.PriorDict()
        spec_map = {s.name: s for s in self.specs}

        for p in param_names:
            s = spec_map[p]
            latex = s.latex_label if s.latex_label is not None else p
            if s.transform == "linear":
                priors[p] = bilby.core.prior.Uniform(
                    float(s.min), float(s.max), name=p, latex_label=latex)
            elif s.transform == "log10":
                priors[p] = bilby.core.prior.LogUniform(
                    10.0 ** float(s.min), 10.0 ** float(s.max),
                    name=p, latex_label=latex)
            else:
                raise ValueError(f"Unknown transform {s.transform} for prior {p}")

        # [FIX-4] k_sigma ~ 1 with full_cov -> narrow prior (0.1, 10)
        # With diagonal likelihood, k_sigma must absorb all uncertainty -> (0.1, 300)
        if mode == "full_cov":
            priors["k_sigma"] = bilby.core.prior.LogUniform(
                0.01, 300.0, name="k_sigma", latex_label=r"$k_\sigma$")
        else:
            priors["k_sigma"] = bilby.core.prior.LogUniform(
                0.01, 300.0, name="k_sigma", latex_label=r"$k_\sigma$")

        if sampler_kwargs is None:
            sampler_kwargs = {"sample": "rwalk", "bound": "multi"}

        if label is None:
            label = f"infer_hashcache_{self.cache_hash}"

        # Resolve npool  [MEM-2]
        npool_eff = int(npool) if npool is not None else self._sampler_pool

        # bilby 3.x: passar kwargs diretamente via **unpacking
        _extra_kwargs = dict(sampler_kwargs)
        _extra_kwargs.pop("pool", None)  # pool gerenciado pelo bilby via npool

        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=sampler,
            npoints=int(self.cfg.npoints),
            walks=int(self.cfg.walks),
            dlogz=float(self.cfg.dlogz),
            npool=npool_eff,
            outdir=self.cfg.outdir,
            label=label,
            check_point=bool(checkpoint),
            resume=bool(resume),
            clean=bool(clean),
            check_point_plot=False,
            **_extra_kwargs,
        )
        return result

    # -------------------------
    # Direct inference (no grid)
    # -------------------------
    def run_inference_direct(
        self,
        *,
        param_specs: List["ParamGridSpec"],
        base_Lambda: Dict[str, float],
        pop_cfg: Dict[str, Any],
        label: Optional[str] = None,
        sampler: str = "dynesty",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        likelihood_mode: Optional[str] = None,
        npool: Optional[int] = None,
        lambda_tol_decimals: int = 4,
        lru_size: int = 2,
        checkpoint: bool = False,
        resume: bool = False,
    ) -> "bilby.core.result.Result":
        """
        Run inference WITHOUT a precomputed grid.

        omega_full is computed directly at each MCMC step via
        omega_full_from_corrections.  A lightweight LRU cache reuses
        (h2_cache, probabilities) across steps that share the same Lambda point
        (up to lambda_tol_decimals decimal places), so that changes only in
        (a, alpha) do NOT trigger a population redraw.

        Parameters
        ----------
        param_specs : list of ParamGridSpec
            One spec per free parameter.  Defines prior bounds and transform
            (no 'n' needed here — grid size is ignored).
            Must include specs for 'a' and either 'alpha_ppE' or all
            alpha_event_model parameters, plus any free Lambda parameters.
        base_Lambda : dict
            Full Lambda dict with fixed values for ALL population hyperparameters.
            Free Lambda parameters will be overridden at each MCMC step.
        pop_cfg : dict
            Population rebuild settings: FMIN, FMAX, N_FREQ, N_SYS,
            binary_type, waveform_approximant, inspiral_only,
            disable_inspiral_cutoff, sampling_frequency, duration.
        label : str, optional
            Bilby output label.
        likelihood_mode : str, optional
            "diagonal" or "full_cov" (default: cfg.likelihood_mode).
        npool : int, optional
            Override sampler npool (default: auto).
        lambda_tol_decimals : int
            Decimal places for Lambda cache key rounding (default 4).
        lru_size : int
            LRU cache size for population objects (default 2).
        """
        if self.f_bin is None or self.Om_bin is None or self.sOm_bin is None:
            raise RuntimeError("Data not loaded; call load() first.")

        # Resolve likelihood mode
        mode = str(likelihood_mode if likelihood_mode is not None else self.cfg.likelihood_mode).lower()
        if mode == "optimal" and (self.P1_bin is None or self.P2_bin is None or self.gamma_bin is None):
            import warnings
            warnings.warn(
                "likelihood_mode='optimal' requested mas P1_bin/P2_bin/gamma_bin "
                "are not available in the .npz. Falling back to 'full_cov'.",
                RuntimeWarning,
            )
            mode = "full_cov"
        if mode == "full_cov" and self.C_pop is None:
            import warnings
            warnings.warn(
                "likelihood_mode='full_cov' requested but C_pop has not been estimated. "
                "Falling back to diagonal likelihood.",
                RuntimeWarning,
            )
            mode = "diagonal"

        if self.C_inst is None:
            self.C_inst = np.diag(np.asarray(self.sOm_bin, dtype=np.float64) ** 2)

        param_names = []
        for i, s in enumerate(param_specs):
            if not isinstance(s, ParamGridSpec):
                raise TypeError(
                    f"param_specs[{i}] must be a ParamGridSpec, "
                    f"got {type(s).__name__!r} (value={s!r}). "
                    "Use ParamGridSpec for all entries in param_specs_direct, "
                    "including those derived from LambdaParamSpec."
                )
            param_names.append(s.name)
        lambda_param_names = [s.name for s in self.lambda_param_specs]
        alpha_param_names  = [s.name for s in self.alpha_param_specs] if self.alpha_event_model is not None else []
        G_param_names      = [s.name for s in self.G_param_specs]

        # Draw one base population for the evaluator's default state
        freqs_pop, omega_fid, dataset, Lambda_eff, ctx, probabilities = \
            precompute_population_objects(
                float(pop_cfg["FMIN"]),
                float(pop_cfg["FMAX"]),
                int(pop_cfg["N_FREQ"]),
                int(pop_cfg["N_SYS"]),
                binary_type=str(pop_cfg["binary_type"]),
                waveform_approximant=str(pop_cfg["waveform_approximant"]),
                inspiral_only=bool(pop_cfg["inspiral_only"]),
                disable_inspiral_cutoff=bool(pop_cfg["disable_inspiral_cutoff"]),
                sampling_frequency=float(pop_cfg["sampling_frequency"]),
                duration=float(pop_cfg["duration"]),
                Lambda=base_Lambda,
            )

        evaluator = DirectModelEvaluator(
            f_bin=self.f_bin,
            freqs_pop=freqs_pop,
            omega_fid=omega_fid,
            dataset=dataset,
            base_Lambda=base_Lambda,
            ctx=ctx,
            probabilities=probabilities,
            lambda_param_names=lambda_param_names,
            alpha_param_names=alpha_param_names,
            G_param_names=G_param_names,
            alpha_event_model=self.alpha_event_model,
            G_event_weight=self.G_event_weight,
            compute_knobs=dict(self.compute_knobs),
            pop_cfg=pop_cfg,
            lambda_tol_decimals=lambda_tol_decimals,
            lru_size=lru_size,
        )

        # Build likelihood
        if mode == "optimal":
            likelihood = OmegaOptimalFilterLikelihood(
                f_bin=self.f_bin,
                omega_hat=self.Om_bin,
                sigma_inst=self.sOm_bin,
                P1_bin=self.P1_bin,
                P2_bin=self.P2_bin,
                gamma_bin=self.gamma_bin,
                evaluator=evaluator,
                param_names=param_names,
            )
        elif mode == "full_cov":
            likelihood = OmegaGaussianLikelihoodDirectFullCov(
                f_bin=self.f_bin,
                omega_hat=self.Om_bin,
                C_inst=self.C_inst,
                C_pop=self.C_pop,
                evaluator=evaluator,
                param_names=param_names,
                jitter=float(self.cfg.cpop_jitter),
            )
        else:
            likelihood = OmegaGaussianLikelihoodDirectDiag(
                f_bin=self.f_bin,
                omega_hat=self.Om_bin,
                sigma_inst=self.sOm_bin,
                evaluator=evaluator,
                param_names=param_names,
            )

        # Build priors from param_specs
        priors = bilby.core.prior.PriorDict()
        for s in param_specs:
            latex = s.latex_label if s.latex_label is not None else s.name
            if s.transform == "linear":
                priors[s.name] = bilby.core.prior.Uniform(
                    float(s.min), float(s.max), name=s.name, latex_label=latex)
            elif s.transform == "log10":
                priors[s.name] = bilby.core.prior.LogUniform(
                    10.0 ** float(s.min), 10.0 ** float(s.max),
                    name=s.name, latex_label=latex)
            else:
                raise ValueError(f"Unknown transform {s.transform!r} for prior {s.name}")

        if mode == "full_cov":
            priors["k_sigma"] = bilby.core.prior.LogUniform(
                0.1, 10.0, name="k_sigma", latex_label=r"$k_\sigma$")
        else:
            priors["k_sigma"] = bilby.core.prior.LogUniform(
                0.1, 300.0, name="k_sigma", latex_label=r"$k_\sigma$")

        if sampler_kwargs is None:
            sampler_kwargs = {"sample": "rwalk", "bound": "multi"}
        # Safe copy so we do not modify the caller's original dictionary
        sampler_kwargs = dict(sampler_kwargs)
        # Point dynesty's internal cache (__pystache__) at the output directory
        sampler_kwargs.setdefault("cache_dir", os.path.join(self.cfg.outdir, "dynesty_cache"))
      
        if label is None:
            label = f"infer_direct_{hashlib.sha256(str(param_names).encode()).hexdigest()[:8]}"

        npool_eff = int(npool) if npool is not None else self._sampler_pool

        # Cria pool explicitamente — bilby 3.x ignora npool silenciosamente
        # para dynesty quando sample="rwalk". Passar o pool via sampler_kwargs
        # makes sure parallelization is enabled.
        import multiprocessing
        _pool = None
        # bilby 3.x: sampler_kwargs is ignored — pass kwargs directly
        # Nota: NÃO criar pool manualmente — bilby/dynesty gerencia o pool
        # internally via npool. An external pool breaks serialization
        # (pickle) dos priors nos workers filhos.
        _extra_kwargs = dict(sampler_kwargs)
        # Remove pool se foi passado externamente — deixa bilby gerenciar
        _extra_kwargs.pop("pool", None)

        print(f"[run_inference_direct] npool={npool_eff} — pool gerenciado pelo bilby.", flush=True)

        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=sampler,
            npoints=int(self.cfg.npoints),
            walks=int(self.cfg.walks),
            dlogz=float(self.cfg.dlogz),
            npool=npool_eff,
            outdir=self.cfg.outdir,
            label=label,
            check_point=bool(checkpoint),
            resume=bool(resume),
            clean=bool(clean),
            check_point_plot=False,
            **_extra_kwargs,
        )

        # Save posterior samples as CSV for future ChainConsumer use
        try:
            samples_path = os.path.join(self.cfg.outdir, f"{label}_samples.txt")
            result.posterior.to_csv(samples_path, index=False)
            print(f"[run_inference_direct] Samples saved to {samples_path}", flush=True)
        except Exception as exc:
            import warnings
            warnings.warn(f"[run_inference_direct] Could not save samples CSV: {exc}", RuntimeWarning)

        return result
