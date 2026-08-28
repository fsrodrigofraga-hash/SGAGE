from __future__ import annotations

# ============================================================
# ParametersEstimator.direct — grid-FREE evaluation: omega_full is
# recomputed at every MCMC step
# ============================================================
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import bilby

import ppEMG as corr

from .config import EventVectorFn
from .models import omega_full_from_corrections
from .population import _redshift_normalisation


# Direct model evaluator (no grid — computes omega_full at each MCMC step)
# ============================================================
class DirectModelEvaluator:
    """
    Computes omega_model(f_bin; theta) directly via omega_full_from_corrections.

    Strategy-1 two-level cache:
      - h2_cache and f_peak_obs are computed ONCE at init from the base population
        and kept fixed for the entire run. They depend only on dataset
        (masses/distances), which does not change with Lambda.
      - probabilities are the only quantity that changes with Lambda. On a cache
        miss only calculate_probabilities() is called (~1 ms, pure NumPy) instead
        of a full population redraw (~2-5 s with waveform calls).
      - omega_fid is importance-reweighted to the proposed Lambda:
            omega_fid_eff = omega_fid · (rate/rate_base) · [V(Λ)/V(Λ_base)]
                            · [Σ w(Λ)h² / Σ w(Λ_base)h²],
        with popstock-consistent weights w = p(θ|Λ)/pdraw (dataset["pdraw"])
        and V(Λ) = normalisation do modelo de redshift (fator Rate_norm do
        popstock). Exato em Λ = Λ_base; consistente com pop.calculate_omega_gw.

    Cache key: tuple of rounded Lambda free-parameter values (tolerance = lambda_tol).
    """

    def __init__(
        self,
        f_bin: np.ndarray,
        freqs_pop: np.ndarray,
        omega_fid: np.ndarray,
        dataset: Dict[str, Any],
        base_Lambda: Dict[str, float],
        ctx: Any,
        probabilities: np.ndarray,
        *,
        lambda_param_names: List[str],
        alpha_param_names: List[str],
        G_param_names: List[str],
        alpha_event_model: Optional[EventVectorFn],
        G_event_weight: Optional[EventVectorFn],
        compute_knobs: Dict[str, Any],
        pop_cfg: Optional[Dict[str, Any]] = None,
        lambda_tol_decimals: int = 4,
        lru_size: int = 2,
    ):
        self.f_bin               = np.asarray(f_bin,     dtype=np.float64)
        self.freqs_pop           = np.asarray(freqs_pop, dtype=np.float64)
        self.omega_fid           = np.asarray(omega_fid, dtype=np.float64)
        self.dataset             = dataset
        self.base_Lambda         = dict(base_Lambda)
        self.ctx                 = ctx
        self._base_probabilities = np.asarray(probabilities, dtype=np.float64)

        self.lambda_param_names = list(lambda_param_names)
        self.alpha_param_names  = list(alpha_param_names)
        self.G_param_names      = list(G_param_names)
        self.alpha_event_model  = alpha_event_model
        self.G_event_weight     = G_event_weight
        self.compute_knobs      = dict(compute_knobs)
        self.pop_cfg            = dict(pop_cfg) if pop_cfg is not None else {}
        self.lambda_tol_dec     = int(lambda_tol_decimals)
        self.lru_size           = int(lru_size)

        # Strategy 1: pre-compute h2_cache and f_peak_obs once at init.
        # These are independent of Lambda (depend only on masses/distances in dataset).
        knobs = self.compute_knobs
        print("[DirectModelEvaluator] Pre-computing h2_cache (once)...", flush=True)
        self._h2_cache, self._f_peak_obs = corr.compute_h2_cache_and_fpeak_parallel(
            dataset=self.dataset,
            idx=self.ctx.idx_freq,
            wf_cfg=self.ctx.wf_cfg,
            frequencies=self.ctx.frequencies,
            nproc=knobs.get("nproc", 1),
            chunksize=int(knobs.get("chunksize", 20)),
            fast_binning=bool(knobs.get("fast_binning", True)),
            max_bins=int(knobs.get("max_bins", 100)),
            use_frequency_warp=bool(knobs.get("use_frequency_warp", True)),
        )
        print("[DirectModelEvaluator] h2_cache ready.", flush=True)

        # den_base(f) = Σᵢ wᵢ(Λ_base)·h²ᵢ(f)  — denominador fixo para reweighting
        # with popstock-consistent weights w = p/pdraw (if the dataset has
        # "pdraw", w(Lambda_base) ~ 1 uniformly, since the samples were drawn
        # from Lambda_base).
        self._w_base = corr.weights_from_probabilities(
            self.dataset, self._base_probabilities
        )
        self._den_base = np.sum(self._w_base[:, None] * self._h2_cache, axis=0)
        # rate_base para reweighting da taxa de merger
        self._rate_base = float(base_Lambda.get("rate", 1.0))

        # V(Λ_base): fator de normalizacao do modelo de redshift, que o
        # popstock multiplica em Omega (Rate_norm ∝ rate·V). Necessario no
        # emulator when gamma/kappa/z_peak are free.
        self._norm_base = _redshift_normalisation(
            getattr(self.ctx, "models", {}), self.base_Lambda
        )
        if self._norm_base is None or not np.isfinite(self._norm_base) or self._norm_base <= 0:
            self._norm_base = None
            print("[DirectModelEvaluator] WARNING: normalisation() of the "
                  "redshift model is unavailable — V(Lambda) factor disabled.",
                  flush=True)

        # LRU cache: key -> probabilities array
        self._prob_cache: OrderedDict = OrderedDict()
        # LRU cache: key -> (new den array, new norm float|None)
        self._den_cache:  OrderedDict = OrderedDict()

    def _lambda_key(self, theta: Dict[str, float]) -> Tuple[float, ...]:
        """Round-based cache key for Lambda free parameters."""
        return tuple(
            round(float(theta[k]), self.lambda_tol_dec)
            for k in self.lambda_param_names
        )

    def _get_probabilities_for_lambda(self, theta: Dict[str, float]) -> np.ndarray:
        """
        Return probabilities for the Lambda values in theta.
        On a cache miss, calls calculate_probabilities() (~1 ms, pure NumPy)
        instead of a full population redraw (~2-5 s). h2_cache stays fixed.
        """
        if not self.lambda_param_names:
            return self._base_probabilities

        key = self._lambda_key(theta)

        if key in self._prob_cache:
            self._prob_cache.move_to_end(key)
            return self._prob_cache[key]

        Lambda_here = dict(self.base_Lambda)
        for k in self.lambda_param_names:
            Lambda_here[k] = float(theta[k])

        probs = np.asarray(
            corr.calculate_probabilities(self.dataset, Lambda_here, self.ctx.models),
            dtype=np.float64,
        )

        if len(self._prob_cache) >= self.lru_size:
            self._prob_cache.popitem(last=False)
        self._prob_cache[key] = probs
        return probs

    def model(self, theta: Dict[str, float]) -> np.ndarray:
        """
        Compute omega_model interpolated onto f_bin for the given parameter point theta.
        theta must contain: 'a', all alpha_param_names or 'alpha_ppE', all lambda_param_names.
        """
        probs = self._get_probabilities_for_lambda(theta)

        a            = float(theta.get("a", self.base_Lambda.get("a", 0.0)))
        alpha_params = {k: float(theta[k]) for k in self.alpha_param_names}
        G_params     = {k: float(theta[k]) for k in self.G_param_names}

        if self.alpha_event_model is None:
            alpha_ppE         = float(theta["alpha_ppE"])
            alpha_event_model = None
        else:
            alpha_ppE         = None
            alpha_event_model = self.alpha_event_model

        # Build Lambda for this point (needed by omega_full_from_corrections)
        Lambda_here = dict(self.base_Lambda)
        for k in self.lambda_param_names:
            Lambda_here[k] = float(theta[k])

        # Importance reweighting de omega_fid para o Lambda proposto:
        #   omega_fid_eff = omega_fid · (rate/rate_base) · [V(Λ)/V(Λ_base)]
        #                   · [Σ w(Λ)h² / Σ w(Λ_base)h²]
        # com w = p/pdraw (popstock-consistente). Exato em Λ = Λ_base.
        rate_novo  = float(Lambda_here.get("rate", self._rate_base))
        rate_ratio = rate_novo / self._rate_base if self._rate_base > 0 else 1.0

        key = self._lambda_key(theta)
        if key in self._den_cache:
            self._den_cache.move_to_end(key)
            den_novo, norm_novo = self._den_cache[key]
        else:
            w_novo   = corr.weights_from_probabilities(self.dataset, probs)
            den_novo = np.sum(w_novo[:, None] * self._h2_cache, axis=0)
            norm_novo = (
                _redshift_normalisation(getattr(self.ctx, "models", {}), Lambda_here)
                if self._norm_base is not None else None
            )
            if len(self._den_cache) >= self.lru_size:
                self._den_cache.popitem(last=False)
            self._den_cache[key] = (den_novo, norm_novo)

        norm_ratio = 1.0
        if (self._norm_base is not None and norm_novo is not None
                and np.isfinite(norm_novo) and norm_novo > 0):
            norm_ratio = norm_novo / self._norm_base

        den_ratio = np.where(
            (self._den_base > 0) & (den_novo > 0),
            den_novo / self._den_base,
            1.0,
        )
        omega_fid_eff = self.omega_fid * rate_ratio * norm_ratio * den_ratio

        omega_full = omega_full_from_corrections(
            self.freqs_pop, omega_fid_eff, self.dataset, Lambda_here,
            self.ctx, probs,
            a=a,
            alpha_ppE=alpha_ppE,
            alpha_event_model=alpha_event_model,
            alpha_params=alpha_params,
            G_event_weight=self.G_event_weight,
            G_params=G_params,
            # Pass pre-computed waveform cache — no LAL calls needed
            precomputed_h2_cache=self._h2_cache,
            precomputed_f_peak_obs=self._f_peak_obs,
            **self.compute_knobs,
        )
        return np.interp(self.f_bin, self.freqs_pop, omega_full,
                         left=0.0, right=0.0).astype(np.float64)


class OmegaGaussianLikelihoodDirectDiag(bilby.Likelihood):
    """Diagonal likelihood using DirectModelEvaluator (no grid)."""

    def __init__(
        self,
        f_bin: np.ndarray,
        omega_hat: np.ndarray,
        sigma_inst: np.ndarray,
        evaluator: DirectModelEvaluator,
        *,
        param_names: List[str],
    ):
        super().__init__(parameters={p: None for p in param_names + ["k_sigma"]})
        self.f          = np.asarray(f_bin,     dtype=np.float64)
        self.omega_hat  = np.asarray(omega_hat,  dtype=np.float64)
        self.sigma_inst = np.asarray(sigma_inst, dtype=np.float64)
        self.evaluator  = evaluator
        self.param_names = list(param_names)

    def log_likelihood(self) -> float:
        k     = float(self.parameters["k_sigma"])
        theta = {p: float(self.parameters[p]) for p in self.param_names}
        m     = self.evaluator.model(theta)
        r     = self.omega_hat - m
        s     = k * self.sigma_inst
        return float(-0.5 * np.sum((r / s) ** 2 + np.log(2.0 * np.pi * s ** 2)))


class OmegaGaussianLikelihoodDirectFullCov(bilby.Likelihood):
    """
    Full-covariance likelihood using DirectModelEvaluator (no grid).
    C_eff(k) = k²·C_inst + C_pop  [FIX-4]
    """
    _LRU_MAXSIZE = 4

    def __init__(
        self,
        f_bin: np.ndarray,
        omega_hat: np.ndarray,
        C_inst: np.ndarray,
        C_pop: np.ndarray,
        evaluator: DirectModelEvaluator,
        *,
        param_names: List[str],
        jitter: float = 1e-10,
        k_cache_decimals: int = 3,
    ):
        from scipy.linalg import cho_factor as _cf, cho_solve as _cs
        self._cho_factor = _cf
        self._cho_solve  = _cs

        super().__init__(parameters={p: None for p in param_names + ["k_sigma"]})
        self.f           = np.asarray(f_bin,    dtype=np.float64)
        self.omega_hat   = np.asarray(omega_hat, dtype=np.float64)
        self.C_inst      = np.asarray(C_inst,    dtype=np.float64)
        self.C_pop       = np.asarray(C_pop,     dtype=np.float64)
        self.evaluator   = evaluator
        self.param_names = list(param_names)
        self.jitter      = float(jitter)
        self.k_cache_decimals = int(k_cache_decimals)

        self._chol_cache: OrderedDict = OrderedDict()
        self._I = np.eye(int(self.omega_hat.size), dtype=np.float64)
        _diag_scale = max(
            float(np.mean(np.abs(np.diag(self.C_pop)))),
            float(np.mean(np.diag(self.C_inst) ** 2)),
            1e-300,
        )
        self._jitter_abs = self.jitter * _diag_scale

    def _get_cholesky(self, k: float):
        kq = float(np.round(k, self.k_cache_decimals))
        if kq in self._chol_cache:
            self._chol_cache.move_to_end(kq)
            return self._chol_cache[kq]
        C = (kq * kq) * self.C_inst + self.C_pop + self._I * self._jitter_abs
        try:
            c_and_lower = self._cho_factor(C)
        except np.linalg.LinAlgError:
            return None
        logdet = 2.0 * float(np.sum(np.log(np.abs(np.diag(c_and_lower[0])))))
        entry  = (c_and_lower, logdet)
        if len(self._chol_cache) >= self._LRU_MAXSIZE:
            self._chol_cache.popitem(last=False)
        self._chol_cache[kq] = entry
        return entry

    def log_likelihood(self) -> float:
        k     = float(self.parameters["k_sigma"])
        theta = {p: float(self.parameters[p]) for p in self.param_names}
        m     = self.evaluator.model(theta)
        d     = self.omega_hat - m
        cached = self._get_cholesky(k)
        if cached is None:
            return -np.inf
        c_and_lower, logdet = cached
        quad = float(np.dot(d, self._cho_solve(c_and_lower, d)))
        return -0.5 * (quad + logdet + d.size * np.log(2.0 * np.pi))
