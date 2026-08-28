from __future__ import annotations

# ============================================================
# ParametersEstimator.likelihoods — likelihoods on the interpolated
# grid (diagonal and full-covariance) plus the optimal filter
# ============================================================
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import bilby

from .interpolation import InterpModelNDWithCache


# Likelihood classes
# ============================================================
class OmegaGaussianLikelihoodDiag(bilby.Likelihood):
    """
    Diagonal Gaussian likelihood on sigma_inst.
    Legacy mode — use when C_pop is unavailable or TURBO_SKIP_CPOP=True.
    """
    def __init__(
        self,
        f_bin: np.ndarray,
        omega_hat: np.ndarray,
        sigma_inst: np.ndarray,
        model_interp: "InterpModelNDWithCache",
        *,
        param_names: List[str],
    ):
        params = {p: None for p in (param_names + ["k_sigma"])}
        super().__init__(parameters=params)
        self.f           = np.asarray(f_bin,    dtype=np.float64)
        self.omega_hat   = np.asarray(omega_hat,  dtype=np.float64)
        self.sigma_inst  = np.asarray(sigma_inst, dtype=np.float64)
        self.model_interp = model_interp
        self.param_names  = list(param_names)

    def log_likelihood(self) -> float:
        k     = float(self.parameters["k_sigma"])
        theta = {p: float(self.parameters[p]) for p in self.param_names}
        m     = np.asarray(self.model_interp.model(theta), dtype=np.float64)
        r     = self.omega_hat - m
        s     = k * self.sigma_inst
        return float(-0.5 * np.sum((r / s) ** 2 + np.log(2.0 * np.pi * s ** 2)))


class OmegaGaussianLikelihoodFullCov(bilby.Likelihood):
    """
    [FIX-4] Full-covariance Gaussian likelihood.

    C_eff(k) = k²·C_inst + C_pop

    k_sigma scales ONLY the instrumental covariance C_inst; C_pop enters
    with fixed weight = 1.  This means k_sigma ~ 1 when the instrument
    noise model is well calibrated (prior LogUniform(0.1, 10) is sufficient).

    Contrast with the incorrect formulation used previously:
        C_eff(k) = k² · (C_inst + C_pop)   ← wrong: k_sigma would absorb C_pop

    The Cholesky factorisation is cached in an LRU dict keyed by rounded k.
    Cache size is kept at 4 entries [MEM-4] (~7 MB each for n_bins ~ 960).
    """
    _LRU_MAXSIZE = 4  # [MEM-4]

    def __init__(
        self,
        f_bin: np.ndarray,
        omega_hat: np.ndarray,
        C_inst: np.ndarray,
        C_pop: np.ndarray,
        model_interp: "InterpModelNDWithCache",
        *,
        param_names: List[str],
        jitter: float = 1e-10,
        k_cache_decimals: int = 3,
    ):
        from scipy.linalg import cho_factor as _cf, cho_solve as _cs
        self._cho_factor = _cf
        self._cho_solve  = _cs

        params = {p: None for p in (param_names + ["k_sigma"])}
        super().__init__(parameters=params)

        self.f            = np.asarray(f_bin,    dtype=np.float64)
        self.omega_hat    = np.asarray(omega_hat, dtype=np.float64)
        self.C_inst       = np.asarray(C_inst,    dtype=np.float64)
        self.C_pop        = np.asarray(C_pop,     dtype=np.float64)
        self.model_interp = model_interp
        self.param_names  = list(param_names)
        self.jitter       = float(jitter)
        self.k_cache_decimals = int(k_cache_decimals)

        self._chol_cache: OrderedDict = OrderedDict()
        self._I = np.eye(int(self.omega_hat.size), dtype=np.float64)

        # Absolute jitter scale based on diagonal magnitudes
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

        # [FIX-4] k² scales ONLY C_inst
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
        m     = np.asarray(self.model_interp.model(theta), dtype=np.float64)
        d     = self.omega_hat - m
        cached = self._get_cholesky(k)
        if cached is None:
            return -np.inf
        c_and_lower, logdet = cached
        quad = float(np.dot(d, self._cho_solve(c_and_lower, d)))
        return -0.5 * (quad + logdet + d.size * np.log(2.0 * np.pi))


# ============================================================
# Optimal filter likelihood (Allen & Romano 1999)
# ============================================================
class OmegaOptimalFilterLikelihood(bilby.Likelihood):
    """
    Allen & Romano optimal-filter likelihood.
    w(f) ∝ γ²(f) / [P1(f) · P2(f)]
    Requer P1_bin, P2_bin e gamma_bin no .npz.
    """
    def __init__(self, f_bin, omega_hat, sigma_inst, P1_bin, P2_bin, gamma_bin,
                 evaluator, *, param_names, floor=1e-60):
        super().__init__(parameters={p: None for p in param_names + ["k_sigma"]})
        self.f          = np.asarray(f_bin,    dtype=np.float64)
        self.omega_hat  = np.asarray(omega_hat, dtype=np.float64)
        self.sigma_inst = np.asarray(sigma_inst, dtype=np.float64)
        self.evaluator  = evaluator
        self.param_names = list(param_names)
        P1 = np.where(np.asarray(P1_bin) > 0, P1_bin, float(floor))
        P2 = np.where(np.asarray(P2_bin) > 0, P2_bin, float(floor))
        g2 = np.asarray(gamma_bin, dtype=np.float64) ** 2
        w_raw = g2 / (P1 * P2)
        w_raw = np.where(np.isfinite(w_raw) & (w_raw > 0), w_raw, 0.0)
        w_sum = float(np.sum(w_raw))
        self._w = w_raw * (len(w_raw) / w_sum) if w_sum > 0 else np.ones_like(w_raw)

    def log_likelihood(self) -> float:
        k     = float(self.parameters["k_sigma"])
        theta = {p: float(self.parameters[p]) for p in self.param_names}
        m     = self.evaluator.model(theta)
        r     = self.omega_hat - m
        s2    = (k * self.sigma_inst) ** 2
        good  = (s2 > 0) & np.isfinite(r) & np.isfinite(s2)
        if not np.any(good):
            return -np.inf
        w_g, r_g, s2_g = self._w[good], r[good], s2[good]
        return -0.5 * float(
            np.sum(w_g * r_g**2 / s2_g)
            + np.sum(np.log(2.0 * np.pi * s2_g / np.where(w_g > 0, w_g, 1.0)))
        )
